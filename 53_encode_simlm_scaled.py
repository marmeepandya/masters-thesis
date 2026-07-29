import os
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

load_dotenv()

RESULT_DIR = Path("result/53_encode_simlm_scaled")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = RESULT_DIR / "company_embeddings_checkpoint.npy"
FINAL_PATH = RESULT_DIR / "company_embeddings.npy"
CHUNK_SIZE = 20_000

combined = pd.read_parquet("result/44_build_scaled_corpus/combined_pool.parquet")


def build_title(row):
    name = row.get("name", "")
    return name.strip() if isinstance(name, str) and name.strip() else "Unknown company"


def build_body(row):
    """Same fields as every other baseline's rich text, minus name (that's the separate title here)."""
    parts = []
    for field, prefix in [
        ("country",           "Country:"),
        ("state",             "State:"),
        ("municipality",      "City:"),
        ("district",          "District:"),
        ("organization_type", "Type:"),
        ("organization_size", "Size:"),
        ("nace_code",         "Industry:"),
        ("summary",           ""),
    ]:
        val = row.get(field, "")
        if isinstance(val, str) and val.strip():
            parts.append(f"{prefix} {val}".strip() if prefix else val)
    kw = row.get("summary_keywords", "")
    if isinstance(kw, str) and kw.strip():
        kw_clean = kw.replace("'", "").replace("[", "").replace("]", "")
        parts.append(f"Keywords: {kw_clean}")
    return " | ".join(parts)


print("[Load] Building title + body fields for each company...")
titles = [build_title(row) for _, row in combined.iterrows()]
bodies = [build_body(row) for _, row in combined.iterrows()]
print(f"[Load] Companies to encode: {len(titles):,}")

print(f"[GPU] CUDA available : {torch.cuda.is_available()}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"[GPU] Device : {torch.cuda.get_device_name(0)}")

print("[Encode] Loading SimLM (intfloat/simlm-base-msmarco-finetuned)...")
t0 = time.time()
REPO = "intfloat/simlm-base-msmarco-finetuned"
# Try local cache first with local_files_only -- avoids a slow/unauthenticated HF Hub network
# call hanging the job even when the model is already fully cached (same fix as notebooks 48-50).
try:
    tokenizer = AutoTokenizer.from_pretrained(REPO, local_files_only=True)
    model = AutoModel.from_pretrained(REPO, local_files_only=True).to(DEVICE)
    print("[Encode] Loaded from local cache -- skipped Hugging Face Hub network calls")
except Exception as e:
    print(f"[Encode] Not fully cached locally yet ({type(e).__name__}) -- retrying with network access (this will be slower)")
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    model = AutoModel.from_pretrained(REPO).to(DEVICE)
model.eval()
print(f"[Encode] Model loaded in {time.time()-t0:.1f}s on {DEVICE}")


def cls_pool(last_hidden_state):
    emb = last_hidden_state[:, 0, :]
    return torch.nn.functional.normalize(emb, p=2, dim=1)

@torch.no_grad()
def encode_passages(titles_batch, bodies_batch, batch_size=128):
    all_embs = []
    for i in range(0, len(titles_batch), batch_size):
        t_batch = titles_batch[i:i + batch_size]
        b_batch = bodies_batch[i:i + batch_size]
        encoded = tokenizer(t_batch, text_pair=b_batch, max_length=144, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        outputs = model(**encoded)
        embs = cls_pool(outputs.last_hidden_state)
        all_embs.append(embs.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


if FINAL_PATH.exists() and np.load(FINAL_PATH, mmap_mode="r").shape[0] == len(titles):
    print("[Encode] Final embeddings already on disk -- skipping")
    embeddings = np.load(FINAL_PATH)
else:
    if CHECKPOINT_PATH.exists():
        done = np.load(CHECKPOINT_PATH)
        start = done.shape[0]
        print(f"[Encode] Resuming from checkpoint -- {start:,}/{len(titles):,} already encoded")
        all_chunks = [done]
    else:
        start = 0
        all_chunks = []

    t0 = time.time()
    for chunk_start in range(start, len(titles), CHUNK_SIZE):
        chunk_embs = encode_passages(
            titles[chunk_start:chunk_start + CHUNK_SIZE],
            bodies[chunk_start:chunk_start + CHUNK_SIZE],
            batch_size=128,
        )
        all_chunks.append(chunk_embs)
        done_so_far = chunk_start + len(chunk_embs)
        np.save(CHECKPOINT_PATH, np.concatenate(all_chunks, axis=0))
        elapsed = time.time() - t0
        print(f"[Encode] {done_so_far:,}/{len(titles):,} encoded ({elapsed/60:.1f} min elapsed)")

    embeddings = np.concatenate(all_chunks, axis=0)
    np.save(FINAL_PATH, embeddings)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    print(f"[Encode] Done. Embeddings shape: {embeddings.shape}")
    print(f"[Encode] Saved -> {FINAL_PATH}")
