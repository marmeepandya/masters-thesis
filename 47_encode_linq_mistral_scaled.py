import time
SCRIPT_START = time.time()  # marks total job elapsed time, used to stop encoding safely before the SLURM wall time

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

RESULT_DIR = Path("result/47_encode_linq_mistral_scaled")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[Setup] Result folder : {RESULT_DIR}/ -- ready")

combined = pd.read_parquet("result/44_build_scaled_corpus/combined_pool.parquet")
rich_texts = combined["rich_text"].tolist()
print(f"[Load] Companies to encode: {len(rich_texts):,}")

print(f"[GPU] CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[GPU] Device : {torch.cuda.get_device_name(0)}")
    print(f"[GPU] VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    DEVICE = "cuda"
else:
    print("[GPU] WARNING: No GPU -- this model needs a GPU to be practical")
    DEVICE = "cpu"

MAX_LENGTH = 512
TIME_BUDGET_MINUTES = 27  # same margin as notebook 20 -- leave time to save a checkpoint before a 30-min job gets killed

CHECKPOINT_PATH = RESULT_DIR / "company_embeddings_checkpoint.npy"
FINAL_PATH = RESULT_DIR / "company_embeddings.npy"
TIME_LOG_PATH = RESULT_DIR / "encode_time_seconds.txt"
prior_encode_secs = float(TIME_LOG_PATH.read_text()) if TIME_LOG_PATH.exists() else 0.0

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

print("[Encode] Loading Linq-Embed-Mistral (fp16)...")
t0 = time.time()
REPO = "Linq-AI-Research/Linq-Embed-Mistral"
try:
    tokenizer = AutoTokenizer.from_pretrained(REPO, local_files_only=True)
    model = AutoModel.from_pretrained(REPO, torch_dtype=torch.float16, device_map=DEVICE, local_files_only=True)
    print("[Encode] Loaded from local cache -- skipped Hugging Face Hub network calls")
except Exception:
    print("[Encode] Not fully cached locally yet -- loading with network access (this will be slower)")
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    model = AutoModel.from_pretrained(REPO, torch_dtype=torch.float16, device_map=DEVICE)
model.eval()
print(f"[Encode] Model loaded in {(time.time()-t0)/60:.1f} minutes")


@torch.no_grad()
def encode_batch(texts, start=0, batch_size=16, prefix_embs=None, checkpoint=False):
    all_embs = [prefix_embs] if prefix_embs is not None else []
    for i in range(start, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_dict = tokenizer(batch, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        outputs = model(**batch_dict)
        embs = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().float().numpy())
        if checkpoint and (i // batch_size + 1) % 50 == 0:
            print(f"[Encode]   {i + len(batch):,}/{len(texts):,} companies encoded...")
        if checkpoint and (i // batch_size + 1) % 200 == 0:
            np.save(CHECKPOINT_PATH, np.concatenate(all_embs, axis=0))
        if checkpoint and (time.time() - SCRIPT_START) / 60 > TIME_BUDGET_MINUTES:
            np.save(CHECKPOINT_PATH, np.concatenate(all_embs, axis=0))
            TIME_LOG_PATH.write_text(str(prior_encode_secs + time.time() - encode_t0))
            print(f"[Encode] Time budget ({TIME_BUDGET_MINUTES} min) reached at {i + len(batch):,}/{len(texts):,} -- checkpoint saved, resubmit run.sh to continue")
            sys.exit(0)
    return np.concatenate(all_embs, axis=0)


if FINAL_PATH.exists() and np.load(FINAL_PATH, mmap_mode="r").shape[0] == len(rich_texts):
    print("[Encode] Final embeddings already on disk -- skipping corpus encoding")
    embeddings = np.load(FINAL_PATH).astype("float32")
    ENCODE_TIME = prior_encode_secs
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
else:
    if CHECKPOINT_PATH.exists():
        done_embs = np.load(CHECKPOINT_PATH)
        start_idx = done_embs.shape[0]
        print(f"[Encode] Resuming from checkpoint -- {start_idx:,}/{len(rich_texts):,} companies already encoded ({prior_encode_secs/60:.1f} min spent so far)")
    else:
        done_embs = None
        start_idx = 0

    print("[Encode] Encoding all companies (no instruction prefix on document side)...")
    encode_t0 = time.time()
    embeddings = encode_batch(rich_texts, start=start_idx, batch_size=16, prefix_embs=done_embs, checkpoint=True)
    ENCODE_TIME = prior_encode_secs + (time.time() - encode_t0)
    print(f"[Encode] Done in {ENCODE_TIME/60:.1f} minutes total (across all resumed runs)")
    print(f"[Encode] Embeddings shape : {embeddings.shape}")
    np.save(FINAL_PATH, embeddings)
    TIME_LOG_PATH.write_text(str(ENCODE_TIME))
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
