#!/usr/bin/env python
# coding: utf-8

# # 54 - Encode the scaled corpus with ColBERTv2 (MaxSim, per-token embeddings)
# 
# Fundamentally different from every other encoder here (45/46/48-53): ColBERT stores one embedding *per token*, not one vector per company, so the checkpoint/final files here are `doc_embs.pt` (shape `[N, 192, 128]`, fp16) and `doc_masks.pt` (`[N, 192]`, bool), not a simple `(N, dim)` `.npy` array. Same overall pattern as `27_baseline_colbert.ipynb`, including its `robust_save` helper (atomic temp-file-then-rename writes with retries, added after Lustre storage I/O errors were seen truncating `torch.save` mid-write on this cluster).
# 
# **Storage warning, worth knowing before running this**: at ~397K companies, `doc_embs.pt` alone will be roughly **19.5 GB** (the original 98,716-company run was ~4.85 GB). Make sure there's enough disk space in `result/54_encode_colbert_scaled/` before starting, and note the checkpoint file exists *alongside* the growing final tensor during encoding, so peak disk usage briefly doubles near the end. This notebook only encodes and saves the per-token document embeddings, it does not run the MaxSim search/scoring step (that's query-time work for a later evaluation notebook, same reasoning as the reranker in notebooks 45-53's markdown cells).

# In[1]:


import sys, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv()

SCRIPT_START = time.time()
TIME_BUDGET_MINUTES = 26
MAX_DOC_LEN = 192
ENCODE_CHUNK_SIZE = 5000

RESULT_DIR = Path("result/54_encode_colbert_scaled")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[Setup] Result folder : {RESULT_DIR}/ -- ready")


def robust_save(save_fn, path, retries=3, delay_seconds=5):
    """Writes to a temp file then atomically renames -- avoids leaving a corrupted checkpoint if the write is
    interrupted (seen: Lustre iostream errors truncating torch.save mid-write). Retries transient I/O failures."""
    p = Path(path)
    tmp_path = str(p.with_suffix(".tmp" + p.suffix))
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            save_fn(tmp_path)
            os.replace(tmp_path, path)
            return
        except Exception as e:
            last_err = e
            print(f"[Checkpoint] Save attempt {attempt}/{retries} to {path} failed: {e} -- retrying in {delay_seconds}s...")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            time.sleep(delay_seconds)
    raise last_err


combined = pd.read_parquet("result/44_build_scaled_corpus/combined_pool.parquet")
rich_texts = combined["rich_text"].tolist()
print(f"[Load] Companies to encode: {len(rich_texts):,}")

print(f"[GPU] CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[GPU] Device : {torch.cuda.get_device_name(0)}")
    DEVICE = "cuda"
else:
    print("[GPU] WARNING: No GPU -- this baseline needs a GPU to be practical")
    DEVICE = "cpu"


# In[2]:


from transformers import AutoTokenizer, BertModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

REPO = "colbert-ir/colbertv2.0"
print("[Load] Loading ColBERTv2 tokenizer + BertModel backbone...")
t0 = time.time()
# Try local cache first with HF_HUB_OFFLINE=1 -- avoids a slow/unauthenticated HF Hub network
# call hanging the job even when everything is already cached (same fix as notebooks 48-50/53).
os.environ["HF_HUB_OFFLINE"] = "1"
try:
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    model = BertModel.from_pretrained(REPO, add_pooling_layer=False).to(DEVICE)
    ckpt_path = hf_hub_download(REPO, "model.safetensors")
    print("[Load] Loaded from local cache -- skipped Hugging Face Hub network calls")
except Exception as e:
    print(f"[Load] Not fully cached locally yet ({type(e).__name__}) -- retrying with network access (this will be slower)")
    os.environ.pop("HF_HUB_OFFLINE", None)
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    model = BertModel.from_pretrained(REPO, add_pooling_layer=False).to(DEVICE)
    ckpt_path = hf_hub_download(REPO, "model.safetensors")
model.eval()

print("[Load] Loading token-projection linear layer from checkpoint...")
state_dict = load_safetensors(ckpt_path)
linear_weight = state_dict["linear.weight"].to(DEVICE)  # [128, 768], no bias
print(f"[Load] Done in {time.time()-t0:.1f}s on {DEVICE} -- linear projection shape {tuple(linear_weight.shape)}")


@torch.no_grad()
def encode_tokens(texts, max_length, batch_size=64):
    """Returns (embeddings [N, max_length, 128] fp16, mask [N, max_length] bool), both on CPU."""
    all_embs, all_masks = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
        outputs = model(**encoded)
        proj = F.linear(outputs.last_hidden_state, linear_weight)  # [B, L, 128]
        proj = F.normalize(proj, p=2, dim=-1)
        all_embs.append(proj.half().cpu())
        all_masks.append(encoded["attention_mask"].bool().cpu())
    return torch.cat(all_embs, dim=0), torch.cat(all_masks, dim=0)


# In[ ]:


CHECKPOINT_EMBS_PATH = RESULT_DIR / "doc_embs_checkpoint.pt"
CHECKPOINT_MASKS_PATH = RESULT_DIR / "doc_masks_checkpoint.pt"
FINAL_EMBS_PATH = RESULT_DIR / "doc_embs.pt"
FINAL_MASKS_PATH = RESULT_DIR / "doc_masks.pt"

print(f"[Encode] Encoding {len(rich_texts):,} companies (max_length={MAX_DOC_LEN})...")
t0 = time.time()

if FINAL_EMBS_PATH.exists():
    print("[Encode] Final embeddings already on disk -- loading, skipping corpus encoding")
    doc_embs = torch.load(FINAL_EMBS_PATH)
    doc_masks = torch.load(FINAL_MASKS_PATH)
else:
    if CHECKPOINT_EMBS_PATH.exists():
        doc_embs = torch.load(CHECKPOINT_EMBS_PATH)
        doc_masks = torch.load(CHECKPOINT_MASKS_PATH)
        start_idx = doc_embs.shape[0]
        pct = start_idx / len(rich_texts) * 100
        print(f"[Encode] Resuming from checkpoint -- {start_idx:,}/{len(rich_texts):,} ({pct:.1f}%) companies already encoded")
    else:
        doc_embs, doc_masks = None, None
        start_idx = 0

    for chunk_start in range(start_idx, len(rich_texts), ENCODE_CHUNK_SIZE):
        chunk_end = min(chunk_start + ENCODE_CHUNK_SIZE, len(rich_texts))
        chunk_texts = rich_texts[chunk_start:chunk_end]
        chunk_embs, chunk_masks = encode_tokens(chunk_texts, max_length=MAX_DOC_LEN, batch_size=64)
        doc_embs = chunk_embs if doc_embs is None else torch.cat([doc_embs, chunk_embs], dim=0)
        doc_masks = chunk_masks if doc_masks is None else torch.cat([doc_masks, chunk_masks], dim=0)
        done = doc_embs.shape[0]
        pct = done / len(rich_texts) * 100
        robust_save(lambda p: torch.save(doc_embs, p), CHECKPOINT_EMBS_PATH)
        robust_save(lambda p: torch.save(doc_masks, p), CHECKPOINT_MASKS_PATH)
        print(f"[Encode]   {done:,}/{len(rich_texts):,} companies encoded ({pct:.1f}%)... checkpoint saved")
        if (time.time() - SCRIPT_START) / 60 > TIME_BUDGET_MINUTES:
            print(f"[Encode] Time budget ({TIME_BUDGET_MINUTES} min) reached at {done:,}/{len(rich_texts):,} ({pct:.1f}%) -- checkpoint saved. Rerun this same job to resume.")
            sys.exit(0)

    robust_save(lambda p: torch.save(doc_embs, p), FINAL_EMBS_PATH)
    robust_save(lambda p: torch.save(doc_masks, p), FINAL_MASKS_PATH)
    if CHECKPOINT_EMBS_PATH.exists():
        CHECKPOINT_EMBS_PATH.unlink()
    if CHECKPOINT_MASKS_PATH.exists():
        CHECKPOINT_MASKS_PATH.unlink()

ENCODE_TIME = time.time() - t0
print(f"[Encode] Done in {ENCODE_TIME/60:.1f} minutes (this session)")
print(f"[Encode] doc_embs shape : {tuple(doc_embs.shape)}  ({doc_embs.element_size()*doc_embs.nelement()/1e9:.2f} GB)")
print(f"[Encode] Saved -> {FINAL_EMBS_PATH}, {FINAL_MASKS_PATH}")

