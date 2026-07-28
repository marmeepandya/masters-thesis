import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

RESULT_DIR = Path("result/48_encode_bge_large_scaled")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = RESULT_DIR / "company_embeddings_checkpoint.npy"
FINAL_PATH = RESULT_DIR / "company_embeddings.npy"
CHUNK_SIZE = 20_000

combined = pd.read_parquet("result/44_build_scaled_corpus/combined_pool.parquet")
rich_texts = combined["rich_text"].tolist()
print(f"[Load] Companies to encode: {len(rich_texts):,}")

print(f"[GPU] CUDA available : {torch.cuda.is_available()}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"[GPU] Device : {torch.cuda.get_device_name(0)}")

print("[Encode] Loading BGE-large (BAAI/bge-large-en-v1.5)...")
t0 = time.time()
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=DEVICE)
print(f"[Encode] Model loaded in {time.time()-t0:.1f}s on {model.device}")
batch_size = 256
print(f"[Encode] Batch size : {batch_size}")

if FINAL_PATH.exists() and np.load(FINAL_PATH, mmap_mode="r").shape[0] == len(rich_texts):
    print("[Encode] Final embeddings already on disk -- skipping")
    embeddings = np.load(FINAL_PATH)
else:
    if CHECKPOINT_PATH.exists():
        done = np.load(CHECKPOINT_PATH)
        start = done.shape[0]
        print(f"[Encode] Resuming from checkpoint -- {start:,}/{len(rich_texts):,} already encoded")
        all_chunks = [done]
    else:
        start = 0
        all_chunks = []

    t0 = time.time()
    for chunk_start in range(start, len(rich_texts), CHUNK_SIZE):
        chunk_texts = rich_texts[chunk_start:chunk_start + CHUNK_SIZE]
        chunk_embs = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # REQUIRED for BGE -- uses cosine similarity
        )
        all_chunks.append(chunk_embs)
        done_so_far = chunk_start + len(chunk_texts)
        np.save(CHECKPOINT_PATH, np.concatenate(all_chunks, axis=0))
        elapsed = time.time() - t0
        print(f"[Encode] {done_so_far:,}/{len(rich_texts):,} encoded ({elapsed/60:.1f} min elapsed)")

    embeddings = np.concatenate(all_chunks, axis=0)
    np.save(FINAL_PATH, embeddings)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    print(f"[Encode] Done. Embeddings shape: {embeddings.shape}")
    print(f"[Encode] Saved -> {FINAL_PATH}")
