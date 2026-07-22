#!/usr/bin/env python
# coding: utf-8

# # BGE-Multilingual-Gemma2 Baseline Retrieval
# ## All-fields dense retrieval baseline over 98,716 companies
# 
# **Why BGE-Multilingual-Gemma2?** Swapped in for `Alibaba-NLP/gte-Qwen2-7B-instruct` in the "leading heavy embedder" slot -- gte-Qwen2 ships custom `trust_remote_code=True` modeling code (bidirectional-attention wrapper over Qwen2), the same class of custom code that already broke `Alibaba-NLP/gte-large-en-v1.5` (see notebook 17) and `nvidia/NV-Embed-v2` (notebook 19) against our installed `transformers==5.9.0`. `BAAI/bge-multilingual-gemma2` is fine-tuned on top of `google/gemma-2-9b`, a standard architecture natively supported in transformers -- no custom modeling file, no `trust_remote_code`. 9B parameters, 3584-dim embeddings, last-token pooling.
# 
# **Text representation:** all fields -- name, country, state, city, district, organization_type, organization_size, nace_code, summary, summary_keywords (identical convention to `03_baseline_minilm.ipynb`, for a fair comparison).

# In[1]:


import time
SCRIPT_START = time.time()  # marks total job elapsed time, used to stop encoding safely before the 30-min SLURM wall time

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RESULT_DIR = Path('result/15_baseline_bge_gemma2')
RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f'[Setup] Result folder : {RESULT_DIR}/ -- ready')


# In[2]:


import json, time
import numpy as np
import pandas as pd
import faiss
print('[Imports] All packages loaded successfully')


# ## 1. Load dataset & build all-fields rich text

# In[3]:


print('[Load] Loading production results...')
results_df = pd.read_excel('dataset/production_results.xlsx')
all_companies = results_df.drop_duplicates(subset='domain').reset_index(drop=True)
print(f'[Load] Unique companies : {len(all_companies):,}')

with open('dataset/goi_search_results.json', 'r') as f:
    queries_data = json.load(f)
print(f'[Load] Queries : {len(queries_data)}')

def build_rich_text(row):
    """Combine all informative fields into one string -- same convention as 03_baseline_minilm.ipynb."""
    parts = []
    for field, prefix in [
        ('name',              'Company:'),
        ('country',           'Country:'),
        ('state',             'State:'),
        ('municipality',      'City:'),
        ('district',          'District:'),
        ('organization_type', 'Type:'),
        ('organization_size', 'Size:'),
        ('nace_code',         'Industry:'),
        ('summary',           ''),
    ]:
        val = row.get(field, '')
        if isinstance(val, str) and val.strip():
            parts.append(f'{prefix} {val}'.strip() if prefix else val)
    kw = row.get('summary_keywords', '')
    if isinstance(kw, str) and kw.strip():
        kw_clean = kw.replace("'", '').replace('[', '').replace(']', '')
        parts.append(f'Keywords: {kw_clean}')
    return ' | '.join(parts)

print('[Load] Building all-fields rich text for each company...')
rich_texts = [build_rich_text(row) for _, row in all_companies.iterrows()]
print('[Load] Sample rich text (first company):')
print(f'  {rich_texts[0][:300]}...')


# ## 2. GPU check

# In[4]:


import torch

print(f'[GPU] CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[GPU] Device         : {torch.cuda.get_device_name(0)}')
    print(f'[GPU] VRAM           : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    DEVICE = 'cuda'
else:
    print('[GPU] WARNING: No GPU -- this model needs a GPU to be practical')
    DEVICE = 'cpu'


# ## 3. Encode all companies with BGE-Multilingual-Gemma2
# 
# **Instruction format:** query-side instruction prefix, same `"Instruct: {task}\nQuery: {query}"` convention used for E5-Mistral. **Not yet verified against the official BGE-Gemma2 model card** -- double-check this template before treating results as final, since it may differ from the e5-mistral lineage's exact convention. Documents get no prefix. Last-token pooling (decoder-based model, like E5-Mistral). Our texts are short (~160 tokens avg) so `max_length=512` is generous headroom.

# In[ ]:


import sys, os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

TASK_INSTRUCTION = 'Given a search query describing a type of company, retrieve relevant company profiles'
MAX_LENGTH = 512
TIME_BUDGET_MINUTES = 26  # switched back to gpu_a100_short (30-min wall time) -- gpu_a100_il had 0 available nodes for 3+ days straight

CHECKPOINT_PATH   = RESULT_DIR / 'company_embeddings_checkpoint.npy'
FINAL_PATH        = RESULT_DIR / 'company_embeddings.npy'
TIME_LOG_PATH     = RESULT_DIR / 'encode_time_seconds.txt'
prior_encode_secs = float(TIME_LOG_PATH.read_text()) if TIME_LOG_PATH.exists() else 0.0

def robust_save(save_fn, path, retries=3, delay_seconds=5):
    """Writes to a temp file then atomically renames -- avoids leaving a corrupted checkpoint if the write is interrupted (seen: Lustre iostream errors truncating writes mid-stream). Retries transient I/O failures. Keeps the original suffix on the temp name since np.save auto-appends .npy to any path that doesn't already end with it."""
    p = Path(path)
    tmp_path = str(p.with_suffix('.tmp' + p.suffix))
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            save_fn(tmp_path)
            os.replace(tmp_path, path)
            return
        except Exception as e:
            last_err = e
            print(f'[Checkpoint] Save attempt {attempt}/{retries} to {path} failed: {e} -- retrying in {delay_seconds}s...')
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            time.sleep(delay_seconds)
    raise last_err

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

print('[Encode] Loading BGE-Multilingual-Gemma2 (fp16)...')
t0   = time.time()
REPO = 'BAAI/bge-multilingual-gemma2'
try:
    tokenizer = AutoTokenizer.from_pretrained(REPO, local_files_only=True)
    model     = AutoModel.from_pretrained(REPO, torch_dtype=torch.float16, device_map=DEVICE, local_files_only=True)
    print('[Encode] Loaded from local cache -- skipped Hugging Face Hub network calls')
except Exception:
    print('[Encode] Not fully cached locally yet -- loading with network access (this will be slower)')
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    model     = AutoModel.from_pretrained(REPO, torch_dtype=torch.float16, device_map=DEVICE)
model.eval()
print(f'[Encode] Model loaded in {(time.time()-t0)/60:.1f} minutes')

@torch.no_grad()
def encode_batch(texts, start=0, batch_size=16, prefix_embs=None, checkpoint=False):
    all_embs = [prefix_embs] if prefix_embs is not None else []
    for i in range(start, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_dict = tokenizer(batch, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        outputs = model(**batch_dict)
        embs = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().float().numpy())
        if checkpoint and (i // batch_size + 1) % 50 == 0:
            done_so_far = i + len(batch)
            pct = done_so_far / len(texts) * 100
            print(f'[Encode]   {done_so_far:,}/{len(texts):,} companies encoded ({pct:.1f}%)...')
        if checkpoint and (i // batch_size + 1) % 200 == 0:
            robust_save(lambda p: np.save(p, np.concatenate(all_embs, axis=0)), CHECKPOINT_PATH)
        if checkpoint and (time.time() - SCRIPT_START) / 60 > TIME_BUDGET_MINUTES:
            robust_save(lambda p: np.save(p, np.concatenate(all_embs, axis=0)), CHECKPOINT_PATH)
            TIME_LOG_PATH.write_text(str(prior_encode_secs + time.time() - encode_t0))
            done_so_far = i + len(batch)
            pct = done_so_far / len(texts) * 100
            print(f'[Encode] Time budget ({TIME_BUDGET_MINUTES} min) reached at {done_so_far:,}/{len(texts):,} ({pct:.1f}%) -- NOT fully encoded, file stays named "checkpoint". Rerun this same script/notebook (any device, batch or interactive) to resume and add more progress.')
            sys.exit(0)
    return np.concatenate(all_embs, axis=0)

if FINAL_PATH.exists() and np.load(FINAL_PATH, mmap_mode='r').shape[0] == len(rich_texts):
    print(f'[Encode] {len(rich_texts):,}/{len(rich_texts):,} (100%) already embedded -- final file on disk, skipping corpus encoding')
    embeddings = np.load(FINAL_PATH).astype('float32')
    ENCODE_TIME = prior_encode_secs
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
else:
    if CHECKPOINT_PATH.exists():
        done_embs = np.load(CHECKPOINT_PATH)
        start_idx = done_embs.shape[0]
        pct = start_idx / len(rich_texts) * 100
        print(f'[Encode] Resuming from checkpoint -- {start_idx:,}/{len(rich_texts):,} ({pct:.1f}%) companies already encoded ({prior_encode_secs/60:.1f} min spent so far). File is named "checkpoint" because encoding is not yet 100% complete -- this is based purely on companies-encoded count, not on how the previous run was launched or how it exited.')
    else:
        done_embs = None
        start_idx = 0

    print('[Encode] Encoding all companies (no instruction prefix on document side)...')
    encode_t0  = time.time()
    embeddings = encode_batch(rich_texts, start=start_idx, batch_size=16, prefix_embs=done_embs, checkpoint=True)
    ENCODE_TIME = prior_encode_secs + (time.time() - encode_t0)
    print(f'[Encode] Done in {ENCODE_TIME/60:.1f} minutes total (across all resumed runs)')
    print(f'[Encode] Embeddings shape : {embeddings.shape} -- all {len(rich_texts):,} companies now embedded, writing final file (no "checkpoint" in the name)')
    robust_save(lambda p: np.save(p, embeddings), FINAL_PATH)
    TIME_LOG_PATH.write_text(str(ENCODE_TIME))
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


# ## 4. Build FAISS index (IndexFlatIP, cosine via normalized vectors)

# In[ ]:


embeddings = np.load(RESULT_DIR / 'company_embeddings.npy').astype('float32')
dimension  = embeddings.shape[1]
index      = faiss.IndexFlatIP(dimension)
index.add(embeddings)
faiss.write_index(index, str(RESULT_DIR / 'company_faiss.index'))
print(f'[FAISS] Index built, {index.ntotal:,} vectors, dim={dimension}')


# ## 5. Run all 101 queries (WITH instruction prefix)

# In[ ]:


query_prefix = f'Instruct: {TASK_INSTRUCTION}\nQuery: '

print(f'[Run] Starting BGE-Multilingual-Gemma2 retrieval for {len(queries_data)} queries...')
all_results = []
query_times = []
total_start = time.time()

for i, item in enumerate(queries_data):
    query_id = item['query_id']
    query    = item['query']
    query_with_instruction = query_prefix + query

    t0        = time.perf_counter()
    query_emb = encode_batch([query_with_instruction], batch_size=1)
    scores, idxs = index.search(query_emb.astype('float32'), 1000)
    query_ms  = (time.perf_counter() - t0) * 1000
    query_times.append(query_ms)

    for rank, (idx, score) in enumerate(zip(idxs[0], scores[0])):
        company = all_companies.iloc[idx]
        all_results.append({
            'query_id': query_id, 'query': query, 'rank': rank + 1,
            'score': float(score), 'domain': company['domain'],
            'name': company.get('name', ''), 'country': company.get('country', ''),
        })

    if (i + 1) % 20 == 0 or (i + 1) == len(queries_data):
        print(f'[Run] {i+1:3d}/{len(queries_data)}  |  avg {sum(query_times)/len(query_times):.1f}ms/query')

results_df_out = pd.DataFrame(all_results)
results_df_out.to_csv(RESULT_DIR / '15_baseline_bge_gemma2_results.csv', index=False)
AVG_LATENCY_MS = sum(query_times) / len(query_times)
print(f'[Run] Done! Avg latency: {AVG_LATENCY_MS:.1f}ms')


# ## Evaluation -- NDCG, Precision, Recall, F1 @ k (same protocol as all other baselines)

# In[ ]:


print('[Eval] Loading production labels...')
production_df = pd.read_excel('dataset/production_results.xlsx')

K_VALUES = [10, 50, 100, 500, 1000]

def get_relevant(query_id, top_k=1000):
    return set(production_df[
        (production_df['query_id'] == query_id) &
        (production_df['rank'] <= top_k)
    ]['domain'].tolist())

def precision_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / k if k else 0

def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0

def f1_at_k(retrieved, relevant, k):
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

def dcg_at_k(retrieved, relevant, k):
    return sum(
        1 / np.log2(i + 2)
        for i, d in enumerate(retrieved[:k]) if d in relevant
    )

def ndcg_at_k(retrieved, relevant, k):
    ideal = dcg_at_k(list(relevant), relevant, k)
    return dcg_at_k(retrieved, relevant, k) / ideal if ideal else 0

print('[Eval] Computing metrics for all queries...')
eval_rows = []

for i, item in enumerate(queries_data):
    qid       = item['query_id']
    query     = item['query']
    relevant  = get_relevant(qid)
    retrieved = (
        results_df_out[results_df_out['query_id'] == qid]
        .sort_values('rank')['domain'].tolist()
    )
    for k in K_VALUES:
        eval_rows.append({
            'query_id':  qid,
            'query':     query,
            'k':         k,
            'precision': precision_at_k(retrieved, relevant, k),
            'recall':    recall_at_k(retrieved, relevant, k),
            'f1':        f1_at_k(retrieved, relevant, k),
            'ndcg':      ndcg_at_k(retrieved, relevant, k),
        })

    if (i + 1) % 25 == 0:
        print(f'[Eval] {i+1}/101 queries evaluated...')

eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(RESULT_DIR / 'evaluation_15_baseline_bge_gemma2.csv', index=False)
print(f'[Eval] Saved to {RESULT_DIR}/evaluation_15_baseline_bge_gemma2.csv')


# ## Final Summary

# In[ ]:


print('[Summary] ============================================================')
print('[Summary] BGE-Multilingual-Gemma2 RESULTS (All-Fields)')
print(f'\n[Summary] Encoding time     : {ENCODE_TIME/60:.1f} minutes')
print(f'[Summary] Avg query latency : {AVG_LATENCY_MS:.1f}ms')
print(f'[Summary] Companies encoded : {len(all_companies):,}')
print('[Summary] Embedding dims    : 3584')
print(f'[Summary] Queries evaluated : {len(queries_data)}')
print()
print(f'  {"k":<6} {"NDCG":>8} {"Precision":>10} {"Recall":>8} {"F1":>8}')
print('  ' + '-' * 46)
for k in K_VALUES:
    sub = eval_df[eval_df['k'] == k]
    print(f'  {k:<6} '
          f'{sub["ndcg"].mean():>8.3f} '
          f'{sub["precision"].mean():>10.3f} '
          f'{sub["recall"].mean():>8.3f} '
          f'{sub["f1"].mean():>8.3f}')

print()
print('[Summary] Comparison against models already tested (Recall@1000):')
print('  BM25              : 0.635')
print('  Nomic             : 0.684')
print('  BGE summary-only  : 0.707  (prefix-fixed)')
print('  BGE all-fields    : 0.727  (prefix-fixed)')
print('  OpenAI large      : 0.728')
print('  MiniLM            : 0.741  (best so far)')
r1000 = eval_df[eval_df["k"] == 1000]["recall"].mean()
print(f'  BGE-Multilingual-Gemma2: {r1000:.3f}')

