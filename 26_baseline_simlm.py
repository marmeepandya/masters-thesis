#!/usr/bin/env python
# coding: utf-8

# # SimLM Baseline Retrieval
# ## Title + passage-body dense retrieval baseline over 98,716 companies
# 
# **Why SimLM?** `intfloat/simlm-base-msmarco-finetuned` is a BERT-base (110M param) bi-encoder pre-trained with a representation-bottleneck objective specifically for MS MARCO passage retrieval -- cheap to encode (comparable to MiniLM/GTE-large, not the 7-9B models), and it targets exactly the recall-vs-scale tradeoff the Relevance Filtering for Embedding-based Retrieval paper discusses.
# 
# **Usage convention differs from other baselines** (per the official model card, not the all-fields concatenated string used elsewhere):
# - **Pooling:** CLS token (`last_hidden_state[:, 0, :]`), then L2-normalized -- not mean pooling.
# - **Passages:** encoded as a `(title, text_pair=body)` pair, max_length=144 -- title and body are tokenized as a sentence pair, not concatenated into one string.
# - **Queries:** no prefix, max_length=32, encoded alone.
# - Here, `title` = company name, `body` = the same country/state/city/type/size/industry/summary/keywords fields used in every other baseline's rich text, just without the name (since the name is now the separate title field).
# 
# **Checkpoint/resume:** the corpus encoding step saves progress periodically and self-exits gracefully before the 30-min SLURM wall time, so a timed-out run can be resubmitted (`sbatch run13.sh` again) to continue rather than restart from zero. SimLM is small (110M params) and should encode the full corpus in a couple of minutes like GTE-large did, so this is mostly insurance.

# In[ ]:


from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RESULT_DIR = Path('result/26_baseline_simlm')
RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f'[Setup] Result folder : {RESULT_DIR}/ -- ready')

ENCODE_CHUNK_SIZE = 10000  # companies per encoding checkpoint interval


# In[ ]:


import sys, json, time
import numpy as np
import pandas as pd
import faiss

SCRIPT_START = time.time()  # marks total job elapsed time, used to stop safely before the 30-min SLURM wall time
TIME_BUDGET_MINUTES = 26    # job wall-time is 30 min -- leaves a buffer before the hard kill so a checkpoint always gets saved

print('[Imports] All packages loaded successfully')


# ## 1. Load dataset & build title + passage-body fields

# In[ ]:


print('[Load] Loading production results...')
results_df = pd.read_excel('dataset/production_results.xlsx')
all_companies = results_df.drop_duplicates(subset='domain').reset_index(drop=True)
print(f'[Load] Unique companies : {len(all_companies):,}')

with open('dataset/goi_search_results.json', 'r') as f:
    queries_data = json.load(f)
print(f'[Load] Queries : {len(queries_data)}')

def build_title(row):
    name = row.get('name', '')
    return name.strip() if isinstance(name, str) and name.strip() else 'Unknown company'

def build_body(row):
    """Same fields as every other baseline's rich text, minus name (that's the separate title here)."""
    parts = []
    for field, prefix in [
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

print('[Load] Building title + body fields for each company...')
titles = [build_title(row) for _, row in all_companies.iterrows()]
bodies = [build_body(row) for _, row in all_companies.iterrows()]
print('[Load] Sample (first company):')
print(f'  title: {titles[0]}')
print(f'  body : {bodies[0][:300]}...')


# ## 2. GPU check

# In[ ]:


import torch

print(f'[GPU] CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[GPU] Device         : {torch.cuda.get_device_name(0)}')
    print(f'[GPU] VRAM           : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    DEVICE = 'cuda'
else:
    print('[GPU] WARNING: No GPU -- this model needs a GPU to be practical')
    DEVICE = 'cpu'


# ## 3. Encode all companies with SimLM (CLS pooling, title/body sentence-pair encoding)
# 
# Checkpointed every `ENCODE_CHUNK_SIZE` companies.

# In[ ]:


from transformers import AutoTokenizer, AutoModel

REPO = 'intfloat/simlm-base-msmarco-finetuned'
print('[Encode] Loading SimLM...')
t0        = time.time()
tokenizer = AutoTokenizer.from_pretrained(REPO)
model     = AutoModel.from_pretrained(REPO).to(DEVICE)
model.eval()
print(f'[Encode] Model loaded in {time.time()-t0:.1f}s on {DEVICE}')

def cls_pool(last_hidden_state):
    emb = last_hidden_state[:, 0, :]
    return torch.nn.functional.normalize(emb, p=2, dim=1)

@torch.no_grad()
def encode_passages(titles_batch, bodies_batch, batch_size=128):
    all_embs = []
    for i in range(0, len(titles_batch), batch_size):
        t_batch = titles_batch[i:i+batch_size]
        b_batch = bodies_batch[i:i+batch_size]
        encoded = tokenizer(t_batch, text_pair=b_batch, max_length=144, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        outputs = model(**encoded)
        embs    = cls_pool(outputs.last_hidden_state)
        all_embs.append(embs.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)

@torch.no_grad()
def encode_queries(query_list, batch_size=32):
    all_embs = []
    for i in range(0, len(query_list), batch_size):
        batch   = query_list[i:i+batch_size]
        encoded = tokenizer(batch, max_length=32, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        outputs = model(**encoded)
        embs    = cls_pool(outputs.last_hidden_state)
        all_embs.append(embs.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)

CHECKPOINT_PATH = RESULT_DIR / 'company_embeddings_checkpoint.npy'
FINAL_PATH      = RESULT_DIR / 'company_embeddings.npy'

print('[Encode] Encoding all companies (title + body as sentence pair)...')
t0 = time.time()

if FINAL_PATH.exists():
    print('[Encode] Final embeddings already on disk -- loading, skipping corpus encoding')
    embeddings = np.load(FINAL_PATH)
else:
    if CHECKPOINT_PATH.exists():
        embeddings = np.load(CHECKPOINT_PATH)
        start_idx  = embeddings.shape[0]
        pct = start_idx / len(titles) * 100
        print(f'[Encode] Resuming from checkpoint -- {start_idx:,}/{len(titles):,} ({pct:.1f}%) companies already encoded')
    else:
        embeddings = None
        start_idx  = 0

    for chunk_start in range(start_idx, len(titles), ENCODE_CHUNK_SIZE):
        chunk_end    = min(chunk_start + ENCODE_CHUNK_SIZE, len(titles))
        chunk_embs   = encode_passages(titles[chunk_start:chunk_end], bodies[chunk_start:chunk_end], batch_size=128)
        embeddings   = chunk_embs if embeddings is None else np.concatenate([embeddings, chunk_embs], axis=0)
        done = embeddings.shape[0]
        pct  = done / len(titles) * 100
        print(f'[Encode]   {done:,}/{len(titles):,} companies encoded ({pct:.1f}%)...')
        np.save(CHECKPOINT_PATH, embeddings)
        if (time.time() - SCRIPT_START) / 60 > TIME_BUDGET_MINUTES:
            print(f'[Encode] Time budget ({TIME_BUDGET_MINUTES} min) reached at {done:,}/{len(titles):,} ({pct:.1f}%) -- checkpoint saved. Rerun this same job to resume.')
            sys.exit(0)

    np.save(FINAL_PATH, embeddings)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

ENCODE_TIME = time.time() - t0
print(f'[Encode] Done in {ENCODE_TIME/60:.1f} minutes (this session)')
print(f'[Encode] Embeddings shape : {embeddings.shape}')


# ## 4. Build FAISS index (IndexFlatIP, cosine via normalized vectors)

# In[ ]:


embeddings = np.load(RESULT_DIR / 'company_embeddings.npy').astype('float32')
dimension  = embeddings.shape[1]
index      = faiss.IndexFlatIP(dimension)
index.add(embeddings)
faiss.write_index(index, str(RESULT_DIR / 'company_faiss.index'))
print(f'[FAISS] Index built, {index.ntotal:,} vectors, dim={dimension}')


# ## 5. Run all 101 queries (no prefix, max_length=32)

# In[ ]:


print(f'[Run] Starting SimLM retrieval for {len(queries_data)} queries...')
all_results = []
query_times = []
total_start = time.time()

for i, item in enumerate(queries_data):
    query_id = item['query_id']
    query    = item['query']

    t0        = time.perf_counter()
    query_emb = encode_queries([query]).astype('float32')
    scores, idxs = index.search(query_emb, 1000)
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
        elapsed = time.time() - total_start
        print(f'[Run] {i+1:3d}/{len(queries_data)}  |  avg {sum(query_times)/len(query_times):.1f}ms/query')

results_df_out = pd.DataFrame(all_results)
results_df_out.to_csv(RESULT_DIR / 'simlm_results.csv', index=False)
AVG_LATENCY_MS = sum(query_times) / len(query_times)
print(f'[Run] Done! Avg latency: {AVG_LATENCY_MS:.1f}ms')


# ## Evaluation -- NDCG, Precision, Recall, F1 @ k=10,50,100,300,500,1000 (same protocol as all other baselines, plus k=300 per Istari's request)

# In[ ]:


print('[Eval] Loading production labels...')
production_df = pd.read_excel('dataset/production_results.xlsx')

K_VALUES = [10, 50, 100, 300, 500, 1000]

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
eval_df.to_csv(RESULT_DIR / 'evaluation_simlm.csv', index=False)
print(f'[Eval] Saved to {RESULT_DIR}/evaluation_simlm.csv')


# ## Final Summary

# In[ ]:


print('[Summary] ============================================================')
print('[Summary] SimLM RESULTS (title + body)')
print(f'\n[Summary] Avg query latency : {AVG_LATENCY_MS:.1f}ms')
print(f'[Summary] Companies encoded : {len(all_companies):,}')
print(f'[Summary] Embedding dims    : {embeddings.shape[1]}')
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
print('  E5-Mistral        : 0.274')
print('  Nomic             : 0.684')
print('  BGE-M3            : 0.704')
print('  BGE summary-only  : 0.707  (prefix-fixed)')
print('  BGE all-fields    : 0.727  (prefix-fixed)')
print('  OpenAI large      : 0.728')
print('  MiniLM            : 0.741')
print('  SFR-Mistral       : 0.746')
print('  Linq-Mistral      : 0.749')
print('  GTE-large         : 0.768  (best solo so far)')
r1000 = eval_df[eval_df["k"] == 1000]["recall"].mean()
print(f'  SimLM             : {r1000:.3f}')

