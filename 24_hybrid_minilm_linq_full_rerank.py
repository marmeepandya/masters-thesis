#!/usr/bin/env python
# coding: utf-8

# # Hybrid Retrieval: MiniLM + Linq-Embed-Mistral -- Full-Pool Rerank (Re2G-style)
# ## No RRF pre-fusion -- rerank the entire union pool, then cut to top-1000
# 
# **Why this variant:** notebook 23 fused MiniLM + Linq_Mistral with RRF and truncated to top-1000 *before* reranking, which left a 0.087 recall gap vs the 0.869 oracle ceiling from notebook 22 (some oracle-reachable candidates never entered the reranked pool). This notebook follows the Re2G paper's actual architecture (Figure 3) more closely: take the raw union of both channels' top-1000 (no RRF, no pre-truncation), rerank the entire pool with the cross-encoder, then keep the top-1000 by reranker score.
# 
# ```
# MiniLM top-1000              --+
#                                 +-- raw union (~1,300 avg candidates, no fusion/truncation)
# Linq-Embed-Mistral top-1000  --+
#                                      |
#                                      v
#                     Cross-encoder reranker scores EVERY candidate
#                                      |
#                                      v
#                           Sort by score, keep top-1000
# ```
# 
# **Cost tradeoff:** reranking ~1,300 candidates/query instead of 50 is ~9 minutes of GPU compute for all 101 queries (vs 253ms/query in notebook 23). This variant measures the *quality ceiling* a reranker can reach over this candidate pool -- it is not meant to be a production-latency pipeline.
# 
# **Sources (all cached, no re-encoding):**
# - `result/03_baseline_minilm/minilm_results.csv`
# - `result/20_baseline_linq_mistral/20_baseline_linq_mistral_results.csv`
# 
# **Folder structure:**
# ```
# result/
#   24_hybrid_minilm_linq_full_rerank/
#     reranked_results.csv     # full-pool reranked top-1000 per query
#     evaluation.csv           # NDCG/Prec/Recall/F1
#     comparison_vs_rrf.csv    # this vs notebook 23's RRF+top-50-rerank vs oracle ceiling
# ```

# In[ ]:


import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RESULT_DIR = Path('result/24_hybrid_minilm_linq_full_rerank')
RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f'[Setup] Result folder : {RESULT_DIR}/ -- ready')

TOP_K_PER_CHANNEL = 1000   # how many each channel contributes to the union pool
FINAL_K           = 1000   # how many survive after reranking
RERANK_BATCH_SIZE = 32
print(f'[Setup] Top-k per channel : {TOP_K_PER_CHANNEL}')
print(f'[Setup] Final k           : {FINAL_K}')


# In[ ]:


import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[Imports] Device      : {DEVICE}')
if torch.cuda.is_available():
    print(f'[Imports] GPU         : {torch.cuda.get_device_name(0)}')
print('[Imports] All packages loaded successfully')


# ## 1. Load saved retrieval results -- both already cached, no re-encoding needed

# In[ ]:


print('[Load] Loading MiniLM results...')
minilm_df = pd.read_csv('result/03_baseline_minilm/minilm_results.csv')
print(f'[Load] MiniLM rows    : {len(minilm_df):,}')

print('[Load] Loading Linq-Embed-Mistral results...')
linq_df = pd.read_csv('result/20_baseline_linq_mistral/20_baseline_linq_mistral_results.csv')
print(f'[Load] Linq rows      : {len(linq_df):,}')

print('[Load] Loading production labels...')
production_df = pd.read_excel('dataset/production_results.xlsx')
print(f'[Load] Production rows: {len(production_df):,}')

print('[Load] Loading corpus...')
all_companies = pd.read_excel('dataset/production_results.xlsx')
all_companies = all_companies.drop_duplicates(subset='domain').reset_index(drop=True)
print(f'[Load] Corpus size    : {len(all_companies):,}')

print('[Load] Loading queries...')
with open('dataset/goi_search_results.json', 'r') as f:
    data = json.load(f)
print(f'[Load] Queries        : {len(data)}')

minilm_queries = minilm_df['query_id'].nunique()
linq_queries   = linq_df['query_id'].nunique()
assert minilm_queries == linq_queries == 101, 'Query count mismatch!'
print('[Load] Sanity check passed')

domain_info = all_companies.set_index('domain')[['name','country','summary']].to_dict('index')


# ## 2. Build raw union pool per query -- no RRF, no pre-truncation

# In[ ]:


pool_sizes = {}
query_pools = {}

for item in data:
    qid = item['query_id']
    minilm_ranked = minilm_df[minilm_df['query_id'] == qid].sort_values('rank')['domain'].tolist()[:TOP_K_PER_CHANNEL]
    linq_ranked   = linq_df[linq_df['query_id'] == qid].sort_values('rank')['domain'].tolist()[:TOP_K_PER_CHANNEL]
    pool = list(dict.fromkeys(minilm_ranked + linq_ranked))  # union, dedup, order not important -- reranker resorts everything
    query_pools[qid] = pool
    pool_sizes[qid] = len(pool)

avg_pool = sum(pool_sizes.values()) / len(pool_sizes)
print(f'[Pool] Avg pool size : {avg_pool:.0f} (matches notebook 22 oracle union computation)')
print(f'[Pool] Min pool size : {min(pool_sizes.values())}')
print(f'[Pool] Max pool size : {max(pool_sizes.values())}')


# ## 3. Load cross-encoder reranker -- `BAAI/bge-reranker-v2-m3`, same model used in notebooks 07 and 23

# In[ ]:


print('[Reranker] Loading BGE cross-encoder reranker...')
t0                 = time.time()
RERANKER_MODEL     = 'BAAI/bge-reranker-v2-m3'
tokenizer_reranker = AutoTokenizer.from_pretrained(RERANKER_MODEL)
model_reranker     = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
model_reranker     = model_reranker.to(DEVICE)
model_reranker.eval()
print(f'[Reranker] Loaded in {time.time()-t0:.1f}s on {DEVICE}')

@torch.no_grad()
def score_pool(query, domains, batch_size=RERANK_BATCH_SIZE):
    summaries  = [domain_info.get(d, {}).get('summary', '') or '' for d in domains]
    pairs      = [[query, s] for s in summaries]
    all_scores = []
    for i in range(0, len(pairs), batch_size):
        batch   = pairs[i:i+batch_size]
        encoded = tokenizer_reranker(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors='pt'
        ).to(DEVICE)
        scores = model_reranker(**encoded).logits.squeeze(-1)
        all_scores.extend(scores.cpu().float().tolist())
    return all_scores

print('[Warmup] Warming up reranker with 3 dummy calls...')
sample_qid   = data[0]['query_id']
sample_query = data[0]['query']
sample_pool  = query_pools[sample_qid][:50]
for _ in range(3):
    score_pool(sample_query, sample_pool)
print('[Warmup] GPU warm')


# ## 4. Rerank the full union pool per query, keep top-1000
# 
# This is the expensive step -- reranks ~1,300 candidates/query instead of the 50 notebook 23 used. Expect several minutes total, not milliseconds.

# In[ ]:


print(f'[Rerank] Scoring full union pool for {len(data)} queries...')
print('-' * 60)

all_results  = []
rerank_times = []
total_start  = time.time()

for i, item in enumerate(data):
    qid   = item['query_id']
    query = item['query']
    pool  = query_pools[qid]

    t0     = time.perf_counter()
    scores = score_pool(query, pool)
    rerank_ms = (time.perf_counter() - t0) * 1000
    rerank_times.append(rerank_ms)

    scored = sorted(zip(pool, scores), key=lambda x: -x[1])[:FINAL_K]

    for rank, (domain, score) in enumerate(scored, start=1):
        info = domain_info.get(domain, {})
        all_results.append({
            'query_id':       qid,
            'query':          query,
            'rank':           rank,
            'reranker_score': score,
            'domain':         domain,
            'name':           info.get('name', ''),
            'country':        info.get('country', ''),
        })

    if (i + 1) % 10 == 0 or (i + 1) == len(data):
        elapsed     = time.time() - total_start
        remaining_t = (len(data) - i - 1) * elapsed / (i + 1)
        print(f'[Rerank] {i+1:3d}/{len(data)}  |  avg {sum(rerank_times)/len(rerank_times)/1000:.1f}s/query  |  ~{remaining_t:.0f}s remaining')

results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULT_DIR / 'reranked_results.csv', index=False)

AVG_RERANK_S = sum(rerank_times) / len(rerank_times) / 1000
print('-' * 60)
print(f'[Rerank] Done!')
print(f'[Rerank] Avg rerank time : {AVG_RERANK_S:.1f}s/query')
print(f'[Rerank] Saved to        : {RESULT_DIR / "reranked_results.csv"}')


# ## 5. Evaluation

# In[ ]:


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
    return sum(1 / np.log2(i + 2) for i, d in enumerate(retrieved[:k]) if d in relevant)

def ndcg_at_k(retrieved, relevant, k):
    ideal = dcg_at_k(list(relevant), relevant, k)
    return dcg_at_k(retrieved, relevant, k) / ideal if ideal else 0

print('[Eval] Evaluating full-pool rerank results...')
eval_rows = []

for i, item in enumerate(data):
    qid       = item['query_id']
    relevant  = get_relevant(qid)
    retrieved = (
        results_df[results_df['query_id'] == qid]
        .sort_values('rank')['domain'].tolist()
    )
    for k in K_VALUES:
        eval_rows.append({
            'query_id':  qid,
            'query':     item['query'],
            'k':         k,
            'precision': precision_at_k(retrieved, relevant, k),
            'recall':    recall_at_k(retrieved, relevant, k),
            'f1':        f1_at_k(retrieved, relevant, k),
            'ndcg':      ndcg_at_k(retrieved, relevant, k),
        })

    if (i + 1) % 25 == 0:
        print(f'[Eval] {i+1}/101 queries evaluated...')

eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(RESULT_DIR / 'evaluation.csv', index=False)

print('[Eval] === FULL-POOL RERANK RESULTS ===')
print(f'  {"k":<6} {"NDCG":>8} {"Precision":>10} {"Recall":>8} {"F1":>8}')
print('  ' + '-' * 46)
for k in K_VALUES:
    sub = eval_df[eval_df['k'] == k]
    print(f'  {k:<6} '
          f'{sub["ndcg"].mean():>8.3f} '
          f'{sub["precision"].mean():>10.3f} '
          f'{sub["recall"].mean():>8.3f} '
          f'{sub["f1"].mean():>8.3f}')


# ## 6. Comparison vs notebook 23 (RRF + top-50 rerank) and the notebook 22 oracle ceiling

# In[ ]:


print('[Compare] Loading notebook 23 results for comparison...')
rrf_eval_df = pd.read_csv('result/23_hybrid_minilm_linq_reranker/evaluation_stage2.csv')

ORACLE_RECALL_1000 = 0.869  # from notebook 22's oracle union recall for this exact pair

comp_rows = []
for k in K_VALUES:
    full_sub = eval_df[eval_df['k'] == k]
    rrf_sub  = rrf_eval_df[rrf_eval_df['k'] == k]
    comp_rows.append({
        'k':                     k,
        'full_pool_rerank_ndcg': round(full_sub['ndcg'].mean(), 3),
        'rrf_top50_rerank_ndcg': round(rrf_sub['ndcg'].mean(), 3),
        'full_pool_rerank_recall': round(full_sub['recall'].mean(), 3),
        'rrf_top50_rerank_recall': round(rrf_sub['recall'].mean(), 3),
    })

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(RESULT_DIR / 'comparison_vs_rrf.csv', index=False)
print(comp_df.to_string(index=False))

full_r1000 = eval_df[eval_df['k']==1000]['recall'].mean()
print(f'\n[Compare] Full-pool rerank Recall@1000 : {full_r1000:.3f}')
print(f'[Compare] Oracle ceiling (notebook 22)  : {ORACLE_RECALL_1000:.3f}')
print(f'[Compare] Remaining gap                 : {ORACLE_RECALL_1000 - full_r1000:.3f}')
print(f'[Compare] Avg rerank cost               : {AVG_RERANK_S:.1f}s/query (vs 0.25s/query for RRF+top-50 in notebook 23)')
print(f'\n[Compare] Saved to {RESULT_DIR / "comparison_vs_rrf.csv"}')

