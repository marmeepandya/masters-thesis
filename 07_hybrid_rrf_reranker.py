#!/usr/bin/env python
# coding: utf-8

# # Hybrid Retrieval: BM25 + MiniLM + RRF + Cross-Encoder Reranker
# ## The complete SOTA retrieval pipeline
# 
# **What this notebook does:**
# 
# Combines all previous methods into one unified pipeline:
# 
# ```
# Stage 1 — Hybrid Retrieval (BM25 + MiniLM fused with RRF)
#           BM25 top-1000   ──┐
#                              ├── RRF fusion ── top-1000 candidates
#           MiniLM top-1000 ──┘
# 
# Stage 2 — Cross-Encoder Reranker
#           top-1000 candidates ── BGE-reranker reads query + company ── top-50 final
# ```
# 
# **Why hybrid RRF works:**
# BM25 and MiniLM make complementary errors:
# - BM25 misses companies that use different vocabulary (vocabulary mismatch)
# - MiniLM misses companies with very specific terms or exact entity names
# - A company ranking high in BOTH gets a strong RRF score
# - Union of two lists = higher recall than either alone
# 
# **RRF formula:**
# ```
# RRF_score(d) = Σ 1 / (k + rank(d))   for each ranker
# ```
# where k=60 is a smoothing constant. No training needed — parameter-free.
# 
# **Folder structure:**
# ```
# result/
# └── 07_hybrid_rrf_reranker/
#     ├── rrf_results.csv              # Stage 1: RRF-fused top-1000 per query
#     ├── rrf_reranked_results.csv     # Stage 2: reranked top-50 per query
#     ├── evaluation_stage1.csv        # NDCG/Prec/Recall/F1 for hybrid RRF
#     ├── evaluation_stage2.csv        # NDCG/Prec/Recall/F1 for RRF + reranker
#     ├── latency_breakdown.csv        # Per-query timing
#     └── comparison_all.csv           # vs all previous methods
# ```
# 
# ### Notebook structure
# 1. Environment setup
# 2. Imports
# 3. Load saved retrieval results (BM25 + MiniLM)
# 4. RRF fusion — Stage 1
# 5. Stage 1 evaluation
# 6. Load cross-encoder reranker
# 7. Rerank top-50 — Stage 2
# 8. Stage 2 evaluation
# 9. Latency breakdown
# 10. Final comparison — all methods
# 11. Key findings

# ## 1 · Environment Setup

# In[ ]:


import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RESULT_DIR = Path('result/07_hybrid_rrf_reranker')
RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f'[Setup] Result folder : {RESULT_DIR}/ — ready')

# RRF hyperparameter — k=60 is the standard from Cormack et al. 2009
# Higher k = less weight on top ranks, more uniform fusion
# k=60 is used universally in the literature
RRF_K      = 1000
TOP_K_RRF  = 1000   # candidates after RRF fusion
TOP_K_RERANK = 50   # how many to pass to cross-encoder
print(f'[Setup] RRF k         : {RRF_K}')
print(f'[Setup] RRF top-k     : {TOP_K_RRF}')
print(f'[Setup] Rerank top-k  : {TOP_K_RERANK}')


# ## 2 · Imports
# 
# | Package | Role |
# |---|---|
# | `pandas / numpy` | Load saved retrieval results and compute metrics |
# | `transformers` | BGE cross-encoder reranker |
# | `torch` | GPU detection |
# | `time` | Latency measurement |
# | `collections.defaultdict` | RRF score accumulation |

# In[ ]:


import json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[Imports] Device      : {DEVICE}')
if torch.cuda.is_available():
    print(f'[Imports] GPU         : {torch.cuda.get_device_name(0)}')
print('[Imports] All packages loaded successfully')


# ## 3 · Load Saved Retrieval Results
# 
# We reuse the already-computed BM25 and MiniLM results.
# No re-encoding or re-indexing needed — RRF only needs the ranked lists.
# 
# **Sources:**
# - `result/01_baseline_bm25/bm25_results.csv` — BM25 top-1000 per query
# - `result/03_baseline_minilm/minilm_results.csv` — MiniLM top-1000 per query

# In[ ]:


print('[Load] Loading BM25 results...')
bm25_df = pd.read_csv('result/01_baseline_bm25/bm25_results.csv')
print(f'[Load] BM25 rows      : {len(bm25_df):,}')

print('[Load] Loading MiniLM results...')
minilm_df = pd.read_csv('result/03_baseline_minilm/minilm_results.csv')
print(f'[Load] MiniLM rows    : {len(minilm_df):,}')

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

# Sanity check — both should have results for all 101 queries
bm25_queries   = bm25_df['query_id'].nunique()
minilm_queries = minilm_df['query_id'].nunique()
print(f'\n[Load] BM25 queries   : {bm25_queries}')
print(f'[Load] MiniLM queries : {minilm_queries}')
assert bm25_queries == minilm_queries == 101, 'Query count mismatch!'
print('[Load] Sanity check passed ✅')


# ## 4 · Reciprocal Rank Fusion — Stage 1
# 
# **RRF algorithm (Cormack et al., 2009):**
# 
# For each document d appearing in any ranked list:
# ```
# RRF_score(d) = Σ  1 / (k + rank_i(d))
#                i
# ```
# where k=60 is the smoothing constant and rank_i(d) is the rank of document d
# in ranker i (1-indexed). Documents not appearing in a list are ignored.
# 
# **Why k=60?**
# Cormack et al. showed k=60 is robust across many retrieval tasks. Lower k
# gives more weight to top-ranked documents; higher k makes fusion more uniform.
# k=60 is the standard default used universally in the literature.
# 
# **Key property:** RRF boosts documents that rank highly in BOTH lists.
# A document ranked #1 in BM25 and #1 in MiniLM gets score 2/(60+1) = 0.033.
# A document ranked #1 in only one list gets 1/(60+1) = 0.016.

# In[ ]:


def rrf_fusion(ranked_lists, k=RRF_K, top_n=TOP_K_RRF):
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists : list of lists of domain strings, each sorted by rank
        k            : RRF smoothing constant (default 60)
        top_n        : number of results to return

    Returns:
        list of (domain, rrf_score) tuples sorted by score descending
    """
    scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, domain in enumerate(ranked_list, start=1):
            scores[domain] += 1.0 / (k + rank)

    # Sort by score descending, return top-n
    sorted_results = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_results[:top_n]


print(f'[RRF] Running fusion for {len(data)} queries...')
print(f'[RRF] k={RRF_K}, top_n={TOP_K_RRF}')
print('-' * 55)

all_rrf_results = []
rrf_times       = []
total_start     = time.time()

# Build domain → company info lookup for fast access
domain_info = all_companies.set_index('domain')[['name','country','summary']].to_dict('index')

for i, item in enumerate(data):
    qid   = item['query_id']
    query = item['query']

    # ── Get ranked domain lists for this query ────────────────────────────────
    t0 = time.perf_counter()

    bm25_ranked = (
        bm25_df[bm25_df['query_id'] == qid]
        .sort_values('rank')['domain'].tolist()
    )
    minilm_ranked = (
        minilm_df[minilm_df['query_id'] == qid]
        .sort_values('rank')['domain'].tolist()
    )

    # ── RRF fusion ────────────────────────────────────────────────────────────
    fused = rrf_fusion([bm25_ranked, minilm_ranked], k=RRF_K, top_n=TOP_K_RRF)
    rrf_ms = (time.perf_counter() - t0) * 1000
    rrf_times.append(rrf_ms)

    for rank, (domain, rrf_score) in enumerate(fused, start=1):
        info = domain_info.get(domain, {})
        all_rrf_results.append({
            'query_id':  qid,
            'query':     query,
            'rank':      rank,
            'rrf_score': rrf_score,
            'domain':    domain,
            'name':      info.get('name', ''),
            'country':   info.get('country', ''),
            'summary':   info.get('summary', ''),
        })

    if (i + 1) % 20 == 0 or (i + 1) == len(data):
        elapsed   = time.time() - total_start
        remaining = (len(data) - i - 1) * elapsed / (i + 1)
        print(f'[RRF] {i+1:3d}/{len(data)}  |  '
              f'avg {sum(rrf_times)/len(rrf_times):.2f}ms/query  |  '
              f'~{remaining:.0f}s remaining')

rrf_df = pd.DataFrame(all_rrf_results)
rrf_df.to_csv(RESULT_DIR / 'rrf_results.csv', index=False)

AVG_RRF_MS = sum(rrf_times) / len(rrf_times)
print('-' * 55)
print(f'[RRF] Done!')
print(f'[RRF] Avg fusion time : {AVG_RRF_MS:.2f}ms/query')
print(f'[RRF] Total results   : {len(rrf_df):,}')
print(f'[RRF] Saved to        : result/07_hybrid_rrf_reranker/rrf_results.csv')

# ── Coverage check — how many unique companies did RRF find? ─────────────────
bm25_unique   = set(bm25_df['domain'].unique())
minilm_unique = set(minilm_df['domain'].unique())
rrf_unique    = set(rrf_df['domain'].unique())
print(f'\n[RRF] Coverage analysis:')
print(f'  BM25 unique domains   : {len(bm25_unique):,}')
print(f'  MiniLM unique domains : {len(minilm_unique):,}')
print(f'  RRF union             : {len(rrf_unique):,}')
print(f'  In BM25 only          : {len(bm25_unique - minilm_unique):,}')
print(f'  In MiniLM only        : {len(minilm_unique - bm25_unique):,}')
print(f'  In both               : {len(bm25_unique & minilm_unique):,}')


# ## 5 · Stage 1 Evaluation — Hybrid RRF
# 
# Evaluate the fused ranking before reranking.
# Expect improvement over both individual methods — especially Recall@k
# since the union of two lists covers more relevant companies.

# In[ ]:


K_VALUES = [10, 50, 100, 500, 1000]

def get_relevant(query_id, top_k=100):
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

print('[Eval1] Evaluating Stage 1 (Hybrid RRF)...')
eval1_rows = []

for i, item in enumerate(data):
    qid       = item['query_id']
    relevant  = get_relevant(qid)
    retrieved = (
        rrf_df[rrf_df['query_id'] == qid]
        .sort_values('rank')['domain'].tolist()
    )
    for k in K_VALUES:
        eval1_rows.append({
            'query_id':  qid,
            'query':     item['query'],
            'k':         k,
            'precision': precision_at_k(retrieved, relevant, k),
            'recall':    recall_at_k(retrieved, relevant, k),
            'f1':        f1_at_k(retrieved, relevant, k),
            'ndcg':      ndcg_at_k(retrieved, relevant, k),
        })

    if (i + 1) % 25 == 0:
        print(f'[Eval1] {i+1}/101 queries evaluated...')

eval1_df = pd.DataFrame(eval1_rows)
eval1_df.to_csv(RESULT_DIR / 'evaluation_stage1.csv', index=False)

print('[Eval1] === STAGE 1 RESULTS: Hybrid BM25 + MiniLM (RRF) ===')
print(f'  {"k":<6} {"NDCG":>8} {"Precision":>10} {"Recall":>8} {"F1":>8}')
print('  ' + '-' * 46)
for k in K_VALUES:
    sub = eval1_df[eval1_df['k'] == k]
    print(f'  {k:<6} '
          f'{sub["ndcg"].mean():>8.3f} '
          f'{sub["precision"].mean():>10.3f} '
          f'{sub["recall"].mean():>8.3f} '
          f'{sub["f1"].mean():>8.3f}')


# ## 6 · Load Cross-Encoder Reranker
# 
# Same model as previous experiments: `BAAI/bge-reranker-v2-m3`.
# Already downloaded — loads from local cache.

# In[ ]:


print('[Reranker] Loading BGE cross-encoder reranker...')
t0                 = time.time()
RERANKER_MODEL     = 'BAAI/bge-reranker-v2-m3'
tokenizer_reranker = AutoTokenizer.from_pretrained(RERANKER_MODEL)
model_reranker     = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
model_reranker     = model_reranker.to(DEVICE)
model_reranker.eval()
print(f'[Reranker] Loaded in {time.time()-t0:.1f}s on {DEVICE}')

# ── GPU warmup ────────────────────────────────────────────────────────────────
print('[Warmup] Warming up reranker with 5 dummy calls...')
sample_qid   = data[0]['query_id']
sample_query = data[0]['query']
sample_cands = rrf_df[rrf_df['query_id'] == sample_qid].head(50)

def rerank(query, candidates_df, top_k=TOP_K_RERANK, batch_size=32):
    candidates = candidates_df.head(top_k).copy()
    summaries  = candidates['summary'].fillna('').tolist()
    pairs      = [[query, s] for s in summaries]
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch   = pairs[i:i+batch_size]
            encoded = tokenizer_reranker(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors='pt'
            ).to(DEVICE)
            scores = model_reranker(**encoded).logits.squeeze(-1)
            all_scores.extend(scores.cpu().float().tolist())
    candidates['reranker_score'] = all_scores
    candidates['rrf_rank']       = candidates['rank'].values
    reranked = candidates.sort_values('reranker_score', ascending=False).reset_index(drop=True)
    reranked['rank'] = reranked.index + 1
    return reranked

for _ in range(5):
    rerank(sample_query, sample_cands)
print('[Warmup] GPU warm ✅')


# ## 7 · Stage 2 — Cross-Encoder Reranker on RRF top-50
# 
# Takes the top-50 from the RRF-fused list and re-scores each one.
# Combined list: reranked top-50 + original RRF order for positions 51-1000.

# In[7]:


print(f'[Stage2] Re-ranking top-{TOP_K_RERANK} for {len(data)} queries...')
print('-' * 60)

all_stage2_results = []
rerank_times       = []
total_start        = time.time()

for i, item in enumerate(data):
    qid   = item['query_id']
    query = item['query']

    rrf_query = rrf_df[rrf_df['query_id'] == qid].sort_values('rank')

    # ── Rerank top-50 ─────────────────────────────────────────────────────────
    t0       = time.perf_counter()
    reranked = rerank(query, rrf_query, top_k=TOP_K_RERANK)
    rerank_ms = (time.perf_counter() - t0) * 1000
    rerank_times.append(rerank_ms)

    # ── Combined: reranked top-50 + RRF positions 51-1000 ────────────────────
    reranked_domains = set(reranked['domain'].tolist())
    remaining        = rrf_query[~rrf_query['domain'].isin(reranked_domains)]

    for _, row in reranked.iterrows():
        all_stage2_results.append({
            'query_id':       qid,
            'query':          query,
            'rank':           int(row['rank']),
            'rrf_rank':       int(row['rrf_rank']),
            'reranker_score': float(row['reranker_score']),
            'rrf_score':      float(row['rrf_score']),
            'domain':         row['domain'],
            'name':           row.get('name', ''),
            'summary':        row.get('summary', ''),
        })

    for rank_offset, (_, row) in enumerate(remaining.iterrows(), start=51):
        all_stage2_results.append({
            'query_id':       qid,
            'query':          query,
            'rank':           rank_offset,
            'rrf_rank':       int(row['rank']),
            'reranker_score': None,
            'rrf_score':      float(row['rrf_score']),
            'domain':         row['domain'],
            'name':           row.get('name', ''),
            'summary':        row.get('summary', ''),
        })

    if (i + 1) % 20 == 0 or (i + 1) == len(data):
        elapsed   = time.time() - total_start
        remaining_t = (len(data) - i - 1) * elapsed / (i + 1)
        steady    = sum(rerank_times[-20:]) / min(20, len(rerank_times))
        print(f'[Stage2] {i+1:3d}/{len(data)}  |  '
              f'avg {sum(rerank_times)/len(rerank_times):.0f}ms  '
              f'steady {steady:.0f}ms  |  '
              f'~{remaining_t:.0f}s remaining')

stage2_df = pd.DataFrame(all_stage2_results)
stage2_df.to_csv(RESULT_DIR / 'rrf_reranked_results.csv', index=False)

AVG_RERANK_MS        = sum(rerank_times) / len(rerank_times)
STEADY_RERANK_MS     = sum(rerank_times[-20:]) / 20
print('-' * 60)
print(f'[Stage2] Done!')
print(f'[Stage2] Avg rerank time    : {AVG_RERANK_MS:.0f}ms')
print(f'[Stage2] Steady-state time  : {STEADY_RERANK_MS:.0f}ms  (last 20 queries)')
print(f'[Stage2] Saved to           : result/07_hybrid_rrf_reranker/rrf_reranked_results.csv')


# ## 8 · Stage 2 Evaluation — RRF + Reranker

# In[ ]:


print('[Eval2] Evaluating Stage 2 (RRF + Reranker)...')
eval2_rows = []

for i, item in enumerate(data):
    qid       = item['query_id']
    relevant  = get_relevant(qid)
    retrieved = (
        stage2_df[stage2_df['query_id'] == qid]
        .sort_values('rank')['domain'].tolist()
    )
    for k in K_VALUES:
        eval2_rows.append({
            'query_id':  qid,
            'query':     item['query'],
            'k':         k,
            'precision': precision_at_k(retrieved, relevant, k),
            'recall':    recall_at_k(retrieved, relevant, k),
            'f1':        f1_at_k(retrieved, relevant, k),
            'ndcg':      ndcg_at_k(retrieved, relevant, k),
        })

    if (i + 1) % 25 == 0:
        print(f'[Eval2] {i+1}/101 queries evaluated...')

eval2_df = pd.DataFrame(eval2_rows)
eval2_df.to_csv(RESULT_DIR / 'evaluation_stage2.csv', index=False)

print('[Eval2] === STAGE 2 RESULTS: RRF + Reranker ===')
print(f'  {"k":<6} {"NDCG":>8} {"Precision":>10} {"Recall":>8} {"F1":>8}')
print('  ' + '-' * 46)
for k in K_VALUES:
    sub = eval2_df[eval2_df['k'] == k]
    print(f'  {k:<6} '
          f'{sub["ndcg"].mean():>8.3f} '
          f'{sub["precision"].mean():>10.3f} '
          f'{sub["recall"].mean():>8.3f} '
          f'{sub["f1"].mean():>8.3f}')


# ## 9 · Latency Breakdown
# 
# **RRF adds near-zero latency** — it is pure Python dictionary operations
# on already-computed ranked lists. No model inference, no GPU needed.
# 
# Total pipeline latency = MiniLM query latency + RRF fusion + Reranker

# In[ ]:


MINILM_MS   = 16.9   # from baseline
BM25_MS     = 102.7  # from baseline (CPU)

# In production BM25 and MiniLM run in parallel
# So pipeline latency = max(BM25, MiniLM) + RRF + Reranker
PARALLEL_RETRIEVAL_MS = max(BM25_MS, MINILM_MS)  # = 102.7ms
TOTAL_STAGE1_MS       = PARALLEL_RETRIEVAL_MS + AVG_RRF_MS
TOTAL_STAGE2_MS       = PARALLEL_RETRIEVAL_MS + AVG_RRF_MS + STEADY_RERANK_MS

print('[Latency] ============================================================')
print('[Latency] PIPELINE LATENCY BREAKDOWN')
print('[Latency] ============================================================')
print(f'\n  Note: BM25 and MiniLM can run in PARALLEL in production')
print(f'  Parallel retrieval latency = max(BM25={BM25_MS:.1f}ms, MiniLM={MINILM_MS:.1f}ms)')
print()
print(f'  Component                 | Stage 1 (RRF) | Stage 2 (RRF+Reranker)')
print(f'  ------------------------- | ------------- | ----------------------')
print(f'  Parallel retrieval        |    {PARALLEL_RETRIEVAL_MS:.1f}ms     |    {PARALLEL_RETRIEVAL_MS:.1f}ms')
print(f'  RRF fusion                |    {AVG_RRF_MS:.2f}ms      |    {AVG_RRF_MS:.2f}ms')
print(f'  Cross-encoder reranker    |      N/A       |    {STEADY_RERANK_MS:.0f}ms')
print(f'  ------------------------- | ------------- | ----------------------')
print(f'  TOTAL                     |    {TOTAL_STAGE1_MS:.1f}ms     |    {TOTAL_STAGE2_MS:.0f}ms')
print()
print(f'  MiniLM baseline (single)  :  16.9ms')
print(f'  Hybrid RRF                :  {TOTAL_STAGE1_MS:.1f}ms  (+{TOTAL_STAGE1_MS-MINILM_MS:.1f}ms vs MiniLM)')
print(f'  Hybrid RRF + Reranker     :  {TOTAL_STAGE2_MS:.0f}ms  (+{TOTAL_STAGE2_MS-MINILM_MS:.0f}ms vs MiniLM)')

latency_df = pd.DataFrame([{
    'pipeline':          'MiniLM baseline',
    'retrieval_ms':      MINILM_MS,
    'rrf_ms':            0,
    'rerank_ms':         0,
    'total_ms':          MINILM_MS,
}, {
    'pipeline':          'Hybrid RRF (BM25 + MiniLM)',
    'retrieval_ms':      PARALLEL_RETRIEVAL_MS,
    'rrf_ms':            round(AVG_RRF_MS, 2),
    'rerank_ms':         0,
    'total_ms':          round(TOTAL_STAGE1_MS, 1),
}, {
    'pipeline':          'Hybrid RRF + Reranker',
    'retrieval_ms':      PARALLEL_RETRIEVAL_MS,
    'rrf_ms':            round(AVG_RRF_MS, 2),
    'rerank_ms':         round(STEADY_RERANK_MS, 0),
    'total_ms':          round(TOTAL_STAGE2_MS, 0),
}])
latency_df.to_csv(RESULT_DIR / 'latency_breakdown.csv', index=False)
print(f'\n[Latency] Saved to result/07_hybrid_rrf_reranker/latency_breakdown.csv')


# ## 10 · Final Comparison — All Methods
# 
# Complete comparison table across every method tested in this thesis.
# This is the master results table for the thesis experiments chapter.

# In[ ]:


print('[Compare] Loading all baseline evaluations...')

# Load all previous evaluation files
def load_eval(path, method_name):
    try:
        df = pd.read_csv(path)
        df['method'] = method_name
        print(f'  ✅ {method_name}')
        return df
    except FileNotFoundError:
        print(f'  ⚠️  {method_name} — file not found: {path}')
        return None

all_evals = {
    'BM25':                    load_eval('result/01_baseline_bm25/evaluation_bm25.csv', 'BM25'),
    'MiniLM':                  load_eval('result/03_baseline_minilm/evaluation_minilm.csv', 'MiniLM'),
    'BGE':                     load_eval('result/02_baseline_bge/evaluation_bge.csv', 'BGE'),
    'Nomic':                   load_eval('result/05_baseline_nomic/evaluation_nomic.csv', 'Nomic'),
    'OpenAI-large':            load_eval('result/04_baseline_openai_large/evaluation_openai_large.csv', 'OpenAI-large'),
    'Hybrid RRF':              eval1_df,
    'Hybrid RRF + Reranker':   eval2_df,
}

# Focus on key methods for the main table
KEY_METHODS = ['BM25', 'MiniLM', 'BGE', 'Hybrid RRF', 'Hybrid RRF + Reranker']

print('\n' + '=' * 75)
print(f'  {"Method":<28} {"k":>6} | {"NDCG":>7} | {"Prec":>7} | {"Recall":>7} | {"F1":>7}')
print('  ' + '=' * 75)

comp_rows = []
for method in KEY_METHODS:
    df = all_evals.get(method)
    if df is None:
        continue
    for k in K_VALUES:
        sub  = df[df['k'] == k]
        ndcg = sub['ndcg'].mean()
        prec = sub['precision'].mean()
        rec  = sub['recall'].mean()
        f1   = sub['f1'].mean()
        print(f'  {method:<28} {k:>6} | {ndcg:>7.3f} | {prec:>7.3f} | {rec:>7.3f} | {f1:>7.3f}')
        comp_rows.append({'method':method,'k':k,
                          'ndcg':round(ndcg,3),'precision':round(prec,3),
                          'recall':round(rec,3),'f1':round(f1,3)})
    print('  ' + '-' * 75)

pd.DataFrame(comp_rows).to_csv(RESULT_DIR / 'comparison_all.csv', index=False)
print(f'\n[Compare] Full table saved to result/07_hybrid_rrf_reranker/comparison_all.csv')


# ## 11 · Key Findings
# 
# Auto-generated summary comparing all stages.

# In[ ]:


print('[Findings] ============================================================')
print('[Findings] KEY FINDINGS — HYBRID RRF PIPELINE')
print('[Findings] ============================================================')

def get_ndcg10(df):
    return df[df['k']==10]['ndcg'].mean() if df is not None else None

def get_prec10(df):
    return df[df['k']==10]['precision'].mean() if df is not None else None

def get_rec1000(df):
    return df[df['k']==1000]['recall'].mean() if df is not None else None

bm25_n10   = get_ndcg10(all_evals.get('BM25'))
mini_n10   = get_ndcg10(all_evals.get('MiniLM'))
rrf_n10    = get_ndcg10(eval1_df)
rrfr_n10   = get_ndcg10(eval2_df)

print(f'\n[Findings] NDCG@10 progression:')
if bm25_n10:  print(f'  BM25 baseline            : {bm25_n10:.3f}')
if mini_n10:  print(f'  MiniLM baseline          : {mini_n10:.3f}')
print(f'  Hybrid RRF               : {rrf_n10:.3f}  ({(rrf_n10-mini_n10)/mini_n10*100:+.1f}% vs MiniLM)')
print(f'  Hybrid RRF + Reranker    : {rrfr_n10:.3f}  ({(rrfr_n10-mini_n10)/mini_n10*100:+.1f}% vs MiniLM)')

rrf_r1000  = get_rec1000(eval1_df)
mini_r1000 = get_rec1000(all_evals.get('MiniLM'))
print(f'\n[Findings] Recall@1000:')
if mini_r1000: print(f'  MiniLM baseline          : {mini_r1000:.3f}')
print(f'  Hybrid RRF               : {rrf_r1000:.3f}  ({(rrf_r1000-mini_r1000)/mini_r1000*100:+.1f}% vs MiniLM)')

print(f'\n[Findings] Latency:')
print(f'  MiniLM baseline          :  {MINILM_MS:.1f}ms')
print(f'  Hybrid RRF               :  {TOTAL_STAGE1_MS:.1f}ms (parallel retrieval + RRF)')
print(f'  Hybrid RRF + Reranker    :  {TOTAL_STAGE2_MS:.0f}ms')
print('[Findings] ============================================================')

