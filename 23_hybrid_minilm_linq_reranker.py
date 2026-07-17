#!/usr/bin/env python
# coding: utf-8

# # Hybrid Retrieval: MiniLM + Linq-Embed-Mistral + RRF + Cross-Encoder Reranker
# ## Testing whether this pair clears the ~0.85 oracle ceiling found in notebook 22
# 
# **Why this pairing:** notebook 22's oracle union recall analysis found MiniLM + Linq-Embed-Mistral has the highest 2-channel oracle recall@1000 (0.869) of any pair, using results that are already cached on disk -- no re-encoding needed. This notebook fuses the two with RRF, then reranks the top-50 with a cross-encoder to see how much of that 0.869 ceiling a real reranker actually captures.
# 
# ```
# Stage 1 -- Hybrid Retrieval (MiniLM + Linq-Embed-Mistral fused with RRF)
#           MiniLM top-1000              --+
#                                           +-- RRF fusion -- top-1000 candidates
#           Linq-Embed-Mistral top-1000  --+
# 
# Stage 2 -- Cross-Encoder Reranker
#           top-1000 candidates -- BGE-reranker reads query + company -- top-50 final
# ```
# 
# **Sources (all cached, no re-encoding):**
# - `result/03_baseline_minilm/minilm_results.csv`
# - `result/20_baseline_linq_mistral/20_baseline_linq_mistral_results.csv`
# 
# **Folder structure:**
# ```
# result/
#   23_hybrid_minilm_linq_reranker/
#     rrf_results.csv              # Stage 1: RRF-fused top-1000 per query
#     rrf_reranked_results.csv     # Stage 2: reranked top-50 per query
#     evaluation_stage1.csv        # NDCG/Prec/Recall/F1 for hybrid RRF
#     evaluation_stage2.csv        # NDCG/Prec/Recall/F1 for RRF + reranker
#     latency_breakdown.csv        # Per-query timing
#     comparison_all.csv           # vs MiniLM, Linq, GTE-large solo
# ```

# In[1]:


import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RESULT_DIR = Path('result/23_hybrid_minilm_linq_reranker')
RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f'[Setup] Result folder : {RESULT_DIR}/ -- ready')

RRF_K        = 60   # standard smoothing constant from Cormack et al. 2009
TOP_K_RRF    = 1000
TOP_K_RERANK = 50
print(f'[Setup] RRF k         : {RRF_K}')
print(f'[Setup] RRF top-k     : {TOP_K_RRF}')
print(f'[Setup] Rerank top-k  : {TOP_K_RERANK}')


# In[2]:


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


# ## 1. Load saved retrieval results (MiniLM + Linq-Embed-Mistral) -- both already cached, no re-encoding needed

# In[3]:


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
print(f'\n[Load] MiniLM queries : {minilm_queries}')
print(f'[Load] Linq queries   : {linq_queries}')
assert minilm_queries == linq_queries == 101, 'Query count mismatch!'
print('[Load] Sanity check passed')


# ## 2. Reciprocal Rank Fusion -- Stage 1
# 
# `RRF_score(d) = sum over rankers of 1 / (k + rank(d))`, k=60 (Cormack et al., 2009). Documents ranking highly in both channels get boosted; documents missing from a channel are simply skipped for that channel's term.

# In[4]:


def rrf_fusion(ranked_lists, k=RRF_K, top_n=TOP_K_RRF):
    scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, domain in enumerate(ranked_list, start=1):
            scores[domain] += 1.0 / (k + rank)
    sorted_results = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_results[:top_n]

print(f'[RRF] Running fusion for {len(data)} queries...')
print(f'[RRF] k={RRF_K}, top_n={TOP_K_RRF}')
print('-' * 55)

all_rrf_results = []
rrf_times       = []
total_start     = time.time()

domain_info = all_companies.set_index('domain')[['name','country','summary']].to_dict('index')

for i, item in enumerate(data):
    qid   = item['query_id']
    query = item['query']

    t0 = time.perf_counter()

    minilm_ranked = (
        minilm_df[minilm_df['query_id'] == qid]
        .sort_values('rank')['domain'].tolist()
    )
    linq_ranked = (
        linq_df[linq_df['query_id'] == qid]
        .sort_values('rank')['domain'].tolist()
    )

    fused = rrf_fusion([minilm_ranked, linq_ranked], k=RRF_K, top_n=TOP_K_RRF)
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
        print(f'[RRF] {i+1:3d}/{len(data)}  |  avg {sum(rrf_times)/len(rrf_times):.2f}ms/query  |  ~{remaining:.0f}s remaining')

rrf_df = pd.DataFrame(all_rrf_results)
rrf_df.to_csv(RESULT_DIR / 'rrf_results.csv', index=False)

AVG_RRF_MS = sum(rrf_times) / len(rrf_times)
print('-' * 55)
print(f'[RRF] Done!')
print(f'[RRF] Avg fusion time : {AVG_RRF_MS:.2f}ms/query')
print(f'[RRF] Total results   : {len(rrf_df):,}')
print(f'[RRF] Saved to        : {RESULT_DIR / "rrf_results.csv"}')

minilm_unique = set(minilm_df['domain'].unique())
linq_unique   = set(linq_df['domain'].unique())
rrf_unique    = set(rrf_df['domain'].unique())
print(f'\n[RRF] Coverage analysis:')
print(f'  MiniLM unique domains : {len(minilm_unique):,}')
print(f'  Linq unique domains   : {len(linq_unique):,}')
print(f'  RRF union             : {len(rrf_unique):,}')
print(f'  In MiniLM only        : {len(minilm_unique - linq_unique):,}')
print(f'  In Linq only          : {len(linq_unique - minilm_unique):,}')
print(f'  In both               : {len(minilm_unique & linq_unique):,}')


# ## 3. Stage 1 evaluation -- hybrid RRF

# In[5]:


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

print('[Eval1] === STAGE 1 RESULTS: Hybrid MiniLM + Linq-Embed-Mistral (RRF) ===')
print(f'  {"k":<6} {"NDCG":>8} {"Precision":>10} {"Recall":>8} {"F1":>8}')
print('  ' + '-' * 46)
for k in K_VALUES:
    sub = eval1_df[eval1_df['k'] == k]
    print(f'  {k:<6} '
          f'{sub["ndcg"].mean():>8.3f} '
          f'{sub["precision"].mean():>10.3f} '
          f'{sub["recall"].mean():>8.3f} '
          f'{sub["f1"].mean():>8.3f}')


# ## 4. Load cross-encoder reranker -- `BAAI/bge-reranker-v2-m3`, same model already used in notebook 07

# In[ ]:


print('[Reranker] Loading BGE cross-encoder reranker...')
t0                 = time.time()
RERANKER_MODEL     = 'BAAI/bge-reranker-v2-m3'
tokenizer_reranker = AutoTokenizer.from_pretrained(RERANKER_MODEL)
model_reranker     = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
model_reranker     = model_reranker.to(DEVICE)
model_reranker.eval()
print(f'[Reranker] Loaded in {time.time()-t0:.1f}s on {DEVICE}')

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

print('[Warmup] Warming up reranker with 5 dummy calls...')
sample_qid   = data[0]['query_id']
sample_query = data[0]['query']
sample_cands = rrf_df[rrf_df['query_id'] == sample_qid].head(50)
for _ in range(5):
    rerank(sample_query, sample_cands)
print('[Warmup] GPU warm')


# ## 5. Stage 2 -- cross-encoder reranker on RRF top-50
# 
# Takes the top-50 from the RRF-fused list and re-scores each one. Combined list: reranked top-50 + original RRF order for positions 51-1000.

# In[ ]:


print(f'[Stage2] Re-ranking top-{TOP_K_RERANK} for {len(data)} queries...')
print('-' * 60)

all_stage2_results = []
rerank_times       = []
total_start        = time.time()

for i, item in enumerate(data):
    qid   = item['query_id']
    query = item['query']

    rrf_query = rrf_df[rrf_df['query_id'] == qid].sort_values('rank')

    t0       = time.perf_counter()
    reranked = rerank(query, rrf_query, top_k=TOP_K_RERANK)
    rerank_ms = (time.perf_counter() - t0) * 1000
    rerank_times.append(rerank_ms)

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
        elapsed     = time.time() - total_start
        remaining_t = (len(data) - i - 1) * elapsed / (i + 1)
        steady      = sum(rerank_times[-20:]) / min(20, len(rerank_times))
        print(f'[Stage2] {i+1:3d}/{len(data)}  |  avg {sum(rerank_times)/len(rerank_times):.0f}ms  steady {steady:.0f}ms  |  ~{remaining_t:.0f}s remaining')

stage2_df = pd.DataFrame(all_stage2_results)
stage2_df.to_csv(RESULT_DIR / 'rrf_reranked_results.csv', index=False)

AVG_RERANK_MS    = sum(rerank_times) / len(rerank_times)
STEADY_RERANK_MS = sum(rerank_times[-20:]) / 20
print('-' * 60)
print(f'[Stage2] Done!')
print(f'[Stage2] Avg rerank time    : {AVG_RERANK_MS:.0f}ms')
print(f'[Stage2] Steady-state time  : {STEADY_RERANK_MS:.0f}ms  (last 20 queries)')
print(f'[Stage2] Saved to           : {RESULT_DIR / "rrf_reranked_results.csv"}')


# ## 6. Stage 2 evaluation -- RRF + reranker

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


# ## 7. Latency breakdown
# 
# MiniLM and Linq-Embed-Mistral run in PARALLEL in production, so retrieval latency is bounded by the slower of the two -- and Linq (a 7B-param model) is ~8x slower per query than MiniLM, so this hybrid pipeline is materially slower end-to-end than MiniLM alone, unlike the near-free BM25+MiniLM pairing in notebook 07.

# In[ ]:


MINILM_MS = 16.9   # from result/03_baseline_minilm run logs (logs/03_baseline_minilm_4693231.out)
LINQ_MS   = 138.0  # from result/20_baseline_linq_mistral run logs (logs/20_baseline_linq_mistral_july_5828151.out) -- 7B-param model, slower single-query encode

PARALLEL_RETRIEVAL_MS = max(MINILM_MS, LINQ_MS)
TOTAL_STAGE1_MS       = PARALLEL_RETRIEVAL_MS + AVG_RRF_MS
TOTAL_STAGE2_MS       = PARALLEL_RETRIEVAL_MS + AVG_RRF_MS + STEADY_RERANK_MS

print('[Latency] ============================================================')
print('[Latency] PIPELINE LATENCY BREAKDOWN')
print('[Latency] ============================================================')
print(f'\n  Note: MiniLM and Linq-Embed-Mistral can run in PARALLEL in production')
print(f'  Parallel retrieval latency = max(MiniLM={MINILM_MS:.1f}ms, Linq={LINQ_MS:.1f}ms)')
print()
print(f'  Component                 | Stage 1 (RRF) | Stage 2 (RRF+Reranker)')
print(f'  ------------------------- | ------------- | ----------------------')
print(f'  Parallel retrieval        |    {PARALLEL_RETRIEVAL_MS:.1f}ms    |    {PARALLEL_RETRIEVAL_MS:.1f}ms')
print(f'  RRF fusion                |    {AVG_RRF_MS:.2f}ms      |    {AVG_RRF_MS:.2f}ms')
print(f'  Cross-encoder reranker    |      N/A       |    {STEADY_RERANK_MS:.0f}ms')
print(f'  ------------------------- | ------------- | ----------------------')
print(f'  TOTAL                     |    {TOTAL_STAGE1_MS:.1f}ms    |    {TOTAL_STAGE2_MS:.0f}ms')
print()
print(f'  MiniLM baseline (single)  :  {MINILM_MS:.1f}ms')
print(f'  Hybrid RRF                :  {TOTAL_STAGE1_MS:.1f}ms  (+{TOTAL_STAGE1_MS-MINILM_MS:.1f}ms vs MiniLM)')
print(f'  Hybrid RRF + Reranker     :  {TOTAL_STAGE2_MS:.0f}ms  (+{TOTAL_STAGE2_MS-MINILM_MS:.0f}ms vs MiniLM)')

latency_df = pd.DataFrame([{
    'pipeline':     'MiniLM baseline',
    'retrieval_ms': MINILM_MS,
    'rrf_ms':       0,
    'rerank_ms':    0,
    'total_ms':     MINILM_MS,
}, {
    'pipeline':     'Hybrid RRF (MiniLM + Linq)',
    'retrieval_ms': PARALLEL_RETRIEVAL_MS,
    'rrf_ms':       round(AVG_RRF_MS, 2),
    'rerank_ms':    0,
    'total_ms':     round(TOTAL_STAGE1_MS, 1),
}, {
    'pipeline':     'Hybrid RRF + Reranker',
    'retrieval_ms': PARALLEL_RETRIEVAL_MS,
    'rrf_ms':       round(AVG_RRF_MS, 2),
    'rerank_ms':    round(STEADY_RERANK_MS, 0),
    'total_ms':     round(TOTAL_STAGE2_MS, 0),
}])
latency_df.to_csv(RESULT_DIR / 'latency_breakdown.csv', index=False)
print(f'\n[Latency] Saved to {RESULT_DIR / "latency_breakdown.csv"}')


# ## 8. Final comparison -- vs MiniLM, Linq-Embed-Mistral, and GTE-large (best solo channel) alone

# In[ ]:


print('[Compare] Loading all baseline evaluations...')

def load_eval(path, method_name):
    try:
        df = pd.read_csv(path)
        df['method'] = method_name
        print(f'  loaded {method_name}')
        return df
    except FileNotFoundError:
        print(f'  missing {method_name} -- file not found: {path}')
        return None

all_evals = {
    'MiniLM':                load_eval('result/03_baseline_minilm/evaluation_minilm.csv', 'MiniLM'),
    'Linq_Mistral':          load_eval('result/20_baseline_linq_mistral/evaluation_20_baseline_linq_mistral.csv', 'Linq_Mistral'),
    'GTE_large':             load_eval('result/17_baseline_gte_large/evaluation_gte_large.csv', 'GTE_large'),
    'Hybrid RRF':            eval1_df,
    'Hybrid RRF + Reranker': eval2_df,
}

KEY_METHODS = ['MiniLM', 'Linq_Mistral', 'GTE_large', 'Hybrid RRF', 'Hybrid RRF + Reranker']

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
        comp_rows.append({'method': method, 'k': k, 'ndcg': round(ndcg,3), 'precision': round(prec,3), 'recall': round(rec,3), 'f1': round(f1,3)})
    print('  ' + '-' * 75)

pd.DataFrame(comp_rows).to_csv(RESULT_DIR / 'comparison_all.csv', index=False)
print(f'\n[Compare] Full table saved to {RESULT_DIR / "comparison_all.csv"}')


# ## 9. Key findings

# In[ ]:


print('[Findings] ============================================================')
print('[Findings] KEY FINDINGS -- HYBRID MINILM + LINQ-MISTRAL PIPELINE')
print('[Findings] ============================================================')

def get_ndcg10(df):
    return df[df['k']==10]['ndcg'].mean() if df is not None else None

def get_rec1000(df):
    return df[df['k']==1000]['recall'].mean() if df is not None else None

minilm_n10 = get_ndcg10(all_evals.get('MiniLM'))
linq_n10   = get_ndcg10(all_evals.get('Linq_Mistral'))
rrf_n10    = get_ndcg10(eval1_df)
rrfr_n10   = get_ndcg10(eval2_df)

print(f'\n[Findings] NDCG@10 progression:')
if minilm_n10: print(f'  MiniLM baseline            : {minilm_n10:.3f}')
if linq_n10:   print(f'  Linq-Embed-Mistral baseline: {linq_n10:.3f}')
print(f'  Hybrid RRF                 : {rrf_n10:.3f}  ({(rrf_n10-minilm_n10)/minilm_n10*100:+.1f}% vs MiniLM)')
print(f'  Hybrid RRF + Reranker      : {rrfr_n10:.3f}  ({(rrfr_n10-minilm_n10)/minilm_n10*100:+.1f}% vs MiniLM)')

rrf_r1000    = get_rec1000(eval1_df)
minilm_r1000 = get_rec1000(all_evals.get('MiniLM'))
oracle_r1000 = 0.869  # from notebook 22's oracle union recall for this exact pair (MiniLM + Linq_Mistral)
print(f'\n[Findings] Recall@1000:')
if minilm_r1000: print(f'  MiniLM baseline            : {minilm_r1000:.3f}')
print(f'  Hybrid RRF                 : {rrf_r1000:.3f}  ({(rrf_r1000-minilm_r1000)/minilm_r1000*100:+.1f}% vs MiniLM)')
print(f'  Oracle ceiling (notebook 22): {oracle_r1000:.3f}  -- gap of {oracle_r1000-rrf_r1000:.3f} is what RRF leaves on the table vs a perfect reranker')

print(f'\n[Findings] Latency:')
print(f'  MiniLM baseline            :  {MINILM_MS:.1f}ms')
print(f'  Hybrid RRF                 :  {TOTAL_STAGE1_MS:.1f}ms (parallel retrieval + RRF)')
print(f'  Hybrid RRF + Reranker      :  {TOTAL_STAGE2_MS:.0f}ms')
print('[Findings] ============================================================')

