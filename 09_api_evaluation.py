#!/usr/bin/env python
# coding: utf-8

# # Real Evaluation: Our Pipeline vs Istari Production API
# ## Full evaluation: 101 queries × 1000 results with graded GT relevance
# 
# **What this notebook does:**
# 
# For all 101 queries, compares retrieval results against the GOI ground truth
# (`goi_search_results.json`) provided by Manav.
# 
# **Three systems compared:**
# 1. **Istari API BM25** — production keyword search on full 20M GOI
# 2. **Istari API BM25 + Filter** — keyword + country filter on 20M GOI
# 3. **Our MiniLM** — our best method on 98,716 corpus
# 
# **Ground truth:** `goi_search_results.json` — Istari's production semantic search results.
# Relevance = GT similarity score (graded, not binary). Threshold = 0.5 for Precision/Recall.
# 
# **Folder structure:**
# ```
# result/
# └── 09_api_evaluation/
#     ├── api_results_cache.pkl          # Cache — resume after timeout
#     ├── api_bm25_results.json          # Istari BM25 top-1000 all queries
#     ├── api_bm25_filter_results.json   # Istari BM25+Filter top-1000 all queries
#     ├── our_minilm_results.json        # Our MiniLM top-1000 all queries
#     ├── latency_per_query.csv          # Per-query latency for all 3 methods
#     └── evaluation_final.csv           # NDCG/Prec/Recall at k∈{10,50,100,500,1000}
# ```

# ## 1 · Environment Setup

# In[2]:


import os, json, time, pickle, requests
import numpy as np
import pandas as pd
import faiss
import torch
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

ISTARI_API_KEY  = os.getenv('API_KEY')
ISTARI_BASE_URL = 'https://api.istari.ai/v1/search'

RESULT_DIR = Path('result/09_api_evaluation')
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

API_COLUMNS = [
    'domain', 'name', 'country', 'state', 'municipality',
    'organization_type', 'organization_size', 'nace_code', 'summary'
]

print(f'[Setup] Device         : {DEVICE}')
if torch.cuda.is_available():
    print(f'[Setup] GPU            : {torch.cuda.get_device_name(0)}')
print(f'[Setup] Result folder  : {RESULT_DIR}/')

# Quick API test
print('[Setup] Testing Istari API v1...')
r = requests.post(
    ISTARI_BASE_URL,
    headers={'x-api-key': ISTARI_API_KEY, 'Content-Type': 'application/json'},
    json={'keywords': {'must_one': ['software'], 'must_all': [], 'must_not': []},
          'locations': {'country': [], 'state': [], 'region': []},
          'custom_filters': {}, 'excludes': [],
          'columns': ['domain', 'name'], 'size': 3},
    timeout=15
)
print(f'[Setup] API status     : {r.status_code} ✅' if r.status_code == 200
      else f'[Setup] API ERROR      : {r.status_code} {r.text[:100]}')


# ## 2 · Load All 101 Queries

# In[3]:


queries_df = pd.read_excel('dataset/queries.xlsx')
print(f'[Queries] Columns    : {list(queries_df.columns)}')
print(queries_df.head(6).to_string())

# Auto-detect text query column
query_col = None
for col in queries_df.columns:
    if queries_df[col].dtype == object:
        query_col = col
        break
if query_col is None:
    query_col = queries_df.columns[0]

# Use ALL queries
TEST_QUERIES = queries_df[query_col].astype(str).tolist()
print(f'\n[Queries] Using column : "{query_col}"')
print(f'[Queries] Total queries: {len(TEST_QUERIES)}')
print(f'[Queries] First 5      : {TEST_QUERIES[:5]}')


# ## 3 · Istari v1 API — Fetch 1000 via 2×500

# In[3]:


COUNTRY_MAP = {
    'germany': 'Germany', 'german': 'Germany',
    'usa': 'United States', 'us': 'United States', 'america': 'United States',
    'uk': 'United Kingdom', 'britain': 'United Kingdom',
    'france': 'France', 'india': 'India', 'china': 'China', 'japan': 'Japan',
    'austria': 'Austria', 'switzerland': 'Switzerland', 'spain': 'Spain',
    'italy': 'Italy', 'netherlands': 'Netherlands', 'sweden': 'Sweden',
}
STOPWORDS = {
    'in','the','and','or','for','of','a','an','with','by',
    'companies','company','firms','firm','providers','provider',
    'services','service','solutions','solution',
}

def parse_query(query):
    tokens   = query.lower().split()
    country  = None
    keywords = []
    for token in tokens:
        clean = token.strip('.,;()')
        if clean in COUNTRY_MAP:
            country = COUNTRY_MAP[clean]
        elif clean not in STOPWORDS and len(clean) > 2:
            keywords.append(clean)
    return keywords, country


def istari_fetch_1000(query, mode='bm25'):
    """
    Fetch top-1000 results from Istari v1 API using 2 × size=500.
    mode: 'bm25' | 'bm25_filter'
    Returns (results_list, total_latency_ms)
    """
    keywords, country = parse_query(query)

    def payload(offset):
        return {
            'keywords':       {'must_one': keywords, 'must_all': [], 'must_not': []},
            'locations':      {'country': [country] if (mode=='bm25_filter' and country) else [],
                               'state': [], 'region': []},
            'custom_filters': {},
            'excludes':       [],
            'columns':        API_COLUMNS,
            'size':           500,
            'from':           offset,
        }

    all_results = []
    total_ms    = 0

    for offset in [0, 500]:
        t0 = time.perf_counter()
        r  = requests.post(
            ISTARI_BASE_URL,
            headers={'x-api-key': ISTARI_API_KEY, 'Content-Type': 'application/json'},
            json=payload(offset), timeout=60
        )
        ms = (time.perf_counter() - t0) * 1000
        total_ms += ms

        if r.status_code != 200:
            print(f'    [API] ERROR offset={offset}: {r.status_code} {r.text[:80]}')
            break
        all_results.extend(r.json().get('data', []))
        time.sleep(0.2)

    for rank, res in enumerate(all_results, 1):
        res['rank'] = rank

    return all_results, total_ms


# Smoke test
print('[Test] Smoke test...')
tr, tms = istari_fetch_1000(TEST_QUERIES[0], mode='bm25')
print(f'[Test] Got {len(tr)} results in {tms:.0f}ms')
print(f'[Test] Sample: {tr[0].get("name","")} ({tr[0].get("domain","")})  ✅')


# ## 4 · Run All API Searches — 101 queries × 2 modes × 1000 results
# 
# Results cached to disk after every query.
# If the job times out, re-run and it resumes from where it left off.

# In[4]:


api_cache_path = RESULT_DIR / 'api_results_cache.pkl'

if api_cache_path.exists():
    print('[API] Loading cached results...')
    with open(api_cache_path, 'rb') as f:
        api_results, api_latencies = pickle.load(f)
    print(f'[API] Cached queries : {len(api_results["bm25"])}/{len(TEST_QUERIES)}')
else:
    api_results   = {'bm25': {}, 'bm25_filter': {}}
    api_latencies = {'bm25': {}, 'bm25_filter': {}}
    print('[API] No cache found — starting fresh')

total_to_fetch = sum(1 for q in TEST_QUERIES if q not in api_results['bm25'])
print(f'[API] Queries to fetch : {total_to_fetch}')
print(f'[API] Est. time        : ~{total_to_fetch * 2 * 2 * 0.7 / 60:.1f} min')
print('=' * 60)

total_start = time.time()
done_count  = len(api_results['bm25'])

for qi, query in enumerate(TEST_QUERIES):

    # Skip already fetched
    if query in api_results['bm25']:
        continue

    for mode in ['bm25', 'bm25_filter']:
        results, ms = istari_fetch_1000(query, mode=mode)
        api_results[mode][query]   = results
        api_latencies[mode][query] = ms
        time.sleep(0.3)

    done_count += 1

    # Save after every query
    with open(api_cache_path, 'wb') as f:
        pickle.dump((api_results, api_latencies), f)

    if done_count % 10 == 0 or done_count == len(TEST_QUERIES):
        elapsed   = time.time() - total_start
        remaining = (len(TEST_QUERIES) - done_count) * elapsed / done_count if done_count else 0
        print(f'[API] {done_count:3d}/{len(TEST_QUERIES)}  |  '
              f'avg {elapsed/done_count*1000:.0f}ms/query  |  '
              f'~{remaining:.0f}s remaining')

# Save final JSON
for mode in ['bm25', 'bm25_filter']:
    path = RESULT_DIR / f'api_{mode}_results.json'
    with open(path, 'w') as f:
        json.dump(api_results[mode], f, indent=2, default=str)
    avg = sum(api_latencies[mode].values()) / len(api_latencies[mode]) if api_latencies[mode] else 0
    print(f'[API] {mode}: avg {avg:.0f}ms/query — saved to {path}')


# ## 5 · Our MiniLM — 101 queries × 1000 results

# In[5]:


minilm_cache_path = RESULT_DIR / 'minilm_results_cache.pkl'

if minilm_cache_path.exists():
    print('[MiniLM] Loading cached results...')
    with open(minilm_cache_path, 'rb') as f:
        our_results, our_latencies = pickle.load(f)
    print(f'[MiniLM] Cached: {len(our_results)}/{len(TEST_QUERIES)} queries')
else:
    our_results   = {}
    our_latencies = {}

if len(our_results) < len(TEST_QUERIES):
    print('[MiniLM] Loading model and index...')
    all_companies = pd.read_excel('dataset/production_results.xlsx')
    all_companies = all_companies.drop_duplicates(subset='domain').reset_index(drop=True)
    embeddings    = np.load('result/03_baseline_minilm/company_embeddings.npy').astype('float32')
    index         = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    model         = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    print(f'[MiniLM] Corpus: {len(all_companies):,}  Embeddings: {embeddings.shape}')

    # GPU warmup
    for _ in range(5):
        dummy = model.encode(['warmup'], convert_to_numpy=True).astype('float32')
        index.search(dummy, 10)
    print('[MiniLM] GPU warm ✅')

    total_start = time.time()
    done_count  = len(our_results)

    for qi, query in enumerate(TEST_QUERIES):
        if query in our_results:
            continue

        t0    = time.perf_counter()
        q_emb = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=False
        ).astype('float32')
        dists, idxs = index.search(q_emb, 1000)
        ms = (time.perf_counter() - t0) * 1000

        results = []
        for rank, (idx, dist) in enumerate(zip(idxs[0], dists[0]), 1):
            company = all_companies.iloc[idx]
            results.append({
                'rank':    rank,
                'domain':  company['domain'],
                'name':    str(company.get('name', '')),
                'country': str(company.get('country', '')),
                'summary': str(company.get('summary', '')),
                'score':   float(dist),
            })

        our_results[query]   = results
        our_latencies[query] = ms
        done_count += 1

        # Save after every query
        with open(minilm_cache_path, 'wb') as f:
            pickle.dump((our_results, our_latencies), f)

        if done_count % 20 == 0 or done_count == len(TEST_QUERIES):
            elapsed   = time.time() - total_start
            remaining = (len(TEST_QUERIES) - done_count) * elapsed / done_count if done_count else 0
            print(f'[MiniLM] {done_count:3d}/{len(TEST_QUERIES)}  |  '
                  f'avg {elapsed/done_count*1000:.1f}ms/query  |  '
                  f'~{remaining:.0f}s remaining')

# Save JSON
with open(RESULT_DIR / 'our_minilm_results.json', 'w') as f:
    json.dump(our_results, f, indent=2, default=str)
avg_ms = sum(our_latencies.values()) / len(our_latencies) if our_latencies else 0
print(f'[MiniLM] Avg latency: {avg_ms:.1f}ms — saved to result/09_api_evaluation/our_minilm_results.json')


# ## 6 · Latency Summary

# In[6]:


print('[Latency] Building latency summary...')

latency_rows = []
for query in TEST_QUERIES:
    latency_rows.append({
        'query':          query,
        'api_bm25_ms':    api_latencies['bm25'].get(query, 0),
        'api_filter_ms':  api_latencies['bm25_filter'].get(query, 0),
        'minilm_ms':      our_latencies.get(query, 0),
    })

latency_df = pd.DataFrame(latency_rows)
latency_df.to_csv(RESULT_DIR / 'latency_per_query.csv', index=False)

print(f'[Latency] === AVERAGE QUERY LATENCY ===')
print(f'  Istari BM25        : {latency_df["api_bm25_ms"].mean():.0f}ms  (2 API calls × 500)')
print(f'  Istari BM25+Filter : {latency_df["api_filter_ms"].mean():.0f}ms  (2 API calls × 500)')
print(f'  Our MiniLM         : {latency_df["minilm_ms"].mean():.1f}ms   (local GPU)')
print(f'[Latency] Saved to result/09_api_evaluation/latency_per_query.csv')


# In[ ]:


# ── Fair latency comparison ───────────────────────────────────────────────────
print('=== LATENCY ANALYSIS ===')
print()
print('NOTE: API latency includes network round-trip — not directly comparable')
print('      to local MiniLM latency which is pure computation.')
print()

# API latency breakdown
api_avg    = latency_df['api_bm25_ms'].mean()
filter_avg = latency_df['api_filter_ms'].mean()
minilm_avg = latency_df['minilm_ms'].mean()

# Steady state MiniLM (exclude first 5 warmup queries)
steady_queries = list(our_latencies.keys())[5:]
steady_ms      = np.mean([our_latencies[q] for q in steady_queries if q in our_latencies])

print(f'  Istari API BM25        : {api_avg:.0f}ms total')
print(f'    → includes 2 network calls × ~{api_avg/2:.0f}ms each')
print(f'  Istari API BM25+Filter : {filter_avg:.0f}ms total')
print()
print(f'  Our MiniLM (all)       : {minilm_avg:.1f}ms avg (includes GPU warmup)')
print(f'  Our MiniLM (steady)    : {steady_ms:.1f}ms avg (last N-5 queries — use this)')
print()
print('  For thesis — report MiniLM steady-state latency:')
print(f'    Query encoding + FAISS search = {steady_ms:.1f}ms on A100 GPU')
print(f'    This is the fair production latency for our method')
print()

# Score distribution sanity check alongside latency
print('  Latency distribution for MiniLM:')
all_minilm_times = [our_latencies[q] for q in our_latencies]
print(f'    Min   : {min(all_minilm_times):.1f}ms')
print(f'    Max   : {max(all_minilm_times):.1f}ms')
print(f'    Mean  : {np.mean(all_minilm_times):.1f}ms')
print(f'    Std   : {np.std(all_minilm_times):.1f}ms')
print(f'    P50   : {np.percentile(all_minilm_times, 50):.1f}ms')
print(f'    P95   : {np.percentile(all_minilm_times, 95):.1f}ms')


# ## 7 · Load Ground Truth & Build Score Dicts
# 
# Ground truth = `goi_search_results.json` (Istari semantic search results).
# Relevance = similarity_score (graded 0.0–1.0). Threshold = 0.5 for Precision/Recall.

# In[7]:


print('[GT] Loading ground truth...')
with open('dataset/goi_search_results.json', 'r') as f:
    goi_data = json.load(f)
print(f'[GT] Queries in GT : {len(goi_data)}')

# Check score distribution
print('[GT] Score distribution (first 3 queries):')
for item in goi_data[:3]:
    scores = [r['similarity_score'] for r in item['results']]
    print(f'  "{item["query"]}": '
          f'min={min(scores):.3f}  max={max(scores):.3f}  '
          f'mean={sum(scores)/len(scores):.3f}')

# Build lookup dicts
gt_scores_by_query = {}   # query → {domain: similarity_score}
gt_ranked_by_query = {}   # query → [domain, ...] in rank order

for item in goi_data:
    q = item['query']
    sorted_results = sorted(item['results'], key=lambda x: x['rank'])
    gt_scores_by_query[q] = {r['domain']: float(r['similarity_score']) for r in sorted_results}
    gt_ranked_by_query[q] = [r['domain'] for r in sorted_results]

# Match TEST_QUERIES to GT
matched_queries = {}
for q in TEST_QUERIES:
    if q in gt_scores_by_query:
        matched_queries[q] = q
    else:
        for gt_q in gt_scores_by_query:
            if gt_q.lower().strip() == q.lower().strip():
                matched_queries[q] = gt_q
                break

print(f'[GT] Matched {len(matched_queries)}/{len(TEST_QUERIES)} queries to ground truth')


# # similarity score of goi 

# In[ ]:


# import json
# import numpy as np

# # Load ground truth — standalone version
# with open('dataset/goi_search_results.json', 'r') as f:
#     goi_data = json.load(f)

# print(f'Loaded {len(goi_data)} queries from goi_search_results.json')

# # ── Confirm similarity score distribution ─────────────────────────────────────
# all_scores = []
# for item in goi_data:
#     for r in item['results']:
#         all_scores.append(r['similarity_score'])

# all_scores = np.array(all_scores)

# print(f'\n=== SIMILARITY SCORE DISTRIBUTION ({len(goi_data)} queries × 1000 results = {len(all_scores):,} scores) ===')
# print(f'  Min score    : {all_scores.min():.4f}')
# print(f'  Max score    : {all_scores.max():.4f}')
# print(f'  Mean score   : {all_scores.mean():.4f}')
# print(f'  Median score : {np.median(all_scores):.4f}')
# print(f'  Std dev      : {all_scores.std():.4f}')
# print()
# print('  Percentiles:')
# for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
#     print(f'    {p:>3}th percentile : {np.percentile(all_scores, p):.4f}')
# print()

# print('  Threshold analysis:')
# for thresh in [0.50, 0.60, 0.65, 0.70, 0.73, 0.75, 0.78, 0.80]:
#     n   = (all_scores >= thresh).sum()
#     pct = 100 * n / len(all_scores)
#     print(f'    threshold={thresh:.2f} → {n:>7,} / {len(all_scores):,} pass ({pct:.1f}%)')
# print()

# print('  Per-query min/max (first 10 queries):')
# for item in goi_data[:10]:
#     scores = [r['similarity_score'] for r in item['results']]
#     print(f'    [{item["query_id"]:>3}] "{item["query"][:35]:<35}" '
#           f'min={min(scores):.3f}  max={max(scores):.3f}  mean={sum(scores)/len(scores):.3f}')


# The scores are extremely tightly clustered:
# 
# - Min: 0.688, Max: 0.862, Mean: 0.765, Std dev: only 0.028
# - This is a very narrow range, all 1000 results per query are considered semantically relevant by Istari's system
# 
# Threshold analysis is revealing:
# 
# - threshold=0.50 → 100% pass — completely useless as a cutoff
# - threshold=0.70 → 98.4% pass — still almost everything
# - threshold=0.73 → 87.6% pass — better but still very liberal
# - threshold=0.75 → 72.8% pass — reasonable split
# - threshold=0.78 → 32.0% pass — only the high quality matches
# - threshold=0.80 → 11.4% pass — only the very best

# ## 8 · Evaluation — NDCG (graded), Precision, Recall, Overlap
# 
# k ∈ {10, 50, 100, 500, 1000}

# In[ ]:


RELEVANCE_THRESHOLD = 0.75  # based on score distribution above
K_VALUES = [10, 50, 100, 500, 1000]

def ndcg_graded_at_k(retrieved, gt_scores, k):
    dcg = sum(
        gt_scores.get(d, 0.0) / np.log2(i + 2)
        for i, d in enumerate(retrieved[:k])
    )
    ideal = sorted(gt_scores.values(), reverse=True)[:k]
    idcg  = sum(s / np.log2(i + 2) for i, s in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0

def precision_at_k(retrieved, gt_scores, k, thresh=RELEVANCE_THRESHOLD):
    return sum(1 for d in retrieved[:k] if gt_scores.get(d,0) >= thresh) / k if k else 0

def recall_at_k(retrieved, gt_scores, k, thresh=RELEVANCE_THRESHOLD):
    total = sum(1 for s in gt_scores.values() if s >= thresh)
    if total == 0: return 0
    return sum(1 for d in retrieved[:k] if gt_scores.get(d,0) >= thresh) / total

def overlap_at_k(retrieved, gt_ranked, k):
    return len(set(retrieved[:k]) & set(gt_ranked[:k]))

METHODS = [
    ('Istari BM25',          api_results['bm25']),
    ('Istari BM25 + Filter', api_results['bm25_filter']),
    ('Our MiniLM',           our_results),
]

print('[Eval] Computing metrics for all queries...')
eval_rows = []

for method_name, results_dict in METHODS:
    for query in TEST_QUERIES:
        gt_q = matched_queries.get(query)
        if not gt_q:
            continue
        gt_scores  = gt_scores_by_query[gt_q]
        gt_ranked  = gt_ranked_by_query[gt_q]
        retrieved  = [r['domain'] for r in
                      sorted(results_dict.get(query, []), key=lambda x: x.get('rank',9999))]

        for k in K_VALUES:
            eval_rows.append({
                'method':    method_name,
                'query':     query,
                'k':         k,
                'ndcg':      round(ndcg_graded_at_k(retrieved, gt_scores, k), 4),
                'precision': round(precision_at_k(retrieved, gt_scores, k), 4),
                'recall':    round(recall_at_k(retrieved, gt_scores, k), 4),
                'overlap':   overlap_at_k(retrieved, gt_ranked, k),
            })

eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(RESULT_DIR / 'evaluation_final.csv', index=False)
print(f'[Eval] Done! {len(eval_df):,} rows saved to result/09_api_evaluation/evaluation_final.csv')


# ## 9 · Final Results Table

# In[9]:


print('\n[Results] ============================================================')
print('[Results] FULL EVALUATION vs GOI GROUND TRUTH (graded relevance)')
print(f'[Results] Relevance threshold: {RELEVANCE_THRESHOLD}')
print('[Results] ============================================================')
print(f'  {"Method":<24} {"k":>6} | {"NDCG":>7} | {"Prec":>7} | {"Recall":>7} | {"Overlap":>8}')
print('  ' + '=' * 70)

for method_name, _ in METHODS:
    for k in K_VALUES:
        sub  = eval_df[(eval_df['method']==method_name) & (eval_df['k']==k)]
        if len(sub) == 0: continue
        ndcg = sub['ndcg'].mean()
        prec = sub['precision'].mean()
        rec  = sub['recall'].mean()
        ovlp = sub['overlap'].mean()
        print(f'  {method_name:<24} {k:>6} | {ndcg:>7.3f} | {prec:>7.3f} | {rec:>7.3f} | {ovlp:>8.1f}')
    print('  ' + '-' * 70)

# Highlight key comparison
print(f'\n[Results] Key comparison at k=1000:')
for method_name, _ in METHODS:
    sub  = eval_df[(eval_df['method']==method_name) & (eval_df['k']==1000)]
    if len(sub) == 0: continue
    ndcg = sub['ndcg'].mean()
    rec  = sub['recall'].mean()
    ovlp = sub['overlap'].mean()
    print(f'  {method_name:<24}: NDCG={ndcg:.3f}  Recall={rec:.3f}  Overlap={ovlp:.0f}/1000')


# ## 10 · Qualitative Check — First 3 Queries

# In[10]:


print('[Qualitative] TOP-10 vs GROUND TRUTH for first 3 queries')

for query in TEST_QUERIES[:3]:
    gt_q = matched_queries.get(query)
    if not gt_q: continue
    gt_scores = gt_scores_by_query[gt_q]
    gt_ranked = gt_ranked_by_query[gt_q]

    print(f'\n{"="*70}')
    print(f'Query: "{query}"')
    print(f'GT top-3 domains: {gt_ranked[:3]}')
    print(f'{"="*70}')

    for method_name, results_dict in METHODS:
        retrieved = [r['domain'] for r in
                     sorted(results_dict.get(query,[]), key=lambda x: x.get('rank',9999))]
        print(f'\n  {method_name}:')
        for rank, domain in enumerate(retrieved[:10], 1):
            score    = gt_scores.get(domain, 0.0)
            gt_rank  = gt_ranked.index(domain)+1 if domain in gt_ranked else '>1000'
            icon     = '✅' if score >= RELEVANCE_THRESHOLD else '❌'
            print(f'    {icon} Rank {rank:>3} | {domain:<40} | GT score={score:.3f}  GT rank={gt_rank}')

