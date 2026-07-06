import os, json, time, pickle, math, requests
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

ISTARI_API_KEY   = os.getenv('API_KEY')
V2_BASE_URL      = 'https://api.istari.ai/v2/search'
RESULT_DIR       = Path('result/03b_minilm_api_rerank')
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RELEVANCE_THRESHOLD = 0.75

print(f'[Setup] V2 URL        : {V2_BASE_URL}')
print(f'[Setup] API key set   : {"✅" if ISTARI_API_KEY else "❌  — check API_KEY in .env"}')
print(f'[Setup] Result dir    : {RESULT_DIR}/')

import torch
from sentence_transformers import SentenceTransformer

print(f'[GPU] CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[GPU] Device          : {torch.cuda.get_device_name(0)}')
    DEVICE = 'cuda'
else:
    print('[GPU] No GPU — running on CPU (MiniLM is fast enough on CPU)')
    DEVICE = 'cpu'

print('[Model] Loading MiniLM (all-MiniLM-L6-v2)...')
t0    = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
print(f'[Model] Loaded in {time.time()-t0:.1f}s on {model.device}')
print('[Model] Embedding dims : 384 (same convention as 03_baseline_minilm.ipynb — '
      'raw embeddings, L2 distance, normalize_embeddings=False)')

print('[Setup] Testing v2 API...')
r = requests.post(
    V2_BASE_URL,
    headers={
        'Accept': 'application/json',
        'x-api-key': ISTARI_API_KEY,
        'Content-Type': 'application/json',
    },
    json={
        'describe': 'software companies',
        'keywords': {'must_all': [], 'must_any': [], 'must_not': []},
        'filters':  {'country': [], 'state': [], 'region': [],
                     'organization_type': [], 'organization_size': [], 'nace_code': []},
        'excludes': [],
        'columns':  ['domain', 'name'],
        'size':     3,
    },
    timeout=15,
)
if r.status_code == 200:
    sample = r.json().get('data', [])
    print(f'[Setup] v2 API OK ✅  — got {len(sample)} results')
    print(f'[Setup] Sample      : {sample[0].get("name","?")} ({sample[0].get("domain","?")})')
else:
    print(f'[Setup] v2 API ERROR: {r.status_code} {r.text[:200]}')
    raise SystemExit('Fix API key or endpoint before continuing.')

# A MISSING rate-limit header is treated as "unlimited", not zero. An unlimited-tier
# API key may simply not send these headers at all (nothing to rate-limit), and the
# previous version of this cell defaulted a missing header to 0 -- which silently
# self-limited an unlimited key down to a tiny SIZE_PER_QUERY instead of the full
# TARGET_SIZE_PER_QUERY. Raw header values are printed below so this is verifiable
# from the log rather than assumed.
REMAINING_RESULTS_HDR  = r.headers.get('X-RateLimit-Results-Remaining')
REMAINING_REQUESTS_HDR = r.headers.get('X-RateLimit-Requests-Remaining')
UNLIMITED_FALLBACK     = 10_000_000

REMAINING_RESULTS  = int(REMAINING_RESULTS_HDR)  if REMAINING_RESULTS_HDR  is not None else UNLIMITED_FALLBACK
REMAINING_REQUESTS = int(REMAINING_REQUESTS_HDR) if REMAINING_REQUESTS_HDR is not None else UNLIMITED_FALLBACK

print(f'[Setup] Raw rate-limit headers  : results={REMAINING_RESULTS_HDR!r}, requests={REMAINING_REQUESTS_HDR!r}')
if REMAINING_RESULTS_HDR is None or REMAINING_REQUESTS_HDR is None:
    print('[Setup] NOTE: header(s) missing from response — treating as unlimited for this run '
          '(expected for an unlimited-quota API key). If this is wrong, the raw values above '
          'will show it.')
print(f'[Setup] Live quota remaining this month → requests: {REMAINING_REQUESTS}, results: {REMAINING_RESULTS}')

print('[Load] Loading queries...')
queries_df   = pd.read_excel('dataset/queries.xlsx')
query_col    = next(c for c in queries_df.columns if queries_df[c].dtype == object)
TEST_QUERIES = queries_df[query_col].astype(str).tolist()
print(f'[Load] {len(TEST_QUERIES)} queries from column "{query_col}"')
print(f'[Load] First 5: {TEST_QUERIES[:5]}')

QUERIES_TO_FETCH = len(TEST_QUERIES)  # 101 total queries in the evaluation set

SAFETY_MARGIN         = 0.9    # leave 10% headroom on top of what the server currently reports
V2_MAX_PAGE_SIZE       = 500   # documented API max per single request
TARGET_SIZE_PER_QUERY  = 1000  # desired depth -- matches k=1000 used across the other
                                # baselines; well under the documented 10,000-row depth
                                # cap for scored/semantic search modes

SIZE_PER_QUERY = min(
    TARGET_SIZE_PER_QUERY,
    int((REMAINING_RESULTS * SAFETY_MARGIN) // QUERIES_TO_FETCH),
)

if SIZE_PER_QUERY < 5:
    raise SystemExit(
        f'[v2] Only {REMAINING_RESULTS} results left this month — '
        f'not enough for {QUERIES_TO_FETCH} queries even at size=5. Stop and wait for quota reset.'
    )

PAGES_PER_QUERY = math.ceil(SIZE_PER_QUERY / V2_MAX_PAGE_SIZE)  # 1 or 2 pages to reach up to 1000

SHALLOW_TOLERANCE = 0.9  # a cached entry within 90% of SIZE_PER_QUERY is NOT considered
                          # stale -- dedup against page-boundary overlaps can legitimately
                          # leave a query a few results short of the exact target

v2_cache_path = RESULT_DIR / 'v2_results_cache.pkl'
if v2_cache_path.exists():
    with open(v2_cache_path, 'rb') as f:
        v2_results, v2_latencies = pickle.load(f)
    print(f'[v2] Loading cache from a previous run: {len(v2_results)}/{len(TEST_QUERIES)} queries')

    stale = [
        q for q, data in v2_results.items()
        if not data
        or len({r['domain'] for r in data}) < len(data)
        or len(data) < SIZE_PER_QUERY * SHALLOW_TOLERANCE
    ]
    for q in stale:
        del v2_results[q]
        v2_latencies.pop(q, None)
    if stale:
        print(f'[v2] Dropped {len(stale)} poisoned/shallow cached entries (refetching at size={SIZE_PER_QUERY}): {stale}')
    print(f'[v2] Valid cached queries at current depth: {len(v2_results)}/{len(TEST_QUERIES)}')
else:
    v2_results   = {}
    v2_latencies = {}
    print('[v2] No cache — starting fresh')

to_fetch = [q for q in TEST_QUERIES if q not in v2_results]

est_results  = SIZE_PER_QUERY * len(to_fetch)
est_requests = PAGES_PER_QUERY * len(to_fetch)

if est_requests > REMAINING_REQUESTS:
    raise SystemExit(
        f'[v2] Fetching the {len(to_fetch)} still-needed queries needs up to ~{est_requests} '
        f'requests ({PAGES_PER_QUERY}/query) but only {REMAINING_REQUESTS} requests remain '
        f'this month. Lower TARGET_SIZE_PER_QUERY or wait for quota reset.'
    )

K_VALUES = sorted({k for k in [10, 30, 50, 100, 200, 500] if k <= SIZE_PER_QUERY} | {SIZE_PER_QUERY})

print(f'[v2] Live remaining budget    : {REMAINING_REQUESTS} requests, {REMAINING_RESULTS} results')
print(f'[v2] Queries to fetch         : {len(to_fetch)}/{QUERIES_TO_FETCH}')
print(f'[v2] Size per query           : {SIZE_PER_QUERY}  ({PAGES_PER_QUERY} page(s)/query, {V2_MAX_PAGE_SIZE} max/page)')
print(f'[v2] Estimated cost this run  : {est_requests} requests, {est_results} results')
print(f'[v2] Evaluation k values      : {K_VALUES}')

V2_COLUMNS = [
    'domain', 'name', 'country', 'state', 'municipality', 'district',
    'organization_type', 'organization_size', 'nace_code', 'summary', 'summary_keywords',
]
V2_RATE_LIMIT_RETRIES = 5
V2_RATE_LIMIT_WAIT    = 10   # seconds, multiplied by attempt number
V2_PAGE_SLEEP         = 1.5  # seconds between pages of the SAME query


def _v2_call_page(query, size, search_after=None):
    """Single page of the v2 API. Pass search_after (the previous response's
    metadata.search_after cursor) to fetch the next page; omit/None for the
    first page. Request field: top-level search_after. Response field:
    metadata.search_after (null when there are no further pages)."""
    payload = {
        'describe': query,
        'keywords': {'must_all': [], 'must_any': [], 'must_not': []},
        'filters':  {'country': [], 'state': [], 'region': [],
                     'organization_type': [], 'organization_size': [], 'nace_code': []},
        'excludes': [],
        'columns':  V2_COLUMNS,
        'size':     size,
    }
    if search_after is not None:
        payload['search_after'] = search_after

    for attempt in range(V2_RATE_LIMIT_RETRIES):
        t0   = time.perf_counter()
        resp = requests.post(
            V2_BASE_URL,
            headers={
                'Accept': 'application/json',
                'x-api-key': ISTARI_API_KEY,
                'Content-Type': 'application/json',
            },
            json=payload,
            timeout=30,
        )
        ms = (time.perf_counter() - t0) * 1000
        remaining_hdr = resp.headers.get('X-RateLimit-Results-Remaining')
        remaining     = int(remaining_hdr) if remaining_hdr is not None else None

        if resp.status_code == 200:
            body        = resp.json()
            data        = body.get('data', [])
            next_cursor = body.get('metadata', {}).get('search_after')
            return data, ms, remaining, next_cursor

        if resp.status_code == 429:
            body_text = resp.text[:200]
            quota_exhausted = (remaining is not None and remaining <= 0) or 'quota' in body_text.lower()
            if quota_exhausted:
                raise SystemExit(
                    f'[v2] 429 — monthly quota genuinely exhausted on "{query}" '
                    f'(results remaining: {remaining}). Stopping — retrying won\'t help. '
                    f'Response: {body_text}'
                )
            wait = V2_RATE_LIMIT_WAIT * (attempt + 1)
            print(f'    [v2] 429 transient rate limit on "{query}" ({remaining} results still '
                  f'remaining — not a quota issue) — waiting {wait}s (attempt {attempt+1}/{V2_RATE_LIMIT_RETRIES})')
            time.sleep(wait)
            continue

        print(f'    [v2] ERROR {resp.status_code} on "{query}": {resp.text[:150]}')
        return None, ms, remaining, None

    print(f'    [v2] Gave up on "{query}" after {V2_RATE_LIMIT_RETRIES} rate-limit retries — skipping (not cached).')
    return None, 0, None, None


def v2_call(query, target_size, max_pages):
    """Fetches up to target_size UNIQUE-domain candidates for one query,
    paginating via search_after, capped at max_pages requests. Dedupes by
    domain across pages (near-tied ranking scores can legitimately cause the
    same domain to reappear at a page boundary). Assigns v2's own native rank
    (1..N) to each candidate -- this is v2's OWN ranking, kept as the baseline
    to compare MiniLM's re-ranking against in Section 7."""
    all_data       = []
    seen_domains   = set()
    total_ms       = 0.0
    search_after   = None
    last_remaining = None
    pages_fetched  = 0

    while len(all_data) < target_size and pages_fetched < max_pages:
        page_size = min(V2_MAX_PAGE_SIZE, target_size - len(all_data))
        data, ms, remaining, next_cursor = _v2_call_page(query, page_size, search_after)
        pages_fetched += 1
        total_ms += ms
        if data is None:
            return None, total_ms, last_remaining

        new_rows = [r for r in data if r['domain'] not in seen_domains]
        seen_domains.update(r['domain'] for r in new_rows)
        for rank, res in enumerate(new_rows, start=len(all_data) + 1):
            res['v2_native_rank'] = rank
        all_data.extend(new_rows)
        last_remaining = remaining if remaining is not None else last_remaining

        if not data or next_cursor is None:
            break
        search_after = next_cursor
        if pages_fetched < max_pages:
            time.sleep(V2_PAGE_SLEEP)

    return all_data, total_ms, last_remaining


print('[v2] Functions defined ✅ (paginates via search_after, dedupes by domain, '
      'capped at max_pages/query, fatal only on real quota exhaustion, '
      'short retry on transient rate limits, no cache poisoning)')

if to_fetch:
    print(f'[v2] Fetching {len(to_fetch)} queries at target size={SIZE_PER_QUERY} each '
          f'({PAGES_PER_QUERY} page(s)/query)...')
    print('-' * 60)
    total_start = time.time()

    for i, query in enumerate(to_fetch):
        data, ms, remaining = v2_call(query, target_size=SIZE_PER_QUERY, max_pages=PAGES_PER_QUERY)

        if data is None:
            print(f'[v2] Skipping "{query}" due to error above — NOT cached, will retry next run.')
            continue

        v2_results[query]   = data
        v2_latencies[query] = ms

        with open(v2_cache_path, 'wb') as f:
            pickle.dump((v2_results, v2_latencies), f)

        if (i + 1) % 10 == 0 or (i + 1) == len(to_fetch):
            elapsed     = time.time() - total_start
            remaining_t = (len(to_fetch) - i - 1) * elapsed / (i + 1) if (i + 1) > 0 else 0
            avg_ms      = sum(v2_latencies.values()) / len(v2_latencies)
            print(f'[v2] {i+1:3d}/{len(to_fetch)}  |  '
                  f'{len(data)} results  |  '
                  f'avg {avg_ms:.0f}ms/query  |  '
                  f'~{remaining_t:.0f}s remaining  |  '
                  f'quota results left: {remaining}')

        time.sleep(3.0)  # paced to avoid tripping the transient per-second rate limit

    print('-' * 60)
    print(f'[v2] Done! {len(v2_results)} queries fetched.')
else:
    print('[v2] All queries already cached ✅')

with open(RESULT_DIR / 'v2_results.json', 'w') as f:
    json.dump(v2_results, f, indent=2, default=str)

sizes = [len(v2_results[q]) for q in v2_results]
print(f'[v2] Candidates per query: min={min(sizes)}  max={max(sizes)}  avg={sum(sizes)/len(sizes):.0f}')
print(f'[v2] Saved to {RESULT_DIR}/v2_results.json')

def build_rich_text(company):
    """Combine all available fields into one string for MiniLM encoding --
    same field set and prefixes as 03_baseline_minilm.ipynb's build_rich_text,
    adapted to the v2 API's response field names."""
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
        val = company.get(field, '')
        if isinstance(val, str) and val.strip():
            parts.append(f'{prefix} {val}'.strip() if prefix else val)

    kw = company.get('summary_keywords', '')
    if isinstance(kw, list) and kw:
        parts.append(f'Keywords: {", ".join(str(k) for k in kw)}')
    elif isinstance(kw, str) and kw.strip():
        kw_clean = kw.replace("'", '').replace('[', '').replace(']', '')
        parts.append(f'Keywords: {kw_clean}')

    return ' | '.join(parts)


print(f'[Rerank] Re-ranking {len(v2_results)} queries with MiniLM...')
print('-' * 55)

all_reranked_results = []
rerank_times         = []
total_start           = time.time()

for i, query in enumerate(v2_results):
    candidates = v2_results[query]
    texts      = [build_rich_text(c) for c in candidates]

    t0 = time.perf_counter()
    query_emb = model.encode([query], normalize_embeddings=False, convert_to_numpy=True).astype('float32')
    cand_embs = model.encode(texts, normalize_embeddings=False, convert_to_numpy=True,
                              batch_size=256 if DEVICE == 'cuda' else 64).astype('float32')
    distances = np.linalg.norm(cand_embs - query_emb, axis=1)  # L2 distance, lower = more similar
    rerank_ms = (time.perf_counter() - t0) * 1000
    rerank_times.append(rerank_ms)

    order = np.argsort(distances)
    for new_rank, idx in enumerate(order, start=1):
        c = candidates[idx]
        all_reranked_results.append({
            'query':          query,
            'rank':           new_rank,
            'v2_native_rank': c.get('v2_native_rank'),
            'minilm_distance': float(distances[idx]),
            'domain':         c.get('domain', ''),
            'name':           c.get('name', ''),
            'country':        c.get('country', ''),
            'summary':        c.get('summary', ''),
        })

    if (i + 1) % 20 == 0 or (i + 1) == len(v2_results):
        elapsed   = time.time() - total_start
        remaining = (len(v2_results) - i - 1) * elapsed / (i + 1)
        print(f'[Rerank] {i+1:3d}/{len(v2_results)}  |  '
              f'avg {sum(rerank_times)/len(rerank_times):.1f}ms/query  |  '
              f'~{remaining:.0f}s remaining')

reranked_df = pd.DataFrame(all_reranked_results)
reranked_df.to_csv(RESULT_DIR / 'minilm_reranked_results.csv', index=False)

AVG_RERANK_MS = sum(rerank_times) / len(rerank_times)
print('-' * 55)
print(f'[Rerank] Done!')
print(f'[Rerank] Avg rerank time (encode + sort) : {AVG_RERANK_MS:.1f}ms/query')
print(f'[Rerank] Saved to : {RESULT_DIR}/minilm_reranked_results.csv')

print('[GT] Loading ground truth...')
with open('dataset/goi_search_results.json', 'r') as f:
    goi_data = json.load(f)

gt_scores_by_query = {}
gt_ranked_by_query = {}
for item in goi_data:
    q = item['query']
    sorted_results = sorted(item['results'], key=lambda x: x['rank'])
    gt_scores_by_query[q] = {r['domain']: float(r['similarity_score']) for r in sorted_results}
    gt_ranked_by_query[q] = [r['domain'] for r in sorted_results]

matched_queries = {}
for q in TEST_QUERIES:
    if q in gt_scores_by_query:
        matched_queries[q] = q
    else:
        for gt_q in gt_scores_by_query:
            if gt_q.lower().strip() == q.lower().strip():
                matched_queries[q] = gt_q
                break

print(f'[GT] Loaded {len(goi_data)} queries')
print(f'[GT] Matched {len(matched_queries)}/{len(TEST_QUERIES)} queries to ground truth')

def ndcg_graded_at_k(retrieved, gt_scores, k):
    dcg  = sum(gt_scores.get(d, 0.0) / np.log2(i + 2)
               for i, d in enumerate(retrieved[:k]))
    idcg = sum(s / np.log2(i + 2)
               for i, s in enumerate(sorted(gt_scores.values(), reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0

def precision_at_k(retrieved, gt_scores, k, thresh=RELEVANCE_THRESHOLD):
    return sum(1 for d in retrieved[:k] if gt_scores.get(d, 0) >= thresh) / k if k else 0

def recall_at_k(retrieved, gt_scores, k, thresh=RELEVANCE_THRESHOLD):
    total = sum(1 for s in gt_scores.values() if s >= thresh)
    return sum(1 for d in retrieved[:k] if gt_scores.get(d, 0) >= thresh) / total if total else 0

def f1_at_k(retrieved, gt_scores, k, thresh=RELEVANCE_THRESHOLD):
    p = precision_at_k(retrieved, gt_scores, k, thresh)
    r = recall_at_k(retrieved, gt_scores, k, thresh)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

def overlap_at_k(retrieved, gt_ranked, k):
    return len(set(retrieved[:k]) & set(gt_ranked[:k]))

print('[Eval] Metric functions defined ✅')

METHODS = {
    'API Native':       {q: sorted(v2_results[q], key=lambda c: c['v2_native_rank']) for q in v2_results},
    'MiniLM Reranked':  {q: reranked_df[reranked_df['query'] == q].sort_values('rank').to_dict('records') for q in v2_results},
}

print('[Eval] Computing metrics for both rankings...')
eval_rows = []

for method_name, results_dict in METHODS.items():
    for query in v2_results:
        gt_q = matched_queries.get(query)
        if not gt_q:
            continue
        gt_scores = gt_scores_by_query[gt_q]
        gt_ranked = gt_ranked_by_query[gt_q]
        retrieved = [c['domain'] for c in results_dict[query]]
        for k in K_VALUES:
            eval_rows.append({
                'method':    method_name,
                'query':     query,
                'k':         k,
                'ndcg':      round(ndcg_graded_at_k(retrieved, gt_scores, k), 4),
                'precision': round(precision_at_k(retrieved, gt_scores, k), 4),
                'recall':    round(recall_at_k(retrieved, gt_scores, k), 4),
                'f1':        round(f1_at_k(retrieved, gt_scores, k), 4),
                'overlap':   overlap_at_k(retrieved, gt_ranked, k),
            })

eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(RESULT_DIR / 'evaluation_comparison.csv', index=False)
print(f'[Eval] Done — {len(eval_df):,} rows saved to {RESULT_DIR}/evaluation_comparison.csv')

print('=' * 75)
print(f'EVALUATION vs GOI GROUND TRUTH  (threshold={RELEVANCE_THRESHOLD})')
print('=' * 75)
print(f'  {"Method":<20} {"k":>6} | {"NDCG":>7} | {"Prec":>7} | {"Recall":>7} | {"F1":>7} | {"Overlap":>8}')
print('  ' + '=' * 75)

for method_name in METHODS:
    for k in K_VALUES:
        sub = eval_df[(eval_df['method'] == method_name) & (eval_df['k'] == k)]
        if len(sub) == 0:
            continue
        print(f'  {method_name:<20} {k:>6} | '
              f'{sub["ndcg"].mean():>7.3f} | '
              f'{sub["precision"].mean():>7.3f} | '
              f'{sub["recall"].mean():>7.3f} | '
              f'{sub["f1"].mean():>7.3f} | '
              f'{sub["overlap"].mean():>8.1f}')
    print('  ' + '-' * 75)

avg_fetch_ms  = np.mean(list(v2_latencies.values()))
avg_rerank_ms = np.mean(rerank_times) if rerank_times else 0

print('[Latency] ============================================================')
print(f'  v2 API fetch (network)      : {avg_fetch_ms:.0f}ms avg  '
      f'({PAGES_PER_QUERY} page(s)/query at size={SIZE_PER_QUERY})')
print(f'  MiniLM re-rank (local GPU)  : {avg_rerank_ms:.1f}ms avg  '
      f'(encode query + ~{int(np.mean([len(v) for v in v2_results.values()]))} candidates + sort)')
print(f'  Combined (fetch + rerank)   : {avg_fetch_ms + avg_rerank_ms:.0f}ms avg')
print('[Latency] NOTE: the fetch component is a full deep fetch (up to 1000 results),')
print('[Latency] not directly comparable to production\'s typical per-search latency')
print('[Latency] (see 09b_api_evaluation_v2.ipynb Section 9b for that comparison).')

latency_summary = pd.DataFrame([
    {'component': 'v2 API fetch',  'avg_ms': round(avg_fetch_ms, 1),  'note': f'{PAGES_PER_QUERY} page(s)/query, network'},
    {'component': 'MiniLM rerank', 'avg_ms': round(avg_rerank_ms, 1), 'note': 'local GPU, encode + sort'},
    {'component': 'Combined',      'avg_ms': round(avg_fetch_ms + avg_rerank_ms, 1), 'note': 'end-to-end this pipeline'},
])
latency_summary.to_csv(RESULT_DIR / 'latency_summary.csv', index=False)
print(f'[Latency] Saved to {RESULT_DIR}/latency_summary.csv')

print('[Findings] ============================================================')
print('[Findings] MiniLM RERANKED vs API NATIVE (same candidate pool)')
print('[Findings] ============================================================')

for k in K_VALUES:
    native_sub  = eval_df[(eval_df['method'] == 'API Native')      & (eval_df['k'] == k)]
    rerank_sub  = eval_df[(eval_df['method'] == 'MiniLM Reranked') & (eval_df['k'] == k)]
    if len(native_sub) == 0:
        continue
    native_ndcg, rerank_ndcg = native_sub['ndcg'].mean(), rerank_sub['ndcg'].mean()
    native_rec,  rerank_rec  = native_sub['recall'].mean(), rerank_sub['recall'].mean()
    delta = (rerank_ndcg / native_ndcg - 1) * 100 if native_ndcg > 0 else float('inf')
    print(f'\n  k={k}:')
    print(f'    API Native        NDCG={native_ndcg:.3f}  Recall={native_rec:.3f}'
          f'{"  ← higher NDCG" if native_ndcg >= rerank_ndcg else ""}')
    print(f'    MiniLM Reranked   NDCG={rerank_ndcg:.3f}  Recall={rerank_rec:.3f}  ({delta:+.1f}% vs native)'
          f'{"  ← higher NDCG" if rerank_ndcg > native_ndcg else ""}')

deepest_k = max(K_VALUES)
native_r  = eval_df[(eval_df['method']=='API Native')      & (eval_df['k']==deepest_k)]['recall'].mean()
rerank_r  = eval_df[(eval_df['method']=='MiniLM Reranked') & (eval_df['k']==deepest_k)]['recall'].mean()
print(f'\n[Findings] Recall@{deepest_k} sanity check: API Native={native_r:.4f}  MiniLM Reranked={rerank_r:.4f}')
print('[Findings] These should be IDENTICAL (or nearly so) -- re-ranking only reorders the')
print('[Findings] same fetched candidate pool, it cannot add companies that were not retrieved.')
print('[Findings] Any real difference between the two methods should show up in NDCG/Precision')
print('[Findings] at shallow k, not in Recall at the deepest fetched depth.')
print('\n[Done] result/03b_minilm_api_rerank/ — all files saved.')
