#!/usr/bin/env python
# coding: utf-8

# # Reranker Ablation: mxbai-rerank-large-v1 vs BAAI/bge-reranker-v2-m3
# ## Same 3-way pool as notebook 25 -- only the reranker model changes
# 
# **Why this experiment:** notebook 25 (3-way hybrid, MiniLM+Linq+GTE-large) and notebook 28 (4-way, +OpenAI-large) scored almost identically (0.804 vs 0.802 Recall@1000) despite notebook 28's much richer candidate pool -- meaning the reranker, not candidate coverage, is now the limiting factor. This notebook tests that hypothesis directly: same exact 3-way union pool as notebook 25, swap only the reranker model, and see whether a different, comparably-sized-but-architecturally-distinct cross-encoder (DeBERTa-v2-based, vs the current XLM-RoBERTa-based BGE reranker) changes the result.
# 
# **Why `mxbai-rerank-large-v1` specifically:** verified via HF config it's a standard `DebertaV2ForSequenceClassification` -- no `trust_remote_code`, no custom architecture class (the exact class of failure that broke `gte-large-v1.5` and `NV-Embed-v2` in this repo). Loads with plain `AutoModelForSequenceClassification`, same pattern as the current reranker.
# 
# **If this changes the result meaningfully:** the bottleneck is reranker *quality/architecture*, and a better reranker is worth pursuing further. **If it doesn't:** the bottleneck is more fundamental (e.g. genuinely hard vocabulary-mismatch cases the whole reranking paradigm can't fix), which is itself a valuable, reportable finding.
# 
# **Checkpoint/resume:** same pattern as notebooks 25/26/27 -- rerank loop checkpointed every 10 queries with atomic-write-with-retry, 26-min internal time budget.

# In[ ]:


import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RESULT_DIR = Path('result/29_hybrid_altreranker')
RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f'[Setup] Result folder : {RESULT_DIR}/ -- ready')

TOP_K_PER_CHANNEL = 1000   # how many each channel contributes to the union pool
FINAL_K           = 1000   # how many survive after reranking
RERANK_BATCH_SIZE = 32
print(f'[Setup] Top-k per channel : {TOP_K_PER_CHANNEL}')
print(f'[Setup] Final k           : {FINAL_K}')


# In[ ]:


import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SCRIPT_START = time.time()  # marks total job elapsed time, used to stop safely before the 30-min SLURM wall time
TIME_BUDGET_MINUTES = 26    # job wall-time is 30 min -- leaves a buffer before the hard kill so a checkpoint always gets saved

def robust_save(save_fn, path, retries=3, delay_seconds=5):
    """Writes to a temp file then atomically renames -- avoids leaving a corrupted checkpoint if the write is interrupted (seen: Lustre iostream errors truncating writes mid-stream). Retries transient I/O failures."""
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[Imports] Device      : {DEVICE}')
if torch.cuda.is_available():
    print(f'[Imports] GPU         : {torch.cuda.get_device_name(0)}')
print('[Imports] All packages loaded successfully')


# ## 1. Load saved retrieval results -- identical 3-way channel set to notebook 25, all cached, no re-encoding needed

# In[ ]:


print('[Load] Loading MiniLM results...')
minilm_df = pd.read_csv('result/03_baseline_minilm/minilm_results.csv')
print(f'[Load] MiniLM rows    : {len(minilm_df):,}')

print('[Load] Loading Linq-Embed-Mistral results...')
linq_df = pd.read_csv('result/20_baseline_linq_mistral/20_baseline_linq_mistral_results.csv')
print(f'[Load] Linq rows      : {len(linq_df):,}')

print('[Load] Loading GTE-large results...')
gte_df = pd.read_csv('result/17_baseline_gte_large/gte_large_results.csv')
print(f'[Load] GTE-large rows : {len(gte_df):,}')

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
gte_queries    = gte_df['query_id'].nunique()
assert minilm_queries == linq_queries == gte_queries == 101, 'Query count mismatch!'
print('[Load] Sanity check passed')

domain_info = all_companies.set_index('domain')[['name','country','summary']].to_dict('index')


# ## 2. Build raw 3-way union pool per query -- identical to notebook 25

# In[ ]:


CHANNEL_DFS = {'MiniLM': minilm_df, 'Linq_Mistral': linq_df, 'GTE_large': gte_df}

pool_sizes = {}
query_pools = {}

for item in data:
    qid = item['query_id']
    ranked_lists = []
    for name, df in CHANNEL_DFS.items():
        ranked = df[df['query_id'] == qid].sort_values('rank')['domain'].tolist()[:TOP_K_PER_CHANNEL]
        ranked_lists.append(ranked)
    pool = list(dict.fromkeys([d for lst in ranked_lists for d in lst]))  # union, dedup, order not important
    query_pools[qid] = pool
    pool_sizes[qid] = len(pool)

avg_pool = sum(pool_sizes.values()) / len(pool_sizes)
print(f'[Pool] Avg pool size : {avg_pool:.0f} (should match notebook 25 -- ~1,424)')


# ## 3. Load the alternative reranker -- `mixedbread-ai/mxbai-rerank-large-v1`
# 
# Standard `DebertaV2ForSequenceClassification` -- no custom code, no `trust_remote_code`. Same query+document pair scoring convention as the BGE reranker, but the docs specify a sigmoid over the raw logit for a calibrated score (doesn't change the ranking, since sigmoid is monotonic, but included for correctness).

# In[ ]:


print('[Reranker] Loading mxbai-rerank-large-v1...')
t0                 = time.time()
RERANKER_MODEL     = 'mixedbread-ai/mxbai-rerank-large-v1'
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
        logits = model_reranker(**encoded).logits.squeeze(-1)
        scores = torch.sigmoid(logits)
        all_scores.extend(scores.cpu().float().tolist())
    return all_scores

print('[Warmup] Warming up reranker with 3 dummy calls...')
sample_qid   = data[0]['query_id']
sample_query = data[0]['query']
sample_pool  = query_pools[sample_qid][:50]
for _ in range(3):
    score_pool(sample_query, sample_pool)
print('[Warmup] GPU warm')


# ## 4. Rerank the full 3-way union pool per query, keep top-1000
# 
# Checkpointed every 10 queries with atomic-write-with-retry.

# In[ ]:


CHECKPOINT_PATH = RESULT_DIR / 'reranked_results_checkpoint.csv'
FINAL_PATH      = RESULT_DIR / 'reranked_results.csv'

print(f'[Rerank] Scoring full union pool for {len(data)} queries...')

if FINAL_PATH.exists():
    print('[Rerank] Final results already on disk -- loading, skipping rerank')
    results_df = pd.read_csv(FINAL_PATH)
    rerank_times = []
else:
    if CHECKPOINT_PATH.exists():
        prior_df  = pd.read_csv(CHECKPOINT_PATH)
        done_qids = set(prior_df['query_id'].unique())
        all_results = prior_df.to_dict('records')
        print(f'[Rerank] Resuming -- {len(done_qids)}/{len(data)} queries already reranked')
    else:
        all_results = []
        done_qids   = set()

    rerank_times = []
    total_start  = time.time()

    for i, item in enumerate(data):
        qid = item['query_id']
        if qid in done_qids:
            continue
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

        if len(rerank_times) % 10 == 0 or (i + 1) == len(data):
            elapsed  = time.time() - total_start
            done_now = len(set(r['query_id'] for r in all_results))
            print(f'[Rerank] {done_now:3d}/{len(data)}  |  avg {sum(rerank_times)/len(rerank_times)/1000:.1f}s/query (this session)')
            robust_save(lambda p: pd.DataFrame(all_results).to_csv(p, index=False), CHECKPOINT_PATH)

        if (time.time() - SCRIPT_START) / 60 > TIME_BUDGET_MINUTES:
            robust_save(lambda p: pd.DataFrame(all_results).to_csv(p, index=False), CHECKPOINT_PATH)
            done_now = len(set(r['query_id'] for r in all_results))
            print(f'[Rerank] Time budget ({TIME_BUDGET_MINUTES} min) reached at {done_now}/{len(data)} queries -- checkpoint saved. Rerun this same job to resume.')
            sys.exit(0)

    results_df = pd.DataFrame(all_results)
    robust_save(lambda p: results_df.to_csv(p, index=False), FINAL_PATH)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

AVG_RERANK_S = (sum(rerank_times) / len(rerank_times) / 1000) if rerank_times else float('nan')
print(f'[Rerank] Done!')
print(f'[Rerank] Avg rerank time (this session) : {AVG_RERANK_S:.1f}s/query')
print(f'[Rerank] Saved to        : {FINAL_PATH}')


# ## 5. Evaluation -- k=10,50,100,300,500,1000

# In[ ]:


def get_relevant(qid, top_k=1000):
    return set(production_df[(production_df['query_id']==qid) & (production_df['rank']<=top_k)]['domain'].tolist())

K_VALUES = [10, 50, 100, 300, 500, 1000]

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

print('[Eval] Evaluating alt-reranker results...')
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

print('[Eval] === MXBAI-RERANK-LARGE-V1 RESULTS (same 3-way pool as notebook 25) ===')
print(f'  {"k":<6} {"NDCG":>8} {"Precision":>10} {"Recall":>8} {"F1":>8}')
print('  ' + '-' * 46)
for k in K_VALUES:
    sub = eval_df[eval_df['k'] == k]
    print(f'  {k:<6} '
          f'{sub["ndcg"].mean():>8.3f} '
          f'{sub["precision"].mean():>10.3f} '
          f'{sub["recall"].mean():>8.3f} '
          f'{sub["f1"].mean():>8.3f}')


# ## 6. Comparison vs notebook 25 (BAAI/bge-reranker-v2-m3, same exact pool)

# In[ ]:


print('[Compare] Loading notebook 25 results (same pool, different reranker)...')
prior_eval_df = pd.read_csv('result/25_hybrid_triple_full_rerank/evaluation.csv')

comp_rows = []
for k in K_VALUES:
    alt_sub   = eval_df[eval_df['k'] == k]
    prior_sub = prior_eval_df[prior_eval_df['k'] == k] if k in prior_eval_df['k'].unique() else None
    row = {
        'k':                    k,
        'mxbai_ndcg':           round(alt_sub['ndcg'].mean(), 3),
        'mxbai_recall':         round(alt_sub['recall'].mean(), 3),
    }
    if prior_sub is not None and len(prior_sub):
        row['bge_v2m3_ndcg']   = round(prior_sub['ndcg'].mean(), 3)
        row['bge_v2m3_recall'] = round(prior_sub['recall'].mean(), 3)
    comp_rows.append(row)

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(RESULT_DIR / 'comparison_vs_notebook25.csv', index=False)
print(comp_df.to_string(index=False))

alt_r1000  = eval_df[eval_df['k']==1000]['recall'].mean()
alt_n10    = eval_df[eval_df['k']==10]['ndcg'].mean()
bge_r1000  = prior_eval_df[prior_eval_df['k']==1000]['recall'].mean()
bge_n10    = prior_eval_df[prior_eval_df['k']==10]['ndcg'].mean()
print(f'\n[Compare] mxbai-rerank-large-v1  : Recall@1000={alt_r1000:.3f}  NDCG@10={alt_n10:.3f}')
print(f'[Compare] bge-reranker-v2-m3     : Recall@1000={bge_r1000:.3f}  NDCG@10={bge_n10:.3f}')
print(f'[Compare] Delta                  : Recall@1000={alt_r1000-bge_r1000:+.3f}  NDCG@10={alt_n10-bge_n10:+.3f}')
print()
if abs(alt_r1000 - bge_r1000) < 0.01 and abs(alt_n10 - bge_n10) < 0.01:
    print('[Compare] Interpretation: near-identical results -- the reranker model choice is NOT the bottleneck.')
    print('[Compare] The residual gap vs the oracle ceiling is likely a harder, more fundamental limitation')
    print('[Compare] (e.g. genuinely ambiguous/vocabulary-mismatch cases no reranker can resolve from summary text alone).')
else:
    print('[Compare] Interpretation: meaningfully different results -- reranker choice DOES matter here.')
    print('[Compare] Worth investigating further (e.g. trying additional rerankers, or the reranker input format).')

