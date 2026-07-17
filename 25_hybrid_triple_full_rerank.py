#!/usr/bin/env python
# coding: utf-8

# # Hybrid Retrieval: MiniLM + Linq-Embed-Mistral + GTE-large -- Full-Pool Rerank
# ## Adding GTE-large (best solo channel, Recall@1000=0.768) to the best pair from notebook 22
# 
# **Why this variant:** notebook 24's 2-channel full-pool rerank (MiniLM + Linq_Mistral) reached Recall@1000=0.802, still 0.067 below the 0.869 oracle ceiling for that pair. GTE-large is actually the single best solo channel (0.768, beating MiniLM's 0.741) and was never combined with the other two. Adding it should both raise the oracle ceiling and give the reranker a richer, more complementary candidate pool.
# 
# ```
# MiniLM top-1000              --+
# Linq-Embed-Mistral top-1000  --+-- raw 3-way union (no fusion/truncation)
# GTE-large top-1000           --+
#                                      |
#                                      v
#                     Cross-encoder reranker scores EVERY candidate
#                                      |
#                                      v
#                           Sort by score, keep top-1000
# ```
# 
# **Sources (all cached, no re-encoding):**
# - `result/03_baseline_minilm/minilm_results.csv`
# - `result/20_baseline_linq_mistral/20_baseline_linq_mistral_results.csv`
# - `result/17_baseline_gte_large/gte_large_results.csv`
# 
# **Checkpoint/resume:** the rerank loop saves progress every 10 queries and self-exits gracefully before the 30-min SLURM wall time, so a timed-out run can be resubmitted (`sbatch run12.sh` again) to continue from the last completed query instead of restarting.
# 
# **Folder structure:**
# ```
# result/
#   25_hybrid_triple_full_rerank/
#     reranked_results.csv     # full-pool reranked top-1000 per query
#     evaluation.csv           # NDCG/Prec/Recall/F1 at k=10,50,100,300,500,1000
#     comparison_vs_prior.csv  # this vs notebook 24 vs notebook 22 oracle ceilings
# ```

# In[ ]:


import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RESULT_DIR = Path('result/25_hybrid_triple_full_rerank')
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[Imports] Device      : {DEVICE}')
if torch.cuda.is_available():
    print(f'[Imports] GPU         : {torch.cuda.get_device_name(0)}')
print('[Imports] All packages loaded successfully')


# ## 1. Load saved retrieval results -- all three already cached, no re-encoding needed

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


# ## 2. Build raw 3-way union pool per query -- no RRF, no pre-truncation

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
print(f'[Pool] Avg pool size : {avg_pool:.0f}')
print(f'[Pool] Min pool size : {min(pool_sizes.values())}')
print(f'[Pool] Max pool size : {max(pool_sizes.values())}')


# ## 3. Oracle ceiling for this 3-way combo (quick sanity check, no model calls)
# 
# Same computation as notebook 22, just for this specific channel combination -- tells us the best-case ceiling before spending GPU time reranking.

# In[ ]:


def get_relevant(qid, top_k=1000):
    return set(production_df[(production_df['query_id']==qid) & (production_df['rank']<=top_k)]['domain'].tolist())

oracle_recalls = []
for item in data:
    qid = item['query_id']
    relevant = get_relevant(qid)
    pool = set(query_pools[qid])
    oracle_recalls.append(len(pool & relevant) / len(relevant) if relevant else 0)

ORACLE_RECALL_1000 = np.mean(oracle_recalls)
print(f'[Oracle] 3-way union oracle Recall@1000 : {ORACLE_RECALL_1000:.3f}')
print(f'[Oracle] (2-way MiniLM+Linq ceiling from notebook 22 was 0.869, for comparison)')


# ## 4. Load cross-encoder reranker -- `BAAI/bge-reranker-v2-m3`, same model used throughout

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


# ## 5. Rerank the full 3-way union pool per query, keep top-1000
# 
# Checkpointed every 10 queries -- if the job hits its time budget mid-loop, already-reranked queries are skipped on resume.

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
            pd.DataFrame(all_results).to_csv(CHECKPOINT_PATH, index=False)

        if (time.time() - SCRIPT_START) / 60 > TIME_BUDGET_MINUTES:
            pd.DataFrame(all_results).to_csv(CHECKPOINT_PATH, index=False)
            done_now = len(set(r['query_id'] for r in all_results))
            print(f'[Rerank] Time budget ({TIME_BUDGET_MINUTES} min) reached at {done_now}/{len(data)} queries -- checkpoint saved. Rerun this same job to resume.')
            sys.exit(0)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(FINAL_PATH, index=False)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

AVG_RERANK_S = (sum(rerank_times) / len(rerank_times) / 1000) if rerank_times else float('nan')
print(f'[Rerank] Done!')
print(f'[Rerank] Avg rerank time (this session) : {AVG_RERANK_S:.1f}s/query')
print(f'[Rerank] Saved to        : {FINAL_PATH}')


# ## 6. Evaluation -- k=10,50,100,300,500,1000 (added 300 to match the list sizes Istari cares about)

# In[ ]:


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

print('[Eval] Evaluating 3-way full-pool rerank results...')
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

print('[Eval] === 3-WAY FULL-POOL RERANK RESULTS ===')
print(f'  {"k":<6} {"NDCG":>8} {"Precision":>10} {"Recall":>8} {"F1":>8}')
print('  ' + '-' * 46)
for k in K_VALUES:
    sub = eval_df[eval_df['k'] == k]
    print(f'  {k:<6} '
          f'{sub["ndcg"].mean():>8.3f} '
          f'{sub["precision"].mean():>10.3f} '
          f'{sub["recall"].mean():>8.3f} '
          f'{sub["f1"].mean():>8.3f}')


# ## 7. Comparison vs notebook 24 (2-way MiniLM+Linq full-pool rerank)

# In[ ]:


print('[Compare] Loading notebook 24 results for comparison...')
prior_eval_df = pd.read_csv('result/24_hybrid_minilm_linq_full_rerank/evaluation.csv')

comp_rows = []
for k in K_VALUES:
    triple_sub = eval_df[eval_df['k'] == k]
    prior_sub  = prior_eval_df[prior_eval_df['k'] == k] if k in prior_eval_df['k'].unique() else None
    row = {
        'k':                          k,
        'triple_ndcg':                round(triple_sub['ndcg'].mean(), 3),
        'triple_recall':              round(triple_sub['recall'].mean(), 3),
    }
    if prior_sub is not None and len(prior_sub):
        row['pair_ndcg']   = round(prior_sub['ndcg'].mean(), 3)
        row['pair_recall'] = round(prior_sub['recall'].mean(), 3)
    comp_rows.append(row)

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(RESULT_DIR / 'comparison_vs_prior.csv', index=False)
print(comp_df.to_string(index=False))

triple_r1000 = eval_df[eval_df['k']==1000]['recall'].mean()
print(f'\n[Compare] 3-way full-pool rerank Recall@1000 : {triple_r1000:.3f}')
print(f'[Compare] 3-way oracle ceiling                : {ORACLE_RECALL_1000:.3f}')
print(f'[Compare] Remaining gap                       : {ORACLE_RECALL_1000 - triple_r1000:.3f}')
print(f'\n[Compare] Saved to {RESULT_DIR / "comparison_vs_prior.csv"}')

