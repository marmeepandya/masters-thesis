#!/usr/bin/env python
# coding: utf-8

# # BM25 Baseline Retrieval
# ## Sparse keyword-based search over 98,716 companies
# 
# **What this notebook does:**
# 
# Implements the BM25 sparse retrieval baseline — the industry-standard keyword search
# method used by systems like Elasticsearch. Used as the comparison point for all
# dense retrieval methods.
# 
# **BM25 in plain English:**
# BM25 counts how often query words appear in each company's text, with two smart adjustments:
# 1. Repeated words give diminishing returns (a word appearing 100× is not 100× more useful than once)
# 2. Longer documents are not unfairly rewarded just for being longer
# 
# **What is being indexed:**
# All available company fields are combined into one rich text string per company —
# not just the summary. This includes name, country, state, city, industry (NACE),
# size, and summary keywords. This gives BM25 more signal to match against.
# 
# **Folder structure:**
# ```
# result/
# └── 01_baseline_bm25/
#     ├── bm25_results.csv          # Top-1000 BM25 results per query (101 queries)
#     └── evaluation_bm25.csv       # NDCG, Precision, Recall, F1 @ k∈{10,50,100, 500, 1000}
# ```
# 
# ### Notebook structure
# 1. Environment setup
# 2. Imports
# 3. Load dataset & build corpus
# 4. Build BM25 index
# 5. Smoke test
# 6. Run all 101 queries → results
# 7. Evaluation — NDCG, Precision, Recall, F1 @ k∈{10,50,100,1000}
# 8. Final summary

# ## 1 · Environment Setup
# 
# Load environment variables and create the output folder if it does not exist.

# In[ ]:


import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY  = os.getenv('API_KEY')
BASE_URL = os.getenv('BASE_URL')

# Create result folder
RESULT_DIR = Path('result/01_baseline_bm25_summary')
RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f'[Setup] Result folder: {RESULT_DIR}/ — ready')


# ## 2 · Imports
# 
# | Package | Role |
# |---|---|
# | `rank_bm25` | BM25Okapi sparse retrieval index |
# | `nltk / word_tokenize` | Splits text into tokens (words) for BM25 |
# | `pandas / numpy` | Data loading and numerical operations |
# | `time` | Measuring query latency |
# | `json` | Loading the queries file |

# In[ ]:


from rank_bm25 import BM25Okapi
import json, time
import numpy as np
import pandas as pd
from pathlib import Path
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

RESULT_DIR = Path('result/01_baseline_bm25_summary')
print('[Imports] All packages loaded successfully')


# ## 3 · Load Dataset & Build Corpus
# 
# **Source:** `dataset/production_results.xlsx`
# 
# This file contains 101 queries × 1,000 production-ranked results = 101,000 rows.
# We de-duplicate on `domain` to get one row per company — our retrieval corpus.
# 
# **Why use all fields for BM25?**
# BM25 is a keyword matcher. If we only index the `summary`, a query like
# *"software companies in Germany"* cannot match the `country` field.
# By combining all fields into one rich text string, BM25 can match on
# country, city, industry code, company name, and keywords — not just the summary.
# 
# **Rich text format per company:**
# ```
# Software Genesis, Inc. | Country: United States | State: Illinois |
# Size: Micro (0-9) | Industry: NACE K: Telecommunication... |
# Software Genesis is a software development company... |
# Keywords: software development, consumer market, software company...
# ```

# In[ ]:


print('[Load] Loading production results...')
results_df = pd.read_excel('dataset/production_results.xlsx')
print(f'[Load] Total rows        : {len(results_df):,}')
print(f'[Load] Columns available : {list(results_df.columns)}')

# De-duplicate: one row per company 
all_companies = results_df.drop_duplicates(subset='domain').reset_index(drop=True)
print(f'[Load] Unique companies  : {len(all_companies):,}')

# Load queries  
with open('dataset/goi_search_results.json', 'r') as f:
    data = json.load(f)
print(f'[Load] Queries           : {len(data)}')

# Build rich text per company combining ALL available fields 
# BM25 can only match what it sees — more fields = better matching
print('[Load] Building rich text for each company...')

def build_rich_text(row):
    """Summary only — Set A baseline experiments."""
    return str(row.get('summary', '')) if pd.notna(row.get('summary')) else ''

rich_texts = [build_rich_text(row) for _, row in all_companies.iterrows()]

# Sanity check
print(f'[Load] Sample rich text (first company):')
print(f'  {rich_texts[0][:250]}...')


# ## 4 · Build BM25 Index
# 
# **Tokenisation:** Each rich text string is split into lowercase words using NLTK's
# punkt tokeniser. This is the same tokeniser applied to queries at search time.
# 
# **Index build time** is measured and logged — this is a one-time offline cost.
# 
# **BM25 hyperparameters** (defaults):
# - `k1 = 1.5` — term frequency saturation: controls how much repeated words contribute
# - `b = 0.75` — length normalisation: penalises very long documents slightly

# In[4]:


print('[Index] Tokenising rich texts...')
t0     = time.time()
corpus = [
    word_tokenize(text.lower()) if isinstance(text, str) and text else []
    for text in rich_texts
]
tok_time = time.time() - t0
print(f'[Index] Tokenisation done : {tok_time:.1f}s')
print(f'[Index] Avg tokens/doc    : {sum(len(c) for c in corpus)/len(corpus):.0f}')

print('[Index] Building BM25 index...')
t0 = time.time()
bm25 = BM25Okapi(corpus)
INDEX_BUILD_TIME = time.time() - t0
print(f'[Index] Index built in    : {INDEX_BUILD_TIME:.2f}s')
print(f'[Index] Companies indexed : {len(corpus):,}')


# ## 5 · Smoke Test
# 
# Test the index on one query before running all 101.
# Also inspect what the top results actually are — this reveals the
# vocabulary mismatch problem (BM25 returning investment banks for
# *"software companies"* because they mention "software" in their portfolio description).

# In[5]:


test_query  = 'software companies'
test_tokens = word_tokenize(test_query.lower())
test_scores = bm25.get_scores(test_tokens)
top10       = np.argsort(test_scores)[::-1][:10]

print(f'[Smoke] Query: "{test_query}"')
print(f'[Smoke] Top-10 BM25 results:')
print(f'  {"Rank":<6} {"Score":>8}  {"Name":<40}  Country')
print('  ' + '-' * 70)
for rank, idx in enumerate(top10, 1):
    row = all_companies.iloc[idx]
    name    = str(row.get('name', ''))[:38]
    country = str(row.get('country', ''))
    print(f'  {rank:<6} {test_scores[idx]:>8.4f}  {name:<40}  {country}')

print('\n[Smoke] Full summary of rank-1 result:')
print(f'  {str(all_companies.iloc[top10[0]]["summary"])[:400]}')


# ## 6 · Run BM25 Across All 101 Queries
# 
# Retrieve the **top 1,000** companies per query.
# Query latency is measured per query and averaged at the end.
# 
# **What latency includes:**
# - Tokenising the query string
# - Computing BM25 scores for all 98,716 companies
# - Sorting to get top-1,000
# 
# **Output:** `result/01_baseline_bm25/bm25_results.csv`

# In[6]:


print(f'[Run] Starting BM25 retrieval for {len(data)} queries...')
print(f'[Run] Retrieving top-1000 per query')
print('-' * 55)

all_bm25_results = []
query_times      = []
total_start      = time.time()

for i, item in enumerate(data):
    query_id = item['query_id']
    query    = item['query']

    # Tokenise + score + sort — time the full query  
    t0          = time.perf_counter()
    query_tokens = word_tokenize(query.lower())
    scores       = bm25.get_scores(query_tokens)
    top_indices  = np.argsort(scores)[::-1][:1000]
    query_ms     = (time.perf_counter() - t0) * 1000
    query_times.append(query_ms)

    for rank, idx in enumerate(top_indices):
        company = all_companies.iloc[idx]
        all_bm25_results.append({
            'query_id':   query_id,
            'query':      query,
            'rank':       rank + 1,
            'bm25_score': float(scores[idx]),
            'domain':     company['domain'],
            'name':       company.get('name', ''),
            'country':    company.get('country', ''),
            'summary':    company.get('summary', ''),
        })

    if (i + 1) % 20 == 0 or (i + 1) == len(data):
        elapsed   = time.time() - total_start
        remaining = (len(data) - i - 1) * elapsed / (i + 1)
        print(f'[Run] {i+1:3d}/{len(data)}  |  '
              f'avg {sum(query_times)/len(query_times):.1f}ms/query  |  '
              f'~{remaining:.0f}s remaining')

# Save 
bm25_df = pd.DataFrame(all_bm25_results)
bm25_df.to_csv(RESULT_DIR / 'bm25_results.csv', index=False)

AVG_LATENCY_MS = sum(query_times) / len(query_times)
print('-' * 55)
print(f'[Run] Done!')
print(f'[Run] Total results      : {len(bm25_df):,}')
print(f'[Run] Avg query latency  : {AVG_LATENCY_MS:.1f}ms')
print(f'[Run] Index build time   : {INDEX_BUILD_TIME:.2f}s')
print(f'[Run] Saved to           : result/01_baseline_bm25/bm25_results.csv')


# ## 7 · Evaluation — NDCG, Precision, Recall, F1 @ k
# 
# ### Pseudo-relevance labels
# A company is **relevant** for a query if it appears in the **production top-100**.
# This is the standard assumption when human annotations are unavailable.
# 
# ### Metrics explained
# 
# | Metric | Formula | What it measures |
# |---|---|---|
# | **Precision@k** | \|Retrieved ∩ Relevant\| / k | Of the top-k results, what fraction are relevant? |
# | **Recall@k** | \|Retrieved ∩ Relevant\| / \|Relevant\| | Of all relevant companies, what fraction did we find? |
# | **F1@k** | 2 × P × R / (P + R) | Harmonic mean of Precision and Recall — balances both |
# | **NDCG@k** | DCG@k / IDCG@k | Ranking quality — rewards relevant results ranked higher |
# 
# All metrics reported at k ∈ {10, 50, 100, 500, 1000}.

# In[7]:


print('[Eval] Loading production labels...')
production_df = pd.read_excel('dataset/production_results.xlsx')

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

# Evaluate all queries 
print('[Eval] Computing metrics for all queries...')
eval_rows = []

for i, item in enumerate(data):
    qid       = item['query_id']
    query     = item['query']
    relevant  = get_relevant(qid)
    retrieved = (
        bm25_df[bm25_df['query_id'] == qid]
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
eval_df.to_csv(RESULT_DIR / 'evaluation_bm25.csv', index=False)
print(f'[Eval] Saved to result/01_baseline_bm25/evaluation_bm25.csv')


# ## 8 · Final Summary
# 
# Full results table averaged across all 101 queries.
# These numbers are the BM25 baseline that all other methods are compared against.

# In[8]:


print('[Summary] BM25 BASELINE RESULTS')
print('[Summary] ============================================================')
print(f'\n[Summary] Index build time  : {INDEX_BUILD_TIME:.2f}s')
print(f'[Summary] Avg query latency : {AVG_LATENCY_MS:.1f}ms')
print(f'[Summary] Companies indexed : {len(all_companies):,}')
print(f'[Summary] Queries evaluated : {len(data)}')
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

