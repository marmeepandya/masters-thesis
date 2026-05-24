#!/usr/bin/env python
# coding: utf-8

# # Embedding Model Comparison: BM25 vs MiniLM vs BGE-large vs OpenAI
# 
# This notebook benchmarks **four retrieval approaches** over a corpus of ~98 700 companies,
# comparing both **retrieval quality** (NDCG@k, Precision@k, Recall@k) and **runtime performance**
# (index build time, encoding time, query latency).
# 
# | # | Method | Type | Dims | Params | Notes |
# |---|---|---|---|---|---|
# | 1 | **BM25** | Sparse | — | 0 | No encoding needed; inverted index |
# | 2 | **MiniLM-L6-v2** | Dense | 384 | 22M | Smallest/fastest open model |
# | 3 | **BGE-large-en-v1.5** | Dense | 1024 | 335M | Strongest open model |
# | 4 | **OpenAI text-embedding-3-small** | Dense API | 1536 | — | Commercial baseline Ralph suggested |
# 
# ### Why these four?
# They cover the full speed–quality–cost spectrum, which is exactly what Istari needs to evaluate:
# - BM25 = production-grade sparse baseline (no GPU needed)
# - MiniLM = smallest viable dense model (fastest, free)
# - BGE-large = strongest open-source dense model (free, needs GPU)
# - OpenAI = commercial API (best quality, ~$0.02/M tokens)
# 
# ### Notebook structure
# 1. Environment & GPU setup
# 2. Imports
# 3. Load dataset & build BM25 index
# 4. **Runtime benchmark** — index build & query latency for all methods
# 5. Encode with MiniLM → `embeddings_minilm.npy`
# 6. Encode with BGE-large → `embeddings_bge.npy`
# 7. Encode with OpenAI → `embeddings_openai.npy`
# 8. Run retrieval for all methods across 101 queries
# 9. Evaluate: Precision@k, Recall@k, NDCG@k
# 10. **Final comparison table** — quality + speed side by side

# ## 1 · Environment & GPU Setup
# 
# Load API keys and verify GPU availability before doing any heavy work.
# The GPU check is critical — BGE-large encoding on CPU takes ~30 min; on a T4/A100 it takes ~3 min.

# In[16]:


import os
from dotenv import load_dotenv
import torch

load_dotenv()

API_KEY    = os.getenv('API_KEY')
BASE_URL   = os.getenv('BASE_URL')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')  # add OPENAI_API_KEY to your .env file

# ── GPU check ──────────────────────────────────────────────────────────────
print(f'CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU            : {torch.cuda.get_device_name(0)}')
    print(f'VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    DEVICE = 'cuda'
else:
    print('WARNING: No GPU — BGE-large encoding will be very slow (~30 min)')
    DEVICE = 'cpu'

print(f'\nDevice: {DEVICE}')


# ## 2 · Imports
# 
# | Package | Role |
# |---|---|
# | `rank_bm25` | BM25Okapi sparse retrieval |
# | `nltk` | Punkt tokeniser for BM25 |
# | `sentence_transformers` | MiniLM and BGE-large encoders |
# | `openai` | OpenAI embedding API |
# | `faiss` | Fast vector similarity search |
# | `time` / `datetime` | Runtime measurement |
# | `pandas` / `numpy` | Data wrangling |

# In[17]:


from rank_bm25 import BM25Okapi
import json, re, time
import numpy as np
import pandas as pd
import faiss
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from datetime import datetime

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

print('All imports OK')


# ## 3 · Load Dataset & Build BM25 Index
# 
# Load the production results Excel file and build the BM25 index over company summaries.
# 
# **Pseudo-relevance labels:** A company is *relevant* for a query if it appears in the production top-100.
# This is the standard assumption when human annotations are unavailable.

# In[18]:


# ── Load production results ────────────────────────────────────────────────
production_df = pd.read_excel('dataset/production_results.xlsx')
print(f'Production results: {len(production_df):,} rows')

# ── Build de-duplicated company corpus ─────────────────────────────────────
all_companies = production_df.drop_duplicates(subset='domain').reset_index(drop=True)
print(f'Unique companies  : {len(all_companies):,}')

summaries = all_companies['summary'].fillna('').tolist()

# ── Load queries ────────────────────────────────────────────────────────────
with open('dataset/goi_search_results.json', 'r') as f:
    data = json.load(f)
print(f'Queries           : {len(data)}')

# ── Build BM25 index (and time it) ─────────────────────────────────────────
print('\nBuilding BM25 index...')
t0 = time.time()
corpus_tokens = [
    word_tokenize(s.lower()) if isinstance(s, str) else []
    for s in summaries
]
bm25 = BM25Okapi(corpus_tokens)
BM25_INDEX_TIME = time.time() - t0
print(f'BM25 index built in {BM25_INDEX_TIME:.2f}s on {len(corpus_tokens):,} companies')

# Save corpus for later use
all_companies[['domain', 'name', 'summary']].to_csv('result/3_baseline_comparison/company_corpus.csv', index=False)
print('Corpus saved to result/3_baseline_comparison/company_corpus.csv')


# In[19]:


import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    # Embed a single short string — costs less than $0.000001
    response = client.embeddings.create(
        input="test",
        model="text-embedding-3-small"
    )
    emb = response.data[0].embedding
    print(f" API key works!")
    print(f"   Model  : text-embedding-3-small")
    print(f"   Vector : {len(emb)} dimensions")
    print(f"   First 5: {[round(x, 4) for x in emb[:5]]}")

except Exception as e:
    print(f" API key failed: {e}")


# ## 4 · Runtime Benchmark
# 
# Before running the full 101-query evaluation, we measure **query latency** for each method
# on a small sample of 10 queries. This gives us the speed numbers for the comparison table.
# 
# > **Why measure separately?** Encoding time (one-off, offline) and query time (online, per-request)
# > are the two metrics Istari actually cares about. Encoding is a setup cost; query latency is what
# > the user experiences.
# 
# ### Measurements we collect per method:
# 
# | Metric | Description |
# |---|---|
# | **Corpus encoding time** | How long to embed all ~99k company summaries (one-time) |
# | **Query latency (ms)** | Average time to encode 1 query + retrieve top-1000 results |
# | **Index build time** | Time to build the FAISS / BM25 index |
# 
# We store all timings in `runtime_results` and display a final summary table at the end.

# In[20]:


# This dict collects all runtime measurements — filled in as we go
runtime_results = {
    'BM25':   {'index_build_s': BM25_INDEX_TIME, 'encode_s': 0.0, 'query_latency_ms': None},
    'MiniLM': {'index_build_s': None, 'encode_s': None, 'query_latency_ms': None},
    'BGE':    {'index_build_s': None, 'encode_s': None, 'query_latency_ms': None},
    'OpenAI': {'index_build_s': None, 'encode_s': None, 'query_latency_ms': None},
}

# Benchmark BM25 query latency (10 queries)
BENCHMARK_QUERIES = [item['query'] for item in data[:10]]
t0 = time.time()
for q in BENCHMARK_QUERIES:
    tokens = word_tokenize(q.lower())
    scores = bm25.get_scores(tokens)
    _ = np.argsort(scores)[::-1][:1000]
bm25_latency = (time.time() - t0) / len(BENCHMARK_QUERIES) * 1000
runtime_results['BM25']['query_latency_ms'] = bm25_latency
print(f'BM25 query latency: {bm25_latency:.1f} ms/query')


# ## 5 · Encode with MiniLM (`all-MiniLM-L6-v2`)
# 
# The smallest and fastest dense model — 22M parameters, 384-dimensional embeddings.
# Used as the lightweight dense baseline.
# 
# **No special flags needed** for MiniLM — unlike BGE, it does not require `normalize_embeddings`.

# In[ ]:


print('Loading MiniLM...')
model_minilm = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

print('Encoding with MiniLM...')
t0 = time.time()
emb_minilm = model_minilm.encode(
    summaries,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
)
minilm_encode_time = time.time() - t0
runtime_results['MiniLM']['encode_s'] = minilm_encode_time
print(f'MiniLM encoding: {minilm_encode_time/60:.1f} min  |  shape: {emb_minilm.shape}')

np.save('result/3_baseline_comparison/embeddings_minilm.npy', emb_minilm)
print('Saved to result/3_baseline_comparison/embeddings_minilm.npy')


# ## 6 · Encode with BGE-large (`BAAI/bge-large-en-v1.5`)
# 
# 335M-parameter BERT-large model, 1024-dimensional embeddings.
# Strongest open-source retrieval model — significantly better than MiniLM on MTEB benchmarks.
# 
# **`normalize_embeddings=True` is required** — BGE is trained with cosine similarity,
# so embeddings must be unit-normalised for correct ranking.

# In[ ]:


print('Loading BGE-large...')
model_bge = SentenceTransformer('BAAI/bge-large-en-v1.5', device=DEVICE)

print('Encoding with BGE-large...')
t0 = time.time()
emb_bge = model_bge.encode(
    summaries,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,  # REQUIRED for BGE
)
bge_encode_time = time.time() - t0
runtime_results['BGE']['encode_s'] = bge_encode_time
print(f'BGE encoding: {bge_encode_time/60:.1f} min  |  shape: {emb_bge.shape}')

np.save('result/3_baseline_comparison/embeddings_bge.npy', emb_bge)
print('Saved to result/3_baseline_comparison/embeddings_bge.npy')


# ## 7 · Encode with OpenAI (`text-embedding-3-small`)
# 
# Commercial API model — 1536-dimensional embeddings, no local compute needed.
# Costs ~$0.02 per million tokens. Encoding ~99k summaries costs approximately **$0.15–0.20 total**.
# 
# **Key differences from local models:**
# - No GPU needed — runs via HTTP requests to OpenAI
# - Slower than local GPU encoding due to network latency
# - Cannot be self-hosted — introduces API dependency and ongoing cost
# - `normalize_embeddings` not needed — OpenAI returns normalised vectors by default

# In[22]:


client = OpenAI(api_key=OPENAI_KEY)

def encode_openai_batched(texts, model='text-embedding-3-small', batch_size=500):
    """Encode texts using OpenAI API in batches. Returns numpy array."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Replace empty strings — OpenAI rejects them
        batch = [t if t.strip() else 'unknown company' for t in batch]
        response = client.embeddings.create(input=batch, model=model)
        batch_embs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embs)
        if (i // batch_size) % 10 == 0:
            print(f'  Encoded {min(i+batch_size, len(texts)):,}/{len(texts):,}', end='\r')
    print()
    return np.array(all_embeddings, dtype='float32')

print('Encoding with OpenAI text-embedding-3-small...')
t0 = time.time()
emb_openai = encode_openai_batched(summaries)
openai_encode_time = time.time() - t0
runtime_results['OpenAI']['encode_s'] = openai_encode_time
print(f'OpenAI encoding: {openai_encode_time/60:.1f} min  |  shape: {emb_openai.shape}')

np.save('result/3_baseline_comparison/embeddings_openai.npy', emb_openai)
print('Saved to result/3_baseline_comparison/embeddings_openai.npy')


# ## 8 · Build FAISS Indices & Measure Query Latency
# 
# We build one FAISS index per embedding model, then measure **query latency** on 10 benchmark queries.
# 
# Index type choice:
# - **MiniLM** → `IndexFlatL2` (not normalised, L2 distance)
# - **BGE** → `IndexFlatIP` (normalised, cosine = inner product)
# - **OpenAI** → `IndexFlatIP` (OpenAI returns unit-normalised vectors)

# In[21]:


def build_faiss_index(embeddings, use_ip=False):
    """Build a flat FAISS index. use_ip=True for normalised embeddings (cosine similarity)."""
    dim = embeddings.shape[1]
    if use_ip:
        index = faiss.IndexFlatIP(dim)  # inner product = cosine for unit vectors
    else:
        index = faiss.IndexFlatL2(dim)  # L2 distance
    index.add(embeddings.astype('float32'))
    return index

def measure_query_latency(model_fn, index, queries, n_results=1000):
    """Average query latency in ms over a list of queries."""
    times = []
    for q in queries:
        t0 = time.time()
        q_emb = model_fn(q)
        index.search(q_emb, n_results)
        times.append((time.time() - t0) * 1000)
    return np.mean(times), np.std(times)

# ── Build MiniLM index ──────────────────────────────────────────────────────
print('Building MiniLM FAISS index...')
t0 = time.time()
index_minilm = build_faiss_index(emb_minilm, use_ip=False)
runtime_results['MiniLM']['index_build_s'] = time.time() - t0
print(f'  MiniLM index: {index_minilm.ntotal:,} vectors ({runtime_results["MiniLM"]["index_build_s"]:.2f}s)')

# ── Build BGE index ─────────────────────────────────────────────────────────
print('Building BGE FAISS index...')
t0 = time.time()
index_bge = build_faiss_index(emb_bge, use_ip=True)
runtime_results['BGE']['index_build_s'] = time.time() - t0
print(f'  BGE index   : {index_bge.ntotal:,} vectors ({runtime_results["BGE"]["index_build_s"]:.2f}s)')

# ── Build OpenAI index ──────────────────────────────────────────────────────
print('Building OpenAI FAISS index...')
t0 = time.time()
index_openai = build_faiss_index(emb_openai, use_ip=True)
runtime_results['OpenAI']['index_build_s'] = time.time() - t0
print(f'  OpenAI index: {index_openai.ntotal:,} vectors ({runtime_results["OpenAI"]["index_build_s"]:.2f}s)')

# ── Query latency: encode 1 query + search ──────────────────────────────────
print('\nMeasuring query latency (10 queries each)...')

def encode_minilm_query(q):
    return model_minilm.encode([q], convert_to_numpy=True).astype('float32')

def encode_bge_query(q):
    return model_bge.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')

def encode_openai_query(q):
    resp = client.embeddings.create(input=[q], model='text-embedding-3-small')
    return np.array([resp.data[0].embedding], dtype='float32')

mean_ms, std_ms = measure_query_latency(encode_minilm_query, index_minilm, BENCHMARK_QUERIES)
runtime_results['MiniLM']['query_latency_ms'] = mean_ms
print(f'  MiniLM  : {mean_ms:.1f} ± {std_ms:.1f} ms/query')

mean_ms, std_ms = measure_query_latency(encode_bge_query, index_bge, BENCHMARK_QUERIES)
runtime_results['BGE']['query_latency_ms'] = mean_ms
print(f'  BGE     : {mean_ms:.1f} ± {std_ms:.1f} ms/query')

mean_ms, std_ms = measure_query_latency(encode_openai_query, index_openai, BENCHMARK_QUERIES)
runtime_results['OpenAI']['query_latency_ms'] = mean_ms
print(f'  OpenAI  : {mean_ms:.1f} ± {std_ms:.1f} ms/query')

print(f'  BM25    : {runtime_results["BM25"]["query_latency_ms"]:.1f} ms/query')


# ## 9 · Run Retrieval for All 101 Queries
# 
# Retrieve top-1000 results per query for all four methods.
# Results are saved to separate CSVs so they can be reloaded without re-running encoding.

# In[ ]:


def run_dense_retrieval(model_fn, index, data, all_companies, label):
    """Run top-1000 retrieval for all queries using a dense model + FAISS index."""
    results = []
    print(f'Running {label} retrieval...')
    for item in data:
        query_id = item['query_id']
        query    = item['query']
        q_emb    = model_fn(query)
        scores, indices = index.search(q_emb, 1000)
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            results.append({
                'query_id': query_id, 'query': query,
                'rank': rank + 1, 'score': float(score),
                'domain': all_companies.iloc[idx]['domain'],
                'name':   all_companies.iloc[idx]['name'],
                'summary':all_companies.iloc[idx]['summary'],
            })
    df = pd.DataFrame(results)
    df.to_csv(f'result/3_baseline_comparison/retrieval_{label.lower()}.csv', index=False)
    print(f'  Saved {len(df):,} rows to result/3_baseline_comparison/retrieval_{label.lower()}.csv')
    return df

# ── BM25 ─────────────────────────────────────────────────────────────────────
print('Running BM25 retrieval...')
bm25_rows = []
for item in data:
    qid, q = item['query_id'], item['query']
    scores = bm25.get_scores(word_tokenize(q.lower()))
    top_idx = np.argsort(scores)[::-1][:1000]
    for rank, idx in enumerate(top_idx):
        bm25_rows.append({
            'query_id': qid, 'query': q,
            'rank': rank+1, 'score': float(scores[idx]),
            'domain': all_companies.iloc[idx]['domain'],
            'name':   all_companies.iloc[idx]['name'],
            'summary':all_companies.iloc[idx]['summary'],
        })
bm25_df = pd.DataFrame(bm25_rows)
bm25_df.to_csv('result/retrieval_bm25.csv', index=False)
print(f'  Saved {len(bm25_df):,} rows to result/retrieval_bm25.csv')

# ── Dense models ─────────────────────────────────────────────────────────────
minilm_df = run_dense_retrieval(encode_minilm_query, index_minilm, data, all_companies, 'MiniLM')
bge_df    = run_dense_retrieval(encode_bge_query,    index_bge,    data, all_companies, 'BGE')
openai_df = run_dense_retrieval(encode_openai_query, index_openai, data, all_companies, 'OpenAI')


# ## 10 · Evaluation — Precision@k, Recall@k, NDCG@k
# 
# ### Metrics
# 
# | Metric | What it measures |
# |---|---|
# | **Precision@k** | Of the top-k results, what fraction are relevant? (user-facing quality) |
# | **Recall@k** | Of all relevant companies, what fraction appear in top-k? (coverage) |
# | **NDCG@k** | Ranking quality — rewards relevant results ranked higher (primary metric) |
# 
# ### Relevance labels
# A company is *relevant* for a query if it appears in the **production top-100** for that query
# (pseudo-relevance assumption — standard when no human labels are available).

# In[ ]:


# ── Metric functions ────────────────────────────────────────────────────────
def get_relevant(query_id, top_k=100):
    return set(production_df[
        (production_df['query_id'] == query_id) &
        (production_df['rank'] <= top_k)
    ]['domain'].tolist())

def precision_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / k if k else 0

def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0

def dcg_at_k(retrieved, relevant, k):
    return sum(
        1 / np.log2(i + 2)
        for i, d in enumerate(retrieved[:k]) if d in relevant
    )

def ndcg_at_k(retrieved, relevant, k):
    ideal = dcg_at_k(list(relevant), relevant, k)
    return dcg_at_k(retrieved, relevant, k) / ideal if ideal else 0

# ── Evaluate all methods ─────────────────────────────────────────────────────
k_values = [10, 50, 100, 500, 1000]

method_dfs = {
    'BM25':   bm25_df,
    'MiniLM': minilm_df,
    'BGE':    bge_df,
    'OpenAI': openai_df,
}

all_eval_rows = []

for item in data:
    qid      = item['query_id']
    query    = item['query']
    relevant = get_relevant(qid)

    for method, df in method_dfs.items():
        retrieved = df[df['query_id'] == qid].sort_values('rank')['domain'].tolist()
        for k in k_values:
            all_eval_rows.append({
                'query_id': qid, 'query': query, 'method': method, 'k': k,
                'precision': precision_at_k(retrieved, relevant, k),
                'recall':    recall_at_k(retrieved, relevant, k),
                'ndcg':      ndcg_at_k(retrieved, relevant, k),
            })

eval_df = pd.DataFrame(all_eval_rows)
eval_df.to_csv('result/3_baseline_comparison/evaluation_all_methods.csv', index=False)
print(f'Evaluation saved: {len(eval_df):,} rows')
print('\n=== NDCG@k RESULTS ===')
pivot = eval_df.groupby(['method','k'])['ndcg'].mean().unstack('k')
print(pivot.round(3).to_string())


# ## 11 · Final Comparison Table — Quality + Speed
# 
# This is the key output for your thesis and your meeting with Ralph.
# It shows both retrieval quality (NDCG@10) and runtime in a single table.

# In[ ]:


# ── Quality summary ─────────────────────────────────────────────────────────
quality = eval_df.groupby(['method','k'])['ndcg'].mean().unstack('k')
prec10  = eval_df[eval_df['k']==10].groupby('method')['precision'].mean()
rec1000 = eval_df[eval_df['k']==1000].groupby('method')['recall'].mean()

# ── Runtime summary ──────────────────────────────────────────────────────────
METHOD_ORDER = ['BM25', 'MiniLM', 'BGE', 'OpenAI']

rows = []
for m in METHOD_ORDER:
    r = runtime_results[m]
    encode_str = f"{r['encode_s']/60:.1f} min" if r['encode_s'] else 'N/A'
    rows.append({
        'Method':          m,
        'Corpus encode':   encode_str,
        'Index build (s)': f"{r['index_build_s']:.1f}" if r['index_build_s'] else '—',
        'Query latency':   f"{r['query_latency_ms']:.0f} ms",
        'NDCG@10':         f"{quality.loc[m, 10]:.3f}" if m in quality.index else '—',
        'NDCG@100':        f"{quality.loc[m, 100]:.3f}" if m in quality.index else '—',
        'Prec@10':         f"{prec10[m]:.3f}" if m in prec10.index else '—',
        'Recall@1000':     f"{rec1000[m]:.3f}" if m in rec1000.index else '—',
        'Free?':           'Yes' if m in ['BM25','MiniLM','BGE'] else '~$0.15 corpus',
        'GPU needed?':     'No' if m in ['BM25','MiniLM','OpenAI'] else 'Yes (encoding)',
    })

summary_df = pd.DataFrame(rows).set_index('Method')
print('=== FINAL COMPARISON: QUALITY + RUNTIME ===')
print(summary_df.to_string())

summary_df.to_csv('result/3_baseline_comparison/final_comparison.csv')
print('\nSaved to result/3_baseline_comparison/final_comparison.csv')


# ## 12 · NDCG@k Plot
# 
# Visualise how each method's ranking quality changes across different cutoff values.
# This plot is suitable for including in your thesis (Ralph said visualisations are fine where they add value).

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
METHOD_ORDER = ['BM25', 'MiniLM', 'BGE', 'OpenAI']
colors  = {'BM25': '#2196F3', 'MiniLM': '#FF9800', 'BGE': '#4CAF50', 'OpenAI': '#9C27B0'}
markers = {'BM25': 'o',       'MiniLM': 's',        'BGE': '^',       'OpenAI': 'D'}

# ── Left: NDCG@k ─────────────────────────────────────────────────────────────
ax = axes[0]
for m in METHOD_ORDER:
    vals = [eval_df[(eval_df['method']==m) & (eval_df['k']==k)]['ndcg'].mean() for k in k_values]
    ax.plot(k_values, vals, marker=markers[m], color=colors[m], label=m, linewidth=2, markersize=6)
ax.set_xscale('log')
ax.set_xticks(k_values)
ax.set_xticklabels(k_values)
ax.set_xlabel('k (cutoff)', fontsize=12)
ax.set_ylabel('NDCG@k', fontsize=12)
ax.set_title('Retrieval Quality (NDCG@k)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ── Right: Query latency bar chart ───────────────────────────────────────────
ax2 = axes[1]
latencies = [runtime_results[m]['query_latency_ms'] for m in METHOD_ORDER]
bars = ax2.bar(METHOD_ORDER, latencies,
               color=[colors[m] for m in METHOD_ORDER], edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, latencies):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.0f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_ylabel('Query latency (ms)', fontsize=12)
ax2.set_title('Query Latency (encode + search)', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('result/3_baseline_comparison/comparison_plot.png', bbox_inches='tight', dpi=150)
plt.show()
print('Plot saved to result/3_baseline_comparison/comparison_plot.png')


# ## 13 · Key Findings & Thesis Narrative
# 
# **Fill in the table below after running the evaluation.**
# 
# ### Speed–Quality tradeoff summary
# 
# | Method | NDCG@10 | Query latency | Verdict |
# |---|---|---|---|
# | BM25 | — | — ms | Fast, interpretable, no GPU. Struggles with vocabulary mismatch |
# | MiniLM | — | — ms | Good quality/speed tradeoff. Best for CPU-only deployment |
# | BGE-large | — | — ms | Strongest open model. Needs GPU for encoding |
# | OpenAI | — | — ms | Best quality? But API dependency and per-query cost |
# 
# ### Expected narrative (based on IR literature)
# - All dense models should outperform BM25 at **low k** (top-10, top-50) — semantic matching surfaces relevant companies faster
# - BM25 may still be competitive at **high k** (top-500, top-1000) — broad keyword coverage
# - BGE-large should beat MiniLM across all k — larger model, better representations
# - OpenAI should be competitive with BGE-large — may win or lose depending on domain
# 
# This motivates **hybrid retrieval** as the next step: combine BM25 + best dense model via
# Reciprocal Rank Fusion (RRF) to get the best of both worlds.
