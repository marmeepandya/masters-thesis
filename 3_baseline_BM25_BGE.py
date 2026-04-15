#!/usr/bin/env python
# coding: utf-8

# # Baseline Retrieval: BM25 vs. BGE-large Embedding Search
# 
# This notebook implements and evaluates **two classical retrieval baselines** over a corpus of ~98 700 companies:
# 
# | Method | How it works |
# |---|---|
# | **BM25** (Okapi BM25) | Sparse term-matching; ranks documents by weighted keyword overlap with the query |
# | **BGE-large-en-v1.5 + FAISS** | Dense semantic search; encodes text into **1024-d** vectors using a 335M-parameter BERT-large model and finds nearest neighbours |
# 
# Both systems are evaluated against a **production ranking** (used as pseudo-relevance labels) using **Recall@k**, **Precision@k**, and **NDCG@k** at cut-offs k ∈ {10, 50, 100, 500, 1000}.
# 
# ### Notebook structure
# 1. Environment setup  
# 2. Import libraries  
# 3. Load dataset (`production_results.xlsx`)  
# 4. Build & test the BM25 index  
# 5. Inspect BM25 top results  
# 6. Run BM25 across all 101 queries → `result/BM25-BGE/bm25_results_BGE.csv`  
# 7. Persist the company corpus  
# 8. Encode company summaries with BGE-large → `result/BM25-BGE/company_embeddings.npy`  
# 9. Build a FAISS index & run embedding search → `result/BM25-BGE/embedding_results.csv`  
# 10. Evaluate Recall & Precision  
# 11. Precision comparison (head-to-head)  
# 12. NDCG evaluation

# ## 1 · Environment Setup
# 
# Load environment variables from a `.env` file.  
# Two variables are expected:
# 
# - `API_KEY` – API key for any external service calls (not used in the retrieval pipeline itself but available for later reranking steps).  
# - `BASE_URL` – Base endpoint URL for the same service.

# In[ ]:


import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")


# ## 2 · Imports
# 
# Key packages:
# 
# | Package | Role |
# |---|---|
# | `rank_bm25` | Pure-Python BM25Okapi implementation |
# | `nltk` / `word_tokenize` | Tokenisation for BM25 (punkt tokeniser) |
# | `sentence_transformers` | Loads `BAAI/bge-large-en-v1.5` to produce **1024-d** dense embeddings |
# | `faiss` | Facebook AI Similarity Search – fast exact/approximate nearest-neighbour lookup |
# | `pandas` / `numpy` | Data wrangling and numerical operations |

# In[ ]:


from rank_bm25 import BM25Okapi
import json
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import requests
import json
import pandas as pd
import re
nltk.download('punkt')
nltk.download('punkt_tab')

from sentence_transformers import SentenceTransformer
import faiss
import time


# ## 3 · Load Production Dataset
# 
# `production_results.xlsx` contains the output of the existing production search system. (EVAL Dataset)  
# Each row represents a (query, company) pair with its production rank.
# 
# This dataframe serves two purposes:
# 1. **Corpus source** – the `domain` and `summary` columns are used to build the retrieval indices.  
# 2. **Pseudo-relevance labels** – companies appearing in the production top-100 are treated as *relevant* for evaluation.

# In[ ]:


results_df = pd.read_excel("dataset/production_results.xlsx")


# ## 4 · Build the BM25 Index & Smoke-Test
# 
# ### What is BM25?
# BM25 (Best Match 25) is the industry-standard sparse retrieval function. It extends TF-IDF with:
# - **Term-frequency saturation** – repeated terms give diminishing returns (controlled by `k1`).
# - **Document-length normalisation** – shorter documents are not unfairly penalised (controlled by `b`).
# 
# `BM25Okapi` from `rank_bm25` uses the Okapi variant with default `k1=1.5, b=0.75`.
# 
# ### Pipeline
# ```
# company summaries ──► lower-case word_tokenize ──► BM25Okapi(corpus)
#                                                         │
# query string ──────► lower-case word_tokenize ──► get_scores() ──► ranked list
# ```
# 
# A quick smoke-test with the query *"software companies"* confirms the index is working.

# In[ ]:


# Step 1: Build corpus from production results
# Get unique companies across all queries
all_companies = results_df.drop_duplicates(subset='domain').reset_index(drop=True)
print(f"Unique companies in dataset: {len(all_companies)}")

# Step 2: Tokenize summaries
print("Tokenizing company summaries...")
corpus = []
for summary in all_companies['summary']:
    if isinstance(summary, str):
        tokens = word_tokenize(summary.lower())
    else:
        tokens = []
    corpus.append(tokens)

# Step 3: Build BM25 index
print("Building BM25 index...")
bm25 = BM25Okapi(corpus)
print(f"BM25 index built on {len(corpus)} companies!")

# Step 4: Test with first query
test_query = "software companies"
query_tokens = word_tokenize(test_query.lower())
scores = bm25.get_scores(query_tokens)

# Get top 10
top_indices = np.argsort(scores)[::-1][:10]
print(f"\nTop 10 BM25 results for: '{test_query}'")
for rank, idx in enumerate(top_indices):
    print(f"{rank+1}. {all_companies.iloc[idx]['domain']} — score: {scores[idx]:.4f}")


# ## 5 · Inspect Top BM25 Results
# 
# Print the full company summary for the top-5 BM25 hits to sanity-check result quality.  
# All returned companies are investment firms that discuss *software* in their descriptions — reasonable given BM25's keyword-matching nature.

# In[ ]:


# Check what these companies actually are
for idx in top_indices[:5]:
    company = all_companies.iloc[idx]
    print(f"Domain: {company['domain']}")
    print(f"Summary: {company['summary']}")
    print("-" * 50)


# ## 6 · Run BM25 Across All 101 Queries
# 
# Iterate over every query in `goi_search_results.json` and retrieve the **top 1 000** BM25 companies.
# 
# - Each query is tokenised the same way as the corpus (lower-case `word_tokenize`).
# - Results are collected into a flat dataframe and written to **`bm25_results.csv`**.
# - Retrieving 1 000 candidates per query gives headroom for later re-ranking stages.

# In[ ]:


with open('dataset/goi_search_results.json', 'r') as f:
    data = json.load(f)

all_bm25_results = []

print("Running BM25 for all queries...")
for item in data:
    query_id = item['query_id']
    query = item['query']

    # Tokenize query
    query_tokens = word_tokenize(query.lower())

    # Get BM25 scores
    scores = bm25.get_scores(query_tokens)

    # Get top 1000
    top_indices = np.argsort(scores)[::-1][:1000]

    for rank, idx in enumerate(top_indices):
        company = all_companies.iloc[idx]
        all_bm25_results.append({
            'query_id': query_id,
            'query': query,
            'rank': rank + 1,
            'bm25_score': scores[idx],
            'domain': company['domain'],
            'name': company['name'],
            'summary': company['summary']
        })

bm25_df = pd.DataFrame(all_bm25_results)
print(f"Total BM25 results: {len(bm25_df)}")

# Save
bm25_df.to_csv('result/BM25-BGE/bm25_results_BGE.csv', index=False)
print("Saved to result/BM25-BGE/bm25_results_BGE.csv!")


# ## 7 · Save Company Corpus
# 
# Persist the de-duplicated company table (`domain`, `name`, `summary`) to **`company_corpus_BGE.csv`**.
# 
# This file is used in the next section to encode summaries into dense embeddings without re-loading the full production Excel file.

# ## 8 · Dense Encoding with BGE-large
# 
# ### Model: `BAAI/bge-large-en-v1.5`
# - A **335M-parameter** BERT-large model fine-tuned for dense retrieval by the Beijing Academy of AI.  
# - Produces **1024-dimensional** L2-normalised embeddings (vs. 384-d for MiniLM).  
# - Significantly stronger than MiniLM on retrieval benchmarks (MTEB ~54 vs ~44), at the cost of longer encoding time.  
# - Requires `normalize_embeddings=True` — BGE models are trained with cosine similarity, so normalisation is essential for correct ranking.
# 
# ### GPU check
# Before encoding, we verify the model is running on GPU (`cuda`). Without GPU acceleration, encoding ~99k summaries would take 20–40 minutes. On a T4/A100 GPU it completes in 2–5 minutes.
# 
# ### Why save embeddings immediately?
# Encoding takes several minutes; saving to **`result/BM25-BGE/company_embeddings.npy`** avoids re-running this step if the kernel restarts.

# In[ ]:


import torch

# ── GPU check — must be True for reasonable speed ──────────────────────────
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected — encoding will be very slow on CPU")


# In[ ]:


# Load corpus
all_companies = pd.read_csv('dataset/company_corpus.csv')
print(f"Total companies to encode: {len(all_companies)}")

# Load model — explicitly move to GPU if available
print("Loading model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
print(f"Model loaded on: {model.device}")

# Encode all summaries
# normalize_embeddings=True is REQUIRED for BGE models — they use cosine similarity
print("Encoding companies...")
start = time.time()

summaries = all_companies['summary'].fillna('').tolist()
embeddings = model.encode(
    summaries,
    batch_size=256,            # larger batch = faster on GPU
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # IMPORTANT: required for BGE models
)

elapsed = time.time() - start
print(f"Done! Took {elapsed/60:.1f} minutes")
print(f"Embeddings shape: {embeddings.shape}")  # should be (98716, 1024)

# Save embeddings immediately — don't lose them!
np.save('result/BM25-BGE/company_embeddings.npy', embeddings)
print("Embeddings saved to result/BM25-BGE/company_embeddings.npy!")


# ## 9 · Build FAISS Index & Run Embedding Search
# 
# ### FAISS (`IndexFlatIP` — Inner Product)
# Because BGE embeddings are **L2-normalised** (via `normalize_embeddings=True`), cosine similarity equals inner product.  
# We therefore use `IndexFlatIP` (Inner Product) rather than `IndexFlatL2` — this is the correct index for normalised vectors.
# 
# > **Why does this matter?** `IndexFlatL2` minimises Euclidean distance, while `IndexFlatIP` maximises inner product. For unit-normalised vectors these are equivalent in ranking order, but `IndexFlatIP` returns the actual cosine similarity score (between -1 and 1), which is easier to interpret.
# 
# ### Query pipeline
# ```
# query string ──► model.encode(normalize=True) ──► 1024-d unit vector ──► index.search(k=1000)
#                                                                                  │
#                                                               (scores, indices) ──► ranked list
# ```
# 
# Results for all 101 queries are saved to **`result/BM25-BGE/embedding_results.csv`** — same schema as `bm25_results_BGE.csv` for easy comparison.

# In[ ]:


# Load embeddings
embeddings = np.load('result/BM25-BGE/company_embeddings.npy').astype('float32')
print(f"Embeddings loaded: {embeddings.shape}")  # (98716, 1024)

# Build FAISS index — use IndexFlatIP for normalised (BGE) embeddings
print("Building FAISS index...")
dimension = embeddings.shape[1]  # 1024 for BGE-large
index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine similarity for unit vectors
index.add(embeddings)
print(f"FAISS index built! Total vectors: {index.ntotal}")

# Save index
faiss.write_index(index, 'result/BM25-BGE/company_faiss.index')
print("Index saved to result/BM25-BGE/company_faiss.index!")

# Run all 101 queries
print("\nRunning embedding retrieval for all 101 queries...")
all_embedding_results = []
start = time.time()

for item in data:
    query_id = item['query_id']
    query = item['query']

    # Encode query — must also normalise for cosine similarity
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype('float32')

    # Search top 1000
    scores, indices = index.search(query_embedding, 1000)

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        all_embedding_results.append({
            'query_id': query_id,
            'query': query,
            'rank': rank + 1,
            'score': float(score),  # cosine similarity (higher = more similar)
            'domain': all_companies.iloc[idx]['domain'],
            'name': all_companies.iloc[idx]['name'],
            'summary': all_companies.iloc[idx]['summary']
        })

elapsed = time.time() - start
print(f"Done! Took {elapsed:.1f} seconds")

# Save results
embedding_df = pd.DataFrame(all_embedding_results)
embedding_df.to_csv('result/BM25-BGE/embedding_results.csv', index=False)
print(f"Saved {len(embedding_df)} results to result/BM25-BGE/embedding_results.csv!")


# ## 10 · Evaluation — Recall@k & Precision@k
# 
# ### Pseudo-relevance labels
# A company is considered **relevant** for a query if it appears in the **production top-100** for that query.  
# This is the standard *pseudo-relevance* assumption when human-judged labels are unavailable.
# 
# ### Metrics
# 
# | Metric | Formula | Interpretation |
# |---|---|---|
# | **Recall@k** | \|Retrieved ∩ Relevant\| / \|Relevant\| | Fraction of relevant companies found in top-k |
# | **Precision@k** | \|Retrieved ∩ Relevant\| / k | Fraction of top-k that are relevant |

# In[ ]:


# Load all three result sets
production_df = pd.read_excel('dataset/production_results.xlsx')
bm25_df = pd.read_csv('result/BM25-BGE/bm25_results_BGE.csv')
embedding_df = pd.read_csv('result/BM25-BGE/embedding_results.csv')

# For evaluation we use production rankings as pseudo-relevance
# A company is "relevant" for a query if it appears in production top-100
def get_relevant_domains(query_id, top_k=100):
    relevant = production_df[
        (production_df['query_id'] == query_id) &
        (production_df['rank'] <= top_k)
    ]['domain'].tolist()
    return set(relevant)

# Compute Recall@k — how many relevant companies did we retrieve?
def recall_at_k(retrieved_domains, relevant_domains, k):
    retrieved_at_k = set(retrieved_domains[:k])
    if len(relevant_domains) == 0:
        return 0
    return len(retrieved_at_k & relevant_domains) / len(relevant_domains)

# Compute Precision@k
def precision_at_k(retrieved_domains, relevant_domains, k):
    retrieved_at_k = set(retrieved_domains[:k])
    if k == 0:
        return 0
    return len(retrieved_at_k & relevant_domains) / k

# Evaluate all queries
k_values = [10, 50, 100, 500, 1000]
results_summary = []

for item in data:
    query_id = item['query_id']
    query = item['query']

    # Get relevant domains from production top-100
    relevant = get_relevant_domains(query_id, top_k=100)

    # Get retrieved domains for each system
    bm25_retrieved = bm25_df[bm25_df['query_id'] == query_id].sort_values('rank')['domain'].tolist()
    emb_retrieved = embedding_df[embedding_df['query_id'] == query_id].sort_values('rank')['domain'].tolist()

    for k in k_values:
        results_summary.append({
            'query_id': query_id,
            'query': query,
            'k': k,
            'bm25_recall': recall_at_k(bm25_retrieved, relevant, k),
            'bm25_precision': precision_at_k(bm25_retrieved, relevant, k),
            'emb_recall': recall_at_k(emb_retrieved, relevant, k),
            'emb_precision': precision_at_k(emb_retrieved, relevant, k),
        })

eval_df = pd.DataFrame(results_summary)

# Print average results
print("=== AVERAGE RESULTS ACROSS 101 QUERIES ===\n")
for k in k_values:
    subset = eval_df[eval_df['k'] == k]
    print(f"--- @{k} ---")
    print(f"BM25     Recall: {subset['bm25_recall'].mean():.3f}  Precision: {subset['bm25_precision'].mean():.3f}")
    print(f"BGE-large  Recall: {subset['emb_recall'].mean():.3f}  Precision: {subset['emb_precision'].mean():.3f}")
    print()

# Save ONCE — outside the loop
eval_df.to_csv('result/BM25-BGE/evaluation_results.csv', index=False)
print("Saved to result/BM25-BGE/evaluation_results.csv!")


# ### Results summary (BGE-large vs BM25)
# *(Update this table after running the cell above)*
# 
# | k | BM25 Recall | BGE Recall | BM25 Prec | BGE Prec |
# |---|---|---|---|---|
# | 10 | — | — | — | — |
# | 50 | — | — | — | — |
# | 100 | — | — | — | — |
# | 500 | — | — | — | — |
# | 1000 | — | — | — | — |
# 
# **Observation:** Fill in after running the evaluation.

# ## 11 · Precision Comparison (Head-to-Head)
# 
# A focused look at the precision gap between BM25 and BGE-large.
# 
# BGE-large is a much stronger model than MiniLM (335M vs 22M parameters, 1024-d vs 384-d embeddings).  
# We expect to see a larger improvement over BM25 at low k compared to MiniLM results.
# 
# This trade-off motivates a **hybrid retrieval** approach (e.g., RRF) as a next step.

# In[ ]:


# Focus on precision comparison
print("=== PRECISION COMPARISON: BM25 vs BGE-large ===\n")
for k in k_values:
    subset = eval_df[eval_df['k'] == k]
    diff = subset['emb_precision'].mean() - subset['bm25_precision'].mean()
    print(f"@{k}: BM25={subset['bm25_precision'].mean():.3f} "
          f"BGE-large={subset['emb_precision'].mean():.3f} "
          f"Diff={diff:+.3f}")


# ## 12 · NDCG Evaluation
# 
# ### What is NDCG?
# **Normalised Discounted Cumulative Gain** measures ranking quality, not just set overlap.  
# Results ranked higher contribute more to the score (logarithmic discount).
# 
# ```
# DCG@k  = Σ rel_i / log2(i + 2)        for i = 0..k-1
# NDCG@k = DCG@k / IDCG@k               (IDCG = ideal / perfect ranking)
# ```
# 
# Here relevance is binary: 1 if the domain is in the production top-100, 0 otherwise.
# 
# ### Results summary (BGE-large vs BM25)
# *(Update this table after running the cell below)*
# 
# | k | BM25 NDCG | BGE-large NDCG | Diff |
# |---|---|---|---|
# | 10 | — | — | — |
# | 50 | — | — | — |
# | 100 | — | — | — |
# | 500 | — | — | — |
# | 1000 | — | — | — |
# 
# **Conclusion:** Fill in after running the evaluation.

# In[ ]:


def dcg_at_k(relevant_domains, retrieved_domains, k):
    """Compute DCG@k"""
    score = 0
    for i, domain in enumerate(retrieved_domains[:k]):
        if domain in relevant_domains:
            score += 1 / np.log2(i + 2)  # +2 because log2(1) = 0
    return score

def ndcg_at_k(relevant_domains, retrieved_domains, k):
    """Compute NDCG@k"""
    # Actual DCG
    actual_dcg = dcg_at_k(relevant_domains, retrieved_domains, k)

    # Ideal DCG — best possible ranking
    ideal_retrieved = list(relevant_domains)[:k]
    ideal_dcg = dcg_at_k(relevant_domains, ideal_retrieved, k)

    if ideal_dcg == 0:
        return 0
    return actual_dcg / ideal_dcg

# Evaluate all queries
k_values = [10, 50, 100, 500, 1000]
ndcg_results = []

print("Computing NDCG for all queries...")
for item in data:
    query_id = item['query_id']
    query = item['query']

    # Get relevant domains from production top-100
    relevant = get_relevant_domains(query_id, top_k=100)

    # Get retrieved domains for each system
    bm25_retrieved = bm25_df[
        bm25_df['query_id'] == query_id
    ].sort_values('rank')['domain'].tolist()

    emb_retrieved = embedding_df[
        embedding_df['query_id'] == query_id
    ].sort_values('rank')['domain'].tolist()

    for k in k_values:
        ndcg_results.append({
            'query_id': query_id,
            'query': query,
            'k': k,
            'bm25_ndcg': ndcg_at_k(relevant, bm25_retrieved, k),
            'emb_ndcg': ndcg_at_k(relevant, emb_retrieved, k),
        })

ndcg_df = pd.DataFrame(ndcg_results)

# Print results
print("\n=== NDCG RESULTS ACROSS 101 QUERIES ===\n")
for k in k_values:
    subset = ndcg_df[ndcg_df['k'] == k]
    bm25_mean = subset['bm25_ndcg'].mean()
    emb_mean = subset['emb_ndcg'].mean()
    diff = emb_mean - bm25_mean
    print(f"@{k}: BM25={bm25_mean:.3f}  Embed={emb_mean:.3f}  Diff={diff:+.3f}")

# Save
ndcg_df.to_csv('result/BM25-BGE/ndcg_results.csv', index=False)
print("\nSaved to result/BM25-BGE/ndcg_results.csv!")


# In[ ]:


# Evaluate NDCG for all queries
k_values = [10, 50, 100, 500, 1000]
ndcg_results = []

print("Computing NDCG for all queries...")
for item in data:
    query_id = item['query_id']
    query = item['query']

    relevant = get_relevant_domains(query_id, top_k=100)

    bm25_retrieved = bm25_df[
        bm25_df['query_id'] == query_id
    ].sort_values('rank')['domain'].tolist()

    emb_retrieved = embedding_df[
        embedding_df['query_id'] == query_id
    ].sort_values('rank')['domain'].tolist()

    for k in k_values:
        ndcg_results.append({
            'query_id': query_id,
            'query': query,
            'k': k,
            'bm25_ndcg': ndcg_at_k(relevant, bm25_retrieved, k),
            'emb_ndcg': ndcg_at_k(relevant, emb_retrieved, k),
        })

ndcg_df = pd.DataFrame(ndcg_results)

# Print results
print("\n=== NDCG RESULTS ACROSS 101 QUERIES ===\n")
for k in k_values:
    subset = ndcg_df[ndcg_df['k'] == k]
    bm25_mean = subset['bm25_ndcg'].mean()
    emb_mean = subset['emb_ndcg'].mean()
    diff = emb_mean - bm25_mean
    print(f"@{k}: BM25={bm25_mean:.3f}  BGE-large={emb_mean:.3f}  Diff={diff:+.3f}")

# Save
ndcg_df.to_csv('result/BM25-BGE/ndcg_results.csv', index=False)
print("\nSaved to result/BM25-BGE/ndcg_results.csv!")

