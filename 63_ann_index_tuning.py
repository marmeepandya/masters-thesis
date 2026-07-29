#!/usr/bin/env python
# coding: utf-8

# # 63 - Approximate Nearest Neighbor (ANN) Index Construction and Tuning
# 
# The thesis proposal explicitly commits to approximate nearest neighbor search ("supports scalable top-$k$ retrieval using approximate nearest neighbor search", "trade-off between retrieval accuracy and latency under different ANN index configurations"). Every retrieval result so far, including the scaling evaluation in notebook 60, uses exact brute-force search (a full dot product against every company in the tier), not ANN at all. This notebook closes that gap: builds a FAISS HNSW index over GTE-large's embeddings (the strongest all-around performer from notebook 60: best NDCG, fastest exact search) at each of the four corpus tiers, and characterizes the recall-vs-latency trade-off as the index's hyperparameters change, compared directly against notebook 60's already-computed exact-search numbers for the same model, tiers, and queries.
# 
# HNSW's hyperparameters are index-structure settings, not model-training settings: `M` (graph connections per node) and `ef_construction` (build-time search width) are fixed when the index is built; `ef_search` (query-time search width) can be swept cheaply against an already-built index without rebuilding. Unlike notebook 61's single best-combination search, there is no one "best" ANN setting here, it is inherently a trade-off between recall and speed, so the natural output is a curve, not a winner.

# In[ ]:


import os
import json
import time
import numpy as np
import pandas as pd
import faiss
from pathlib import Path

OUTPUT_DIR = Path("result/63_ann_index_tuning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIERS = [("100k", "in_100k"), ("200k", "in_200k"), ("300k", "in_300k"), ("400k", "in_400k")]
DEEP_QUERY_IDS = [1, 2, 3, 4, 5, 11, 12, 15, 34, 14, 27, 66, 72, 99, 101, 56, 82, 91, 92]
K_VALUES = [10, 50, 100, 300, 500, 1000]
RELEVANT_THRESHOLD = 2
TOP_K = 1000

# (M, ef_construction) index-build combinations to try; ef_search is swept separately per built index
BUILD_COMBOS = [(16, 40), (32, 200)]
EF_SEARCH_VALUES = [10, 50, 100, 200, 500, 1000]

combined = pd.read_parquet("result/44_build_scaled_corpus/combined_pool.parquet")
gte_embeddings = np.load("result/46_encode_gte_scaled/company_embeddings.npy", mmap_mode="r")
print(f"[Load] Combined pool: {len(combined):,} companies, GTE embeddings shape: {gte_embeddings.shape}")

train_queries = json.load(open("result/08_llm_relevance_judge/train_queries.json"))
held_out_queries = json.load(open("result/08_llm_relevance_judge/held_out_queries.json"))
query_lookup = {item["query_id"]: item["query"] for item in train_queries + held_out_queries}

base_gold = pd.read_json("result/40_active_learning_labeling_queue/expanded_gold_labels.json")
round_gold = pd.read_json("result/42_headline_query_deepening/round_gold_labels.json")
gold = pd.concat([base_gold[["query_id", "domain", "gold_label"]], round_gold[["query_id", "domain", "gold_label"]]], ignore_index=True).drop_duplicates(subset=["query_id", "domain"])
relevant_sets = {
    qid: set(g.loc[g["gold_label"] >= RELEVANT_THRESHOLD, "domain"])
    for qid, g in gold.groupby("query_id")
}


# In[ ]:


from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_OFFLINE"] = "1"
try:
    gte_model = SentenceTransformer("thenlper/gte-large", local_files_only=True)
except Exception:
    gte_model = SentenceTransformer("thenlper/gte-large")

deep_queries = [(qid, query_lookup[qid]) for qid in DEEP_QUERY_IDS]
query_texts = [qtext for _, qtext in deep_queries]
query_embs = gte_model.encode(query_texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
print(f"[Encode] Query embeddings: {query_embs.shape}")

del gte_model
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# In[ ]:


def precision_at_k(retrieved, relevant, k):
    if k == 0 or not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return None
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def dcg_at_k(retrieved, relevant, k):
    return sum(1 / np.log2(i + 2) for i, d in enumerate(retrieved[:k]) if d in relevant)


def ndcg_at_k(retrieved, relevant, k):
    if not relevant:
        return None
    ideal = dcg_at_k(list(relevant), relevant, k)
    return dcg_at_k(retrieved, relevant, k) / ideal if ideal else 0.0


# In[ ]:


all_rows = []

for tier_name, flag_col in TIERS:
    mask = combined[flag_col].values
    tier_domains = combined.loc[mask, "domain"].reset_index(drop=True)
    tier_embs = np.ascontiguousarray(gte_embeddings[mask].astype("float32"))
    dim = tier_embs.shape[1]

    for M, ef_construction in BUILD_COMBOS:
        combo_path = OUTPUT_DIR / f"results_{tier_name}_M{M}_efc{ef_construction}.json"
        if combo_path.exists():
            print(f"[Tier {tier_name}, M={M}, ef_construction={ef_construction}] Already done -- loading from disk")
            all_rows.extend(json.load(open(combo_path)))
            continue

        print(f"[Tier {tier_name}] Building HNSW index over {tier_embs.shape[0]:,} companies (M={M}, ef_construction={ef_construction})...")
        t0 = time.time()
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.add(tier_embs)
        build_time_s = time.time() - t0
        print(f"[Tier {tier_name}] Index built in {build_time_s:.1f}s")

        combo_rows = []
        for ef_search in EF_SEARCH_VALUES:
            index.hnsw.efSearch = ef_search
            for qid, q_emb in zip(DEEP_QUERY_IDS, query_embs):
                t0 = time.perf_counter()
                sims, idxs = index.search(q_emb.reshape(1, -1), TOP_K)
                query_ms = (time.perf_counter() - t0) * 1000
                retrieved = [tier_domains.iloc[i] for i in idxs[0] if i != -1]
                relevant = relevant_sets.get(qid, set())
                for k in K_VALUES:
                    combo_rows.append({
                        "tier": tier_name, "M": M, "ef_construction": ef_construction, "ef_search": ef_search,
                        "query_id": qid, "k": k, "query_latency_ms": query_ms, "index_build_s": build_time_s,
                        "precision": precision_at_k(retrieved, relevant, k),
                        "recall": recall_at_k(retrieved, relevant, k),
                        "ndcg": ndcg_at_k(retrieved, relevant, k),
                    })
            print(f"  ef_search={ef_search}: avg latency {np.mean([r['query_latency_ms'] for r in combo_rows if r['ef_search']==ef_search and r['k']==10]):.2f}ms")

        json.dump(combo_rows, open(combo_path, "w"))
        all_rows.extend(combo_rows)

ann_results = pd.DataFrame(all_rows)
ann_results.to_csv(OUTPUT_DIR / "ann_results_all.csv", index=False)
print(f"[Done] {len(ann_results):,} rows")


# In[ ]:


ann_summary = ann_results.groupby(["tier", "M", "ef_construction", "ef_search", "k"])[["recall", "ndcg", "query_latency_ms"]].mean().reset_index()
ann_summary.to_csv(OUTPUT_DIR / "ann_summary.csv", index=False)

exact_gte = pd.read_csv("result/60_scaling_evaluation/scaling_eval_per_query.csv")
exact_gte = exact_gte[exact_gte["method"] == "gte"]
exact_summary = exact_gte.groupby(["tier", "k"])[["recall", "ndcg"]].mean().reset_index()
exact_timing = pd.read_json("result/60_scaling_evaluation/gte_timing.json") if Path("result/60_scaling_evaluation/gte_timing.json").exists() else None

print("=== Recall@1000 and NDCG@1000: exact search vs. ANN (best ef_search per tier/build-combo) ===")
k1000 = ann_summary[ann_summary["k"] == 1000]
for tier_name, _ in TIERS:
    exact_row = exact_summary[(exact_summary["tier"] == tier_name) & (exact_summary["k"] == 1000)]
    print(f"\nTier {tier_name} -- exact search: recall={exact_row['recall'].values[0]:.3f}  ndcg={exact_row['ndcg'].values[0]:.3f}")
    tier_ann = k1000[k1000["tier"] == tier_name].sort_values("recall", ascending=False)
    print(tier_ann[["M", "ef_construction", "ef_search", "recall", "ndcg", "query_latency_ms"]].to_string(index=False))

