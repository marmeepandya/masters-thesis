#!/usr/bin/env python
# coding: utf-8

# # 69 - Per-System Label Coverage and bpref
# 
# Ralph's methodological feedback pointed out a specific bias: every precision/recall/NDCG figure in this thesis implicitly treats an unjudged candidate as irrelevant, which punishes a system exactly when it retrieves something good the pooling process happened to miss. Since different systems' top-k results carry very different fractions of already-judged candidates (a system that was itself one of the six pooling sources trivially has higher coverage than one that wasn't), their Recall/NDCG scores are not directly comparable without knowing that fraction.
# 
# This notebook computes, for six systems, label coverage at each k and bpref \citep{buckley2004retrieval}, a TREC-standard metric designed for exactly this incomplete-judgment setting: rather than treating an unjudged document as confirmed non-relevant, bpref only counts judged non-relevant documents when penalising a relevant document's rank position, so a system is never docked for retrieving something genuinely unjudged rather than genuinely irrelevant. This reuses the exact leave-one-query-out cross-validation setup from Notebook 56, so the numbers here are directly comparable to Section~\ref{subsec:goi_vs_ours_ranking}'s reported Recall@1000 figures.

# In[ ]:


import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut

OUTPUT_DIR = Path("result/69_label_coverage_bpref")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RELEVANT_THRESHOLD = 2
K_VALUES = [10, 50, 100, 300, 500, 1000]

feature_cols = ["score_minilm", "score_linq", "score_gte", "score_bm25",
                "invrank_minilm", "invrank_linq", "invrank_gte", "invrank_bm25",
                "reranker_score", "n_channels"]

base_gold = pd.read_json("result/40_active_learning_labeling_queue/expanded_gold_labels.json")
round_gold = pd.read_json("result/42_headline_query_deepening/round_gold_labels.json")
features = pd.read_csv("result/30_learned_fusion_ranker/scored_candidates.csv")
production = pd.read_excel("dataset/production_results.xlsx")[["query_id", "domain", "rank"]].rename(columns={"rank": "production_rank"})

gold = pd.concat([base_gold[["query_id", "domain", "gold_label"]], round_gold[["query_id", "domain", "gold_label"]]], ignore_index=True).drop_duplicates(subset=["query_id", "domain"])
labeled = gold.merge(features, on=["query_id", "domain"], how="inner")
labeled["relevant"] = (labeled["gold_label"] >= RELEVANT_THRESHOLD).astype(int)
print(f"Silver labels matched to features: {len(labeled)} across {labeled['query_id'].nunique()} queries")

judged_lookup = {qid: set(g["domain"]) for qid, g in gold.groupby("query_id")}
relevant_lookup = {qid: set(g.loc[g["gold_label"] >= RELEVANT_THRESHOLD, "domain"]) for qid, g in gold.groupby("query_id")}
nonrelevant_lookup = {qid: set(g.loc[g["gold_label"] < RELEVANT_THRESHOLD, "domain"]) for qid, g in gold.groupby("query_id")}


# In[ ]:


# System 1: production's own ranking.
goi_rankings = {qid: g.sort_values("production_rank")["domain"].tolist() for qid, g in production.groupby("query_id")}

# System 2: leave-one-query-out cross-validated fusion ranker, identical setup to Notebook 56.
X, y, groups = labeled[feature_cols].values, labeled["relevant"].values, labeled["query_id"].values
logo = LeaveOneGroupOut()
our_rankings = {}
t0 = time.time()
for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
    held_out_query = groups[test_idx][0]
    clf = HistGradientBoostingClassifier(max_iter=150, max_depth=4, class_weight="balanced", random_state=0)
    clf.fit(X[train_idx], y[train_idx])
    query_candidates = features[features["query_id"] == held_out_query]
    scores = clf.predict_proba(query_candidates[feature_cols].values)[:, 1]
    ranked = query_candidates.assign(_score=scores).sort_values("_score", ascending=False)
    our_rankings[held_out_query] = ranked["domain"].tolist()[:1000]
print(f"LOGO CV done in {time.time()-t0:.1f}s")

# Systems 3-6: each individual retrieval channel's own top-1000 ranking within the same candidate pool,
# using that channel's raw score column. A channel that never retrieved a given candidate scores it
# near-zero, so sorting descending naturally recovers that channel's own preferences.
channel_rankings = {}
for ch in ["bm25", "minilm", "gte", "linq"]:
    col = f"score_{ch}"
    channel_rankings[ch] = {qid: g.sort_values(col, ascending=False)["domain"].tolist()[:1000] for qid, g in features.groupby("query_id")}

systems = {"production": goi_rankings, "ours_logo_cv": our_rankings, **{f"channel_{c}": r for c, r in channel_rankings.items()}}
print(f"Systems compared: {list(systems.keys())}")


# In[ ]:


def bpref(retrieved, rel, nonrel):
    # Buckley and Voorhees (2004): only judged non-relevant docs count against a relevant doc's rank position.
    R, N = len(rel), len(nonrel)
    if R == 0:
        return None
    denom = min(R, N) if min(R, N) > 0 else 1
    total, n_seen = 0.0, 0
    for d in retrieved:
        if d in nonrel:
            n_seen += 1
        elif d in rel:
            total += 1 - min(n_seen, R) / denom
    return total / R


def recall_at_k(retrieved, rel, k):
    if not rel:
        return None
    return len(set(retrieved[:k]) & rel) / len(rel)


rows = []
for sysname, rankings in systems.items():
    for qid, ranked in rankings.items():
        if qid not in judged_lookup:
            continue
        judged = judged_lookup[qid]
        rel = relevant_lookup.get(qid, set())
        nonrel = nonrelevant_lookup.get(qid, set())
        row = {"system": sysname, "query_id": qid,
               "recall_at_1000": recall_at_k(ranked, rel, 1000),
               "bpref": bpref(ranked, rel, nonrel)}
        for k in K_VALUES:
            topk = ranked[:k]
            row[f"coverage_at_{k}"] = len(set(topk) & judged) / len(topk) if topk else 0
        rows.append(row)

per_query = pd.DataFrame(rows)
per_query.to_csv(OUTPUT_DIR / "per_query_coverage_bpref.csv", index=False)

coverage_cols = [f"coverage_at_{k}" for k in K_VALUES]
summary = per_query.groupby("system")[coverage_cols + ["recall_at_1000", "bpref"]].mean().round(3)
summary.to_csv(OUTPUT_DIR / "summary_coverage_bpref.csv")

print("=== Ranked by standard Recall@1000 ===")
print(summary[["coverage_at_10", "recall_at_1000", "bpref"]].sort_values("recall_at_1000", ascending=False))
print()
print("=== Ranked by bpref ===")
print(summary[["coverage_at_10", "recall_at_1000", "bpref"]].sort_values("bpref", ascending=False))
print()
print("Full coverage-by-k table:")
print(summary[coverage_cols].sort_values("coverage_at_10", ascending=False))

