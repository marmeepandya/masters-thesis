#!/usr/bin/env python
# coding: utf-8

# # 61 - Fusion Ranker Hyperparameter Tuning
# 
# Every reported fusion-ranker result so far (Sections~\ref{sec:learned_fusion}, \ref{subsec:refit_fusion_ranker}, and notebook 56) uses the same hand-picked `HistGradientBoostingClassifier(max_iter=150, max_depth=4, class_weight="balanced")` -- never systematically tuned. This notebook runs a randomized hyperparameter search over the same architecture.
# 
# Picking the best hyperparameters using the same cross-validated metric that then gets reported as the final result would be leakage (the "winner's curse" in model selection): a combination chosen because it scored best across all 101 queries' held-out folds is no longer a fair, unbiased estimate of how it performs on genuinely unseen data. To avoid this, the search is restricted to the 82 queries *without* deep gold coverage -- the search never sees the 19 deep-coverage queries (5-query pilot + 14 headline queries) at all. Once the best hyperparameters are chosen from that 82-query search alone, they are evaluated fresh on the full 101-query set, and the 19-deep-query subset of that final evaluation is a genuinely leakage-free number, since those queries never influenced which hyperparameters were picked. The all-101-query number is still reported for comparability with earlier sections, but read it as directionally informative rather than strictly unbiased, since 82 of those 101 queries did contribute to the search.

# In[ ]:


import time
import random
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut

OUTPUT_DIR = Path("result/61_fusion_ranker_hyperparameter_tuning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RELEVANT_THRESHOLD = 2
K_VALUES = [10, 50, 100, 300, 500, 1000]
DEEP_QUERY_IDS = [1, 2, 3, 4, 5, 11, 12, 15, 34, 14, 27, 66, 72, 99, 101, 56, 82, 91, 92]  # pilot + headline

feature_cols = ["score_minilm", "score_linq", "score_gte", "score_bm25",
                "invrank_minilm", "invrank_linq", "invrank_gte", "invrank_bm25",
                "reranker_score", "n_channels"]

base_gold = pd.read_json("result/40_active_learning_labeling_queue/expanded_gold_labels.json")
round_gold = pd.read_json("result/42_headline_query_deepening/round_gold_labels.json")
features = pd.read_csv("result/30_learned_fusion_ranker/scored_candidates.csv")

gold = pd.concat([
    base_gold[["query_id", "domain", "gold_label"]],
    round_gold[["query_id", "domain", "gold_label"]],
], ignore_index=True).drop_duplicates(subset=["query_id", "domain"])

labeled = gold.merge(features, on=["query_id", "domain"], how="inner")
labeled["relevant"] = (labeled["gold_label"] >= RELEVANT_THRESHOLD).astype(int)
print(f"Gold labels for cross-validation: {len(labeled)} across {labeled['query_id'].nunique()} queries")

relevant_sets = {
    qid: set(g.loc[g["gold_label"] >= RELEVANT_THRESHOLD, "domain"])
    for qid, g in labeled.groupby("query_id")
}

X = labeled[feature_cols].values
y = labeled["relevant"].values
groups = labeled["query_id"].values

ALL_QUERY_IDS = sorted(labeled["query_id"].unique().tolist())
TUNING_QUERY_IDS = sorted(set(ALL_QUERY_IDS) - set(DEEP_QUERY_IDS))
print(f"Held-out deep queries (never touched during search): {len(DEEP_QUERY_IDS)}")
print(f"Tuning-only queries (used for hyperparameter search): {len(TUNING_QUERY_IDS)}")


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


def cv_rank(params, pool_query_ids):
    """Leave-one-query-out CV restricted to pool_query_ids -- every fold's training data comes
    only from other queries within this same pool, so queries outside the pool never influence
    training or, when this is used inside the hyperparameter search, model selection either."""
    pool_mask = np.isin(groups, pool_query_ids)
    X_pool, y_pool, groups_pool = X[pool_mask], y[pool_mask], groups[pool_mask]
    features_pool = features[features["query_id"].isin(pool_query_ids)]
    logo = LeaveOneGroupOut()
    rankings = {}
    for train_idx, test_idx in logo.split(X_pool, y_pool, groups_pool):
        held_out_query = groups_pool[test_idx][0]
        clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=0, **params)
        clf.fit(X_pool[train_idx], y_pool[train_idx])
        query_candidates = features_pool[features_pool["query_id"] == held_out_query]
        scores = clf.predict_proba(query_candidates[feature_cols].values)[:, 1]
        ranked = query_candidates.assign(cv_score=scores).sort_values("cv_score", ascending=False)
        rankings[held_out_query] = ranked["domain"].head(1000).tolist()
    return rankings


def evaluate(rankings, query_ids, label):
    rows = []
    for qid in query_ids:
        retrieved = rankings.get(qid, [])
        relevant = relevant_sets.get(qid, set())
        for k in K_VALUES:
            rows.append({
                "system": label, "query_id": qid, "k": k,
                "precision": precision_at_k(retrieved, relevant, k),
                "recall": recall_at_k(retrieved, relevant, k),
                "ndcg": ndcg_at_k(retrieved, relevant, k),
            })
    return pd.DataFrame(rows)


def mean_at_k(eval_df, k, col):
    return eval_df[eval_df["k"] == k][col].mean()


# In[ ]:


baseline_path = OUTPUT_DIR / "baseline_eval.csv"
if baseline_path.exists():
    print("[Baseline] Already evaluated -- loading from disk")
    baseline_eval = pd.read_csv(baseline_path)
else:
    print("[Baseline] Running the current, untuned hyperparameters (max_iter=150, max_depth=4) on all 101 queries...")
    t0 = time.time()
    baseline_rankings = cv_rank({"max_iter": 150, "max_depth": 4}, ALL_QUERY_IDS)
    baseline_eval = evaluate(baseline_rankings, list(baseline_rankings.keys()), "baseline (untuned)")
    baseline_eval.to_csv(baseline_path, index=False)
    print(f"[Baseline] Done in {time.time()-t0:.1f}s")

print(f"[Baseline] All-101-queries  Recall@1000={mean_at_k(baseline_eval, 1000, 'recall'):.3f}  NDCG@1000={mean_at_k(baseline_eval, 1000, 'ndcg'):.3f}")
deep_mask = baseline_eval["query_id"].isin(DEEP_QUERY_IDS)
print(f"[Baseline] 19-deep-queries  Recall@1000={mean_at_k(baseline_eval[deep_mask], 1000, 'recall'):.3f}  NDCG@1000={mean_at_k(baseline_eval[deep_mask], 1000, 'ndcg'):.3f}")


# In[ ]:


SEARCH_SPACE = {
    "max_depth": [2, 3, 4, 5, 6, 8, None],
    "max_iter": [50, 100, 150, 200, 300],
    "learning_rate": [0.03, 0.05, 0.1, 0.15, 0.2, 0.3],
    "min_samples_leaf": [5, 10, 20, 30],
    "l2_regularization": [0.0, 0.1, 0.5, 1.0, 2.0],
}
N_SEARCH = 40
SEARCH_SEED = 42

leaderboard_path = OUTPUT_DIR / "search_leaderboard.json"
leaderboard = json.load(open(leaderboard_path)) if leaderboard_path.exists() else []
done_combo_ids = {row["combo_id"] for row in leaderboard}

rng = random.Random(SEARCH_SEED)
sampled_combos = []
for combo_id in range(N_SEARCH):
    params = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
    sampled_combos.append((combo_id, params))

print(f"[Search] Scoring every combination on the {len(TUNING_QUERY_IDS)} tuning-only queries -- the 19 deep queries are never touched here")
print(f"[Search] {len(done_combo_ids)}/{N_SEARCH} combinations already evaluated -- resuming")
t0 = time.time()
for combo_id, params in sampled_combos:
    if combo_id in done_combo_ids:
        continue
    rankings = cv_rank(params, TUNING_QUERY_IDS)
    combo_eval = evaluate(rankings, list(rankings.keys()), f"combo_{combo_id}")
    leaderboard.append({
        "combo_id": combo_id, **params,
        "recall_at_1000": mean_at_k(combo_eval, 1000, "recall"),
        "ndcg_at_1000": mean_at_k(combo_eval, 1000, "ndcg"),
        "ndcg_at_10": mean_at_k(combo_eval, 10, "ndcg"),
    })
    json.dump(leaderboard, open(leaderboard_path, "w"), indent=2)  # checkpoint after every combo
    print(f"  [{combo_id+1}/{N_SEARCH}] NDCG@1000={leaderboard[-1]['ndcg_at_1000']:.4f}  Recall@1000={leaderboard[-1]['recall_at_1000']:.4f}  params={params}")

print(f"[Search] Done in {(time.time()-t0)/60:.1f} min")


# In[ ]:


leaderboard_df = pd.DataFrame(leaderboard).sort_values("ndcg_at_1000", ascending=False).reset_index(drop=True)
leaderboard_df.to_csv(OUTPUT_DIR / "search_leaderboard.csv", index=False)
print("Top 10 hyperparameter combinations by NDCG@1000:")
print(leaderboard_df.head(10).to_string(index=False))

best_row = leaderboard_df.iloc[0]
best_params = {k: (None if pd.isna(best_row[k]) else best_row[k]) for k in SEARCH_SPACE}
for int_key in ["max_iter", "min_samples_leaf"]:
    if best_params[int_key] is not None:
        best_params[int_key] = int(best_params[int_key])
if best_params["max_depth"] is not None:
    best_params["max_depth"] = int(best_params["max_depth"])
print(f"\nBest hyperparameters: {best_params}")


# In[ ]:


tuned_path = OUTPUT_DIR / "tuned_eval.csv"
if tuned_path.exists():
    print("[Tuned] Already evaluated -- loading from disk")
    tuned_eval = pd.read_csv(tuned_path)
else:
    print(f"[Tuned] Running the best hyperparameters found on all 101 queries: {best_params}")
    t0 = time.time()
    tuned_rankings = cv_rank(best_params, ALL_QUERY_IDS)
    tuned_eval = evaluate(tuned_rankings, list(tuned_rankings.keys()), "tuned")
    tuned_eval.to_csv(tuned_path, index=False)
    print(f"[Tuned] Done in {time.time()-t0:.1f}s")

comparison_rows = []
for label, df in [("baseline (untuned)", baseline_eval), ("tuned", tuned_eval)]:
    for subset_name, mask in [("all_101_queries", df["query_id"].notna()), ("19_deep_queries", df["query_id"].isin(DEEP_QUERY_IDS))]:
        sub = df[mask]
        for k in K_VALUES:
            comparison_rows.append({
                "model": label, "query_subset": subset_name, "k": k,
                "precision": mean_at_k(sub, k, "precision"),
                "recall": mean_at_k(sub, k, "recall"),
                "ndcg": mean_at_k(sub, k, "ndcg"),
            })
comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUTPUT_DIR / "baseline_vs_tuned_comparison.csv", index=False)
print(comparison_df.to_string(index=False))
print()
print("Read \"tuned / 19_deep_queries\" as the genuinely leakage-free comparison -- those 19 queries")
print("never contributed to the hyperparameter search. \"tuned / all_101_queries\" is still useful for")
print("comparability with earlier sections, but 82 of those 101 queries did influence which")
print("hyperparameters were picked, so treat it as directional rather than strictly unbiased.")

