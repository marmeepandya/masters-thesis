#!/usr/bin/env python
# coding: utf-8

# # 64 - A Neural Network Fusion Ranker
# 
# Every fusion ranker result so far uses `HistGradientBoostingClassifier` (Section~\ref{sec:learned_fusion}, tuned in Section~\ref{subsec:hyperparameter_tuning}). This notebook tries a genuinely different model class on the same ten features: a small multi-layer perceptron (MLP), with its own hyperparameter search over architecture (hidden layer sizes/depth), optimization (learning rate), and regularization (dropout, weight decay), and asks whether a neural network does any better than the tuned tree ensemble.
# 
# Full `LeaveOneGroupOut` cross-validation (as used for the GBDT) is too expensive to repeat per hyperparameter combination for a neural network, each fold requires actually training a network from scratch, not a near-instant tree fit, so the search phase uses a single, fixed 80/20 split of the 82 tuning-only queries (never touching the 19 held-out deep queries) instead of full cross-validation, a standard, well-justified simplification for neural hyperparameter search. Once the best combination is chosen, it is evaluated with the same full `LeaveOneGroupOut` protocol over all 101 queries used everywhere else in this thesis, so the final comparison against the tuned GBDT (Section~\ref{subsec:hyperparameter_tuning}) on the 19 leakage-free deep queries is directly comparable.

# In[ ]:


import time
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path("result/64_neural_fusion_ranker")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GBDT_DIR = Path("result/61_fusion_ranker_hyperparameter_tuning")

RELEVANT_THRESHOLD = 2
K_VALUES = [10, 50, 100, 300, 500, 1000]
DEEP_QUERY_IDS = [1, 2, 3, 4, 5, 11, 12, 15, 34, 14, 27, 66, 72, 99, 101, 56, 82, 91, 92]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Setup] Device: {DEVICE}")

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

X_all = labeled[feature_cols].values.astype("float32")
y_all = labeled["relevant"].values.astype("float32")
groups_all = labeled["query_id"].values

ALL_QUERY_IDS = sorted(labeled["query_id"].unique().tolist())
TUNING_QUERY_IDS = sorted(set(ALL_QUERY_IDS) - set(DEEP_QUERY_IDS))
print(f"Held-out deep queries (never touched during search): {len(DEEP_QUERY_IDS)}")
print(f"Tuning-only queries (used for search): {len(TUNING_QUERY_IDS)}")

rng = random.Random(42)
shuffled_tuning = TUNING_QUERY_IDS.copy()
rng.shuffle(shuffled_tuning)
n_search_val = max(1, round(len(shuffled_tuning) * 0.2))
SEARCH_VAL_QUERY_IDS = sorted(shuffled_tuning[:n_search_val])
SEARCH_TRAIN_QUERY_IDS = sorted(shuffled_tuning[n_search_val:])
print(f"Search train/val split: {len(SEARCH_TRAIN_QUERY_IDS)} train queries, {len(SEARCH_VAL_QUERY_IDS)} validation queries")


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


class FusionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, params, n_epochs=80, batch_size=64, seed=0):
    """Standardizes features (fit on train only -- an MLP, unlike the tree-based GBDT, is scale-sensitive),
    trains with a class-imbalance-weighted BCE loss (mirroring the class_weight="balanced" convention used
    for every other classifier in this thesis), returns the trained model and the fitted scaler."""
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train).astype("float32")

    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, len(y_train) - n_pos)
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=DEVICE)

    model = FusionMLP(X_train.shape[1], params["hidden_dims"], params["dropout"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_t = torch.tensor(X_scaled, device=DEVICE)
    y_t = torch.tensor(y_train, device=DEVICE)
    n = len(X_t)
    rng_np = np.random.RandomState(seed)

    model.train()
    for epoch in range(n_epochs):
        order = rng_np.permutation(n)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            optimizer.zero_grad()
            logits = model(X_t[idx])
            loss = loss_fn(logits, y_t[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    return model, scaler


@torch.no_grad()
def score_mlp(model, scaler, X):
    X_scaled = scaler.transform(X).astype("float32")
    logits = model(torch.tensor(X_scaled, device=DEVICE))
    return torch.sigmoid(logits).cpu().numpy()


# In[ ]:


SEARCH_SPACE = {
    "hidden_dims": [(16,), (32,), (64,), (32, 16), (64, 32), (128, 64, 32)],
    "dropout": [0.0, 0.1, 0.3, 0.5],
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
    "weight_decay": [0.0, 1e-4, 1e-3, 1e-2],
}
N_SEARCH = 20
SEARCH_SEED = 42

train_mask = np.isin(groups_all, SEARCH_TRAIN_QUERY_IDS)
X_search_train, y_search_train = X_all[train_mask], y_all[train_mask]

leaderboard_path = OUTPUT_DIR / "search_leaderboard.json"
leaderboard = json.load(open(leaderboard_path)) if leaderboard_path.exists() else []
done_combo_ids = {row["combo_id"] for row in leaderboard}

rng = random.Random(SEARCH_SEED)
sampled_combos = [(i, {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}) for i in range(N_SEARCH)]

print(f"[Search] {len(done_combo_ids)}/{N_SEARCH} combinations already evaluated -- resuming")
t0 = time.time()
for combo_id, params in sampled_combos:
    if combo_id in done_combo_ids:
        continue
    model, scaler = train_mlp(X_search_train, y_search_train, params, seed=combo_id)

    val_rankings = {}
    for qid in SEARCH_VAL_QUERY_IDS:
        query_candidates = features[features["query_id"] == qid]
        scores = score_mlp(model, scaler, query_candidates[feature_cols].values.astype("float32"))
        ranked = query_candidates.assign(score=scores).sort_values("score", ascending=False)
        val_rankings[qid] = ranked["domain"].head(1000).tolist()
    val_eval = evaluate(val_rankings, SEARCH_VAL_QUERY_IDS, f"combo_{combo_id}")

    leaderboard.append({
        "combo_id": combo_id,
        "hidden_dims": list(params["hidden_dims"]), "dropout": params["dropout"],
        "learning_rate": params["learning_rate"], "weight_decay": params["weight_decay"],
        "recall_at_1000": mean_at_k(val_eval, 1000, "recall"),
        "ndcg_at_1000": mean_at_k(val_eval, 1000, "ndcg"),
    })
    json.dump(leaderboard, open(leaderboard_path, "w"), indent=2)
    print(f"  [{combo_id+1}/{N_SEARCH}] NDCG@1000={leaderboard[-1]['ndcg_at_1000']:.4f}  Recall@1000={leaderboard[-1]['recall_at_1000']:.4f}  params={params}")

print(f"[Search] Done in {(time.time()-t0)/60:.1f} min")


# In[ ]:


leaderboard_df = pd.DataFrame(leaderboard).sort_values("ndcg_at_1000", ascending=False).reset_index(drop=True)
leaderboard_df.to_csv(OUTPUT_DIR / "search_leaderboard.csv", index=False)
print("Top 10 hyperparameter combinations by NDCG@1000 (on the search-validation queries):")
print(leaderboard_df.head(10).to_string(index=False))

best_row = leaderboard_df.iloc[0]
best_params = {
    "hidden_dims": tuple(best_row["hidden_dims"]),
    "dropout": float(best_row["dropout"]),
    "learning_rate": float(best_row["learning_rate"]),
    "weight_decay": float(best_row["weight_decay"]),
}
print(f"\nBest hyperparameters: {best_params}")


# In[ ]:


final_eval_path = OUTPUT_DIR / "final_eval.csv"
if final_eval_path.exists():
    print("[Final] Already evaluated -- loading from disk")
    final_eval = pd.read_csv(final_eval_path)
else:
    print(f"[Final] Running full LeaveOneGroupOut CV over all 101 queries with: {best_params}")
    t0 = time.time()
    rankings = {}
    for fold_i, held_out_query in enumerate(ALL_QUERY_IDS):
        train_mask_fold = groups_all != held_out_query
        model, scaler = train_mlp(X_all[train_mask_fold], y_all[train_mask_fold], best_params, seed=fold_i)
        query_candidates = features[features["query_id"] == held_out_query]
        scores = score_mlp(model, scaler, query_candidates[feature_cols].values.astype("float32"))
        ranked = query_candidates.assign(score=scores).sort_values("score", ascending=False)
        rankings[held_out_query] = ranked["domain"].head(1000).tolist()
        if (fold_i + 1) % 25 == 0 or (fold_i + 1) == len(ALL_QUERY_IDS):
            print(f"  {fold_i+1}/{len(ALL_QUERY_IDS)} queries cross-validated ({(time.time()-t0)/60:.1f} min elapsed)")
    final_eval = evaluate(rankings, ALL_QUERY_IDS, "neural_net")
    final_eval.to_csv(final_eval_path, index=False)
    print(f"[Final] Done in {(time.time()-t0)/60:.1f} min")


# In[ ]:


gbdt_baseline = pd.read_csv(GBDT_DIR / "baseline_eval.csv")
gbdt_baseline["system"] = "gbdt_baseline_untuned"
gbdt_tuned = pd.read_csv(GBDT_DIR / "tuned_eval.csv")
gbdt_tuned["system"] = "gbdt_tuned"

comparison = pd.concat([gbdt_baseline, gbdt_tuned, final_eval], ignore_index=True)

rows = []
for label, df in [("gbdt_baseline_untuned", gbdt_baseline), ("gbdt_tuned", gbdt_tuned), ("neural_net", final_eval)]:
    for subset_name, mask in [("all_101_queries", df["query_id"].notna()), ("19_deep_queries", df["query_id"].isin(DEEP_QUERY_IDS))]:
        sub = df[mask]
        for k in K_VALUES:
            rows.append({
                "model": label, "query_subset": subset_name, "k": k,
                "precision": mean_at_k(sub, k, "precision"),
                "recall": mean_at_k(sub, k, "recall"),
                "ndcg": mean_at_k(sub, k, "ndcg"),
            })
final_comparison = pd.DataFrame(rows)
final_comparison.to_csv(OUTPUT_DIR / "three_way_comparison.csv", index=False)
print(final_comparison.to_string(index=False))
print()
print("19_deep_queries is the leakage-free comparison for the neural net (those queries never")
print("contributed to the architecture/hyperparameter search); all_101_queries reported for")
print("comparability but the GBDT-tuned row there is not leakage-free either (Section on GBDT tuning).")

