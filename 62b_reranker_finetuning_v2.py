#!/usr/bin/env python
# coding: utf-8

# # 62b - Reranker Fine-Tuning, Attempt 2: Hard Negatives + Pairwise Ranking Loss
# 
# Notebook 62's fine-tuning attempt made the reranker worse at every $k$ value tested. The most plausible cause, diagnosed there, was severe class imbalance in the training data (only 2.3% of examples labelled "not relevant"), likely collapsing the model toward scoring most candidates as relevant. This second attempt fixes that directly rather than abandoning fine-tuning: (1) the negative side of training is supplemented with candidates sampled from each query's own retrieval pool that were never gold-labelled at all (a standard, well-justified assumption given how sparse true relevance is in a raw candidate pool), and (2) training switches from MSE regression on an absolute score to a pairwise margin ranking loss, which directly optimizes for what is actually being measured, whether a relevant candidate outscores a non-relevant one for the same query, and is inherently more robust to label imbalance than absolute score regression. Same 82-tuning/19-held-out query split as notebooks 61 and 62, so the held-out comparison stays leakage-free and directly comparable to both prior experiments.

# In[ ]:


import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

OUTPUT_DIR = Path("result/62b_reranker_finetuning_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
V1_DIR = Path("result/62_reranker_finetuning")

RELEVANT_THRESHOLD = 2
K_VALUES = [10, 50, 100, 300, 500, 1000]
DEEP_QUERY_IDS = [1, 2, 3, 4, 5, 11, 12, 15, 34, 14, 27, 66, 72, 99, 101, 56, 82, 91, 92]  # pilot + headline, held out entirely
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Setup] Device: {DEVICE}")

base_gold = pd.read_json("result/40_active_learning_labeling_queue/expanded_gold_labels.json")
round_gold = pd.read_json("result/42_headline_query_deepening/round_gold_labels.json")

train_queries = json.load(open("result/08_llm_relevance_judge/train_queries.json"))
held_out_queries = json.load(open("result/08_llm_relevance_judge/held_out_queries.json"))
query_lookup = {item["query_id"]: item["query"] for item in train_queries + held_out_queries}

corpus = pd.read_parquet("result/44_build_scaled_corpus/combined_pool.parquet")[["domain", "summary"]]

round_gold = round_gold.copy()
round_gold["query"] = round_gold["query_id"].map(query_lookup)
round_gold = round_gold.merge(corpus, on="domain", how="left")

gold = pd.concat([
    base_gold[["query_id", "domain", "gold_label", "query", "summary"]],
    round_gold[["query_id", "domain", "gold_label", "query", "summary"]],
], ignore_index=True).drop_duplicates(subset=["query_id", "domain"])
gold["summary"] = gold["summary"].fillna("")

ALL_QUERY_IDS = sorted(gold["query_id"].unique().tolist())
TUNING_QUERY_IDS = sorted(set(ALL_QUERY_IDS) - set(DEEP_QUERY_IDS))
print(f"Held-out deep queries (never touched during fine-tuning): {len(DEEP_QUERY_IDS)}")
print(f"Tuning-only queries (used for fine-tuning): {len(TUNING_QUERY_IDS)}")

features = pd.read_csv("result/30_learned_fusion_ranker/scored_candidates.csv")[["query_id", "domain"]]
features = features.merge(corpus, on="domain", how="left")
features["summary"] = features["summary"].fillna("")

relevant_sets = {
    qid: set(g.loc[g["gold_label"] >= RELEVANT_THRESHOLD, "domain"])
    for qid, g in gold.groupby("query_id")
}


# In[ ]:


NEG_SAMPLES_PER_QUERY = 15  # presumed-negative candidates sampled per tuning query, supplementing the known (gold-labelled but not highly relevant) negatives
NEGATIVES_PER_POSITIVE = 3  # how many negatives each positive example is paired against
PAIR_SEED = 42

train_gold = gold[gold["query_id"].isin(TUNING_QUERY_IDS)]
positives_df = train_gold[train_gold["gold_label"] >= RELEVANT_THRESHOLD]
known_negatives_df = train_gold[train_gold["gold_label"] < RELEVANT_THRESHOLD]

pair_rows = []
seed_counter = 0
for qid in TUNING_QUERY_IDS:
    q_positives = positives_df[positives_df["query_id"] == qid]
    if len(q_positives) == 0:
        continue
    q_known_neg = known_negatives_df[known_negatives_df["query_id"] == qid][["domain", "summary"]]

    gold_domains_this_query = set(train_gold[train_gold["query_id"] == qid]["domain"])
    pool = features[(features["query_id"] == qid) & (~features["domain"].isin(gold_domains_this_query))]
    n_sample = min(NEG_SAMPLES_PER_QUERY, len(pool))
    sampled_neg = pool.sample(n=n_sample, random_state=PAIR_SEED)[["domain", "summary"]] if n_sample > 0 else pool[["domain", "summary"]]

    query_negatives = pd.concat([q_known_neg, sampled_neg], ignore_index=True)
    if len(query_negatives) == 0:
        continue
    query_text = query_lookup[qid]

    for _, pos_row in q_positives.iterrows():
        n_neg = min(NEGATIVES_PER_POSITIVE, len(query_negatives))
        neg_sample = query_negatives.sample(n=n_neg, random_state=PAIR_SEED + seed_counter)
        seed_counter += 1
        for _, neg_row in neg_sample.iterrows():
            pair_rows.append({
                "query_id": qid, "query": query_text,
                "pos_summary": pos_row["summary"], "neg_summary": neg_row["summary"],
            })

pairs_df = pd.DataFrame(pair_rows)
print(f"Training pairs: {len(pairs_df):,} from {pairs_df['query_id'].nunique()} queries")
print(f"(v1 comparison: 820 individual examples, only 19 of them negative; here every pair has an explicit negative)")


# In[ ]:


REPO = "BAAI/bge-reranker-v2-m3"
print(f"[Load] Loading {REPO}...")
t0 = time.time()
os.environ["HF_HUB_OFFLINE"] = "1"
try:
    tokenizer = AutoTokenizer.from_pretrained(REPO, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(REPO, local_files_only=True).to(DEVICE)
    print("[Load] Loaded from local cache -- skipped Hugging Face Hub network calls")
except Exception:
    print("[Load] Not fully cached locally yet -- retrying with network access")
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForSequenceClassification.from_pretrained(REPO).to(DEVICE)
print(f"[Load] Model loaded in {time.time()-t0:.1f}s on {DEVICE}")


# In[ ]:


finetuned_path = OUTPUT_DIR / "finetuned_model"

BATCH_SIZE = 16  # pairs per batch (32 individual examples scored per forward pass, same per-pass size as v1)
N_EPOCHS = 4
LEARNING_RATE = 2e-5
MARGIN = 1.0

if finetuned_path.exists():
    print("[Train] Already fine-tuned -- loading saved checkpoint instead of retraining")
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_path).to(DEVICE)
    model.eval()
else:
    print(f"[Train] Fine-tuning for {N_EPOCHS} epochs, batch size {BATCH_SIZE} pairs, lr {LEARNING_RATE}, margin {MARGIN}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MarginRankingLoss(margin=MARGIN)
    model.train()

    n = len(pairs_df)
    rng = np.random.RandomState(42)
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        order = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, BATCH_SIZE):
            batch = pairs_df.iloc[order[i:i + BATCH_SIZE]]
            pos_pairs = [[q, s] for q, s in zip(batch["query"], batch["pos_summary"])]
            neg_pairs = [[q, s] for q, s in zip(batch["query"], batch["neg_summary"])]
            all_pairs = pos_pairs + neg_pairs

            encoded = tokenizer(all_pairs, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
            logits = model(**encoded).logits.squeeze(-1)
            pos_scores, neg_scores = logits[:len(pos_pairs)], logits[len(pos_pairs):]
            target = torch.ones_like(pos_scores)
            loss = loss_fn(pos_scores, neg_scores, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        print(f"[Train] Epoch {epoch+1}/{N_EPOCHS}  mean margin loss={epoch_loss/n_batches:.4f}  ({(time.time()-t0)/60:.1f} min elapsed)")

    model.eval()
    model.save_pretrained(finetuned_path)
    tokenizer.save_pretrained(finetuned_path)
    print(f"[Train] Done. Saved -> {finetuned_path}")


# In[ ]:


@torch.no_grad()
def score_pool(model, query, summaries, batch_size=32):
    pairs = [[query, s] for s in summaries]
    all_scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        scores = model(**encoded).logits.squeeze(-1)
        all_scores.extend(scores.cpu().float().tolist())
    return all_scores


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


def evaluate_model(model, label):
    rows = []
    t0 = time.time()
    for qi, qid in enumerate(DEEP_QUERY_IDS):
        pool = features[features["query_id"] == qid]
        query_text = query_lookup[qid]
        scores = score_pool(model, query_text, pool["summary"].tolist())
        ranked = pool.assign(score=scores).sort_values("score", ascending=False)
        retrieved = ranked["domain"].tolist()
        relevant = relevant_sets.get(qid, set())
        for k in K_VALUES:
            rows.append({
                "model": label, "query_id": qid, "k": k,
                "precision": precision_at_k(retrieved, relevant, k),
                "recall": recall_at_k(retrieved, relevant, k),
                "ndcg": ndcg_at_k(retrieved, relevant, k),
            })
        print(f"  [{label}] {qi+1}/{len(DEEP_QUERY_IDS)} queries scored ({(time.time()-t0)/60:.1f} min elapsed)")
    return pd.DataFrame(rows)


# In[ ]:


finetuned_v2_eval_path = OUTPUT_DIR / "finetuned_v2_eval.csv"
if finetuned_v2_eval_path.exists():
    print("[Fine-tuned v2] Already evaluated -- loading from disk")
    finetuned_v2_eval = pd.read_csv(finetuned_v2_eval_path)
else:
    print("[Fine-tuned v2] Evaluating the pairwise-trained reranker on the 19 held-out queries...")
    finetuned_v2_eval = evaluate_model(model, "finetuned_v2")
    finetuned_v2_eval.to_csv(finetuned_v2_eval_path, index=False)

# Reuse notebook 62's already-computed pretrained and v1 (MSE regression) evaluations for a full 3-way comparison
pretrained_eval = pd.read_csv(V1_DIR / "pretrained_eval.csv")
pretrained_eval["model"] = "pretrained"
finetuned_v1_eval = pd.read_csv(V1_DIR / "finetuned_eval.csv")
finetuned_v1_eval["model"] = "finetuned_v1_mse"

comparison = pd.concat([pretrained_eval, finetuned_v1_eval, finetuned_v2_eval], ignore_index=True)
summary = comparison.groupby(["model", "k"])[["precision", "recall", "ndcg"]].mean().round(4)
summary.to_csv(OUTPUT_DIR / "three_way_comparison.csv")
print(summary)

