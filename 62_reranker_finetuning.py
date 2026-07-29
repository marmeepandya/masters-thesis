#!/usr/bin/env python
# coding: utf-8

# # 62 - Fine-Tuning the Reranker on Gold-Labeled Pairs
# 
# `BAAI/bge-reranker-v2-m3` is used everywhere in this pipeline as a frozen, pretrained cross-encoder, never fit to this thesis's own queries or labels (Sections on hybrid reranking, and one of the ten fusion-ranker features). This notebook fine-tunes it on the gold-labeled (query, candidate) pairs and checks whether that measurably improves ranking quality.
# 
# Same query-level split as notebook 61, reused rather than reinvented: the 82 queries without deep gold coverage are used for fine-tuning, and the 19 deep-coverage queries (5-query pilot + 14 headline queries) are held out completely, never touched during training, so the final comparison against the pretrained baseline is leakage-free.

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

OUTPUT_DIR = Path("result/62_reranker_finetuning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# round_gold only carries query_id/domain/gold_label -- fill in query text and summary the same way notebooks 58-60 do
round_gold = round_gold.copy()
round_gold["query"] = round_gold["query_id"].map(query_lookup)
round_gold = round_gold.merge(corpus, on="domain", how="left")

gold = pd.concat([
    base_gold[["query_id", "domain", "gold_label", "query", "summary"]],
    round_gold[["query_id", "domain", "gold_label", "query", "summary"]],
], ignore_index=True).drop_duplicates(subset=["query_id", "domain"])
gold["summary"] = gold["summary"].fillna("")
print(f"Combined gold standard: {len(gold):,} candidates across {gold['query_id'].nunique()} queries")

ALL_QUERY_IDS = sorted(gold["query_id"].unique().tolist())
TUNING_QUERY_IDS = sorted(set(ALL_QUERY_IDS) - set(DEEP_QUERY_IDS))
print(f"Held-out deep queries (never touched during fine-tuning): {len(DEEP_QUERY_IDS)}")
print(f"Tuning-only queries (used for fine-tuning): {len(TUNING_QUERY_IDS)}")


# In[ ]:


train_gold = gold[gold["query_id"].isin(TUNING_QUERY_IDS)].reset_index(drop=True)
train_pairs = list(zip(train_gold["query"], train_gold["summary"]))
train_targets = (train_gold["gold_label"] / 2.0).tolist()  # 0/1/2 -> 0.0/0.5/1.0, matching the reranker's continuous score output
print(f"Training examples: {len(train_pairs):,} pairs from {train_gold['query_id'].nunique()} queries")
print(f"Target score distribution: {pd.Series(train_targets).value_counts().sort_index().to_dict()}")


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

BATCH_SIZE = 32
N_EPOCHS = 4
LEARNING_RATE = 2e-5

if finetuned_path.exists():
    print("[Train] Already fine-tuned -- loading saved checkpoint instead of retraining")
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_path).to(DEVICE)
    model.eval()
else:
    print(f"[Train] Fine-tuning for {N_EPOCHS} epochs, batch size {BATCH_SIZE}, lr {LEARNING_RATE}")
    print("[Train] Small dataset (~1,000 examples) for a large model -- real overfitting risk, watch the held-out evaluation below")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    model.train()

    n = len(train_pairs)
    rng = np.random.RandomState(42)
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        order = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, BATCH_SIZE):
            batch_idx = order[i:i + BATCH_SIZE]
            batch_pairs = [[train_pairs[j][0], train_pairs[j][1]] for j in batch_idx]
            batch_targets = torch.tensor([train_targets[j] for j in batch_idx], dtype=torch.float32, device=DEVICE)

            encoded = tokenizer(batch_pairs, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
            logits = model(**encoded).logits.squeeze(-1)
            loss = loss_fn(logits, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        print(f"[Train] Epoch {epoch+1}/{N_EPOCHS}  mean MSE loss={epoch_loss/n_batches:.4f}  ({(time.time()-t0)/60:.1f} min elapsed)")

    model.eval()
    model.save_pretrained(finetuned_path)
    tokenizer.save_pretrained(finetuned_path)
    print(f"[Train] Done. Saved -> {finetuned_path}")


# In[ ]:


features = pd.read_csv("result/30_learned_fusion_ranker/scored_candidates.csv")[["query_id", "domain"]]
features = features.merge(corpus, on="domain", how="left")
features["summary"] = features["summary"].fillna("")

relevant_sets = {
    qid: set(g.loc[g["gold_label"] >= RELEVANT_THRESHOLD, "domain"])
    for qid, g in gold.groupby("query_id")
}


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


pretrained_path = OUTPUT_DIR / "pretrained_eval.csv"
if pretrained_path.exists():
    print("[Pretrained] Already evaluated -- loading from disk")
    pretrained_eval = pd.read_csv(pretrained_path)
else:
    print("[Pretrained] Evaluating the original, un-fine-tuned reranker on the 19 held-out queries...")
    try:
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(REPO, local_files_only=True).to(DEVICE)
    except Exception:
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(REPO).to(DEVICE)
    pretrained_model.eval()
    pretrained_eval = evaluate_model(pretrained_model, "pretrained")
    pretrained_eval.to_csv(pretrained_path, index=False)
    del pretrained_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

finetuned_eval_path = OUTPUT_DIR / "finetuned_eval.csv"
if finetuned_eval_path.exists():
    print("[Fine-tuned] Already evaluated -- loading from disk")
    finetuned_eval = pd.read_csv(finetuned_eval_path)
else:
    print("[Fine-tuned] Evaluating the fine-tuned reranker on the 19 held-out queries...")
    finetuned_eval = evaluate_model(model, "finetuned")
    finetuned_eval.to_csv(finetuned_eval_path, index=False)

comparison = pd.concat([pretrained_eval, finetuned_eval], ignore_index=True)
summary = comparison.groupby(["model", "k"])[["precision", "recall", "ndcg"]].mean().round(4)
summary.to_csv(OUTPUT_DIR / "pretrained_vs_finetuned_summary.csv")
print(summary)

