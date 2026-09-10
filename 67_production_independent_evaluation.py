#!/usr/bin/env python
# coding: utf-8

# # 67 - Closing the Gap: Can This Pipeline Find Production-Blind, Gold-Verified Relevant Companies?
# 
# Production has no ranking data at all for `source == "new"` companies (the random 300K sample), since `production_results.xlsx` only ever covers the original 98,716-company corpus. This means a symmetric "production vs. ours" ranking comparison, like Section~\ref{subsec:goi_vs_ours_ranking}'s, isn't even defined here. The meaningful, cleaner test this notebook runs instead: take the gold-verified relevant companies from notebook 66 (independently judged, unanimous agreement, drawn exclusively from candidates production was never asked to rank for any of these queries), and check whether this thesis's own retrieval pipeline, searching the *full*, realistic corpus (not restricted to `source == "new"` this time, the real search space a user would actually query), finds and ranks them well. This directly tests the coverage-gain claim from Section~\ref{subsec:production_divergence} on a candidate pool that is clean of the circularity problem identified in Section~\ref{subsec:goi_vs_ours_ranking}, by construction, not by argument.
# 
# Run across all four corpus tiers (100K-400K), the same methods and metrics as the scaling evaluation (Section~\ref{sec:scaling_corpus}), so this result is directly comparable to, and can be read alongside, everything already reported there.

# In[1]:


import os
import json
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

OUTPUT_DIR = Path("result/67_production_independent_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIERS = [("100k", "in_100k"), ("200k", "in_200k"), ("300k", "in_300k"), ("400k", "in_400k")]
K_VALUES = [10, 50, 100, 300, 500, 1000]
TOP_K = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

gold_path = Path("result/66_production_independent_judging/production_independent_gold_labels.json")
if not gold_path.exists():
    raise FileNotFoundError(
        "Notebook 66 hasn't been run yet (or produced no output) -- this notebook needs its "
        "production_independent_gold_labels.json to know which companies are actually relevant."
    )
gold = pd.read_json(gold_path)

manual_path = Path("result/66_production_independent_judging/manual_review_queue.json")
if manual_path.exists():
    manual = pd.read_json(manual_path)
    n_resolved = manual["resolved_label"].notna().sum() if "resolved_label" in manual.columns else 0
    print(f"[Note] {len(manual)} candidates are in the manual tie-break queue; {n_resolved} have a resolved_label so far.")
    if "resolved_label" in manual.columns:
        resolved = manual[manual["resolved_label"].notna()].rename(columns={"resolved_label": "gold_label"})
        resolved["agreement"] = "manual"
        gold = pd.concat([gold, resolved[["query_id", "query", "domain", "name", "summary", "sources", "gold_label", "agreement"]]], ignore_index=True)

print(f"[Load] Production-independent gold labels: {len(gold):,} candidates across {gold['query_id'].nunique()} queries")
print(gold["gold_label"].value_counts().sort_index())

relevant_sets = {
    qid: set(g.loc[g["gold_label"] >= 2, "domain"])
    for qid, g in gold.groupby("query_id")
}
n_relevant_total = sum(len(v) for v in relevant_sets.values())
print(f"[Load] Highly-relevant (gold_label>=2), production-independent companies: {n_relevant_total} across {len(relevant_sets)} queries")
for qid, rel in relevant_sets.items():
    print(f"  query {qid}: {len(rel)} gold-verified relevant")


# In[2]:


DEEP_QUERY_IDS = sorted(relevant_sets.keys())
combined = pd.read_parquet("result/44_build_scaled_corpus/combined_pool.parquet")

train_queries = json.load(open("result/08_llm_relevance_judge/train_queries.json"))
held_out_queries = json.load(open("result/08_llm_relevance_judge/held_out_queries.json"))
query_lookup = {item["query_id"]: item["query"] for item in train_queries + held_out_queries}

os.environ["HF_HUB_OFFLINE"] = "1"
query_texts = [query_lookup[qid] for qid in DEEP_QUERY_IDS]

from sentence_transformers import SentenceTransformer

try:
    minilm_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE, local_files_only=True)
except Exception:
    minilm_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
minilm_query_embs = minilm_model.encode(query_texts, normalize_embeddings=False, convert_to_numpy=True).astype("float32")
del minilm_model

try:
    gte_model = SentenceTransformer("thenlper/gte-large", device=DEVICE, local_files_only=True)
except Exception:
    gte_model = SentenceTransformer("thenlper/gte-large", device=DEVICE)
gte_query_embs = gte_model.encode(query_texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
del gte_model
if DEVICE == "cuda":
    torch.cuda.empty_cache()

from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

TASK_INSTRUCTION = "Given a search query describing a type of company, retrieve relevant company profiles"
query_prefix = f"Instruct: {TASK_INSTRUCTION}\nQuery: "
REPO = "Linq-AI-Research/Linq-Embed-Mistral"
try:
    linq_tokenizer = AutoTokenizer.from_pretrained(REPO, local_files_only=True)
    linq_model = AutoModel.from_pretrained(REPO, torch_dtype=torch.float16, device_map=DEVICE, local_files_only=True)
except Exception:
    linq_tokenizer = AutoTokenizer.from_pretrained(REPO)
    linq_model = AutoModel.from_pretrained(REPO, torch_dtype=torch.float16, device_map=DEVICE)


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


@torch.no_grad()
def encode_linq(texts):
    batch_dict = linq_tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    outputs = linq_model(**batch_dict)
    embs = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    embs = F.normalize(embs, p=2, dim=1)
    return embs.cpu().float().numpy()


linq_query_embs = encode_linq([query_prefix + q for q in query_texts]).astype("float32")
del linq_model
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print(f"[Encode] Done for {len(DEEP_QUERY_IDS)} queries with gold-verified production-independent relevant companies.")


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


EMBEDDING_METHODS = {
    "minilm": ("result/45_encode_minilm_scaled/company_embeddings.npy", minilm_query_embs, "l2"),
    "gte": ("result/46_encode_gte_scaled/company_embeddings.npy", gte_query_embs, "cosine"),
    "linq": ("result/47_encode_linq_mistral_scaled/company_embeddings.npy", linq_query_embs, "cosine"),
}

eval_rows = []
for method_name, (emb_path, query_embs, metric) in EMBEDDING_METHODS.items():
    full_embs = np.load(emb_path, mmap_mode="r")
    for tier_name, flag_col in TIERS:
        mask = combined[flag_col].values
        tier_domains = combined.loc[mask, "domain"].reset_index(drop=True)
        tier_embs = np.asarray(full_embs[mask]).astype("float32")

        for qi, qid in enumerate(DEEP_QUERY_IDS):
            q_emb = query_embs[qi]
            if metric == "l2":
                dists = np.sum((tier_embs - q_emb) ** 2, axis=1)
                top_idx = np.argsort(dists)[:TOP_K]
            else:
                sims = tier_embs @ q_emb
                top_idx = np.argsort(sims)[::-1][:TOP_K]
            retrieved = [tier_domains.iloc[i] for i in top_idx]
            relevant = relevant_sets[qid]
            for k in K_VALUES:
                eval_rows.append({
                    "method": method_name, "tier": tier_name, "query_id": qid, "k": k,
                    "precision": precision_at_k(retrieved, relevant, k),
                    "recall": recall_at_k(retrieved, relevant, k),
                    "ndcg": ndcg_at_k(retrieved, relevant, k),
                })
        print(f"[{method_name}] tier {tier_name} done")
    del full_embs

eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(OUTPUT_DIR / "production_independent_eval_per_query.csv", index=False)
print(f"[Done] {len(eval_df):,} eval rows")


# In[ ]:


summary = eval_df.groupby(["method", "tier", "k"])[["precision", "recall", "ndcg"]].mean().reset_index()
summary["tier"] = pd.Categorical(summary["tier"], categories=["100k", "200k", "300k", "400k"], ordered=True)
summary = summary.sort_values(["method", "tier", "k"])
summary.to_csv(OUTPUT_DIR / "production_independent_eval_summary.csv", index=False)
print(summary.to_string(index=False))
print()
print("Read this as: of the companies independently, blindly verified as relevant, none of which")
print("production had any opportunity to already know about, how many does this pipeline actually")
print("find and rank well, as the corpus scales from 100K to 400K companies.")

