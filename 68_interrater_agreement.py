#!/usr/bin/env python
# coding: utf-8

# # 68 - Inter-Rater Agreement: Cohen's Kappa for the Silver Standard's LLM Judges
# 
# Ralph's methodological feedback asked for kappa to be reported for LLM-LLM, LLM-human, and human-human agreement, rather than the raw percent-agreement figures used throughout this thesis so far. Percent agreement alone is known to overstate reliability on skewed label distributions (most candidates end up highly relevant), since two judges can agree often just by both defaulting to the majority label; Cohen's kappa \citep{cohen1960coefficient} corrects for this by subtracting out the agreement expected by chance.
# 
# This notebook covers two parts. Part 1 (LLM-LLM) can be computed right now, from judge-cache files this thesis already produced across five separate judging rounds. Part 2 (LLM-human and human-human) depends on the external relevance review sent to Istari's Kerem Cerit and Manav (Section~\ref{subsec:external_review_prep}), which had not yet been returned at the time this notebook was written; that part is written to run automatically the moment reviewed files are placed in the expected folder, rather than needing to be rebuilt later.

# In[ ]:


import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

OUTPUT_DIR = Path("result/68_interrater_agreement")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Every judging round this thesis has run so far, with the two judge keys used in that round's cache file.
# Rounds 1-3 use the original, cheaper pair (gpt-4o-mini/claude-haiku-4-5, cached under "openai"/"claude").
# Rounds 4-5 use the upgraded pair (gpt-5.4/claude-sonnet-5, cached under "gpt54"/"sonnet5").
ROUNDS = [
    ("35_llm_judge_ensemble", "result/35_llm_judge_ensemble/judge_cache.json", "openai", "claude", "original 5-query silver-standard pilot"),
    ("40_active_learning_labeling_queue", "result/40_active_learning_labeling_queue/judge_cache.json", "openai", "claude", "101-query active-learning expansion"),
    ("42_headline_query_deepening", "result/42_headline_query_deepening/judge_cache.json", "openai", "claude", "headline query deepening"),
    ("42_headline_query_deepening_validation", "result/42_headline_query_deepening/validation_judge_cache.json", "gpt54", "sonnet5", "bronze-tier spot-check, upgraded judge pair"),
    ("66_production_independent_judging_pilot", "result/66_production_independent_judging/judge_cache.json", "gpt54", "sonnet5", "production-independent pilot, pre-redesign GPT+Sonnet pair"),
]

rows = []
all_a, all_b = [], []
for round_id, path, key_a, key_b, description in ROUNDS:
    if not Path(path).exists():
        print(f"[skip] {path} not found")
        continue
    cache = json.load(open(path))
    la, lb = [], []
    for entry in cache.values():
        a = entry.get(key_a, {}).get("label")
        b = entry.get(key_b, {}).get("label")
        if a is None or b is None:
            continue
        la.append(a)
        lb.append(b)
    kappa = cohen_kappa_score(la, lb)
    agreement = sum(1 for x, y in zip(la, lb) if x == y) / len(la)
    rows.append({"round": round_id, "description": description, "judge_a": key_a, "judge_b": key_b,
                 "n_candidates": len(la), "pct_agreement": agreement, "cohens_kappa": kappa})
    all_a.extend(la)
    all_b.extend(lb)
    print(f"{round_id:<45} n={len(la):>5}  agreement={agreement*100:5.1f}%  kappa={kappa:.3f}")

pooled_kappa = cohen_kappa_score(all_a, all_b)
pooled_agreement = sum(1 for x, y in zip(all_a, all_b) if x == y) / len(all_a)
rows.append({"round": "pooled_all_rounds", "description": "every LLM-LLM judging round combined", "judge_a": "mixed", "judge_b": "mixed",
             "n_candidates": len(all_a), "pct_agreement": pooled_agreement, "cohens_kappa": pooled_kappa})
print(f"\n{'POOLED':<45} n={len(all_a):>5}  agreement={pooled_agreement*100:5.1f}%  kappa={pooled_kappa:.3f}")

llm_llm_df = pd.DataFrame(rows)
llm_llm_df.to_csv(OUTPUT_DIR / "llm_llm_kappa.csv", index=False)
print(f"\nSaved to {OUTPUT_DIR / 'llm_llm_kappa.csv'}")


# ## Part 2: LLM-human and human-human agreement
# 
# Requires the relevance review Kerem and Manav were sent (Section~\ref{subsec:external_review_prep}, four `.xlsx` files under `result/58_gold_label_review_export/`, each with a blank `relevance_label` column). Once a reviewer fills in and returns a file, save it to `result/58_gold_label_review_export/returned/<reviewer_name>/<original_filename>.xlsx`, keeping the filename and row order unchanged, only `relevance_label` filled in. This cell picks up any reviewer folders it finds automatically; run again after each new batch comes back rather than waiting for both reviewers to finish everything.

# In[ ]:


RETURNED_DIR = Path("result/58_gold_label_review_export/returned")
reviewer_dirs = sorted([d for d in RETURNED_DIR.glob("*") if d.is_dir()]) if RETURNED_DIR.exists() else []

if not reviewer_dirs:
    print(f"No returned reviews found yet under {RETURNED_DIR} -- Part 2 cannot run until at least one reviewer's files are saved there.")
    print("Expected layout: result/58_gold_label_review_export/returned/<reviewer_name>/<original_filename>.xlsx")
else:
    QUERY_IDS = {1: "software companies", 92: "wealth management firms"}

    def load_reviewer(reviewer_dir):
        frames = []
        for f in sorted(reviewer_dir.glob("*.xlsx")):
            df = pd.read_excel(f)
            if "relevance_label" not in df.columns or df["relevance_label"].isna().all():
                continue
            qid = next((q for q, name in QUERY_IDS.items() if name.replace(" ", "_") in f.stem), None)
            df["query_id"] = qid
            df["reviewer"] = reviewer_dir.name
            frames.append(df[["query_id", "domain", "relevance_label", "reviewer"]])
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["query_id", "domain", "relevance_label", "reviewer"])

    reviews = pd.concat([load_reviewer(d) for d in reviewer_dirs], ignore_index=True)
    reviews = reviews.dropna(subset=["relevance_label"])
    print(f"Loaded {len(reviews)} filled-in human labels across reviewers: {reviews['reviewer'].unique().tolist()}")

    # LLM-human: compare each human label against this thesis's own existing silver-standard label,
    # only where that specific (query_id, domain) candidate already happened to carry one.
    base_gold = pd.read_json("result/40_active_learning_labeling_queue/expanded_gold_labels.json")
    round_gold = pd.read_json("result/42_headline_query_deepening/round_gold_labels.json")
    silver_standard = pd.concat([base_gold[["query_id", "domain", "gold_label"]], round_gold[["query_id", "domain", "gold_label"]]], ignore_index=True).drop_duplicates(subset=["query_id", "domain"])

    llm_human = reviews.merge(silver_standard, on=["query_id", "domain"], how="inner")
    if len(llm_human):
        k = cohen_kappa_score(llm_human["relevance_label"], llm_human["gold_label"])
        agree = (llm_human["relevance_label"] == llm_human["gold_label"]).mean()
        print(f"\nLLM-human kappa (n={len(llm_human)} overlapping candidates): agreement={agree*100:.1f}%, kappa={k:.3f}")
        llm_human.to_csv(OUTPUT_DIR / "llm_human_overlap.csv", index=False)
    else:
        print("\nNo overlap yet between returned human labels and this thesis's existing silver-standard candidates -- LLM-human kappa not computable yet.")

    # human-human: only meaningful once at least two reviewers have returned labels for the same candidates.
    if reviews["reviewer"].nunique() >= 2:
        pivot = reviews.pivot_table(index=["query_id", "domain"], columns="reviewer", values="relevance_label", aggfunc="first")
        pivot = pivot.dropna()
        if len(pivot) and pivot.shape[1] >= 2:
            r1, r2 = pivot.columns[0], pivot.columns[1]
            k = cohen_kappa_score(pivot[r1], pivot[r2])
            agree = (pivot[r1] == pivot[r2]).mean()
            print(f"\nHuman-human kappa ({r1} vs {r2}, n={len(pivot)} shared candidates): agreement={agree*100:.1f}%, kappa={k:.3f}")
            pivot.to_csv(OUTPUT_DIR / "human_human_overlap.csv")
        else:
            print("\nReviewers have not yet labelled any of the same candidates -- human-human kappa not computable yet.")
    else:
        print(f"\nOnly {reviews['reviewer'].nunique()} reviewer(s) returned so far -- human-human kappa needs at least two.")

