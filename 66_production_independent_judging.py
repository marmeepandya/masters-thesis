#!/usr/bin/env python
# coding: utf-8

# # 66 - LLM Judging of the Production-Independent Pool
# 
# Judges every trustworthy candidate from notebook 65 with two LLM judges from two different labs, Claude Sonnet 5 (Anthropic) and Gemini 3.6 Flash (Google), on the same 0/1/2 relevance scale used throughout this thesis. This replaces the earlier GPT-5.4 + Claude Sonnet 5 pair originally planned to match the silver-label spot-checks in Section~\ref{sec:silver_labels}. GPT-5.4 is dropped here because the live v2 API production baseline, one of the pooling sources behind the main silver-standard labels, is OpenAI-embedding-based, so an OpenAI judge grading candidates that baseline helped surface risks correlated, same-family bias, exactly the concern Ralph raised. Claude and Gemini have no overlap with any embedding or retrieval model used anywhere else in this thesis.
# 
# Both judges are current, non-preview, pinned production releases rather than preview/experimental tiers, chosen for reproducibility: a preview model (e.g. Gemini's 3.1 Pro preview tier) can be silently updated by the provider underneath a fixed model ID, which would undermine a thesis result tied to a specific run. Claude Sonnet 5 also carries over the empirical precedent from the earlier pilot (97.3% unanimous agreement with GPT-5.4 on the original 2-query production-independent pilot), so at minimum one judge in the panel has a track record on this exact task.
# 
# Where the two judges agree, that label is accepted directly. Where they disagree, the candidate is written to a separate manual-review file instead of being silently resolved, matching the tie-breaking principle used for every other gold/silver-standard pass in this thesis.
# 
# **Costs real money per API call.** Run the pilot cell first (a couple of queries) to sanity-check cost and agreement rate before judging the full pool, especially since the Gemini integration here is new and untested in this pipeline. At the pooling depth used in notebook 65 (roughly 19 queries x up to 80 candidates each), this should cost on the order of a few dollars for the Claude side; Gemini pricing for this pair wasn't independently verified here, so treat the pilot cell as the real cost check rather than trusting this estimate.
# 
# **Manual step after this notebook runs:** open `manual_review_queue.json` and add a `resolved_label` field (0/1/2) to each row by hand, the same human tie-breaking already done for every other gold/silver-standard pass in this thesis. Notebook 67 looks for that `resolved_label` column and folds it back in automatically if present; if you skip this step, notebook 67 still runs fine using only the unanimous-agreement labels, just with a smaller relevant set.

# In[ ]:


import os
import json
import time
import pandas as pd
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

OUTPUT_DIR = Path("result/66_production_independent_judging")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = OUTPUT_DIR / "judge_cache.json"
MANUAL_REVIEW_PATH = OUTPUT_DIR / "manual_review_queue.json"

PILOT_N_QUERIES = None  # 2-query pilot succeeded (144/147 judged, 96.5% unanimous agreement) -- go to the full 19-query run

if GEMINI_API_KEY is None or ANTHROPIC_API_KEY is None:
    print("WARNING: missing GEMINI_API_KEY or ANTHROPIC_API_KEY in .env -- both judges are needed for agreement/disagreement voting.", flush=True)

pooled_df = pd.read_json("result/65_production_independent_pooling/pooled_candidates.json")
pooled_df = pooled_df[pooled_df["summary_trustworthy"]].reset_index(drop=True)
print(f"Trustworthy pooled candidates available for judging: {len(pooled_df)}", flush=True)

# cache from the earlier GPT-5.4 pilot still has entries under the "gpt54" key -- harmless, just ignored now,
# and the "sonnet5" entries under those same keys are reused as-is, so no extra Claude cost for re-judging the pilot queries
if PILOT_N_QUERIES is not None:
    pilot_query_ids = sorted(pooled_df["query_id"].unique())[:PILOT_N_QUERIES]
    judge_df = pooled_df[pooled_df["query_id"].isin(pilot_query_ids)].reset_index(drop=True)
    print(f"PILOT MODE: judging {len(judge_df)} candidates across {PILOT_N_QUERIES} queries", flush=True)
else:
    judge_df = pooled_df
    print(f"FULL RUN: judging all {len(judge_df)} candidates across {judge_df['query_id'].nunique()} queries", flush=True)


# In[ ]:


ENRICHED_JUDGE_PROMPT_TEMPLATE = """You are judging search result relevance for a company search engine.

Search query: "{query}"

Candidate company:
Name: {name}
Country: {country}
State/region: {state}
District: {district}
Municipality: {municipality}
Organization type: {organization_type}
Organization size: {organization_size}
NACE industry code: {nace_code}
Summary: {summary}
Summary keywords: {summary_keywords}

Rate how relevant this company is to the search query, using exactly one of these labels:
2 = highly relevant: a strong, direct match, the kind of company someone searching this query would expect and want to find.
1 = partially relevant: related to the query but not a strong direct match, e.g. an adjacent business, or a company that only partially fits the query's intent, or fits it for one line of business among several unrelated ones.
0 = not relevant: nothing meaningful to do with the query, including a company that merely mentions a query-related term in passing without it describing what the company actually does (for example, a venture capital firm whose portfolio includes software startups is NOT itself a software company for the query "software companies").

If the summary gives no verifiable content to judge from (placeholder text, clearly broken description), use 0 rather than guessing based on the name alone. Judge based on what the company currently does, not speculation about unstated subsidiaries or past business lines.

These are the same assessor guidelines given to the human reviewers labelling a parallel sample of this data, so your judgments can be meaningfully compared against theirs.

Respond with ONLY a JSON object: {{"label": <0, 1, or 2>, "reason": "<one short sentence>"}}"""


def build_prompt(row):
    fields = {c: (row.get(c, "") if pd.notna(row.get(c, "")) else "unknown") for c in
              ["query", "name", "country", "state", "district", "municipality",
               "organization_type", "organization_size", "nace_code", "summary", "summary_keywords"]}
    return ENRICHED_JUDGE_PROMPT_TEMPLATE.format(**fields)


def parse_judge_reply(text):
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        parsed = json.loads(text[start:end])
        return int(parsed["label"]), parsed.get("reason", "")
    except (ValueError, KeyError, json.JSONDecodeError):
        return None, f"UNPARSEABLE: {text[:200]}"


def judge_gemini(prompt, max_retries=6):
    # Uses the long-established generateContent endpoint rather than Google's newer "Interactions API" --
    # the first pilot run against Interactions API failed on every single call (some with a missing
    # "output_text" key on 200 responses, meaning that endpoint's response shape did not match what its
    # own docs described at the time this was written; some with rate-limit errors that had no retry logic
    # to recover from). generateContent is a much longer-standing, stable surface, so this is the safer bet.
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=90,
            )
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (2 ** attempt))
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            # 429 (rate limit) and 5xx (transient server-side issues, e.g. the 503s seen mid-run on
            # 2026-08-04) are both worth retrying with backoff; a 4xx other than 429 is a real request
            # problem and should surface immediately via raise_for_status below instead of being retried.
            if attempt == max_retries - 1:
                resp.raise_for_status()
            wait_s = int(resp.headers.get("Retry-After", 5 * (2 ** attempt)))
            time.sleep(wait_s)
            continue
        resp.raise_for_status()
        body = resp.json()
        try:
            text = body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            # Capture the real response shape instead of a bare KeyError, so a repeat failure is diagnosable.
            return None, f"UNEXPECTED RESPONSE SHAPE: {json.dumps(body)[:300]}"
        return parse_judge_reply(text)
    return None, "EXHAUSTED RETRIES: repeated 429/5xx from Gemini"


def judge_claude_sonnet5(prompt):
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
        json={
            "model": "claude-sonnet-5", "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    # Sonnet 5 runs adaptive thinking by default, so content[0] may be a "thinking" block,
    # not "text" -- find the actual text block instead of assuming it's first.
    content_blocks = resp.json()["content"]
    text_block = next((b["text"] for b in content_blocks if b.get("type") == "text"), None)
    if text_block is None:
        return None, f"NO TEXT BLOCK: {content_blocks}"
    return parse_judge_reply(text_block)


# In[ ]:


cache = json.load(open(CACHE_PATH)) if CACHE_PATH.exists() else {}
print(f"Loaded {len(cache)} cached entries ({sum(1 for e in cache.values() if e.get('gemini', {}).get('label') is not None)} with a successful gemini label)", flush=True)

for i, row in judge_df.iterrows():
    key = f'{row["query_id"]}::{row["domain"]}'
    entry = cache.get(key, {})
    needs_gemini = entry.get("gemini", {}).get("label") is None
    needs_sonnet5 = entry.get("sonnet5", {}).get("label") is None

    if not (needs_gemini or needs_sonnet5):
        continue  # both judges already cached for this candidate -- no API call, no disk write, no sleep needed

    prompt = build_prompt(row)

    # Checking the actual label (not just key presence) matters: the first pilot run left 147 entries
    # with a "gemini" key whose label is None (a failed call, not a skipped one) -- re-running without
    # this check would treat those as already-done and never retry them.
    if needs_gemini:
        try:
            label, reason = judge_gemini(prompt)
            entry["gemini"] = {"label": label, "reason": reason}
        except (requests.exceptions.RequestException, KeyError, IndexError) as e:
            print(f"  [gemini] error on {row['domain']}: {e}", flush=True)
    if needs_sonnet5:
        try:
            label, reason = judge_claude_sonnet5(prompt)
            entry["sonnet5"] = {"label": label, "reason": reason}
        except (requests.exceptions.RequestException, KeyError, IndexError) as e:
            print(f"  [sonnet5] error on {row['domain']}: {e}", flush=True)

    cache[key] = entry
    tmp_path = CACHE_PATH.with_suffix(".json.tmp")
    json.dump(cache, open(tmp_path, "w"), indent=2, default=str)
    tmp_path.replace(CACHE_PATH)  # atomic save after every single candidate -- paid calls, never redo work

    if (i + 1) % 10 == 0 or (i + 1) == len(judge_df):
        n_done = sum(1 for e in cache.values() if e.get("gemini", {}).get("label") is not None and e.get("sonnet5", {}).get("label") is not None)
        print(f"  row {i+1}/{len(judge_df)} reached, {n_done} candidates fully judged so far", flush=True)
    time.sleep(2.0)  # only after a real API call was made -- this was previously unconditional, so resuming a run with hundreds of already-cached candidates spent minutes sleeping through pure skips before any new work happened

print("Done judging.", flush=True)


# In[ ]:


gold_rows = []
manual_review_rows = []

for i, row in judge_df.iterrows():
    key = f'{row["query_id"]}::{row["domain"]}'
    entry = cache.get(key, {})
    gemini_label = entry.get("gemini", {}).get("label")
    sonnet_label = entry.get("sonnet5", {}).get("label")

    if gemini_label is None or sonnet_label is None:
        continue  # a judge call failed or was unparseable -- exclude rather than guess

    base = {"query_id": row["query_id"], "query": row["query"], "domain": row["domain"],
            "name": row["name"], "summary": row["summary"], "sources": row["sources"]}

    if gemini_label == sonnet_label:
        gold_rows.append({**base, "gold_label": gemini_label, "agreement": "unanimous"})
    else:
        manual_review_rows.append({**base, "gemini_label": gemini_label, "sonnet5_label": sonnet_label,
                                     "gemini_reason": entry["gemini"]["reason"], "sonnet5_reason": entry["sonnet5"]["reason"]})

gold_df = pd.DataFrame(gold_rows)
manual_df = pd.DataFrame(manual_review_rows)

gold_df.to_json(OUTPUT_DIR / "production_independent_gold_labels.json", orient="records", indent=2)
manual_df.to_json(MANUAL_REVIEW_PATH, orient="records", indent=2)

n_total = len(gold_df) + len(manual_df)
print(f"Unanimous agreement: {len(gold_df)}/{n_total} ({100*len(gold_df)/n_total:.1f}%)" if n_total else "No judged candidates yet.", flush=True)
print(f"Needs manual tie-break: {len(manual_df)} -- see {MANUAL_REVIEW_PATH}", flush=True)
if len(gold_df):
    print(gold_df["gold_label"].value_counts().sort_index(), flush=True)

