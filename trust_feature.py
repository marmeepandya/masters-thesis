#!/usr/bin/env python
# coding: utf-8

# trust_filter.py — shared summary / summary_keywords trustworthiness verdicts
#
# Single source of truth for judging whether a GOI company's `summary` and `summary_keywords` fields are usable retrieval metadata for that specific company, or should be excluded because they're boilerplate (hosting provider/registrar/website-builder filler) or topically mismatched with each other.
#
# `summary_trustworthy` and `keywords_trustworthy` are computed and returned SEPARATELY and are not derived from one another -- validated against result/llm_summary_trustworthiness/llm_judged_sample.csv (300 LLM-judged companies from notebook 10): 16/300 had a bad summary but trustworthy keywords, 35/300 had the reverse, so one flag cannot stand in for the other.
#
# v2 design note: an earlier version of this module detected boilerplate via a
# per-corpus duplicate_count (how many *other* domains in the local 98,716-row
# corpus share the exact same summary text) and flagged summary/domain mismatch
# via literal name/domain token matching. Both break at GOI's real scale
# (~20-40M companies): duplicate_count requires holding the whole corpus and
# recomputing a groupby for every lookup, and literal token matching doesn't
# generalize to boilerplate templates or identity mismatches it hasn't seen
# verbatim before (validated: it also just hurt precision, see git history).
#
# This version replaces both with EMBEDDING SIMILARITY against a small, fixed
# reference set (~300 known boilerplate templates, computed once via
# build_boilerplate_audit() and cached to result/summary_quality/
# summary_quality_audit.csv). That reference set's size does not grow with
# the corpus, so a verdict on any single company -- cached or freshly pulled
# from the live v2 API -- costs one embedding + a similarity check against
# ~300 vectors, regardless of whether the total corpus is 98,716 or 20-40
# million companies.
#
# This module is self-sufficient: build_boilerplate_audit(),
# compute_query_contamination(), and compute_sensitivity_analysis() (formerly
# a separate summary_quality_audit.py script) generate and analyse the
# reference set from the raw dataset files directly, so nothing outside this
# file is required to regenerate any of it from scratch.
#
# Meant to be imported identically by:
#   - local batch scripts, via the *_batch() functions (vectorized encode calls
#     over many rows at once -- do not loop encode() one row at a time), and
#   - live Istari v2 API evaluation code, per-result, via summary_verdict() /
#     keywords_verdict() (single-row convenience wrappers around the batch
#     functions) -- so the definition of "trustworthy" never drifts between
#     the two and neither one needs the full corpus in memory.
#
# Run this file directly to calibrate similarity thresholds and validate
# against the notebook 10 LLM judgments (prints precision/recall per threshold
# for both summary and keywords verdicts).

import ast
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MIN_WORDS = 15
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'

SUMMARY_JUNK_PHRASES = [
    'web hosting', 'website hosting', 'hosting provider', 'shared hosting', 'vps hosting',
    'reseller hosting', 'domain registrar', 'domain registration', 'register your domain',
    'register a domain', 'buy this domain', 'this domain is for sale', 'domain name is for sale',
    'domain is available', 'parked domain', 'parked page', 'website builder', 'site builder',
    'under construction', 'coming soon', 'default page', 'this website is for sale',
    'godaddy', 'namecheap', 'bluehost', 'hostgator', 'wordpress.com', 'wix.com', 'squarespace',
]
import re
SUMMARY_JUNK_RE = re.compile('|'.join(re.escape(p) for p in SUMMARY_JUNK_PHRASES), re.IGNORECASE)

DEFAULT_BOILERPLATE_THRESHOLD = 0.97  # cosine similarity to nearest known template -- best F1 (0.59) from validate_against_llm_judgments() threshold sweep
DEFAULT_KEYWORDS_ALIGNMENT_THRESHOLD = 0.25  # cosine similarity between keywords and summary

_embedder = None


def get_embedder():
    """Lazily loads and caches the MiniLM model (same one used in 03_baseline_minilm)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def _embed(texts, embedder=None):
    embedder = embedder or get_embedder()
    embeddings = embedder.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings)


def _coerce_keywords(summary_keywords):
    if isinstance(summary_keywords, str):
        try:
            parsed = ast.literal_eval(summary_keywords)
            return parsed if isinstance(parsed, list) else [summary_keywords]
        except (ValueError, SyntaxError):
            return [summary_keywords] if summary_keywords else []
    return summary_keywords or []


def build_boilerplate_audit(
    corpus_xlsx='dataset/production_results.xlsx',
    audit_csv='result/summary_quality/summary_quality_audit.csv',
    duplicate_threshold=5,
):
    """Identifies boilerplate/junk-summary domains in the corpus: any exact
    summary text shared by >= duplicate_threshold distinct domains is treated
    as a hosting-provider/registrar/website-builder template rather than a
    real company-specific description. Saves the junk rows to audit_csv and
    returns them as a DataFrame. This is the generator behind the reference
    set that load_boilerplate_templates() embeds -- folded in here (formerly
    a separate summary_quality_audit.py script) so this module is
    self-sufficient and can regenerate its own reference set from scratch.
    """
    audit_csv = Path(audit_csv)
    print(f'[Audit] Loading {corpus_xlsx} (read-only)...')
    corpus = pd.read_excel(corpus_xlsx)

    dup_counts = corpus.groupby('summary')['domain'].nunique().sort_values(ascending=False)
    junk_summaries = dup_counts[dup_counts >= duplicate_threshold]
    junk_mask = corpus['summary'].isin(junk_summaries.index)
    junk_rows = corpus[junk_mask].drop_duplicates('domain')[['domain', 'name', 'country', 'summary']].copy()
    junk_rows['duplicate_count'] = junk_rows['summary'].map(dup_counts)
    junk_rows = junk_rows.sort_values('duplicate_count', ascending=False)

    n_total_domains = corpus['domain'].nunique()
    n_junk_domains = junk_rows['domain'].nunique()
    print(f'[Audit] Total unique domains in corpus : {n_total_domains}')
    print(f'[Audit] Distinct boilerplate summaries  : {len(junk_summaries)} (each shared by >= {duplicate_threshold} companies)')
    print(f'[Audit] Domains affected                : {n_junk_domains} ({100*n_junk_domains/n_total_domains:.1f}% of corpus)')

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    junk_rows.to_csv(audit_csv, index=False)
    print(f'[Audit] Saved {audit_csv} ({len(junk_rows)} rows)')
    return junk_rows


def compute_query_contamination(
    goi_json='dataset/goi_search_results.json',
    audit_csv='result/summary_quality/summary_quality_audit.csv',
    out_csv='result/summary_quality/summary_quality_query_contamination.csv',
    relevance_threshold=0.75,
):
    """For each query in goi_json, measures what fraction of its
    pseudo-relevant ground truth (similarity_score >= relevance_threshold) is
    a boilerplate/junk-summary domain per build_boilerplate_audit(). High
    pct_junk means that query's ground truth is itself contaminated with
    hosting-provider/registrar templates rather than genuine matches."""
    junk_domains = set(pd.read_csv(audit_csv)['domain'])

    print(f'[GT] Loading {goi_json} (read-only)...')
    with open(goi_json) as f:
        goi_data = json.load(f)

    rows = []
    for item in goi_data:
        relevant = [r for r in item['results'] if r['similarity_score'] >= relevance_threshold]
        relevant_junk = [r for r in relevant if r['domain'] in junk_domains]
        rows.append({
            'query': item['query'],
            'n_relevant_total': len(relevant),
            'n_relevant_junk': len(relevant_junk),
            'pct_junk': round(100 * len(relevant_junk) / len(relevant), 1) if relevant else 0.0,
        })

    contamination_df = pd.DataFrame(rows).sort_values('pct_junk', ascending=False)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    contamination_df.to_csv(out_csv, index=False)

    n_contaminated = (contamination_df['n_relevant_junk'] > 0).sum()
    print(f'[GT] Queries with >=1 junk-summary company marked relevant: {n_contaminated}/{len(goi_data)}')
    print(f'[GT] Saved {out_csv}')
    return contamination_df


def compute_sensitivity_analysis(
    goi_json='dataset/goi_search_results.json',
    audit_csv='result/summary_quality/summary_quality_audit.csv',
    out_csv='result/summary_quality/summary_quality_sensitivity.csv',
    relevance_threshold=0.75,
    k_values=(10, 50, 100, 200, 500, 1000),
    v1_cache='result/09_api_evaluation/api_results_cache.pkl',
    minilm_cache='result/09_api_evaluation/minilm_results_cache.pkl',
    v2_cache='result/09b_api_v2/v2_results_cache.pkl',
):
    """Compares NDCG/Precision/Recall/F1 for cached retrieval methods against
    the ground truth as-is vs. a 'cleaned' in-memory ground truth with junk
    domains (per build_boilerplate_audit()) removed from the relevant set.
    Quantifies how much boilerplate-contaminated ground truth inflates or
    deflates each method's reported quality; does not modify any source file."""
    junk_domains = set(pd.read_csv(audit_csv)['domain'])

    with open(goi_json) as f:
        goi_data = json.load(f)
    gt_scores_by_query = {
        item['query']: {r['domain']: float(r['similarity_score']) for r in item['results']}
        for item in goi_data
    }
    gt_scores_by_query_cleaned = {
        q: {d: s for d, s in scores.items() if d not in junk_domains}
        for q, scores in gt_scores_by_query.items()
    }

    print('[Load] Loading cached retrieval results...')
    with open(v1_cache, 'rb') as f:
        api_results, _ = pickle.load(f)
    with open(minilm_cache, 'rb') as f:
        our_results, _ = pickle.load(f)

    methods = [
        ('Istari v1 BM25',   api_results['bm25']),
        ('Our MiniLM (98K)', our_results),
    ]
    v2_cache = Path(v2_cache)
    if v2_cache.exists():
        with open(v2_cache, 'rb') as f:
            v2_results, _ = pickle.load(f)
        n_v2 = sum(1 for v in v2_results.values() if v)
        print(f'[Load] v2 Semantic cache found: {n_v2} queries fetched so far')
        methods.append(('Istari v2 Semantic', v2_results))
    else:
        print('[Load] No v2 cache found yet -- skipping v2 in this comparison')

    def _ndcg(retrieved, gt_scores, k):
        dcg = sum(gt_scores.get(d, 0.0) / np.log2(i + 2) for i, d in enumerate(retrieved[:k]))
        idcg = sum(s / np.log2(i + 2) for i, s in enumerate(sorted(gt_scores.values(), reverse=True)[:k]))
        return dcg / idcg if idcg > 0 else 0

    def _precision(retrieved, gt_scores, k):
        return sum(1 for d in retrieved[:k] if gt_scores.get(d, 0) >= relevance_threshold) / k if k else 0

    def _recall(retrieved, gt_scores, k):
        total = sum(1 for s in gt_scores.values() if s >= relevance_threshold)
        return sum(1 for d in retrieved[:k] if gt_scores.get(d, 0) >= relevance_threshold) / total if total else 0

    def _f1(retrieved, gt_scores, k):
        p, r = _precision(retrieved, gt_scores, k), _recall(retrieved, gt_scores, k)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    print('[Eval] Computing metrics: original GT vs. cleaned GT (junk domains removed)...')
    detail_rows = []
    for method_name, results_dict in methods:
        for query, gt_orig in gt_scores_by_query.items():
            if query not in results_dict or not results_dict[query]:
                continue
            retrieved = [r['domain'] for r in sorted(results_dict[query], key=lambda x: x.get('rank', 9999))]
            gt_clean = gt_scores_by_query_cleaned[query]
            for k in k_values:
                if k > len(retrieved):
                    continue
                detail_rows.append({
                    'method': method_name, 'query': query, 'k': k,
                    'ndcg_original': round(_ndcg(retrieved, gt_orig, k), 4),
                    'ndcg_cleaned':  round(_ndcg(retrieved, gt_clean, k), 4),
                    'precision_original': round(_precision(retrieved, gt_orig, k), 4),
                    'precision_cleaned':  round(_precision(retrieved, gt_clean, k), 4),
                    'recall_original': round(_recall(retrieved, gt_orig, k), 4),
                    'recall_cleaned':  round(_recall(retrieved, gt_clean, k), 4),
                    'f1_original': round(_f1(retrieved, gt_orig, k), 4),
                    'f1_cleaned':  round(_f1(retrieved, gt_clean, k), 4),
                })
    detail_df = pd.DataFrame(detail_rows)

    summary_rows = []
    for method_name, _ in methods:
        for k in k_values:
            sub = detail_df[(detail_df['method'] == method_name) & (detail_df['k'] == k)]
            if len(sub) == 0:
                continue
            for metric in ['ndcg', 'precision', 'recall', 'f1']:
                orig = sub[f'{metric}_original'].mean()
                clean = sub[f'{metric}_cleaned'].mean()
                summary_rows.append({
                    'method': method_name, 'k': k, 'metric': metric,
                    'original': round(orig, 4), 'cleaned': round(clean, 4),
                    'delta': round(clean - orig, 4),
                    'delta_pct': round(100 * (clean - orig) / orig, 1) if orig > 0 else float('nan'),
                })

    sensitivity_df = pd.DataFrame(summary_rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sensitivity_df.to_csv(out_csv, index=False)
    print(f'[Eval] Saved {out_csv}')
    return sensitivity_df


def load_boilerplate_templates(
    audit_csv='result/summary_quality/summary_quality_audit.csv',
    cache_path='result/trust_filter/boilerplate_template_embeddings.npz',
    embedder=None,
    corpus_xlsx='dataset/production_results.xlsx',
):
    """Loads (and caches) embeddings for the distinct known boilerplate
    summary templates identified by build_boilerplate_audit() (exact text
    shared by >=5 companies in the local sample -- hosting providers,
    registrars, website builders, etc.). This reference set is small and
    fixed-size: it is computed once, not per-corpus-size, which is what makes
    the boilerplate check below scale to 20-40M companies instead of just the
    98,716-domain local sample. If audit_csv doesn't exist yet, it is
    generated on the fly via build_boilerplate_audit() -- this module no
    longer depends on a separate script to bootstrap its reference set.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return list(data['texts']), data['embeddings']

    audit_csv = Path(audit_csv)
    if not audit_csv.exists():
        build_boilerplate_audit(corpus_xlsx=corpus_xlsx, audit_csv=audit_csv)

    df = pd.read_csv(audit_csv)
    texts = df['summary'].drop_duplicates().tolist()
    embeddings = _embed(texts, embedder)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, texts=np.array(texts, dtype=object), embeddings=embeddings)
    return texts, embeddings


def boilerplate_similarity_batch(summaries, template_embeddings, embedder=None):
    """Max cosine similarity of each summary to the nearest known boilerplate
    template. High similarity = looks like a hosting-provider/registrar/
    website-builder template rather than a company-specific description."""
    summary_embeddings = _embed(summaries, embedder)
    sims = summary_embeddings @ template_embeddings.T  # both L2-normalized -> dot product = cosine
    return sims.max(axis=1)


def keywords_alignment_batch(summaries, keywords_lists, embedder=None):
    """Cosine similarity between each summary and its own joined keyword list.
    Low similarity = keywords are topically unrelated to what the summary
    actually describes (the pattern notebook 10's LLM judgments actually show
    -- e.g. logistics summary paired with 'Consulting, Business Continuity'
    keywords -- not generic vocabulary, which a fixed term-list can't catch)."""
    joined_keywords = [', '.join(_coerce_keywords(k)) or '(none)' for k in keywords_lists]
    summary_embeddings = _embed(summaries, embedder)
    keyword_embeddings = _embed(joined_keywords, embedder)
    return (summary_embeddings * keyword_embeddings).sum(axis=1)  # row-wise cosine, both normalized


def summary_verdicts_batch(
    names, domains, summaries, template_embeddings, embedder=None,
    boilerplate_threshold=DEFAULT_BOILERPLATE_THRESHOLD,
):
    """Vectorized summary_verdict over many rows at once -- use this (not a
    per-row loop) for any full-corpus pass, local or live-batch."""
    summaries = [str(s or '') for s in summaries]
    sims = boilerplate_similarity_batch(summaries, template_embeddings, embedder)

    results = []
    for summary, sim in zip(summaries, sims):
        text = summary.strip()
        word_count = len(text.split())
        reasons = []
        if not text:
            results.append((False, ['empty summary']))
            continue
        if word_count < MIN_WORDS:
            reasons.append(f'too short ({word_count} words)')
        if SUMMARY_JUNK_RE.search(text):
            reasons.append('generic hosting/registrar/parked-domain phrase')
        if sim >= boilerplate_threshold:
            reasons.append(f'boilerplate (similarity {sim:.2f} to a known template)')
        results.append((len(reasons) == 0, reasons))
    return results


def keywords_verdicts_batch(
    summaries, keywords_lists, embedder=None,
    alignment_threshold=DEFAULT_KEYWORDS_ALIGNMENT_THRESHOLD,
):
    """Vectorized keywords_verdict over many rows at once."""
    sims = keywords_alignment_batch(summaries, keywords_lists, embedder)
    results = []
    for kws, sim in zip(keywords_lists, sims):
        kws = _coerce_keywords(kws)
        if not kws:
            results.append((False, ['no keywords provided']))
            continue
        reasons = []
        if sim < alignment_threshold:
            reasons.append(f'keywords not topically aligned with summary (similarity {sim:.2f})')
        results.append((len(reasons) == 0, reasons))
    return results


def summary_verdict(name, domain, summary, template_embeddings, embedder=None, **kwargs):
    """Single-row convenience wrapper -- fine for live per-result API
    evaluation, but use summary_verdicts_batch() for any full-corpus pass."""
    return summary_verdicts_batch([name], [domain], [summary], template_embeddings, embedder, **kwargs)[0]


def keywords_verdict(summary, summary_keywords, embedder=None, **kwargs):
    """Single-row convenience wrapper -- fine for live per-result API
    evaluation, but use keywords_verdicts_batch() for any full-corpus pass."""
    return keywords_verdicts_batch([summary], [summary_keywords], embedder, **kwargs)[0]


def trust_verdict(name, domain, summary, summary_keywords, template_embeddings, embedder=None):
    """Single entry point -- call this from both local batch scripts and live
    API evaluation code so the definition of "trustworthy" never drifts."""
    summary_ok, summary_reasons = summary_verdict(name, domain, summary, template_embeddings, embedder)
    keywords_ok, keywords_reasons = keywords_verdict(summary, summary_keywords, embedder)
    return {
        'summary_trustworthy': summary_ok,
        'summary_reasons': summary_reasons,
        'keywords_trustworthy': keywords_ok,
        'keywords_reasons': keywords_reasons,
    }


def build_corpus_trust_table(
    corpus_df,
    domain_col='domain',
    name_col='name',
    summary_col='summary',
    cache_path='result/trust_feature/corpus_trust_table.csv',
    audit_csv='result/summary_quality/summary_quality_audit.csv',
    embedder=None,
    force_recompute=False,
    batch_size=256,
):
    """Computes (and caches) summary_trustworthy for every row of a corpus, ONCE.

    Every retrieval notebook over the local 98,716-domain corpus (BM25, MiniLM,
    BGE, ...) shares the exact same domain/summary text, so the trust verdict
    is identical across all of them -- compute it here once and every notebook
    just loads the cached table by domain, instead of each one re-embedding the
    same ~98,716 summaries again. Only summary_trustworthy is included (not
    keywords_trustworthy): the keywords alignment check hasn't validated well
    enough yet to act on (see validate_against_llm_judgments()).
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute:
        return pd.read_csv(cache_path)

    embedder = embedder or get_embedder()
    template_texts, template_embeddings = load_boilerplate_templates(audit_csv, embedder=embedder)
    print(f'[TrustTable] {len(template_texts)} boilerplate templates loaded')

    domains = corpus_df[domain_col].tolist()
    names = corpus_df[name_col].tolist()
    summaries = corpus_df[summary_col].tolist()

    trustworthy, reasons_list = [], []
    n = len(corpus_df)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_results = summary_verdicts_batch(
            names[start:end], domains[start:end], summaries[start:end],
            template_embeddings, embedder,
        )
        for ok, reasons in batch_results:
            trustworthy.append(ok)
            reasons_list.append('; '.join(reasons))
        if end % (batch_size * 10) == 0 or end == n:
            print(f'[TrustTable] {end}/{n} companies scored...')

    out = pd.DataFrame({
        'domain': domains,
        'summary_trustworthy': trustworthy,
        'summary_reasons': reasons_list,
    })
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    n_bad = (~out['summary_trustworthy']).sum()
    print(f'[TrustTable] Saved {cache_path} -- {n_bad}/{n} ({100*n_bad/n:.1f}%) flagged untrustworthy')
    return out


def _prf(llm_bad, heur_bad):
    agreement = (llm_bad == heur_bad).mean()
    tp = (llm_bad & heur_bad).sum()
    fp = (~llm_bad & heur_bad).sum()
    fn = (llm_bad & ~heur_bad).sum()
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float('nan')
    return agreement, precision, recall, f1


def validate_against_llm_judgments(
    llm_judged_csv='result/llm_summary_trustworthiness/llm_judged_sample.csv',
    audit_csv='result/summary_quality/summary_quality_audit.csv',
):
    """Sweeps boilerplate/alignment similarity thresholds and reports
    precision/recall/F1 against the 300 LLM judgments from notebook 10, so a
    threshold gets picked by evidence rather than a guess."""
    llm_df = pd.read_csv(llm_judged_csv)
    embedder = get_embedder()
    template_texts, template_embeddings = load_boilerplate_templates(audit_csv, embedder=embedder)
    print(f'[Setup] {len(template_texts)} known boilerplate templates loaded/embedded')

    summaries = llm_df['summary'].tolist()
    keywords_lists = llm_df['summary_keywords'].tolist()
    llm_summary_bad = ~llm_df['summary_trustable'].astype(bool)
    llm_keywords_bad = ~llm_df['keywords_trustable'].astype(bool)

    boilerplate_sims = boilerplate_similarity_batch(summaries, template_embeddings, embedder)
    alignment_sims = keywords_alignment_batch(summaries, keywords_lists, embedder)

    print('\n' + '=' * 78)
    print('SUMMARY: boilerplate-similarity threshold sweep')
    print('=' * 78)
    print(f'{"threshold":>10} {"agreement":>10} {"precision":>10} {"recall":>10} {"f1":>10}')
    best_summary = (None, -1)
    for thresh in [0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97]:
        word_counts = llm_df['summary'].apply(lambda s: len(str(s).split()))
        junk_hit = llm_df['summary'].apply(lambda s: bool(SUMMARY_JUNK_RE.search(str(s))))
        heur_bad = (word_counts < MIN_WORDS) | junk_hit | (boilerplate_sims >= thresh)
        agreement, precision, recall, f1 = _prf(llm_summary_bad.values, heur_bad.values)
        print(f'{thresh:>10.2f} {100*agreement:>9.1f}% {precision:>10.2f} {recall:>10.2f} {f1:>10.2f}')
        if not np.isnan(f1) and f1 > best_summary[1]:
            best_summary = (thresh, f1)

    print('\n' + '=' * 78)
    print('KEYWORDS: alignment-similarity threshold sweep')
    print('=' * 78)
    print(f'{"threshold":>10} {"agreement":>10} {"precision":>10} {"recall":>10} {"f1":>10}')
    best_keywords = (None, -1)
    for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        heur_bad = alignment_sims < thresh
        agreement, precision, recall, f1 = _prf(llm_keywords_bad.values, heur_bad)
        print(f'{thresh:>10.2f} {100*agreement:>9.1f}% {precision:>10.2f} {recall:>10.2f} {f1:>10.2f}')
        if not np.isnan(f1) and f1 > best_keywords[1]:
            best_keywords = (thresh, f1)

    print(f'\n[Best] summary boilerplate threshold  : {best_summary[0]} (F1={best_summary[1]:.2f})')
    print(f'[Best] keywords alignment threshold   : {best_keywords[0]} (F1={best_keywords[1]:.2f})')

    out = llm_df[['domain', 'summary_trustable', 'keywords_trustable']].copy()
    out['boilerplate_similarity'] = boilerplate_sims
    out['keywords_alignment_similarity'] = alignment_sims
    RESULT_DIR = Path('result/llm_summary_trustworthiness')
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / 'trust_filter_v2_validation.csv'
    out.to_csv(out_path, index=False)
    print(f'\n[Saved] {out_path} ({len(out)} rows)')

    return best_summary, best_keywords


if __name__ == '__main__':
    audit_csv = Path('result/summary_quality/summary_quality_audit.csv')
    if not audit_csv.exists():
        build_boilerplate_audit(audit_csv=audit_csv)
    compute_query_contamination(audit_csv=audit_csv)
    compute_sensitivity_analysis(audit_csv=audit_csv)
    validate_against_llm_judgments()
