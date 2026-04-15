# Semantic Company Retrieval at Scale
### Master's Thesis — Data Science, University of Mannheim
**Author:** Marmee Pandya  
**Industry Partner:** [Istari.ai](https://istari.ai)  
**Date:** March 2026

---

## Project Overview

Modern company intelligence platforms maintain datasets containing millions of company profiles. Despite the availability of large amounts of structured and unstructured data, retrieving relevant companies remains challenging. Current workflows often rely on manually configured filters and keyword-based queries, which require multiple iterations before satisfactory results are obtained.

This thesis investigates whether **embedding-based semantic retrieval** combined with **approximate nearest neighbor (ANN) search** can efficiently generate accurate top-k candidate company sets from large-scale datasets while maintaining low query latency.

---

## Thesis Goal

Design, implement, and evaluate an efficient embedding-based candidate retrieval system for large-scale company data, using Istari's **Global Organization Index (GOI)** — a dataset of ~20 million verified and active organizations across 232 countries.

### Research Goals

| Goal | Description |
|------|-------------|
| **Goal 1** | Develop a semantic retrieval pipeline that encodes company descriptions into dense vector representations and supports scalable top-k retrieval using ANN search |
| **Goal 2** | Systematically compare the proposed approach against the production search system and a BM25 baseline using standard retrieval metrics (Precision@k, Recall@k, NDCG@k) and efficiency measures |
| **Goal 3 *(optional)*** | Explore similarity-based explanatory techniques such as embedding neighborhood analysis and visualization of retrieved company clusters |

---

## Dataset

The thesis uses Istari's **Global Organization Index (GOI)**, built through a multi-step validation pipeline:

- ~400M organizations collected from national registries worldwide
- ~40M attributed to identifiable web domains
- ~20M verified as actively operated — this is the final dataset

Each organization record includes an AI-generated `summary`, `keywords`, `nace_code`, `organization_type`, `organization_size`, and location fields.

---

## Baselines

1. **Production Search Baseline** — Istari's existing keyword-based search system
2. **BM25 Baseline** — Classical probabilistic retrieval model
3. **Embedding-Based Literature Baseline** — A recent dense retrieval method from scientific literature

---

## Evaluation

- **Retrieval Quality:** Precision@k, Recall@k, NDCG@k
- **Efficiency:** Query latency, throughput, memory consumption
- **Error Analysis:** Qualitative analysis of vocabulary mismatch and failure modes

---

*More sections will be added as the project progresses.*