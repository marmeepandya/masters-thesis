# Semantic Company Retrieval at Scale

**Master's Thesis — Data Science, University of Mannheim**
**Author:** Marmee Pandya
**Advisor:** Prof. Dr. Ralph Peeters
**Industry Partner:** [ISTARI.AI](https://istari.ai)

---

## Overview

A company that is a perfect match for a search query can still be invisible to a keyword-based search system, simply because its own description never repeats the query's exact wording. This project investigates whether embedding-based semantic retrieval, paired with approximate nearest-neighbour (ANN) search, can generate accurate candidate sets efficiently at scale, and whether it can outperform the keyword-driven production search system currently used by ISTARI.AI.

The work compares more than ten dense embedding models, a late-interaction architecture, and several hybrid retrieval pipelines over a corpus of company profiles drawn from ISTARI's Global Organization Index (GOI), a dataset of roughly 20 million verified, actively operated organizations across 232 countries. A learned fusion ranker trained on the task's own queries is the strongest performer under a standard retrieval protocol. Because no independent human relevance judgements existed for this task at the outset, a large part of the project is dedicated to building an independent, multi-judge silver-labeling pipeline and testing how much of that initial result depended on using production's own output as ground truth. At the largest tested scale, ANN search matches exact search's retrieval quality within a small margin while running significantly faster, a result confirmed with statistical testing rather than asserted from raw numbers alone.

The full write-up, including methodology, results, and discussion of the evaluation-circularity problem, is in [`Thesis_Report/thesis.pdf`](Thesis_Report/thesis.pdf).

---

## Repository Structure

```
.
├── 01_..._76_...ipynb       # Numbered analysis pipeline (see below)
├── 66_..._69_...py          # Standalone scripts mirroring the corresponding notebooks, for batch/SLURM runs
├── result/                  # Per-notebook output artefacts (metrics, figures, intermediate tables)
├── dataset/                 # Not included in this repository — see Data Availability
├── Thesis_Report/           # LaTeX source and compiled PDF of the thesis
├── Thesis_Proposal/         # Original thesis proposal
├── assessor_guidelines.md   # Guidelines given to human relevance assessors
├── requirements.txt         # Python dependencies
└── sync_to_drive.sh         # Syncs result/ artefacts to Google Drive backup
```

### Pipeline Overview

The notebooks are numbered in the order they were run and roughly fall into the following stages:

| Range | Stage |
|-------|-------|
| `01`–`05` | Baseline dense retrievers (BM25, BGE, MiniLM, OpenAI, Nomic) |
| `06`–`30` | Hybrid retrieval, reranking, and fusion ranker experiments |
| `32`–`43` | Evaluation-set construction, LLM judge ensembles, and silver-label calibration |
| `44`–`64` | Scaling embeddings and retrieval to the full corpus, reranker fine-tuning, ANN tuning |
| `65`–`76` | Production-independent evaluation, inter-rater agreement, and significance testing |

Each notebook writes its outputs to a correspondingly named folder under `result/`.

---

## Data Availability

Company-level data from ISTARI's Global Organization Index (GOI), the evaluation queries, and the production search results used as a comparison baseline are proprietary to ISTARI.AI and are **not included** in this repository.

Result outputs, trained model artefacts, and other intermediate files that do not expose raw company data (relevance labels, evaluation metrics, figures) are backed up separately at this [Google Drive folder](https://drive.google.com/drive/u/3/folders/1dRm6Oj8p3JlzWZsJoMJukuh19AkFJi1m), so the analysis can be reproduced conditional on independent access to the underlying GOI data.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Notebooks expect a `dataset/` directory populated with the GOI export (see Data Availability above) and a `.env` file with any required API keys (OpenAI, Anthropic, Google) for the embedding and LLM-judging notebooks.

---

## Key Results

- A learned fusion ranker reaches **Recall@1,000 = 0.823** on a corpus of 98,716 companies under the standard evaluation protocol.
- Retrained against an independently built, multi-judge silver standard, production's own results miss **27.3%** of independently verified relevant companies.
- On a pool of companies production had never ranked, the same pipeline recovers **88.5%** of them.
- At the largest tested scale (397,025 companies, ~2% of ISTARI's full database), ANN search matches exact search's retrieval quality within 2.5% while running up to **78x faster**.

See [`Thesis_Report/thesis.pdf`](Thesis_Report/thesis.pdf) for full methodology and discussion.

---

## License

This repository accompanies an academic thesis produced in collaboration with ISTARI.AI. Code is shared for reproducibility; the underlying GOI dataset remains proprietary to ISTARI.AI.
