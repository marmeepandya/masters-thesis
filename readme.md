# Semantic Company Retrieval at Scale

**Master's Thesis - Data Science University of Mannheim**. 
**Author:** Marmee Pandya

---

## Overview

A company that is a perfect match for a search query can still be invisible to a keyword-based search system, simply because its own description never repeats the query's exact wording. This project investigates whether embedding-based semantic retrieval, paired with approximate nearest-neighbour (ANN) search, can generate accurate candidate sets efficiently at scale, and whether it can outperform the keyword-driven production search system currently used by industry partner.

The work compares ten dense embedding models, a late-interaction architecture, and several hybrid retrieval pipelines over a corpus of company profiles drawn from the industry partner, a dataset of roughly 20 million verified, actively operated organisations across 232 countries and territories. A learned fusion ranker trained on the task's own queries is the strongest performer under a standard retrieval protocol. Because no independent human relevance judgements existed for this task at the outset, a large part of the project is dedicated to building an independent, multi-judge silver-labelling pipeline and testing how much of that initial result depended on using production's own output as ground truth. At the largest tested scale, ANN search matches exact search's retrieval quality within 0.5-2.5 percentage points while running 25-78x faster, a result confirmed with statistical testing rather than asserted from raw numbers alone.

The full write-up, including methodology, results, and discussion of the evaluation-circularity problem, is in [`Thesis_Report/thesis.pdf`](Thesis_Report/thesis.pdf).

---

## Data Availability

Company-level data and the production search results used as a comparison baseline are proprietary to industry partner and are **not included** in this repository.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Notebooks expect a `dataset/` directory populated with the GOI export (see Data Availability above) and a `.env` file with any required API keys (OpenAI, Anthropic, Google) for the embedding and LLM-judging notebooks.

The HyDE pipeline and the original LLM judge pair notebooks also expect a local [Ollama](https://ollama.com) server running `llama3.1:8b` (`OLLAMA_BASE_URL`, default `http://localhost:11434`); this is a separate, non-Python installation and is not covered by `pip install`.

---

## Key Results

- A learned fusion ranker reaches **Recall@1,000 = 0.823** on a corpus of 98,716 companies under the standard evaluation protocol.
- Retrained against an independently built, multi-judge silver standard, production's own results miss **27.3%** of independently verified relevant companies.
- On a pool of companies production had never ranked, the strongest single embedding model, **Linq-Embed-Mistral**, recovers **88.5%** of them.
- At the largest tested scale (397,025 companies, ~2% of  full database), ANN search matches exact search's retrieval quality within 0.5-2.5 percentage points while running **25-78x faster**.

See [`Thesis_Report/thesis.pdf`](Thesis_Report/thesis.pdf) for full methodology and discussion.

---

## License

 Code is shared for reproducibility; the underlying GOI dataset remains proprietary.
