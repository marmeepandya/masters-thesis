# Company Search Relevance: Assessor Guidelines

This document explains how to label company search results for relevance. The same criteria are also given to the LLM judges used elsewhere in this evaluation, so your labels and the model's labels can be meaningfully compared against each other.

## The task

For a given search query (e.g. "software companies", "wealth management firms"), you will see a list of candidate companies. For each one, read the company name, summary, and any other fields shown, and decide how relevant that company is to the query.

## The scale

Use exactly one of three labels:

**2 = Highly relevant.** The company is a strong, direct match for the query. If someone searched for this query, this is a company they would expect and want to find.

**1 = Partially relevant.** The company is related to the query but is not a strong direct match. This includes companies that operate in an adjacent space, only partially fit the query's intent, or fit the query for one line of business among several unrelated ones.

**0 = Not relevant.** The company has nothing meaningful to do with the query. This includes companies that merely mention a query-related term in passing without it describing what they actually do.

## Worked example: the query "software companies"

This exact query has already surfaced a real failure mode in this project worth calibrating against: investment firms and venture capital companies whose summaries mention "software" because they describe their investment portfolio (e.g. "we invest in software startups"), without the company itself being a software company.

- A company that builds and sells software products or services → **2**
- A company that provides IT consulting or custom software development as one of several services → **2** or **1**, depending on how central software is to what they do
- A venture capital firm whose portfolio includes software companies, but which is not itself a software company → **0**
- A manufacturing company that mentions using internal software tools → **0**

## Edge cases

- **Unreachable domain / placeholder website / no verifiable content.** Skip the row (leave it blank) rather than guessing. Do not label based on the company name alone if the summary gives you nothing to go on.
- **Holding companies / conglomerates.** Judge based on whether the specific business description shown fits the query, not on speculation about subsidiaries not mentioned in the summary.
- **Non-English company names or summaries.** Judge on the same criteria; translate mentally if needed. Do not penalize a company just for not being English-language.
- **Company plausibly fits the query but the summary is thin or generic.** Use your judgment about whether the available evidence, even if brief, supports a specific label, rather than defaulting to 1 for every ambiguous case. If genuinely 50/50, 1 is the reasonable default.
- **A company that used to do X but has since pivoted away from it.** Judge based on what the summary says the company currently does.

## How to submit

Fill in the `relevance_label` column in the spreadsheet you were sent, using only 0, 1, or 2 (leave blank if you're skipping a row per the edge cases above). Please label top-down for as long as time allows; rows are already ordered so that the most uncertain part of the ranking comes first. There's no need to sort, filter, or reorder anything, and no need to explain your reasoning per row, just the label.

If you find a case this document doesn't clearly cover, use your best judgment and, if you have a moment, note briefly what was ambiguous about it, that kind of feedback is useful for improving these guidelines.
