1. Structural: "Chapter 4 has too many sections" (the dominant, most-repeated theme)
- Explicit note on the TOC: "4 has too many sections! consider grouping e.g. by your contribution." Later, at the Chapter 4 opening: "you already group [this] in the text! Regroup all sections into 4-5 main sections," with an arrow pointing at the existing "Roadmap" paragraph's five Phases, i.e., turn Phase I–V (already described in prose) into the actual section structure, rather than 26 flat sections.
- Also flagged: never leave a section with exactly one subsection (4.4 has only 4.4.1). Several sections called out as "very short, try fusing them" (ANN scaling section, several 4.19–4.22 subsections). Table 4.1's "Phase" column singled out directly: "sections" — use these.
- Figure 4.1 should appear before the paragraph explaining it, not after.
- Terminology (silver/gold/bronze) should be defined earlier — flagged at Table 4.5 ("where gold?") and again at the TOC/Bibliography ("Eidesstattlich?" — questioning the German declaration title too).

2. Recurring gap: Findings state what happened, rarely why. Explicit meta-comment: "I like the clear structure between method → results → finding? Findings often miss an explanation." Called out specifically for "why is MiniLM best?" and "why does quality not climb with size?" — and conversely, praised the SimLM and Reranker Ablation sections' "Interpretation" paragraphs as the model to replicate more broadly.

3. Model/literature recency. MiniLM and BGE flagged "old," literature review called "missing literature after 2024," and judge model choice ("why old model? GPT5.6, Luna?"). Independent skepticism that Claude Haiku 4.5 vs GPT-4o-mini are comparably reliable ("Haiku 4.5 is significantly better than 4o-mini" — pushing back on the blind-calibration conclusion).

4. Active learning — independently asks Ralph's exact question: "How is this active learning?" with no other context, a second, independent reviewer landing on the same concern is worth knowing.

5. Writing style. Prefers active voice/"we" over passive in Methods sections; likes bullet points (Contributions/Thesis-structure paragraphs feel dense); one sentence flagged directly as "reads like AI." Sharpest comment: the footnote justifying why gold_label wasn't renamed in code ("risked introducing bugs") was called "bad! nothing Claude couldn't fix :(" and "jupyter as prod: double bad!" — direct instruction: cut the coding-practice excuse, just state the naming choice.

6. Missing ML formalization. Wants an explicit input/output/loss statement for both the learned fusion ranker (4.18) and the neural network ranker (4.21.2), plus the MLP's activation function.
                                                                                                                                                        7. Completeness/technical requests (mostly minor, several alriant (PCIe/SXM), CPU model, benchmark comparison for baselinemodel choices, proof for the "network-latency-dominant" OpenAI claim, cost comparison of OpenAI API vs A100 GPU, a results table for Section 4.13       (currently prose-only), justification for the rank-bucket thr, a clearer/shown cost-estimation methodology for the $22/$28kjudging-cost figures, converting Figure 4.1 to TikZ, formal Recall@k formula, RRF as a displayed equation, and moving the Precision/Recall relationship math to where metrics are first defined (4.1.2) rather than a— interesting, since that's almost exactly where I placed itafter your last review pass, so this is a vote to move it even earlier.

8. What's landing well (worth preserving, not just noting problems): the abstract-writing funnel diagram tip, Chapter 2 ("Nice!"), the AI-summary data-quality paragraph in the Introduction, Section 4.12 Corpn!! I really like these small intros"), Section 4.22's opening, and a genuine "a bit surprised by that" reaction to the corrected MiniLM-wins finding in Table 4.2, external, independent confirmation that this result
is notable and lands as intended.

- The methods-detail point has a second motive: expanding the methods explanations isn't just for reader clarity, he's explicitly saying it would also organically grow the page count, reducing pressure to pad the thesis with additional experiments. That's a useful reframe, "more detail" is doing double duty.
- The interpretation gap has a shape: it's not that interpretation is missing everywhere, it's that later sections (SimLM, Reranker Ablation, the ones I saw praised in the PDF) already do it well, while earlier sections don't. So the fix is applying the pattern already working later in the chapter backward, not inventing a new pattern.
- The active-voice point is more specific than style preference: he wants methodological choices explicitly owned as this thesis's decisions ("we did this") rather than framed as inherited defaults ("this is standard practice, everyone does it"). That's a framing/ownership issue as much as a grammar one.
- Shorter sentences, and otherwise no language concerns.


Structural changes (the big one):
1. Split the current Chapter 4 into two chapters: Methodology (dataset, evaluation protocol, computational setup, what each baseline/experiment is and why) and Results (the numbers, tables, findings, and — per the recurring "why" feedback — interpretation, organized under the 5 existing Phases as actual \sections, with the current ~24 sections demoted to \subsection or \paragraph level so they stop cluttering the ToC).
2. Move Literature Review, either right after the Introduction, or to the very end of the thesis (Ralph offered both).
3. Never a single subpoint under a point (already flagged repeatedly — I'll audit for this everywhere, not just Chapter 4).
4. Add explicit Introduction subsections for research questions / thesis structure (formalize content that's mostly already there in prose).

Content changes (smaller, but numerous):
5. Systematic "why" pass: add interpretation to the earlier sections that currently lack it, matching the pattern already used later in the chapter.
6. Active voice / explicit ownership pass ("we did X" instead of "X was done" or "X is standard practice").
7. Cut the gold_label naming footnote's "risked introducing bugs" justification, per your friend's sharp note.
8. Expand method explanations (BM25, dense retrieval, ANN, etc.) for a reader who doesn't already know them, without just duplicating the Literature Review.

New task (resolves Ralph's grade concern directly):
9. Prepare a 10–15 query stratified export for you to self-label blind (following the existing guidelines), so LLM-human kappa can finally be reported, independent of whether Kerem/Manav ever respond.
