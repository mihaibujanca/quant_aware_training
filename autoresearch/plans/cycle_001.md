# Task
Literature survey of Balestriero's piecewise linear / spline theory of deep learning. Read 3 papers, extract implementable techniques for analyzing/compressing ReLU networks via their piecewise linear structure.

# Hypothesis
Verbatim from agenda:
- **Confirmed if**: identify ≥3 concrete, implementable techniques for analyzing or compressing networks via their piecewise linear structure
- **Refuted if**: papers are purely theoretical with no computational pathway applicable to model compression
- **Inconclusive if**: techniques exist but applicability to constructing compressed models is unclear

# Pre-flight Results
- P1 (duplicate): PASS — PAST EXPERIMENTS is empty, no prior survey done.
- P2 (dependency): PASS — no dependencies; this is the root item.
- P3 (concrete/testable): PASS — deliverable is a structured summary with implementability verdicts; criteria are clear.
- P4 (modifies program/prompts): PASS — only modifies agenda.md (moving item to PAST).

# Steps
1. **Fetch papers via web search.** Search queries:
   - `"A Spline Theory of Deep Learning" Balestriero 2018 arxiv`
   - `"Mad Max: Affine Spline Insights into Deep Learning" Balestriero Baraniuk 2020 arxiv`
   - `"From deep to shallow" ReLU network transformations arxiv`
   Read each paper (arxiv PDF). If "From deep to shallow" is unfindable after 3 query variants, note it and proceed with the other two.

2. **Extract structured notes per paper.** For each paper, record:
   - Core theoretical result (1-2 sentences)
   - How ReLU networks create affine spline partitions (mechanism)
   - Linear region enumeration: any algorithm or bound given? Complexity?
   - Compression / minimal representation results: any constructive method?
   - Implementability verdict: {ready, needs-work, theoretical-only}

3. **Synthesize across papers.** Produce a techniques inventory — a numbered list of concrete, implementable techniques found. For each: name, paper source, inputs required, outputs produced, estimated implementation difficulty (hours), and relevance to agenda items 3-8.

4. **Write output.** Save structured summary to `autoresearch/surveys/balestriero_spline_theory.md` (≤3 pages). Sections: Background, Per-Paper Notes, Techniques Inventory, Verdict (Confirmed/Refuted/Inconclusive with justification), Implications for Agenda Items 3-8.

5. **Update agenda.** In `autoresearch/agenda.md`, move item 1 from NEXT EXPERIMENTS → PAST EXPERIMENTS with: cycle number, commit hash (placeholder until commit), verdict, artifact path.

6. **Commit.** `[autoresearch] cycle 001: survey Balestriero piecewise linear / spline theory`

# Validation
- Survey file exists at `autoresearch/surveys/balestriero_spline_theory.md` and contains per-paper notes for ≥2 papers.
- Techniques inventory lists ≥0 techniques with implementability verdicts (the count determines Confirmed/Refuted).
- Verdict section explicitly states Confirmed, Refuted, or Inconclusive with reasoning tied to criteria.
- Agenda item 1 appears in PAST EXPERIMENTS section.

# Artifacts
- `autoresearch/surveys/balestriero_spline_theory.md` — structured literature summary
- Updated `autoresearch/agenda.md` — item 1 moved to PAST EXPERIMENTS

# Fallbacks
- If "From deep to shallow" is unfindable: proceed with the two confirmed papers; 2 papers is sufficient for a verdict.
- If arxiv PDFs are inaccessible via web fetch: search for HTML versions, blog summaries, or slides by the same authors that cover the same content. Note reduced confidence.
- If papers are purely theoretical (Refuted): document what theoretical tools exist and flag item 3 (region census) as needing an independent implementation approach not derived from these papers.
