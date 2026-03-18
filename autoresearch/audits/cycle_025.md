# Audit — Cycle 025

## Item
33 — GELU → 3-piece PWL approximation fidelity — GPT-2

## Phase 1: Verdict Check

Claimed: **refuted** — perplexity increases > 20% for all piece counts ≤ 10.

Criteria: "Refuted if: perplexity increases > 20%, indicating that GELU's smooth transition zone is functionally critical at the LLM scale."

Evidence: 3-piece → +5466%, 5-piece → +2462%, 10-piece → +57.8%. All > 20%. Only 20-piece (+1.1%) is viable.

**Verdict correct.** Clear refutation.

## Phase 2: Quality Checks

### G1: Methodology
- Full model GELU replacement — correct approach.
- Block-by-block analysis at K=3 — good diagnostic.
- Range [-5, 5] is wider than toy's [-3, 3] — appropriate.
- No retraining — correct per agenda specification.

### G3: Tautology Check
Not tautological. The sensitivity to GELU smoothness at LLM scale is a genuine finding.

### G5: Counterintuitive
The much worse result on GPT-2 vs toy (K=10: +58% vs +2.6%) is notable. GPT-2's FFNs operate at much higher precision requirements than toy classifiers.

## Status: PASS

original_outcome: refuted
audited_outcome: refuted
guardrail_flags: none
reason: Clear refutation. GELU smoothness is critical at LLM scale. Partition-based theory requires ≥20 pieces to be a viable approximation, making it impractical.
recommendations: This closes the partition-theory-for-GELU line. Compression of GELU transformers must work directly with the smooth activation, not through PWL approximation. Procrustes-based depth pruning (cycle 018) remains the most viable no-SGD approach.
