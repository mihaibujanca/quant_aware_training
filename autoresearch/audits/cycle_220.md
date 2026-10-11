# Cycle 220 — Item 234 audit (Surgical bump removal)

**Status: PASS** — literal verdict INCONCLUSIVE is correct per the criterion.

## Numbers
- baseline_ppl = 45.064 (in-cycle); low=49.57, high=69.85.
- gelu_noneg_ppl = 2,093.296 (+4,545%).
- gelu_nobump_ppl = 11,455.595 (+25,321%).
- sanity_ppl_after_restore = 45.06370... (exact match → patch/restore clean).

## Verdict logic check
- CONFIRMED clause: requires one variant ≤49.57. Neither is. False.
- REFUTED clause: |2093 − 11455| > 3. False.
- → INCONCLUSIVE per the literal rules. Correct.

## Implementation checks (G1)
- GeluNoneg = relu(gelu(z, tanh)): correct. For z<0 → 0; for z≥0 → GELU(z) (bump preserved).
- GeluNobump: continuity at z=±2 verified (z=2 → 0.5·4=2; z=−2 → 0).
- All 12 `transformer.h[i].mlp.act` patched with a single shared nn.Module (no params, safe).
- Post-restore baseline PPL equals the pre-patch baseline to float precision — patching has no side effects.
- Eval protocol reuses cycle 206's loader (wikitext-2 validation, N=200, SEQ_LEN=128, seed 42).

## Baseline discrepancy (documented, not blocking)
- Cycle 184 reported baseline ≈22.7; cycle 215 reported ≈53.9; this cycle 45.06. Plan used relative thresholds against the in-cycle baseline so the discrepancy does not invalidate the decision, but the eval-set construction drift across cycles is a recurring hazard worth tracking (G1).

## Scientific interpretation (G3 — partial flag)
The summary says "neither hits the clean-separation thresholds; the load is shared" — that phrasing undersells the result. A +4,545% / +25,321% collapse is not "load sharing", it is **double necessity**: removing either sub-region catastrophically breaks the model. The manifest's `learned` field captures the asymmetry (nobump ~5.5× worse than noneg) correctly.

## Why noneg is ×46 baseline when cycle 184 ReLU+bias was only +88%
This is the key methodological gap. `gelu_noneg(z) = relu(gelu(z))` has **no first-moment correction**. The GELU negative leak is small (≈−0.17 max) but is applied on ~half the tokens, so zeroing it introduces a **positive mean shift of ~0.08** per channel that cascades across 12 blocks. Cycle 184's ReLU+bias explicitly compensated this shift; that is why it degraded only +88%. The plan's hypothesis ("one sub-region is near-linearizable") is not tested fairly without this compensation.

## Counterintuitive check (G4)
Surprise is resolved: the catastrophe is expected once you notice the missing mean-shift correction. Consistent with cycles 184/188/190/191 showing constant-bias removal is already costly per-block and compounds across depth.

## Recommendations
1. **Rerun with mean-corrected variants**: `gelu_noneg_c(z) = relu(gelu(z)) − E[relu(gelu(z)) − gelu(z)]` (per-channel, measured on the eval activations) and similarly for nobump. This tests the *shape* hypothesis without the first-moment confound.
2. **Reframe the hypothesis**: "decomposing GELU into independently-linearizable sub-regions" is likely incorrect; results support "each sub-region individually necessary under all-12 substitution". A depth-gated test (1, 2, 4 block partial patches — already proposed in the manifest `next` field) would quantify where the destructive cascade begins.

## Guardrail flags
- G1 (minor): baseline drift vs cycle 184 unexplained but used consistently.
- G3: summary phrasing underplays the double-necessity finding.
- G4: counterintuitive result has a concrete methodological explanation (missing mean compensation).

## Interesting
Yes. Double-necessity of the two sub-regions + asymmetry (nobump ~5.5× worse than noneg) is a genuinely informative null. Combined with cycles 188/217 (constant bias dominates Δ), it suggests GELU's functional content lives at two distinct scales: a per-block constant that cycles 188/191 can absorb, and per-token curvature that the nobump test kills catastrophically.
