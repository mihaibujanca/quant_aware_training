# Cycle 317 Design Review — Item 327

## Decision: APPROVE

## Strengths
- Plan implements item 327 verbatim: TRIPLES = C({1,2,3,10,11},3) = 10, ranks {1:2,2:2,3:24,10:5,11:22}, joint Adam LR=1e-4, SEEDS=[42], N_STEPS=4000, EVAL_EVERY=100, best-state restore, disjoint calib=all_ids[:200]/eval=all_ids[200:300], SEQ_LEN=128.
- Reuses the cycle 310 `LowRankCProj` template bit-identically (3072×r frozen A = U[:,:r]·diag(S[:r]), r×768 trainable B = Vh[:r,:], 768 trainable bias from W_proj.bias) so triple best_mean values are apples-to-apples with the cycle 310 pair matrix and the cycle 305 5-stack baseline.
- Ranks are the cycle-282-family single-block anchors exactly (b1 r=2 cycle 280, b2 r=2 cycle 285, b3 r=24 cycle 288, b10 r=5 cycle 289, b11 r=22 cycle 270), matching how cycle 310 and cycle 305 both installed simultaneous fits. No rank mixing, no new choices.
- Verdict tree matches the brief exactly with 3 branches and INCONCLUSIVE fallback.
- Cycle 310 + 305 baselines recorded in the JSON give a clean "pair → triple → 5-stack" progression for downstream readers (pair sum prediction vs measured triple delta).
- Wall ~40-50 min fits inside the normal experiment cap; doubling N_STEPS from cycle 310's 2000 → 4000 is justified because cycle 310's (1,11) 2000-step sanity gap was +0.70 PPL vs the 4000-step cycle 284 anchor — a meaningful budget-bounded undershoot at the pair level would be worse at the triple level where more trainable parameters compete for Adam updates.

## Risks (acknowledged in plan)
- Single seed (item 327 verbatim). The verdict bands (+5 PPL destructive; +2 PPL non-destructive) are wide enough to tolerate 1-seed noise for the CONFIRMED-b10b11-carries-destruction branch; the tighter +2 PPL non-destructive clause could flip on a borderline triple (cycle 310 showed (b1,b2) at +4.87, just below +5).
- The b10b11-carries branch requires BOTH: (a) 4/4 (b10,b11)-containing triples destruct AND (b) 6/6 non-(b10,b11) triples have delta ≤ +2 PPL. Given cycle 310's pair-level results — (b1,b2) +4.87, (b1,b3) +1.06, (b1,b10) +0.72, (b2,b3) +1.77, (b2,b10) +1.44, (b3,b10) +0.76 — any triple containing (b1,b2) (namely (1,2,3), (1,2,10), (1,2,11)) plausibly crosses +2 and breaks the clean branch. The verdict may land INCONCLUSIVE even if the substantive reading is clean.
- 4000 steps × 10 triples × heavier per-fit compute than pairs (3 blocks active) could run ~50 min, near the upper wall budget.

## Required execution-time addenda
1. For each triple, report which cycle 310 pair deltas it contains (3 per triple) and whether (b10,b11) is among them. This lets the audit layer factor "pair-inherited destruction" vs "new triple-level destruction".
2. Record the `init_ppl` for each triple; these are a direct, free check on whether simultaneous installation correctly chains the 3 LowRankCProj modules (init PPL should be dominated by the largest-delta pair in the triple's SVD truncation).
3. Assert `n_trainable == sum_{l in triple} ((r_l + 1) * 768)` exactly per triple.
4. On best-state restore: `|restored_ppl − best_ppl| < 1e-3` per triple.

## Approval
APPROVE.
