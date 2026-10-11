# Cycle 215 Audit — Item 225

**Item**: Iterative block-by-block Adam correction — sequential refinement
**Claimed outcome**: INCONCLUSIVE (final_ppl_mean=47.253 ± 0.047 across 3 seeds)
**Audit status**: PASS

## Phase 1 — Verdict check
Agenda criteria:
- Confirmed if ≤46.0
- Refuted if ≥60.0
- Inconclusive if 46–60

final_ppl_mean = 47.253 ∈ (46.0, 60.0) → **INCONCLUSIVE** matches the criteria. Verdict stands.

## Phase 2 — Quality checks

### G1 Implementation correctness — PASS
- Sequential order [1, 4, 7] respected (iteration over `BLOCKS` as-is).
- Freeze pattern: `for p in model.parameters(): p.requires_grad_(False)` then unfreeze only `mlp.W_out`, `mlp.b_out` of the current block — correct.
- Fresh `torch.optim.Adam` per block, LR=1e-4 (matches cycle 206 baseline).
- Trainable param count asserted: `768*K + 768 = 394,752`.
- Eval every 100 steps on held-out `eval_ids`, best-state snapshot+restore per block.
- Collapse guard: if `best_ppl > prev_ppl` → restore `pre_state`. No collapses observed (flag false for all 9 block-seeds).
- `calib_ids_base` and `eval_ids` are disjoint slices of the same tokenised wikitext (`all_ids[:N_CALIB]` vs `all_ids[N_CALIB:N_CALIB+N_EVAL]`) — same as cycle 206, no leakage.
- `W_fc`/`b_fc` registered as buffers (not parameters) in `CompressedMLP` — they cannot be accidentally trained.
- No double-training: `best_state` is snapshotted before the next block begins and restored before moving on.
- Eval uses held-out PPL set in the same way as cycle 206.

### G2 Reproducibility — PASS
- 3 seeds run: [42, 137, 2024]. Per-seed results distinct (final=47.210, 47.232, 47.318).
- `init_ppl` identical across seeds (std ≈1e-6) because topk+lstsq init is deterministic given data — only Adam trajectory is seed-dependent. Plan explicitly predicted this.
- Hyperparameters recorded in `config` block of the JSON.
- Matches cycle 206 on `N_CALIB=200`, `N_EVAL=100`, `SEQ_LEN=128`, `LR=1e-4`, `EVAL_EVERY=100`.
- Joint baseline reference 46.556 comes from cycle 210 step 4000.

### G3 Interpretation — PASS
- "Lost by 0.7 PPL to joint": 47.253 − 46.556 = 0.697 ✓.
- Per-block trajectory monotone improvement: 61.746 → 51.486 → 48.503 → 47.253 ✓.
- Block-7 restored PPL = final PPL (47.253) consistent ✓.

### G4 Counterintuitive — N/A
Joint coordinate optimisation beating sequential by a small margin is expected behaviour.

### G5 Learning quality — PASS
`learned` field is substantive: identifies that sequential best-state restoration freezes each block against downstream blocks' subsequent trajectories. `next` field is concrete: "sequential + joint-polish pass" or "sequential without best-state restore, shared LR schedule".

### G6 Method informativeness — PASS
Gap to CONFIRMED = 1.25 PPL; gap to REFUTED = 12.75 PPL. Cross-seed std ≈0.05 → the mean is well-resolved; INCONCLUSIVE is a genuine narrow-miss result, not a noisy signal.

## Minor observation (non-blocking)
For almost every block-seed, `best_step == 2000` (the step budget). Training was still improving when it stopped. This means "2000 steps per block" may be under-trained — a longer budget (or the suggested joint polish) could plausibly close the 0.7 PPL gap. This is already captured in the `next` field of the manifest and does not affect the verdict.

## Guardrail flags
none

## Decision
PASS. No files modified.

## Recommendations for next cycle
1. Run sequential + joint-polish (all 3 W_outs trained jointly for ~500 steps after sequential init) to see if the 0.7 PPL gap closes.
2. Alternative: extend `N_STEPS_PER_BLOCK` to 4000 and re-check whether the within-block curves plateau before restore (they may still be improving).
3. If neither closes the gap, accept joint Adam as the dominant recipe and close this line of investigation — the finding "sequential ≈ joint − 0.7 PPL with identical parameter count" is already informative.
