# Cycle 220 Review — Item 234 (Surgical bump removal)

Verdict: REVISE

## R1 Novelty
PASS. GELU_noneg and GELU_nobump as defined here are not tested in cycles 184/188/190/193. Complements cycle 190's hybrid via a different cut.

## R2 Falsifiability
PASS in structure (3 outcomes: CONFIRMED / REFUTED / INCONCLUSIVE). But the numeric thresholds are brittle — see R4.

## R3 Leakage
PASS. Single deterministic eval pass, model and data fixed, seed 42, no training, no stochastic augmentation. Patch/restore cleanly verified by a post-restore baseline re-eval.

## R4 Thresholds (ISSUE)
The criteria `noneg_ppl <= 25` and `nobump_ppl >= 35` are hardcoded against an assumed baseline of ~22.7 from cycle 184. If this cycle's measured baseline drifts (e.g. 24-26 as observed in other cycles on the same eval loader), the ≤25 cutoff becomes nearly unsatisfiable or incoherent. Cycle 184 measured 22.7 but cycle 206-218 baselines have varied. The 25 and 35 values encode "+10%" and "+55%" relative to 22.7 — the spec should reflect that intent.

REQUIRED change to step 9:
```
low_thresh  = baseline_ppl * 1.10   # within 10% of float
high_thresh = baseline_ppl * 1.55   # 55% degraded
if (noneg_ppl <= low_thresh and nobump_ppl >= high_thresh) or \
   (nobump_ppl <= low_thresh and noneg_ppl >= high_thresh):
    verdict = "CONFIRMED"
elif abs(noneg_ppl - nobump_ppl) <= 3:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"
```
Record `low_thresh`, `high_thresh`, and `baseline_ppl` in the JSON so the verdict is reproducible from the numbers.

Note: the agenda's verbatim "≤25 / ≥35" text can stay in the brief for traceability, but the executable verdict must be relative to the in-cycle baseline.

## R5 Geometric/methodological soundness
PASS.
- `gelu_noneg`: relu(gelu(z)) clamps the ~-0.17 negative dip to 0; matches gelu for z>0, ≈0 for z<0 (with smoothing near 0). Matches intent.
- `gelu_nobump`: continuity verified at z=±2 (kink, but continuous); asymptotes match gelu (identity above, 0 below). Replaces the [-2,2] "bump" window with slope-0.5 ramp. Matches intent.
- Sanity anchors at z ∈ {-3,-1,0,1,3} will catch definition errors before patching.
- Patch/restore protocol with post-restore baseline re-eval is the right safety net.
- Fallback to forward pre-hook on `mlp.c_proj` is reasonable if `mlp.act` swap fails.

## Action
REVISE step 9 to compute thresholds as `baseline * 1.10` and `baseline * 1.55` and log both the thresholds and the baseline in `cycle_220_surgical_bump.json`. Otherwise the plan is ready to execute.

---

# Round 2 Verdict: APPROVE

The revised plan at `autoresearch/plans/cycle_220.md` step 9 now computes:
```
low_thresh  = baseline_ppl * 1.10   # within 10% of float
high_thresh = baseline_ppl * 1.55   # >55% degraded
```
and uses these in the verdict logic (CONFIRMED / REFUTED / INCONCLUSIVE branches). The JSON schema in step 10 adds `"low_thresh"` and `"high_thresh"` fields alongside `baseline_ppl`, making the verdict reproducible from the recorded numbers.

All R1-R5 checks now PASS. Round-2 verdict: APPROVE. Ready to execute.
