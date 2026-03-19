# Review (Round 2): Cycle 064 — Topology-preserving distillation (Item 42)

Verdict: **APPROVE**

## Round 1 Issue Resolution

### 1. Student capacity (was BLOCKING) — RESOLVED
Student changed from 2→16→16→1 to 2→32→1. Width matches teacher (32), structural difference is now depth (1 vs 4 layers). This is a cleaner test: the distillation challenge is depth compression, not working around a width bottleneck. Output-only floor at 70% acknowledges single-layer capacity limits on 3-turn spirals.

### 2. Surrogate gradient = scaled activation-MSE (was BLOCKING) — RESOLVED
PH surrogate abandoned entirely. Replaced with distance-matrix matching (Frobenius on pairwise distance matrices), which has a genuinely different gradient direction from pointwise activation-MSE. PH is now post-hoc evaluation only. Activation-MSE included as explicit baseline (method D), so if distance-matrix ≈ activation-MSE, the topology proxy is inert.

### 3. Dimension mismatch (was BLOCKING) — RESOLVED
Both teacher and student activations projected to 2D via PCA fitted on teacher. Fixed projection matrix is differentiable w.r.t. student params. Fallback to 3D or full 32D if PCA variance < 60%.

### 4. Missing baselines (was MODERATE) — RESOLVED
4-method comparison: output-only / distance-matrix / CKA-matching / activation-MSE. Jacobian-matching omitted but the activation-MSE baseline directly tests "any intermediate matching helps" vs "distance-preserving matching specifically helps," which is the key distinction.

## Remaining Notes (non-blocking)

- The 1-layer student on 3-turn spirals will likely cap around 75-85% regardless of distillation method due to limited linear regions. If all methods cluster within 2pp, the "inconclusive" criterion (< 2pp gain) correctly catches this. The moons dataset provides a second signal where the student should have ample capacity.
- Consider also recording a standalone student baseline (trained on ground truth labels, no teacher) to separate "teacher helps" from "topology helps."
