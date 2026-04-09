# Cycle 328 Review

## Plan summary
Spectrum preflight on b12's `blocks.12.mlp.proj.weight` (640x320), then conditional rank-7 LASER retrain. Gate: top-7 energy fraction must exceed 10% to proceed with retrain. Three possible ranks if gate passes: {3, 7, 14}.

## R1: Premise validity
Sound with appropriate caveats. Cycle 327 (item 340) proved eff_rank is not a proxy for weight SVD compressibility. The plan correctly adds a spectrum gate instead of assuming b12 will work because of low eff_rank. The structural argument (b12 weight is 640x320 vs b3's 640x1920, squarer = possibly more concentrated) is plausible but explicitly flagged as speculation. The gate resolves this empirically in seconds.

## R2: Measurement soundness
BPB is deterministic given fixed val tokens, same as cycle 327. The spectrum check (sum-of-squared singular values) is a standard, well-defined measurement. The plan includes a Frobenius norm cross-check, which is the right validation. Energy fraction = sum(S[:k]^2) / sum(S^2) is the correct formulation.

## R3: Statistical adequacy
Single seed for retrain (if triggered). Same rationale as cycle 327: 3-rank sweep provides robustness, effect size thresholds are above typical seed noise. For the spectrum check, no statistics needed — SVD is deterministic.

## R4: Method-architecture alignment
**Weight shape**: b12 has hidden=320, so `CastedLinear(hidden=320, dim=640)` stores weight as `(640, 320)` in MLX's `(out, in)` convention. Wait — need to check: MLX `nn.Linear(input_dims, output_dims)` stores weight as `(output_dims, input_dims)`. For `mlp.proj = CastedLinear(hidden, dim)` where hidden=320 (input to proj) and dim=640 (output), weight shape is `(640, 320)`. SVD of (640, 320): U(640,320) S(320) Vh(320,320) with full_matrices=False. A_frozen = U[:,:7] @ diag(S[:7]) shape (640,7), B_trainable = Vh[:7,:] shape (7,320). Forward: `x @ B.T @ A.T` where x is (..., 320), gives (..., 7) then (..., 640). Correct.

**Concern**: The existing `pgolf_laser_retrain.py` was written for b3 (weight 640x1920). For b12 (weight 640x320), the SVD shapes change: U goes from (640,640) to (640,320), S from (640,) to (320,). The script should handle this if it uses `full_matrices=False` and indexes `[:r]` generically. But worth verifying during execution that min(640,320)=320 >= rank 14.

## R5: Known pitfall scan
- MP3 (underdetermined): If retrain runs, rank-7 B has 7x320=2240 params. 512K tokens / 2240 = 229:1. Well above minimum.
- MP4 (domain mismatch): Same cross-architecture concern as 327, but now gated by spectrum check.
- Gate thresholds: 10% is reasonable. b3 had 2.6% at rank-5, so 10% would clearly separate flat from concentrated. 50% is a high bar — plan could use 20% as the "likely to work" threshold, but the gate at 10% is the critical decision point.

## Issues
- **Minor: weight shape in brief says (320, 640) but should be (640, 320)**. The brief says `proj.weight` is "320x640" but MLX stores as (out_dims, in_dims) = (640, 320). The plan's Step 1 and R4 analysis use the correct shape. Non-blocking — exec agent will see actual shape from the checkpoint.
- **Minor: rank-14 sensitivity test**. rank-14 out of 320 total singular values = 4.4% of directions. This is a reasonable sensitivity point. But if the spectrum IS concentrated and rank-7 captures >80% energy, rank-3 is the more interesting test (how low can we go?). Plan already includes both. Good.

## Verdict
APPROVE (round 1). The spectrum gate is the right modification given cycle 327's refutation. Plan is technically sound, shapes verified, fallbacks adequate. The gate turns a potentially wasted 15-minute retrain into a 10-second check when the spectrum is flat.
