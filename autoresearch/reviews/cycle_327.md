# Cycle 327 Review

## Plan summary
LASER rank-{2,5,10} weight-SVD of `blocks.3.mlp.proj.weight` (640x1920) on parameter-golf 13L_640d_322 + B-factor retrain (500 Adam steps, lr=1e-4). First cross-project backward pass. Two scripts: pgolf-side retrain script, qat-side orchestrator.

## R1: Premise validity
Sound. Cycle 318 measured eff_rank(b3) = 4.89 with 78% top-1 eigenvalue fraction. Cycle 280 established the LASER recipe on GPT-2 GELU b1. Testing transfer to a different architecture (ReLU², variable mlp_mult, no bias) is the right next step. Dependencies 335/336 were REFUTED for universality but the plan correctly notes those concern multi-checkpoint stability, not validity for this specific checkpoint. The premise stands.

## R2: Measurement soundness
BPB is a clean, deterministic metric (given fixed val tokens). The plan reports three BPB values per rank (baseline, b_init, retrained), which gives proper controls. The validation checks (baseline matches training summary, b_init worse than baseline, monotonic in rank) are well-chosen. No activation-space measurement issues.

## R3: Statistical adequacy
Single seed per rank. For SGD retrain, seed sensitivity is real (optimizer noise, data ordering). However: (a) the 3-rank sweep provides robustness -- if the BPP-vs-rank curve is smooth and monotonic, seed variance is unlikely to flip the verdict, (b) the effect size thresholds (0.001 BPB) are well above typical seed noise for 500-step Adam. Acceptable. If the result is borderline (CONFIRMED-partial zone), a follow-up with 3 seeds at rank-5 would be warranted.

## R4: Method-architecture alignment
**Monkey-patching concern**: The plan proposes `model.blocks[b].mlp.proj.__call__ = ...` and `model.blocks[b].mlp.proj.B = B_init` for gradient flow. This should work in MLX: assigning `B` as an attribute on the module makes it visible in the parameter tree, and `nn.value_and_grad` differentiates through all non-frozen parameters. The fallback (explicit `mx.grad`) is correctly identified. Verified that `nn.value_and_grad` is used extensively in the pgolf codebase (train_gpt_mlx.py, correction_ce.py, sanity_checks.py).

**CastedLinear.__call__** does `x @ self.weight.T`. The plan replaces this with `x @ B.T @ A.T` where A=(640,r) and B=(r,1920). Dimensionally: `x@B.T` = (...,r), then `@A.T` = (...,640). This equals `x @ (A@B).T = x @ W.T`. Correct.

**Weight shape**: Confirmed. `CastedLinear(hidden=1920, dim=640)` stores weight as `(640, 1920)` via MLX's `nn.Linear(in, out)` convention `(out, in)`. Plan says `(640, 1920)`. Correct.

## R5: Known pitfall scan
- MP1-MP2: Not applicable (no threshold detection, no activation-regime issues).
- MP3 (underdetermined): 512K tokens for 9.6K params = 53:1 ratio. Well above the 2:1 minimum.
- MP4 (domain mismatch): Explicitly the point of the experiment. The plan acknowledges this (GPT-2 GELU -> pgolf ReLU²) and defines appropriate verdict thresholds.
- MP5 (insufficient capacity): The base model is trained and achieves 1.3474 BPB. Capacity is not the question.
- MP6 (verdict misclassification): Thresholds are concrete (0.001, 0.005, 0.01 BPB). BPB varies continuously with rank. No risk of uninformative metric.
- MP7-MP8: Not applicable (not measuring post-LayerNorm geometry or representation dimension).

## Issues
- **Minor: missing `cwd=PGOLF_ROOT`** in the qat-side subprocess call. All existing pgolf cross-project scripts (cycles 318, 319, 323, 335) pass `cwd=PGOLF_ROOT` to `subprocess.run` so that `Hyperparameters().data_path` (default `./data/datasets/...`) resolves correctly. The plan's Step 2 description doesn't mention this. The exec agent will likely follow the existing pattern, but the plan should state it explicitly. Non-blocking -- the exec agent has the cross_project doc as Step 0.
- **Minor: `TokenStream` vs `TokenLoader`**. The plan says `TokenStream(hp.train_files)`. The training script uses `TokenLoader` (line 1696), and `TokenStream` is a different class (line 348). Both exist and `TokenStream` is the simpler one (no vocab validation, no skip shards). For a 500-step retrain, `TokenStream` is fine. No issue.

## Verdict
APPROVE (round 1). Plan is technically sound, shapes are verified correct, fallbacks cover the two realistic failure modes (bfloat16 SVD, monkey-patch grad). The `cwd` omission is non-blocking since the cross_project doc (read as Step 0) and existing scripts establish the pattern.
