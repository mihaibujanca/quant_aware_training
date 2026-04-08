# Cycle 318 Audit — Item 333 (parameter-golf eff_rank census)

## Audit status: PASS

## Provisional outcome (from execute)
CONFIRMED-collapse-generalizes (any block ≤ 20 — 3 blocks: b3=4.89, b11=13.40, b12=6.94) AND CONFIRMED-different-shape (some ≤ 20, some ≥ 100 — b0=187.61, b1=160.08).

## Verdict logic check (item 333 verbatim)
The agenda's verdict tree:
1. CONFIRMED-collapse-generalizes if any block has eff_rank ≤ 20.
2. CONFIRMED-no-slack if all 13 blocks have eff_rank ≥ 100.
3. CONFIRMED-different-shape if eff_rank profile is non-empty in {≤20} but also has some blocks ≥ 100.
4. INCONCLUSIVE if all blocks fall in (20, 100).

Measured per-block eff_rank: b0=187.61, b1=160.08, b2=86.05, b3=4.89, b4=38.30, b5=39.96, b6=28.21, b7=23.23, b8=30.78, b9=28.85, b10=21.55, b11=13.40, b12=6.94.

- Branch 1 (any ≤ 20): TRUE — 3 blocks (b3, b11, b12).
- Branch 2 (all ≥ 100): FALSE — 11 blocks below 100.
- Branch 3 (mixed): TRUE — 3 blocks ≤ 20 AND 2 blocks ≥ 100.
- Branch 4 (all in (20, 100)): FALSE — both endpoints are populated.

Branches 1 and 3 are simultaneously TRUE. Branch 3 is the more specific outcome (it implies branch 1 by construction) and is the verdict label written by the executor. Both labels are consistent. The headline finding is the more general one ("rank-collapse generalizes to ReLU²"); the secondary finding is "the U-shape is differently shaped from GPT-2's".

The exec script's verdict logic at `experiments/pgolf_eff_rank_census_cycle318.py:write_summary` was traced: the `n_collapsed > 0 and n_full > 0 → CONFIRMED-different-shape` branch is hit before the `n_collapsed > 0 → CONFIRMED-collapse-generalizes` branch, which is consistent with the more-specific-first ordering above. PASS.

## Cross-checks (all PASS)
- 13-block profile present (asserted by sample of `cycle_318_pgolf_eff_rank.json`); n_blocks = 13 in the JSON header. ✓
- Each block reports `dim=640`, `n_tokens=25600`. Consistent across all 13 blocks. ✓
- Trainable params = 0 (forward-only measurement). The pgolf script never instantiates an optimizer, never wraps any parameter in Linear, never calls .backward() — all weights remain `mx.array` constants. ✓
- The captured tensor is `MLP.__call__` return value, i.e. **pre-`mlp_scale` MLP outputs** as required by `autoresearch/cross_project/parameter_golf.md` section 4 (gotcha 4). The hook in `pgolf_eff_rank_census.py` wraps `MLP.__call__` directly and captures `out = original_mlp_call(self, x, slope=slope)` BEFORE the Block's per-channel scale multiplication. ✓
- The eff_rank values are computed from `C = (Y^T Y) / N` with `N = 25600` and `dim = 640`, producing a (640, 640) PSD matrix and `eigvalsh`. eff_rank = exp(-Σ p log p) where p = λ/Σλ. The computation is in float64 to avoid eigenvalue precision drift. ✓
- Forward pass uses `n_seqs = 25` × `seq_len = 1024` = 25,600 tokens (matches request).

## Numerical sanity (re-run)
- Eigenvalue sums per block (trace of covariance) are positive and not near-zero, ruling out numerical degeneracy.
- For block 3 (the lowest eff_rank, 4.89): top-1 eigenvalue accounts for 78.22% of total variance — consistent with the eff_rank ≈ 5 reading (a near-rank-1 dominant subspace plus 3-4 minor subspaces). The interpretation is that block 3's MLP outputs concentrate into ~5 effective directions out of 640.
- For block 0 (the highest eff_rank, 187.61): top-1 fraction is 11.96% — closest to "diffuse" of any block, consistent with eff_rank in the hundreds.
- For block 12 (the output-adjacent block, eff_rank 6.94): top-1 fraction 65.10%, top-5 from rank_99=92 — extreme late-stack concentration, similar to GPT-2's b11 narrowing but more pronounced.

## Independent interpretation
This is the FIRST measurement of any kind on a non-GPT-2, non-GELU-trained model in the project. Item 333 is the headline cross-activation control test for the cycle 280 LASER finding.

**Headline reading**: rank-collapse phenomenon is NOT GELU-specific and NOT GPT-2-specific. A 13-layer 640-dim ReLU²-trained transformer (parameter-golf), trained under explicit parameter pressure (16MB artifact, ~4M params), still concentrates output covariance to the rank ≤ 20 regime in 3/13 blocks. The "GELU-specific" hypothesis and the "overparameterization-specific" hypothesis are both refuted in one shot.

**Shape comparison vs GPT-2 cycle 278** (cycle 278 numbers from `autoresearch/results/cycle_278_all_blocks_eff_rank.json`):

| block | GPT-2 | parameter-golf |
|---|---|---|
| 0 | 71.93 | 187.61 |
| 1 | 1.94 | 160.08 |
| 2 | 1.05 | 86.05 |
| 3 | 24.29 | 4.89 |
| 4 | 49.04 | 38.30 |
| 5 | 179.19 | 39.96 |
| 6 | 290.97 | 28.21 |
| 7 | 297.79 | 23.23 |
| 8 | 291.41 | 30.78 |
| 9 | 195.50 | 28.85 |
| 10 | 16.70 | 21.55 |
| 11 | 22.49 | 13.40 |
| 12 | — | 6.94 |

The U-shape is structurally different. GPT-2's leading concentration is at b1/b2 (eff_rank 1-2); parameter-golf's b1/b2 are HIGH (160, 86) and the leading concentration is at b3 (4.89). Both models concentrate near the output (GPT-2: b10/b11 = 16.70/22.49; parameter-golf: b11/b12 = 13.40/6.94), supporting the "late residual-stream concentrator" pattern from cycles 304/311 as a universal late-stack feature.

**Practical implication**: the cycle 280 LASER recipe (rank-r SVD of `c_proj` weight + Adam retrain on B + bias) should be tried at parameter-golf b3 first, where eff_rank=4.89 is the analog of GPT-2's b1 1.94. The expected payoff is comparable: a 2,304-param patch that beats float held-out BPB. This is the natural follow-up cycle.

## Cross-project infrastructure check
This is the first cycle to use `autoresearch/cross_project/parameter_golf.md` end-to-end. The plan correctly listed Step 0 as "cat the doc". The executor wrote both halves of the bridge:
- pgolf side: `parameter-golf/local_scripts/pgolf_eff_rank_census.py` (permanent infrastructure, reusable across future cycles).
- qat side: `experiments/pgolf_eff_rank_census_cycle318.py` (per-cycle wrapper).

**Infrastructure issues encountered and fixed during exec**:
1. `analyze_checkpoint.load_model` does NOT pass `bigram_vocab_size` to the GPT constructor, so the 13L_640d checkpoint (which contains `bigram.embed.weight` etc.) would fail to load with `ValueError: Received 3 parameters not in model`. Fixed by writing a custom `load_model_v2` in `pgolf_eff_rank_census.py` that detects bigram from the .npz keys and passes `bigram_vocab_size=4096, bigram_dim=128` accordingly.
2. `analyze_checkpoint.load_model` also doesn't pass `mlp_mult_per_layer`. The 13L_640d checkpoint has variable `mlp_mult` per block: 7 blocks at 3, 4 blocks at 2, 1 block at 0.5 (b12 has hidden=320 vs dim=640). `load_model_v2` infers mlp_mult per block from `blocks.{i}.mlp.fc.weight.shape[0] / dim`.
3. `Hyperparameters().model_dim` defaults to 512 but the checkpoint is 640. `load_model_v2` overwrites `hp.model_dim` from `tok_emb.weight.shape[1]`.
4. `load_val_tokens(hp, n)` reads `hp.val_files` which is built from `hp.data_path = "./data/datasets/fineweb10B_sp1024"`. The relative path requires the subprocess to run with `cwd=PGOLF_ROOT`, AND/OR `DATA_PATH` env var set. The qat-side wrapper sets both for safety.
5. `analyze_checkpoint.py` imports `from train_gpt_mlx import (...)` which only resolves if `pgolf_root` is on `sys.path`. The pgolf script inserts both `local_scripts/` and the project root.

These fixes are upstream improvements to the cross-project bridge — future cycles invoking other pgolf measurements should reuse `load_model_v2` or pass the same args to a shared loader. **Recommendation**: extract `load_model_v2` into a shared module (e.g. `parameter-golf/local_scripts/cross_project_loader.py`) before cycle 334 (item 334) needs the same loader.

## Auditor recommendation
PASS the CONFIRMED-collapse-generalizes / CONFIRMED-different-shape verdict. Both the headline finding (rank-collapse is generic) and the secondary finding (the U-shape differs from GPT-2's) are well-supported. The cross-project bridge worked end-to-end on its first execution, modulo the load_model gotchas which are fixed in this cycle's infrastructure.

The natural next step is to attempt cycle 280 LASER replication at parameter-golf b3 (eff_rank=4.89, dim=640) — this is the cross-project payoff item 333 was designed to enable. File as a new agenda item; do NOT retroactively modify this cycle.

## Status: PASS (provisional outcome accepted unchanged)
