# Cycle 333 Brief

## Item
Item 332: LASER recipe replication on tiny GPT activation variants — does cycle 280's rank-r-beats-baseline transfer across activations on a controlled testbed?

## Context
- Cycle 332 trained 3 tiny GPTs (4L 256d) on wikitext-2 with GELU/ReLU²/LeakyReLU², 3 seeds each. All converged. Val PPL: GELU=198, ReLU²=185, LeakyReLU²=185.
- Checkpoints at `experiments/tiny_gpt_act_compare_checkpoints/{gelu,relu2,lrelu2}_s{seed}.pt`
- Eff_rank: GELU=[90,108,114,89], ReLU²=[109,138,138,111], lrelu²=[107,152,149,119] — GELU has lowest eff_rank at all blocks
- Cycle 280: GPT-2 GELU b1 rank-2 LASER → PPL 49.03 (vs float 53.90, −4.86). That GPT-2 model had eff_rank 1.95 at b1 and top-1 energy 93%.
- Cycles 327-329: LASER on pgolf REFUTED due to flat weight spectra. Lesson: always check spectrum before committing to retrain.
- Tiny GPT `c_proj.weight` shape: (256, 1024). Much smaller than GPT-2's (3072, 768) or pgolf (640, 1920).
- Tiny GPT uses `nn.Linear` (weight shape transposed vs HF Conv1D); no bias on c_proj.

## Dependencies
- Checkpoints from cycle 332 (confirmed exist, 9 files, seed 42/137/2024 × 3 activations)
- Model class in `experiments/tiny_gpt_act_compare_cycle331.py`
- LASER template: `experiments/block1_lowrank_cycle280.py` (LowRankCProj class)
- wikitext-2 data (same as cycle 332)

## Key Files
- Checkpoints: `experiments/tiny_gpt_act_compare_checkpoints/{gelu,relu2,lrelu2}_s{42,137,2024}.pt`
- Model + data: `experiments/tiny_gpt_act_compare_cycle331.py`
- LASER template: `experiments/block1_lowrank_cycle280.py`
- Spectrum preflight template: `experiments/pgolf_laser_b11_spectrum_cycle329.py`

## Constraints
- Run cap: 15 minutes (3 variants × 4 ranks × ~1 min each, plus spectrum preflight)
- Spectrum preflight MUST gate retrain: if top-5 energy < 10%, skip retrain, report REFUTED-flat-spectrum
- Must adapt LowRankCProj for `nn.Linear` (not HF Conv1D): weight is (256, 1024), forward is `F.linear(x, W)` not `x @ W + b`
- No bias on c_proj in tiny model (unlike GPT-2)

## Critical Difference: nn.Linear vs HF Conv1D
- GPT-2 (HF): `c_proj.weight` is (3072, 768), forward = `x @ W + b`
- Tiny GPT: `c_proj.weight` is (256, 1024), forward = `F.linear(x, W) = x @ W.T`
- SVD of (256, 1024): U (256, 256), S (256,), Vh (256, 1024)
- Rank-r: A = U[:, :r] * S[:r] (frozen, 256×r), B = Vh[:r, :] (trainable, r×1024)
- Forward: F.linear(x, A @ B) = x @ (A @ B).T

## Handoff
Write spectrum preflight + LASER retrain script. Lowest output eff_rank blocks: GELU=b3(89.0), ReLU²=b0(109.2), lrelu²=b0(107.2). But output eff_rank != weight spectrum — the preflight checks the WEIGHT SVD of c_proj at the target block. Run spectrum preflight on all 4 blocks × 3 variants to pick the best LASER target per variant (most concentrated weight spectrum). If concentrated (top-5 ≥ 10%): sweep ranks {1,2,5,10} with 500 steps Adam lr=1e-4, 3 seeds. Report 3×4 PPL table. Verdict per agenda item 332.
