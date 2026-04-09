# Task
LASER on parameter-golf b3: rank-5 weight-SVD of `blocks.3.mlp.proj.weight` + B-factor retrain. First cross-project backward pass. Test ranks {2, 5, 10}.

# Hypothesis
Rank-5 retrain BPB <= baseline + 0.001. Baseline BPB = 1.3474 (pre-quant, from training summary).
- **CONFIRMED-LASER-transfers** if rank-5 retrain BPB ≤ baseline + 0.001: cycle 280 LASER recipe is generic, transfers to parameter-golf, ~1.2MB savings on the budget. **Major leaderboard-relevant result.** Schedule items 341/342 (LASER on b12/b11) as immediate follow-ups.
- **CONFIRMED-partial** if rank-5 retrain is in (baseline + 0.001, baseline + 0.005): partial recovery; rank-10 might do better. Worth a small rank sweep follow-up.
- **REFUTED-LASER-fails** if rank-5 retrain BPB ≥ baseline + 0.01: parameter-golf b3's rank-5 capacity is genuinely insufficient; cycle 280 doesn't transfer. Need to understand WHY (training-distribution mismatch? bfloat16 precision? composition with bigram/skip layers?).
- **INCONCLUSIVE** otherwise.

# Pre-flight Results
- P1: Not a duplicate. No LASER retrain on pgolf in PAST.
- P2: Dependencies 335/336 REFUTED, but both concern *universality* of b3 as target, not validity for this checkpoint. b3 pre-scale eff_rank=4.89, post-scale=4.01 — still lowest in early stack. Experiment is valid for 13L_640d_322 specifically. PASS.
- P3: Concrete — write 2 scripts, run 3 ranks, compare BPB. PASS.
- P4: No program.md changes needed. PASS.
- P5 (soft): Null baseline = no retrain after SVD truncation (B_init BPB). Script reports this as `b_init_bpb`.
- P6 (soft): ReLU² activation — SVD is on `proj.weight` (640×1920), not through the activation. The factored forward `x @ B.T @ A.T` replaces `x @ W.T` at the linear layer. Activation-agnostic.
- P7 (soft): 500 steps × seq_len 1024 ≈ 512K tokens for 9.6K trainable params. Ratio ~53:1. Adequate.

# Steps
0. `cat autoresearch/cross_project/parameter_golf.md` (REQUIRED by agenda).
1. Write `parameter-golf/local_scripts/pgolf_laser_retrain.py`:
   - Reuse `load_model_v2` from `pgolf_eff_rank_census_post_scale.py` (handles variable mlp_mult).
   - Args: `--checkpoint`, `--target-block`, `--rank`, `--n-train-steps`, `--lr`.
   - SVD: `W = model.blocks[b].mlp.proj.weight` shape (640, 1920). `U, S, Vh = mx.linalg.svd(W, full_matrices=False)`. `A_frozen = U[:,:r] @ mx.diag(S[:r])` (640,r). `B_init = Vh[:r,:]` (r,1920).
   - Replace proj forward: monkey-patch `model.blocks[b].mlp.proj.__call__` to compute `x @ B.T @ A.T` where B is trainable, A is frozen.
   - Freeze: `model.freeze()`, then unfreeze only B via `model.blocks[b].mlp.proj.B = B_init`.
   - Train: load train tokens via `TokenStream(hp.train_files)`. Use `nn.value_and_grad` on `model.loss`. Adam lr=1e-4, 500 steps, seq_len=1024, batch=1 seq.
   - Eval: `eval_bpb(model, val_tokens)` before SVD, after SVD (no retrain), after retrain.
   - Print JSON: `{baseline_bpb, b_init_bpb, retrained_bpb, rank, block, n_steps, n_params_trainable}`.
2. Write `experiments/pgolf_laser_retrain_cycle340.py` (qat side):
   - Subprocess-invoke pgolf script 3 times: `--rank 2`, `--rank 5`, `--rank 10`.
   - Collect results, write `autoresearch/results/cycle_327_pgolf_laser_retrain.json`.
   - Build `autoresearch/results/cycle_327_summary.md` with BPB-vs-rank table.
3. Run the qat-side script. Timeout 1800s (3 ranks × ~5 min each).
4. Move item 340 from NEXT to PAST in `agenda.md` with verdict and commit hash.
5. Commit: `[autoresearch] cycle 327: LASER rank-5 SVD + B-retrain on pgolf b3 — {VERDICT}`.

# Validation
- Baseline BPB from script must match training summary (1.3474 ± 0.001).
- `b_init_bpb` (no retrain) should be worse than baseline (confirms SVD truncation actually removes information).
- `retrained_bpb` for rank-10 should be <= rank-5 <= rank-2 (monotonic in rank).
- If any BPB is NaN/inf, the forward pass is broken — check shape conventions.

# Artifacts
- `parameter-golf/local_scripts/pgolf_laser_retrain.py` (permanent pgolf infrastructure)
- `experiments/pgolf_laser_retrain_cycle340.py` (cycle artifact)
- `autoresearch/results/cycle_327_pgolf_laser_retrain.json`
- `autoresearch/results/cycle_327_summary.md`

# Fallbacks
- If `mx.linalg.svd` fails on bfloat16: cast W to float32 before SVD, cast A/B back.
- If `nn.value_and_grad` can't differentiate through monkey-patched forward: use explicit `mx.grad` on a standalone loss function that takes B as argument.
- If 500 steps insufficient (retrained_bpb still >> baseline): try 2000 steps with lr=3e-5.
- If baseline BPB mismatch: check that `load_model_v2` loads all weights correctly (compare param count).
