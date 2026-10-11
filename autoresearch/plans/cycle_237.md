# Task

Cycle 237 / item 247 (P0, T3): bump-subspace ablation vs random subspace baseline. Build `experiments/bump_t3_ablation_cycle237.py` measuring full-model PPL on wikitext-2 val (N_SEQS=200, SEQ_LEN=512) with the bump projector `P_l = V_l[:,:10] @ V_l[:,:10].T` from cycle 235's `t1_features.pt`, ablating residual stream `x_l ← x_l − x_l @ P_l` after each block's forward.

# Hypothesis

Verbatim from agenda item 247:
> T3: Bump subspace ablation vs random subspace baseline. For each block, take the top-10 right singular vectors of centred Δ̃ from item 245's `cycle_<N>_t1_features.pt`. Define the projector `P_l = V_l[:,:10] @ V_l[:,:10].T` and the ablation: after the FFN output of block l is added to the residual stream, replace `x_l ← x_l − P_l x_l`. Measure full-model PPL on held-out wikitext-2 val (N_SEQS=200, SEQ_LEN=512). Confirmed if joint bump-subspace ablation PPL ≥ 1.5× random-subspace ablation mean PPL, AND at least 3 individual blocks show single-block ablation PPL ≥ 2× their random baseline. Refuted if bump ablation PPL within 1.1× random baseline.

# Pre-flight Results

- P1 (deps): clean — cycle 235 done, `cycle_235_t1_features.pt` on disk with `block_l → V (768,32)`.
- P2 (no overwrite): clean — `cycle_237_*` filenames are fresh.
- P3 (data loaders): cycle 220 eval_ppl pattern + cycle 235 wikitext-2 val tokenisation are reusable.
- P4 (budget): 145 PPL evals at SEQ_LEN=512 ≈ 70 min walltime; budget tightened by dropping random seed count from 10→3 and gating per-block-random/full-FFN comparisons behind fallbacks.

# Steps

1. New file `experiments/bump_t3_ablation_cycle237.py`.
2. Load HF `gpt2` (eval, no_grad), tokenizer; SEED=42, DEVICE=mps/cpu, N_SEQS=200, SEQ_LEN=512.
3. Load `autoresearch/results/cycle_235_t1_features.pt`; for each block l extract `V_l = features[f"block_{l}"]["V"][:, :10]` (768,10) → `P_bump_l = V_l @ V_l.T` (768,768) on DEVICE.
4. Build wikitext-2 val eval set: replicate cycle 235 tokenise+concat+chunk pattern but on `split="validation"`, take 200 × 512 = 102400 tokens → `eval_ids` shape (200, 512).
5. Define `eval_ppl(model, eval_ids, batch=8)`: standard CE-over-tokens loss → exp; stream batches under `torch.no_grad()`.
6. Define hook factory:
   ```
   def make_ablation_hook(P):  # P: (768, 768) on DEVICE
       def hook(module, inputs, output):
           hs = output[0] if isinstance(output, tuple) else output
           new_hs = hs - hs @ P
           return ((new_hs,) + output[1:]) if isinstance(output, tuple) else new_hs
       return hook
   ```
   Register on `model.transformer.h[l]` (block-level forward_hook). Verify tuple-vs-tensor on first call.
7. baseline_ppl: no hooks installed.
8. joint_bump_ppl: install `make_ablation_hook(P_bump_l)` on all 12 blocks simultaneously, eval, remove all hooks.
9. per_block_bump_ppl[l] for l in 0..11: install only block l, eval, remove. 12 evals.
10. Random subspace baseline (3 seeds {42, 137, 2024}):
    - Per seed s, per block l: `Q,_ = torch.linalg.qr(torch.randn(768,10,generator=g_s))`; `P_rand[s][l] = Q @ Q.T`.
    - joint_random_ppl[s]: install all 12 P_rand[s][l] simultaneously.
    - per_block_random_ppl[l][s]: install only block l. 12×3 = 36 evals.
    - Aggregate: random_joint_mean ± std (n=3), per_block_random_mean[l] ± std (n=3).
11. Top-10 right SVs of FULL FFN output (calibration on wikitext-2 train, 200×512, seed 42 — match cycle 235 calibration): for each block l, register a forward_hook on `model.transformer.h[l].mlp` capturing its output, run forward over calib in batches, stack → (n_tokens, 768), `_, _, Vh = torch.linalg.svd(X, full_matrices=False)`, `V_full_l = Vh[:10].T`, `P_full_l = V_full_l @ V_full_l.T`. Then joint_full_ppl + per_block_full_ppl[l].
12. Verdict (item 247):
    - CONFIRMED if `joint_bump_ppl ≥ 1.5 * random_joint_mean` AND `≥3` of 12 blocks have `per_block_bump_ppl[l] ≥ 2.0 * per_block_random_mean[l]`.
    - REFUTED if `joint_bump_ppl ≤ 1.1 * random_joint_mean`.
    - INCONCLUSIVE otherwise.
13. Write `autoresearch/results/cycle_237_t3_ablation_ppl.json` with: baseline, joint_bump, per_block_bump (12), random_joint_per_seed + mean/std, per_block_random_per_seed + mean/std (12), joint_full_ffn, per_block_full_ffn (12), verdict, config dict (SEED, N_SEQS, SEQ_LEN, DEVICE, seeds list).
14. Write `autoresearch/results/cycle_237_t3_summary.md` with verdict + table (block | bump | random_mean±std | full_ffn | bump/random ratio) + joint comparison + 2-3 line interpretation.
15. Print verdict to stdout.

# Validation

- baseline_ppl is at SEQ_LEN=512 — will differ from cycle 220 (SEQ_LEN=128, ~45). Document the actual number; cycle 218 used SEQ_LEN=512 and is the closer reference.
- joint_bump_ppl > baseline (some damage expected from removing 10/768 dims at every block).
- per_block_bump_ppl[l] < joint_bump_ppl for all l (single-block damage strictly less than 12-block).
- Restore-sanity: re-eval baseline after final hook removal must match initial baseline within 1e-4.
- Random projector idempotency check (one block): `(P @ P - P).abs().max() < 1e-4`.

# Artifacts

- `experiments/bump_t3_ablation_cycle237.py`
- `autoresearch/results/cycle_237_t3_ablation_ppl.json`
- `autoresearch/results/cycle_237_t3_summary.md`
- `autoresearch/results/cycle_237_manifest.txt` (written by execute)

# Fallbacks

- Walltime tight (>60 min projected after baseline+joint_bump timing): drop per-block random comparison (keep joint_random only); leave per_block_random_mean as `null` in JSON and skip the second confirmation clause (fall through to INCONCLUSIVE if walltime forces it).
- Walltime still tight: drop the full-FFN top-10 comparison entirely; record `full_ffn: null`.
- Hook output shape mismatch: HF GPT2 block forward returns either tensor (use_cache=False, no past) or tuple. The factory above branches on `isinstance(output, tuple)`. On first call, print `type(output)` to verify.
- Ablation kills the model entirely (PPL ≫ 10× baseline at joint_bump): suspect projector orientation. Sanity: check `P_bump @ V_l[:,0] ≈ V_l[:,0]` (eigenvector of projector). If not, transpose convention is flipped.
- `linalg.qr` on MPS may fail — fall back to CPU for the QR/SVD steps and move P matrices back to DEVICE.

# Update bookkeeping

- Move item 247 from NEXT/P0 to PAST under `### Cycle 237` in `autoresearch/agenda.md`.
- Append TSV row to `autoresearch/STATUS.tsv` with cycle 237, item 247, verdict, key numbers.
- Commit on `geometric_compression` with message `[autoresearch] cycle 237: T3 bump-subspace ablation — <verdict>`.
