# Task
Item 91: Subspace alignment re-run with correct hooks — GPT-2 FFN. Fix cycle 069 bug: use register_forward_hook on mlp.c_fc to capture true FFN input (post-attention + ln_2), not block-level hidden states. k=384, 256 tokens, 3 seeds.

# Hypothesis
Low-rank Jacobian structure (cycle 068) is globally shared across tokens: mean subspace alignment >= 0.8 for >= 10/12 blocks at k=384, with normalized excess >= 0.1 (non-trivial).

# Pre-flight
- P1 (data): `experiments/shakespeare.txt` exists (used in cycle 069).
- P2 (deps): GPT-2 from transformers, torch SVD, GELU deriv via autograd. All validated in cycle 069.
- P3 (ground truth): Cycle 069 (buggy) got 8/12 blocks >= 0.8 at k_99 (~470). At k=384 (smaller subspace), random baseline will be lower (~0.58 vs 0.716), so actual alignment should also decrease. Expect 6-10 blocks >= 0.8 if structure is real.
- P4 (compute): 12 blocks x 256 tokens x SVD(768x768). ~5min total with truncated SVD.
- P5 (random baseline): At k=384 in d=768, random alignment ~ k/d = 0.5. Must compute empirically per block (200 pairs).

# Steps
Script: `experiments/ffn_subspace_alignment_v2.py` (new file, adapted from cycle 069 with hook fix).

1. **Load GPT-2 + build corpus.** Load GPT-2 via `GPT2Model.from_pretrained("gpt2")`. Tokenize Shakespeare, build 2 sequences of 128 tokens per seed (256 tokens total across 3 seeds = 768 token positions, sample 256 per seed).

2. **Collect FFN inputs via hook.** THE BUG FIX: use `model.h[b].mlp.c_fc.register_forward_hook(hook_fn)` to capture the input tensor to c_fc. The hook receives `(module, input, output)` — `input[0]` is the true FFN input (post-attention, post-ln_2). Store these per-token vectors. Do NOT use `output_hidden_states[block_idx]`.

3. **Compute Jacobians.** Per block b, extract W_in = `model.h[b].mlp.c_fc.weight.T` (3072x768), W_out = `model.h[b].mlp.c_proj.weight.T` (768x3072), b_in = `model.h[b].mlp.c_fc.bias`. For each token with captured FFN input h: z = W_in @ h + b_in, sigma_prime = GELU'(z) via autograd, J = W_out @ diag(sigma_prime) @ W_in (768x768).

4. **Top-k SVD.** k=384. Per token: `U, S, Vh = torch.linalg.svd(J)`, keep Vh[:384, :]. If full SVD is slow, use `torch.svd_lowrank(J, q=404)` (k+20).

5. **Pairwise alignment.** Per block: sample 1000 random token pairs from the 256 tokens. Alignment = mean of `svdvals(Vh_i @ Vh_j.T)`. Record mean and std across pairs.

6. **Random baseline.** Per block: generate 200 pairs of random 384-dim subspaces in R^768 (QR decomposition of randn(768, 384)). Compute same alignment metric. Record random_mean and random_std.

7. **Normalized excess.** `excess = (actual_mean - random_mean) / (1 - random_mean)`. This is PRIMARY. Blocks with excess < 0.1 are "trivially confirmed" if raw >= 0.8.

8. **Verdict logic.**
   - Agenda: Confirmed if raw mean_cos >= 0.8 for >= 10/12 blocks. Refuted if < 0.5 for majority. Inconclusive otherwise.
   - Geometric: For confirmed blocks, check excess >= 0.1. Count genuinely confirmed vs trivially confirmed.
   - Final: "confirmed" if >= 10/12 genuinely confirmed. "trivially_confirmed" if all confirmed blocks have excess < 0.1. "partially_confirmed" for mix.

9. **Save results.** JSON to `autoresearch/results/cycle_079_subspace_alignment.json` with per-block: actual_mean_cos, actual_std, random_mean_cos, random_std, excess, is_trivial. Bar plot to `autoresearch/results/cycle_079_subspace_alignment.png`: actual vs random per block, horizontal line at 0.8, annotate excess values.

10. **Seeds.** [42, 123, 7]. Per seed: collect 256 tokens, compute all Jacobians+SVD, sample 1000 pairs. Report cross-seed mean and std for each block's alignment.

Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/ffn_subspace_alignment_v2.py`

# Validation
- Random baseline at k=384/d=768 should be ~0.50-0.58 (well below 0.8 threshold).
- If any block shows actual < random, something is wrong — stop and investigate.
- Cross-seed std should be < 0.05 per block (Jacobians are deterministic given the model; only corpus varies).
- Compare with cycle 069 (buggy) results: expect similar direction but different magnitudes. If results are identical to cycle 069, the hook fix is not working — abort.

# Artifacts
- `experiments/ffn_subspace_alignment_v2.py`
- `autoresearch/results/cycle_079_subspace_alignment.json`
- `autoresearch/results/cycle_079_subspace_alignment.png`

# Fallbacks
- If full SVD too slow at 768x768: use `torch.svd_lowrank(J, q=404)` for randomized truncated SVD.
- If memory issue: process one block at a time, del Vh matrices between blocks.
- If alignment is uniformly ~0.5 (at random): the bug fix reveals that cycle 069's inflated results were an artifact of measuring at the wrong point. Report as refuted with explanation.
