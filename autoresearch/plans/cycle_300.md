# Cycle 300 Plan — Item 310: Semantic decomposition of cycle 280's frozen A factor

## Item 310 (verbatim Hypothesis)
direction 0 loads on `<eos>`/punctuation/stopwords (the unigram-frequency axis seen in cycles 217 and 235). Direction 1 loads on a more semantically structured subspace. Both directions are dominated by mostly-negative-regime FFN channels — i.e., the rank-2 c_proj reads mostly from the "GELU bump" channels, not the positive ones.

## Dependencies
- `experiments/block1_lowrank_cycle280.py` (SVD of block-1 c_proj, `LowRankCProj`, 4000-step Adam retrain).
- `experiments/cycle280_decomp_cycle299.py` (seed 42 retrain template, trajectory infra).
- `experiments/delta_centred_alignment_cycle218.py` (Q/K/V slicing: `W_attn[:, 0:768], [768:1536], [1536:2304]`; top-5 cos_sum via `torch.linalg.svdvals`).
- `experiments/wdown_sign_geometry_cycle261.py` (σ_c = `(preact > 0).float().mean(dim=0)`, pre-activation `z = ln2_out @ W_fc + b_fc`).
- GPT-2 small, wikitext-2-raw-v1 val, 200-seq calib slice (seed 42 as in cycle 280/299).

## Approach
1. New script `experiments/cycle280_A_semantic_cycle300.py`. Load GPT-2 small + wikitext-2 val; 200 calib / 100 eval × 128 tok (identical slice to cycle 280/299). Assert float PPL ≈ 53.895.
2. SVD block-1 c_proj.weight (3072, 768), float32 CPU. Build `A = U[:, :2] * S[:2]` (3072, 2) frozen buffer, `B_init = Vh[:2, :]` (2, 768). Keep `Vh_init = Vh[:2, :].clone()` for the retrain-vs-init comparison (step 4). Install `LowRankCProj` on block 1.
3. Retrain B + bias Adam lr=1e-4, 4000 steps, eval every 100, SEED=42. Assert init PPL ≈ 57.664, best < init, best ≈ 49.06 (matches cycle 299). Save trained `B_final` (2, 768) and `bias_final` (768).
4. Cache `sigma_c` (3072,) and `W_fc_col_norms` (3072,) on block 1 once: compute `preact = ln_2(x) @ W_fc + b_fc` over the 200-calib slice (N_TOKENS = 25600), set `sigma_c = (preact > 0).float().mean(0)`, `W_fc_col_norms = ||W_fc[:, c]||_2`. Also cache `preact_mean_c` and `preact_std_c` per channel for the pre-activation stats field.
5. For each direction i ∈ {0, 1}, on the TRAINED A @ B_final:
   - **Input side**: `a_i = A[:, i]` (3072,). Top-20 channels by `|a_i[c]|` (descending). For each of the 20, record `(c, a_i[c], sigma_c[c], W_fc_col_norms[c], preact_mean[c], preact_std[c])`. Report group split: count how many are mostly_neg (σ ≤ 0.1), bal, mostly_pos (σ ≥ 0.9), "GELU bump" mid.
   - **Output side**: `b_i = B_final[i, :]` (768,). Compute `logits_i = wte.weight @ b_i` (50257,); top-20 by +logit (boost) and top-20 by −logit (suppress). List tokens via tokenizer.
   - **Block 2 Q/K/V alignment**: slice block 2 `W_attn` into Q, K, V (768×768 each); compute left SVs `U_q, U_k, U_v` (768, 20). For a rank-1 vector `b_i`, principal cosines against (768,20) reader basis: `cos_vec = (U_reader.T @ (b_i / ||b_i||))` (20,), take top-5 absolute values, sum. Report (Q, K, V) top-5 cos sum and max.
6. **SVD-init comparison**: repeat step 5 output-side with `b_i = Vh_init[i, :]` (no retrain). List top-20 boost tokens for init directions. Compare to trained boost-token lists — intersect and report overlap / new tokens per direction.
7. **Verdict**:
   - **CONFIRMED structured** iff ≥1 direction has top-20 boost cosine ≥ 0.3 with a semantically coherent cluster (recorded in JSON; coherence judged from the token list itself — programmatic proxy: at least 10/20 tokens are non-punctuation, non-stopword alphabetic subwords).
   - **CONFIRMED unigram** iff both directions' top-20 boost token lists are ≥ 14/20 punctuation/stopwords/whitespace/high-frequency tokens (programmatic: token in a fixed stopword/punct set).
   - **REFUTED interpretable** iff neither top-20 cluster is coherent AND all 6 Q/K/V cos_sums for both directions are below random baseline mean + 1σ (random baseline: 100 Gaussian unit vectors in R^768; top-5 absolute cosine sum; compute mean, std).
   - Otherwise **INCONCLUSIVE**.

## Files
- New: `experiments/cycle280_A_semantic_cycle300.py`
- New: `autoresearch/results/cycle_300_semantic.json`
- New: `autoresearch/results/cycle_300_summary.md`
- New: `autoresearch/logs/cycle_300_run.log`
- New: `autoresearch/results/cycle_300_manifest.txt`, `cycle_300_exec_result.txt`, `cycle_300_audit.txt`, `autoresearch/audits/cycle_300.md`
- Updated: `autoresearch/agenda.md`, `autoresearch/logs/experiments.tsv`, `autoresearch/STATUS.md`

## Config
TARGET_BLOCK=1, RANK=2, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, N_STEPS=4000, LR=1e-4, EVAL_EVERY=100, SEED=42, DEVICE="mps" if available, TOPK=20, RANDOM_SEEDS=100 (for null), STOPWORD_SET = `{",", ".", ":", ";", "(", ")", "?", "!", "'", '"', " ", "the", "a", "an", "of", "in", "to", "is", "was", " the", " a", " of", " to", " and", " in", " is", " it"}` (leading-space GPT-2 BPE forms), trainable params 2·768+768=2304 (asserted).

## Verification
- Float PPL within 0.5 of 53.895.
- Init PPL within 0.05 of 57.664.
- Best retrain PPL < init (task-performance verify).
- `A.shape == (3072, 2)` and `B_final.shape == (2, 768)`.
- `|sigma_c|` in [0, 1], `W_fc_col_norms` strictly positive.

## Baselines
- Cycle 218 c_fc top-5 cos sum 2.97 vs random 1.27 (d=768, subspace dim 20).
- Cycle 217 SV0 unigram-frequency axis (block 0 Δ̃).
- Cycle 261 σ_c per channel definition.
- Cycle 299 init/best PPL reference (57.658, 49.060 @ step 3900).

# APPROVE
