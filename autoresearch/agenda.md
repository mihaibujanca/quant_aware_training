# Research Agenda — Transformer-First Geometric Compression

This queue supersedes the old toy-model-first plan. The goal is to answer the right transformer FFN compression questions, in the right order, with distillation as a control and not the mainline.

## NEXT EXPERIMENTS

### P0

GELU bump interpretability items (230-234) reorient the investigation from activation statistics toward the W_down column geometry and the per-token difference vector Δ(x) = GELU_FFN(x) − ReLU_FFN(x). See `docs/gelu_bump_audit.md` §11 for why activation-only analysis has stalled. The goal is interpretable claims a layperson can parse ("the bump pushes in direction X / writes into block l+1's reading subspace / uses anti-podal channel pairs"), not more statistics. Items 230-234 take priority over the compression item 226 below.

All items 230-234 operate on GPT-2 small on OWT validation tokens. Each produces a report with plots/tables plus a 1-sentence plain-language claim, not a PPL threshold. `Confirmed/Refuted` criteria are phrased as whether the interpretable claim holds with a clear pattern, not numeric thresholds.

233. **Anti-podal channel twin census — does activation pattern match column geometry?**
   - For each block, list all (c, c') channel pairs where cos(W_down[:,c], W_down[:,c']) < −0.9 (from item 230). For each such pair, measure across 200 OWT sequences: (a) corr(z_c, z_{c'}), (b) joint distribution of (z_c, z_{c'}) — do they fire in complementary regimes (one positive, one negative)?, (c) for how many tokens is sign(z_c) ≠ sign(z_{c'})? Test the hypothesis that anti-podal W_down twins have anti-correlated pre-activations — i.e., the model is actually using them as mirror channels where only one fires at a time.
   - Output: per-block twin list with activation statistics, scatter plots of (z_c, z_{c'}) for top-5 most anti-podal pairs. One-sentence conclusion: "Anti-podal W_down pairs fire complementarily in X% of cases, confirming/refuting the mirror-channel hypothesis."
   - Type: interpretability / mechanism confirmation
   - Claim to test: "Anti-podal W_down pairs have anti-correlated pre-activations — they are mirror channels, and GELU lets both directions be expressed by a single channel via the bump."
   - Confirmed if: in blocks with ≥100 anti-podal pairs (from item 230), median corr(z_c, z_{c'}) < −0.3, AND sign disagreement rate > 60%.
   - Refuted if: median corr > 0 (twins fire together, not complementarily) in all blocks.
   - Inconclusive if: weak negative correlation (-0.3 to 0) or mixed across blocks.
   - Depends on: 230

234. **Surgical bump removal — isolate the negative leak from the positive-zone bump**
   - Past experiments conflated two sub-effects: (i) the negative leak (GELU(z) < 0 for z < 0), and (ii) the positive-zone bump (GELU(z) < z for 0 < z < 2). Cycle 190 did one hybrid split but not the complementary clamping. Define two new activations: (A) `GELU_noneg(z) = max(0, GELU(z))` — kills the negative leak, keeps the positive bump shape; (B) `GELU_nobump(z) = z * 1[z > 2] + 0 * 1[z < −2] + linear_interp(z) for z in [−2, 2]` — kills the bump knee, keeps the asymptotic regimes. Run each all-12-blocks and measure PPL + Δ(x) to isolate which sub-region of the bump carries the functional load.
   - Output: 4-point comparison table {baseline, hybrid (190), GELU_noneg, GELU_nobump} per block + all-12. Rank of Δ(x) under each variant. One-sentence conclusion: "The negative leak carries X%, the positive-zone bump carries Y% of the GELU-vs-ReLU functional advantage."
   - Type: empirical / decomposition
   - Claim to test: "The negative leak and positive-zone bump contribute unequally to model performance. One of them is functionally dominant; the other is near-linearizable."
   - Confirmed if: one of {GELU_noneg, GELU_nobump} all-12 PPL ≤ 25 (within 10% of float) AND the other ≥ 35 (>55% degraded) — clean separation.
   - Refuted if: both variants degrade similarly (within 3 PPL of each other) — the two sub-regions are entangled.
   - Inconclusive if: both degrade 20-50% with no clean separation.
   - Depends on: none

226. **Adam correction for 6 blocks with proper early stopping — confirm INCONCLUSIVE**
   - Cycle 211 showed 6-block best at step 1500 (PPL=59.68) but used fixed 4000 steps. Rerun with eval every 100 steps and early stopping to reliably find the optimal step count. This settles the INCONCLUSIVE from cycle 211.
   - Type: empirical
   - Confirmed if: 6-block PPL ≤58.0 with proper early stopping
   - Refuted if: 6-block PPL >62.0 even with early stopping
   - Inconclusive if: PPL 58-62 (same as cycle 211)
   - Depends on: none

### P1

227. **Frobenius-weighted channel selection — select channels that minimize output Frobenius error**
   - Current topk scores = act_std × W_proj_norm (cycles 207 showed this is near-optimal for single-block PPL). But for the CASCADE context, the damage accumulates block-by-block. Select channels that minimize the full-model Frobenius error of the block output (||float_out - compressed_out||_F), using cross-validation on a held-out calibration split. May give better cascade resistance.
   - Type: empirical
   - Confirmed if: Frobenius-optimal selection improves 3-block init PPL by ≥1 PPL (≤60.75)
   - Refuted if: improvement <0.3 PPL vs standard init
   - Inconclusive if: 0.3-1 PPL improvement

228. **Lower K with Adam correction — compression efficiency curve**
   - All Adam experiments used K=512 (50% of full FFN). Test K=256 (25%) and K=128 (12.5%) with Adam correction for BLOCKS=[1,4,7]. If Adam can correct lower K, we get higher compression at similar PPL.
   - Type: empirical
   - Confirmed if: K=256 + Adam achieves PPL ≤52.0 (+3.5% vs float)
   - Refuted if: K=256 + Adam PPL >58.0
   - Inconclusive if: PPL 52-58

### P2

229. **Distillation baseline for 3-block compression — matched budget**
   - Establish a distillation upper bound for comparison: KD from float GPT-2 to compressed GPT-2 with BLOCKS=[1,4,7] K=512 (same architecture as cycle 206). Train W_out + W_fc for 10k steps with KD loss. This is the "trained compressor" ceiling for this architecture.
   - Type: empirical
   - Confirmed if: KD achieves PPL ≤44.0 (meaningfully better than Adam W_out correction 46.556)
   - Refuted if: KD PPL ≥47.0 (Adam W_out correction is already at the ceiling)
   - Inconclusive if: PPL 44-47





### Closed — SKIPPED, PENDING FIX (Mixtral — need pretrained model or OWT data)

138. ~~**Train MoE testbed — Tiny Mixtral**~~ SKIPPED — WikiText-2 too small, overfits at PPL 65. Need pretrained model or larger dataset.
139. ~~**REAP expert pruning baseline — Tiny Mixtral**~~ SKIPPED — depends on 138.
129. ~~**Geometric decomposition of expert pruning error — Tiny Mixtral**~~ SKIPPED — depends on 139.
130. ~~**Low-rank linear correction for expert pruning error — Tiny Mixtral**~~ SKIPPED — depends on 129.
131. ~~**Combined pruning + quantization + correction — Tiny Mixtral**~~ SKIPPED — depends on 130.

### Closed — OBSOLETE items (do not run)

23. ~~**Effective number of affine behaviors via Jacobian k-means — GPT-2 FFN**~~ ⚠ OBSOLETE: Superseded by item 80 (Jacobian effective rank census).
   - Apply the same clustering idea from experiment 22 to GPT-2 FFN blocks. Instead of MASO affine maps (unavailable for GELU networks), use per-token Jacobians J(x) ∈ ℝ^{768×768} computed via `torch.func.jacrev`. Run k-means on the flattened Jacobians for each block. Measure the reconstruction error of block outputs when each token's output is replaced by J̄_cluster(x − x̄_cluster) + f(x̄_cluster) (first-order Taylor approximation using the cluster centroid Jacobian). Find minimum K* such that reconstruction error < 5% of block output norm. Compare K* across blocks and correlate with single-block swap damage from cycle 005.
   - Depends on: 9, 22 (refuted)

41. ~~**Constructive partition synthesis — toy ReLU (2D input) [MOONSHOT]**~~ ⚠ OBSOLETE: Depends on item 22 (refuted).
   - THE open problem in the literature. Given a trained spirals network (2→32→32→1): (a) Extract all observed activation patterns + affine maps via MASO formula (A_ω = ∏ diag(q_ω^i) W^i). (b) Cluster into K macro-regions via k-means on affine maps (leverage item 22). (c) For each macro-region, compute target affine map (support-weighted average of constituent maps). (d) SYNTHESIS: solve for weights W'_1, ..., W'_L of a smaller network such that for each macro-region k with pattern q_k, the product ∏ diag(q'_k^i) W'^i ≈ target A_k. Boundary continuity adds constraints: adjacent macro-regions' affine maps must agree on the shared boundary hyperplane. Method: Solve via least-squares with boundary penalty.
   - Depends on: 22 (refuted)

44. ~~**Topological autoencoder for partition space — toy ReLU [MOONSHOT]**~~ ⚠ OBSOLETE: Depends on items 22 (refuted) and 24 (dead end).
   - Build an autoencoder that learns a compressed encoding of the activation pattern space: Input = per-sample activation pattern (binary vector, one bit per neuron across all layers). Encoder: MLP mapping to low-dimensional latent z ∈ ℝ^K. Decoder: from z, predict the affine map A_ω. Loss = MSE(A_ω_predicted, A_ω_true) + λ · PH_reg (persistent homology regularization matching latent topology to RTG topology). If the autoencoder achieves good reconstruction with K << number_of_patterns, it proves the partition space has low intrinsic dimensionality and the decoder provides a direct "K codes → affine map" compressed functional representation.
   - Depends on: 22 (refuted), 24 (dead end)

## PAST EXPERIMENTS

### Cycle 218

232. **Where does Δ point? — centred Δ̃ alignment with next block's reading directions** *(CONFIRMED — cycle 218)*
   - **Outcome: CONFIRMED** — After centring Δ̃ = Δ − mean(Δ), top-20 right singular vectors of Δ̃_l exceed the random baseline (10 seeds Gaussian orthonormal 20×768, mean+3σ threshold ≈ 3.0) against at least one downstream reader in **11/11** cross-block pairs. Per-reader significant counts: c_fc 11/11, Q 10/11, K 6/11, V 8/11, residual-PC 11/11. Centred rank_95 jumps from raw median 1 (c217) to 552 (range 489-567 across blocks) — the per-block constant bias was hiding the true per-token geometry. σ₀ (centred) 260-1293 (block 11 highest), σ₁ 193-503. Mean ||Δ|| per block 16-153. Top-3 SV vocab projection: SV0 in blocks 2/3/10 loads on stopwords/punctuation (expected, the dominant per-token axis correlates with unigram frequency after centring); SV1-2 show mixed word/tokenizer-noise clusters (vocab heuristic counts all 12 blocks but that's primarily via SV2 word tokens, not semantically clean clusters). **Key finding**: the bump IS structured inter-block communication — downstream c_fc and residual stream both read from the Δ̃ subspace with margin, and Q/K/V attention reads partially from it. Connects to cycle 216 W_down→c_fc alignment baseline 0.649. Item 234 (surgical bump removal) is now well-motivated: killing the bump should disrupt cross-block information flow.
   - Artifacts: `experiments/delta_centred_alignment_cycle218.py`, `autoresearch/results/cycle_218_delta_centred_alignment.json`, `autoresearch/results/cycle_218_summary.md`

### Cycle 217

231. **Per-token Δ(x) = GELU_FFN − ReLU_FFN — rank, subspace, logit impact** *(INCONCLUSIVE — corrected from CONFIRMED — cycle 217)*
   - **Outcome: INCONCLUSIVE** (audit-corrected from CONFIRMED; semantic-cluster clause not met in spirit). Rank clause ✓: rank_95 ≤ 64 in 9/12 blocks (median 1; exceptions blocks 1/8/9 at 73-132). Interpretability clause ✗: TI/TD ratio 0.87-0.99 shows Δ is dominated by a **per-block constant bias**, not a structured token-dependent message. SV0 in rank_95=1 blocks is the mean direction projecting trivially to the unigram-frequency axis (stopwords boosted, tokenizer artefacts suppressed, or its sign-flip). σ₁ ≈ 200-350 vs σ₀ ≈ 3k-7k (3-9%). The per-token component lives in SV1+ and is untested here. Cross-block top-3 principal cosines Δ_l → c_fc_{l+1} left SVs: mean 0.369 vs random 0.168 vs cycle 216 W_down→c_fc baseline 0.649 — above noise but does not exceed structural baseline. Connects cycles 188 (C3 mean replacement +20.33% PPL) and 191 (constant residual correction +12 PPL): the constant IS the dominant Δ component but the model depends on the small SV1+ residual. Item 232 must centre Δ̃ = Δ − mean(Δ) to actually test the hypothesis. Audit: `autoresearch/audits/cycle_217.md`.
   - Artifacts: `experiments/delta_per_token_cycle217.py`, `autoresearch/results/cycle_217_delta_per_token.json`, `autoresearch/results/cycle_217_summary.md`, `autoresearch/audits/cycle_217.md`

### Cycle 216

230. **W_down column geometry census — anti-podal pairs, clusters, cross-block alignment** *(REFUTED — cycle 216)*
   - **Outcome: REFUTED** — All 12 blocks have <1.2% columns with anti-podal twin at cos<-0.9 (max block 0: 1.17%; blocks 1-5: 0.00-0.26%). Within-cluster cos 0.03-0.07 (near-random). Random baseline (3 seeds, R^768): 0 anti-podal pairs. W_down columns are NOT geometrically paired. **Notable secondary finding**: cross-block top-5 principal cosines between block l's W_down writing subspace and block l+1's c_fc reading subspace average 0.649 (mid-blocks 4-7 hit 0.70-0.78), so adjacent blocks do share writing/reading directions despite no column-level anti-podality.
   - Artifacts: `experiments/wdown_column_census_cycle216.py`, `autoresearch/results/cycle_216_wdown_geometry.json`, `autoresearch/results/cycle_216_summary.md`

### Cycle 215

225. **Sequential block-by-block Adam W_out correction** *(INCONCLUSIVE — cycle 215)*
   - **Outcome: INCONCLUSIVE** — Sequential (block 1→4→7, 2000 Adam steps each, best-state restore) final PPL=47.253 ± 0.047 across 3 seeds vs joint Adam best 46.556. Per-block trajectory: init 61.746 → b1 51.486 → b4 48.503 → b7 47.253. Monotone, no collapse, but 0.7 PPL short of confirmed threshold — joint optimisation still wins marginally.
   - Artifacts: `experiments/adam_sequential_cycle215.py`, `autoresearch/results/cycle_215_sequential_adam.json`

### Cycle 214

224. **Analytical W_out correction** *(REFUTED — cycle 214)*
   - **Outcome: REFUTED** — Analytical PPL=61.767 ≈ init 61.746 (no improvement). Distribution mismatch: compressed-context activations ≠ float-context activations. Cascade correction is non-local and requires CE gradient descent. Analytical lstsq cannot replace Adam.

### Cycle 213

223. **Adam with trainable W_fc — does unfreezing fc improve?** *(INCONCLUSIVE — cycle 213)*
   - **Outcome: INCONCLUSIVE** — W_out+W_fc best PPL=46.424 vs W_out-only best=46.556 (delta=0.13 PPL). Faster convergence (1500 vs 4000 steps) but same quality. W_fc training adds no meaningful advantage.

### Cycle 212

222. **10-block full compression with Adam** *(REFUTED — cycle 212)*
   - **Outcome: REFUTED** — Best PPL 98.493 at step 800 (+82.7% vs float). Adam W_out correction cannot fix 10-block cascade. Scaling pattern: 3-block (below float), 6-block (+10.7% early stop), 10-block (+82.7% REFUTED). Practical limit ~6 blocks with strict early stopping.

### Cycle 211

221. **6-block Adam correction — BLOCKS=[1,2,4,5,7,8] K=512** *(INCONCLUSIVE — cycle 211)*
   - **Outcome: INCONCLUSIVE** — Best PPL 59.680 at step 1500 (criterion ≤60.0 met), but rapid overfitting: step 2000=61.141, step 4000=72.250. More params → faster overfitting (6-block optimal 1500 steps vs 3-block 4000 steps). Achievable with strict early stopping but fragile.

### Cycle 210

220. **Adam W_out correction step-count curve** *(INCONCLUSIVE — cycle 210)*
   - **Outcome: INCONCLUSIVE** — Plateau between steps 2000-4000 (0.25 PPL delta). Overfitting at step 8000 (+4.67 PPL). Best PPL: 46.556 at step 4000. Practical: early stop at ~4000 steps for N_CALIB=200.

### Cycle 209

212. **Rank-1 input FFN compression — explicit 1D subspace projection** *(REFUTED — cycle 209)*
   - **Outcome: REFUTED** — Block 4 MLP input (after LN2) has rank_90=395, not rank-1. LN2 per-token normalization spreads residual stream into ~395D. Rank-1 K=512 PPL=58.207 vs standard K=512 PPL=55.715 (gap=+2.49). Rank-1 saturates at K≥256 (same PPL at K=256, 512, 1024). Key correction: residual stream rank-1 ≠ MLP input rank-1.

### Cycle 208

219. **PCA-conditioned lstsq for block 1** *(REFUTED — cycle 208)*
   - **Outcome: REFUTED** — All PCA variants worse than standard (54.770–55.074 vs 54.689). Best improvement: −0.08 PPL. lstsq handles near-zero-variance input dimensions through numerical conditioning; explicit PCA adds approximation error without benefit.

### Cycle 207

218. **Block 1 channel selection — magnitude topk vs output-contribution topk** *(REFUTED — cycle 207)*
   - **Outcome: REFUTED** — All 4 criteria within 1 PPL (range 54.629–55.679). Best alternative (pure act_std, S3) improves by 0.06 PPL despite orderings being genuinely different (min Kendall-tau=−0.13). lstsq W_out adapts to whichever 512 channels are selected. Selection criterion irrelevant at K=512.

### Cycle 206

217. **Adam correction on W_out of compressed MLPs — topk+lstsq init, CE target** *(CONFIRMED — cycle 206)*
   - BLOCKS=[1,4,7] K=512 model; frozen fc weights; 1.18M trainable params (W_out/b_out of 3 compressed MLPs).
   - **Outcome: CONFIRMED** — PPL drops 61.746→48.111 (improvement=13.635), which is 10.7% BELOW float (53.895). Target was ≤59.0. Cascade barrier is not structural: suboptimal W_out init fully explains the +14.6% cascade, correctable with 1.18M params and 1000 Adam steps. Eval PPL still decreasing at step 1000.

### Cycle 205

216. **Block 1 single-block K-sweep** *(INCONCLUSIVE — cycle 205)*
   - Block 1 alone K=512: +1.5% PPL (54.689 vs float 53.895). 3-block +14.6% is entirely cascade. Improvement beyond K=512: 0.493 PPL (below 0.5 threshold). K=512 near-optimal; cascade amplification (10×) is the bottleneck, not single-block quality.
   - Artifacts: `experiments/block1_k_sweep_cycle205.py`, `autoresearch/results/cycle_205_block1_k_sweep.json`

### Cycle 204

215. **3-block selection sweep** *(REFUTED — cycle 204)*
   - [1,4,10]=61.455 best, anchor [1,4,7]=61.746 (0.29 PPL gap = within noise). Block 1 dominates: all top-4 include block 1 (input_rank=192). Blocks starting at 2+ consistently worse (+1.8-7.6 PPL). Best achievable with 3-block topk+lstsq is ~+14-15% above float. Current direction closed.
   - Artifacts: `experiments/block3_selection_sweep_cycle204.py`, `autoresearch/results/cycle_204_block3_selection.json`

### Cycle 203

214. **Block subset search — more compressible blocks from {2,3,5,6,8,9,10}** *(REFUTED — cycle 203)*
   - No superset of [1,4,7] within 2 PPL of 61.75. Adding 2 blocks costs +13-40 PPL. 3 blocks is the ceiling. Cascade near-symmetric across early/middle/late blocks (+40/43/46%). Each extra block adds ~14-15% PPL.
   - Artifacts: `experiments/block_subset_search_cycle203.py`, `autoresearch/results/cycle_203_block_subset_search.json`

### Cycle 202

213. **BLOCKS=[1,4,7] compression (exclude block 11)** *(CONFIRMED — cycle 202)*
   - Excluding block 11 improves PPL by 6.4 (68.14→61.75; +14.6% vs +26.4% above float). Block 11 rank-expansion (rank-2→231) cannot be approximated by K=512 topk+lstsq from rank-2 input. New best recipe: BLOCKS=[1,4,7], K=512 → +14.6% above float.
   - Artifacts: `experiments/block11_exclusion_cycle202.py`, `autoresearch/results/cycle_202_block11_exclusion.json`

### Cycle 201

211. **Block 11 geometry deep-dive** *(CONFIRMED — cycle 201)*
   - Block 11 output decompressor: in_rank=2→out_rank=231 (154× expansion, mirror of block 0). Delta_rank=1 (rank-1 perturbation, 97% top-3 SV energy). Token-type concentration: top-20 mean delta 848.7 vs overall 335.7 (2.5×). Block 11 expands rank-1/2 residual for LM head. Borderline droppability explained: LM head receives rank-2 not rank-231.
   - Artifacts: `experiments/block11_geometry_cycle201.py`, `autoresearch/results/cycle_201_block11_geometry.json`

### Cycle 200

210. **Residual stream rank trajectory under compression** *(REFUTED — cycle 200)*
   - Float ranks 244→192→23→1; compressed ranks 244→192→20→1. Max rank ratio=1.0. Rank-1 collapse robustly preserved. Cascade barrier is from Frobenius CONTENT error, not rank disruption. Frob div peaks at block 8-9 (~0.24), consistent with cycle 194 passive self-correction.
   - Artifacts: `experiments/rank_trajectory_cycle200.py`, `autoresearch/results/cycle_200_rank_trajectory.json`

### Cycle 199

209. **K-sweep for near-rank-1-input blocks** *(REFUTED — cycle 199)*
   - Degradation +57% at K=8 → +26% at K=512, no plateau. Rank-1 residual stream does NOT enable smaller K. LN normalization creates token-specific variation in the full 768D space regardless of residual rank. Block input rank ≠ MLP input diversity after LN.
   - Artifacts: `experiments/k_sweep_rank1_cycle199.py`, `autoresearch/results/cycle_199_k_sweep.json`

### Cycle 198

205. **Block 0 criticality analysis — what makes block 0 unique** *(CONFIRMED — cycle 198)*
   - Block 0 input_rank_90=269 vs median_others=1 (269× outlier). Raw token embeddings span 269 dimensions; later blocks operate on a near-rank-1 residual stream. Input rank collapse: 269→207→25→1 by block 3. Block 0 is irreplaceable because it transforms high-diversity embeddings into the residual stream; dropping it leaves later blocks operating on the wrong subspace. Cannot be compressed by the same approach as blocks 1-11.
   - Artifacts: `experiments/block0_criticality_cycle198.py`, `autoresearch/results/cycle_198_block0_criticality.json`

### Cycle 197

208. **Block-pair joint compression — sequential progressive fitting** *(INCONCLUSIVE — cycle 197)*
   - Strategy A (all-float): 30.089 PPL. Strategy B (sequential progressive): 29.954 PPL. Improvement: 0.135 PPL (within noise). Distribution mismatch in training data is NOT the cascade bottleneck. Cascade barrier is from fundamental information loss (K=512 channels cannot represent 3072), not from distribution shift during fitting.
   - Artifacts: `experiments/joint_compression_cycle197.py`, `autoresearch/results/cycle_197_joint_compression.json`

### Cycle 196

207. **LayerNorm recalibration for stacked compressed blocks** *(REFUTED — cycle 196)*
   - compressed_ppl=30.089 (blocks [1,4,7,11], K=512). LN-2 recal → 30.337 (worse). All-12 LN-2 recal → 30.645 (worse). Best improvement = -0.248. Per-channel cross-token statistics irrelevant to LN's per-token normalization. Cascade barrier is in residual content, not distributional shift. LN recalibration direction closed.
   - Artifacts: `experiments/ln_recalibration_cycle196.py`, `autoresearch/results/cycle_196_ln_recal.json`

### Cycle 195

206. **Jacobian effective rank census — data-active rank of GPT-2 FFN blocks** *(REFUTED — cycle 195)*
   - overall_mean_rank_90=472.4/768. All 12 blocks high-rank; 0/12 meet <100 criterion. Blocks 1-11: mean_rank 387-543/768 (50-71%). Block 0 lower at 166.8/768 (22%). Rank gradient: low early (b0: 166) → high middle (b9: 543) → slight drop (b11: 459 — consistent with block 11 anomaly). R3 MASO active-subspace hypothesis refuted: GPT-2 FFN blocks operate in near-full-rank subspaces. No exploitable low-rank active subspace.
   - Artifacts: `experiments/jacobian_rank_census_cycle195.py`, `autoresearch/results/cycle_195_jacobian_rank.json`

### Cycle 194

204. **Self-correction depth analysis — blocks 8-10 partial compensation** *(REFUTED — cycle 194)*
   - max_decrease=2/12 layers (only the specific layer where the newly swapped block sits). Error magnitude generally keeps growing; PPL improvement at k=8-10 is error REDIRECTION to low-PPL subspace, not cancellation. Consistent with cycle 184 (diffuse high-rank errors, no low-rank correction viable). Block 11 breaks self-correction with large PPL jump.
   - Artifacts: `experiments/self_correction_analysis_cycle194.py`, `autoresearch/results/cycle_194_self_correction.json`

### Cycle 193

203. **Superadditive interaction isolation — C3+Hybrid combined modification** *(INCONCLUSIVE — cycle 193)*
   - C3+Hybrid all-12 PPL=41.447 (+82.59%) vs ReLU+bias ~42.6 (+88%). Nearly identical to full ReLU+bias. Simple sum prediction was +48% (33.6 PPL); actual +83% — 35-point superadditivity. Confirms superadditivity is GENERAL (not specific to zero-point). Both-regime modification is the root cause regardless of specific form of negative replacement. Progressive non-monotone at k=9-10 (self-correction pattern).
   - Artifacts: `experiments/c3_hybrid_combined_cycle193.py`, `autoresearch/results/cycle_193_c3_hybrid_combined.json`

### Cycle 192

202. **Exact dead-channel pruning baseline — GPT-2 FFN (Serra/Berzins)** *(REFUTED — cycle 192)*
   - GPT-2 FFN has essentially zero dead channels. eps=0.001: 0 dead; eps=0.01: 0.003% (1 channel); eps=0.1: 0.03% (11 channels total); eps=1.0: 2.64% (974 channels) but PPL jumps 33.4→60.0 (+80% — not dead, just lightly used). Always-active: only 2 channels total. Exact dead-channel pruning not applicable to pretrained GELU networks.
   - Artifacts: `experiments/dead_channel_pruning_cycle192.py`, `autoresearch/results/cycle_192_dead_channel_pruning.json`

### Cycle 191

201. **Mean residual error correction — directional coherence-based bias (9216 params)** *(REFUTED — cycle 191)*
   - Corrected PPL=56.781 vs uncorrected=44.842 (WORSENED by 11.938 PPL). Mean error vectors computed from calibration data, applied as constant residual stream shifts, hurt subsequent layers by shifting their inputs outside operating range. Layer 0 correction≈0 (bias already compensates), layers 1-11 fraction_of_frob 0.13-0.35 (non-trivial mean) but applying it as constant bias is counterproductive. Confirms: residual stream corrections require downstream re-calibration.
   - Artifacts: `experiments/mean_error_correction_cycle191.py`, `autoresearch/results/cycle_191_mean_error_correction.json`

### Cycle 190

200. **Positive-regime shape isolation — GELU-neg + ReLU-pos hybrid activation** *(INCONCLUSIVE — cycle 190)*
   - Hybrid (GELU for z<0, ReLU for z>=0) all-12 PPL=28.980 (+27.67%) — in the inconclusive 28-35 range. Positive-regime shape change alone costs +27.67%, comparable to negative-regime constant replacement (C3 +20.33%). Full ReLU+bias swap +88% implies superadditive interaction between both modifications. Non-monotone progressive: peaks at k=7 (28.484), decreases through k=10 (27.851), rises at k=11 (28.980) — some deep-layer self-correction.
   - Artifacts: `experiments/gelu_neg_relu_pos_cycle190.py`, `autoresearch/results/cycle_190_gelu_neg_relu_pos.json`

### Cycle 189

199. **Cascade error direction analysis — GELU vs ReLU+bias residual stream** *(CONFIRMED — cycle 189)*
   - Mean cos(e_l, e_{l-1})=0.716; all 11/11 inter-layer pairs exceed 0.3 threshold. Errors compound directionally: ||e||_F grows 685→11472 (17x) over 12 layers. Rank_90 stays high (320-553/768) — errors are diffuse in content but directionally coherent. Block 11 anomaly: ||e||_F jumps 6054→11472 (largest single-block amplification), rank_90 drops to 320 (consistent with cycle 184). Baseline PPL=35.9 (SEQ_LEN=128), full-swap PPL=44.8 (+25%).
   - Artifacts: `experiments/cascade_error_direction_cycle189.py`, `autoresearch/results/cycle_189_cascade_error_direction.json`

### Cycle 188

198. **Progressive C3 composition — per-channel mean replacement across all 12 blocks** *(INCONCLUSIVE — cycle 188)*
   - Baseline 22.700 PPL. Progressive C3: +0.53%, +1.79%, +3.30%, +5.26%, +7.94%, +10.76%, +13.34%, +15.22%, +17.01%, +18.30%, +18.98%, +20.33% for blocks 0-11. All-12 PPL=27.314 (+20.33%), between +15% and +40% thresholds. Shape-preserving replacement composes far better than ReLU+bias (27.3 vs ~42 PPL). Constant replacement itself accounts for ~4.6 PPL; shape change accounts for remaining ~14.3 PPL of the 18.9-PPL gap in cycles 181/182.
   - Artifacts: `experiments/c3_composition_cycle188.py`, `autoresearch/results/cycle_188_c3_composition.json`

### Cycle 187

196. **Layer-drop baseline — GPT-2 FFN skip blocks [1,4,7,11]** *(CONFIRMED — cycle 187)*
   - Baseline PPL 34.9. Block 0 critical (PPL 1923 when dropped). Blocks 1-10 all droppable (PPL 36.4–39.1, damage +1.5–4.3). Block 11 borderline (41.4 PPL). Multi-block drop [1,4,7,11]: 53.0 PPL (+18). GPT-2 FFN blocks are individually near-redundant but cascade interactions produce nonlinear damage (single-block damage 2-4 vs multi-block 18). Block 0 uniquely critical.
   - Artifacts: `experiments/layer_drop_cycle187.py`, `autoresearch/results/cycle_187_layer_drop.json`

### Cycle 186

195. **Fewer-block compression — GPT-2 FFN k=4 on 2 blocks vs 4 blocks** *(REFUTED — cycle 186)*
   - Best 2-block pair (1,4) achieves mean PPL 50.4 ±5.3 vs 4-block 59.4 ±8.2 — 9 PPL improvement from halving block count, but still above 48 threshold. High cross-seed variance (std ~5-8 PPL) driven by seed-42 teacher PPL of 57.6. All 6 pairs range 50.4–54.8, tightly clustered. Block pair (1,4) consistently best.
   - Artifacts: `experiments/fewer_blocks_cycle186.py`, `autoresearch/results/cycle_186_fewer_blocks.json`

### Cycle 185

194. **Compression ratio sweep k={2,4,6,8} — GPT-2 FFN** *(REFUTED — cycle 185)*
   - PPL curve nearly flat: k=2→61.56, k=4→59.42, k=6→58.30, k=8→57.16 (only 4.4 PPL gain for 4x more params, 12x→3x compression). k=8 PPL=57.2 >55 threshold — compression floor is ratio-independent. Cascade/composition barrier dominates regardless of per-block channel count. Diminishing returns confirmed.
   - Artifacts: `experiments/compression_ratio_sweep_cycle185.py`, `autoresearch/results/cycle_185_compression_ratio_sweep.json`

### Cycle 184

197. **Residual stream trajectory divergence: GELU vs ReLU+bias — GPT-2** *(REFUTED — cycle 184)*
   - Divergence is exponential (R²=0.972, rate a=0.223/block) and α-smooth (monotone PPL from 57.7→38.1, sharp_ratio=2.17), but DIFFUSE (rank_90=300-434 out of 768 max). Low-rank correction would need ~434 dims to capture 90% of damage. Transition-zone fraction ~0.49 for both divergent and non-divergent tokens — transition zone does NOT predict which tokens diverge.
   - Artifacts: `experiments/trajectory_divergence_cycle184.py`, `autoresearch/results/cycle_184_trajectory_divergence.json`, `autoresearch/results/cycle_184_trajectory_divergence.png`

### Cycle 183

193. **Matched-budget distillation control baseline — GPT-2 FFN k=4** *(CONFIRMED — cycle 183)*
   - Local Adam+MSE distillation (cached block I/O, up to 2000 steps) vs topk+lstsq+polish: stacked PPL 80.7 vs 59.4 (+21.3 PPL). Single-block PPLs nearly identical (within 0.3-3.3 PPL), but stacking diverges sharply — topk+lstsq preserves compositional geometry that Adam-from-scratch does not.
   - Artifacts: `experiments/distillation_baseline_cycle183.py`, `autoresearch/results/cycle_183_distillation_baseline.json`

### Cycle 182

192. **Iterative bias recomputation for ReLU swap — GPT-2** *(INCONCLUSIVE — cycle 182)*
   - Iterative PPL 37.0 vs precomputed 42.6 — 5.6 PPL improvement (+63% vs baseline). Distribution shift is real (iterative helps) but not the dominant error source. Constant-bias correction insufficient even with correct input distributions. Later blocks still degrade: blocks 0-6 at +44%, all 12 at +63%. Block 11 μ_neg norm grows to 134.9 (vs 120.3 precomputed) — iterative sees larger negative-regime signals as upstream ReLU changes propagate.
   - Artifacts: `experiments/iterative_relu_bias_swap_cycle182.py`, `autoresearch/results/cycle_182_iterative_relu_swap.json`, `autoresearch/results/cycle_182_iterative_relu_swap.png`

### Cycle 181

191. **Full-model ReLU + per-block bias vector swap — GPT-2** *(REFUTED — cycle 181)*
   - Full-model ReLU+bias swap: PPL 42.6 (+87.9% vs float 22.7). Single-block swaps all within +6% (confirming cycle 173), but simultaneous 12-block swap cascades severely. Incremental accumulation shows smooth degradation: blocks 0-2 within +10%, blocks 0-4 at +30%, blocks 0-6 at +56%, all 12 at +88%. Precomputed biases from float model become stale as upstream blocks change the residual stream — exactly the distribution-shift failure predicted in R4 review. Block 11's μ_neg norm (120.3) is 4x larger than other blocks, suggesting the last block amplifies accumulated errors. Item 192 (iterative recomputation) is now eligible.
   - Artifacts: `experiments/full_relu_bias_swap_cycle181.py`, `autoresearch/results/cycle_181_relu_bias_swap.json`, `autoresearch/results/cycle_181_relu_bias_swap.png`

### Cycle 180

190. **FLAP-style bias compensation for topk pruning — GPT-2 FFN k=4** *(INCONCLUSIVE — cycle 180)*
   - Analytic bias compensation (absorbing removed channels' mean contribution as a constant bias) is significantly worse than lstsq pre-polish: mean PPL 91.9 vs 73.7. However, after 500-step Adam polish, the gap narrows dramatically: bias+polish 52.3 vs lstsq+polish 49.8 (only +2.5 PPL). Per-block MSE: bias reconstruction is ~2x worse than lstsq (e.g. block 11: 4.15 vs 1.62), but polish compensates. The analytic bias does not eliminate the need for data-dependent fitting — it's a worse initialization than lstsq, but polish can mostly close the gap.
   - Artifacts: `experiments/flap_bias_compensation_cycle180.py`, `autoresearch/results/cycle_180_flap_bias.json`

### Cycle 179

189. **PruneNet spectral channel selection — GPT-2 FFN k=4** *(REFUTED — cycle 179)*
   - Spectral greedy selection (minimizing KS-distance of W_down singular value distribution) selects very different channels from activation-magnitude topk (overlap: 7-38% depending on block) but performs consistently worse. Mean spectral PPL 53.8 vs topk 50.1 (+3.78 PPL, above +2 threshold). Pre-polish: spectral 76.2 vs topk 74.7 (+1.5). The weight-only spectral criterion misses data-dependent activation importance — channels with large W_down column norms don't necessarily carry high activation variance.
   - Artifacts: `experiments/spectral_channel_selection_cycle179.py`, `autoresearch/results/cycle_179_spectral_selection.json`, `autoresearch/results/cycle_179_spectral_selection.png`

### Cycle 178

188. **Negative-regime bias absorption + positive-only compression — GPT-2 FFN k=4** *(REFUTED — cycle 178)*
   - Two failure modes: (1) GELU->ReLU+b_leak absorption in 4 blocks costs +12.3% PPL (mean 58.7 vs 52.3 baseline), failing the <5% absorption criterion; (2) positive-only topk+lstsq with ReLU gives mean PPL 70.5 vs 50.1 for standard GELU topk (+34.7% worse). Root cause: ~94-99% of channels are sometimes positive (not ~9-13% as estimated), so there's no meaningful channel reduction. The GELU-to-ReLU substitution destroys the positive-regime shape near z=0, which cycle 175 showed is where the real prediction impact lies.
   - Artifacts: `experiments/neg_bias_absorption_compression_cycle178.py`, `autoresearch/results/cycle_178_neg_absorption.json`, `autoresearch/results/cycle_178_neg_absorption.png`

### Cycle 177

187. **SoLA-style split compression — GPT-2 FFN k=4** *(INCONCLUSIVE — cycle 177)*
   - Keep 512 top channels full-rank + SVD-compress remaining 2560 tail channels at rank r={2,4,8,16}. Best r=16 improves polished PPL by only 0.97 (58.42 vs 59.39) and unpolished by 0.55 (74.20 vs 74.75) — within ±2 PPL threshold. The split approach uses 3.57x more params per block (2.81M vs 787K) because it must keep the full c_fc to compute tail activations. The tail SVD captures only 15-22% of tail variance at r=16; the additional information is real but marginal for stacked PPL. Polish absorbs most of the local fit improvement, narrowing the gap.
   - Artifacts: `experiments/split_compression_cycle177.py`, `autoresearch/results/cycle_177_split_compression.json`, `autoresearch/results/cycle_177_split_compression.png`

### Cycle 176

183. **Sign-pattern-conditioned lstsq — GPT-2 FFN k=4** *(REFUTED — cycle 176)*
   - Piecewise lstsq conditioned on k-means sign-pattern clusters makes stacked PPL catastrophically WORSE: K=4 +13, K=8 +131, K=16 +76 PPL vs single lstsq baseline (2 seeds). Root cause: small cluster sizes (100-1000 tokens) make per-cluster lstsq underdetermined (512x768 params from <1000 samples), causing severe overfitting. Vindicates cycle 167 census: sign patterns don't cluster tractably and forced clustering destroys compression quality.
   - Artifacts: `experiments/sign_pattern_lstsq_cycle176.py`, `autoresearch/results/cycle_176_sign_lstsq.json`, `autoresearch/results/cycle_176_sign_lstsq.png`

### Cycle 175

186. **Per-token prediction impact of negative regime — GPT-2 FFN** *(CONFIRMED — cycle 175)*
   - GELU→ReLU substitution produces mean KL > 0.1 nats in 9/12 blocks. Block 0 dominates (KL=1.42). However, KL does NOT correlate with negative-regime norms (r=-0.16 with ||y_neg||, r=-0.33 with ||s_neg||). The prediction impact comes from the positive-regime shape difference (GELU≠ReLU near z=0), not from the negative regime. ReLU hurts 60-80% of tokens in early blocks but is coin-flip in blocks 9-10. Top-50 tokens show common-token overlap, not genuine semantic pattern.
   - Artifacts: `experiments/prediction_impact_gelu_relu_cycle175.py`, `autoresearch/results/cycle_175_prediction_impact.json`, `autoresearch/results/cycle_175_prediction_impact.png`

### Cycle 174

185. **Geometry of token-dependent negative signal — GPT-2 FFN** *(REFUTED — cycle 174)*
   - s_neg is orthogonal to prediction gradients: mean|cos(s_neg,J)| = 0.031 across all 12 blocks (well below 0.05 threshold). rank(S_neg) averages 469 vs rank(S_pos) 335 — negative signal is higher-rank, not lower (opposite of hypothesis). Neither s_neg nor s_pos shows meaningful per-token prediction alignment. Combined with cycle 173 (absorbable as bias), this confirms the negative regime is geometrically unstructured noise.
   - Artifacts: `experiments/neg_signal_geometry_cycle174.py`, `autoresearch/results/cycle_174_neg_signal_geometry.json`

### Cycle 172

182. **Lstsq regression target: residual-stream delta vs raw output — GPT-2 FFN k=4** *(REFUTED — cycle 172)*
   - All alternatives within 1 PPL of raw lstsq or worse. norm_wt (1/||x|| weighting): -0.12 PPL (noise). svd_sub (top-64 SVD subspace): +0.50 PPL (slightly worse). freq_wt (token-frequency weighting): +2.17 PPL (actively harmful). Polish absorbs initial differences, confirming the raw lstsq target is already near-optimal for the c_proj fitting step.
   - Artifacts: `experiments/lstsq_target_alt_cycle172.py`, `autoresearch/results/cycle_172_lstsq_targets.json`

### Cycle 171

181. **Wider-shallower compression: all 12 blocks at k=8 vs 4 blocks at k=4 — GPT-2 FFN** *(REFUTED — cycle 171)*
   - All wider-shallower configs are catastrophically worse: 12@3x PPL=183.4 vs 4@6x PPL=59.4 (joint polish). 12@2x=155.1, 8@4x=127.4. The c=0.41/block compounding rate dominates: touching more blocks always compounds more error, even at very light per-block compression. Block count is the primary cost, not per-block ratio. This definitively closes the "spread compression thinly" hypothesis.
   - Artifacts: `experiments/wider_shallower_compression_cycle171.py`, `autoresearch/results/cycle_171_wider_shallower.json`

### Cycle 170

180. **Literature scan: FFN compression 2025-2026 — what are we missing?** *(CONFIRMED — cycle 170)*
   - Scanned 11 methods from AAAI/ICLR/NeurIPS/NAACL/ACL/COLING 2024-2025. Found >=2 untried methods with comparable approaches: (1) **SoLA split** — preserve top-k channels full-rank + SVD-compress the rest (vs our prune+reconstruct), (2) **PruneNet spectral selection** — calibration-free channel selection via singular value distribution preservation (vs our activation-magnitude topk). Also notable: 2SSP combines FFN width pruning with attention depth pruning (untried combo). FLAP, CFSP, LLM-BIP are structured neuron pruning variants close to what we do. StructuredFFN/Spark train from scratch (out of scope).
   - Artifacts: `autoresearch/results/cycle_170_literature_scan.md`

### Cycle 169

179. **Does constant bias shift under compression? — GPT-2 FFN** *(REFUTED — cycle 169)*
   - b_leak is stable at uncompressed blocks under k=4 compression: blocks 0,2,3,6 shift <3.4%, blocks 8-11 shift 7-10% (cascade accumulation). Compressed blocks shift 59-93% but this is expected — they have entirely different W_down from lstsq fitting, not a cascade effect. Fixing compressed blocks' b_leak to float values hurts PPL (52.3 → 119.5), confirming the compressed model's b_leak is already adapted to its new projection. The 41%/block cascade is NOT caused by bias instability.
   - Artifacts: `experiments/bias_shift_compression_cycle169.py`, `autoresearch/results/cycle_169_bias_shift.json`, `autoresearch/results/cycle_169_bias_shift.png`

### Cycle 168

178. **Training dynamics of negative z — GPT-2 FFN** *(CONFIRMED — cycle 168)*
   - z_mean negativity is universal across GPT-2 scales: small=-0.97, medium=-1.12, large=-1.43 (max_ratio=1.47, all < -0.3). Mild scale trend: larger models more negative (1.47x). Dead fraction grows 84.5% → 90.8% with scale. Bias contribution shrinks from ~8% (small) to ~2% (large) — Wx anti-alignment increasingly dominant at scale. Combined with cycle 165 (random-init ≈ 0), negativity is both learned and universal.
   - Artifacts: `experiments/zmean_scale_comparison_cycle168.py`, `autoresearch/results/cycle_168_scale_comparison.json`, `autoresearch/results/cycle_168_scale_comparison.png`

### Cycle 167

177. **Sign-pattern census on actual data — GPT-2 FFN** *(REFUTED — cycle 167)*
   - All 4096 tokens produce unique sign patterns in every block (zero redundancy). K* > 256 in 11/12 blocks (best within-cluster variance ratio at k=128: 0.68 block 0). Even with 87-88% channels locked negative, the ~12-27% flipping channels create enough combinatorial diversity that sign patterns are effectively unique per token. Hamming distances: 400-900 (mean), spanning 13-30% of 3072 channels. k-means clustering fails to capture FFN output structure — sign patterns alone do not define a tractable piecewise partition. Block 2 anomaly at k=256 (ratio=0.02) is a statistical artifact from 16 samples/cluster.
   - Artifacts: `experiments/sign_pattern_census_cycle167.py`, `autoresearch/results/cycle_167_sign_census.json`, `autoresearch/results/cycle_167_sign_census.png`

### Cycle 166

176. **Functional role of constant negative bias — GPT-2 FFN** *(CONFIRMED — cycle 166)*
   - ReLU + b_leak (mean negative output) recovers within +5% PPL in 11/12 blocks (mean delta +2.38%, max +5.91% at block 4). The negative GELU regime is functionally a constant bias per block. However, full-model replacement (all 12 blocks) degrades PPL by +87% — cascade accumulation of small per-block errors. Per-block, the negative regime = 3072 fixed params of redundancy. Block 4 is the only outlier (5.91%). b_leak norms increase with depth (4.05 → 5.32).
   - Artifacts: `experiments/constant_neg_bias_gpt2_cycle166.py`, `autoresearch/results/cycle_166_neg_bias.json`, `autoresearch/results/cycle_166_neg_bias.png`

### Cycle 165

175. **Why is z_mean negative? Weight-data alignment decomposition — GPT-2 FFN** *(CONFIRMED — cycle 165)*
   - Negativity is LEARNED, not geometric accident. Trained W_up @ x mean ≈ -0.61 to -1.06 per block; random-init W_rand @ x mean ≈ 0 (±0.001). Mean shift = 0.886 across all 12 blocks. Per-channel cosine alignment: 83-98% of c_fc rows are anti-aligned with E[x], mean cos ≈ -0.06 to -0.11. Decomposition: z_mean ≈ -1, b_mean ≈ -0.08, Wx_mean ≈ -0.9 (dominant). Bias contributes only ~8-10% of total negativity; W_up @ x is the primary driver. Network has learned weight rows that systematically point opposite to the mean input direction.
   - Artifacts: `experiments/weight_data_alignment_cycle165.py`, `autoresearch/results/cycle_165_alignment.json`, `autoresearch/results/cycle_165_alignment.png`

### Cycle 164

174. **Positive-channel ablation — GPT-2 FFN** *(CONFIRMED — cycle 164)*
   - Killing positive channels (z > 0 → 0, keep negative leak) causes +3,141,890% PPL full-model (22.7 → 713,227). Token-mean replacement even worse: +149,216,407% (33.9M PPL). Per-block: block 0 alone +970,095%, middle blocks 6-23% each, block 11 +89%. Positive channels carry the primary computational signal; negative leak alone is insufficient. ||y_pos|| > ||y_neg|| at all blocks, ratio increasing with depth (1.37→2.56). Positive regime is essential — both regimes needed.
   - Artifacts: `experiments/positive_channel_ablation_gpt2.py`, `autoresearch/results/cycle_164_positive_ablation.json`, `autoresearch/results/cycle_164_positive_ablation.png`

### Cycle 163

173. **Per-block error contribution profiling — stacked topk+lstsq — GPT-2 FFN k=4** *(REFUTED — cycle 163)*
   - 3 seeds [42,137,7], 12 blocks progressive stacking, baseline PPL=52.3. Degradation is approximately uniform: top-3 blocks contribute only 36% of total (need >60%), max single-block contribution 12.6% (need >15%). Correction concentration: Condition A (top-3 r=8)=60.3±9.2, Condition B (top-3 r=11)=60.4±9.3, uniform (4 blocks r=8)=58.6±8.3. Concentrated corrections 1.7 PPL worse than uniform. Degradation profile is approximately uniform (blocks 7-11 each 10-12%, blocks 0-6 each 3-10%), supporting hypothesis that error accumulates diffusely across all blocks rather than concentrating in a few.
   - Artifacts: `experiments/per_block_stacking_profile_cycle163.py`, `autoresearch/results/cycle_163_per_block_profile.json`, `autoresearch/results/cycle_163_per_block_profile.png`

### Cycle 162

172. **GELU-aware dead and near-saturated channel census — GPT-2 FFN** *(REFUTED — cycle 162)*
   - 3 seeds × 256 seqs × 256 tokens, T∈{2,3,4}. Total qualifying at T=2: 5/36864 (0.01%). Zero qualifying at T=3,4. Near-linear channels: 0 at all thresholds. Only blocks 0 (4 channels) and 1 (1 channel) had any dead channels at T=2. Block 0 prune (4 channels): PPL delta +0.00%. Bottom 1% by contribution score (30 ch): PPL delta -0.07%. Bottom 5% (153 ch): PPL delta +0.12%. T=2 threshold adds nothing vs cycle 128: GPT-2 FFN channels are fully active across the OWT distribution at all tested thresholds. Contribution scores follow a smooth distribution with no clear mass below null baseline (1/3072 = 0.033%).
   - Artifacts: `experiments/gelu_channel_census_cycle162.py`, `autoresearch/results/cycle_162_census.json`, `autoresearch/results/cycle_162_census.png`

### Cycle 161

171. **Centroid-affinity channel selection vs activation-norm topk — GPT-2 FFN k=4** *(REFUTED — cycle 161)*
   - Centroid-FPS stacked PPL = 88.311 ± 0.046 vs topk = 87.684 ± 0.062. Delta = -0.627 (centroid *worse* by 0.6 PPL). Jaccard = 0.333 on all blocks (methods select distinct channels), so the geometric signal simply points to suboptimal channels. The cycle-27 centroid signal (inter-block merge predictor, rho=0.853) does not transfer to intra-block channel selection.
   - Artifacts: `experiments/centroid_affinity_selection_cycle161.py`, `autoresearch/results/cycle_161_centroid_affinity.json`, `autoresearch/results/cycle_161_centroid_affinity.png`

### Cycle 160

170. **Block-adaptive compression ratio via per-block sensitivity — GPT-2 FFN** *(INCONCLUSIVE — cycle 160, PLAN INVALID)*
   - Duplicates refuted items 110 (cycle 94) and 65 (cycle 50). Non-uniform allocation already shown to be absorbed by polish.

### Cycle 159

169. **Correction-first compression: train corrections, then compress residual — GPT-2 FFN k=4** *(INCONCLUSIVE — cycle 159)*
   - Correction-first (retrained) PPL = 60.0 ± 7.4 vs Phase A (compress+correct) = 59.2 ± 7.4. Improvement vs A = -0.81 (marginal, in [57, 62] range). OOD triggered on all seeds (cosine sims ~0, corrections trained on compressed residuals are OOD for float residuals). lstsq ratio <1 for all blocks (cf targets are easier to fit), but PPL difference is noise-level. Corrections can't transfer directly: transferred PPL=75.3 vs retrained 60.0. The subspace corrections occupy is not separable from what compression occupies.
   - Artifacts: `experiments/correction_first_cycle159.py`, `autoresearch/results/cycle_159_correction_first.json`, `autoresearch/results/cycle_159_correction_first.png`

### Cycle 158

168. **Topk+lstsq init for joint training — GPT-2 FFN k=4** *(REFUTED — cycle 158, audit-corrected)*
   - Warm-start PPL = 73.0 ± 0.23 (3 seeds). Better than random-init (91.1) but worse than topk+polish (70.7). PPL 73.0 >= 69.3 (REFUTED threshold: within 2% of topk+polish). Rank expands 4→21.5 confirming training left the init subspace, but warm start provides no PPL advantage over polish. Joint training from warm start is strictly worse than simple polish.
   - Artifacts: `experiments/topk_init_joint_cycle158.py`, `autoresearch/results/cycle_158_topk_init_joint.json`

### Cycle 157

167. **Higher-rank joint training (rank-128, rank-256) — GPT-2 FFN k=4** *(REFUTED — cycle 157)*
   - Higher rank dramatically worse: r64=91.1, r128=175.4, r256=621.9. More params = worse optimization, not better. Problem is optimization landscape, not capacity.
   - Artifacts: `experiments/higher_rank_joint_cycle157.py`, `autoresearch/results/cycle_157_higher_rank_joint.json`

### Cycle 156

166. **Joint training with correction-aware loss — GPT-2 FFN k=4** *(REFUTED — cycle 156)*
   - Joint compress+correct PPL=110.0, worse than compress-only (91) and topk+corr (59). Corrections during joint training make optimization harder.
   - Artifacts: `experiments/joint_compress_correct_cycle156.py`, `autoresearch/results/cycle_156_joint_compress_correct.json`

### Cycle 155

165. **Multi-seed validation of joint training — GPT-2 FFN k=4** *(REFUTED — cycle 155)*
   - Median 91.1±1.0 across 8 seeds. Cycle 149's 69.76 was an anomaly. Rank-64 joint training consistently ~91 PPL.
   - Artifacts: `experiments/joint_multiseed_cycle155.py`, `autoresearch/results/cycle_155_joint_multiseed.json`

### Cycle 154

164. **Joint training step budget sweep — GPT-2 FFN k=4** *(INCONCLUSIVE — cycle 154, PLAN INVALID)*
   - Joint training seed-dependent (91.9 vs 69.76) and CE models resist corrections. Sweep pointless until seed-dependence resolved.

### Cycle 153

163. **Joint training + LoRA corrections — GPT-2 FFN k=4** *(REFUTED — cycle 153)*
   - **Outcome: REFUTED** — CE-trained models resist post-hoc corrections. Joint+corrections = 107.5 (catastrophic). Topk+corrections = 69.8 (works). Joint PPL = 91.9 (seed-dependent, vs 69.76 in cycle 149).
   - Commit: 34ea5f8
   - Artifacts: experiments/joint_plus_corrections_cycle153.py, autoresearch/results/cycle_153_joint_corrections.json

### Cycle 152

162. **Progressive stacking — train-then-freeze — GPT-2 FFN k=4** *(REFUTED — cycle 152)*
   - **Outcome: REFUTED** — Progressive = 78.2, worse than joint (69.8), topk (70.7), and independent (74.4). Greedy per-block can't adapt to cumulative error.
   - Commit: 7911858
   - Artifacts: experiments/progressive_stacking_cycle152.py, autoresearch/results/cycle_152_progressive_stacking.json

### Cycle 151

161. **Correction subspace analysis on unpolished model — GPT-2 FFN k=4** *(CONFIRMED — cycle 151)*
   - **Outcome: CONFIRMED** — Novel subspace dominates corrections (93-97%). Block 7 correction is almost entirely cascade compensation (novel recovery=110%, error recovery=34%). Corrections reroute around damaged subspace.
   - Commit: 6fefec8
   - Artifacts: experiments/correction_subspace_unpolished_cycle151.py, autoresearch/results/cycle_151_correction_subspace.json

### Cycle 150

160. **Linear factorization + CE training (single-block) — GPT-2 FFN k=4** *(INCONCLUSIVE — cycle 150, PLAN INVALID)*
   - Superseded by cycles 148 (linear=ReLU at single-block) and 149 (joint training works at stacking).

### Cycle 149

159. **Joint stacking-aware compression training — GPT-2 FFN k=4** *(INCONCLUSIVE — cycle 149)*
   - **Outcome: INCONCLUSIVE** — Random-init rank-64 = 69.76 (beats topk 70.70 by 1.3%), but SVD init fails (78.2) and 2% threshold not met. Random beats structured init with 1.6x fewer params, but seed-dependent.
   - Commit: 7344625
   - Artifacts: experiments/joint_stacking_training_cycle149.py, autoresearch/results/cycle_149_joint_stacking.json

### Cycle 148

158. **Linear AE ablation — is encoder nonlinearity necessary? — GPT-2 FFN** *(REFUTED — cycle 148)*
   - **Outcome: REFUTED** — Linear AE matches ReLU AE within 0.1-0.7% (B4: relu=54.34, linear=54.30; B7: relu=55.28, linear=54.91). Nonlinearity is cosmetic. The advantage over topk comes from the CE training objective, not encoder architecture.
   - Commit: 0844df4
   - Artifacts: experiments/linear_ae_ablation_cycle148.py, autoresearch/results/cycle_148_linear_ae_ablation.json

### Cycle 147

157. **Stacked AE with inter-block error feedback — GPT-2 FFN k=4** *(INCONCLUSIVE — cycle 147, PLAN INVALID)*
   - Predecessor 152 REFUTED (AE stacking worse). Plan invalid.

### Cycle 146

156. **Nanochat asset audit and benchmark lock-in** *(INCONCLUSIVE — cycle 146)*
   - **Outcome: INCONCLUSIVE** — No nanochat checkpoints or climbmix data available. Model code present but not importable. All cycles remain GPT-2 proxy.
   - Artifacts: autoresearch/results/cycle_146_nanochat_audit.md

### Cycle 145

150. **Curvature-salience guided correction rank allocation — GPT-2 FFN k=4** *(INCONCLUSIVE — cycle 145, PLAN INVALID)*
   - Triple invalidation: dep 146 refuted, allocation doesn't help (cycle 114), polish convergence issues.

### Cycle 144

149. **Layer-local pullback metric via rest-of-network Jacobian — GPT-2 FFN** *(REFUTED — cycle 144)*
   - **Outcome: REFUTED** — Layer-local pullback rho=0.538, WORSE than Frobenius (rho=0.797). Downstream amplification is non-monotonic. Global Frobenius remains best per-block error metric.
   - Commit: f35ece2
   - Artifacts: experiments/layer_local_pullback_cycle144.py, autoresearch/results/cycle_144_layer_local_pullback.json

### Cycle 143

155. **Eigenstructure-aware channel selection — GPT-2 FFN** *(REFUTED — cycle 143)*
   - **Outcome: REFUTED** — Whitened importance is worse than topk on 3/4 blocks (B1 +0.3%, B4 +0.4%, B7 +2.8% worse) and marginally better on B11 (-1.3%). Covariance whitening hurts channel selection — decorrelating removes structural information that lstsq uses. Learned isotropy doesn't help channel selection. Zero seed variance confirms deterministic result.
   - Commit: PENDING
   - Artifacts: experiments/eigenstructure_channel_selection_cycle143.py, autoresearch/results/cycle_143_eigenstructure.json

### Cycle 142

154. **Correction subspace analysis — GPT-2 FFN k=4** *(INCONCLUSIVE — cycle 142)*
   - **Outcome: INCONCLUSIVE** — Method failure: corrections trained on already-polished model WORSENED PPL (71.0 → 88.7). Corrections compete with polish optimization. Subspace decomposition and ablation results are not interpretable when corrections are harmful. Need to reproduce the cycle 115 protocol (corrections on unpolished model) for valid results. Not a genuine negative — the cycle 115 corrections DO work (PPL 59), this was a training setup error.
   - Commit: PENDING
   - Artifacts: experiments/correction_subspace_cycle142.py, autoresearch/results/cycle_142_correction_subspace.json

### Cycle 141

153. **Graduated importance pruning — GPT-2 FFN** *(INCONCLUSIVE — cycle 141)*
   - **Outcome: INCONCLUSIVE** — Importance pruning consistently beats topk at all 12 configs (4 blocks × 3 pruning levels) by 0.2-2.0%. Results are deterministic (zero seed variance). B11@20% hits exactly 2.0%. Advantage grows with pruning aggressiveness. But doesn't reach 2% threshold on >=2 blocks. Importance signal is real but topk's activation-magnitude selection is already good — importance adds marginal refinement, not a step change.
   - Commit: PENDING
   - Artifacts: experiments/importance_pruning_k4_cycle141.py, autoresearch/results/cycle_141_importance_pruning.json

### Cycle 139

147. **G-weighted lstsq — GPT-2 FFN** *(INCONCLUSIVE — cycle 139, PLAN INVALID)*
   - Predecessor 146 REFUTED: G-weighted is constant rescaling of Frobenius. G-lstsq would be equivalent to standard lstsq.

### Cycle 140

148. **Curvature-guided allocation — GPT-2 FFN** *(INCONCLUSIVE — cycle 140, PLAN INVALID)*
   - Predecessor 146 REFUTED + cycle 114 showed allocation doesn't help. Double invalidation.

### Cycle 138

152. **Bottleneck AE k=4 stacking — GPT-2 FFN** *(REFUTED — cycle 138)*
   - **Outcome: REFUTED** — AE stacking (74.4) is 4.7% WORSE than topk (71.0). Pre-polish AE stacking is catastrophic (259.8 vs 84.9) — independently trained AEs don't compose. Even after 500-step joint polish, AE can't match topk. AE+LoRA corrections (76.7) even worse. The 3% single-block AE advantage (cycle 137) does NOT transfer to multi-block stacking. Cascade/composition dominates local quality.
   - Commit: 1626479
   - Artifacts: experiments/ae_stacking_k4_cycle138.py, autoresearch/results/cycle_138_ae_stacking.json

### Cycle 137

151. **Bottleneck AE param-matched — GPT-2 FFN k=512** *(CONFIRMED — cycle 137)*
   - **Outcome: CONFIRMED** — Param-matched AE (d_bn=64, 247K params) beats topk+lstsq (788K params) on both blocks post-polish. Block 4: AE=53.65±0.10, topk=55.32±0.04 (-3.0%). Block 7: AE=54.46±0.17, topk=56.36±0.02 (-3.4%). The AE uses 3x FEWER learnable params. Full AE (1.97M) actually worsens after polish (+1-4% vs topk). The nonlinear ReLU bottleneck at d_bn=64 learns a better subspace than axis-aligned top-512 selection, and this advantage SURVIVES 500-step Adam polish.
   - Commit: 1d89948
   - Artifacts: experiments/ae_param_match_k512_cycle137.py, autoresearch/results/cycle_137_ae_param_match.json

### Cycle 136

146. **Pullback-metric error measurement — GPT-2 FFN** *(REFUTED — cycle 136)*
   - **Outcome: REFUTED** — Frobenius (rho=0.797) beats G-weighted (rho=0.734) for predicting PPL damage. G/Frob ratio is nearly constant (~34-35) across blocks 0-10, meaning pullback metric is a constant rescaling at FFN output level. Block 11 anomalous (60x ratio) but this outlier hurts G-weighted correlation. Frobenius is already sufficient as per-block error metric.
   - Commit: e1f3c06
   - Artifacts: experiments/pullback_error_cycle136.py, autoresearch/results/cycle_136_pullback_error.json, autoresearch/results/cycle_136_scatter.png

### Cycle 135

145. **Curved inference census — GPT-2 residual stream** *(CONFIRMED — cycle 135)*
   - **Outcome: CONFIRMED** — Strong salience-curvature anticorrelation rho=-0.745 (p=0.009) and curvature-damage correlation |rho|=0.645 (p=0.032). Curvature peaks in early layers (2-4, kappa≈0.002) and decays to near-zero in late layers (10-11). Late layers have low curvature but highest compression damage — the trajectory straightens under G where semantic precision matters most. Block 0 is anomalous (high damage, near-zero curvature). Consistent across 3 seeds.
   - Commit: 275e8c3
   - Artifacts: experiments/curved_inference_census_cycle135.py, autoresearch/results/cycle_135_curved_inference.json, autoresearch/results/cycle_135_curved_inference.png

### Cycle 134

135. **Data-conditioned affine map extraction — GPT-2 FFN** *(INCONCLUSIVE — cycle 134)*
   - **Outcome: INCONCLUSIVE** — No meaningful Jacobian clustering. Block 4: 128/128 groups at all thresholds (max pairwise cos=0.87, mean=0.33). Block 7: 124/128 groups (few pairs with cos>0.95, mean=0.27). GELU FFNs lack the discrete piecewise-affine structure of ReLU networks. Where rare similar pairs exist (14 in block 7), interchangeability works (mean error 1.8%, 86% below 5%). The MASO framework doesn't directly apply to GELU networks.
   - Commit: e2e1136
   - Artifacts: experiments/affine_map_extraction_cycle134.py, autoresearch/results/cycle_134_affine_maps.json

### Cycle 133

127. **Direct regression hybrid — GPT-2 FFN** *(REFUTED — cycle 133)*
   - **Outcome: REFUTED** — All-topk (70.12±0.05) beats hybrid (70.49±0.02) post-polish. All-direct worst (72.84±0.09). Hybrid wins pre-polish (83.6 vs 84.9) but polish erases the difference. Topk is robust enough that block-adaptive method selection doesn't help. Very low seed variance (~0.05 PPL) confirms the result.
   - Commit: 83098db
   - Artifacts: experiments/hybrid_regression_topk_cycle133.py, autoresearch/results/cycle_133_hybrid_regression.json

### Cycle 132

126. **Attention projection pruning on high-amplification heads — GPT-2** *(REFUTED — cycle 132)*
   - **Outcome: REFUTED** — Composed W_V@W_O SVD pruning catastrophically degrades PPL. Best rank=16 (25% of head dim): FFN-only=95.3, FFN+SVD=338.5 (+243). Even preserving 25% of each head's value-output subspace destroys model function. Attention heads carrying cascade error also carry essential information. 3 seeds, all consistent (std ~10 on delta >240).
   - Commit: 02be78c
   - Artifacts: experiments/attn_proj_pruning_cycle132.py, autoresearch/results/cycle_132_attn_pruning.json

### Cycle 131

88. **Learned bottleneck AE for FFN compression — GPT-2 FFN** *(INCONCLUSIVE — cycle 131)*
   - **Outcome: INCONCLUSIVE** — AE consistently beats topk+lstsq at k=384: block 4 by -2.0% (56.15 vs 57.28), block 7 by -3.4% (55.86 vs 57.85). Both blocks beat float PPL (57.65). Nonlinearity ~5% (boundary). AE improvement is real and consistent (3 seeds, low std) but doesn't reach 5% CONFIRMED threshold. AE has 2.5x more params than topk+lstsq — param-matched test needed. POC only; full k-sweep, stacking, and k=128 test deferred.
   - Commit: ce37a8d
   - Artifacts: experiments/bottleneck_ae_poc_cycle131.py, autoresearch/results/cycle_131_bottleneck_ae.json

### Cycle 130

87. **Cross-block weight sharing via Procrustes alignment — GPT-2 FFN** *(REFUTED — cycle 130)*
   - **Outcome: REFUTED** — All 6 pairwise Procrustes residuals > 84% (min: blocks 4-5 = 84.4%, max: blocks 4-7 = 98.0%). Compressed W_down matrices are completely functionally dissimilar — no cross-block weight sharing is possible. Each block's compression occupies a distinct subspace.
   - Commit: 190ee71
   - Artifacts: experiments/procrustes_alignment_cycle130.py, autoresearch/results/cycle_130_procrustes.json, autoresearch/results/cycle_130_heatmap.png

### Cycle 129

100. **Is FFN isotropy architectural or learned? — GPT-2 FFN** *(CONFIRMED — cycle 129)*
   - **Outcome: CONFIRMED** — Random-init GPT-2 has isotropy ratio 0.002 across all blocks (3 seeds, std<0.0002). Trained GPT-2: block 1=0.004, blocks 4-7=0.015. Training increases isotropy 7-8x in blocks 4-7 (learned). Block 1 is anomalous (only 1.5x increase — partially architectural). ReLU swap (qualitative): 0.014 vs GELU 0.015, negligible change — activation function doesn't drive isotropy.
   - Commit: 6a82731
   - Artifacts: experiments/ffn_isotropy_architectural_gpt2.py, autoresearch/results/cycle_129_isotropy_architectural.json, autoresearch/results/cycle_129_isotropy_comparison.png

### Cycle 128

134. **Dead/always-active neuron census + exact pruning — GPT-2 FFN** *(INCONCLUSIVE — cycle 128)*
   - **Outcome: INCONCLUSIVE** — 0/36864 strict-removable (0.00%). No dead, always-active, or near-linear neurons under universal thresholds. Soft census: 14 channels in blocks 0-4 have >95% tokens in dead zone (z<-3), consistent across 3 seeds. These are approximate pruning candidates but too few (<0.5%) for meaningful compression. GELU networks have no free compression from stable channels.
   - Commit: 5ae03ec
   - Artifacts: experiments/dead_neuron_census_gpt2_cycle128.py, autoresearch/results/cycle_128_census.json, autoresearch/results/cycle_128_census.png

### Cycle 127

144. **SVD of learned corrections: geometric alignment — GPT-2** *(INCONCLUSIVE — cycle 127)*
   - **Outcome: INCONCLUSIVE** — Corrections don't strongly align with either compression error or attention amplification directions. Max alignment with error: 0.46 (block 5), 0.44 (block 4) — above null p99 (0.13) but below 0.5 threshold. Attention alignment weak (max 0.25). Block 1 attention delta is zero (no upstream compressed blocks to perturb). Corrections learn a partially novel subspace; moderate overlap with error directions in middle blocks.
   - Commit: f41a7e7
   - Artifacts: experiments/svd_alignment_cycle127.py, autoresearch/results/cycle_127_svd_alignment.json, autoresearch/results/cycle_127_alignment_heatmap.png

### Cycle 126

143. **Best correction placement at scale — GPT-2 FFN k=4** *(INCONCLUSIVE — cycle 126, PLAN INVALID)*
   - **Outcome: INCONCLUSIVE** — Predecessor item 142 (cycle 125) was REFUTED, invalidating premise. Placement is near-optimal at P1; no scale validation needed.

### Cycle 125

142. **Correction placement sweep — GPT-2 FFN k=4** *(REFUTED — cycle 125)*
   - **Outcome: REFUTED** — P1 (after FFN, residual stream) is within 2% of best alternative. Ranking: P1=69.52 (69.0% recovery), P5=70.18 (67.3%), P2=70.97 (65.2%), P4=71.65 (63.5%), P3=80.85 (39.4%). P1 reproduction matches cycle 115 (69.52 vs 69.53). Residual-stream correction is already near-optimal; attention-internal corrections (P3/P4) are worse, especially Q+K (r=4 may be insufficient but P4 at r=4 also underperforms P1 at r=8). Item 143 deprioritized.
   - Commit: 4371880
   - Artifacts: experiments/placement_sweep_k4_cycle125.py, autoresearch/results/cycle_125_placement_sweep.json, autoresearch/results/cycle_125_ppl_bar.png, autoresearch/results/cycle_125_loss_curves.png

### Cycle 124

141. **Attention entropy shift under FFN compression — GPT-2** *(INCONCLUSIVE — cycle 124)*
   - **Outcome: INCONCLUSIVE** — Entropy shift exists but is modest. Only 2/12 layers exceed 0.1 nat threshold (L5: 0.113, L9: 0.103). Mean |delta_H| across all layers = 0.059 nats. Shift is non-zero (not Refuted: max = 0.113 nats, vs 0.02 floor). Significant negative Spearman correlation with cycle 104 attn_frac: rho=-0.55, p=0.083 — borderline significant but wrong direction (high-cascade-amplification layers show LESS entropy shift, not more). Cross-seed std < 0.01 (very stable across seeds). Float PPL=57.65, compressed PPL=71.65 (+24.3%). Interpretation: FFN compression does perturb attention distributions, but the perturbation is concentrated in downstream layers (L5, L8-L10), not at the compressed blocks [1,4,7,11] themselves. The negative correlation with attn_frac suggests cascade amplification is not driven by attention pattern disturbance but by V-space corruption (consistent with cycle 123 INCONCLUSIVE finding).
   - Commit: 2186270baa5f3f170685241a11290816c7b6bf15
   - Artifacts: experiments/attn_entropy_shift_gpt2.py, autoresearch/results/cycle_124_attn_entropy_shift.json, autoresearch/results/cycle_124_attn_entropy_shift.png

### Cycle 123

140. **QK vs V error decomposition under FFN compression — GPT-2** *(INCONCLUSIVE — cycle 123)*
   - **Outcome: INCONCLUSIVE** — Dominant source varies by layer position: V dominates early layers (L2-L4: V_frac ~0.65-0.68), QK dominates late layers (L9-L11: QK_frac ~0.55-0.58). No single component exceeds 70% threshold across all 3 seeds in any layer. Float PPL=57.65, compressed k=4 PPL=71.66 (+24.3%). Cross-seed std < 0.02 (very stable). Layers 0-1 show E_total≈0 (clean residual stream before first compressed block, confirming sanity). Interaction terms generally small (|inter_frac| < 0.1 in most layers). Layer 5 anomaly: QK_frac=0.578 (high QK, likely attention layer sensitive to early residual stream corruption from block 1 and block 4 compressions). No surgical single correction target exists; correction strategies must target both QK and V pathways.
   - Commit: dbfd59fc17e1c08e00b04e407bd768449cc62c65
   - Artifacts: experiments/qk_v_decomposition_gpt2.py, autoresearch/results/cycle_123_qk_v_decomposition.json, autoresearch/results/cycle_123_qk_v_heatmap.png

### Cycle 121

138. **Train MoE testbed — Tiny Mixtral** *(INCONCLUSIVE — cycle 121, audit-corrected)*
   - **Outcome: INCONCLUSIVE** — Best val PPL=65.08 at 15M tokens, then overfitting (no dropout used). Training recipe failure, not architecture limitation: no dropout/regularization beyond weight_decay=0.01, and WikiText-2 (~2M tokens) causes 7.5x epoch repeats at 15M tokens. Effective model size ~5.5M params (top-2 routing). Needs retry with dropout=0.1 and wider lr sweep.
   - Artifacts: experiments/moe_train_cycle121.py, experiments/results/cycle_121_tiny_mixtral.pt, autoresearch/results/cycle_121_moe_training.json

### Cycle 120

137. **Cascade corrections at k=8 stacking — GPT-2 FFN** *(CONFIRMED with caveat — cycle 120)*
   - **Outcome: CONFIRMED (metric caveat)** — r=8 corrections on k=8 blocks [1,2,3,4,5,6,7,9] at 6x recovered 91.4% of degradation by formula (seed 42). Baseline 57.65, uncorrected 1055.3, corrected 143.6. Recovery exceeds 20% threshold. However, the 91.4% figure is misleading: it is driven by the enormous 997.7-point denominator. Corrected PPL 143.6 is +149% above baseline — worse than k=5 uncorrected (122.1). Absolute corrected PPL trend across k: 59.1 (k=4) -> 70.1 (k=5) -> 143.6 (k=8) shows corrections become less effective as k grows. 98,304 correction params. [Audit: cycle_120.md — CORRECTED, misleading recovery metric]
   - Artifacts: experiments/cascade_correction_k8_cycle120.py, autoresearch/results/cycle_120_cascade_correction_k8.json, autoresearch/results/cycle_120_cascade_correction_k8.png

### Cycle 119

136. **Cascade corrections at k=5 stacking — GPT-2 FFN** *(CONFIRMED — cycle 119)*
   - **Outcome: CONFIRMED** — r=8 corrections on k=5 blocks [7,5,4,1,6] at 6x recovered 73.8% of degradation (mean across 3 seeds). Baseline ~51.7 PPL, uncorrected ~122.1 PPL (+70.5 deg), corrected ~70.1 PPL (+18.5 residual). Recovery 73.8% exceeds 30% threshold. Near-identical to k=4 recovery (74.3%, cycle 115), showing cascade corrections scale gracefully from 4→5 blocks. 61,440 correction params.
   - Artifacts: experiments/cascade_correction_k5_gpt2.py, autoresearch/results/cycle_119_cascade_correction_k5.json, autoresearch/results/cycle_119_cascade_correction_k5.png

### Cycle 118

128. **Train MoE testbed + REAP expert pruning baseline — Tiny Mixtral** *(INCONCLUSIVE — cycle 118)*
   - **Outcome: INCONCLUSIVE** — Model trained 20M tokens on WikiText-2, final val PPL=78.65 (>50 threshold). Extended run (20M tokens, fallback) still did not reach PPL<50. REAP saliency scoring works correctly: Gini>0.3 in all 4 layers (0.481, 0.351, 0.345, 0.377). Pruning is monotone (baseline=78.65, REAP25%=90.21, REAP50%=103.25, REAP75%=231.98). REAP decisively outperforms random (REAP50%=103.25 vs random50%=299.58±57.65 — better by >3 std). Per-layer ablation: Layer 0 most sensitive (ΔPPL=+8.58), Layer 1 least sensitive (Δ=+0.06). PPL>50 due to under-training (tiny vocab=4096, shallow 4-layer model). Saliency infrastructure and REAP implementation confirmed working.
   - Artifacts: experiments/moe_reap_pruning_tiny_mixtral.py, experiments/results/cycle_118_tokenizer/, experiments/results/cycle_118_tiny_mixtral.pt, autoresearch/results/cycle_118_moe_reap.json, autoresearch/results/cycle_118_moe_reap.png
   - Commit: 8a9d0f2

### Cycle 117

133. **Cascade corrections at k=5 and k=8 stacking — GPT-2 FFN** *(CRASH — cycle 117)*
   - **Outcome: CRASH (timeout)** — Killed after 140+ min CPU. Design problem: k=5+k=8 × 3 seeds × corrections training far exceeds 30-min cap. Superseded by items 136/137.
   - Artifacts: autoresearch/briefs/cycle_117.md, autoresearch/plans/cycle_117.md, autoresearch/reviews/cycle_117.md

### Cycle 106

101. **Gradient polish on k=5 stacking — GPT-2 FFN** *(CONFIRMED — cycle 106)*
   - **Outcome: CONFIRMED** — Joint W_down polish (500 Adam steps, lr=1e-4) reduces k=5 stacking PPL delta by ~61% relative for both configs. Config [7,5,1,3,6]: 0-step +62.8% → 500-step +24.3% (61.3% reduction). Config [7,5,4,1,6]: 0-step +57.7% → 500-step +22.4% (61.2% reduction). Baseline GPT-2 PPL 57.65. At 200 steps: ~52% reduction already achieved. No overfitting flagged. Train loss monotonically decreasing (minor non-monotonicity in 1 run first 10 steps only).
   - Artifacts: experiments/gradient_polish_k5_stacking_gpt2.py, autoresearch/results/cycle_106_k5_polish.json, autoresearch/results/cycle_106_k5_polish.png

### Cycle 113

124. **Attention-mediated cascade: per-head amplification census — GPT-2** *(REFUTED — cycle 113)*
   - **Outcome: REFUTED** — Amplification is near-uniform across heads in 8/11 layers (max per head ≤15%). top3_frac ranges 0.29–0.39 in 10/11 transitions (far below 70% confirmed threshold). One exception: Block 10→11, top3_frac=0.722, max_frac=0.352 (only layer). Gini coefficients low (0.06–0.28) confirming near-uniform spread. The cascade error induced by FFN compression propagates roughly equally through all 12 attention heads — no surgical target heads exist. Ablation PPL deltas (block 6): 0.04–0.59, Spearman rho=0.636 (p=0.026) — moderate correlation between amplification and importance, but amplification is too diffuse for targeted suppression.
   - Artifacts: experiments/per_head_amplification_census_gpt2.py, autoresearch/results/cycle_113_head_amplification.json, autoresearch/results/cycle_113_heatmap.png, autoresearch/results/cycle_113_summary.png, autoresearch/results/cycle_113_top10_heads.png

### Cycle 104

115. **Cascade error source: residual-stream vs nonlinearity amplification — GPT-2 FFN** *(CONFIRMED — cycle 104)*
   - **Outcome: CONFIRMED_ATTN** — Attention dominates (≥70%) in 8/11 transitions. Early layers (0→6): attn_frac > 1.0, gelu_frac < 0 (attention hyperamplifies, GELU attenuates). Transition 6→7 is the pivot: attn drops to 0.88, GELU turns positive. Layers 8→10: attn_frac ~0.5, gelu_frac ~0.5–0.63 (balanced). LN passthrough ratio = 1.000 everywhere (LayerNorm is neutral). Dominant source is attention, particularly self-attention redistribution of error across sequence positions. The cascade is primarily an attention-mediated phenomenon, not GELU nonlinearity. Intervention target: attention suppression or error-orthogonalization before attention sub-layer.
   - Artifacts: experiments/cascade_error_source_gpt2.py, autoresearch/results/cycle_104_cascade_source.json, autoresearch/results/cycle_104_cascade_source.png

### Cycle 102

107. **Zero-shot GELU→ReLU swap — GPT-2** *(REFUTED — cycle 102)*
   - **Outcome: REFUTED** — Full-model swap: baseline PPL 22.26 → 1874.11 (+8320% relative). Block 0 alone: +2664%. Late blocks (9, 10) tolerate swap (+6.7%, +5.3%). Sparsity 56-98% post-ReLU but PPL is unusable. GELU bump is critical at inference.
   - Artifacts: experiments/gelu_relu_swap_gpt2.py, autoresearch/results/cycle_102_gelu_relu_swap.json, autoresearch/results/cycle_102_gelu_relu_swap.png

### Cycle 101

106. **GELU output clustering (corrected) — GPT-2 FFN** *(REFUTED — cycle 101)*
   - **Outcome: REFUTED** — K* > 256 for all 12/12 blocks. Even with correct methodology (cluster-mean of actual FFN outputs, no linearization), tokens in the same GELU' cluster do NOT produce similar outputs. Mean reconstruction error at k=256 remains 0.67–1.02 (far above 5% threshold). Between-cluster variance ratio is low (<1 in most cases), confirming GELU' cluster membership provides essentially no predictive power for FFN output. This definitively closes the GELU clustering approach.
   - Artifacts: experiments/gelu_output_clustering_corrected.py, autoresearch/results/cycle_101_gelu_output_clustering.json, autoresearch/results/cycle_101_gelu_output_clustering.png

### Cycle 100

103. **Low-rank Jacobian factorization for FFN compression — GPT-2 FFN** *(REFUTED — cycle 100)*
   - **Outcome: REFUTED** — Mean-Jacobian SVD factorization U@diag(S(x))@V^T fails at r=128 (median rel. error 0.809 > threshold 0.5). Per-token Jacobians have full rank (~720-730 at 95% energy) and vary enormously across tokens; their mean's singular vectors do not span the individual J(x). Oracle PPL: +13.7% to +74.0%. Random U,V beats SVD U,V in 3/4 blocks — the SVD basis provides no advantage. TopK+lstsq (k=96) achieves near-baseline PPL (−1.8% to +6.3%). The factored linear form cannot represent the nonlinear FFN even with exact S(x).
   - Artifacts: experiments/jacobian_lowrank_factorization_gpt2.py, autoresearch/results/cycle_100_jacobian_factorization.json, autoresearch/results/cycle_100_jacobian_factorization.png

### Cycle 098

113. **Joint FFN+attention compression on viable blocks — GPT-2** *(INCONCLUSIVE — cycle 098)*
   - **Outcome: INCONCLUSIVE** — Joint k=4 blocks achieved +133.55% PPL (far above +20% confirmed threshold, but also above +40% inconclusive ceiling). However, joint k=2 (FFN blocks [7,5] + attn blocks [1,4]) achieved +19.38% — just under the +20% refutation threshold. Key findings: (1) FFN-only k=4 = +56.18% (matches cycle 089 exactly, not +5% as plan misstated — that was single-block); (2) Attn-only k=3 = +13.08% (matches cycle 095); (3) Interaction is strongly super-additive: +64.29% above additive prediction — joint damage is not the sum of parts, it amplifies. (4) k=2 joint is viable at +19.38%. The confirmed criterion was miscalibrated (assumed FFN k=4 = +5% single-block result, but actual accumulated is +56%). The core finding: FFN and attention errors interact super-additively through the shared residual stream, amplifying by ~64% above additive. Compression must target k≤2 blocks per modality for joint viability.
   - Artifacts: experiments/joint_ffn_attn_compression_gpt2.py, autoresearch/results/cycle_098_joint_compression.json, autoresearch/results/cycle_098_joint_compression.png

### Cycle 097

114. **Block-0-skip attention stacking — GPT-2** *(REFUTED — cycle 097)*
   - **Outcome: REFUTED** — 5-block simultaneous attention compression (blocks [1,3,4,9,10], skip block 0) achieved +42.93% ± 1.57% PPL degradation (threshold: >+20% = refuted). Cascade dominates regardless of block-0 exclusion. 3-block control [1,3,4] also worsened: +25.05% ± 0.20% (vs cycle 095 top-3 [1,4,9] = +12.98%). Position probe on block 0: early (pos 0-16) L2 ratio=0.727, late (pos 112-128) L2 ratio=0.703, late/early=0.968 — no evidence of position-embedding-driven sensitivity (ratios nearly equal). Block 0 sensitivity appears intrinsic, not positional. Adding 2 more blocks to the viable set increases cascade substantially; the +5% single-block threshold does not predict stackability.
   - Artifacts: experiments/block0_skip_attention_gpt2.py, autoresearch/results/cycle_097_block0_skip_attention.json, autoresearch/results/cycle_097_block0_skip.png

### Cycle 096

111. **Cascade attribution: FFN vs attention — GPT-2** *(inconclusive — cycle 096, design review REJECT round 2)*
   - Plan rejected: verdict logic applied thresholds to raw fractions including interaction term, inconsistent with the clean decomposition. Also undefined variable reference.
   - Item valid, needs cleaner verdict logic using interaction-corrected quantities.
   - Outcome: inconclusive (plan rejected, not executed)
   - Commit: NONE

### Cycle 095

109. **Attention block compression feasibility scan v2 — GPT-2** *(INCONCLUSIVE — cycle 095)*
   - **Outcome: INCONCLUSIVE** — Block 0 is catastrophically sensitive (+102.4%); blocks 1,3,4,9,10 pass (≤+5%): pass_count=5/12, fail_count=1/12. Most blocks fall in the +5–15% range. Phase 2 (simultaneous rank-11 on top-3 blocks): +13.0% ± 1.4%. Rank-16 milder compression gives identical result (+13.0% ± 1.5%) — striking: rank-16 vs rank-11 makes no difference for simultaneous multi-block compression. QKV SVD compression is feasible for most blocks individually but not globally at 5.4×. Block 0 appears structurally dependent on full-rank attention. Baseline PPL=51.66.
   - Artifacts: experiments/attention_compression_v2_gpt2.py, autoresearch/results/cycle_095_attention_compression_v2.json, autoresearch/results/cycle_095_attention_compression_v2_barchart.png
   - Cycle: 095

### Cycle 094

110. **Non-uniform compression ratio allocation — GPT-2 FFN** *(REFUTED — cycle 094)*
   - **Outcome: REFUTED** — Uniform allocation is near-optimal. Baseline PPL=51.66±5.49. uniform_6x: PPL=69.40±9.69 (+34.3%), mild_nonunif: PPL=69.56±9.64 (+34.6%), aggressive_nonunif: PPL=70.25±9.83 (+36.0%). Best non-uniform delta vs uniform = +0.24% (negligible). Polish absorbs single-block damage differences; allocation is irrelevant after the output-projection is refit. 100-step Adam polish, 3 seeds, OWT.
   - Artifacts: experiments/nonuniform_compression_gpt2.py, autoresearch/results/cycle_094_nonuniform_compression.json, autoresearch/results/cycle_094_nonuniform_compression.png
   - Cycle: 094

### Cycle 093

108. **Lower compression ratio stacking — GPT-2 FFN** *(REFUTED — cycle 093)*
   - **Outcome: REFUTED** — c is ratio-independent: c_3x=0.406, c_4x=0.404, c_6x=0.408 (all ~0.41). k=8 PPL delta: +916% (3x), +896% (4x), +914% (6x). The ~41%/block compounding rate does not decrease with lower compression. Single-block PPL improves slightly at lower ratios but stacking cascades identically. 6x sanity check passed (c=0.408 ≈ cycle 089 c~0.41).
   - Artifacts: experiments/lower_ratio_stacking_gpt2.py, autoresearch/results/cycle_093_lower_ratio_stacking.json, autoresearch/results/cycle_093_lower_ratio_stacking.png
   - Cycle: 093

### Cycle 092

107. **Adam polish dissection — GPT-2 FFN** *(INCONCLUSIVE — cycle 092, audit: UNCERTAIN — metric issues)*
   - **Outcome: INCONCLUSIVE** — Delta rank95≈390/768 (ratio≈0.89× null); residual_frac≈0.58-0.62. Audit flagged: rank95 used sigma not sigma², compressing ratios toward 1.0. Energy-based rank could be much lower. Residual projection confounded by truncation to rank95 dims (random baseline ~50.8%). Results unreliable but directionally suggestive of diffuse update.
   - Artifacts: autoresearch/results/cycle_092_polish_dissection.json, autoresearch/results/cycle_092_convergence.png
   - Cycle: 092

### Cycle 090

105. **Attention block compression feasibility scan — GPT-2** *(inconclusive — cycle 090, design review REJECT round 2)*
   - Plan rejected: rank-128 SVD on [768,768] attention matrix = 3× compression, not 6×. Thresholds calibrated to wrong compression ratio.
   - Item is valid but needs corrected compression calculation (rank-64 for true 6×, or adjusted thresholds for 3×).
   - Outcome: inconclusive (plan rejected, not executed)
   - Commit: NONE

### Cycle 089

104. **Compositional error accumulation curve — GPT-2 FFN** *(REFUTED — cycle 089)*
   - **Outcome: REFUTED** — Super-multiplicative error accumulation. Baseline PPL=51.66±5.49. Forward k=4: +56%, k=5: +145%, k=8: +914%. Multiplicative fit R²=0.933, c=0.408/block. Reverse ordering reaches identical PPL=524 at k=8. Hard ceiling confirmed: >4 simultaneous compressed blocks at 6× is not viable. Key finding: errors compound at ~41% per block rather than summing.
   - Artifacts: autoresearch/results/cycle_089_accumulation_curve.json, autoresearch/results/cycle_089_accumulation_curve.png
   - Cycle: 089

### Cycle 088

102. **Input-space token clustering for local Jacobian approximation — GPT-2 FFN** *(REFUTED — cycle 088)*
   - K* > 64 for all 4/4 blocks [7, 5, 4, 1]. Reconstruction errors at K=64: Block 7=1.53, Block 5=1.90, Block 4=2.13, Block 1=1.82 (far above 5% threshold). K=1 null baseline: 1.93–2.68 — clustering barely helps. Local Jacobians are nearly full-rank (rank95: 319–478, rank99: 473–621 out of 768). The FFN behaves in a continuously distinct linear regime per token; input-space clustering is no more helpful than output-space clustering (cycle 085).
   - Cycle: 088
   - Artifacts: `experiments/input_space_jacobian_clustering_gpt2.py`, `autoresearch/results/cycle_088_input_jacobian_clustering.json`, `autoresearch/results/cycle_088_input_jacobian_clustering.png`

### Cycle 087

96. **Mixed-corpus stacking evaluation — GPT-2 FFN** *(INCONCLUSIVE — cycle 087)*
   - Polished (500 steps) results: A (OWT-only) → OWT=55.24±0.16, Wiki=111.13±1.20; B (mixed) → OWT=56.23±0.52, Wiki=102.74±1.81. Baseline (uncompressed): OWT=45.84, Wiki=70.13. Wiki improvement B vs A: +7.5% relative. OWT degradation B vs A: +1.8%. Neither confirmed (needs ≥10% Wiki improvement) nor refuted (needs >5% OWT degradation).
   - Cycle: 087
   - Artifacts: `experiments/mixed_corpus_stacking_gpt2.py`, `autoresearch/results/cycle_087_mixed_stacking.json`, `autoresearch/results/cycle_087_mixed_stacking.png`

### Cycle 086

99. **FFN output manifold geometry — GPT-2 FFN** *(INCONCLUSIVE — cycle 086)*
   - Isotropy ratios (lambda_512/lambda_1): Block 7=0.0151, Block 5=0.0157, Block 4=0.0151, Block 1=0.0043. All moderate (0.01-0.1) — confirmed/refuted criteria not met. Spectrum steeper than MP (KS=0.49-0.64, 53-72% eigenvalues outside support). JL empirical distortion is 3-4x tighter than bound (max 0.11 vs eps_jl=0.39). Top-k SVD projections violate isometry (max distortion 1.4) due to over-compensation by sqrt(d/k) scale factor.
   - Cycle: 086
   - Artifacts: `experiments/ffn_manifold_geometry_gpt2.py`, `autoresearch/results/cycle_086_manifold_geometry.json`, `autoresearch/results/cycle_086_manifold_geometry.png`

### Cycle 085

97. **GELU activation pattern diversity — GPT-2 FFN** *(REFUTED — cycle 085)*
   - K-means on continuous GELU'(z) vectors (3072-dim) for all 12 blocks, k in {8,16,32,64,128,256}. Binary pattern analysis (binarize at 0.5). Centroid reconstruction using consistent affine linearization: d_c = GELU'(z_c) via autograd.
   - Results: K* > 256 for all 12/12 blocks. Reconstruction error at k=256: range 0.69–1.93 (far above 5% threshold). Binary patterns: 2045/2048 unique per block (near-maximum diversity). Hamming mean 0.13–0.29. Jensen gap (mean ||d_c_recomputed - d_kmeans_centroid||₁/3072) mostly < 0.05, confirming the linearization itself is sound — the problem is that 256 clusters is simply insufficient for 3072-dim GELU patterns with near-maximum diversity.
   - Verdict: REFUTED. GPT-2 FFN GELU patterns are too diverse to cluster: each token's activation pattern is distinct enough that centroid approximation fails badly even at k=256. Centroid affine map compression (item 98) is not viable via this route.
   - Cycle: 085
   - Artifacts: `experiments/gelu_pattern_diversity.py`, `autoresearch/results/cycle_085_gelu_pattern_diversity.json`, `autoresearch/results/cycle_085_gelu_pattern_diversity.png`

### Cycle 084

95. **Data-conditioned Jacobian clustering for merge candidates — GPT-2 FFN** *(inconclusive — cycle 084, design review REJECT round 2)*
   - Plan rejected: affine linearization inconsistency — k-means centroid d_c = mean(GELU'(z_i)) ≠ GELU'(z_c) due to Jensen's inequality. Reconstruction formula unsound without overriding d_c = GELU'(z_c).
   - Item is valid but needs a corrected plan. Can be re-proposed with consistent linearization.
   - Outcome: inconclusive (plan rejected, not executed)
   - Commit: NONE

### Cycle 083

94. **GELU near-linearity census v2 — GPT-2 FFN (corrected methodology)** *(REFUTED — cycle 083)*
   - Correct GELU-derivative methodology (GELU'(z) via autograd, tanh-approx) on 4128 mixed-corpus tokens (4×8×129), 12 blocks × 3072 channels.
   - Results: Total dead: 14/36864 (0.04%), near-linear: 3/36864 (0.01%). 0/12 blocks meet threshold.
   - GELU nonlinearity is fully utilized. GPT-2 activations occupy the nonlinear regime — no per-channel dead or near-linear structure exploitable for compression.
   - Note: tanh-approx GELU' is slightly negative at x≪0, unlike exact GELU. Null N(0,1) baseline: dead=1.54%, linear=8.15%.
   - Verdict: REFUTED — GELU nonlinearity fully utilized.
   - Cycle: 083
   - Artifacts: `experiments/gelu_nearlinearity_census_v2.py`, `autoresearch/results/cycle_083_gelu_census_v2.json`, `autoresearch/results/cycle_083_gelu_census_v2.png`

### Cycle 082

93. **Gauss-Newton correction retry with true J^T J — GPT-2 FFN** *(INCONCLUSIVE — cycle 082)*
   - Retry of item 85 using true GN Hv via jvp+vjp (flash attn disabled for forward-mode AD). CG residuals monotonically decrease (0.12→0.004 at iter 30) confirming correct PSD J^T J. However, the system is severely underdetermined: 8 seqs × 127 tokens = 1016 tokens < 98K params. The Tikhonov solution is close to zero correction in the null space.
   - Results (3 seeds, blocks [7,5,4,1], 6× compression): GN +21.5±23.7% PPL vs unpolished +21.7±23.7% vs steepest descent +21.6% vs Adam 500 +7.7%. GN and steepest descent are essentially identical — single Newton step cannot overcome the underdetermined system's null space.
   - Key finding: With only 1016 tokens and 98K parameters, J^T J has effective rank ≤ 1016 << 98K. The Tikhonov correction is essentially zero in the 97K-dim null space. More data or per-block linearization needed to make GN viable.
   - Verdict: INCONCLUSIVE (+21.5% PPL, between +12% and +25% threshold). GN step helps marginally (0.2pp) but is far from Adam's +7.7%.
   - Cycle: 082
   - Artifacts: `experiments/gauss_newton_true_gpt2.py`, `autoresearch/results/cycle_082_gauss_newton_true.json`, `autoresearch/results/cycle_082_gauss_newton_true.png`

### Cycle 081

85. **Analytic stacking correction via Jacobian chain rule — GPT-2 FFN [MOONSHOT]** *(inconclusive — cycle 081, audit: UNCERTAIN — method failure, not genuine refutation)*
   - Implementation used finite-difference full-Hessian Hv (indefinite, not PSD) instead of true Gauss-Newton J^T J. Tikhonov damping ~1e-10 was negligible. Only 10 CG iterations on ~98K params.
   - Newton PPL mean +902% (diverged), but steepest descent +14.3% (gradient signal valid). CG 0/12 converged (residual ~0.3).
   - Audit: method failure (wrong Hv approximation + insufficient CG iters), not evidence against the hypothesis itself.
   - Outcome: inconclusive (audit downgrade from refuted)
   - Commit: 5a2b487
   - Artifacts: `experiments/newton_correction_stacked_gpt2.py`, `autoresearch/results/cycle_081_newton_correction.json`, `autoresearch/results/cycle_081_newton_correction.png`

### Cycle 080

92. **Retained-spectrum denoising via analytic projection — GPT-2 FFN** *(REFUTED — cycle 080)*
   - Method is algebraically vacuous: re-solving lstsq with same h_comp and shifted target ≈ no-op. Analytic 1-pass +10.6% ≈ unpolished +10.9% (0.3pp = noise). Adam comparison invalid (Shakespeare domain adaptation artifact, not compression polishing).
   - Cycle: 080
   - Artifacts: `experiments/retained_spectrum_denoising_gpt2.py`, `autoresearch/results/cycle_080_analytic_projection.json`, `autoresearch/results/cycle_080_analytic_projection.png`

### Cycle 079

91. **Subspace alignment re-run with correct hooks — GPT-2 FFN** *(REFUTED — cycle 079)*
   - "Globally shared" spectral structure hypothesis fails: 0/12 blocks hit 0.8 alignment threshold (range 0.74–0.79, random=0.637). Structure is partially shared but not global. topk is reasonable but not near-optimal.
   - Cycle: 079
   - Artifacts: `experiments/ffn_subspace_alignment_v2.py`, `autoresearch/results/cycle_079_subspace_alignment.json`, `autoresearch/results/cycle_079_subspace_alignment.png`

### Cycle 078

90. **Mixed-corpus lstsq refit with matched sample sizes — GPT-2 FFN** *(CONFIRMED — cycle 078)*
   - Reran cycle 076 with matched n=2048 for both conditions: A'=Shakespeare-only 2048 tokens, C=mixed 2048 tokens (512 each from Shakespeare, WikiText-2, OWT, C4). Channel selection fixed from Shakespeare for both. Blocks [7,5,4,1] at 6×, 3 seeds [42,123,7], eval on WikiText-2 validation (200 sequences).
   - Key numbers (mean across 3 seeds): A' PPL=138.90±2.37 (+98.1% vs baseline 70.13), C PPL=112.44±1.10 (+60.3%). Relative improvement C over A': +19.1%.
   - Interpretation: corpus diversity genuinely helps lstsq conditioning — not just a sample-size artifact. Mixed refit reduces PPL by 26.5 points (19.1% relative) vs Shakespeare-only at matched n=2048. The cycle 076 confound is resolved: both sample size AND corpus diversity contribute. At adequate n, diverse data significantly outperforms single-domain refit.
   - Verdict: CONFIRMED (threshold: ≥5% relative, actual: +19.1%)
   - Cycle: 078
   - Artifacts: `experiments/mixed_corpus_matched_refit_gpt2.py`, `autoresearch/results/cycle_078_matched_refit.json`, `autoresearch/results/cycle_078_matched_refit.png`

### Cycle 077

89. **SVD vs topk vs random basis at aggressive compression (k/n ≤ 0.3) — GPT-2 FFN** *(REFUTED — cycle 077)*
   - Re-ran cycle 071 comparison at k/n ∈ {0.3, 0.2, 0.1} on blocks [1, 4, 7, 11] with 3 seeds. SVD+lstsq and random+lstsq match within <2% relative PPL at all k/n ratios and all 4 blocks. Criterion: within 3% at all conditions.
   - Key numbers (mean across blocks, 3 seeds): k/n=0.3: SVD 122.4, random 122.8; k/n=0.2: SVD 122.7, random 123.4; k/n=0.1: SVD 123.4, random 123.6. topk+lstsq substantially worse at all k/n (126–165 PPL).
   - Interpretation: the lstsq refit entirely dominates basis choice. The output manifold of GPT-2 FFNs is so isotropic that any random k-dimensional subspace is as good as the Jacobian SVD subspace. Compression quality is determined by k/n (dimensionality retained), not the basis orientation.
   - Verdict: REFUTED
   - Cycle: 077
   - Artifacts: `experiments/ffn_aggressive_basis_compression.py`, `autoresearch/results/cycle_077_aggressive_basis.json`, `autoresearch/results/cycle_077_aggressive_basis.png`

### Cycle 076

66. **Lstsq refit with mixed-corpus probe data — GPT-2 FFN** *(INCONCLUSIVE — cycle 076, audit-corrected from CONFIRMED)*
   - ⚠ AUDIT G1/G3: Condition A (512 Shakespeare tokens) has n=512 samples for p=513 lstsq params — underdetermined system. PPL 3.7M is lstsq degeneracy, not corpus mismatch. Condition C (2048 mixed tokens) is well-conditioned and works (PPL 112.9), but the A→C improvement is confounded by sample size vs corpus diversity.
   - Needs rerun with matched sample sizes: A'=Shakespeare 2048 tokens vs C=mixed 2048 tokens to isolate corpus diversity effect. Cycle 044 showed Shakespeare-only at 2048 tokens gives PPL ~135, not millions.
   - Verdict: INCONCLUSIVE — lstsq degeneracy confound prevents isolating corpus diversity effect.
   - Cycle: 076
   - Artifacts: `experiments/mixed_corpus_lstsq_refit_gpt2.py`, `autoresearch/results/cycle_076_mixed_refit.json`, `autoresearch/results/cycle_076_mixed_refit.png`
   - Depends on: 64

### Cycle 075

35. **Region volume Pareto — GPT-2 FFN analog** *(REFUTED — cycle 075)*
   - Pareto hypothesis fails decisively: top-20% coverage 5–26% (need ≥70%), 0/12 blocks. FFN Jacobian structure is diffusely distributed across clusters, not Pareto-concentrated. Gini above null (0.316 vs 0.179) confirms non-uniform structure exists, but it's spread out, not dominated by a few regions.
   - Cycle: 075
   - Artifacts: `experiments/region_volume_pareto_gpt2.py`, `autoresearch/results/cycle_075_pareto_gpt2.json`, `autoresearch/results/cycle_075_pareto_gpt2.png`

### Cycle 074

28. **Persistent homology of activation manifold through depth — GPT-2** *(INCONCLUSIVE — cycle 074)*
   - PCA 30D, ripser, 3 seeds [42,123,7], 512 tokens/seed. Damage from cycle 048.
   - TP grows monotonically through ALL 12 blocks (0/11 decreasing pairs). Spearman(TP, damage_8x)=0.448, but without block 0 it jumps to 0.882 (L0 leverage artifact). Null baseline TP > block TP for 11/12 blocks.
   - PCA variance @ 30D: 72.4% for blocks 0-1 (< 80% threshold). Pre-LN verified (norm std 5.5–132). PCA sensitivity: Spearman(20D,40D)=1.000.
   - Inconclusive because pca_ok=False for blocks 0-1. Main signal: TP increases monotonically, opposite of confirm hypothesis; no compression predictor.
   - Outcome: INCONCLUSIVE (PCA variance fails threshold for early blocks; TP monotonically increases, not decreases)
   - Cycle: 074
   - Artifacts: `experiments/phom_gpt2.py`, `autoresearch/results/cycle_074_phom_gpt2.json`, `autoresearch/results/cycle_074_phom_gpt2.png`

### Cycle 073

84. **Rank-matched adaptive compression — GPT-2 FFN** *(CONFIRMED — cycle 073)*
   - Uniform allocation is near-optimal for GPT-2: block ranks vary only 1.27× (477–611), making uniform vs rank-matched indistinguishable (0.1pp diff over 15pp std, 3 seeds). GPT-2 block structure is near-homogeneous — adaptive allocation only matters for models with heterogeneous blocks (rank ratio >2×).
   - Cycle: 073
   - Artifacts: `experiments/rank_matched_compression_gpt2.py`, `autoresearch/results/cycle_073_rank_matched_compression.json`, `autoresearch/results/cycle_073_rank_matched_compression.png`

### Cycle 072

83. **Stacking error spectral decomposition — GPT-2 FFN** *(REFUTED — cycle 072)*
   - Blocks [7,5,4,1] at 6x, 3 seeds [42,123,7], 512 probe tokens. rank_99 ≈ 481–640 (blocks 7/5/4 ~485, block 1=640).
   - frac_retained (error in retained top-rank_99 directions): single-block 0.775–0.913, stacked 0.775–0.913.
   - Null (random basis at same rank_99): 0.635–0.822.
   - Key finding: compression error is concentrated in the RETAINED spectrum (75–91% in retained directions), NOT the discarded directions. Error lands in what the model is trying to preserve. Stacking does not shift the spectral profile vs single-block (difference < 0.01 across all blocks/seeds).
   - Refuted: discarded fraction never reaches 70% threshold (retained fraction is 75–91%). Error is structured (above random null) but in the wrong direction — it corrupts retained subspace, not just the discarded one.
   - Implication: gradient polish works not by routing around missing info but by correcting corruption of the retained subspace. The stacking problem is a retained-spectrum corruption problem.
   - Outcome: REFUTED
   - Cycle: 072
   - Artifacts: `experiments/stacking_error_spectral_decomposition.py`, `autoresearch/results/cycle_072_stacking_error_spectrum.json`, `autoresearch/results/cycle_072_stacking_error_spectrum.png`

### Cycle 071

82. **Spectral compression (Jacobian SVD basis) vs channel selection (topk) — GPT-2 FFN** *(CONFIRMED with caveat — cycle 071, audit-corrected)*
   - 4 methods at k=512 on blocks [1, 4, 7, 11], 3 seeds [42, 123, 7]. Baseline PPL: 121.5.
   - SVD+lstsq: mean 121.84. topk+lstsq: mean 126.16. Relative diffs: -1.1%, -4.0%, -2.4%, -6.1%. Wins ≥3% in 2/4 blocks.
   - ⚠ AUDIT CAVEAT: Random+lstsq (122.31) ≈ SVD+lstsq (121.84). Jacobian SVD basis not special vs random at k/n=0.67. Real mechanism: "any rotation+refit >> axis-aligned topk", not "Jacobian directions are special." Rerun at k=128/256 needed to test if SVD basis matters at higher compression.
   - Outcome: CONFIRMED (literal criterion met) but lstsq refit dominates basis choice at this retention ratio.
   - Cycle: 071
   - Artifacts: `experiments/ffn_jacobian_svd_compression.py`, `autoresearch/results/cycle_071_jacobian_svd_basis.json`, `autoresearch/results/cycle_071_jacobian_svd_basis.png`

### Cycle 070

64. **Corpus-diverse probing: multi-domain channel selection stability — GPT-2 FFN** *(INCONCLUSIVE — cycle 070)*
   - 4 corpora (Shakespeare, WikiText-2, OpenWebText, C4), blocks [7, 5, 4, 1], 6×, seeds [42, 123, 7].
   - Cross-corpus Jaccard: moderate (0.34–0.58). Block 1 lowest (0.34), block 5 highest (0.58). Shakespeare vs. web corpora: 0.41–0.53 across blocks. Web corpora vs. each other (WT2/OWT/C4): 0.41–0.68. No block reaches ≥ 0.7 threshold (0/4). Within-corpus cross-seed Jaccard: 0.56–0.74, consistent with cycle 067.
   - PPL: max_rel_PPL_gap ≤ 10% for blocks 4 and 1 (PASS), exceeds threshold for blocks 7 (+12.4%) and 5 (+15.3%). 2/4 blocks pass PPL criterion.
   - Jaccard < 0.4 only for block 1 Shakespeare vs. other pairs (not majority of blocks) — refuted criterion not met.
   - Key finding: channel selection IS partially corpus-specific (J ≈ 0.35–0.58, vs random baseline 0.091), especially Shakespeare vs. web corpora. Web corpora are more mutually consistent. The +120% PPL degradation from cycle 044 is caused by BOTH unstable channel selection AND corpus-specific lstsq coefficients.
   - Outcome: INCONCLUSIVE — Jaccard in moderate range (0.4–0.7); channel selection is meaningfully but not fully corpus-specific.
   - Cycle: 070
   - Artifacts: `experiments/corpus_diverse_channel_selection.py`, `autoresearch/results/cycle_070_corpus_diverse_channels.json`, `autoresearch/results/cycle_070_corpus_diverse_channels.png`, `data/c4_sample.txt`

### Cycle 069

81. *(invalid — implementation bug; re-ran as item 91, cycle 079: REFUTED)*

### Cycle 068

80. **FFN Jacobian effective rank census — GPT-2 FFN** *(CONFIRMED — cycle 068)*
   - Per-token Jacobian J(x) = W_out @ diag(GELU'(z)) @ W_in (768×768) SVD'd over 516 tokens × 3 seeds [42, 123, 7]. Random matrix control: rank_99(768×768 Gaussian) = 592 (Marchenko-Pastur).
   - rank_99 per block: 10/12 blocks ≤ 500 (range 434–510), genuinely low-rank vs 592 random baseline. Block 1 outlier at 588 (near-random, likely uniformly active GELU). Block 2 at 511.
   - rank_95: blocks 0,3–11 in range 268–314 (40% of full rank). Block 1 at 458, block 2 at 366.
   - Spearman(rank_99, P1_frob_std) = -0.497: anti-correlation — higher rank → lower P1 variation. Inverted from naive expectation, meaning rank alone doesn't predict compressibility in the expected direction.
   - Outcome: CONFIRMED — FFN blocks are genuinely low-rank (~60–75% of random baseline at 99% energy). Anti-correlation with P1 is an additional finding: compression difficulty is not explained by rank.
   - Cycle: 068
   - Artifacts: `experiments/ffn_jacobian_rank_census.py`, `autoresearch/results/cycle_068_jacobian_rank_census.json`, `autoresearch/results/cycle_068_jacobian_rank_census.png`
   - Depends on: none

### Cycle 067

50. **Centroid-merge quality as a function of probe corpus size — GPT-2 FFN** *(REFUTED — cycle 067)*
   - Measured topk channel selection stability (Jaccard across 3 seeds) and single-block swap PPL as probe corpus size varied from 64 to 2048 tokens on GPT-2 block 7 (6× compression, k=512 of 3072).
   - Jaccard@2048=0.844 (3-way), 0.893 (pairwise mean). Self-consistency WARNING: even at 2048 tokens, 3-way Jaccard is below 0.9, indicating the method is inherently variable across seeds.
   - Jaccard by probe size: 64→0.276, 128→0.356, 256→0.517, 512→0.640, 1024→0.733, 2048→0.844. Monotonically increasing but never reaches 0.85 threshold until above 2048.
   - PPL at probe_size=512 is catastrophically bad (mean 44147, std 37729, baseline 123.4) due to lstsq near-singularity: 512 probe tokens × 512 selected channels creates a near-square system in the output projection refit. PPL normalizes at 1024+ tokens (~+3–4% vs baseline).
   - At probe_size=256: Jaccard=0.517 < 0.6 refutation threshold, PPL vs 2048-ref +3.91%.
   - Outcome: REFUTED — Jaccard@256=0.517 < 0.6; channel selection requires large probe corpora (>1024 tokens) to reach acceptable stability. The centroid-merge signal is data-hungry, not data-efficient.
   - Secondary finding: lstsq output-projection refit is numerically unstable when probe_tokens ≈ hidden_dim (512 tokens, 512 channels). This is an implementation hazard for small-corpus deployments.
   - Cycle: 067
   - Artifacts: `experiments/centroid_merge_corpus_size.py`, `autoresearch/results/cycle_067_centroid_merge_corpus_size.json`, `autoresearch/results/cycle_067_centroid_merge_corpus_size.png`
   - Depends on: 26, 27

### Cycle 066

45. **Weight-space diffusion on topological level sets — toy ReLU [MOONSHOT]** *(REFUTED — cycle 066)*
   - Starting from trained weights θ₀, explore the weight space to find maximally compressed parameters that preserve the representation topology. Method: (a) Compute baseline persistent diagram PD₀ of representations at each layer (on training data). (b) Run projected Langevin dynamics: θ_{t+1} = θ_t − ε·∇||θ||₁ + √(2ε)·noise (drift toward minimum L1 norm = sparsity). (c) After each step, compute PD_t. If Wasserstein₁(PD_t, PD₀) > δ (topology changed), reject the step. (d) Run for T steps. The trajectory traces the "topological level set" — all weights producing the same representational topology. (e) Select the point θ* with minimum ||θ||₁ on the trajectory. If the topological level set is large (θ* has much lower norm than θ₀), the network is over-parameterized for its topology. Then: prune θ* (zero out small weights) and measure accuracy. Compare with: (a) standard magnitude pruning of θ₀, (b) L1-regularized retraining.
   - Cycle 066 (rev2, penalized Langevin variant): 3 seeds × spirals (2→32×4→1). Penalized Langevin without task loss: L(θ) = ||θ||₁ + λ_topo·W₁(PD_θ, PD₀). W₁ drift controlled (<0.20 of W₁_ref for all seeds). Norm reduction: 3.8–7.6% (far below 30% threshold). acc_topo best checkpoint: 83–91%, vs magnitude pruning 99–100%, vs L1-retrain 97–99%. acc_topo is -6pp to -17pp BELOW both baselines.
   - Root cause: L1 subgradient (sign(θ)) without a task loss term destroys accuracy immediately (acc drops to ~45-50% on step 0, recovers transiently due to topology pressure). The topological level set constraint is either too weak (W₁ can be preserved while accuracy collapses) or the sign gradient is too destructive. Norm reduction stays below 8% because the optimizer can't reduce norm while maintaining any acc ≥ 80% floor.
   - Outcome: REFUTED — 0/3 seeds meet the ≥30% norm reduction criterion; acc_topo is consistently worse than both baselines. The penalized Langevin approach without task loss does not work.
   - Cycle: 066
   - Artifacts: `experiments/weight_space_topo_diffusion.py`, `autoresearch/results/cycle_066_topo_diffusion.json`, `autoresearch/results/cycle_066_topo_diffusion.png`
   - Depends on: none

### Cycle 065

43. **Topological truncation: Betti-guided depth reduction [MOONSHOT]** *(INCONCLUSIVE — cycle 065)*
   - 2→32×4→1 networks trained on spirals (acc≈98.8%) and moons (acc≈99.9-100%). 3 seeds each. Ripser on 32D post-ReLU activations, 500 stratified subsampled points.
   - Total persistence grows monotonically across all 4 layers in all 6 runs (spirals: ~100-290 H0; moons: ~44-160 H0). Stabilization T is always 3 (the final layer). T ≤ L−2 criterion NOT MET for any seed/dataset.
   - Affine replacement at T=2 (L-1): ≤0.001 drop on spirals (3/3 seeds), ≤0.001 drop on moons (3/3 seeds). Moons is nearly linearly separable — early affine works trivially regardless of topology. Spirals at T=2: 98.7-98.9% (excellent). At T=1: 75.7-94.4% (inconsistent). At T=0: 61.8-64.7% (poor).
   - Outcome: INCONCLUSIVE — topology never stabilizes early (Betti-guided T criterion fails); affine replacement works well but only at T=L-1 (last layer), not earlier. The "topological triviality point" concept is not supported: TP grows monotonically indicating topology keeps evolving. Penultimate-layer affine replacement works (≤0.001 drop) but that is the trivially easy case, not a moonshot result.
   - Cycle: 065
   - Artifacts: `experiments/topo_truncation_betti.py`, `autoresearch/results/cycle_065_topo_truncation.json`, `autoresearch/results/cycle_065_topo_truncation.png`

### Cycle 064

42. **Topology-preserving distillation — toy ReLU [MOONSHOT]** *(INCONCLUSIVE — cycle 064, audit-corrected)* ⚠ **FLAWED: sqrt gradient singularity**
   - Teacher 2→32×4→1, student 2→32→1 (depth is the variable). 4 distillation methods: output-only, distance-matrix matching, CKA, activation-MSE. 3 seeds × 2 datasets (spirals, moons).
   - ⚠ **Audit correction**: `d2.sqrt()` in distance_matrix_loss has divergent gradient at zero. Post-ReLU activations guarantee zero pairwise distances. All 24 dmat runs went NaN by epoch 100. The negative deltas are random-init noise, not topology signal. Output-only baseline also uses suboptimal raw-logit MSE instead of KL+temperature.
   - Outcome: INCONCLUSIVE (audit-corrected from REFUTED) — method failed to run due to sqrt singularity, not genuine topology signal failure.
   - Cycle: 064
   - Commit: 3dca04c
   - Artifacts: `experiments/topo_preserving_distillation.py`, `autoresearch/results/cycle_064_topo_distillation.json`

### Cycle 056

### Cycle 057

### Cycle 059

74. **Gradient-polished lstsq: 10-50 Adam steps after analytic refit — GPT-2 FFN** *(CONFIRMED — cycle 059)*
   - Stacked k=4@6× (blocks [7,5,4,1]) on OpenWebText, 3 seeds [42,123,7]. After topk+lstsq, run {10,25,50} Adam steps on W_down columns only (lr=1e-4, polish data = probe_holdout, disjoint from lstsq probe and eval).
   - Baseline GPT-2 PPL (OWT): 57.653. Pure lstsq (0 steps): PPL=75.06 (+30.2%).
   - After polishing: 10 steps=72.63 (+25.98%, 14% rel improvement), 25 steps=70.51 (+22.30%, 26% rel), 50 steps=68.29 (+18.44%, 38.9% rel improvement).
   - Criterion: 50 steps reduce degradation by ≥30% relative. Result: 38.9% → CONFIRMED.
   - Polish loss descends consistently at 50 steps (e.g., 4.96→4.86, 4.72→4.17 across blocks/seeds). No overfitting observed (eval is disjoint from polish data).
   - Practical implication: 50 Adam steps at lr=1e-4 on a ~25K-token holdout reduces stacking degradation from +30.2% to +18.4%. Still +18.4% above baseline, so gradient polish alone does not fully close the gap.

### Cycle 063

79. **Per-block boundary correction revisited with stacking-aware fitting — GPT-2 FFN** *(INCONCLUSIVE — cycle 063)*
   - Cascaded affine corrections (each fitted on the actual comp_model outputs, including prior corrections) vs naive corrections (each fitted on float activations). k=4@6× blocks [1,4,5,7], OWT probe/eval, 3 seeds [42,123,7]. 25,600 probe tokens with 80/20 held-out split; all 12 correction points show held-out/fit MSE ratio ~1.1× (no overfitting flagged, threshold 1.5×).
   - Baseline GPT-2 PPL (OWT): 57.653. Uncorrected stacked: +30.20%±15.47%. Naive corrections: +50.77%±26.90% (worse than uncorrected — interference confirmed). Cascaded corrections: +25.35%±18.78%.
   - Criterion: cascaded ≤+12% → CONFIRMED; cascaded ≥ uncorrected → REFUTED. Result: +25.35% → INCONCLUSIVE (12–30% range).
   - Cascaded corrections do beat naive corrections substantially (50.8% vs 25.4%) and slightly beat uncorrected (+30.2% vs +25.4%), but do not reach the +12% threshold. Naive interference effect from cycle 043 persists and is worse — fitting on float activations makes corrections interfere destructively when stacked.
   - Cascaded correction reduces the MSE at each correction point vs naive (e.g., block 1: 0.097 vs 0.290 fit MSE), but later blocks still accumulate large error (block 7 cascaded fit MSE ~1.32). This shows corrections partially absorb error but do not eliminate compounding.
   - Practical implication: stacking-aware corrections are directionally correct but insufficient alone. Combining cascaded corrections + gradient polish (cycle 059) may close the gap further. The stacking problem remains a fundamental challenge at 6× compression with 4 blocks.
   - Artifacts: `experiments/stacking_aware_correction_gpt2.py`, `autoresearch/results/cycle_063_stacking_correction.json`, `autoresearch/results/cycle_063_stacking_correction.png`

### Cycle 062

78. **Reduced compression ratio for stacking: k=4@3× and k=4@2× — GPT-2 FFN** *(INCONCLUSIVE — cycle 062)*
   - Stacked k=4 (blocks [7,5,4,1]) at ratios {2×, 3×, 4×, 6×}, pure topk+lstsq, no gradient polish. 3 seeds [42,123,7], 8192 probe tokens per layer (critical fix from cycle 056 to avoid underdetermined lstsq at 2×). OWT corpus.
   - Baseline GPT-2 PPL (OWT): 57.653. Results: 2×=+11.15%±14.85%, 3×=+18.53%±15.10%, 4×=+22.09%±15.36%, 6×=+25.91%±15.13%.
   - Criterion: k=4@3× ≤+5% → CONFIRMED; ≥+20% → REFUTED. Measured +18.53% → INCONCLUSIVE (within +5–20% band).
   - 6× calibration: measured +25.9% vs cycle 058's +30.2% (+4.3pp deviation, attributable to improved atlas with 8192 vs 4096 probe tokens).
   - High seed variance (std ~15pp): seed 42 consistently 15–20pp worse than seeds 123 and 7. Reduces ratio demonstrates benefit at seeds 123/7 (+6.2% and +14.0% at 3×), but seed 42 shows +27.8% — suggesting the stacking degradation is sensitive to corpus split.
   - Practical implication: compounding is not fully super-additive in ratio. 3× stacking degrades half as much as 6× on average, but remains above the +5% threshold. Reducing ratio helps but does not eliminate the stacking problem.
   - Artifacts: `experiments/stacking_ratio_sweep_gpt2.py`, `autoresearch/results/cycle_062_stacking_ratio_sweep.json`, `autoresearch/results/cycle_062_stacking_ratio_sweep.png`

### Cycle 061

77. **Gradient polish scaling: 100–500 Adam steps on stacked blocks — GPT-2 FFN** *(CONFIRMED — cycle 061)*
   - Extended cycle 059's sweep to {0, 50, 100, 200, 500} steps, adding joint polish (all 4 W_down matrices updated simultaneously). Same setup: stacked k=4@6× (blocks [7,5,4,1]), OWT probe/eval, lr=1e-4, 3 seeds [42,123,7].
   - Baseline GPT-2 PPL (OWT): 57.653. 0-step (lstsq only): +30.20% (replicates cycle 059 exactly).
   - Independent polish: 50 steps=+18.44% (38.9% rel improvement), 100=+14.0% (53.6%), 200=+9.6% (68.4%), 500=+4.9% (83.9%).
   - Joint polish: 50 steps=+18.6%, 100=+13.8%, 200=+8.9%, 500=+3.8%. Consistently ~0.5–1.1pp better than independent at each step count, but not ≥5pp advantage.
   - Primary criterion (500 steps ≤ +8%): 500-step independent=+4.86%, joint=+3.79% — BOTH meet the +8% threshold. CONFIRMED.
   - No plateau detected: 200→500 improvement = 4.7pp (well above the 2pp plateau threshold).
   - Joint vs independent advantage at 500 steps: only 1.07pp (does not meet the ≥5pp criterion, but CONFIRMED by primary criterion).
   - Practical implication: 500 Adam steps at lr=1e-4 reduces stacked 4-block 6× degradation from +30.2% to +4.9% (independent) or +3.8% (joint). This effectively eliminates the stacking penalty for this configuration.
   - Artifacts: `experiments/gradient_polish_scaling_gpt2.py`, `autoresearch/results/cycle_061_gradient_polish_scaling.json`, `autoresearch/results/cycle_061_gradient_polish_scaling.png`

### Cycle 060

76. **Iterative cascaded compression: re-probe through compressed blocks — GPT-2 FFN** *(REFUTED — cycle 060)*
   - Baseline GPT-2 PPL (OWT): 57.653. Independent (float activations, 4 blocks simultaneous): +30.20%±15.47% (mean PPL=75.06). Cascaded (layer order [1,4,5,7], re-probe through partially-compressed model): +28.37%±15.08% (mean PPL=74.01).
   - Cascaded is 1.83pp better than independent — negligible improvement.
   - Independent baseline replication confirmed: +30.20% matches cycle 058's +30.2%.
   - Block 1 cascaded = block 1 independent (no prior distortion), as expected.
   - Per-block reconstruction MSE: cascaded shows ~10–20% lower FFN output MSE per block (distorted inputs are better matched), but this does not translate to PPL improvement because the residual stream distortion propagates from all prior compressed blocks regardless.
   - Criterion: cascaded ≥ +25% = REFUTED. Re-probing through compressed blocks does not meaningfully reduce stacking degradation.
   - Implication: the compounding PPL penalty is not primarily caused by lstsq input mismatch. The distortion propagates through the residual stream (attention, layer norm, etc.), which cascaded re-probing does not address.
   - Cycle: 059
   - Artifacts: `experiments/gradient_polish_stacked_gpt2.py`, `autoresearch/results/cycle_059_gradient_polish.json`, `autoresearch/results/cycle_059_gradient_polish.png`

### Cycle 058

75. **Multi-block stacking error budget: additive vs compounding degradation — GPT-2 FFN** *(INCONCLUSIVE — cycle 058)*
   - Progressive stacking k=1..4 at 6× on OpenWebText, blocks [7, 5, 4, 1], 3 seeds [42, 123, 7].
   - Baseline GPT-2 PPL (OWT): 57.653. Single-block deltas: all negative (−3.3% to −7.6%) — each individual block swap improves OWT PPL (probe alignment effect from cycle 057).
   - Stacked PPL: k=1 55.74 (−3.3%), k=2 61.75 (+7.1%), k=3 69.29 (+20.2%), k=4 75.06 (+30.2%).
   - Marginals: m1=−3.32, m2=+10.42, m3=+13.08, m4=+10.01 pp. Second differences: [+13.74, +2.66, −3.07]. Mixed-sign — not monotonically convex.
   - Ratio metric (stacked/sum-of-singles): ill-defined in all cases (denominator negative, all single deltas ≤ 0).
   - Residual-stream MSE grows with k; block 7 dominates cumulative MSE at each stacking level.
   - Outcome: INCONCLUSIVE — ratio metric is ill-defined due to negative single-block deltas. Sharp nonlinearity at k=2 (+13.7 pp second diff) is consistent with super-additive compounding onset, but the marginal curve is not monotonically convex. k=1 negative baseline confounds ratio analysis; primary criterion is inapplicable as written.
   - Cycle: 058
   - Artifacts: `experiments/stacking_error_budget_gpt2.py`, `autoresearch/results/cycle_058_stacking_budget.json`, `autoresearch/results/cycle_058_stacking_budget.png`

### Cycle 057

73. **Failure decomposition: channel selection vs lstsq refit vs stacking — GPT-2 FFN** *(INCONCLUSIVE — cycle 057)*
   - Decomposed +48% PPL gap (cycle 054, stacked k=4@6×) into three components for blocks [7, 5, 4] at 6×, seeds [42, 123, 7].
   - Key results: Variant A (OWT-native single-block): −3.3% to −5.6% PPL vs baseline (all three blocks, mean ≈−4.4%). Variant B (Shakespeare channels + OWT lstsq refit): essentially identical to A (B-A gap ≤ 0.2pp). Variant C (3-block stacked): +20.2% PPL degradation (mean across blocks, ±13.8%). Jaccard(OWT, Shakespeare) channel overlap: 0.47–0.54 (moderate).
   - Stacking overhead (C-A): +23.5pp to +25.8pp per block. Selection effect (B-A): −0.2pp to +0.1pp (negligible).
   - Outcome: INCONCLUSIVE by criterion — single-block OWT is <10% (in fact ≈−4 to −5%, within evaluation noise), confirming stacking is the sole failure mode and per-block method (selection, refit) is not the problem. Channel selection is interchangeable across corpora at moderate Jaccard (0.5); lstsq refit quality is insensitive to which channel set is selected.
   - Cycle: 057
   - Artifacts: `experiments/failure_decomposition_gpt2.py`, `autoresearch/results/cycle_057_decomposition.json`, `autoresearch/results/cycle_057_decomposition.png`

72. **Compression ratio sweep on OpenWebText: finding the viable frontier — GPT-2 FFN** *(INCONCLUSIVE — cycle 056, audit-corrected)* ⚠ **FLAWED: MP3 (lstsq underdetermination at 2×)**
   - Single-block swaps (no correction) at ratios {2×, 3×, 4×, 6×} on blocks [7, 5, 4] using OpenWebText as BOTH probe and eval corpus. 3 seeds [42, 123, 7], 200 eval sequences each.
   - Per-block mean rel PPL (across 3 seeds): block7: 2x=+16.74%, 3x=+11.50%, 4x=+10.24%, 6x=+9.44% | block5: 2x=+15.53%, 3x=+10.12%, 4x=+9.10%, 6x=+8.25% | block4: 2x=+13.60%, 3x=+8.70%, 4x=+7.52%, 6x=+6.61%
   - ⚠ **Audit correction**: Non-monotonic pattern (2× worse than 4×) is an lstsq overfitting artifact. At 2× (1536 channels kept), the lstsq system has only 1.33× overdetermination (2048 tokens / 1537 params). At 4× (768 channels), overdetermination is 2.66×. The 2× and 3× results are unreliable due to insufficient probe budget relative to parameter count. 4× and 6× results may be valid (overdetermination ≥2.66×) but the frontier conclusion is premature without fixing the probe budget.
   - Outcome: INCONCLUSIVE (audit-corrected from CONFIRMED) — 4× passes criterion on 2/3 blocks but 2× failure is an artifact of lstsq underdetermination, not genuine compression difficulty. Needs re-run with more probe tokens or L2 regularization.
   - Cycle: 056
   - Commit: 29e3a14
   - Artifacts: `experiments/owt_ratio_sweep.py`, `autoresearch/results/cycle_056_owt_ratio_sweep.json`, `autoresearch/results/cycle_056_owt_ratio_sweep.png`

### Cycle 055

71. **Jacobian Frobenius std as corpus-sensitivity predictor — GPT-2 FFN** *(CONFIRMED — cycle 055)*
   - Cycle 048 confirmed P1 (Jacobian Frobenius std) as a strong compression-difficulty oracle (rho=0.84 at 8×). This experiment tests whether the SAME signal predicts corpus sensitivity: compute P1 on Shakespeare probes and on WikiText-2 probes separately for all 12 blocks. If P1 is corpus-invariant (Spearman between Shakespeare-P1 and WikiText-P1 rankings ≥ 0.85), then compression difficulty is an intrinsic block property and the oracle generalizes. If P1 diverges across corpora, blocks that are "easy" on Shakespeare may be "hard" on WikiText-2 — and the oracle must be re-run on each target distribution.
   - WikiText-2 P1: [10.571, 3.874, 3.592, 4.630, 5.266, 6.211, 6.429, 8.356, 11.400, 10.727, 11.571, 15.628]
   - Shakespeare P1 (cycle 048): [12.566, 5.894, 5.010, 5.687, 5.788, 6.456, 6.189, 7.339, 11.009, 12.210, 13.682, 19.539]
   - Spearman rho (all 12 blocks): 0.9441 (p<0.0001). Spearman rho (blocks 1-10, extreme removed): 0.9394 (p=0.0001).
   - Verdict: CONFIRMED. P1 block ranking is corpus-invariant at rho=0.944, well above the 0.85 threshold. The Jacobian Frobenius std oracle is an intrinsic geometric property of each GPT-2 FFN block and does not require re-running on each target distribution.
   - Cycle: 055
   - Artifacts: `experiments/gpt2_p1_corpus_sensitivity.py`, `autoresearch/results/cycle_055_p1_corpus_sensitivity.json`, `autoresearch/results/cycle_055_p1_corpus_sensitivity.png`

### Cycle 054

70. **Training-distribution probing with OpenWebText subset — GPT-2 FFN** *(REFUTED — cycle 054)*
   - Full k=4@6× corrected recipe (blocks 7, 5, 4, 1) using OpenWebText as BOTH probe and eval corpus. 3 seeds [42, 123, 7].
   - Per-seed corrected PPL degradation: seed 42 = +54.01%, seed 123 = +45.68%, seed 7 = +44.94%.
   - Mean corrected degradation: +48.21% ±5.03 (std across seeds).
   - Reference comparison: Shakespeare-probed = +8.5%, WikiText-2-probed = +44%.
   - Verdict: REFUTED. Even using GPT-2's training distribution (OpenWebText) as the probe corpus, corrected PPL degradation is +48.21% — well above the +30% refutation threshold. OpenWebText performance is comparable to WikiText-2 (+44%), not to Shakespeare (+8.5%). The method's corpus specificity is NOT a probe-mismatch artifact: the fundamental limitation is the k=4@6× compression ratio, not the probe corpus identity. Training-distribution probing does not rescue the method.
   - Cycle: 054
   - Artifacts: `experiments/openwebtext_training_dist_probe.py`, `data/openwebtext_sample.txt`, `autoresearch/results/cycle_054_openwebtext_probe.json`, `autoresearch/results/cycle_054_openwebtext_probe.png`

### Cycle 053

69. **Per-block corpus sensitivity isolation — GPT-2 FFN** *(INCONCLUSIVE — cycle 053)*
   - Single-block 6× compression (topk, seed 42) for blocks [7, 5, 4, 1] with Shakespeare and WikiText-2 probes separately. Baselines: Shakespeare=123.35 PPL, WikiText-2=71.84 PPL.
   - Shakespeare probes: all 4 blocks show positive sensitivity delta (WikiText-2 degrades more than Shakespeare). Per-block deltas: block 7=+10.81pp, block 5=+11.20pp, block 4=+14.52pp, block 1=+6.07pp. Block 4 is most sensitive at 34.1% of total — below the 50% concentration threshold.
   - WikiText-2 probes: smaller overall deltas (blocks closer to matched performance). Block 4=+3.23pp (45.7%), block 1=-2.17pp, block 7=-1.13pp, block 5=+0.54pp.
   - Verdict: INCONCLUSIVE. No single block reaches 50% concentration. Block 4 is consistently the most sensitive under both probe regimes (34.1% and 45.7%), and the sensitivity is spread across all 4 blocks under Shakespeare probes (+6 to +14pp range). The gap is systemic — all middle blocks contribute — but block 4 is consistently the worst.
   - Outcome: INCONCLUSIVE — concentration < 50% for both probe corpora. Block 4 is the leading bottleneck (34–46% of total) but does not dominate. Corpus sensitivity appears to be a distributed property of topk channel selection, not a single-block artifact.
   - Cycle: 053
   - Artifacts: `experiments/per_block_corpus_sensitivity.py`, `autoresearch/results/cycle_053_per_block_sensitivity.json`, `autoresearch/results/cycle_053_per_block_sensitivity.png`

### Cycle 050

65. **Oracle-guided adaptive compression ratio — GPT-2 FFN** *(REFUTED — cycle 050)*
   - P1-guided per-block ratios (block 4: 10×, block 1: 9.82×, block 5: 9.0×, block 7: 7.89×, mean=9.17×) compared to uniform control at same mean ratio (~9.17×). Oracle k=4 corrected: +12.33% ±0.82 (mean, 3 seeds). Critically, the boundary lstsq correction worsened oracle performance (MSE increased: 1.71→1.88), while the uniform control at the same mean ratio got +6.36% ±0.69 with correction working normally (MSE: 1.71→1.36). The oracle's variable hidden dims per block produce hidden-state residuals that are incompatible with a single affine boundary correction. The P1 signal predicts block-level damage correctly but variable ratios in a stacked pipeline interact destructively with the boundary refit.
   - Outcome: REFUTED — oracle mean +12.33% is worse than uniform 10× corrected (+8.45%) and worse than uniform 6× uncorrected (+9.59%). Ratio allocation predicts the wrong direction when combined with boundary correction.
   - Cycle: 050
   - Artifacts: `experiments/oracle_adaptive_compression_gpt2.py`, `autoresearch/results/cycle_050_oracle_adaptive.json`, `autoresearch/results/cycle_050_oracle_adaptive.png`

### Cycle 049

51. **Per-block learned compressor with block-conditioned bottleneck — GPT-2 FFN** *(REFUTED — cycle 049)*
   - 32K-token per-block distillation (768->384->768, 500 steps, Adam 1e-3) tested on all 12 GPT-2 FFN blocks. Geometric (topk+lstsq) won 11/12 blocks; distillation beat geometric by ≥1 PPL on only 1/12 (block 1: +1.96 PPL gain). Reproduction with 4K tokens (blocks 8-11) matches cycle 030 Baseline B numbers within noise. Key result: 8× more training data (4K→32K) did not help. The bottleneck is not data volume — distillation at 500 steps does not recover the lstsq-quality fit even with full data. Block-MSE for distillation consistently higher than geometric (1.0–55.5 vs 0.08–1.06), confirming underfitting is structural, not due to data scarcity.
   - Outcome: REFUTED — geometric wins 11/12 blocks even with 32K tokens; criterion requires ≥8/12 wins for distillation.
   - Cycle: 049
   - Artifacts: `experiments/per_block_compressor_data_scaling.py`, `autoresearch/results/cycle_049_per_block_data_scaling.json`

### Cycle 048

30. **Local complexity proxy as a compression difficulty predictor — GPT-2** *(CONFIRMED — cycle 048)*
   - Jacobian Frobenius std (P1) over k=5 nearest neighbors in FFN input space, aggregated over 128 probe tokens. Correlated with cycle 005 single-block swap damage at 8x and 16x compression.
   - Outcome: CONFIRMED. P1 rho_full=0.839 (8x, p=0.000), rho_without_L0=0.818 (p=0.001). rho_full=0.608 (16x, p=0.018), rho_without_L0=0.527 (p=0.048). Leverage-point guard passed (rho_without_L0 ≥ 0.2 in all confirmed pairs). P2 (spectral norm) is leverage-dependent (rho_full=0.399) but strong after L0 removal. P3 (effective rank) anti-correlates and fails the criterion.
   - Cycle: 048
   - Artifacts: `experiments/gpt2_local_complexity.py`, `autoresearch/results/cycle_048_local_complexity_gpt2.json`, `autoresearch/results/cycle_048_local_complexity_gpt2.png`, `autoresearch/results/cycle_048_summary.md`

### Cycle 047

27. **Persistent homology of activation manifold through depth — toy ReLU** *(REFUTED — cycle 047)*
   - Trained 2→32×4→1 spirals network (acc=98.4%). Total persistence L1-L4: 122.7, 119.1, 113.4, 227.1. Only 2/3 transitions decrease (L4 spikes up); 80% monotonicity criterion NOT MET. Largest Betti drop at L3 but lowest r is at L4 — rank-1 prediction wrong. Spearman(drop, r) = +0.5 (wrong sign, n=3). Random-init monotonically decreases (81→39→18→4), trained network does not. Gaussian null (1917-3562) much higher than trained, confirming PH measures real structure — but the structure does not simplify monotonically through depth.
   - Outcome: REFUTED — topology does not decrease monotonically (2/3 transitions, need ≥ 80%); largest Betti drop layer does not match lowest-r layer
   - Cycle: 047
   - Commit: f2d83c5
   - Artifacts: `experiments/phom_toy_relu.py`, `autoresearch/results/cycle_047_phom_toy.json`, `autoresearch/results/cycle_047_phom_toy.png`

### Cycle 032

47. **Dead channel census — GPT-2 FFN v2 (GELU-correct)** *(REFUTED — cycle 032)* ⚠ **FLAWED: MP1 (threshold above trivial baseline)**
   - Pre-GELU dead (T=-3): 0.00%. Near-linear (T=+3): 0.00%. Pruning 1% (by contribution): +1.14% PPL > 1% threshold. Max safe prune = 0%. Contribution-based metric: 93% of channels below 0.05% threshold — but this is an artifact of near-uniform distribution (equal share = 0.033%); not meaningful "dead" channels. Confirms and closes cycle 028 finding with correct GELU-aware definitions.
   - ⚠ **Post-hoc review**: the 0.05% contribution threshold exceeds the equal-share baseline (1/3072 = 0.033%), so nearly all channels are trivially flagged. A design review would have caught this via R2 (measurement soundness) and required a threshold well below equal-share.
   - Outcome: REFUTED
   - Commit: b83ddd0
   - Artifacts: autoresearch/results/cycle_032_prune_collapse.json, autoresearch/results/cycle_032_prune_collapse.png

### Cycle 031

46. **Grokking as compression catalyst — modular arithmetic (v2)** *(REFUTED — cycle 031)* ⚠ **FLAWED: MP5 (insufficient model capacity)**
   - Task: (a+b) mod 97. Grokking detected (memorized epoch 1000, grokked epoch 17500, delay 16500 epochs). BUT: distilled student acc = 0.002 both pre and post grokking (essentially chance). Compression gain = +0.001, below 2% threshold. Pattern count stays at 9409 (= P², all examples unique) throughout — no geometric compaction signal. Post-grokking algebraic structure cannot be distilled from training data alone into a 4× smaller network.
   - Outcome: REFUTED — grokking occurred but compression gain <2% (0.001 actual)
   - ⚠ **Post-hoc review**: student hidden=32 is insufficient capacity for mod-97 (97 residue classes). A design review would have caught this via R3 (capacity adequacy) and required either a larger student or a smaller prime.
   - Commit: c0cb0dd
   - Artifacts: autoresearch/results/cycle_031_grokking.json, autoresearch/results/cycle_031_grokking_trajectory.png

### Cycle 028

16. **Exact prune plus linear collapse baseline** *(REFUTED — cycle 028)* ⚠ **FLAWED METHODOLOGY**
   - Implement the strongest exact baseline from the literature: stable-channel deletion and always-active / near-linear collapse before any heuristic merging. Compare exact-prune-only, exact-prune-plus-least-squares-refit, and the current geometric method on single-block swaps. This is the cleanest test of how much "free" no-SGD compression exists.
   - Type: implementation
   - Confirmed if: exact or near-exact pruning alone yields at least `1.1x` FFN parameter reduction with `<=1%` relative perplexity increase on the tested single-block swaps.
   - Refuted if: exact or near-exact pruning yields negligible savings or immediately breaks block behavior.
   - Inconclusive if: savings exist only under loose thresholding that is hard to justify.
   - Depends on: 11
   - **Outcome: REFUTED** — GPT-2 FFN has effectively zero dead or near-linear channels at threshold 1e-5/5%. Mean dead=0.00%, mean saved (dead+near-linear collapse)=0.008% across 12 blocks. Only 3 blocks had any classifiable channels at all (blocks 8, 9, 11: 1 channel each at 0.033%). Param reduction=1.000x. Prune-only and prune+collapse PPL unchanged (+0.00%). Prune+collapse+lstsq refit improves PPL by 0.21% (127.44→127.18) but that is a full lstsq refit on all live channels (3072/block), not structural pruning. GELU activations in GPT-2 use all channels meaningfully — no free structural compression exists at any practical threshold.
   - ⚠ **Post-hoc review: flawed experiment.** Dead channel definition used `max|post-GELU act| < 1e-5`, but GELU is never truly zero: gelu(-3)≈-0.004, gelu(-5)≈-0.00002 — both well above 1e-5. Threshold is calibrated for ReLU, not GELU. Correct definition: check *pre*-GELU values (max(h_j) < -T for T∈{2,3,4}) to find channels stuck in GELU's saturated tail. Also missing: output-contribution-based metric (||W_down[:,j]|| × rms(act_j)) and smooth pruning curve. **Superseded by item 47.**
   - Cycle: 028
   - Artifacts: `experiments/exact_prune_linear_collapse_gpt2.py`, `autoresearch/results/cycle_028_prune_collapse.json`, `autoresearch/results/cycle_028_prune_collapse.png`

### Cycle 027

15. **Centroid and Jacobian affinity as merge signals** *(CONFIRMED — centroid; REFUTED — Jacobian; cycle 027)*
   - Evaluate whether data-conditioned similarity measures from the centroid-affinity / local-map literature predict safe compression better than raw weight cosine. For each block, compare weight similarity, centroid similarity, Jacobian-spectrum similarity, and any combined score against actual swap damage under controlled budgeted compressions.
   - Type: empirical
   - Confirmed if: a centroid/Jacobian-based score improves Spearman correlation with swap error by at least `0.15` over raw weight cosine on a majority of tested blocks.
   - Refuted if: raw weight cosine is just as good or better across the board.
   - Inconclusive if: the better signal changes by block depth or corpus.
   - Depends on: 10, 11
   - **Outcome: CONFIRMED (centroid) / REFUTED (Jacobian)**
     - Weight cosine: mean ρ = +0.041 (effectively zero — no predictive power)
     - Centroid similarity: mean ρ = +0.853, Δ = +0.812 vs weight cosine, criterion met in 12/12 blocks
     - Jacobian-channel similarity: mean ρ = −0.037, Δ = −0.078 — slightly anti-predictive
     - Centroid (activation time-series cosine) is a dominant merge predictor across all depths
     - Jacobian signal (mean weighted projection) fails — likely because it collapses the token-level variation that centroid preserves
   - Cycle: 027
   - Artifacts: `autoresearch/results/cycle_027_affinity_signals.json`, `autoresearch/results/cycle_027_affinity_signals.png`

### Cycle 026

14. **Rank-only baseline versus geometry versus distillation**
   - Add a matched-budget SVD / low-rank FFN baseline and compare it directly to the current geometric method and distillation on the same single-block swaps and budgets. This asks whether the current "geometric" method is actually better than a plain matrix-factorization baseline.
   - Type: empirical
   - Confirmed if: the geometric method beats the rank-only baseline on full-model metrics in more than `50%` of tested block-budget pairs at equal FFN parameter count.
   - Refuted if: rank-only beats the geometric method in more than `80%` of tested block-budget pairs.
   - Inconclusive if: wins trade back and forth by block or budget with no consistent edge.
   - Depends on: 10
   - **Outcome: CONFIRMED** (cycle 026) — Two SVD baselines tested: (1) SVD on composed map W_down@W_up: catastrophically bad (PPL 3K–350M) because GELU does not commute with change-of-basis; (2) SVD-norm: select top-k channels by W_up row-norm + lstsq refit (proper rank baseline). Geometric beats svd_norm in 11/12 (92%) block-budget pairs. Baseline PPL 122.4; geometric +0.8–5.8%; svd_norm +2.2–7.2%; distillation (200 steps) +0.1–10.1%. Only exception: block 11 at 16× (deep layer, aggressive compression, svd_norm=129.0 vs geo=129.5). Key insight: data-driven channel scoring (activity × output-norm) outperforms weight-magnitude-based selection at all depths and budgets.

### Cycle 025

33. **GELU → 3-piece PWL approximation fidelity — GPT-2** *(REFUTED — cycle 025)*
   - Apply the same PWL substitution from experiment 32 to GPT-2: replace all GELU activations with the 3-piece PWL function, no retraining, measure perplexity on the probe validation set. If perplexity degrades < 5% relative, then GPT-2's FFN blocks are approximately piecewise-affine and all the partition-based theory (activation patterns, RTG, SplineCam) applies as an approximation. This would justify using ReLU-derived compression signals on GELU networks. Also measure block-by-block contribution: which blocks' perplexity degrades most when only that block's GELU is replaced, to identify which blocks are most sensitive to smooth activation curvature.
   - Type: empirical
   - Confirmed if: full GELU → 3-piece PWL substitution causes < 5% relative perplexity increase on the validation set.
   - Refuted if: perplexity increases > 20%, indicating that GELU's smooth transition zone is functionally critical at the LLM scale.
   - Inconclusive if: perplexity increase is moderate (5–20%) and is concentrated in specific blocks rather than spread evenly.
   - Depends on: 32. Skip if experiment 32 is refuted (the PWL approximation is fundamentally inadequate even for toy GELU networks).
   - **Outcome: REFUTED** — Baseline PPL=125.7. 3-piece: 6997 (+5466%), 5-piece: 3221 (+2462%), 10-piece: 198 (+57.8%), 20-piece: 127 (+1.1%). Even single-block 3-piece replacement causes catastrophic damage (block 0: +1239%; best block 9: +135%). GELU curvature is functionally critical at every granularity in GPT-2; 3-piece PWL is completely inadequate. Partition-based theory (SplineCam, RTG, MASO) does NOT apply to GELU networks even as a first-order approximation at this scale. 20-piece PWL is the minimum for near-lossless approximation.
   - Cycle: 025
   - Artifacts: `experiments/gelu_pwl_gpt2.py`, `autoresearch/results/cycle_025_gelu_pwl_gpt2.json`, `autoresearch/results/cycle_025_gelu_pwl_gpt2.png`

### Cycle 024

31. **CKA representational similarity matrix across layers (GPT-2)** *(INCONCLUSIVE — cycle 024)*
   - Compute centered kernel alignment (CKA) between the token representation matrices at all pairs of (layer_i, layer_j) in GPT-2. CKA is a scale-invariant similarity measure between two sets of representations: CKA ≈ 1 means the two layers are computing essentially the same geometry (up to rotation/scale), CKA ≈ 0 means they are unrelated. Plot the full 13×13 CKA heatmap. Identify "representational plateaus" — groups of consecutive layers with high mutual CKA (≥ 0.8) — as candidates for block-level depth pruning. Correlate the CKA drop between layer i and layer i+1 with the single-block swap damage for block i from cycle 005: blocks that contribute large CKA drops are doing new work and should be harder to remove.
   - Type: empirical
   - Confirmed if: at least one contiguous group of ≥ 3 GPT-2 layers has pairwise CKA ≥ 0.80 AND the (CKA drop to next layer) × (compression difficulty) Spearman correlation is ≥ 0.4.
   - Refuted if: CKA is uniformly low (< 0.5) for all non-adjacent layer pairs, indicating every layer computes a distinct representation.
   - Inconclusive if: CKA plateaus exist but removing one layer from a plateau still causes > 20% relative perplexity increase, indicating that representational similarity does not imply functional redundancy.
   - Depends on: 9
   - **Outcome: INCONCLUSIVE** — Massive 10-block plateau found (blocks 1–10, pairwise CKA ≥ 0.991), satisfying plateau criterion. But Spearman(CKA_drop, damage) = 0.32 (p=0.31), below ρ≥0.40 threshold. Root cause: CKA drops within the plateau are near-zero (0.0002–0.004) for all middle transitions, even though damage varies (0.12%–3.07%). Residual connections make the Gram matrix geometry nearly constant despite meaningful functional differences per block. CKA is a poor per-block compression predictor in GPT-2: representational similarity does not imply functional redundancy. The plateau exists but cannot identify which blocks are safe to remove.
   - Cycle: 024
   - Artifacts: `autoresearch/results/cycle_024_cka_matrix.json`, `autoresearch/results/cycle_024_cka_heatmap.png`, `autoresearch/results/cycle_024_summary.md`

### Cycle 023

29. **Local complexity as a compression difficulty predictor — toy ReLU**
   - Implement the local complexity measure from Patel & Montúfar 2025: for each probe input x, count the number of distinct activation patterns observed in a ball of radius ε around x (using probe samples). Average over x. This is the empirical local complexity. A network with high local complexity has many linear regions packed near the data — it is using its nonlinearity heavily and will be hard to compress. For the spirals toy network, compute local complexity at each layer and correlate it with the layer-wise single-affine residual r from experiment 20. Also compare local complexity across training checkpoints (early, mid, late training) to test the Humayun/Patel prediction that local complexity decreases during grokking.
   - Type: empirical
   - Confirmed if: local complexity correlates with single-affine residual r (Spearman ≥ 0.5) AND local complexity decreases by ≥ 30% between early and late training checkpoints.
   - Refuted if: local complexity shows no correlation with r and no consistent trend across training.
   - Inconclusive if: the correlation with r holds but the training-dynamics prediction fails, or vice versa.
   - Depends on: 20
   - **Outcome: INCONCLUSIVE** — Spearman ρ=0.80 (p=0.20, n=4) between local complexity and r_held PASSES the ≥0.5 threshold. Layer ordering by LC matches compression difficulty (L2 highest LC, L1 next, consistent with large r drop from L1→L2 in cycle 010). However, local complexity does NOT decrease during training: early=15.08, mid=14.32, late=15.91 (−5.5% change vs required −30%). The Humayun/Patel grokking prediction fails — spirals may not exhibit the same training-phase collapse as reported. Note: ε=0.39 yielded avg 72 neighbors (2× intended ~30); this doesn't change the qualitative finding. Experiment 30 (GPT-2 proxy) can proceed since the correlation signal holds.
   - Cycle: 023
   - Artifacts: `autoresearch/results/cycle_023_local_complexity.json`, `autoresearch/results/cycle_023_local_complexity.png`

### Cycle 022

40. **Attention head geometry census — GPT-2** *(INCONCLUSIVE — cycle 022)*
   - Extend beyond FFN-only analysis. For each attention head in GPT-2 (12 layers × 12 heads = 144 heads): (a) effective rank of attention matrix mean over probe sequences (nuclear norm / operator norm), (b) head output Jacobian variation (std of head output norms across tokens), (c) pairwise CKA between heads within the same layer to measure redundancy. Rank heads by a combined "prunability score" = low_rank × low_variation × high_redundancy. Test: zero-out the top-K most prunable heads, measure perplexity cost. This answers: is there geometric compression opportunity in attention, not just FFN?
   - Type: empirical + intervention
   - Confirmed if: ≥20% of heads have effective rank < 0.3 × head_dim AND zeroing the 10 most prunable heads causes ≤5% perplexity increase.
   - Refuted if: all heads have high effective rank and zeroing any 5 heads causes >15% perplexity increase.
   - Inconclusive if: effective rank is low but zeroing heads causes high perplexity increase, indicating low-rank heads are still functionally critical.
   - Depends on: 9
   - **Outcome: INCONCLUSIVE** — 89.6% of heads (129/144) have effective rank < 19.2 (0.3×64) — strong geometric low-rank signal across all layers. Eff_rank concentrates 1–7 in deeper layers (5–11), up to 104 in early layers. But zeroing top-10 most prunable heads causes +14.6% PPL increase (criterion: ≤5%). Only K=5 is tolerable at +2.7%. Low-rank geometry does not imply prunability: heads are individually redundant but collectively load-bearing.
   - Cycle: 022
   - Artifacts: `experiments/attn_head_census.py`, `autoresearch/results/cycle_022_head_census.json`, `autoresearch/results/cycle_022_head_census.png`

### Cycle 021

39. **Grokking as compression catalyst — toy ReLU** *(REFUTED — cycle 021)* ⚠ **FLAWED METHODOLOGY**
   - Train spirals network for 10× normal epochs (well past convergence into overfitting/grokking regime). At regular checkpoints: (a) count distinct activation patterns on held-out data, (b) compute pattern entropy (item 37), (c) measure single-affine residual per layer (item 20 method). Look for a phase transition where partition complexity drops sharply — this is grokking manifesting as geometric simplification. If found: run the geometric compression pipeline (from `relu_geometric_compression.py`) AT the grokking point vs BEFORE — quantify how much easier compression becomes post-grokking. This bridges "Deep Grokking" theory (already in lit map) with actionable compression timing.
   - Type: empirical
   - **Outcome: REFUTED** — Classic grokking (late test accuracy rise) IS detected on spirals: train saturates at 92% by epoch 2000, test accuracy jumps from 0.747 → 0.805+ around epoch 17000. However, the geometry signal is opposite to the hypothesis: pattern count INCREASES through grokking (369 → 564+, rising to 723 at epoch 29000 with no drop). Pattern complexity is NOT reduced by grokking on spirals — instead the network develops a richer partition. Compression quality improves slightly (+5.7% accuracy at 4× compression) but this tracks the teacher's improved accuracy, not geometric simplification. Pattern drop from peak = 4.3% (need 40%), compression gain = 5.7% (need 10%). Verdict: REFUTED on both criteria. The "partition collapses during grokking" hypothesis is wrong for continuous 2D tasks; it may only apply to discrete/algebraic tasks (modular arithmetic) where grokking enables compact algorithmic solutions.
   - ⚠ **Post-hoc review: flawed experiment.** (1) Spirals don't grok — continuous classification tasks lack the memorization→generalization phase transition. The "grokking" detected is just gradual fitting. (2) Raw pattern count is the wrong metric — grokking is about partition *rearrangement* (regions align with task structure), not pattern count *reduction*. Should track region-label purity. (3) Pattern increase is expected and correct for a network refining spiral boundaries on 2000 test points. **Superseded by item 46.**
   - Cycle: 021
   - Artifacts: `experiments/grokking_compression_catalyst.py`, `autoresearch/results/cycle_021_grokking.json`, `autoresearch/results/cycle_021_grokking_trajectory.png`

### Cycle 020

38. **Manifold curvature trajectory — GPT-2** *(REFUTED — cycle 020)*
   - At each layer, estimate local manifold curvature of the representation space. Method: for each probe token, fit local PCA in its k-NN neighborhood (k=50 in activation space). Curvature proxy = fraction of variance NOT explained by top-d principal components (d = estimated local intrinsic dimension). Average over tokens to get per-layer curvature. The curvature trajectory should reveal: (a) where the model builds curvature (early layers creating nonlinear features), (b) where it flattens (late layers approaching linearity). This is the geometric dual of the Procrustes test (item 36): Procrustes measures global linearity, curvature measures local manifold geometry. Correlate curvature with single-block swap damage (cycle 005).
   - Type: empirical
   - Confirmed if: curvature decreases monotonically in the final third of layers AND Spearman correlation between curvature and compression difficulty ≥ 0.4.
   - Refuted if: curvature has no clear trend or anti-correlates with compression difficulty.
   - Inconclusive if: curvature decreases but correlation with compression difficulty is weak (0.2–0.4).
   - Depends on: 9
   - **Outcome: REFUTED** — Curvature is nearly uniform across all 13 layers (range 0.0451–0.0471, <5% unexplained variance everywhere), with no monotone trend in the final third. Spearman rho = −0.42 (p=0.17, non-significant), indicating if anything a slight anti-correlation. The k-NN local PCA curvature proxy is insensitive to GPT-2 residual-stream geometry: LayerNorm at each block normalises representations before they are read by subsequent layers, collapsing curvature differences that would otherwise be visible. The signal is too compressed and homogeneous to serve as a compression predictor.
   - Cycle: 020
   - Artifacts: `autoresearch/results/cycle_020_curvature.json`, `autoresearch/results/cycle_020_curvature_trajectory.png`, `autoresearch/results/cycle_020_curvature_vs_damage.png`

### Cycle 019

37. **Activation pattern entropy cascade — toy ReLU** *(CONFIRMED — cycle 019)*
   - For each layer in a trained spirals network, compute Shannon entropy H(layer) of the activation pattern distribution over data. High H = many equally-used patterns = complex = hard to compress. Low H = few dominant patterns = compressible. Also compute conditional entropy H(pattern_L | pattern_{L-1}) — how much NEW partition information each layer adds. If conditional entropy is near zero for some layer, that layer is partitioning redundantly relative to the previous layer → safe to remove. Plot the entropy cascade: H(L1), H(L2|L1), H(L3|L1,L2), ... This is an information-theoretic characterization of partition redundancy, different from affine-map clustering (item 22) or volume analysis (item 34).
   - Type: empirical
   - Confirmed if: conditional entropy drops ≥50% for at least one layer AND low-conditional-entropy layers also have low single-affine residual (item 20 correlation).
   - Refuted if: conditional entropy is uniformly high across all layers (each layer adds substantial new partition structure).
   - Inconclusive if: conditional entropy varies but does not correlate with any compression signal from items 20/22.
   - Depends on: none
   - **Outcome: CONFIRMED** — Conditional entropy drops sharply after layer 1: cH(L2|L1)=1.64 bits (ratio 0.26), cH(L3|L1,L2)=0.26 bits (ratio 0.040), cH(L4|L1-3)=0.27 bits (ratio 0.043). Layers 2-4 all have single-affine residual r < 0.013, far below layer 1's r=0.168. The entropy cascade reveals that layer 1 establishes the bulk of the partition structure; later layers refine minimally. Spearman(cH_ratio, r)=0.40 (underpowered at n=4 but direction consistent).
   - Cycle: 019
   - Artifacts: `autoresearch/results/cycle_019_entropy_cascade.json`, `autoresearch/results/cycle_019_entropy_cascade.png`, `autoresearch/results/cycle_019_summary.md`

### Cycle 017

34. **Region volume Pareto: do large regions carry most task accuracy? — toy ReLU** *(INCONCLUSIVE — cycle 017)*
   - For a trained spirals ReLU network (2D input → 100D embedding → 4 layers × 64 hidden), enumerate all observed activation patterns on the full probe set. Define region "volume" as the fraction of probe inputs that fall in each region. Sort regions by volume in descending order. Compute the cumulative accuracy recovery as a function of regions kept (the "Pareto compression curve"): start with only the largest region, classify all test points using that region's A_ω, add the next largest region and allow each test point to use its own region's A_ω if it falls there, otherwise fall back to the nearest large region. If the Pareto curve is steep (top 20% of regions by volume covers ≥ 80% of correctly classified points), compression can be achieved by keeping only large regions and routing small-region inputs to the nearest large region.
   - Type: empirical
   - Confirmed if: the top 20% of regions by volume covers ≥ 70% of correctly classified test points.
   - Refuted if: regions have an approximately uniform volume distribution (Gini coefficient < 0.3), indicating no Pareto structure.
   - Inconclusive if: Pareto holds in 2D but disappears when the same analysis is run in the 100D embedding, suggesting the structure is input-space-specific.
   - Depends on: none
   - **Outcome: INCONCLUSIVE** — Gini=0.408 confirms a skewed (non-uniform) volume distribution, but the top 20% of regions (75 of 375) covers only 48.2% of correctly-classified test points, falling short of the 70% threshold. The region-count distribution is Pareto-skewed, but coverage is spread across many medium-sized regions rather than concentrated in the top 20%. Accuracy with A_ω fallback at 20% kept = 86.7% vs float 98.3%. Item 35 (GPT-2 analog) depends on 34 not being refuted — it can proceed since Pareto structure is present, just weaker than expected.
   - Cycle: 017
   - Artifacts: `autoresearch/results/cycle_017_pareto.json`, `autoresearch/results/cycle_017_pareto.png`

### Cycle 016

32. **GELU → 3-piece PWL approximation fidelity — toy GELU MLP** *(INCONCLUSIVE — cycle 016)*

### Cycle 012

22. **Effective number of affine behaviors via Jacobian k-means — toy ReLU** *(REFUTED — cycle 012)*
   - A ReLU network with K observed activation patterns has exactly K distinct affine maps A_ω. In practice, many A_ω are similar and can be served by a single representative. For a trained spirals ReLU network, extract all distinct activation patterns on probe data, compute their A_ω matrices using the MASO formula A_ω = W_L·diag(q^L)·…·W_1·diag(q^1), then run k-means on the flattened A_ω vectors. Find the minimum K* such that cluster-centroid A_ω matrices reproduce network outputs with ≤ 1% accuracy drop. K* / N_patterns is the functional redundancy ratio: how much of the activation-pattern diversity is semantically irrelevant. Plot K* vs. accuracy for a range of K to get a compression-accuracy curve.
   - Type: empirical
   - Confirmed if: K* / N_patterns ≤ 0.2 for the spirals network (i.e., fewer than 20% of observed patterns are functionally distinct), indicating high redundancy exploitable for compression.
   - Refuted if: K* / N_patterns > 0.7, indicating that most activation patterns compute meaningfully different functions.
   - Inconclusive if: accuracy drops sharply as soon as any merging is applied (K* ≈ N_patterns) even though many A_ω look geometrically similar, suggesting that functional similarity does not survive k-means clustering.
   - Depends on: none
   - **Outcome: REFUTED** — K*/N_patterns = 1.00 (375/375). Accuracy at K=64 (17% of patterns) = 74.9%, never recovers to within 1% of float (98.3%) until K=N_patterns. Affine maps are genuinely diverse (std 42–83 per component); k-means clustering cannot find redundancy. Implies experiment 23 (GPT-2 Jacobian k-means) is unmotivated per agenda dependency note. RTG spectral coarsening (exp 24) may still work as it respects local adjacency structure.
   - Cycle: 012
   - Commit: see below
   - Artifacts: `autoresearch/results/cycle_012_kmeans_toy.json`, `autoresearch/results/cycle_012_kmeans_curve.png`, `autoresearch/results/cycle_012_summary.md`


### Cycle 030

18. **Learned compressor generalization on held-out blocks** — **INCONCLUSIVE** *(cycle 030)*
   - Shared low-rank bottleneck (768→384→768) trained on blocks 0-7, evaluated on blocks 8-11 with bias-only adaptation (mean-shift). Beats geometric on only 1/4 held-out blocks (block 10 only, where shared PPL=117.4 vs geometric=122.1). Generalization ratio = 0.76 (held-out PPL / in-distribution train PPL), within the 1.25 threshold. But the train-block PPL itself is poor (mean=163.4 vs baseline=122.4): the shared U,V underfit because each block computes a qualitatively different function. The compressor memorizes a rough average rather than reusable structure. Per-block distillation achieves 122-132 PPL (near baseline) with the same parameter budget, confirming the bottleneck is task-specific. Geometric baseline wins on 3/4 blocks without any training.
   - Artifacts: `experiments/learned_compressor_generalization.py`, `autoresearch/results/cycle_030_generalization.json`

### Cycle 029

17. **Local-chart analytic compression inside one FFN block** — **REFUTED**: k-means clustering of FFN inputs into K local charts (K=4, K=8) at matched budget (384 total hidden channels) consistently hurts vs global geometric. Block MSE increases on all blocks (+41% for K=4 on block 2, +53% on block 11). Full-model PPL worse on all 3 blocks for both K values. Splitting the channel budget per chart causes each chart to be too thin; chart-local channel selection cannot overcome the loss from reduced per-chart capacity. The global top-k + lstsq refit remains the stronger approach because GELU FFN outputs are approximately a global low-dimensional function of the input, not a collection of locally separable functions.
   - Artifacts: `experiments/local_chart_ffn_compression.py`, `autoresearch/results/cycle_029_local_chart.json`

### Cycle 025

### Cycle 022

### Cycle 021

### Cycle 018

36. **Procrustes linearity profile + identity-layer pruning — GPT-2**
   - Outcome: **INCONCLUSIVE** — Linearity scores confirmed (7/12 layers ≥0.98, layers 3–10 all highly linear, consistent with "Secretly Linear" paper). Spearman correlation with cycle 005 damage = 0.557 (PASS). But identity replacement of top-2 most linear blocks (3, 4) causes 33.6% PPL increase (FAIL; criterion ≤10%). Individually: block 3 alone = 10.2%, block 4 = 8.5%, block 6 cheapest at 3.9%. Compounding makes simultaneous removal far worse than either alone. Linearity ≠ removability for joint removal — though single blocks near threshold. Block 0 catastrophic (2314%); block 11 severe (595%). Layers 1–2 have lower linearity (0.846, 0.852) consistent with their early representational role.
   - Cycle: 018
   - Artifacts: `experiments/procrustes_linearity_gpt2.py`, `autoresearch/results/cycle_018_linearity.json`, `autoresearch/results/cycle_018_linearity.png`

### Cycle 017

### Cycle 016

### Cycle 015

26. **Intrinsic dimension trajectory through depth (GPT-2)**
   - Outcome: **INCONCLUSIVE** — ID increases monotonically (1.6 → 7.5, not decreases) through depth. Final-third mean ID (7.32) is 62% higher than first-third (4.52), opposite of the ≥30% decrease criterion. Spearman correlation between |ΔID| and cycle-005 damage = -0.16 (p=0.62), well below the ρ≥0.4 threshold. The token-representation manifold becomes progressively higher-dimensional through GPT-2 depth; later layers do not simplify structure — they elaborate it.
   - Cycle: 015
   - Artifacts: `experiments/intrinsic_dim_trajectory.py`, `autoresearch/results/cycle_015_intrinsic_dim.json`, `autoresearch/results/cycle_015_intrinsic_dim.png`

### Cycle 014

25. **RTG spectral coarsening — GPT-2 FFN analog**
   - Outcome: **REFUTED** — Spectral clustering of k-NN Jacobian graph provides no advantage over k-means on raw Jacobians. Maximum relative improvement across all blocks and K values was 1.3% (block 2, K=2), far below the 15% threshold. Both methods show monotonically decreasing error with K, but at parity. Graphs were connected (1 component) in all 4 blocks tested. Result aligns with cycle 013 toy result — spectral coarsening adds no value when the Jacobian space is genuinely high-dimensional and non-clustered.
   - Cycle: 014
   - Artifacts: `experiments/rtg_spectral_gpt2.py`, `autoresearch/results/cycle_014_rtg_spectral_gpt2.json`, `autoresearch/results/cycle_014_rtg_spectral_gpt2.png`, `autoresearch/results/cycle_014_summary.md`

### Cycle 012

### Cycle 005

10. **Single-block swap localization curve**
   - Type: empirical
   - Outcome: **confirmed (proxy)** -- on the latest `gpt2` + Shakespeare sweep, `22/24` latest `>=4x` geometric swaps met the agenda criterion; only layer `0` failed
   - Cycle: 005
   - Commit: NONE (recovered from existing sweep artifacts; no replay commit)
   - Artifact: `autoresearch/results/cycle_005_summary.md`
   - Key finding: Single-block FFN compression is sharply depth-sensitive rather than uniformly broken; most middle and late blocks survive `8x-16x` analytic compression with `<=10%` relative perplexity increase, while layer `0` is a clear hard failure case.

### Cycle 001

1. **Survey: Balestriero's piecewise linear / spline theory of deep learning**
   - Type: literature
   - Outcome: **confirmed** — 6 concrete, implementable techniques identified (≥3 required)
   - Cycle: 001
   - Commit: (see cycle_001_manifest.txt)
   - Artifact: `autoresearch/surveys/balestriero_spline_theory.md`
   - Key finding: Activation pattern census (Technique 1) and affine map extraction (Technique 2) are directly actionable from MASO theory; SplineCam (Technique 3) provides exact 2D enumeration. Gap: analysis-to-synthesis route for minimal network construction remains open.

### Cycle 003

7. **Survey: Non-distillation model compression methods (2023-2026)**
   - Type: literature
   - Outcome: **confirmed** — found ≥2 papers with working no-SGD compression methods
   - Cycle: 003
   - Commit: 66021d0
   - Artifact: `autoresearch/surveys/literature_map.md`
   - Key findings:
     - **Serra et al. NeurIPS 2021 (2102.07804)**: ReLU stability criterion — MIP-based dead/always-active neuron detection → 56% connection removal, 183x faster than prior art. No SGD required.
     - **Berzins 2023 (2306.07212)**: GPU-accelerated partition extraction + §5.4 pruning demo → 78% parameter reduction (2,50,50,1 → 2,25,19,1) via dead + always-active removal. Constructive, no SGD.
     - **3-Phase algorithm** identified: Phase 1 (dead neuron removal) + Phase 2 (always-active collapse to ≤D neurons/layer) = solvable analytically. Phase 3 (merging genuinely unstable neurons) = open problem.
     - Depth reduction ruled out: Arora et al. (1611.01491) proves exponential neuron cost for depth reduction → target width compression only.

### Cycle 002

2. **Survey: Tropical geometry and neural networks**
   - Type: literature
   - Outcome: **inconclusive** — tropical polynomial form is exponentially large (intractable for deep nets); practical TropNNC algorithm uses zonotope clustering but never builds a compact tropical form
   - Cycle: 002
   - Commit: TBD
   - Artifact: `autoresearch/results/cycle_002_summary.md`
   - Key finding: Tropical geometry = dead end for direct compression. One salvageable idea: neurons-as-zonotope-generators motivates clustering-based pruning (overlaps with MASO region merging). Do NOT pursue tropical algebra further.

### Cycle 004

9. **Benchmark lock-in and asset audit**
   - Type: analysis
   - Outcome: **confirmed** — exact-open nanochat benchmark and explicit `gpt2` proxy benchmark are both now documented, with concrete path conventions and acquisition procedures
   - Cycle: 004
   - Commit: e25470e
   - Artifact: `autoresearch/results/cycle_004_summary.md`
   - Key finding: the exact-open target is `karpathy/nanochat` base checkpoints plus `climbmix-400b-shuffle`, but those assets are not present locally; the only benchmark runnable immediately on this machine is the cached `gpt2` proxy path, which must remain clearly labeled proxy-only.

### Cycle 006

11. **Stable and near-linear channel census in pretrained FFNs** ⚠ **FLAWED: MP2 (detection method mismatched to GELU)**
   - Type: empirical
   - Outcome: **refuted (proxy)** — 0/12 proxy gpt2 FFN blocks reached ≥5% stable-or-near-linear channels; 10/12 stayed below 1%
   - Cycle: 006
   - Commit: NONE (recovered from hidden runtime artifacts)
   - Artifact: `autoresearch/results/cycle_006_summary.md`
   - Key finding: Exact stable-channel pruning (dead + always-active) is insufficient for GELU transformer FFNs — stable channels account for < 1% of hidden width in most blocks on the proxy benchmark.
   - ⚠ **Post-hoc review**: same GELU detection flaw as cycle 028 — stability thresholds calibrated for ReLU zero-output, but GELU never truly zeros. Pre-GELU input values should be used for dead channel detection. Superseded by cycle 032 (item 47).

### Cycle 007

12. **Progressive multi-block accumulation curve**
   - Type: empirical
   - Outcome: **confirmed (proxy)** — easiest-first outperforms hardest-first by 27.4% at k=6 and 36.2% at k=9, exceeding the ≥25% threshold. Advantage grows with block count.
   - Cycle: 007
   - Commit: 980dec5
   - Artifact: `autoresearch/results/cycle_007_summary.md`, `autoresearch/results/cycle_007_accumulation.json`
   - Key finding: Ordering matters enormously for multi-block accumulation. Hardest-first is catastrophically worse. Single-block damage ranking from cycle 005 correctly predicts safe accumulation order. Sweet spot is k=4-5 easiest blocks (ppl < 150, < 25% degradation). All-block 16x compression fails regardless of ordering (ppl=403).

### Cycle 008

13. **Probe-corpus robustness of geometry signals**
   - Type: empirical
   - Outcome: **confirmed (proxy)** — 12/12 blocks pass both criteria; channel score Spearman ≥ 0.95, selected-set Jaccard ≥ 0.82, Jacobian cosine ≥ 0.997 across disjoint Shakespeare probe halves
   - Cycle: 008
   - Commit: 63fb41d
   - Artifact: `autoresearch/results/cycle_008_summary.md`, `autoresearch/results/cycle_008_robustness.json`
   - Key finding: Compression geometry signals are highly robust to probe-corpus splits. Channel rankings, Jacobian spectra, and selected hidden-unit sets are nearly identical across disjoint probe halves. This validates the single-probe approach used in all prior cycles.

### Cycle 009

19. **Jacobian variance atlas: which FFN blocks are near-linear on probe data?**
   - Type: empirical
   - Outcome: **refuted (proxy)** — 0/12 blocks have L ≥ 0.85; all blocks have L < 0.6 (max L = 0.38). Spearman ρ = −0.14, p = 0.66.
   - Cycle: 009
   - Commit: NONE
   - Artifact: `autoresearch/results/cycle_009_summary.md`, `autoresearch/results/cycle_009_jacobian_atlas.json`
   - Key finding: No GPT-2 FFN block is near-linear on the probe distribution. Most blocks have negative L (mean Jacobian deviation exceeds mean norm), indicating substantial per-token functional diversity. Single-affine collapse is not viable for any block. Multi-region approaches (items 22–25) are the necessary path.

### Cycle 011

21. **Single-affine collapse quality — GPT-2 FFN blocks** ⚠ **FLAWED: MP3 (underdetermined fitting)**
   - Type: empirical
   - Outcome: **refuted (proxy)** — 0/12 blocks have r_held < 0.15 (best: block 2 r_held=0.178). Swap of best block causes +33.7% PPL increase (> 20% threshold). Best-two simultaneous swap causes +344% PPL increase.
   - Cycle: 011
   - Commit: c4ea941
   - Artifact: `autoresearch/results/cycle_011_summary.md`, `autoresearch/results/cycle_011_single_affine_gpt2.json`, `autoresearch/results/cycle_011_single_affine_gpt2.png`
   - Key finding: No GPT-2 FFN block is approximable by a single global affine map on held-out tokens. Block 2 comes closest (r_fit=0.018) but fails to generalize (r_held=0.178), indicating the affine fit memorizes rather than captures structure. Even the best block causes 33.7% PPL increase when swapped, confirming that GELU nonlinearity is functionally critical. This is consistent with cycle 009 (Jacobian variance too high for single-affine). Multi-region approaches (items 22–25) are required.
   - ⚠ **Post-hoc review**: 512 probe tokens fitting 769 parameters (768 weights + 1 bias) is underdetermined. The r_fit=0.018 → r_held=0.178 gap confirms overfitting. A design review would have caught this via R3 and required n_samples >= 1538.

### Cycle 010

20. **Single-affine collapse quality — toy ReLU**
   - Type: empirical
   - Outcome: **confirmed** — 3/4 hidden layers have r < 0.15 on held-out data. r values: L1=0.430, L2=0.050, L3=0.033, L4=0.024. r decreases monotonically with depth.
   - Cycle: 010
   - Commit: 3d07fcd
   - Artifact: `autoresearch/results/cycle_010_summary.md`, `autoresearch/results/cycle_010_single_affine.json`
   - Key finding: Deep layers of a trained ReLU MLP (layers 2–4) are near-linear on the data distribution (r < 0.06). Only the first hidden layer performs genuinely nonlinear computation (r=0.43). Dead/always-active neuron fraction correlates negatively with r: layers with more stable neurons are more affinely collapsible. Train accuracy was 99.45%. The single-affine collapse is viable for compression of deep layers.

### Cycle 012

### Cycle 013

24. **RTG spectral coarsening: Fiedler-based region merging — toy ReLU**
   - Type: implementation + empirical
   - Outcome: **inconclusive** — spectral beats random by up to +9.4pp at K=8,16 (signal exists), but accuracy plateaus at 55-60% vs float 95.5%. Hamming-1 RTG is highly fragmented (79 components, 330 nodes).
   - Cycle: 013
   - Commit: 202daf9
   - Artifact: `autoresearch/results/cycle_013_rtg_spectral.json`, `experiments/rtg_spectral_toy.py`
   - Key finding: RTG spectral signal is real (beats random significantly) but Hamming-1 graph is too sparse for useful compression — 79 disconnected components means spectral clustering must average across unrelated regions. Item 25 (GPT-2 analog with k-NN Jacobian graph) avoids this sparsity issue and remains eligible.

### Cycle 033

48. **Easiest-first multi-block accumulation up to 6 blocks — GPT-2 FFN** *(CONFIRMED — cycle 033)*
   - Re-ran the easiest-first multi-block accumulation (cycle 007) with the improved geometric method (cycles 026-027): topk channel selection + centroid similarity merge signal at 4× compression. Blocks [7,5,4,1,3,6] accumulated in easiest-first order. k=1: +0.56%, k=2: +2.60%, k=3: +6.38%, k=4: +8.84%, k=5: +17.46%, k=6: +25.99%. Degradation is sub-additive through k=4, then becomes super-additive (k=4→k=5 jump is +8.6pp). The improved method significantly outperforms cycle 007 (which had k=4 at +9.89% with the old method).
   - Outcome: CONFIRMED — k=4 PPL=130.96 (+8.84%) is within the ≤10% threshold
   - Cycle: 033
   - Commit: NONE (recovered from incomplete prior session)
   - Artifacts: `autoresearch/results/cycle_033_accumulation.json`, `autoresearch/results/cycle_033_accumulation.png`, `experiments/multiblock_accumulation_6block.py`

### Cycle 034

49. **Block-specific compression ratio sweep — GPT-2 FFN** *(INCONCLUSIVE — cycle 034, audit-downgraded)*
   - Swept compression ratios {2×, 3×, 4×, 5×, 6×} independently on each of 6 easiest blocks [7,5,4,1,3,6]. All 6 blocks tolerate 6× compression within 5% PPL threshold (29/30 points). Block 5 at 2× is the only failure (+5.41%). PPL is non-monotonic with ratio (higher compression sometimes gives sub-baseline PPL). Reconstruction MSE increases monotonically. Per-block max safe ratios: {7:6, 5:6, 4:6, 1:6, 3:6, 6:6}. Audit downgraded to INCONCLUSIVE: single seed + 50 eval sequences insufficient to confirm sub-baseline PPL values are real vs noise. The directional finding (high compression tolerance) is likely correct but needs multi-seed confirmation.
   - Outcome: INCONCLUSIVE (audit-downgraded from CONFIRMED)
   - Cycle: 034
   - Commit: b8dad28
   - Artifacts: `autoresearch/results/cycle_034_ratio_sweep.json`, `autoresearch/results/cycle_034_ratio_sweep.png`, `experiments/block_ratio_sweep_gpt2.py`

### Cycle 035

52. **Multi-seed confirmation of single-block compression tolerance at 6x — GPT-2 FFN** *(INCONCLUSIVE — cycle 035)*
   - 3 seeds (42, 123, 7), 200 eval sequences each, blocks [7,5,4,1,3,6] at 6× with topk + random null baseline. Baselines differ across seeds (123.4, 125.9, 125.0) confirming shuffle_seed works. All 6/6 blocks within 5% threshold (mean rel PPL: block 7 +0.51%, block 5 -0.20%, block 4 -0.13%, block 1 -1.00%, block 3 -1.55%, block 6 -0.57%). Cross-seed std low (0.18-0.83%). Topk beats random on all 6 blocks. BUT: 5/6 blocks show sub-baseline mean PPL — the non-monotonicity from cycle 034 is REAL, not noise. The lstsq refit at 6× acts as regularization, removing channels that hurt generalization. Verdict: INCONCLUSIVE because confirmation criterion requires no sub-baseline mean PPL.
   - Outcome: INCONCLUSIVE — 6/6 within 5% but 5/6 sub-baseline (regularization effect confirmed real)
   - Cycle: 035
   - Artifacts: `autoresearch/results/cycle_035_multiseed_6x.json`, `autoresearch/results/cycle_035_multiseed_6x.png`, `experiments/multiseed_6x_confirmation_gpt2.py`

### Cycle 036

53. **Heterogeneous-ratio multi-block accumulation — GPT-2 FFN** *(INCONCLUSIVE — cycle 036)*
   - 6× per-block accumulation on blocks [7,5,4,1,3,6]. k=1: -0.16%, k=2: +2.04%, k=3: +6.40%, k=4: +9.59%, k=5: +17.40%, k=6: +26.57%. Baseline 120.32. Compared to cycle 033 (uniform 4×): k=4 at +9.59% vs +8.84% — only +0.75pp worse despite 50% more per-block compression (6× vs 4×). The compounding degradation is dominated by block count, not per-block ratio. Curves nearly overlap at k=3 and diverge slightly at k=4+. Average compression at k=4 is 6.0× (vs criterion >4×).
   - Outcome: INCONCLUSIVE — k=4 at +9.59% is in the 8-12% range; better compression but unclear win over uniform 4×
   - Cycle: 036
   - Artifacts: `autoresearch/results/cycle_036_accumulation_6x.json`, `autoresearch/results/cycle_036_accumulation_6x.png`, `experiments/multiblock_accumulation_6x.py`

### Cycle 037

56. **Total FFN parameter reduction budget at <10% PPL — GPT-2 FFN** *(REFUTED — cycle 037)*
   - k=4@6× (blocks 7,5,4,1): mean +9.77% PPL (±0.26%, 2 seeds), 27.8% total FFN param reduction. k=4 budget is solid. k=5 heterogeneous: tested blocks 3 and 6 as 5th at 2×/3×/4×. ALL k=5 configs exceed +15% (best: block 6 at 4× = +16.99%). The block count wall is hard — reducing the 5th block's ratio to 2× still gives +20%. Per-block ratio doesn't matter; the compounding through the residual stream is the binding constraint.
   - Outcome: REFUTED — all k=5 configs exceed 15% PPL threshold (best +16.99%)
   - Positive finding: k=4@6× achieves 27.8% FFN param reduction at <10% PPL
   - Cycle: 037
   - Artifacts: `autoresearch/results/cycle_037_param_budget.json`, `autoresearch/results/cycle_037_param_budget.png`, `experiments/ffn_param_budget_hetero_k5.py`

### Cycle 038

57. **Block ordering optimization for 5-block stacking — GPT-2 FFN** *(REFUTED — cycle 038)*
   - All C(6,5)=6 five-block combos from the 6 easiest blocks at 6×. Best: drop block 4 (+16.12%). Worst: drop block 1 (+20.88%). Spread 4.76pp — combo matters but not enough. All combos >15%. Insertion order is a no-op (pipeline compresses all blocks independently from float activations). The k=5 wall is robust across all block selections.
   - Outcome: REFUTED — all 5-block configs exceed 15% regardless of combo
   - Cycle: 038
   - Artifacts: `autoresearch/results/cycle_038_block_ordering.json`, `autoresearch/results/cycle_038_block_ordering.png`, `experiments/ffn_block_ordering_k5.py`

### Cycle 039

58. **Inter-block error correction via residual-stream lstsq adjustment — GPT-2 FFN** *(CONFIRMED — cycle 039)*
   - Affine correction C fit via lstsq on probe data at the residual stream after the last compressed block (block 8 input). k=5 (blocks 7,5,1,3,6 at 6×): +16.12% → +12.53%, recovery 3.6pp (≥3pp threshold). k=4 also benefits: +9.59% → +8.35%, recovery 1.23pp (no overcorrection). Correction MSE reduction: 20.8%. Correction is high-rank (rank-95 = 392/768) — the error is distributed, not low-rank, but a significant affine component exists. Zero extra inference parameters (correction applied via hook).
   - Outcome: CONFIRMED — k=5 correction recovers 3.6pp (≥3pp criterion met), k=4 not worsened
   - Cycle: 039
   - Artifacts: `autoresearch/results/cycle_039_residual_correction.json`, `autoresearch/results/cycle_039_residual_correction.png`, `experiments/ffn_residual_correction_k5.py`

### Cycle 040

54. **Compression ceiling scan: 8x and 10x single-block — GPT-2 FFN** *(CONFIRMED — cycle 040)*
   - All 3 blocks (7, 5, 4) tolerate 8× AND 10× compression with mean PPL within 5% (most sub-baseline). 3 seeds, 200 eval seqs. Results: block 7: 6×=+0.51%, 8×=+0.12%, 10×=+0.05%. Block 5: 6×=-0.20%, 8×=-0.49%, 10×=-0.59%. Block 4: 6×=-0.13%, 8×=-0.56%, 10×=-0.54%. Cross-seed std 0.17-0.59% (low). 6× cross-check matches cycle 035 exactly. The ceiling has NOT been found — even 10× compression is essentially free for single-block swaps. The lstsq refit regularization effect strengthens with compression.
   - Outcome: CONFIRMED — 3/3 blocks tolerate 8× at ≤10% (all also tolerate 10×)
   - Cycle: 040
   - Commit: pending
   - Artifacts: `autoresearch/results/cycle_040_ceiling_scan.json`, `autoresearch/results/cycle_040_ceiling_scan.png`, `experiments/compression_ceiling_scan_gpt2.py`

### Cycle 041

59. **Generalization probe: single-block 4× geometric compression on GPT-2-medium — FFN** *(CONFIRMED — cycle 041)*
   - GPT-2-medium (24 layers, d=1024, FFN=4096). 6 blocks (0,4,8,12,16,20) at 4× with 3 seeds, 200 eval seqs. Baseline PPL: 94-97. Block 0: +29.74% (catastrophic, same as GPT-2-small). Blocks 4,8,12,16,20: all within +1% PPL (block 8 actually sub-baseline at -0.16%). 5/6 blocks pass ≤10% threshold. The method generalizes cleanly from GPT-2-small to GPT-2-medium with identical block-0-hard, rest-easy pattern.
   - Outcome: CONFIRMED — 5/6 blocks pass (≥4 required)
   - Cycle: 041
   - Artifacts: `autoresearch/results/cycle_041_medium_generalization.json`, `autoresearch/results/cycle_041_medium_generalization.png`, `experiments/gpt2_medium_generalization_probe.py`

### Cycle 042

60. **Multi-block stacking at 10× per-block — GPT-2 FFN** *(CONFIRMED — cycle 042)*
   - k=4 blocks [7,5,4,1] at 10× compression, 3 seeds, 200 eval seqs. Uncorrected: +10.62% ±0.82%. With residual correction at block 8: **+8.45% ±0.66%**, recovery 2.17pp. This essentially matches k=4@6× corrected (+8.35% from cycle 039). Per-block ratio (6× vs 10×) makes no difference once correction is applied. Total FFN param reduction at 10×: ~30%.
   - Outcome: CONFIRMED — k=4@10x corrected mean +8.45% (≤12% threshold)
   - Cycle: 042
   - Artifacts: `autoresearch/results/cycle_042_multiblock_10x.json`, `autoresearch/results/cycle_042_multiblock_10x.png`, `experiments/multiblock_10x_stacking_gpt2.py`

### Cycle 043

61. **Multi-point residual correction at every compressed block boundary — GPT-2 FFN** *(INCONCLUSIVE — cycle 043)*
   - Fitted N sequential affine corrections (one per compressed block in layer order). k=4: multi-point (+7.68%) is WORSE than single-point (+6.22%) by 1.45pp — intermediate corrections interfere (block 4 MSE increases 41% after its correction). k=5: multi-point (+12.08%) barely improves over single-point (+12.31%) by 0.23pp — within noise. Per-correction diagnostic: first correction (block 1) reduces MSE by 40.8%, but subsequent corrections are destructive or neutral. Cascading lstsq corrections are not composable.
   - Outcome: INCONCLUSIVE — k=5 multi-point at +12.08% touches but doesn't clearly beat 12% threshold; k=4 multi-point is worse than single-point
   - Cycle: 043
   - Artifacts: `autoresearch/results/cycle_043_multipoint_correction.json`, `autoresearch/results/cycle_043_multipoint_correction.png`, `experiments/ffn_multipoint_correction.py`

### Cycle 044

62. **Domain robustness: compression quality on non-Shakespeare text — GPT-2 FFN** *(INCONCLUSIVE — cycle 044)* ⚠ **CRITICAL FINDING**
   - Shakespeare→Shakespeare (A): corrected +6.29% — works great. WikiText-2→WikiText-2 corpus-specific (B): corrected +31.86% — severe degradation even with corpus-specific fitting. Shakespeare→WikiText zero-shot (C): corrected +120.57% — catastrophic. Shakespeare channels + WikiText correction refit (D): +41.20% — correction refit helps but still bad. The method is HIGHLY CORPUS-SPECIFIC. Shakespeare's narrow distribution allowed high compression that doesn't transfer. On broader text (WikiText-2), even corpus-specific compression degrades PPL by 32%. All prior "good" results are Shakespeare-specific.
   - Outcome: INCONCLUSIVE per criteria (gap 88.7pp > 3pp, but B is also >10% so refutation criterion isn't met)
   - Cycle: 044
   - Artifacts: `autoresearch/results/cycle_044_domain_robustness.json`, `autoresearch/results/cycle_044_domain_robustness.png`, `experiments/domain_robustness_wikitext.py`

### Cycle 045

63. **Practical compression recipe: consolidated findings summary — GPT-2 FFN** *(INCONCLUSIVE — cycle 045)*
   - Recipe document written and validated end-to-end. Shakespeare replication: +8.52% ±0.62% (matches target +8.35% within 1pp). WikiText-2 corpus-specific: +44.31% ±1.02% (confirms corpus specificity from cycle 044). Recipe is self-contained and reproducible for narrow-distribution text but requires distribution-matched probing data. The fundamental limitation is not the compression method but the corpus sensitivity.
   - Outcome: INCONCLUSIVE — recipe works but requires corpus-specific tuning (as predicted by cycle 044)
   - Cycle: 045
   - Artifacts: `autoresearch/results/cycle_045_recipe.md`, `autoresearch/results/cycle_045_validation.json`, `autoresearch/results/cycle_045_validation.png`, `experiments/compression_recipe_validation.py`

### Cycle 046

55. **Lstsq refit quality degradation analysis — GPT-2 FFN** *(CRASH — cycle 046)*
   - SIGSEGV during model loading. System memory pressure after 13 consecutive experiment cycles. Not a code bug — retry in fresh session.
   - Outcome: CRASH
   - Cycle: 046

### Cycle 047

27. **Persistent homology of activation manifold through depth — toy ReLU** *(REFUTED — cycle 047)*
   - Trained 2→32×4→1 spirals network (acc=98.4%). Total persistence L1-L4: 122.7, 119.1, 113.4, 227.1. Only 2/3 transitions decrease (L4 spikes up); 80% monotonicity criterion NOT MET. Largest Betti drop at L3 but lowest r is at L4 — rank-1 prediction wrong. Spearman(drop, r) = +0.5 (wrong sign, n=3). Random-init monotonically decreases (81→39→18→4), trained network does not. Gaussian null (1917-3562) much higher than trained, confirming PH measures real structure — but the structure does not simplify monotonically through depth.
   - Outcome: REFUTED
   - Cycle: 047
   - Commit: f2d83c5
   - Artifacts: `experiments/phom_toy_relu.py`, `autoresearch/results/cycle_047_phom_toy.json`, `autoresearch/results/cycle_047_phom_toy.png`

### Cycle 048

30. **Local complexity proxy as a compression difficulty predictor — GPT-2** *(CONFIRMED — cycle 048)*
   - Jacobian Frobenius std (P1) strongly predicts compression difficulty: rho_full=0.839 at 8x (p=0.0003), rho_without_L0=0.818 (p=0.001). At 16x: rho_full=0.608, rho_without_L0=0.527. Leverage-point guard passed. Spectral norm and effective rank proxies failed. P1 is a cheap compression-difficulty oracle.
   - Outcome: CONFIRMED
   - Cycle: 048
   - Commit: 0079ae1
   - Artifacts: `autoresearch/results/cycle_048_local_complexity_gpt2.json`, `autoresearch/results/cycle_048_local_complexity_gpt2.png`, `experiments/gpt2_local_complexity.py`

### Cycle 049

51. **Per-block learned compressor with block-conditioned bottleneck — GPT-2 FFN** *(REFUTED — cycle 049)*
   - Geometric baseline wins 11/12 blocks even with 8x more training data (32K tokens). 500-step per-block distillation cannot match analytical lstsq refit. Data scarcity was not the bottleneck in cycle 030.
   - Outcome: REFUTED
   - Cycle: 049
   - Commit: 72ebee5
   - Artifacts: `autoresearch/results/cycle_049_per_block_data_scaling.json`, `experiments/per_block_compressor_data_scaling.py`

### Cycle 050

65. **Oracle-guided adaptive compression ratio — GPT-2 FFN** *(REFUTED — cycle 050)*
   - Variable hidden dims break residual correction. Oracle +12.33% vs uniform control +6.36%. Pre-correction PPL is identical (~10.4%) — divergence is entirely at the correction step. Variable dims produce incompatible residual structure for single affine boundary correction.
   - Outcome: REFUTED
   - Cycle: 050
   - Commit: 37bdb1d
   - Artifacts: `autoresearch/results/cycle_050_oracle_adaptive.json`, `autoresearch/results/cycle_050_oracle_adaptive.png`, `experiments/oracle_adaptive_compression_gpt2.py`

### Cycle 051

67. **Lstsq refit residual analysis — GPT-2 FFN** *(INCONCLUSIVE — cycle 051)*
   - Design review rejected in round 2 due to incorrect discarded-subspace projection formula. Item remains scientifically valid and can be retried with corrected methodology.
   - Outcome: INCONCLUSIVE (design review reject)
   - Cycle: 051
   - Commit: NONE

### Cycle 052

68. **Corrected lstsq refit residual rank analysis — GPT-2 FFN** *(INCONCLUSIVE — cycle 052)*
   - Design review REJECTED: at 6× compression, 2560 discarded neurons' weight columns over-span the 768-dim output space → QR projection becomes identity (discarded_energy = 100% trivially). Subspace decomposition is uninformative by construction (MP6). The core question (is residual rank corpus-dependent?) remains valid but requires a different decomposition approach — e.g., project onto col(W_kept) instead, or drop subspace decomposition and compare total residual SVD rank directly.
   - Outcome: INCONCLUSIVE (design review reject — MP6 uninformative metric)
   - Cycle: 052
   - Commit: NONE

### Cycle 099

112. **Inter-block error correction via residual-stream projection — GPT-2 FFN** *(REFUTED — cycle 099)*
   - Residual-stream error after compressed FFN blocks is diffuse, NOT low-rank. rank_95 ≈ 500–636 (out of 768 dims). Projecting out top-r error directions causes catastrophic signal destruction: k=4 PPL goes from ~70 to ~33k (r=4), k=8 from ~199 to ~184k. Error subspace overlaps heavily with useful-signal subspace. Even Adam polish on top of projection can't recover (PPL ~200 vs uncorrected ~70). Corrected cascade rate is nonsensical (c ≈ 4.7 vs uncorrected c ≈ 0.06). The inter-block projection correction approach is fundamentally flawed for this architecture.
   - Outcome: REFUTED
   - Cycle: 099
   - Artifacts: `experiments/residual_projection_correction_gpt2.py`, `autoresearch/results/cycle_099_residual_projection.json`, `autoresearch/results/cycle_099_residual_projection_spectra.png`, `autoresearch/results/cycle_099_residual_projection.png`

### Cycle 105

116. **GELU negative-leak aware compression — GPT-2 FFN** *(INCONCLUSIVE — cycle 105)*
   - Design review REJECT round 2: lstsq target ordering issue (b_leak added post-hoc to lstsq creates systematic overshoot; target should be y - b_leak). Item remains scientifically valid and can be retried with corrected methodology.
   - Outcome: INCONCLUSIVE (design review reject)
   - Cycle: 105
   - Commit: NONE

### Cycle 107

117. **Output-space direct regression bypass — GPT-2 FFN** *(INCONCLUSIVE — cycle 107, audit corrected from confirmed)*
   - Direct rank-r regression vs topk+lstsq at matched params on blocks [1,4,7,11]. Middle blocks (4, 7) show near-parity (direct wins by 0.1-0.5%), early/late blocks (1, 11) topk wins by 2-27%. Audit downgraded: topk baseline had pseudo-replication (identical across seeds), margins within noise. Key interpretability finding: block-dependent linearity gradient (middle blocks nearly linear, edges need nonlinearity).
   - Outcome: INCONCLUSIVE (audit corrected — pseudo-replication, margins within noise)
   - Cycle: 107
   - Artifacts: `experiments/direct_regression_bypass_gpt2.py`, `autoresearch/results/cycle_107_direct_regression.json`, `autoresearch/results/cycle_107_direct_regression.png`

### Cycle 173

184. **Unified per-block regime ablation table — GPT-2 FFN** *(CONFIRMED — cycle 173)*
   - C3 (replace negative with token-mean) within +2% PPL of float baseline in 12/12 blocks (max delta +1.4%). Negative regime IS absorbable as constant bias. C1 (kill neg) causes +4-233% damage; C2 (kill pos) causes +6-970k% damage. Block 0 is uniquely sensitive to both positive regime removal and constant-output replacement. Interaction gap always negative: independent kills worse than mean replacement.
   - Outcome: CONFIRMED
   - Cycle: 173
   - Artifacts: `experiments/regime_ablation_table_cycle173.py`, `autoresearch/results/cycle_173_regime_ablation.json`, `autoresearch/results/cycle_173_regime_ablation.png`
