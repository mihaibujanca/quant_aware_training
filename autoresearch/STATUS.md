# Research Status — 240 Cycles (001–240)

## Latest: Cycle 240

250. **K=1 Adam correction at 6 blocks** — REFUTED (interesting=true)
   - BLOCKS=[1,2,4,5,7,8] K=1 (9216 trainable), 3 seeds. **best_mean=96.886 ± 0.071** (96.943/96.786/96.929) ≥ 70 refuted threshold by +26.9 PPL. Init=203.246 (much worse than cycle 221's K=512 init 86.7). Cycle 230's K=1 3-block (59.245) WORKED on mid-cascade slack; at 6 blocks the single-channel projection compounds catastrophically. **Channel capacity binds at 6 blocks even though it doesn't at 3.** Combined with cycle 231 (K=128 6-block 67.562 vs K=512 59.515), the K-vs-PPL curve at 6 blocks is much steeper than at 3. The 59.5 ceiling is NOT a pure structural cascade-depth floor — channel capacity becomes a real constraint when stacked.

## Cycle 239

249. **gelu_noneg + μ + LoRA r=8 disjoint splits** — CONFIRMED (interesting=true)
   - Cycle 234 leakage rerun with disjoint calib[:200] / eval[200:300] (asserted disjoint). 3 seeds: **final_eval_ppl = 47.692 ± 0.143** (47.666/47.531/47.879). ≤48 confirmed by 0.31 PPL. **LoRA beats float on eval slice by 6.20 PPL (-11.5%)** — gelu_noneg + per-channel μ + LoRA r=8 (368K trainable params) GENERALIZES on held-out. Cycle 234's leaked 32.411 was indeed ~15 PPL of calib overfit (47.69 - 32.41 = +15.28); the residual generalizes. Lands between cycle 210 K=512 W_out (46.556) and cycle 223 K=128 W_out (48.989) at ~similar 368K trainable param budget but via a STRUCTURALLY DIFFERENT mechanism (activation modification + low-rank c_proj residual instead of channel selection). Closes cycle 234 audit loop.

## Cycle 238

248. **T4: Activation patching across token pairs (bump-only)** — REFUTED (interesting=true)
   - 100 (A, B, p) pairs from wikitext-2 val (margin>2 both sides), 5 blocks × 100 = 500 records. Per-block MEDIAN bump_attribution: [0.0083, 0.0117, -0.0093, -0.0038, 0.0167] — all under 0.05 refuted gate. Mean diverges (block 9 mean 2.79 vs median 0.017, std 27.15) due to unbounded ratio tails — design review pre-approved median substitution. Median was pre-declared in script, not post-hoc. **Critical metric weakness**: full_patched_delta on B_top is positive in only 42-64% of pairs (denominator unreliable), itself a negative result on token-specific transport. **Closes the 4-item bump investigation coherently**: cycles 235 (T1) CONFIRMED-G3, 236 (T2) INCONCLUSIVE wte mismatch, 237 (T3) CONFIRMED 23k× subspace ablation, 238 (T4) REFUTED per-pair attribution. **The bump subspace is causally critical aggregate-wise but NOT linearly decodable AND NOT token-specific.** Substantial reframing of cycle 218's "inter-block communication" claim — the bump encodes collective cascade behavior, not interpretable per-token features.

## Cycle 237

247. **T3: Bump subspace ablation vs random subspace baseline** — CONFIRMED (interesting=true)
   - Top-10 right SVs of centred Δ̃ per block (from cycle 235 T1) ablated via projector P_l = V_l V_l^T applied to block output. baseline=34.514 (wikitext-2 val 200×512). **joint_bump_ppl = 2,344,386 vs joint_random_mean = 101.298 ± 8.137 (3 seeds) → ratio 23,143× ≥ 1.5 gate ✓**. **9/12 blocks ≥ 2× per-block random gate ✓**. Per-block ratios: [19.9, 12.9, 10.6, 30.2, 22.1, 25.2, **797.7**, 11.2, 1.65, 1.45, 1.51, **928.4**]. **Block 6 and 11 are single-block catastrophic** (must NOT compress); **blocks 8/9/10 are bump-random-like** (compression candidates). Full-FFN top-10 baseline destroys DIFFERENT blocks (3 and 11), confirming bump subspace is structurally distinct from raw FFN top PCs. **Reconciles cycle 236 T2 INCONCLUSIVE**: T2 killed the wte@v linear DECODER, but T3 confirms the SUBSPACE is causally critical via direct PPL ablation. The bump directions are real, just non-linearly decoded.

## Cycle 236

246. **T2: Causal intervention on top bump directions** — INCONCLUSIVE (interesting=true)
   - For each block, top 5 T1 directions intervened with α∈[-3,-1,0,1,3]·std on held-out 1024 tokens. **Effects are real**: max effect_size = 29.23× random baseline (block 11, dir 0); 18/60 directions cross effect≥3 gate; per-block max effect monotone with depth [5.23, 3.62, 2.34, 2.72, 2.62, 2.20, 3.47, 2.85, 4.20, 5.88, 8.05, **29.23**]. Refute branch ruled out. **BUT T1's wte @ v vocab predictions FAIL to match actual logit shifts**: only 1/60 directions reach jaccard ≥ 0.3. Joint confirm gate (effect≥3 AND jaccard≥0.3) passes 0/12 blocks. This is exactly the "directions cause shifts but don't match T1 predictions" inconclusive case. **T1's wte-column-space decoder is the wrong probe** — the bump directions are causally real but their effect propagates through non-linear downstream paths, not through a simple unembedding readout. Cycle 218's "Δ̃ writes into c_fc reading subspace" story survives only at the weak (correlational) level. T3 (cycle 237) is the next gating test.

## Cycle 235

245. **T1: Centred Δ̃ feature SVD with concrete example extraction** — CONFIRMED (G3 flagged)
   - For each of 12 blocks, computed Δ̃_l per token on wikitext-2 train (200 seqs × 512 tokens = ~100K tokens), SVD → top 32 right singular vectors, decoded each direction via wte.weight @ v for vocab projection + top-100 corpus token contexts. Per-block coherent counts: [27,23,24,27,21,19,19,15,17,15,12,10] — **all 12 blocks ≥10, satisfying ≥5 threshold in ≥6 blocks**. Audit PASS but G3 flagged: dominant label is `repeated` (top-100 concentrated on a single high-frequency token id). Strict non-`repeated` coherent count is only 5/12 blocks — would fail ≥6 threshold on stricter reading. Block 11 SV0 (σ=2556, var=24%) looks like unigram-axis reboot (boost = `[, the, -, and, ...]`). T2 (item 246) is the gating causal test. Artifacts: cycle_235_t1_features.pt + decoded.json + examples.md + summary.md per the notebook loader contract.

## Cycle 234

244. **gelu_noneg + first-moment + LoRA r=8 W_out** — INCONCLUSIVE (audit-corrected from CONFIRMED, G1+G6 data leakage)
   - Cycle 228 base (relu(gelu(z)) + μ_l) + LoRA r=8 on c_proj per block (368,640 trainable params), Adam 2000 steps, seed 42. Final PPL = **32.411** (cycle 228 baseline 61.664; -29.25 PPL gain). LoRA-zero init reproduced cycle 228 exactly. **Audit downgrade**: cycle 234 trains LoRA on the SAME 200 wikitext-2 val sequences it then evals on (calib==eval). 368K params >> ~25K eval tokens → in-distribution overfitting, not held-out generalization. The 32.411 below-float PPL is contaminated. Cycles 220/228 used calib==eval safely (no optimizer); cycle 234 inherits that scaffolding and adds Adam, turning eval into training. Recommended: rerun with disjoint splits like cycles 206/223 (`all_ids[:200]` / `[200:300]`).

## Cycle 233

243. **Alternative 6-block subsets — is BLOCKS=[1,2,4,5,7,8] representative?** — INCONCLUSIVE
   - Phase 1 single-seed sweep: [1,3,5,7,8,10]=**60.981** (best, +1.47 vs cycle 221), [1,2,3,4,5,6]=**61.785**, [3,4,5,6,7,8]=**67.826** (structural outlier — consecutive middle blocks +8 PPL worse). Spread 6.845 PPL → INCONCLUSIVE per criterion. No subset cleared the 59.0 phase-2 gate; no 3-seed rerun. Cycle 221's [1,2,4,5,7,8] (59.515) remains the best 6-block selection. **6-block ceiling is structural to the count, but specific subset matters: middle-only is much worse.** Sub-58 PPL at 6 blocks not reachable by reselection alone.

## Cycle 232

242. **Sequential init + joint polish hybrid at 3 blocks** — REFUTED
   - BLOCKS=[1,4,7] K=512, 3 seeds. Phase 1 (cycle 215 sequential, 6000 steps): post_sequential_mean=47.253 ± 0.047 (matches cycle 215 exactly). Phase 2 (joint polish LR=5e-5, 1000 steps): **final_mean=46.911 ± 0.026** ≥ 46.5 refuted threshold. Polish gain: +0.342 PPL (real but insufficient). Δ vs joint-from-lstsq cycle 210 (46.556) = +0.355 PPL WORSE. **Sequential warm-start is strictly worse than fresh joint Adam from lstsq init.** Cycle 215 audit's "sequential opens a better basin" hypothesis killed. 3-block ceiling remains cycle 213 W_out+W_fc joint Adam at 46.424.

## Cycle 231

241. **K=128 Adam correction at 6 blocks — does the flat curve hold at higher block count?** — REFUTED
   - BLOCKS=[1,2,4,5,7,8] K=128 Adam W_out+b_out, 3 seeds (42/137/2024), 4000 steps + best-state restore. Init PPL = **115.056** (+28.4 vs cycle 221 K=512 init 86.686 — lstsq at K=128 much worse at 6-block depth than the plan's ~88 prediction). Trained best_mean = **67.562 ± 0.061** (per-seed: 67.630 / 67.483 / 67.574, best_step 3833 ± 170, all near end of budget). **Δ vs K=512 cycle 221 ceiling (59.515) = +8.047 PPL**, well above 65.0 REFUTED threshold. The K=128 penalty goes from +2.43 PPL at 3 blocks (cycle 223 48.989 vs cycle 206 46.556) to +8.05 PPL at 6 blocks — cascade depth triples the compression cost. Cycle 230's "no compression cliff in [1,512] at 3 blocks" does NOT generalise; at 6 blocks the K-vs-PPL curve steepens. Combined with cycle 221 (joint 6-block 59.515) and cycle 229 (sequential 6-block 60.071), the 6-block W_out+b_out surface has a structural ~59.5 PPL floor that neither optimizer change NOR K reduction breaks. K=128 is NOT a cheaper 6-block substitute for K=512.

## Cycle 230

240. **K∈{8,4,2,1} Adam compression cliff sweep — find the absolute floor** — CONFIRMED (interesting=true)
   - BLOCKS=[1,4,7], 3 seeds. K=8: 56.323 ± 0.011, **K=4: 57.328 ± 0.026** (≤59.28 confirmed), K=2: 58.289 ± 0.035, **K=1: 59.245 ± 0.029** (still within +10% of float). Full curve {512..1}: {46.556, 47.509, 48.989, 50.781, 53.038, 55.133, 56.323, 57.328, 58.289, 59.245} — strictly monotone, ~1.0-1.2 PPL per halving in sub-16 regime. **NO compression cliff anywhere in [1, 512].** K=1 = 4,608 trainable params total (1× channel + 768-dim W_out per block × 3), still within +10% of float. Compressed MLP at K=1 = `scalar_gate(x)·v + c` with v,c∈R^768 free — sufficient additive residual update for 3 blocks. Notably K=1 3-block (59.245) ≈ K=512 6-block (59.515 cycle 221) — both regimes hit the same STRUCTURAL cascade-depth floor. Channel capacity is NOT the 3-block bottleneck.

## Cycle 229

239. **Sequential vs joint Adam at 6 blocks** — REFUTED
   - BLOCKS=[1,2,4,5,7,8] K=512, sequential per-block Adam (2000 steps each, best-state restore), 3 seeds. **final_mean = 60.071 ± 0.086** ≥ 60.0 threshold (barely). Per-seed: 59.949 / 60.144 / 60.119. Joint baseline (cycle 221) = 59.515 → **sequential is +0.556 PPL worse**. Block trajectory: 86.7 → 66.3 → 62.8 → 61.4 → 60.7 → 60.1 → 60.1 (block 1 removes 20.4/26.6 PPL = 59% of total; blocks 7+8 add 0.63 — severe diminishing returns). Monotone, zero collapses, restore sanity clean. **The 58.0 floor for 6-block W_out+b_out is structural, not optimizer-limited.** Joint Adam is consistently slightly better than sequential at both 3 blocks (cycle 215) and 6 blocks (this).

## Cycle 228

238. **gelu_noneg + per-channel first-moment correction** — INCONCLUSIVE (interesting=true)
   - Precomputed μ_l = E[gelu(z) − relu(gelu(z))] per (block, channel) on calibration set, patched all 12 blocks with relu(gelu(z)) + μ_l. **corrected_noneg_ppl = 61.664** vs uncorrected cycle 220 = 2093.296 (~34× recovery, ~99.2% of the absolute cycle 220 gap), still +16.6 PPL above baseline 45.064 (+36.84%). μ ≤ 0 elementwise; per-block mean(μ) -0.070 to -0.095 (~0.5× cycle 184's −0.17). Residual 16.6 PPL is higher-order beyond per-channel mean. Same family as cycles 188 (C3 +20.3%), 190 (hybrid +27.7%), strictly better than 184 (ReLU+bias +87.7%). First moment is necessary but not sufficient.

## Cycle 227

237. **Δ̃ specificity control vs non-adjacent c_fc(j)** — INCONCLUSIVE (interesting=true)
   - For each l ∈ {0..10}, top-20 right SVs of centred Δ̃_l vs top-20 left SVs of c_fc(j) for ALL j > l. **median_specificity = 0.791** (max@l+1 − mean over j>l+1, top-5 cos_sum) ∈ [0.3, 1.0) → INCONCLUSIVE. **0/10 rows have quiet non-adjacent entries** — ALL 55/55 non-adjacent (l, j) pairs clear 3σ random threshold. Sanity: adjacent top-5 mean reproduces cycle 218 exactly (2.9687). **Cycle 218's "Δ̃ targets next block" interpretation is partially walked back**: alignment is broadly distributed across all downstream c_fc reading subspaces with mild (+0.79 cos_sum) adjacency preference, not specifically inter-block targeting.

## Cycle 226

236. **KD distillation rerun with corrected KD normalization** — INCONCLUSIVE
   - Loss fix verified: KD now O(0.5-0.8) vs cycle 224's broken O(30-50). α-sweep (SEED=42, 2000 steps): α=0.0 → 46.430 (best step 1200), α=0.5 → 47.096, α=1.0 → 61.746 (no learning). Pure CE wins. Full run (3 seeds, α=0): best_mean=**46.364 ± 0.047**, reproducing cycle 213 (46.424) within 0.06 PPL. **KD provides no benefit at matched budget on this trainable surface.** INCONCLUSIVE per criterion (46.364 ∈ (44, 47]); KD direction effectively closed for this setting.

## Cycle 225

235. **K=64/32/16 Adam compression cliff sweep** — CONFIRMED (interesting=true)
   - BLOCKS=[1,4,7], 3 seeds. **K=64 best_mean = 50.781 ± 0.021** (-3.11 PPL vs float). **K=32 = 53.038 ± 0.009** (-0.86 PPL vs float, still beats it). K=16 = 55.133 ± 0.010 (+2.30% above float — the cliff). Extended efficiency curve {512, 256, 128, 64, 32, 16} → {46.556, 47.509, 48.989, 50.781, 53.038, 55.133}. Monotone, ~1-2 PPL per halving. **Cliff sits between K=32 and K=16.** K=32 = ~1% of intermediate width (32/3072), only 25K trainable params per block (75K total) → still beats float. Channel capacity is not the bottleneck even at 32× compression.

## Cycle 224

229. **Distillation baseline (KD+CE)** — INCONCLUSIVE (audit-corrected from REFUTED, G6 method failure)
   - **best_mean=60.938 ± 0.103** (init 61.746) — barely moves. **Method failure**: `F.kl_div(..., reduction='batchmean')` on (B=1, T=128, V=50257) divides only by B, returning the SUM over T·V ≈ 6.4M entries (~30-50), while CE averages over tokens (~4). Effective 50/50 mix is ~10× KD-weighted, swamping CE. The same trainable surface (W_out+W_fc on blocks [1,4,7]) reached 46.424 under pure CE in cycle 213 — capacity is fine; the loss formulation is broken. Audit downgrade per G6. Fix: reshape `(B*T, V)` before kl_div, or use `reduction='mean'`, or weight by 1/(T·V); sweep KD/CE ratios on a short budget before recommitting 10k-step distillation.

## Cycle 223

228. **Lower K with Adam correction — compression efficiency curve** — CONFIRMED (interesting=true)
   - BLOCKS=[1,4,7], 3 seeds. **K=256 best_mean = 47.509 ± 0.023** (≤52.0 confirmed; -11.85% vs float). K=128 best_mean = 48.989 ± 0.033 (-9.10% vs float). K=512 (cycle 210) = 46.556 (-13.62%). Compression efficiency curve is **monotone but very flat**: 4× W_out compression (512→128) costs only +2.43 PPL. All three K beat float. **Channel capacity is NOT the bottleneck for low-K compression — Adam W_out fitting is.** Calib/eval disjoint (`all_ids[:200]` / `all_ids[200:300]`); no leakage. Best-state snapshot/restore round-trips ≤1e-3. 297K params (K=128) is enough to clear float.

## Cycle 222

227. **Frobenius-weighted channel selection — cascade init PPL** — REFUTED
   - Frobenius-importance scoring (||W_proj[c]||² × Σ gelu²(z_c[t])) swaps 66/52/29 channels per block (1/4/7) vs standard topk but improves 3-block init PPL by only **0.111** (61.7464 → 61.6355). Below 0.3 threshold → refuted. **Extends cycle 207's single-block finding to the cascade**: lstsq W_out absorbs whichever K=512 channels are selected. Channel selection direction is closed for K=512.

## Cycle 221

226. **Adam 6-block correction with proper early stopping** — INCONCLUSIVE
   - 3-seed (42, 137, 2024) formal early-stop rerun of cycle 211. best_ppl_mean=**59.515 ± 0.094** at best_step_mean=1367 ± 170. -0.165 PPL vs cycle 211's 59.680 single-seed. Overfitting reproduced (final@4000 = 71.054). Confirms 6-block ceiling at ~59.5 PPL — 1.5 PPL above the 58.0 Confirmed threshold. **13 PPL cascade gap**: 3-block joint Adam (46.556) → 6-block joint Adam (59.5). Sequential Adam unlikely to help (cycle 215 showed sequential ≈ joint at 3 blocks).

## Cycle 220

234. **Surgical bump removal — GELU_noneg vs GELU_nobump** — INCONCLUSIVE (interesting=true)
   - All-12 patch on GPT-2 small. baseline=45.064 PPL (in-cycle, wikitext-2 val N=200). gelu_noneg = max(0, GELU(z)) → **2093 PPL (+4545%)**. gelu_nobump = piecewise-linear (z>2: id; z<-2: 0; mid: (z+2)/2) → **11456 PPL (+25321%)**. Both variants destroy the model catastrophically; nobump ~5.5× worse than noneg. Sanity-after-restore matches baseline exactly. Verdict literal-INCONCLUSIVE (neither variant is below low_thresh=49.57; |delta|=9362 PPL ≠ ≤3 PPL). **Real finding**: GELU's negative leak and positive-zone bump are BOTH individually necessary; "decomposable GELU" hypothesis is wrong. Audit (G4): the gelu_noneg result is ~46× harsher than cycle 184's ReLU+bias (+88%) because gelu_noneg has NO mean/bias correction — the negative leak's mean shift cascades through 12 blocks. Recommended: rerun with per-channel first-moment correction to test "near-linearizable" claim fairly, OR reframe hypothesis to "double necessity + cascade fragility".

## Cycle 219

233. **Anti-podal channel twin census** — INCONCLUSIVE (PLAN INVALID)
   - Dependency 230 (cycle 216) found NO blocks with ≥100 anti-podal pairs (total ~31 across all 12 blocks). Confirmed criterion unmeasurable. No experiment run; commit=NONE.

## Cycle 218

232. **Where does Δ point? — centred Δ̃ alignment with downstream readers** — CONFIRMED
   - Centred Δ̃_l = Δ_l − mean(Δ_l) per block, top-20 right SVs vs block l+1 reading subspaces (c_fc, attention Q/K/V, residual PCs). 11/11 cross-block pairs pass mean+3σ random baseline. **c_fc cos_sums 4.42-7.07 (+17 to +33σ above random 2.72±0.10)**, **resPC cos_sums 4.77-8.38 (+20 to +33σ)** — both reader subspaces are robustly aligned with the bump's per-token signal in EVERY block. **V-reader anti-aligned by -10 to -13σ in deep blocks 8-10** (surprising negative finding). Centred rank_95 jumps from raw cycle 217 median=1 to 552 (range 489-567) — the per-token signal is essentially full-rank in R^768. The alignment is therefore "high-rank Δ̃ leaks into the right places" rather than "low-rank message in specific directions". Audit (G3): missing specificity control — Δ̃_l vs non-adjacent c_fc(j) NOT tested.

## Cycle 217

231. **Per-token Δ(x) = GELU_FFN − ReLU_FFN — rank, subspace, logit impact** — INCONCLUSIVE *(audit-corrected from CONFIRMED)*
   - Literal rank clause met (rank_95 ≤ 64 in 9/12 blocks; per-block ranks [10,132,1,1,1,1,1,8,73,110,1,1]) BUT semantic clause failed: TI/TD ratio 0.87-0.99 shows Δ is dominated by a per-block token-independent constant bias. SV0 is essentially mean(Δ) projected to unigram-frequency tokens (commas, articles, tokenizer artefacts). σ₁/σ₀ ≈ 0.03-0.09 — the per-token component the hypothesis tested lives in SV1+ and is much smaller. Cross-block Δ_l top-3 → c_fc_{l+1} top-20 left SV alignment: 0.369 vs random 0.168 (vs cycle 216 W_down→c_fc 0.649). **Connects cycles 188 (C3 mean-replacement +20.33% PPL) and 191 (constant residual correction worsens PPL by 12)**: same object — constant Δ component absorbable alone, but GPT-2 needs the small SV1+ residual that this measurement did not isolate.
   - Recommended next: SVD CENTRED Δ̃ = Δ − mean(Δ) per block (item 232 should be rewritten to do this).

## Cycle 216

230. **W_down column geometry census** — REFUTED
   - Anti-podal mirror-channel hypothesis killed: max anti-podal fraction (cos<-0.9) in blocks 1-5 is 0.26%; max across all 12 blocks 1.17% (block 0); random R^768 baseline = 0. Within-cluster cos 0.03-0.07 — W_down columns are near-random on the sphere with no k=32 cluster structure. **Notable secondary positive finding (interesting=true)**: cross-block alignment between block l's W_down writing subspace (top-20 right SVs of c_proj) and block l+1's c_fc reading subspace (top-20 left SVs of c_fc) has top-5 principal cosines averaging 0.649 (peak 0.778 at l=4→5). Adjacent blocks DO share writing/reading directions, but not via columnwise anti-podal pairs.

## Cycle 215

225. **Iterative block-by-block Adam W_out correction** — INCONCLUSIVE
   - Sequential per-block Adam (BLOCKS=[1,4,7] K=512, 2000 steps each, best-state restore per block, 3 seeds): final_ppl 47.253 ± 0.047 (init 61.746 → after b1 51.486 → after b4 48.503 → after b7 47.253). Monotone no collapse. 0.7 PPL worse than joint Adam best 46.556 (cycle 210). Per-block best-state restoration freezes each block against downstream blocks' eventual trajectories — joint optimization wins. Best-step at 2000 for nearly every block-seed (under-trained at budget cap).

## Cycle 214

224. **Analytical W_out correction** — REFUTED
   - Analytical PPL=61.767 ≈ init 61.746 (no improvement). Distribution mismatch: compressed-context activations ≠ float-context activations. Cascade correction requires CE gradient (Adam); analytical lstsq cannot replace it.

## Cycle 213

223. **Adam W_out+W_fc correction** — INCONCLUSIVE
   - W_out+W_fc best=46.424 vs W_out-only best=46.556 (delta=0.13 PPL, negligible). Converges faster (1500 vs 4000 steps) to same quality. W_fc training adds no meaningful advantage; W_out-only Adam is preferred.

## Cycle 212

222. **10-block Adam correction** — REFUTED
   - Best PPL 98.493 (+82.7% vs float). Adam scaling limit: 3-block below float, 6-block +10.7% (early stop), 10-block +82.7% (REFUTED). W_out-only correction cannot fix the 10-block cascade. Practical limit ~6 blocks.

## Cycle 211

221. **6-block Adam correction** — INCONCLUSIVE
   - Best PPL=59.680 at step 1500 (criterion ≤60.0 met), then overfitting to 72.250 at step 4000. More trainable params → faster overfitting: 6-block (2.36M) saturates at ~1500 steps vs 3-block (1.18M) at ~4000 steps. Early stopping mandatory.

## Cycle 210

220. **Adam W_out step-count curve** — INCONCLUSIVE
   - Plateau between steps 2000-4000 (delta=0.25 PPL). Overfitting at step 8000 (+4.67 PPL). Best: step 4000 = PPL 46.556 (13.6% below float 53.895). Practical: early stop at ~4000 steps for N_CALIB=200.

## Cycle 209

212. **Rank-1 input FFN compression** — REFUTED
   - Block 4 MLP input rank_90=395 (not rank-1). LN2 per-token normalization spreads residual stream into ~395D effective variation. Rank-1 K=512 PPL=58.207 vs standard K=512 PPL=55.715 (gap=+2.49). Key correction: residual stream rank-1 ≠ MLP input rank-1 after LayerNorm. Rank-1 model saturates at K≥256.

## Cycle 208

219. **PCA-conditioned lstsq for block 1** — REFUTED
   - All PCA variants worse than standard (54.770–55.074 vs 54.689). PCA adds approximation error without reducing lstsq noise. lstsq handles near-zero-variance input dimensions implicitly through numerical conditioning.

## Cycle 207

218. **Block 1 channel selection scoring** — REFUTED
   - All 4 criteria (act_std×W_proj_norm, mean_abs×W_proj_norm, pure act_std, pure weight) within 1 PPL (range 54.629–55.679). Best improvement: 0.06 PPL. Despite orderings differing (min Kendall-tau=−0.13), lstsq W_out adapts to any reasonable 512-channel subset. Selection criterion is irrelevant at K=512.

## Cycle 206

217. **Adam W_out correction on BLOCKS=[1,4,7] K=512** — CONFIRMED
   - Adam on W_out/b_out only (1.18M params, frozen fc weights): PPL 61.746→48.111, improvement=13.635 PPL. Final PPL is 10.7% BELOW float (53.895). Cascade barrier is not structural — it is suboptimal W_out init for the 512-channel compressed activation space. 1.18M params and 1000 Adam steps fully dissolve the +14.6% cascade.

## Cycle 205

216. **Block 1 single-block K-sweep** — INCONCLUSIVE
   - Block 1 alone at K=512: +1.5% PPL (54.689 vs float 53.895). The 3-block +14.6% is entirely cascade, not poor single-block quality. Improvement beyond K=512: 0.493 PPL (below 0.5 threshold). K=512 is near-optimal; the bottleneck is cascade amplification (10×), not single-block approximation quality.

## Cycle 204

215. **3-block selection sweep** — REFUTED
   - Best [1,4,10]=61.455, anchor [1,4,7]=61.746 (0.29 PPL gap, within noise). Block 1 dominates: all top-4 include it (input_rank=192, highest-value compressible block). Blocks starting at 2+ consistently worse (+1.8-7.6 PPL). 3-block topk+lstsq ceiling is ~+14-15% above float. P0/P1/P2 queue now empty — agenda refresh needed.

## Cycle 203

214. **Block subset search — supersets of [1,4,7]** — REFUTED
   - No superset achieves within 2 PPL of anchor 61.75. Adding 2 blocks costs +13-40 PPL (cascade). 3 blocks is the practical ceiling for topk+lstsq. Cascade near-symmetric: early/middle/late blocks all +40-46% degradation per extra pair. Each additional block adds ~14-15% PPL.

## Cycle 202

213. **BLOCKS=[1,4,7] compression — exclude block 11** — CONFIRMED
   - Excluding block 11 improves PPL by 6.4 (68.14→61.75; +14.6% vs +26.4% above float). Block 11's rank-expansion (rank-2→231) cannot be approximated by K=512 topk+lstsq from degenerate rank-2 input. New best recipe: BLOCKS=[1,4,7], K=512. Blocks 0 and 11 are both bookend blocks that must not be compressed.

## Cycle 201

211. **Block 11 geometry deep-dive** — CONFIRMED
   - Block 11 is an output decompressor (mirror of block 0): in_rank=2→out_rank=231 (154× expansion). Delta rank=1 (rank-1 perturbation, 97% energy in top-3 SVs). Token-type delta concentration: top-20 mean 848.7 vs overall 335.7 (2.5×). Block 11 expands rank-1/2 residual stream for LM head projection. Borderline droppability (PPL 41.4, cycle 187) explained: LM head receives rank-2 instead of rank-231. Self-correction in Frobenius divergence (cycles 194/200) is block 11's rank-expansion absorbing upstream errors.

## Cycle 200

210. **Residual stream rank trajectory under compression** — REFUTED
   - Float ranks 244→192→23→1; compressed ranks 244→192→20→1 (max ratio=1.0). Rank-1 collapse is robustly preserved. Cascade barrier is PURELY from Frobenius content error, not rank disruption. Frob div peaks at blocks 8-9 (~0.24) then decreases (consistent with cycle 194 passive self-correction). Closes the "rank disruption" hypothesis.

## Cycle 199

209. **K-sweep for near-rank-1-input blocks** — REFUTED
   - Degradation vs float: +57% at K=8, +47% at K=32, +26% at K=512. No plateau. Rank-1 residual stream (blocks 4,7,11 from cycle 198) does NOT enable smaller K. After LayerNorm, each token has distinct 768D variation regardless of residual rank. Block input rank ≠ MLP input diversity after LN.

## Cycle 198

205. **Block 0 criticality analysis — what makes block 0 unique** — CONFIRMED
   - Block 0 input_rank_90=269 vs median_others=1 (269×). Raw token embeddings span 269 dimensions; later blocks operate on near-rank-1 residual stream. Input rank collapse: b0=269→b1=207→b2=25→b3+=1. Block 0 is irreplaceable because it transforms high-diversity embeddings into the residual stream. Cannot be compressed by the same topk+lstsq approach as blocks 1-11 (which operate on rank-1 input).

## Cycle 197

208. **Block-pair joint compression via sequential progressive fitting** — INCONCLUSIVE
   - Sequential fitting (each block on compressed predecessor's I/O): 29.954 PPL vs all-float 30.089 PPL. Improvement 0.135 PPL (noise-level). Distribution mismatch in fitting data is NOT the cascade bottleneck. Cascade is from intrinsic information loss per block (K=512 cannot represent 3072), not from training/inference distribution mismatch.

## Cycle 196

207. **LayerNorm recalibration for stacked compressed blocks** — REFUTED
   - Compressed PPL 30.089 (blocks [1,4,7,11], K=512). LN-2 recal → 30.337 (WORSE by 0.248). Per-channel cross-token statistics irrelevant to LN's per-token normalization — LN normalizes each token independently over 768 channels. Cascade barrier is in residual CONTENT (correlational structure), not distributional shift. LN direction closed.

## Cycle 195

206. **Jacobian effective rank census — data-active rank of GPT-2 FFN blocks** — REFUTED
   - overall_mean_rank_90=472.4/768 (61%). 0/12 blocks below 100/768 low-rank threshold. Blocks 1-11: mean 387-543. Block 0 lower (166.8/768=22%) but not exploitable. Rank gradient: low early → high middle (b9 peaks at 543) → slight drop b11 (459). Stacked ranks 359-642. R3 MASO active-subspace hypothesis REFUTED: GPT-2 FFN blocks are near-full-rank. Consistent with cycles 184/189 (diffuse high-rank error structure).

## Cycle 194

204. **Self-correction depth analysis — blocks 8-10 partial compensation** — REFUTED
   - max_decrease=2/12 layers. Error magnitude generally keeps growing; PPL improvement at k=8-10 is error REDIRECTION to low-PPL subspace (error rotates into less PPL-sensitive dimensions), not cancellation. Self-correction is passive, not active. Cascade barrier cannot be broken by relying on self-correction.

## Cycle 193

203. **Superadditive interaction isolation — C3+Hybrid combined** — INCONCLUSIVE
   - C3+Hybrid all-12 PPL=41.447 (+82.59%) vs ReLU+bias ~42.6 (+88%). Nearly identical. Superadditivity is GENERAL — not specific to zero-point. Simple sum prediction +48% vs actual +83%: 35-point excess from combining both-regime modifications.

## Cycle 192

202. **Exact dead-channel pruning baseline — GPT-2 FFN** — REFUTED
   - Near-zero dead channels at any viable threshold (eps=0.01: 0.003%; eps=1.0: 2.64% but PPL +80%). GELU networks don't have exact zeros — Serra/Berzins exact pruning is not applicable. Only 2 always-active channels total. R2 exact pruning mandate is closed for GELU architectures.

## Cycle 191

201. **Mean residual error correction — directional coherence-based bias** — REFUTED
   - Corrected PPL=56.781 vs uncorrected=44.842 (WORSENED by 11.938). Mean error vectors (fraction_of_frob 0.13-0.35) are non-trivial but applying as constant residual shifts hurts downstream layers. Cascade barrier is fundamental: any constant modification to the residual stream compounds through subsequent layers.

## Cycle 190

200. **Positive-regime shape isolation — GELU-neg + ReLU-pos hybrid activation** — INCONCLUSIVE
   - Hybrid all-12 PPL=28.980 (+27.67%, baseline 22.700) — just above the refuted threshold (28). Positive-regime shape change alone (+27.67%) costs nearly as much as C3 negative-regime replacement (+20.33%). Full ReLU+bias swap (+88%) implies ~40% superadditive interaction when both regimes are modified simultaneously.
   - Non-monotone progressive: PPL peaks at k=7 (28.484), decreases to k=10 (27.851), then rises at k=11 (28.980) — some deep-layer self-correction.

## Cycle 189

199. **Cascade error direction analysis — GELU vs ReLU+bias residual stream** — CONFIRMED
   - Mean cos(e_l, e_{l-1})=0.716; all 11/11 inter-layer pairs exceed 0.3. Errors compound directionally: ||e||_F grows 685→11472 (17x over 12 layers). Rank_90 stays high (320-553/768) — errors are diffuse in content but directionally coherent. Block 11 anomaly: largest ||e||_F jump + rank_90 drops to 320 (consistent with cycle 184).
   - Reconciles with cycle 184 (diffuse rank, low-rank correction not viable): both are true. Errors have structured direction despite high rank.

## Cycle 188

198. **Progressive C3 composition — per-channel mean replacement across all 12 blocks** — INCONCLUSIVE
   - All-12 C3: PPL=27.314 (+20.33%), between +15% and +40% thresholds. Composes much better than ReLU+bias (~42 PPL). Barrier materializes after block 6-7.
   - Shape change accounts for ~14.3 PPL of the 18.9-PPL cycle-181 cascade gap; constant replacement accounts for ~4.6 PPL.

## Cycle 187

196. **Layer-drop baseline — GPT-2 FFN skip blocks [1,4,7,11]** — CONFIRMED
   - 10/12 blocks individually near-redundant (PPL <40 when dropped; damage +1.5–4.3). Block 0 critical (1923 PPL). Block 11 borderline (41.4 PPL).
   - Multi-block [1,4,7,11] drop: 53 PPL (+18). Cascade interactions produce nonlinear superadditivity even for near-free individual drops — confirms cascade barrier is structural, not due to individual block importance.

## Cycle 186

195. **Fewer-block compression — GPT-2 FFN k=4 on 2 blocks vs 4 blocks** — REFUTED
   - Best 2-block pair (1,4): mean PPL 50.4 ±5.3 vs 4-block 59.4 ±8.2. Halving block count gains 9 PPL but stays above threshold 48.
   - All 6 pairs tightly clustered (50.4–54.8); block pair (1,4) consistently best.
   - High cross-seed variance driven by seed-42 teacher PPL 57.6 (vs 48.8/50.4 for other seeds).

## Cycle 185

194. **Compression ratio sweep k={2,4,6,8} — GPT-2 FFN** — REFUTED
   - PPL floor is ratio-independent: k=2→61.56, k=4→59.42, k=6→58.30, k=8→57.16 (only 4.4 PPL for 4x more params, 12x→3x compression). k=8 PPL=57.2 > 55 threshold.
   - Per-PPL gain from doubling params is ~1.1 PPL/step (k=4→6→8), suggesting composition-driven floor not capacity-driven.
   - Cascade/composition barrier confirmed again: increasing channel count does not break the stacking floor.

## Cycle 184

197. **Residual stream trajectory divergence: GELU vs ReLU+bias — GPT-2** — REFUTED
   - Divergence is exponential (R²=0.972) and α-smooth (monotone PPL 57.7→38.1 as α: ReLU→GELU), but DIFFUSE (rank_90=300-434 out of 768). Low-rank correction not viable.
   - Transition-zone fraction identical (~0.49) for most- and least-divergent tokens — activation |z|<1 does NOT predict which tokens diverge.
   - Block 11 anomaly: rank_90 drops from 434→215 at final block (possible self-correction).

## Session 125-184 (59 cycles this session)

### Key Findings
1. **GELU negative regime = constant bias** (173, 174): absorbable; token-dependent signal is noise
2. **Single-block ReLU+bias works, full-model fails** (173, 181): per-block +6%, cascade +88%
3. **z negativity is learned, scales with model size** (165, 168)
4. **Block count is primary cascade cost** (171): 12@3x=183 vs 4@6x=59 PPL
5. **Topk+lstsq+corrections remains best**: PPL ~59 at 1.6M + 49K params
6. **Distillation control confirmed** (183): topk beats local Adam+MSE by +21 PPL; single-block quality ≠ stacking quality
7. **Divergence is exponential but diffuse** (184): rank_90=300-434/768, low-rank correction not viable; α-smooth interpolation confirms shape-mismatch as continuous, not phase-transition
8. **Block count matters, floor persists** (185, 186): ratio-independent compression floor ~57 PPL; 2-block gains 9 PPL over 4-block but threshold unbreached; cascade barrier is structural
9. **Individual blocks near-redundant, cascade is not** (187): blocks 1-10 all droppable (PPL <40); multi-block drop 53 PPL despite single drops being 36-39. Superadditivity is the core problem
10. **C3 composes better than ReLU, still breaks after block 7** (188): +20.33% all-12 vs +88% for ReLU+bias. Shape change is the larger culprit (~14.3 PPL); constant replacement alone causes ~4.6 PPL
11. **Cascade errors are directionally coherent despite high rank** (189): mean cos(e_l,e_{l-1})=0.716, 11/11 layers. Errors are diffuse in content (rank_90=300-550) but aligned in direction. 17x Frobenius amplification over 12 layers. Not low-rank correctable (cycle 184), but directionally structured.
12. **Both activation error types cost equally; interaction is superadditive** (190): positive-regime shape change alone +27.67%, negative-regime constant replacement alone (C3) +20.33%. Full modification (ReLU+bias) +88% — ~40% extra from superadditive interaction between both types simultaneously. Neither type dominates.
13. **Residual stream corrections are counterproductive without downstream re-calibration** (191): first-moment correction worsens PPL by 12 PPL. Cascade barrier fundamental — any modification to residual stream at depth l compounds through subsequent layers.
14. **Exact dead-channel pruning not applicable to GELU networks** (192): GPT-2 has near-zero dead channels (0.003% at eps=0.01). GELU never reaches exactly zero. Serra/Berzins exact pruning requires ReLU hard zeros. R2 baseline is closed.
15. **Superadditivity is general — not specific to zero-point** (193): C3+Hybrid (non-zero neg replacement) PPL=41.4 ≈ ReLU+bias PPL=42.6. Combining both-regime modifications produces ~35% extra damage beyond simple sum, regardless of the specific form of negative-regime replacement.
16. **Self-correction at k=8-10 is passive error rotation, not cancellation** (194): max 2/12 layers show magnitude decrease. PPL improves because errors rotate into low-PPL subspace (consistent with diffuse high-rank structure from cycle 184). Self-correction is a statistical artifact, not exploitable.
17. **FFN Jacobians are near-full-rank — no active subspace** (195): overall_mean_rank_90=472.4/768 (61%). 0/12 blocks low-rank (<100). Rank gradient: b0=166 → b9=543 → b11=459. Stacked ranks 359-642. R3 MASO active-subspace hypothesis fully refuted. Consistent with cycles 184/189 (diffuse errors). GPT-2 FFN functions are globally distributed across nearly all 768 dimensions.
18. **LN recalibration is ineffective** (196): LN-2 recal worsens PPL by 0.248 (30.089→30.337). Per-channel cross-token statistics don't match what per-token LN normalization needs. Cascade error lives in normalized residual content, not distributional shift.
19. **Sequential fitting provides negligible benefit** (197): fitting each block on compressed predecessors' I/O gains 0.135 PPL (noise-level). Distribution mismatch in fitting data is NOT the cascade bottleneck — cascade is from intrinsic information loss per compressed block.
20. **Block 0 criticality is from high input diversity, not specialization** (198): block 0 input_rank_90=269 vs median=1 for blocks 2-11. Input rank collapse: 269→207→25→1 across blocks 0-3. Block 0 must transform high-dimensional raw embeddings (50257-type vocabulary) into the residual stream. Cannot be dropped or replaced by later-block compression approach.
21. **Rank-1 residual stream does NOT reduce required K** (199): K-sweep for BLOCKS=[1,4,7,11] shows monotone improvement +57%→+26% above float (K=8→512) with no plateau. LayerNorm creates token-specific 768D variation after per-token centering/scaling; block input rank ≠ MLP input diversity. K=512 remains necessary baseline.
27. **Block 1 single-block quality is excellent; cascade is the bottleneck** (205): block 1 alone K=512 costs only +1.5% PPL (54.689 vs float 53.895). 3-block [1,4,7] costs +14.6% — gap of 13.1% is entirely cascade amplification, not block 1 compression error. K=512 is near-optimal (improvement to K=1536 is only 0.5 PPL). Cascade must be addressed directly.
26. **[1,4,7] is effectively optimal among 3-block selections** (204): best [1,4,10]=61.455 beats anchor 61.746 by only 0.29 PPL (noise). All top-4 include block 1 (input_rank=192). Blocks 2+ starts consistently worse. 3-block topk+lstsq ceiling ~+14-15% above float is confirmed optimal for current approach.
25. **3 blocks is the practical ceiling for topk+lstsq compression** (203): supersets of [1,4,7] all give PPL ≥75.65 vs anchor 61.75. Adding 2 extra blocks costs +13-40 PPL regardless of position (early/middle/late all similar). Cascade is structural and symmetric across block positions. The improvement must come from a better single-block approximator or a fundamentally different approach.
24. **Excluding block 11 improves compression by 6.4 PPL** (202): BLOCKS=[1,4,7] K=512 → +14.6% above float (61.75) vs BLOCKS=[1,4,7,11] → +26.4% (68.14). Block 11 rank-expansion (rank-2→231) cannot be approximated by topk+lstsq. New best recipe: BLOCKS=[1,4,7], K=512. Both block 0 and block 11 are bookend blocks that must not be compressed.
23. **Block 11 is an output decompressor** (201): in_rank=2→out_rank=231 (154×). Delta_rank=1. Mirror of block 0 (which collapses high-rank input into residual). Blocks 1-10 process rank-1 residual; block 11 expands it to rank-231 for LM head. Self-correction in cycles 194/200 explained: block 11 rank-expansion absorbs upstream errors. Recommendation: exclude block 11 from compression set.
22. **Rank collapse is robustly preserved by topk+lstsq** (200): Float ranks 244→192→23→1; compressed ranks 244→192→20→1. Max rank ratio=1.0. Cascade barrier is from Frobenius content error, not rank structure disruption. Frob div grows 0→0.24 (blocks 0-9) then decreases 0.17 (blocks 10-11), consistent with cycles 189 (directional coherence) and 194 (passive self-correction).

### Working Recipes (ranked)
1. **topk+lstsq + corrections** -> PPL ~59 (49K correction params, total ~1.6M)
2. **topk+lstsq+polish** -> PPL ~70.7 (1.58M params)

### Exhausted Directions
Channel selection, joint training (5 refutations), dead channel pruning (5 refutations), block-adaptive ratio, correction transfer, lstsq targets, sign-pattern partitioning, SoLA split, negative absorption, spectral selection, FLAP bias, wider-shallower, full-model ReLU swap, iterative bias recomputation, low-rank residual correction, compression ratio sweep (ratio-independent floor), fewer-block compression (cascade barrier persists at 2 blocks), Jacobian active subspace compression (R3 MASO hypothesis refuted — near-full-rank Jacobians), LN recalibration (per-channel stats irrelevant to per-token LN)

## Next
- Item 217: Adam correction on W_out of compressed MLPs (P0) — key experiment
- Item 218: Block 1 channel selection quality (P1)
- Item 219: PCA-conditioned lstsq for block 1 (P2)
- Item 218: Block 1 channel selection quality analysis (P1)
- Item 219: PCA-conditioned lstsq for block 1 (P2)

### Working Recipes (updated)
1. **BLOCKS=[1,4,7] K=512** → +14.6% above float PPL (BEST — held-out eval)
2. **BLOCKS=[1,4,7,11] K=512** → +26.4% above float PPL
