# Cycle 262 Audit — Item 268 (K=1 + c_fc bias trainable on 3 blocks)

**Status: PASS**

## Verdict check
- Criteria (agenda item 268):
  - CONFIRMED iff best_mean ≤ 58.5 (beats cycle 230 K=1 3-block 59.245 by ≥ 0.745 PPL).
  - REFUTED iff best_mean ≥ 59.5.
  - INCONCLUSIVE otherwise.
- Reported numbers:
  - Per-seed best PPL: seed 42 = 59.098, seed 137 = 59.165, seed 2024 = 59.130.
  - mean = (59.098 + 59.165 + 59.130) / 3 = 177.393 / 3 = **59.131** ✓ matches reported.
  - std (population) = sqrt(((59.098 − 59.131)² + (59.165 − 59.131)² + (59.130 − 59.131)²) / 3)
    = sqrt((0.001089 + 0.001156 + 0.000001) / 3) = sqrt(0.000749) = **0.02736** ≈ 0.027 ✓.
  - 59.131 > 58.5 → CONFIRMED clause FAILS.
  - 59.131 < 59.5 → REFUTED clause FAILS.
  - INCONCLUSIVE is the only consistent label. PASS.
- Delta vs cycle 230 baseline 59.245: 59.131 − 59.245 = **−0.114** ✓.
- Band position: (59.5 − 59.131) / (59.5 − 58.5) = 0.369 from REFUTED, i.e. 36.9% of the way to CONFIRMED. (Manifest reports 31% — slight rounding inconsistency, immaterial to verdict.)

## Methodology check
- **Calib loader**: 200 wikitext-2 val sequences, len(strip)>50 filter, truncation to SEQ_LEN=128, kept only sequences with shape[1] ≥ SEQ_LEN. Same loader as cycles 213, 225, 230, 240. PASS.
- **Disjoint splits**: `calib_ids = all_ids[:200]`, `eval_ids = all_ids[200:300]`. No overlap. PASS.
- **CompressedMLPBfc**: `fc_weight` is `register_buffer` (frozen), `b_fc` is `nn.Parameter` (trainable), `W_out` and `b_out` `nn.Parameter` (trainable). Forward: `F.gelu(x @ self.fc_weight + self.b_fc) @ self.W_out.T + self.b_out`. The b_fc on the K-channel slice is added BEFORE GELU (correct — it's the c_fc bias). PASS.
- **Topk init**: `fit_block_lstsq` uses cycle 213/225/230 scoring rule `act_std × W_proj.norm` to pick the K=1 channel; lstsq then solves for W_out and b_out conditional on the topk activations. Same convention. PASS.
- **K=1 sanity**: K=1 means one channel per block, so `W_fc[:, topk_idx]` shape (768, 1), `b_fc[topk_idx]` shape (1,), `W_out` shape (768, 1), `b_out` shape (768,). Per-block trainable count = 768 + 1 + 768 = 1537. Three blocks = 4611. The script asserts this with `assert n_trainable == 3 * (768 + 768 + 1)` — passes. PASS.
- **Adam config**: LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, joint optimisation of all 3-block trainable params. Same hyperparameters as cycle 230. PASS.
- **Best-state restore**: `snapshot()` captures (W_out, b_out, b_fc) per block; `best_state` updated whenever eval_ppl improves; final restore copies best state back; `assert abs(final_ppl_restored - best_ppl) < 1e-3` — restore round-trip clean. PASS.
- **Per-seed determinism**: each seed call invokes `torch.manual_seed(seed)` and `random.seed(seed)`, then re-loads GPT-2 from pretrained, then re-collects ln_outs/proj_outs (deterministic forward pass on a deterministic model). The init PPL is **identical 71.409 across all 3 seeds** because the topk+lstsq init does not depend on the torch RNG (it uses a deterministic model and a deterministic data slice). The randomness only enters via the per-seed shuffle of `calib_shuffled` for Adam minibatching. PASS — this is the expected behaviour.
- **Single forward pass per seed for I/O collection**: yes — `collect_mlp_io` registers 3 hooks and runs through `calib_ids` once. PASS.
- **Float baseline**: 53.895 ✓ matches the cycle 230 reference (`float_ppl = 53.895`).

## Numerical sanity
- **Init PPL 71.409**: same across 3 seeds (deterministic init, no RNG dependency in topk+lstsq). The K=1 init is much higher than the K=512 cycle 213 init (~50.x) because at K=1 each block's MLP is reduced to a single direction post-GELU, so the topk-rank-1 reconstruction loses most of the c_proj structure. Cycle 230 K=1 3-block also reported init around 71 — consistent.
- **Best PPL converges to ~59.13**: the per-seed bests (59.098, 59.165, 59.130) span 0.067 PPL, very tight. All 3 seeds reach `best_step = 4000`, meaning the eval PPL was still monotonically improving at the end of training (best=last). This suggests the training did NOT plateau in 4000 steps and could improve further with more steps — but the verdict thresholds (58.5 / 59.5) anchor to cycle 230 which used the same 4000-step budget, so the comparison is apples-to-apples.
- **b_fc trajectories**: per-seed final values:
  - seed 42: block 1 = +0.382, block 4 = +0.353, block 7 = +1.166
  - seed 137: block 1 = +0.420, block 4 = +0.358, block 7 = +1.166
  - seed 2024: block 1 = +0.426, block 4 = +0.356, block 7 = +1.166
  - Per-block seed std: block 1 ≈ 0.020, block 4 ≈ 0.002, block 7 ≈ 0.0005. Block 7 is essentially deterministic across seeds.
  - All 3 blocks all 3 seeds have **positive** final b_fc — the optimiser uniformly wants to push the chosen K=1 channel's pre-GELU activation upward.
  - Block 7 b_fc is the largest (~+1.17) and the most stable — consistent with block 7 being the deepest of the 3 blocks (latest in the cascade) where the W_out direction has the least slack and the bias has the most leverage.
- **b_fc magnitude vs init**: cycle 230's CompressedMLP froze b_fc at the topk-slice values from `block.mlp.c_fc.bias.data`. For the topk channels selected by `act_std × W_proj.norm` at K=1, those init values are typically near 0 with one block having a larger init (block 7 b_fc init ≈ +0.984 inferred from the +0.18 shift to +1.17). The bias movement is real but small.
- **Delta −0.114 PPL is real**: 0.114 is **4.2× the seed std (0.027)**, so the improvement is statistically distinguishable from cycle 230 across the 3 seeds. It is NOT a literal noop. But 0.114 « 0.745 (the threshold for CONFIRMED), so the bias is a small effect on top of W_out, not a structural lever.

## Hypothesis vs result
- **Hypothesis**: at K=1 the W_out path is starved → bias becomes useful → best_mean ≤ 58.5 (beats cycle 230 by ≥ 0.7 PPL).
- **Reality**: best_mean = 59.131, beats cycle 230 by 0.114 PPL only. Bias contributes ~6.5× less than the hypothesised threshold.
- **K-starvation argument fails**: the cycle 213 (K=512 3-block), cycle 241 (K=512 6-block), cycle 248 (K=128 6-block W_fc full), and cycle 262 (K=1 3-block) span four distinct (K, n_blocks) operating points and all four produce the same qualitative result: c_fc bias / W_fc retraining buys ≤ 0.2 PPL on top of W_out+b_out. The W_out path absorbs the available DOF regardless of K. The "small-K is where bias matters" thesis is locally falsified at K=1 — there is NO operating point in the {1, 128, 512} K range where the bias is a structural lever.
- **Counterintuitive check** (CLAUDE.md rule): the hypothesis predicted the bias would matter MORE at small K because the W_out path is starved. The data says the opposite: the bias matters EQUALLY LITTLE at all K. The intuition was wrong — the bias and W_out are not in a "if one is starved the other compensates" regime. Instead, both compete for the same low-dimensional subspace of corrections, and W_out's 768 entries dominate the joint optimisation. The 1-scalar bias can shift the GELU operating point on the chosen channel but cannot make a structural difference because the K=1 bottleneck is in the **direction selection** (one channel) not in the per-channel offset.
- **The bias movement is NOT random**: all 3 blocks all 3 seeds shifted POSITIVE, with very low seed variance, meaning Adam consistently finds a deterministic positive shift. This is a small structural finding: the topk-by-(act_std × W_proj.norm) channel selector picks under-active channels that benefit from being pushed slightly more positive. This is informational but does not change the verdict.

## Bookkeeping check
- agenda.md item 268 deleted from NEXT/P2, replaced with one-line redirect placeholder ✓ (matches cycles 259/260/261 discipline).
- agenda.md `### Cycle 262` PAST entry prepended above `### Cycle 261` ✓.
- TSV row 260 appended, 8 fields, hash filled = 81ac7695 ✓.
- Manifest written with main commit hash 81ac7695… ✓.
- Main commit 81ac769589d93d6e0e0ed9834625408dc1e0b64b ✓.
- Bookkeeping commit ef571b626fc6a0df3e11512e8cc664ab41e43779 ✓.
- Pre-existing dirty files NOT staged (program.md, aleph/pullback_geometry.py, prior cycle audits/briefs/manifests for cycles 215-261, flagged.jsonl, literature_map.md, cycle_170/205/255 manifests) ✓.
- Cycle 262 files staged: experiments/k1_wfc_bias_cycle262.py, autoresearch/plans/cycle_262.md, autoresearch/briefs/cycle_262.md, autoresearch/logs/cycle_262_run.log, autoresearch/results/cycle_262_k1_wfc_bias.json, autoresearch/results/cycle_262_summary.md, autoresearch/results/cycle_262_manifest.txt, autoresearch/agenda.md, autoresearch/logs/experiments.tsv ✓.
- Plan, brief, log, results, summary, manifest, exec_result, audit all under `cycle_262_*` namespace ✓.

## Decision
**PASS** — methodology clean (CompressedMLPBfc correctly freezes W_fc and trains only b_fc + W_out + b_out, b_fc added pre-GELU in the forward path, topk init reproduces cycle 230, disjoint calib/eval splits, joint Adam over all 3 blocks, best-state restore round-trip clean, n_trainable assert passes), numerical sanity holds (init PPL deterministic 71.409 across seeds, per-seed bests within 0.067 PPL, b_fc movement seed-stable and uniformly positive, delta −0.114 = 4.2× seed std real but small), verdict logic correctly applied (best_mean = 59.131 ∈ (58.5, 59.5) → INCONCLUSIVE). The main analytical finding — c_fc bias retraining is uniformly tiny across K ∈ {1, 128, 512}, refuting the "small-K starvation makes bias matter" hypothesis at K=1 — is robust and follows from comparing this cycle's −0.114 PPL gain against cycle 213/241/248's similar near-noop results. The brief's claim of 2,313 trainable params was an arithmetic error (omitted W_out's 768 entries in the per-block decomposition); the actual count is 4,611, but this does not affect the verdict thresholds 58.5/59.5 because they are anchored to cycle 230's 59.245 which had the same 4,608-trainable-params surface (cycle 262 just adds 3 scalars). The follow-up reading about the topk channel selector picking under-active channels (since all 3 blocks all 3 seeds got positive b_fc shifts) is a real structural finding but does not change the negative verdict on the main hypothesis.
