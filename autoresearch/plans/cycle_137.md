# Cycle 137 Plan: Bottleneck AE Param-Matched Comparison (Item 151)

## Task
Does AE's advantage over topk+lstsq survive when (a) params are matched, and
(b) 500-step Adam polish is applied? 4 blocks x 3 methods x 3 seeds x pre/post polish.

## Hypothesis
Cycle 131's 2-3.4% AE advantage came from 2.5x more params, not architecture.
At matched params, topk+lstsq will equal or beat AE post-polish.

## Pre-flight
- P1 (duplicate): no prior cycle ran item 151.
- P2 (deps): cycle 131 completed (INCONCLUSIVE). No blockers.
- P3 (scope): 12 method-block combos x 3 seeds = 36 training runs. Tight but single-cycle.
- P4 (null baseline): uncorrected single-block PPL at k=512 (6x) from working recipe.
  Float PPL must be ~57.6. Abort if >60.
- P5 (param matching): topk+lstsq at k=512: W_compressed = 512*768 = 393,216 params.
  Param-matched AE: d_bn = 64 gives 3072*64 + 64 + 64*3072 + 3072 = 396,352.
  Close enough (<1% overshoot). Print and verify at runtime.
  Full AE: d_bn = 512 gives 3072*512 + 512 + 512*3072 + 3072 = 3,149,312.

## Methods
1. **topk+lstsq** (baseline): derive_ffn_compression at k=512. 393K params.
   Polish: 500 Adam steps, lr=1e-4, CE loss.
2. **full AE** (d_bn=512): E(h) = ReLU(W_enc @ h + b_enc), D(z) = W_dec @ z + b_dec.
   Train 1000 steps, lr=1e-4, Adam, CE loss. Then 500 polish steps.
   ~3.15M params. This is the "was the advantage real at all?" control.
3. **param-matched AE** (d_bn=64): same architecture, d_bn=64. ~396K params.
   Train 1000 steps + 500 polish. This is the critical test.

Polish for ALL methods: 500 Adam steps on the compressed params (topk: W_compressed;
AE: W_enc, b_enc, W_dec, b_dec). Same lr, optimizer, loss for all.

## Steps
1. Create `experiments/ae_param_match_k512_cycle137.py`.
2. Load GPT-2 + OWT corpus (reuse existing infra).
3. Float baseline: evaluate unmodified GPT-2 PPL (200 seqs, 3 seeds). Must be ~57.6.
4. For each seed in [42, 137, 7], for each block in [1, 4, 7, 11]:
   a. **topk+lstsq**: derive at k=512, eval PPL (pre-polish). Polish 500 steps, eval (post).
   b. **full AE**: train 1000 steps, eval (pre-polish). Polish 500 steps, eval (post).
   c. **param-matched AE**: train 1000 steps, eval (pre-polish). Polish 500 steps, eval (post).
5. Record: PPL (pre/post), param count, training loss curve.
6. Aggregate: mean +/- std across 3 seeds per (block, method, phase).
7. Save JSON + summary table + bar chart (block x method, pre vs post).

## Validation
- Float PPL must be ~57.6 (+/- 1). Abort if violated.
- topk+lstsq single-block PPL must degrade <25% over float. If not, pipeline is broken.
- All AE variants must achieve PPL < 2x float after training. If not, training diverged.
- Param counts: print at init. topk ~393K, param-matched ~396K, full ~3.15M.

## Criteria (verbatim from agenda)
- **Confirmed**: param-matched AE beats topk+lstsq by >=3% relative PPL on >=2/4 blocks, POST-polish.
- **Refuted**: param-matched AE matches or loses to topk+lstsq post-polish.
- **Inconclusive**: full AE beats topk but param-matched doesn't, or advantage is pre-polish only.

## Artifacts
- `experiments/ae_param_match_k512_cycle137.py`
- `autoresearch/results/cycle_137_ae_param_match.json`
- `autoresearch/results/cycle_137_ppl_bar.png`

## Fallbacks
- If 1000 AE training steps are insufficient (loss not converged), extend to 2000.
  This doubles runtime but the verdict requires converged AE.
- If param-matched AE (d_bn=64) is too narrow to learn (degenerate ReLU dead zones),
  test d_bn=96 (~590K, still within 1.5x of topk). Note the param gap in verdict.
