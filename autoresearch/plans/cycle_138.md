# Cycle 138 Plan: AE k=4 Stacking (Item 152)

## Task
Does AE's 3-3.4% single-block advantage (cycle 137) translate to better k=4
stacking behavior? Compress blocks [1, 4, 7, 11] at 6x with AE, stack all 4,
measure full-model PPL. Three conditions: topk baseline, AE stacking, AE+LoRA.

## Hypothesis
AE's advantage compounds: 4 blocks each 3% better locally should yield ~10-12%
stacking improvement (if errors are roughly additive). But cascade interactions
may dominate local quality — cycle 104 showed 70%+ of stacking error is
attention-driven. Null hypothesis: cascade dominates, AE stacking PPL >= 69.

## Pre-flight
- P1 (duplicate): no prior cycle ran item 152.
- P2 (deps): cycle 137 completed (CONFIRMED). No blockers.
- P3 (scope): 3 conditions x 1 seed = 3 runs (~2h total). Tight but single-cycle.
  Escalate to 3 seeds only if AE crosses CONFIRMED threshold.
- P4 (null baseline): topk k=4 stacking PPL ~70 from working recipe. Float ~57.65.
  Abort if float PPL > 60 or topk stacking > 80 (pipeline broken).
- P5 (param accounting): Per-block AE (d_bn=64): 3072*64+64+64*768+3072 = ~249K.
  4 blocks: ~996K. Topk at k=512 per block: 512*768 = 393K, 4 blocks: ~1.57M.
  AE uses fewer params. Print all counts at init.

## Methods

### Condition 1: topk+lstsq (baseline)
Per-block derive_ffn_compression at k=512 for blocks [1, 4, 7, 11].
Joint 500-step Adam polish on all 4 compressed blocks simultaneously.
Expected PPL ~70 (reproduces working recipe).

### Condition 2: AE stacking
Per-block AE training: d_bn=64, 500 steps each, lr=1e-4, CE loss.
Then joint 500-step Adam polish on all 4 AE blocks simultaneously.
Critical test: does local AE advantage survive stacking?

### Condition 3: AE + LoRA corrections
Same AE stacking as condition 2, then add LoRA corrections (r=8) on attention
projections of the 4 compressed blocks. Train corrections 500 steps, lr=1e-4.
Compare to topk+corrections PPL ~59.

### Training details
- Polish and correction training: CE loss, Adam, lr=1e-4, batch_size=4.
- AE per-block training: CE loss on full model with single block swapped.
  This is important: train each AE independently (not jointly) to match the
  single-block results from cycle 137, then stack and joint-polish.
- LoRA: r=8 on Q,K,V,O projections of blocks [1,4,7,11]. ~49K params total.

## Steps
1. Create `experiments/ae_stacking_k4_cycle138.py`.
2. Load GPT-2 + OWT corpus (reuse existing infra). Verify float PPL ~57.65.
3. **Condition 1 (topk baseline):**
   a. derive_ffn_compression at k=512 for each of blocks [1,4,7,11].
   b. swap_compressed_ffns (all 4 blocks). Eval pre-polish PPL.
   c. Joint polish 500 steps (all 4 c_proj weights). Eval post-polish PPL.
4. **Condition 2 (AE stacking):**
   a. For each block in [1,4,7,11]: swap MLP with BottleneckFFN(d_bn=64),
      train 500 steps (single-block CE loss), record single-block PPL.
      Restore original before next block.
   b. Swap ALL 4 blocks with their trained AE modules. Eval stacked pre-polish PPL.
   c. Joint polish 500 steps (all 4 AE encoder+decoder params). Eval post-polish PPL.
5. **Condition 3 (AE + LoRA):**
   a. Start from condition 2 post-polish model.
   b. Add LoRA (r=8) to attention projections of blocks [1,4,7,11].
   c. Train corrections 500 steps. Eval PPL.
6. Record: PPL at each stage, param counts, training loss curves.
7. Save JSON results + summary table + comparison bar chart.
8. If AE stacking PPL < 67 → escalate: rerun conditions 1-3 with seeds [137, 7].

## Validation
- Float PPL must be 57.65 +/- 1. Abort if violated.
- topk stacking pre-polish PPL should be >65 (compression damage) and <80 (not broken).
- topk stacking post-polish PPL should be ~70 (+/- 3). If >75, pipeline is broken.
- AE single-block PPL should roughly match cycle 137 numbers (within 5% relative).
  If not, AE training diverged — extend to 1000 steps and retry.
- LoRA param count: print and verify ~49K total across 4 blocks.

## Criteria (verbatim from agenda)
- **Confirmed**: AE stacking PPL < 67 (>4% over topk ~70), OR AE+corrections < 57.
- **Refuted**: AE stacking PPL >= 69 (within 2% of topk, cascade dominates).
- **Inconclusive**: AE stacking PPL 67-69 (marginal).

## Artifacts
- `experiments/ae_stacking_k4_cycle138.py`
- `autoresearch/results/cycle_138_ae_stacking.json`
- `autoresearch/results/cycle_138_stacking_bar.png`

## Fallbacks
- If AE per-block training doesn't converge in 500 steps, extend to 1000.
  Cycle 137 used 500 steps successfully, so this is unlikely.
- If LoRA infrastructure doesn't exist for AE blocks, skip condition 3 and
  mark AE+corrections as "deferred." The core question (conditions 1 vs 2)
  is independent of corrections.
- If topk baseline diverges significantly from ~70, use measured topk PPL as
  the comparison point (not the historical ~70). The relative comparison matters.
