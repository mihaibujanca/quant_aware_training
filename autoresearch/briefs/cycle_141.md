# Item 153: Graduated Importance Pruning — GPT-2 FFN

## Item
Cycle 128 found 0 strict-removable channels but 14 with >95% dead zone activity
in blocks 0-4. Instead of binary dead/alive, test graduated importance pruning:
rank all 3072 intermediate channels by importance = ||W_down[:,j]||_2 * rms(GELU(z_j))
over 2048 probe tokens. Prune bottom 5%/10%/20%, refit W_down via lstsq on
remaining channels. Compare to topk at matched k (2918/2765/2458). Test on
blocks [1, 4, 7, 11]. 3 seeds, 200 OWT eval sequences.

- **Confirmed** if importance pruning beats topk by >1% PPL at any pruning level
  (pre-polish), across 3 seeds.
- **Refuted** if topk matches or beats importance at all levels (pre- AND post-polish).
- **Inconclusive** if pre-polish advantage exists but post-polish washes it out
  (replicating cycle 133 pattern).

## Context
- **Cycle 128** (item 134): dead channel census. 0 strict-removable, 14 soft95-dead
  in blocks 0-4. Binary pruning is not viable; graduated approach needed.
- **Cycle 133** (item 127): hybrid regression REFUTED. Polish erases pre-polish
  method differences. topk+lstsq post-polish = 70.12, hybrid = 70.49.
  Lesson: must report BOTH pre-polish and post-polish numbers.
- **Cycle 129** (item 100): isotropy learned. Channel importance may be more
  structured than topk's activation-magnitude selection assumes.
- **Working recipe**: topk k=4 (512/3072 channels) + lstsq + 500 Adam -> PPL ~70.

## Dependencies
- Cycle 128 (soft): census provides motivation. No code dependency.
- `aleph/lm_geometric_compression.py`: derive_ffn_compression, swap_compressed_ffns.
- Float PPL baseline: 57.65 (cycle 115 manifest).

## Key Files
- `aleph/lm_geometric_compression.py` — compression infrastructure
- `experiments/dead_neuron_census_gpt2_cycle128.py` — census reference (importance
  metric computation pattern)

## Constraints
- CPU-only, ~15-20 min wall time. Analysis + lstsq refit, no training.
- 500-step Adam polish budget matches working recipe.
- Must measure PPL at two checkpoints: pre-polish (after lstsq) and post-polish.
