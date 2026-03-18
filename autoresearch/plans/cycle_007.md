# Task
Progressive multi-block FFN accumulation curve on proxy gpt2+Shakespeare at 16x compression.

# Hypothesis
From agenda item 12:
- **Confirmed if**: easiest-first cumulative compression materially outperforms hardest-first and random, e.g. `>=25%` lower perplexity or KL at the same number of swapped blocks.
- **Refuted if**: cumulative degradation is nearly identical across orders, implying that single-block localization is not informative.
- **Inconclusive if**: order helps only at very small block counts and disappears quickly.

# Pre-flight Results
- P1: Not a duplicate — no multi-block accumulation in PAST.
- P2: Depends on item 10 (cycle 005, confirmed). Data in `runs/nanochat_single_block_localization_overnight_main/results.jsonl`.
- P3: Concrete and testable in one cycle.
- P4: No prompt/program modifications needed.

# Steps

1. **Write experiment script** `experiments/progressive_multiblock_accumulation.py`:
   - Load gpt2 teacher + Shakespeare corpus (reuse existing `load_nanochat_teacher`, `build_probe_corpus`)
   - Read cycle 005 results to rank layers by 16x single-block damage (relative ppl increase)
   - Define 3 orderings: easiest-first (ascending damage), hardest-first (descending damage), random (seed=42)
   - For each ordering, iterate k=1..12:
     - Call `run_ffn_geometric_compression(teacher, corpus, budget, layer_ids=tuple(ordering[:k]))` with hf=0.0625
     - Record full-model perplexity, KL divergence, k, ordering name
   - Save results JSON to `autoresearch/results/cycle_007_accumulation.json`
   - Save accumulation plot to `autoresearch/results/cycle_007_accumulation.png`
   - Print summary table

2. **Run the script**: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/progressive_multiblock_accumulation.py`

3. **Evaluate criteria**:
   - Compare perplexity at each k across orderings
   - Compute the >=25% lower perplexity test at k=3,6,9,12
   - Determine confirmed/refuted/inconclusive

4. **Move item 12** from NEXT to PAST in `autoresearch/agenda.md` with outcome and artifacts.

5. **Commit**: `[autoresearch] cycle 007: progressive multi-block accumulation curve`

6. **Write manifest** `autoresearch/results/cycle_007_manifest.txt`

# Validation
- Teacher ppl should be ~122.35 (sanity check)
- k=1 easiest-first should match cycle 005 layer 7 result (~121.6)
- k=12 all orderings should converge to same ppl (all blocks compressed)
- Plot must show 3 distinguishable curves

# Fallbacks
- If full 12-layer sweep exceeds 30 min, subsample to k=1,2,4,6,8,10,12
- If memory issues, reduce probe corpus size (probe_train_seqs=32)

# Artifacts
- `experiments/progressive_multiblock_accumulation.py`
- `autoresearch/results/cycle_007_accumulation.json`
- `autoresearch/results/cycle_007_accumulation.png`
- `autoresearch/results/cycle_007_summary.md`
