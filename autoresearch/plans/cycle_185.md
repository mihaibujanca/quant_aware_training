# Cycle 185 — Item 194: Compression Ratio Sweep k={2,4,6,8}

## Task

Sweep STUDENT_HIDDEN across k=2 (256), k=4 (512), k=6 (768), k=8 (1024) on GPT-2 FFN blocks [1,4,7,11] using topk+lstsq+500-step Adam polish. 3 seeds. Report stacked PPL, total FFN params, compression ratio.

## Hypothesis

PPL decreases monotonically with k. At k=6 or k=8 the recipe achieves PPL <45, confirming a practical operating point on the compression curve.

## Pre-flight

- **P1 Duplicate**: No prior k-sweep exists. k=4 tested many times but k=2/6/8 never measured with the current recipe.
- **P2 Dependencies**: None. Uses library API directly.
- **P3 Concrete**: 4 k-values x 3 seeds = 12 runs. One output metric (stacked PPL).
- **P5 Null baseline**: Teacher float PPL ~30 (200 eval seqs). k=4 baseline ~59.4 from cycle 183.
- **P6 Model**: GPT-2 pretrained from HuggingFace, no custom checkpoint.

## Criteria (verbatim from agenda)

- Confirmed if: PPL decreases monotonically with k AND k=6 or k=8 achieves PPL <45 (practical operating point exists).
- Refuted if: diminishing returns — k=8 still >55 PPL (compression approach has a floor unrelated to ratio).
- Inconclusive if: non-monotonic or only marginal improvement at higher k.

## Steps

### Step 1: Write `experiments/compression_ratio_sweep_cycle185.py`

Based on cycle 183's topk+lstsq+polish section. Config:

```python
SEEDS = [42, 137, 7]
K_VALUES = [2, 4, 6, 8]          # k groups of 128 channels
HIDDEN_DIMS = {2: 256, 4: 512, 6: 768, 8: 1024}
BLOCKS = [1, 4, 7, 11]
GPT2_FFN_HIDDEN = 3072
GPT2_HIDDEN = 768
SEQ_LEN = 128
EVAL_SEQS = 200
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200
MAX_TOKENS_PER_LAYER = 4096
POLISH_STEPS = 500
LR = 1e-4
BATCH_SIZE = 4
```

For each seed:
1. Load teacher, build probe corpus (same as cycle 183).
2. Evaluate teacher baseline PPL. Sanity: must be in [15, 65].
3. For each k in K_VALUES:
   a. Probe FFN geometry on BLOCKS.
   b. Build topk pack with HIDDEN_DIMS[k].
   c. Swap compressed FFNs (stacked, all 4 blocks).
   d. Polish 500 steps (Adam, c_proj weights only, cross-entropy loss).
   e. Evaluate stacked PPL.
   f. Compute params: per_block = hidden*768 + hidden + 768*hidden + 768; total = 4*per_block.
   g. Compute compression ratio: (4 * original_ffn_params) / total_compressed_params.

Output JSON: `autoresearch/results/cycle_185_compression_ratio_sweep.json`
```json
{
  "per_seed": { "42": { "baseline_ppl": ..., "k2": {"stacked_ppl": ..., "params": ..., "ratio": ...}, ... } },
  "summary": { "k2": {"mean_ppl": ..., "std_ppl": ..., "params": ..., "ratio": ...}, ... },
  "verdict": "CONFIRMED|REFUTED|INCONCLUSIVE",
  "verdict_note": "..."
}
```

Print summary table at end:
```
k  | hidden | params   | ratio | mean_PPL +/- std
2  | 256    | ...      | 12.0x | ...
4  | 512    | ...      | 6.0x  | ...
6  | 768    | ...      | 4.0x  | ...
8  | 1024   | ...      | 3.0x  | ...
```

### Step 2: Run

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/compression_ratio_sweep_cycle185.py
```

Expected runtime: ~20-30 min (4 k-values x 3 seeds = 12 topk+lstsq+polish runs).

### Step 3: Write results manifest

`autoresearch/results/cycle_185_manifest.txt` with verdict, key numbers, artifact paths.

### Step 4: Update agenda.md

Move item 194 from NEXT to PAST under `### Cycle 185` with status and one-line summary.

### Step 5: Commit

```bash
git add experiments/compression_ratio_sweep_cycle185.py autoresearch/results/cycle_185_* autoresearch/plans/cycle_185.md autoresearch/briefs/cycle_185.md autoresearch/agenda.md
git commit -m "[autoresearch] cycle 185: compression ratio sweep k={2,4,6,8} — <VERDICT>"
```

## Validation

- Teacher baseline PPL must be in [15, 65]. Abort if outside.
- k=4 stacked polished PPL should be near 59.4 (cycle 183). Accept 45-80 range (seed variance).
- PPL at k=8 must be <= PPL at k=4 (monotonicity sanity — if violated, flag immediately).
- Compression ratios: k=2 -> ~12x, k=4 -> ~6x, k=6 -> ~4x, k=8 -> ~3x. Verify computed ratios match expectations.

## Artifacts

- `experiments/compression_ratio_sweep_cycle185.py`
- `autoresearch/results/cycle_185_compression_ratio_sweep.json`
- `autoresearch/results/cycle_185_manifest.txt`

## Fallbacks

- If k=8 OOM on MPS (1024-dim hidden + 768 output = ~1.6M params/block, 4 blocks): reduce PROBE_TRAIN_SEQS to 100.
- If one seed diverges wildly (>2x others): report it but base verdict on median, not mean.
