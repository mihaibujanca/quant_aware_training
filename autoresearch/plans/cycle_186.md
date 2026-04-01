# Cycle 186 — Item 195: Fewer-block compression — GPT-2 FFN k=4 on 2 blocks vs 4 blocks

## Task

Test whether compressing only 2 FFN blocks (from {1,4,7,11}) with topk+lstsq+500-step Adam polish gives a better PPL/compression tradeoff than compressing all 4 blocks. Run all 6 pairs from {1,4,7,11}: (1,4), (1,7), (1,11), (4,7), (4,11), (7,11). Compare best 2-block PPL vs the 4-block baseline (~59 PPL from cycle 185).

## Hypothesis

Block count is the primary cascade cost driver (cycle 171: 12@3x=183 PPL vs 4@6x=59 PPL). Halving the number of compressed blocks reduces cascade depth and should yield better PPL at half the compression scope.

## Pre-flight

- **P1 Duplicate**: No prior 2-block-pair experiment.
- **P2 Dependencies**: None. Uses library API directly.
- **P3 Concrete**: 3 seeds × (1 four-block baseline + 6 pairs) = 21 evaluations per seed.
- **P5 Null baseline**: Teacher float PPL ~29-58 (seed-dependent). 4-block k=4 baseline ~59.4 mean PPL (cycle 185).
- **P6 Model**: GPT-2 pretrained from HuggingFace, no custom checkpoint.

## Criteria (verbatim from agenda)

- **Confirmed** if: best 2-block pair achieves PPL <42
- **Refuted** if: all 2-block pairs give PPL >48
- **Inconclusive** if: 42-48 PPL range

Verdict applied to best-pair **mean PPL across 3 seeds**.

## Steps

### Step 1: Write `experiments/fewer_blocks_cycle186.py`

Config:
```python
SEEDS = [42, 137, 7]
K = 4
HIDDEN_DIM = 512
ALL_BLOCKS = [1, 4, 7, 11]
BLOCK_PAIRS = list(itertools.combinations([1, 4, 7, 11], 2))  # 6 pairs
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

Per seed:
1. Build probe corpus
2. Evaluate teacher baseline PPL — abort if outside [15, 65]
3. Probe atlas on ALL 4 blocks at once (reuse for all pair runs)
4. Run 4-block baseline: `_derive_topk_pack(atlas, ALL_BLOCKS, HIDDEN_DIM)` → swap → polish → eval
5. For each pair in BLOCK_PAIRS:
   - `_derive_topk_pack(atlas, list(pair), HIDDEN_DIM)` → swap → polish → eval
   - Use `pair_seed = seed + pair[0]*100 + pair[1]` for reproducibility

Key implementation notes:
- `_derive_topk_pack`, `_joint_polish` from cycle 185 work verbatim (both take `block_ids: list[int]`)
- `swap_compressed_ffns(..., inplace=False)` ensures teacher is never mutated between runs
- Use `str(pair)` as JSON dict key (e.g., `"(1, 4)"`)

Aggregation:
```python
best_pair_key = min(pair_summary, key=lambda k: pair_summary[k]["mean_ppl"])
best_ppl = pair_summary[best_pair_key]["mean_ppl"]

if best_ppl < 42.0:   verdict = "CONFIRMED"
elif best_ppl > 48.0: verdict = "REFUTED"
else:                  verdict = "INCONCLUSIVE"
```

Output: `autoresearch/results/cycle_186_fewer_blocks.json`

### Step 2: Run

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/fewer_blocks_cycle186.py
```

Expected runtime: ~15-25 min (3 seeds × 7 runs each).

### Step 3: Write results manifest

`autoresearch/results/cycle_186_manifest.txt`

### Step 4: Update agenda.md

Move item 195 from NEXT P2 to PAST under `### Cycle 186`.

### Step 5: Commit

```bash
git add experiments/fewer_blocks_cycle186.py autoresearch/results/cycle_186_* autoresearch/plans/cycle_186.md autoresearch/briefs/cycle_186.md autoresearch/agenda.md autoresearch/STATUS.md
git commit -m "[autoresearch] cycle 186: fewer-block compression k=4 2-block pairs — <VERDICT>"
```

## Validation

- Teacher baseline PPL in [15, 65] — hard abort if outside
- 4-block baseline PPL should be near 59.4 (accept 45-80)
- Per-pair PPL below teacher baseline = bug; above 200 = polish divergence; flag but continue
- Atlas is probed on tuple(ALL_BLOCKS)=(1,4,7,11) — verify `_derive_topk_pack` filters by block_ids correctly

## Artifacts

- `experiments/fewer_blocks_cycle186.py`
- `autoresearch/results/cycle_186_fewer_blocks.json`
- `autoresearch/results/cycle_186_manifest.txt`

## Fallbacks

- If one seed diverges (PPL >200): base verdict on median, not mean
- If MPS OOM: reduce BATCH_SIZE from 4 to 2
