# Cycle 044 Plan — Domain Robustness: Non-Shakespeare Eval

## Goal
Test whether k=4@6x channel selection + lstsq correction transfers to WikiText-2.

## Script
New file: `experiments/domain_robustness_wikitext.py` (adapt from `ffn_residual_correction_k5.py`).

## Data Preparation
Download WikiText-2 raw validation split to `data/wikitext2_valid.txt`:
```
wget -O data/wikitext2_valid.txt \
  "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/validation/0000.parquet"
```
Actually: WikiText-2 raw is distributed as parquet. Easier: download via a small Python fetch from the HuggingFace raw text URL, or extract text from parquet using pyarrow.

**Simpler**: Use `urllib` to download the raw text version from the Salesforce mirror or use `pyarrow` to read the parquet. The script should handle this automatically: if `data/wikitext2_valid.txt` does not exist, download and cache it.

## Procedure

### Step 1 — Load GPT-2 + Shakespeare baseline
Load teacher, build Shakespeare ProbeCorpusSpec (200 train / 50 holdout / 200 eval, seq_len=128). Measure baseline PPL on Shakespeare eval.

### Step 2 — Load WikiText-2
Download/cache WikiText-2 validation. Build a second ProbeCorpusSpec with `text_path="data/wikitext2_valid.txt"` and `dataset_name="wikitext2"`. Measure baseline PPL on WikiText-2 eval. Note: GPT-2 baseline PPL on WikiText-2 will differ from Shakespeare — all comparisons must be relative (% increase).

### Step 3 — Condition A: Shakespeare probe + Shakespeare eval (replication)
Compress k=4 blocks [7,5,4,1] at 6x using Shakespeare probe. Fit correction at block 8 input using Shakespeare probe. Eval on Shakespeare. This replicates cycle 039 — sanity check.

### Step 4 — Condition B: WikiText-2 probe + WikiText-2 eval (corpus-specific)
Compress k=4 blocks [7,5,4,1] at 6x using WikiText-2 probe. Fit correction at block 8 using WikiText-2 probe. Eval on WikiText-2. This is the corpus-specific upper bound.

### Step 5 — Condition C: Shakespeare probe + WikiText-2 eval (zero-shot transfer)
Use the compressed model from Condition A (Shakespeare-fit channels + Shakespeare-fit correction). Eval on WikiText-2. This is the transfer test.

### Step 6 — Additional: Shakespeare probe + WikiText-2 eval, re-fit correction only
Same compressed blocks as Condition A, but re-fit correction on WikiText-2 probe. Eval on WikiText-2. This isolates whether transfer failure (if any) is from channel selection or correction.

### Step 7 — 3 seeds, 200 eval seqs per condition
Seeds [42, 137, 2024]. Per-seed: compress + correct + eval. Report mean +/- std for each condition.

## Verdict Criteria (from agenda)
- **CONFIRMED**: zero-shot transfer (C) PPL cost within 3pp of corpus-specific (B) on WikiText-2.
- **REFUTED**: zero-shot transfer causes >20% PPL increase on WikiText-2 while corpus-specific stays <10%.
- **INCONCLUSIVE**: intermediate — transfer works for one metric but not the other.

## Outputs
- `autoresearch/results/cycle_044_domain_robustness.json`
- `autoresearch/results/cycle_044_domain_robustness.png` (bar chart: conditions A/B/C/D by seed)

## Estimated cost
~15 min: model load (1x), 2 corpora builds, 4 conditions x 3 seeds x (compress + correct + eval).
WikiText-2 download: <5 seconds (~4MB).
