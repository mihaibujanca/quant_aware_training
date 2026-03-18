# Cycle 028 Plan — Exact Prune Plus Linear Collapse Baseline

## Script
`experiments/exact_prune_linear_collapse_gpt2.py`

## Pre-flight
- P1: `from transformers import GPT2LMHeadModel` — already used in prior cycles, import available
- P2: Shakespeare corpus at `data/shakespeare.txt` — verified in cycle 006
- P3: `torch.linalg.lstsq` available in PyTorch >= 1.9 — confirmed in repo
- P4: No new dependencies; uses gpt2 cached model weights

## Algorithm

### Data
- Load gpt2, load Shakespeare first 50K tokens, seq_len=128
- Probe set: 200 random sequences → collect FFN pre-activation and post-activation per block

### Per Block Analysis
For each FFN block (W_up: 768→3072, W_down: 3072→768):
1. Collect h = W_up @ x for all probe tokens → shape (N_tokens, 3072)
2. Collect act = gelu(h) → shape (N_tokens, 3072)
3. Dead: channels where max(|act|, dim=0) < 1e-5
4. Always-active: channels where min(act, dim=0) > 0 AND
   |gelu(h) - h| / (|h| + 1e-8) < 0.05 for >95% of tokens
   (i.e., channel is in near-linear GELU regime)
5. Near-linear (softer): channels where median(|gelu(h) - h| / (|h| + 1e-8)) < 0.05

### Compression Methods (in order)
1. **Prune-only**: zero W_up rows and W_down cols for dead channels
2. **Prune+collapse**: additionally replace always-active channels with
   linear pass: output += W_down[:, linear_mask] @ h[:, linear_mask]
   instead of W_down[:, linear_mask] @ gelu(h[:, linear_mask])
3. **Prune+collapse+refit**: after zeroing dead and bypassing linear,
   solve lstsq: find W_down_new @ act_reduced ≈ original_output
   on probe data for each block independently

### Evaluation
- Wrap GPT-2 with patched FFN blocks; measure ppl on held-out 50 sequences
- Compute: param_reduction = 3072 / (3072 - n_pruned) for exact pruning
- Report per-block fractions and aggregate

## Results File
`autoresearch/results/cycle_028_prune_collapse.json`
`autoresearch/results/cycle_028_prune_collapse.png`

## Criteria
- Confirmed: any method achieves ≥1.1x param reduction with ≤1% ppl increase
- Refuted: best savings <5% of params, or ppl increase >1% at any useful pruning
- Inconclusive: savings only with thresholds that are hard to justify (>1e-3)
