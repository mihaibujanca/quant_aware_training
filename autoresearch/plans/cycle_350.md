# Cycle 350 Plan — Item 359: MLP Adapter Hidden-Size Sweep

## Files
- Experiment: `experiments/gpt2_adapter_sweep_cycle359.py`
- Results JSON: `autoresearch/results/cycle_350_adapter_sweep.json`
- Manifest: `autoresearch/results/cycle_350_manifest.txt`

## Infrastructure reuse
- Reuse collect_block_activations, topk_lstsq_compress_block, CompressedGPT2MLP from cycle 348
- Block: b2 only (highest concentration: 99.6% top-5 output-covariance energy)
- HIDDEN_FRACTION=1/6, k=512

## New components
- `MLPAdapter(nn.Module)`: residual adapter x → x + W2·ReLU(W1·x + b1) + b2
  - W1: (h, 768), W2: (768, h), initialized small (std=0.01)
  - Installed after compressed MLP in block forward via hook or module replacement
- `CompressedGPT2MLPWithAdapter`: wraps CompressedGPT2MLP + adapter

## Sweep
- h ∈ {16, 32, 64, 128}
- For each h: 3 seeds × 500 steps
- Trainable params per config:
  - compressed proj: 768*512 + 768 = 393,984 (trainable during finetuning)
  - adapter: 2*h*768 + h + 768 params
  - h=16: 24,592; h=32: 49,184; h=64: 98,368; h=128: 196,736

## Evaluation
- eval_ppl same as cycle 348
- Track best PPL per seed (checkpoint best)
- Aggregate: mean ± std over 3 seeds

## Verdict
- CONFIRMED if >=1 h gives PPL < 53.89 with visible Pareto structure
- REFUTED if adapter never beats topk+lstsq alone (48.47) at any h
- INCONCLUSIVE otherwise

## Verification steps
1. Float PPL should be 53.89 ± 0.5
2. topk+lstsq init PPL before adapter training should be ~48.5 (consistent with c348)
3. Final PPLs should monotonically decrease with h (or saturate)
