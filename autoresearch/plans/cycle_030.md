# Cycle 030 Plan — Learned Compressor Generalization on Held-Out Blocks

## Pre-flight
- P1: GPT-2 loads via transformers; Shakespeare at data/shakespeare.txt — confirmed cycle 029
- P2: aleph/lm_geometric_compression.py exports load_nanochat_teacher, build_probe_corpus,
      probe_ffn_geometry, _apply_activation, _evaluate_model — confirmed cycle 029
- P3: Geometric baseline (channel-select + lstsq) available via derive_ffn_compression
- P4: .venv/bin/python functional; results dir exists at autoresearch/results/

## Script: experiments/learned_compressor_generalization.py

### Data
- Load GPT-2 via transformers, Shakespeare text
- Probe corpus: 128 tokens × 32 sequences (same as cycle 029)
- Collect (h_in, h_out) pairs for all 12 blocks using forward hooks

### Architecture
SharedBottleneck:
  - U: nn.Linear(768, 384, bias=False)  — shared encoder
  - V: nn.Linear(384, 768, bias=False)  — shared decoder
  - b: nn.Parameter(shape=[n_train_blocks, 768])  — per-block bias
  - forward(h, block_idx): V(relu(U(h))) + b[block_idx]

### Training (train blocks 0-7)
- Loss: MSE(SharedBottleneck(h_in, i), h_out) summed over all train blocks
- Optimizer: Adam, lr=1e-3, 500 steps, batch=64 tokens randomly sampled per step
- No SGD; pure supervised regression on probe activations

### Evaluation (held-out blocks 8-11)
For each held-out block b:
  1. Fix U, V from training
  2. Fit b_new = lstsq(V(relu(U(H_in))), H_out - V(relu(U(H_in)))) solved as:
     b_new = mean(H_out - V(relu(U(H_in))), dim=0)  [bias-only adaptation]
  3. Inject SharedBottleneck(h_in, b_new) as the FFN replacement
  4. Measure PPL on Shakespeare

### Baselines
A. Geometric (per-block): top-384 channels by activity score + lstsq W_down refit
   (same as cycle 029 global geometric)
B. Per-block distillation: train separate 768→384→768 MLP for each held-out block
   on the same probe tokens (500 steps, Adam lr=1e-3) — in-distribution upper bound

### Metrics saved to autoresearch/results/cycle_030_generalization.json
- baseline_ppl (uncompressed GPT-2)
- per block: geometric_ppl, distill_ppl, shared_ppl
- in_distribution: shared_ppl on train blocks (sanity check)
- generalization_ratio: shared_ppl_held_out / shared_ppl_train (target ≤ 1.25)
- beats_geometric_count: count of held-out blocks where shared_ppl < geometric_ppl

### Verdict
- CONFIRMED: beats_geometric_count >= 3 AND generalization_ratio <= 1.25
- REFUTED: shared_ppl_held_out >> both geometric and distill on majority of blocks
- INCONCLUSIVE: otherwise
