# Task
Train a tiny Mixtral MoE (15.5M params) on WikiText-2, implement REAP expert saliency scoring, and measure PPL degradation under 25/50/75% global pruning. First item of MoE sequence (blocks 129-131).

# Hypothesis
- **Confirmed** if: model trains to PPL < 50 AND REAP pruning shows graded, monotonic PPL degradation (25% < 50% < 75%) AND saliency scores have meaningful spread (Gini > 0.3 in majority of layers).
- **Refuted** if: PPL is insensitive to which experts are pruned (random pruning ≈ REAP), indicating saliency scoring is ineffective at this scale.
- **Inconclusive** if: graded degradation but some layers are insensitive (flat saliency), suggesting per-layer pruning rates would outperform global.

# Pre-flight Results
- P1 PASS: No MoE experiments in PAST EXPERIMENTS.
- P2 PASS: Item 128 depends on nothing.
- P3 PASS: Fully specified config, dataset, metrics, thresholds.
- P5 (null baseline): Random pruning at each level (3 seeds) reported alongside REAP.
- P6 (methods): SiLU-gated FFN experts; REAP scoring is architecture-appropriate.
- P7 (power): 65K routing events per layer across 8 experts — ample for Gini and ranking.

# Steps
1. Create `experiments/moe_reap_pruning_tiny_mixtral.py`:
   - **Tokenizer**: Train a BPE tokenizer on WikiText-2 train split using `tokenizers` library (BPE, vocab_size=4096, min_frequency=2, special tokens: `<pad>`, `<eos>`, `<unk>`). Save to `experiments/results/cycle_118_tokenizer/`. ~2 min.
   - MixtralConfig as specified in agenda (hidden=256, intermediate=512, 4 layers, 8 experts, top-2, vocab=4096, router_aux_loss_coef=0.001).
   - Tokenize WikiText-2 train/val with the trained BPE tokenizer. Pack sequences to seq_len=256 without padding (concatenate all tokens, chunk into 256-length blocks, drop final short block).
   - Training: 10M tokens, batch=4, seq_len=256, AdamW lr=1e-4, cosine schedule. Log train loss every 100 steps. device="cpu".
   - Validation: PPL on 256 val sequences. If PPL > 50, extend to 20M tokens.
   - **Router entropy check**: after training, forward 256 val sequences. Per layer, compute router entropy: `H = -sum(p * log2(p))` where p = mean softmax gate distribution over all tokens. Flag any layer with H < 1.0 bit (fewer than 2 effective experts = degenerate specialization). Log per-layer entropy values.
   - Save checkpoint to `experiments/results/cycle_118_tiny_mixtral.pt`.

2. REAP saliency scoring:
   - Forward 256 val sequences with `output_router_logits=True`.
   - Per layer, per token: `score[l,k] += gate_softmax[k] * ||expert_k_output||_2`. Average over **non-padding tokens only** (moot with packing, but guard against future changes: skip any token where input_id == pad_token_id).
   - Per-layer Gini coefficient, top-4 vs bottom-4 spread, cross-layer Spearman correlation.

3. Pruning evaluation:
   - Global ranking: flatten (layer, expert) saliency scores, sort ascending, prune lowest N.
   - At each level (25%, 50%, 75%): zero out pruned expert router logits before softmax (renormalize remaining), measure PPL on 256 val sequences.
   - **Random baseline**: 3 random seeds per level, same global count. Report mean +/- std.

4. Per-layer ablation: for each of 4 layers, prune that layer's lowest-saliency experts to 50% (keep other layers full). Measure PPL delta vs baseline.

5. Save results to `autoresearch/results/cycle_118_moe_reap.json` and `autoresearch/results/cycle_118_moe_reap.png` (3-panel: saliency heatmap, PPL curve with random baseline, per-layer damage).

6. Update `autoresearch/agenda.md`: move item 128 from NEXT to PAST EXPERIMENTS with outcome, commit hash, artifact paths.

7. Commit: `[autoresearch] cycle 118: MoE REAP pruning baseline — Tiny Mixtral`

# Validation
- Training: val PPL < 50. If PPL > 100, model is undertrained — do not proceed.
- Router entropy: all layers H >= 1.0 bit. If any layer is degenerate, note in results but continue.
- Pruning monotonicity: PPL(25%) < PPL(50%) < PPL(75%). Verify numerically.
- REAP vs random: at 50%, REAP PPL should be lower than random mean by > 1 std.
- Gini > 0.3 in at least 3/4 layers.
- Baseline PPL (0% pruned) matches training val PPL exactly.

# Artifacts
- `experiments/moe_reap_pruning_tiny_mixtral.py`
- `experiments/results/cycle_118_tokenizer/` — trained BPE tokenizer
- `experiments/results/cycle_118_tiny_mixtral.pt` — checkpoint
- `autoresearch/results/cycle_118_moe_reap.json` — all metrics (includes router entropy)
- `autoresearch/results/cycle_118_moe_reap.png` — visualization

# Fallbacks
- If PPL > 50 at 10M tokens: extend to 20M (~28 min).
- If PPL > 50 at 20M: increase model (hidden=384, intermediate=768) or reduce vocab to 2048.
- If `output_router_logits` API differs: hook via `register_forward_hook` on `MixtralSparseMoeBlock`.
- If CPU training > 30 min: reduce to 5M tokens, check loss trajectory, decide.
