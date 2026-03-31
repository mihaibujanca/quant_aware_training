# Item
158 — Linear AE ablation — is encoder nonlinearity necessary? — GPT-2 FFN

# Context
- Cycle 137 (item 151): d_bn=64 AE beats topk by 3-3.4% at k=512 on blocks [4, 7]. Pre-polish AE PPL: B4=54.2, B7=55.3; topk: B4=60.8, B7=62.7. Post-polish AE: B4=53.5, B7=54.5; topk: B4=55.3, B7=56.3. Float PPL=57.65.
- Cycle 131: measured ~5% nonlinearity in AE encoder (fraction of tokens where ReLU flips sign). If the ReLU barely activates, a linear AE (= low-rank factorization) trained end-to-end should match.
- This ablation disentangles architecture (ReLU) from training objective (end-to-end CE).

# Dependencies
- 151 (cycle 137, CONFIRMED): AE baseline numbers. MET.

# Key Files
- `experiments/ae_param_match_k512_cycle137.py` — template (BottleneckFFN class, train_model, eval harness)
- `autoresearch/results/cycle_137_ae_param_match.json` — ReLU AE and topk baselines
- `aleph/lm_geometric_compression.py` — teacher loading, probe corpus, swap infra

# Constraints
- 3 methods: ReLU AE (d_bn=64), linear AE (d_bn=64, no ReLU), topk+lstsq. All at k=512.
- Blocks [4, 7] (matched to cycle 137).
- 3 seeds [42, 137, 7], 200 eval sequences, pre-polish only.
- Linear AE: identical to BottleneckFFN but forward does `z = self.encoder(h)` instead of `z = F.relu(self.encoder(h))`.
- ~20 min estimated runtime.

# Handoff
Write `experiments/linear_ae_ablation_cycle148.py`. Fork BottleneckFFN from cycle 137 with a `use_relu` flag. Three methods per block per seed. Report pre-polish PPL only. Save JSON to `autoresearch/results/cycle_148_linear_ae_ablation.json`.
