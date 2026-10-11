# Item
234 (P0): Surgical bump removal — isolate negative leak from positive-zone bump.
Define two new activations:
(A) `GELU_noneg(z) = max(0, GELU(z))` — kills the negative leak (clamps GELU's negative tail to 0), keeps the positive bump shape.
(B) `GELU_nobump(z) = z * 1[z > 2] + 0 * 1[z < −2] + linear_interp(z) for z in [−2, 2]` — kills the bump knee, keeps the asymptotic regimes (a piecewise-linear approximation).
Run each replacing the activation in all 12 blocks of GPT-2 and measure full-model PPL.

# Context
- Cycle 184 (refuted): GELU vs ReLU+bias divergence is exponential, R²=0.972, but DIFFUSE rank.
- Cycle 188 (inconclusive): C3 per-channel mean replacement → +20.33% PPL all-12.
- Cycle 190 (inconclusive): GELU_neg + ReLU_pos hybrid (kept GELU below 0, used ReLU above 0) → +27.67% all-12.
- Cycle 193 (inconclusive): C3+Hybrid → +82.59% (superadditive).
- Cycle 218 (CONFIRMED): centred Δ̃ writes structured info into next block's c_fc and residual PCs in 11/11 pairs → bump is real inter-block communication.
- This item is the cleanest decomposition: split GELU into "negative leak" and "positive-zone smoothness" and remove each independently. This complements cycle 190's hybrid, which removed both half-effects via a different cut.

# Dependencies
- None. Standalone PPL test.

# Key Files
- experiments/adam_wout_correction_cycle206.py — has the data loader (200 sequences × 128 tokens) and PPL eval.
- HF GPT-2: model.transformer.h[i].mlp.act is the activation module (NewGELUActivation or similar). To swap: monkey-patch all 12 blocks to use a custom activation function.

# Constraints
- Same eval set as cycles 206/210/215/216/217/218 — N=200 sequences, SEQ_LEN=128. Match for comparability.
- Use the EXACT same eval_ppl() function as cycle 206.
- Three activations to test:
  1. Baseline GELU (the model's float behavior; PPL ~22.7-26 depending on eval set; cycle 184 measured 22.7).
  2. GELU_noneg(z) = max(0, GELU(z)).
  3. GELU_nobump(z) = a piecewise-linear approximation that matches GELU asymptotically:
     - z > 2: identity (z)
     - z < -2: 0
     - -2 ≤ z ≤ 2: linear interpolation between the two anchor values (z=−2 → 0, z=2 → 2)
       So linear part is (z+2)/2 for z in [−2, 2]. Gradient = 0.5.
- Plus the baselines from prior cycles for context: ReLU (cycle 184) and Hybrid (cycle 190).
- Confirmed criteria from agenda:
  - Confirmed: one of {GELU_noneg, GELU_nobump} all-12 PPL ≤ 25 AND the other ≥ 35 — clean separation.
  - Refuted: both within 3 PPL of each other.
  - Inconclusive: both degrade 20-50% with no clean separation.
- Single forward pass per variant; no training. Should run in ~30 seconds total.
- 3 random seeds NOT applicable here — model is deterministic given the activation. Use ONE deterministic eval pass.

# Handoff
Plan agent: design exact computation. New script `experiments/surgical_bump_removal_cycle220.py`. Steps:
1. Reuse cycle 206 data loader.
2. Compute baseline GELU PPL (sanity check ~22-26).
3. Define gelu_noneg(z) = torch.relu(F.gelu(z, approximate='tanh')).
4. Define gelu_nobump(z) = piecewise: z * (z > 2).float() + 0.5 * (z + 2) * ((z >= -2) & (z <= 2)).float() + 0 * (z < -2).float(). Verify the asymptotic behaviors.
5. Patch all 12 blocks' mlp.act module to a Function-based wrapper.
6. Run eval_ppl() per variant. Print all-12 PPL.
7. Verdict: compare with thresholds {≤25, ≥35} and {3 PPL gap}.
8. Output JSON + summary markdown.

Tight scope, ≤30 min wall time.
