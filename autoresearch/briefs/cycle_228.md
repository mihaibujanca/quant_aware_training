# Item
238 (P1): Cycle 220 fair gelu_noneg rerun with per-channel first-moment correction.
For each block l and each FFN hidden channel c, precompute the mean leak μ_{l,c} = E_x[GELU(z_{l,c}(x)) − max(0, GELU(z_{l,c}(x)))] on the calibration split. Add μ_{l,c} as a per-channel post-activation bias when patching gelu_noneg into all 12 blocks.

# Context
- Cycle 220 (INCONCLUSIVE): gelu_noneg(z) = max(0, GELU(z)) all-12 patch → 2093 PPL (+4545%). Cycle 220 audit (G4): no first-moment correction.
- Cycle 184 ReLU+bias variant has first-moment correction and was only +88% — 46× milder than this cycle's gelu_noneg.
- The hypothesis: with mean shift compensation, gelu_noneg should be much closer to float (test the "near-linearizable" claim).

# Dependencies
- 234 / cycle 220 (done).

# Key Files
- experiments/surgical_bump_removal_cycle220.py — base patching infrastructure.
- experiments/adam_wout_correction_cycle206.py — calibration data loader.

# Constraints
- N=200 wikitext-2 calibration sequences × 128 tokens, same as cycle 220.
- Same eval split (N=200 wikitext-2 val).
- Bias correction is OFFLINE: precompute μ_{l,c} from a clean float forward pass on calibration data, then add it to the c_fc post-activation when patching with gelu_noneg.
- More precisely: replace the activation in block l with `gelu_noneg_l(z) = max(0, GELU(z)) + μ_l` where μ_l ∈ R^3072 is a per-channel mean bias.
- Single deterministic eval; no seeds needed.
- Confirmed if: corrected_ppl ≤ 49.57 (the cycle 220 low_thresh = baseline_ppl × 1.10 = 45.064 × 1.10).
- Refuted if: corrected_ppl ≥ 100 (>2× float).
- Inconclusive if: 49.57 < corrected_ppl < 100.

# Handoff
Plan agent: design the precomputation step. New script `experiments/gelu_noneg_corrected_cycle228.py`. Steps:
1. Load HF gpt2.
2. Forward pass with hooks on each block's c_fc to capture pre-activation z_{l} (shape (n_tokens, 3072)).
3. Compute float gelu(z_{l}) and noneg = max(0, gelu(z_{l})). μ_l = (gelu - noneg).mean(dim=0). Shape (3072,).
4. Define corrected activation per block:
   class GeluNonegCorrected(nn.Module):
       def __init__(self, mu_l):
           super().__init__()
           self.register_buffer('mu', mu_l)
       def forward(self, z):
           return torch.relu(F.gelu(z, approximate='tanh')) + self.mu
5. Patch all 12 blocks with the corresponding GeluNonegCorrected(mu_l). Eval PPL.
6. Compare to cycle 220 baseline 45.064 and uncorrected 2093.

Tight scope. ≤30 min wall time.
