# Task
Cycle 220 fair gelu_noneg rerun with per-channel first-moment correction. Test whether adding the offline-precomputed mean leak μ_l ∈ R^3072 as a post-activation bias rescues the gelu_noneg all-12-block patch from PPL=2093 (cycle 220) toward the float baseline 45.064.

# Hypothesis
> 238 (P1): Cycle 220 fair gelu_noneg rerun with per-channel first-moment correction. For each block l and each FFN hidden channel c, precompute the mean leak μ_{l,c} = E_x[GELU(z_{l,c}(x)) − max(0, GELU(z_{l,c}(x)))] on the calibration split. Add μ_{l,c} as a per-channel post-activation bias when patching gelu_noneg into all 12 blocks.

# Pre-flight Results
- P1 (novelty): cycle 220 had no first-moment correction. NEW.
- P2 (dependencies): cycle 220 done.
- P3 (budget): ≤30 min — 1 forward for μ collection + 1 eval for corrected variant.
- P4 (overwrite): no — new file.

# Steps
1. Create `experiments/gelu_noneg_corrected_cycle228.py` from `experiments/surgical_bump_removal_cycle220.py`. Reuse loader, eval split (N=200, SEQ_LEN=128, seed 42), `eval_ppl`, baseline+restore protocol.
2. Phase 1 — μ precomputation (calibration = same eval split, no_grad):
   - Register forward pre-hook on each `model.transformer.h[l].mlp.c_fc` capturing INPUT (LN2 output) into `stash[l]` (list of tensors).
   - Forward all 200 calibration sequences.
   - For each block l ∈ [0,12):
     - `x_l = torch.cat(stash[l], 0).reshape(-1, 768)`
     - `c_fc = model.transformer.h[l].mlp.c_fc`
     - `preact_l = x_l @ c_fc.weight + c_fc.bias`  # (n_tokens, 3072); Conv1D weight is (768, 3072)
     - `gelu_l = F.gelu(preact_l, approximate='tanh')`
     - `noneg_l = torch.relu(gelu_l)`
     - `mu_l = (gelu_l - noneg_l).mean(dim=0)`  # (3072,); negative entries
   - `mu_dict = {l: mu_l for l in range(12)}`. Remove hooks.
3. Phase 2 — patched eval:
   ```python
   class GeluNonegCorrected(nn.Module):
       def __init__(self, mu):
           super().__init__()
           self.register_buffer('mu', mu)
       def forward(self, z):
           return torch.relu(F.gelu(z, approximate='tanh')) + self.mu
   ```
   - Patch each block l with `GeluNonegCorrected(mu_dict[l]).to(DEVICE)`.
   - Eval PPL → `corrected_noneg_ppl`. Restore originals; sanity-eval baseline.
4. Verdict per item 238: `low_thresh = 49.57`, `refute_thresh = 100`. CONFIRMED if ≤49.57; REFUTED if ≥100; INCONCLUSIVE otherwise.
5. Save JSON `autoresearch/results/cycle_228_gelu_noneg_corrected.json` with: cycle=228, item=238, verdict, baseline_ppl=45.064 (cycle 220), uncorrected_noneg_ppl=2093 (cycle 220), corrected_noneg_ppl, sanity_ppl_after_restore, delta_corrected_vs_baseline, mu_stats per block (mean, std, min, max of |mu_l|), thresholds {low_thresh: 49.57, refute_thresh: 100}, config.
6. Save `autoresearch/results/cycle_228_summary.md` with table {baseline, uncorrected, corrected} and verdict prose.

# Validation
- |mu_l| should be small per channel (cycle 184 mean leak ≈ -0.17 across blocks, so |mu_l| ≈ 0.17 magnitude). Print mean(|mu_l|) per block at compute time.
- Restore-and-eval sanity must equal baseline within 1e-4 (assert).
- Corrected PPL must be MUCH less than 2093 (uncorrected). If it's still ≥ 1000, the hook captured the wrong tensor (e.g., post-c_fc instead of LN2 output) — investigate before finalizing.

# Artifacts
- `experiments/gelu_noneg_corrected_cycle228.py`
- `autoresearch/results/cycle_228_gelu_noneg_corrected.json`
- `autoresearch/results/cycle_228_summary.md`
- `autoresearch/results/cycle_228_manifest.txt` (written by execute)

# Fallbacks
- If `mu` has wrong sign and corrected PPL > uncorrected, sanity-check by printing `mu_dict[0][:5]` and comparing against cycle 184's reported ~-0.17 per channel; flip sign convention if needed (μ defined as `gelu - noneg` is non-positive, added it cancels the missing negative leak).
- If hook input shape unexpected (Conv1D quirks), fall back to capturing OUTPUT of `c_fc` directly (== preact) and skip the matmul step.

# Update bookkeeping
- Move 238 from NEXT/P1 to PAST/Cycle 228 in `autoresearch/agenda.md`.
- Append TSV row to `autoresearch/results/cycles.tsv`.
- Update `autoresearch/STATUS.md`.
- Commit plan.
