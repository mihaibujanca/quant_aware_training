# Cycle 373 — Item 381: Probe FFN local Jacobians on calibration data (R3)

## Verdict: PLAN INVALID — duplicate of cycles 068 and 195

## Pre-flight

### P1: Duplicate check — FAIL

Item 381 asks to measure per-token FFN Jacobian effective rank on GPT-2 calibration data for blocks b1, b5, b11, then compare to weight matrix SVD rank. This has been done twice:

- **Cycle 068** (item 80): Per-token Jacobian rank census, all 12 blocks, 512 tokens × 3 seeds. Found mean rank_95 = 305–588/768 across blocks. Random matrix control rank_99 = 592. Outcome: inconclusive (correlation metric).
- **Cycle 195** (item 206): Per-token Jacobian rank census, all 12 blocks, 160 tokens, jacfwd+vmap. Found mean rank_90 = 167–544/768 across blocks. **Verdict: REFUTED.** Learned: "GPT-2 FFN blocks operate in near-full-rank subspaces. No exploitable low-rank active subspace exists."

Item 381 differs only in: (a) nuclear/Frobenius effective rank metric instead of energy-threshold rank, (b) explicit top-10 SV concentration, (c) explicit weight matrix SVD comparison. None of these change the conclusion. The Jacobian rank is 50–71% of full rank at 90% energy (c195). The confirmation criterion requires Jacobian eff_rank < 0.5 × weight eff_rank. Since weight matrices are also 768×768 (the composed FFN Jacobian shape), and the Jacobian ranks are already 50–71% of full rank, the nuclear/Frobenius ratio will not suddenly drop below half the weight rank.

### P2: Dependency check — N/A (blocked by P1)

## Why a different effective rank metric doesn't help

The nuclear/Frobenius ratio (sum(σ)/sqrt(sum(σ²))) is a smooth measure of rank. For a matrix with 50–71% of its singular values contributing to 90% of the energy, this ratio will be high — comparable to the weight matrix ratio. The only scenario where this metric would diverge from the energy-threshold metric is if the Jacobian has a long tail of tiny singular values that still sum to significant nuclear norm, but c068's data shows the opposite: rank_95 and rank_99 are separated by ~100–150 dimensions, meaning the tail drops off reasonably fast.

The top-10 SV concentration is a new metric but it's a weaker test: even if 10 SVs capture 30% of the energy, the remaining 70% still lives in hundreds of dimensions. This doesn't enable practical compression.

## Recommendation

Close item 381. The R3 data-active compression direction has been thoroughly tested on GPT-2 FFNs (2 cycles, 2 metrics, up to 1536 tokens across 3 seeds). The finding is robust: GPT-2 FFN Jacobians are near full-rank on natural text. R3 is dead for GPT-2 FFNs.

If the R3 direction is to be revisited, it would need a genuinely different angle — e.g., Jacobian rank conditional on activation region clusters, or on a different model architecture (MoE experts, which might have sparser activation patterns).
