verdict: APPROVE
round: 1
issues: minor (no validity risk)
feedback: |
  Clean sweep design. The 6x replication check against cycle 054 uncorrected numbers is good
  experimental hygiene — flag the run early if it diverges >5%.

  1. The plan omits affine correction, justified by cycle 054 showing negligible benefit
     (+48% corrected vs +56% uncorrected at 6x). Worth noting in the results write-up that
     the frontier found here is for raw topk+lstsq, not topk+lstsq+correction. If a ratio
     lands near the 10% boundary, a follow-up with correction could push it under.

  2. 3 seeds is thin for per-cell confidence intervals. Consider reporting per-cell std in
     the JSON output alongside the mean, so the runner can flag any cell where std > 5%
     (indicating the mean is unreliable). No design change needed — just an output addition.

  Review checks:
  - R1 (premise): Sound. Cycle 054 established 6x fails on OWT (+48%), Shakespeare sweeps
    (034-036) showed 6x = +8.5%. Testing lower ratios {2x, 3x, 4x} on OWT is the obvious
    next step. Dependency on cycle 054 satisfied.
  - R2 (measurement): Sound. PPL relative change is deterministic and well-calibrated. The
    10% threshold is meaningfully separated from both the baseline (0%) and the known 6x
    failure (+48%). The 6x replication point provides an internal consistency check.
  - R3 (statistical): Adequate. 200 seqs x 128 tokens = 25,600 samples. At 2x compression
    (1536 channels), lstsq system is 25600 x 1536 — ratio ~17:1, well above MP3 threshold.
    3 seeds per cell is standard for frontier mapping (not per-cell hypothesis testing).
  - R4 (method-architecture): Sound. Fork from established cycle 054 pipeline. Single-block
    swaps isolate per-block behavior. Dropping block 1 (worst in cycle 054) focuses on the
    most promising blocks.
  - R5 (pitfalls): No pitfall applies. MP3 cleared (17:1 overdetermination at lowest ratio).
    MP4 is precisely what the experiment tests (domain transfer). MP6 low risk — PPL will
    vary meaningfully across the 2x-to-6x range.

pitfall_scan: No pitfalls apply.
