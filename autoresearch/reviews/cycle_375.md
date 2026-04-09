verdict: APPROVE
round: 1
issues: 0 blocking, 1 non-blocking
feedback:
  - R1 (premise): Sound. ReLU adapters are the correct target for dead-neuron analysis — prior GELU FFN censuses (c128/c162/c192) found 0 dead neurons because GELU has no hard-zero region. Serra/Berzins criteria apply directly to ReLU. The hypothesis that 64-dim adapters are over-provisioned is plausible and actionable.
  - R2 (measurement): Census methodology is exact. Dead: max(pre_relu[:, j]) <= 0 means zero output contribution, row/column deletion is lossless. Always-active: min(pre_relu[:, j]) > 0 means ReLU = identity on that neuron, linear folding into proj_weight/proj_bias is exact. Verification correctly requires PPL delta = 0.000 for both operations.
  - R3 (statistical): Single seed is appropriate — the census is deterministic given calibration data, not a noisy estimate. No replication needed.
  - R4 (method alignment): **Non-blocking issue**: The plan hooks "all calibration sequences" but does not specify the token count. If the standard calibration set is ~200 sequences x ~1024 tokens = ~200K tokens, coverage is adequate (a neuron firing on 0.1% of tokens = ~200 activations, easily detected). If fewer sequences are used, rare-activation neurons could be misclassified as dead. Add a comment in the script logging the total token count used for the census. The built-in safety net (verification that PPL delta = 0.000) would catch any misclassification, so this is non-blocking.
  - R5 (pitfall scan): MP2 — handled correctly: census uses pre-ReLU activations and the activation regime distinction (ReLU vs GELU) is the entire premise. MP6 — thresholds are concrete (10%/5%, 0.1/0.5 PPL). MP3/MP5/MP7/MP8 — not applicable (no fitting, no capacity comparison, no geometric measurements).
