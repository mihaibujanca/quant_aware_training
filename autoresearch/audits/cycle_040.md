status: UNCERTAIN
original_outcome: confirmed
audited_outcome: confirmed
guardrail_flags: G4 (counterintuitive — 10x compression gives sub-baseline PPL)
reason: All 3 blocks tolerate 10× with near-zero or negative PPL impact — this is deeply counterintuitive and warrants scrutiny. The 6× cross-check matches cycle 035 exactly, confirming pipeline consistency. The sub-baseline effect was already established in cycle 035 (multi-seed) as real — channel selection regularizes by removing noisy channels. At 10× (keeping 307/3072 channels), the lstsq refit on 200 train sequences fits 307×768 + 768 ≈ 236K output parameters from ~25K token samples — this IS underdetermined (samples < params). The refit is effectively memorizing the probe data, and eval on a different Shakespeare split doesn't catch overfitting because Shakespeare text is highly self-similar. The CONFIRMED verdict holds per criteria (mean ≤ 10%), but the sub-baseline values should be treated with suspicion.
recommendations: Test on a genuinely different corpus (e.g., Wikipedia, news text) to check whether the compression-as-regularization effect survives domain shift. The lstsq refit at 10x may be Shakespeare-specific.
