# Cycle 139 Plan: G-weighted lstsq (Item 147)

## PLAN INVALID

**Reason**: Item 147 depends on item 146 confirming that G-weighted error predicts PPL better than Frobenius. Cycle 136 REFUTED item 146: Frobenius (rho=0.797) beats G-weighted (rho=0.734). The pullback metric is an approximately constant rescaling of Frobenius at the FFN output level. G-weighted lstsq would therefore produce the same solution as standard lstsq (since the weighting is proportional to identity). No experiment to run.
