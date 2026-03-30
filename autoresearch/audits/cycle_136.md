status=PASS
old_outcome=refuted
new_outcome=refuted
reason=Frobenius rho=0.797 >= G-weighted rho=0.734. G/Frob ratio nearly constant (34-35x), meaning pullback is isotropic at FFN output level. Clear REFUTED.
guardrail_flags=none
recommendations=G-weighted lstsq (147) loses justification from this result but may still work within-block. Prioritize item 151 (AE param-matched) next.
interesting=true
interesting_reason=The near-constant G/Frob ratio means the unembedding matrix projects FFN output errors approximately uniformly — the pullback metric is isotropic at the FFN output layer. This is surprising given cycle 135 showed non-trivial curvature in the trajectory. The resolution: curvature captures DIRECTION of residual stream change, not ERROR magnitude. The pullback amplification of errors is uniform, but the direction of errors relative to trajectory curvature is what matters — and that's captured by Frobenius since the amplification is constant.
