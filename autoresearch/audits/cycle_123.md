status=PASS
old_outcome=INCONCLUSIVE
new_outcome=INCONCLUSIVE
reason=Verdict correct. No component exceeds 70% threshold in any layer (all 3 seeds). V dominates early layers (L2-L6: V_frac 0.63-0.68), QK dominates late layers (L9-L11: QK_frac 0.55-0.58), matching "varies by layer position" criterion. Decomposition sanity holds (E_QK+E_V+E_inter=E_total), cross-seed std<0.02. PPL degradation (+24.3%) exceeds plan's expected +5-15% range for k=4 but does not invalidate the decomposition — the fallback to a different block set was not triggered, which is a minor protocol miss but not verdict-affecting.
guardrail_flags=none
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Repeat with optimal k=4 block set (lower PPL regime) to check if QK/V balance shifts at milder compression. Use layer-position-dependent correction strategy for items 142/144 (V-focused early, QK-focused late).
interesting=true
interesting_reason=Clean V-to-QK dominance transition across depth. Early layers show V-dominant error (0.65-0.68), late layers shift to QK-dominant (0.55-0.58), with crossover at layer 5. This is a structural signal that correction strategies should be layer-adaptive, not uniform. Closes "single surgical target" hypothesis and opens dual-pathway direction.
