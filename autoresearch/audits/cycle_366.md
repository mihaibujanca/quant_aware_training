status=PASS
old_outcome=CONFIRMED — Inflection block is 9 in all 3/3 seeds
new_outcome=CONFIRMED — Inflection at transition 9→10 in 3/3 seeds; b10+b11 drive error explosion; amplification ratio 0.91 (actual < additive baseline)
reason=Inflection calculation verified manually: median growth=5.82, 2×median=11.63, first g_i exceeding this is g_9=21.9 (transition 9→10). All 3 seeds agree (block 9). Additive baseline and amplification ratio recomputed and match JSON. Cross-cycle consistency with cycle 362 Frobenius norms confirmed (identical block ordering: b11>b10>b4>b3>b2>b1).
guardrail_flags=G4-minor: "attention cancels error" attribution is imprecise
tsv_path=autoresearch/logs/experiments.tsv
recommendations=The manifest claims "attention does NOT amplify error" but the experiment does not isolate attention from layernorm or non-compressed FFN processing on perturbed inputs. The ratio < 1 is genuine but should be attributed to "combined non-compressed processing" not "attention" specifically. For follow-up: prioritize compression quality on b10/b11 (they account for 77% of total local FFN error). Consider adaptive k allocation giving b10/b11 higher k at the expense of early blocks.
interesting=true
interesting_reason=First quantitative map of WHERE the composition barrier manifests in GPT-2 residual stream. The hockey-stick error profile (slow growth blocks 0-9, explosion at 10-11) with identical inflection across 3 seeds provides a clear compression priority ordering. The sub-unity amplification ratio (0.91) is a useful finding: non-compressed blocks partially absorb error, meaning the stacking problem is local to the late compressed blocks, not a cascading amplification.
