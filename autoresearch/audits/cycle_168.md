status=PASS
old_outcome=CONFIRMED
new_outcome=CONFIRMED
reason=Criteria: CONFIRMED if max_ratio(|z_mean|) < 2.0 across small/medium/large AND all z_mean < -0.3. Result: max_ratio=1.47, all three scales firmly negative (small=-0.97, medium=-1.12, large=-1.43). Code correctly hooks mlp.c_fc output, decomposes z = Wx + b, and computes per-block means. PPL sanity passes for all three models (small=32.8, medium=24.5, large=20.9 on wikitext-2). GPT-2 small z_mean=-0.97 is consistent with cycle 165 result (-0.93 on 1024 tokens; minor token-sample variance). The mild scale trend (1.47x) is within CONFIRMED bounds but worth noting: larger models have more anti-alignment and less bias contribution.
guardrail_flags=none
tsv_path=autoresearch/logs/experiments.tsv
recommendations=The 1.47x scale trend suggests larger models push more of the FFN activation mass into the negative (GELU-suppressed) regime. Combined with cycle 165 (negativity is learned) and cycle 166 (negative regime = constant bias), this means the "constant bias" component grows with scale. For compression: the dead-zone fraction increases from 84.5% to 90.8% at larger scales, potentially making topk selection even more effective for larger models. Consider testing topk compression on GPT-2 medium to see if the higher dead fraction translates to better compression ratios.
interesting=true
interesting_reason=The mild but consistent scale trend (larger models = more negative z_mean, higher dead fraction) is a novel empirical observation. If this extends to GPT-2 XL and larger models, it implies the compression opportunity grows with scale — a useful signal for practical deployment. The bias contribution also shifts: at large scale, nearly all negativity comes from learned W-x anti-alignment, not from the bias term.
