status=PASS
old_outcome=REFUTED
new_outcome=REFUTED
reason=Centroid-FPS selects distinct channels (Jaccard=0.333) that are strictly worse than topk across all 3 seeds and all blocks. Delta = -0.627 PPL (centroid worse). Implementation correct: centroid computation (A.T@H/N), FPS seeding, lstsq refit, polish all match baseline protocol. Verdict logic slightly loose (checks delta<=0.5 rather than abs(delta)<=0.5, but directionally correct for refutation since centroid failing to beat topk is sufficient).
guardrail_flags=none
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Topk near-optimality further confirmed. Geometric channel selection via coverage/diversity in d_model space does not outperform importance-weighted selection. Future channel selection ideas should explain why they would beat activation-norm ranking, not just offer diversity.
interesting=false
interesting_reason=
