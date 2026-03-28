cycle=078
item=90
status=CONFIRMED
verdict_correct=yes
G1_implementation=PASS
G2_seeds=PASS (3 seeds: 42, 123, 7)
G3_interpretation=FLAG
G4_counterintuitive=PASS
G5_learning=N/A (lstsq refit, no training)
G6_method_failure=PASS
interesting=yes

## Notes

G3 flag: Condition C includes 512 WikiText-2 tokens in the refit data, and eval is WikiText-2 validation.
This is distribution-level leakage: the mixed condition sees tokens from the same distribution as the
eval set, inflating C's advantage. The 19.1% improvement conflates "diversity helps lstsq conditioning"
with "including eval-domain data in refit helps." A cleaner test would use 3 non-WikiText corpora in C
and still eval on WikiText-2. The claim "corpus diversity genuinely helps" is overstated; a significant
portion of the gain may be from eval-domain inclusion rather than diversity per se.

Verdict is still CONFIRMED because the 19.1% exceeds the 5% threshold regardless of mechanism, but the
causal interpretation in the agenda writeup should be softened.
