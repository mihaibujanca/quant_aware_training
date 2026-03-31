# Cycle 171 Audit — Item 181: Wider-Shallower Compression

## G1 (Correctness)
PASS. Script uses established topk+lstsq infrastructure from `aleph/lm_geometric_compression.py`.
Joint polish (all c_proj weights simultaneously) is a valid variant; baseline PPL 59.4 is actually
better than per-block polish baseline (~70.7), which makes the comparison MORE favorable to
wider-shallower configs. Despite this advantage, all wider configs are 68-124 PPL points worse.

## G2 (Seeds)
PASS. 3 seeds (42, 137, 7). Per-seed values stored in JSON. Consistent pattern across all seeds:
wider-shallower always worse by large margins. Std deviation is 8-34 PPL.

## G3 (Verdict criteria)
PASS. Item criterion: "Refuted if 12@k=8 PPL >= 4@k=4 PPL." Result: 183.4 >= 59.4. REFUTED.

## G4 (Bookkeeping)
PASS. Item 181 moved from NEXT to PAST in agenda.md. TSV entry appended. Manifest written.

## G5 (Artifacts)
PASS. JSON results at `autoresearch/results/cycle_171_wider_shallower.json`.
Script at `experiments/wider_shallower_compression_cycle171.py`.

## G6 (Interesting?)
interesting=false
interesting_reason=Result matches the null hypothesis (c=0.41 compounding). No surprise — this
is what cycles 089/093 predicted: touching more blocks compounds geometrically. The experiment
was still worth running to close the hypothesis explicitly.
