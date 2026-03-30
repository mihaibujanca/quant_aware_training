# Cycle 120 Review (Round 2) -- Item 137: Cascade corrections at k=8 stacking

## Round 1 issue resolution -- RESOLVED

The blocking issue from round 1 was the block set. The agenda specified [1,2,3,4,5,7,9,11] which (a) was not a superset of the k=5 set [7,5,4,1,6] (missing block 6), and (b) included block 11 (second-hardest). The revised plan adopts **[1,2,3,4,5,6,7,9]** = k=5 set {1,4,5,6,7} + next 3 easiest {2,3,9}. Deviation from agenda is documented with scientific rationale. Correct.

## R1: Premise validity -- PASS
Unchanged from round 1. Cycle 119 confirmed cascade corrections at k=5 (73.8% recovery, r=8). k=8 is the natural scaling test.

## R2: Measurement soundness -- PASS
Same recovery metric as cycles 115/119. Sanity checks P1-P7 are comprehensive. P5 now correctly references cycle 119's k=5 PPL (~70) as the comparison baseline. Fallback logic (r=16 if r=8 < 10%) is cleanly gated.

## R3: Capacity and statistical adequacy -- FLAG (accepted)
Single seed. Agenda-mandated scope decision. Plan documents the limitation and recommends 3-seed follow-up if recovery is ambiguous (10-30%). Acceptable for exploratory.

## R4: Method-architecture alignment -- PASS
Same LoRA correction architecture (U_l, V_l per block) and pipeline (topk+lstsq+500 Adam) as cycles 115 and 119. No novel choices.

## R5: Known pitfall scan -- PASS
- MP3 (underdetermined fitting): r=8 → 98,304 params vs ~1M tokens. 10:1 ratio. Adequate. r=16 fallback at ~5:1 is marginal but above 2x.
- All other pitfalls (MP1-MP8) clear, same analysis as round 1.

## Advisory (carried forward)

**A3 (Correction training stability).** 8 simultaneous corrections may interact. If training diverges, reduce lr to 5e-5 before declaring failure. Plan incorporates this.

## Verdict: APPROVE
