verdict: APPROVE
round: 1
issues: 0 blocking, 2 minor

## R1: Premise
Sound. Cycle 071 established that SVD+lstsq ~ random+lstsq at k/n=0.67. The natural follow-up is whether this parity holds at aggressive compression. The information-theoretic argument is clear: at k/n=0.1, only 77 directions survive out of 768. If the FFN output concentrates on a low-rank subspace (rank_99 ~ 434-510 from cycle 068), the SVD basis should capture more variance per retained direction than random. If SVD still matches random at k/n=0.1, the function is genuinely diffuse across all directions and no linear basis is privileged.

## R2: Measurement
Single-block swap PPL is the established metric from cycle 071, enabling direct comparison. The 5% relative threshold for confirmation is reasonable: at k/n=0.67, the SVD-random gap was ~0.4% (121.84 vs 122.31). A 5% threshold at k/n<=0.3 is generous enough to detect meaningful differences while filtering noise. The 3% matching threshold for refutation is appropriately tighter.

## R3: Seeds
3 seeds [42, 123, 7] matching cycle 071. Cross-seed std < 20% guard. Adequate for the 3-way comparison. The random basis is also seeded (seed+block_idx), ensuring reproducibility without accidentally correlating random bases across blocks.

## R4: Method-architecture match
All three methods are established from cycle 071. The adaptation to multiple k values is mechanical — same hooks, same lstsq, just varying the number of retained directions. The plan correctly notes Conv1D transposition and autograd-based GELU derivative (both validated in cycle 071). The lstsq overdetermination check at k=230 (N/p=4.4) is adequate per MP3.

## R5: Pitfalls
- MP3 (underdetermined fitting): At k=230 (worst case), lstsq has 231 parameters (including bias). 1024 tokens gives N/p=4.4, above the 2x threshold. At k=77, N/p=13.3. All safe. NOT TRIGGERED.
- MP5 (insufficient capacity): At k/n=0.1, all methods will degrade severely. But the comparison is between methods at matched budget, so capacity is equalized. NOT TRIGGERED.
- MP6 (verdict misclassification): If all methods produce PPL > 1000 at k/n=0.1, the comparison may be between failure modes. The fallback correctly addresses this — focus on k/n=0.2-0.3 if k/n=0.1 blows up. NOT TRIGGERED (mitigated by fallback).
- MP7 (post-LayerNorm): Hooks on mlp.c_fc capture FFN internals, not post-LayerNorm. NOT TRIGGERED.
- MP8 (representation vs functional rank): The Jacobian SVD measures functional rank directly. The experiment tests whether this functional basis outperforms random. Correctly aligned. NOT TRIGGERED.

## Minor observations

1. **topk+lstsq parameter count mismatch.** At k/n=0.67 in cycle 071, topk+lstsq stored 787K params vs SVD+lstsq 393K. At k/n=0.1 (k=77), topk+lstsq retains 77 hidden channels → W_down is 768x77 = 59K params. SVD+lstsq stores B (768x77) = 59K params. Methods are now parameter-matched, unlike at k/n=0.67. This is actually cleaner — note it in results.

2. **Random basis seed scheme.** Using seed+block_idx means block 1 seed 42 uses random seed 43, block 4 uses 46, etc. Fine for reproducibility, but ensure the implementation matches so results are reproducible. Consider documenting the exact seed formula in the script.
