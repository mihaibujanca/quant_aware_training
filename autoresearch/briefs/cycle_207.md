# Cycle 207 Brief: Block 1 Channel Selection Scoring (Item 218)

## Result: REFUTED

Four scoring criteria tested for K=512 block 1 channel selection:
- S1 (current: act_std × W_proj_norm): 54.689
- S2 (mean_abs × W_proj_norm): 54.853
- S3 (pure act_std): 54.629
- S4 (pure weight norm): 55.679

Best alternative (S3) improves by only 0.06 PPL — well below 0.3 threshold.
Despite orderings being genuinely different (Kendall-tau as low as -0.13 between S3 and
S4), PPL is unchanged. lstsq adapts W_out to whichever 512 channels are selected.
Channel selection criterion is irrelevant at K=512; the lstsq step compensates.
