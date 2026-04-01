# Design Review — Cycle 192 (Item 202): Exact Dead-Channel Pruning

## Correctness

**R1 — Conv1D weight layout**: GPT-2 uses `Conv1D` (not nn.Linear). Conv1D stores weight as
`(in_features, out_features)` = transposed vs nn.Linear. For c_fc: (768, 3072). For c_proj:
(3072, 768). The matmul is `x @ weight`. So to zero dead channels at position c in the 3072 dim:
- `c_fc.weight[:, c] = 0` (column c of the (768, 3072) matrix)
- `c_fc.bias[c] = 0`
- `c_proj.weight[c, :] = 0` (row c of the (3072, 768) matrix)
**CORRECT per Conv1D convention**.

**R2 — Max GELU computation**: `max_gelu[b]` tracks max over ALL calibration tokens. This is
a valid "near-dead" criterion: if GELU(z_c) never exceeds epsilon, the channel contributes
near-zero output for the entire calibration distribution. **CORRECT**.

**R3 — Pruning effect**: Zeroing `c_fc.weight[:, c]` means channel c of the FFN hidden layer
always gets input weighted to 0. Zeroing `c_fc.bias[c]` means z_c = 0 always. GELU(0) = 0.
Zeroing `c_proj.weight[c, :]` means the zero GELU output has zero contribution to the output.
The pruning is consistent: dead channels contribute nothing. **CORRECT**.

**R4 — GELU expected to have near-dead channels**: GPT-2 FFN processes tokens from a
pretrained model. The GELU negative regime creates suppressed activations, but whether
any channels are ALWAYS near-dead across 500 sequences is uncertain. This is the core
empirical question — the experiment answers it. **APPROPRIATE UNCERTAINTY**.

## Potential Issues

**P1 — Verdict PPL threshold**: Plan says "ppl < 25" but then corrects to "< 38". Use 38
(+6% from 35.9 baseline at SEQ_LEN=128) as the near-zero-loss criterion. **NEEDS FIXING IN CODE**.

**P2 — max_gelu accumulation dtype**: torch.zeros(N_CHANNELS) → float32. Should be fine for
tracking max values in [0, ~100] range. **OK**.

**P3 — N=500 calib seqs**: Larger than previous cycles. May expose channels that are very
rarely but non-zero activated. This is desirable — larger N = stricter dead-channel criteria.

## Flags

**F1**: Code must use threshold < 38 (not < 25) for "CONFIRMED" PPL criterion. Fix this in
the implementation.

**REVIEW RESULT: APPROVED with flag F1 (fix PPL threshold in code).**
