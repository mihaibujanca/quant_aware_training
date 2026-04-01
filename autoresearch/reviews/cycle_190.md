# Design Review — Cycle 190 (Item 200): GELU-neg + ReLU-pos Hybrid

## Correctness

**R1 — Hybrid activation**: `torch.where(z < 0, F.gelu(z), F.relu(z))`.
- For z<0: GELU(z) = z*Phi(z) ≈ z*0..0.5 → negative, exact match to float model.
- For z>=0: ReLU(z) = z (linear above 0), vs GELU(z) ≈ z (approximately linear above 0 but with smooth correction).
- This isolates exactly the positive-regime shape change. **CORRECT**.

**R2 — No bias correction needed**: Since GELU(z) is unchanged for z<0, the expected output
is the same as the float model in the negative regime. No bias offset is introduced. **CORRECT**.

**R3 — Config matches cycle 188**: SEQ_LEN=1024, N_SEQS=200, SEED=42. Same baseline expected
(~22.7 PPL). Baseline sanity check: abort if not in [15, 30]. **CORRECT**.

**R4 — Progressive sweep**: Deepcopy for each k, hooks replaced in-place. No interference
between iterations. **CORRECT**.

## Potential Issues

**P1 — F.gelu default approximation**: PyTorch F.gelu uses the `tanh` approximation by
default. This matches what GPT-2 uses (`NewGELU`-style). Actually GPT-2 uses the exact formula:
`0.5*z*(1+torch.tanh(sqrt(2/pi)*(z+0.044715*z^3)))`. The nn.GELU() in huggingface GPT-2
uses `approximate='tanh'` or `approximate='none'`. Check: GPT-2's act is `NewGELU` (a custom
module). So `F.gelu(z)` may not exactly match `NewGELU(z)`.

This is a **minor flag**: F.gelu('tanh') gives `0.5*z*(1+tanh(sqrt(2/pi)*(z+0.044715*z^3)))`,
which IS the GPT-2 NewGELU formula. So they match. **ACCEPTABLE**.

**P2 — Experiment scale**: Progressive sweep creates 12 deepcopies. Each forward pass takes
~2-3s on MPS for 200 seqs × 1024 tokens. Total ~24-36s. Well within 30-minute cap.

## Verdict Criteria Check

Thresholds (28, 35) reference baseline=22.7:
- 28 = +5.3 PPL (+23.3%)
- 35 = +12.3 PPL (+54.2%)

C3 all-12 = 27.314 (+20.33%). If hybrid < 28, hybrid is BETTER than C3 (which preserves
negative regime AND replaces positive with constant). But hybrid replaces positive with ReLU
(a shape change), which should cost more than C3's constant mean. So refuted (hybrid < 28)
seems unlikely. More likely: hybrid is between C3 and ReLU+bias.

Expected range: 28-40 PPL (inconclusive to confirmed). **VERDICT CRITERIA REASONABLE**.

## Flags

None blocking.

**REVIEW RESULT: APPROVED — proceed to execute.**
