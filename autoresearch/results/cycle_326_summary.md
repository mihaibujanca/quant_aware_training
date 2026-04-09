# Cycle 326 — Item 339: Encoder/Decoder Boundary vs b2→b3 Eff_Rank Drop

**Verdict: REFUTED-no-correlation**

## Setup

Checkpoint: `13L_640d_322_7k_wd70_batch20k_best.npz` (13 blocks, 640 dim, ReLU²/LeakyReLU² with slope=0.5)

## Encoder/Decoder Architecture

| Parameter | Value |
|---|---|
| num_layers | 13 |
| num_encoder_layers (config) | 0 → auto = 13 // 2 = **6** |
| Encoder blocks | 0–5 |
| Decoder blocks | 6–12 |
| Boundary (first decoder block) | **block 6** |

## Skip Connection Norms

`skip_weights` shape: (6, 640) — one 640-dim scale vector per encoder block.

| Skip | Frobenius Norm |
|---|---|
| skip[0] | 13.09 |
| skip[1] | 14.24 |
| skip[2] | 12.40 |
| skip[3] | 7.84 |
| skip[4] | 15.68 |
| skip[5] | **31.45** |

Skip[5] (connecting encoder block 5 → decoder) is notably stronger (31.45 vs avg ~12.6 for others), but all skips are within the encoder range (blocks 0–5).

## Eff_Rank Profile (from Cycle 318)

| Block | Eff_Rank | Region |
|---|---|---|
| 0 | 187.61 | encoder |
| 1 | 160.08 | encoder |
| **2** | **86.05** | **encoder** |
| **3** | **4.89** | **encoder ← b2→b3 collapse** |
| 4 | 38.30 | encoder |
| 5 | 39.96 | encoder |
| **6** | 28.21 | **decoder (boundary)** |
| 7 | 23.23 | decoder |
| 8 | 30.78 | decoder |
| 9 | 28.85 | decoder |
| 10 | 21.55 | decoder |
| 11 | 13.40 | decoder |
| 12 | 6.94 | decoder |

## Analysis

The b2→b3 eff_rank collapse (86.05 → 4.89, ~94% drop) occurs at block 3 — **3 blocks before the encoder/decoder boundary at block 6**. Block 3 is deep inside the encoder, not at or near any architectural boundary.

No structural transition occurs between blocks 2 and 3:
- Both are encoder blocks with skip connections
- mlp_mult_per_layer at block 3 = 3 (same 3× mult as blocks 0–6)
- No attention pattern change at this boundary

The recovery at block 4 (4.89 → 38.30) also contradicts an architectural explanation: if the drop were due to a structural boundary, it would not recover at the very next block.

## Conclusion

**REFUTED**: The b2→b3 eff_rank collapse does not coincide with the encoder/decoder boundary. The boundary is at block 6; the collapse is at block 3. The collapse is likely a training dynamics artifact (rank collapse in the middle of deep encoder training, quickly self-correcting at block 4).

## References

- Cycle 318: eff_rank census for 13L_640d_322 (`cycle_318_pgolf_eff_rank.json`)
- Config source: `13L_640d_322_7k_wd70_batch20k_summary.json`
