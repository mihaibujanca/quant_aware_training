# Cycle 131 Brief — Item 88

**Item**: Learned bottleneck autoencoder for FFN compression (MOONSHOT)

**Context**: Status quo: topk+lstsq at k≤4 achieves ≈PPL 80 (stacked with LoRA +14%). Open question: can nonlinear learned autoencoders beat linear topk+lstsq? Information bottleneck principle (Tishby) predicts that at high compression (k=64, 48× ratio), surviving encoder features are irreducible structural essentials — the functional core of each block.

**Design**: Replace axis-aligned topk+lstsq with learned bottleneck. Encoder E: h ∈ R^3072 → z ∈ R^k; Decoder D: z → y ∈ R^768. Train E+D under model loss (next-token CE with bottleneck swapped into live GPT-2), not per-block MSE. This forces bottleneck to preserve downstream relevance, not just magnitude.

**Full Scope** (4 blocks × 5 k values + stacking + joint polish):
- Sweep: k ∈ {512, 384, 256, 128, 64} on blocks [1, 4, 7, 11]. Per-block training 2000 steps (lr=1e-4, OWT).
- Track: (a) AE PPL vs topk+lstsq at matched k, (b) encoder weight structure vs topk ranking, (c) encoder nonlinearity ||E(h) − E_linear(h)||/||E(h)||.
- Stacking: train k=4 AE on all 4 blocks, then joint fine-tune E+D pairs under model loss.
- Visualization: encoder top principal directions at k=64 (what survives max compression).

**One-Cycle Scope** (proof of concept):
Recommend k=384 on 2 blocks (blocks 1, 7) as POC. Full 4×5 grid is ~2–3 cycles. Single-block encoder training is fast (2000 steps); bottleneck that beats topk+lstsq by ≥5% relative PPL on even 1/2 blocks is strong signal.

**Key Files**:
- Baseline: `experiments/placement_sweep_k4_cycle125.py` (topk+lstsq reference)
- Status: `autoresearch/STATUS.md` (compression recipe, cycle 129 isotropy findings)

**Constraints**:
- Must train under model loss (not MSE) to preserve downstream relevance.
- Encoder nonlinearity vs PPL gain: if nonlinear but <3% PPL improvement → INCONCLUSIVE.
- Depends on: none.

**Handoff**: Plan should decide: full sweep vs POC (k=384 on 2 blocks)? If POC: which 2 blocks? Implement, train, evaluate AE vs topk+lstsq PPL + encoder analysis. Report encoder weight alignment to topk ranking and nonlinearity metric.
