# Cycle 024 Summary — CKA Representational Similarity (GPT-2)

## Verdict: INCONCLUSIVE

## Setup
- Model: GPT-2 (12 blocks, 768 hidden dim)
- Corpus: Shakespeare (4096 tokens used for CKA)
- CKA: linear kernel, centered, Frobenius norm formulation
- 13 checkpoints: embedding + 12 block outputs (indices 0=emb, 1=block0, ..., 12=block11)

## Key Findings

### Massive representational plateau (blocks 1–10)
CKA[i,j] ≥ 0.991 for all pairs within checkpoints 2–11 (block 1 through block 10).
This is a single 10-block plateau — the residual stream barely changes its geometry
across blocks 1–10 (CKA ≥ 0.99). This is a direct consequence of GPT-2's residual
connections: each block adds a small delta to the stream, so the Gram matrices remain
nearly identical.

### Three distinct representational regimes
1. **Embedding regime** (checkpoint 0): CKA=0.826 with block 0, then ~0.17 with all others.
2. **After-block-0 regime** (checkpoint 1): CKA=0.616 with block 1, ~0.55 with blocks 1–10, then 0.17 with embedding, 0.17 with block 11.
3. **Residual plateau** (checkpoints 2–11, blocks 1–10): pairwise CKA ≥ 0.991 everywhere.
4. **LM head output** (checkpoint 12 = block 11 output): CKA ≈ 0.04–0.07 with ALL earlier layers — completely distinct.

### CKA drops between adjacent checkpoints
- emb→block0: 0.174 (large — first block transforms embedding significantly)
- block0→block1: 0.384 (largest — block 0 does the most new work)
- blocks 1–9 transitions: 0.0002–0.004 (effectively zero — plateau interior)
- block10→block11: 0.927 (extreme — block 11 breaks from the plateau)

### Spearman correlation with cycle 005 damage
Spearman(CKA_drop, damage_8x) = ρ=0.32, p=0.31 — below the 0.40 threshold.

Why it fails: CKA drops are near-zero for blocks 1–9 (all fall in the plateau
interior), yet damage varies meaningfully between these blocks (0.12%–3.07%).
The correlation is diluted because CKA cannot distinguish between blocks that
are all nearly identical in representation space. The signal does exist at the
extremes (L0 and L11 both have large drops AND notable damage) but n=12 is too
small and the interior points dominate.

## Criteria Evaluation
- Plateau criterion (≥1 group ≥3 layers with CKA ≥ 0.80): **PASS** — one plateau of size 10
- Spearman criterion (ρ ≥ 0.40): **FAIL** — ρ=0.32
- Refuted criterion (all non-adj CKA < 0.50): **NO** — max non-adj = 0.9995

## Interpretation
The massive plateau is real but expected from residual connections: each block
contributes a small incremental update to a 768-dim stream. CKA measures total
geometry and is insensitive to these small deltas. This makes CKA a poor predictor
of per-block compression difficulty. The extreme values (block 0 entry, block 11 exit)
track damage, but the middle plateau does not.

Implication: CKA plateaus identify geometrically-redundant representations but do NOT
imply that blocks within the plateau can be removed cheaply. A block can be
functionally critical (hard to remove) while leaving the Gram matrix nearly unchanged —
it computes a small but essential additive correction to the stream.

## Artifacts
- `autoresearch/results/cycle_024_cka_matrix.json`
- `autoresearch/results/cycle_024_cka_heatmap.png`
