# Cycle 055 Plan: Jacobian Frobenius Std Corpus-Sensitivity — GPT-2 FFN

## Hypothesis
P1 (Jacobian Frobenius std) block ranking is an intrinsic geometric property,
not corpus-dependent. Spearman(P1_shakespeare, P1_wikitext) >= 0.85.

## Method
Fork `experiments/gpt2_local_complexity.py` into
`experiments/gpt2_p1_corpus_sensitivity.py`. Changes:

1. Load WikiText-2 from `data/wikitext2_valid.txt` instead of Shakespeare.
2. Tokenize, extract 128 probe tokens (same N_PROBES=128, K_NEIGHBORS=5).
3. Compute P1 (Frobenius std) for all 12 blocks — identical Jacobian code.
4. Load Shakespeare P1 values from
   `autoresearch/results/cycle_048_local_complexity_gpt2.json`.
5. Compute Spearman rank correlation between the two P1 vectors (n=12).
6. Save JSON to `autoresearch/results/cycle_055_p1_corpus_sensitivity.json`:
   - `p1_wikitext` (12 values), `p1_shakespeare` (12 values from cycle 048)
   - `spearman_rho`, `spearman_p`
   - `outcome`: "confirmed" / "refuted" / "inconclusive"
7. Bar chart: P1 per block, Shakespeare vs WikiText-2 side by side.
   Save to `autoresearch/results/cycle_055_p1_corpus_sensitivity.png`.

## Pre-flight Checks
- P1: data file exists (`data/wikitext2_valid.txt`) — verified
- P2: cycle 048 JSON exists with 12 P1 values — verified
- P3: GPT-2 weights downloadable (HuggingFace, cached) — standard
- P4: Jacobian computation fits in CPU RAM (128 probes, k=5) — proven in cycle 048

## Success Criteria
- Confirmed: Spearman rho >= 0.85 (corpus-invariant oracle)
- Refuted: Spearman rho < 0.5 (corpus-dependent oracle)
- Inconclusive: 0.5 <= rho < 0.85

## Risks
- WikiText-2 valid set may be short; ensure >= 128 usable tokens (it has ~36K tokens, fine).
- Token distribution shift could flip a few middle layers without changing overall rank — watch for block 0/11 dominance inflating rho.

## Output Files
- `experiments/gpt2_p1_corpus_sensitivity.py`
- `autoresearch/results/cycle_055_p1_corpus_sensitivity.json`
- `autoresearch/results/cycle_055_p1_corpus_sensitivity.png`
