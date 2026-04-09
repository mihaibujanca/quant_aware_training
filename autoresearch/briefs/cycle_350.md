# Cycle 350 Brief — Item 359: MLP Adapter Hidden-Size Sweep on GPT-2 b2 topk+lstsq

## Motivation
Cycle 347 confirmed MLP adapter (h=64) improves LASER b2: PPL 50.09 → 48.51 (246K params).
Cycle 348 confirmed topk+lstsq b2 alone: PPL 48.47.
Item 361 asks whether adapter gain is generic; item 359 first maps the Pareto frontier (params vs PPL).

## Question
What is the minimal adapter hidden size that yields PPL < float baseline (53.89) on topk+lstsq b2?
Is the relationship monotonic: bigger h → lower PPL, or does it saturate?

## Method
- topk+lstsq b2, same compression recipe as cycle 348 (k=512 of 3072, HIDDEN_FRACTION=1/6)
- Add MLP adapter: x → x + W2·ReLU(W1·x + b1) + b2, hidden dim h ∈ {16, 32, 64, 128}
- Training: freeze compressed MLP + all GPT-2 weights; train adapter + lstsq proj 500 steps, LR=1e-4
- Evaluation: 3 seeds, eval PPL on wikitext-2-raw-v1 validation (100 seqs, seq_len=128)
- Report: h, n_params, mean PPL ± std

## Expected outcome
- All h should beat topk+lstsq alone (48.47) given adapter adds capacity
- Possible saturation at h≥64; h=16 may be sufficient
- Pareto frontier: smallest h crossing below some threshold

## Verdict thresholds
- CONFIRMED: at least one h gives PPL < 53.89 (float) with clear Pareto structure visible
- REFUTED: adapter never improves topk+lstsq baseline (48.47) at any h
- INCONCLUSIVE: improvement exists but no Pareto structure (all similar or non-monotonic)
