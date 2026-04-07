# Cycle 235 T1 Summary

**Verdict:** CONFIRMED

**Headline:** 12/12 blocks have ≥5 coherent directions (coherence_label ∉ {mixed, incoherent}).

**Config:** gpt2 / wikitext-2 train split / N_SEQS=200 × SEQ_LEN=512 = 102400 tokens / seed 42 / device mps

## Per-block coherent direction counts

| block | coherent | total | labels |
|------:|---------:|------:|:-------|
| 0 | 27 | 32 | repeated:24, incoherent:3, mixed:2, stopword:1, punctuation:1, number:1 |
| 1 | 23 | 32 | repeated:22, mixed:8, capitalized:1, incoherent:1 |
| 2 | 24 | 32 | repeated:22, mixed:5, incoherent:3, stopword:2 |
| 3 | 27 | 32 | repeated:22, stopword:4, mixed:3, incoherent:2, capitalized:1 |
| 4 | 21 | 32 | repeated:18, mixed:10, number:2, capitalized:1, incoherent:1 |
| 5 | 19 | 32 | repeated:14, mixed:12, stopword:4, incoherent:1, capitalized:1 |
| 6 | 19 | 32 | repeated:16, mixed:12, stopword:3, incoherent:1 |
| 7 | 15 | 32 | mixed:16, repeated:11, capitalized:2, stopword:2, incoherent:1 |
| 8 | 17 | 32 | mixed:13, repeated:12, stopword:2, incoherent:2, capitalized:1, number:1, punctuation:1 |
| 9 | 15 | 32 | mixed:13, repeated:11, incoherent:4, stopword:3, number:1 |
| 10 | 12 | 32 | mixed:14, incoherent:6, repeated:6, capitalized:3, stopword:2, punctuation:1 |
| 11 | 10 | 32 | mixed:18, repeated:5, incoherent:4, capitalized:2, stopword:2, punctuation:1 |

## Interpretation

The centred per-token bump space contains interpretable latent features: 12/12 blocks have ≥5 of 32 top directions whose top-100 corpus tokens form a coherent heuristic class (punctuation / stopword / capitalized / number / repeated). Coherence here is a cheap surface heuristic — a direction labelled 'punctuation' means ≥40 of its top-100 corpus tokens are single-character punctuation, not that the direction causally boosts punctuation logits. T2 (item 246) is the causal test: it adds α·v_{l,i} to the residual stream and checks whether the actual logit shift matches the T1 vocab prediction. Until then, these features are a correlational feature inventory — the story is alive but unproven.
