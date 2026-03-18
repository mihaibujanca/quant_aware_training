# Cycle 031 Brief — Item 46: Grokking as Compression Catalyst (Modular Arithmetic v2)

## Selected Item
**Item 46** — Grokking as compression catalyst — modular arithmetic (v2, supersedes item 39)

## Context
- Cycle 021 (item 39) used spirals, which don't grok, and raw pattern count as metric — both wrong.
- This redesign uses canonical grokking: (a+b) mod 97, AdamW WD=1.0, tracks region-label purity.
- Script already exists: `experiments/grokking_compression_catalyst_v2.py` (untracked, has bugs).
- Prior cycle 021 result: refuted (pattern count *increased* through grokking, no compression gain).

## Key Prior Findings
- Cycle 027 confirmed centroid activation similarity (ρ=0.85) is the right channel-merge signal.
- Cycle 026 confirmed geometric channel selection beats SVD for GPT-2 FFN compression.
- No grokking signal was tested on the correct task (mod arithmetic) — open question.

## Known Script Issues
1. Lines 315-318: dead code calls `eval_acc(None, X_test, y_test)` which will crash.
2. Output file names say `cycle_021` — must be `cycle_031`.
3. Both fixable with targeted edits.

## Hypothesis
Post-grokking teacher encodes algebraic structure (not a lookup table), so a 4× smaller student
can distill it more accurately than a pre-grokking teacher. Measured by: student test accuracy
after distillation at both memorization checkpoint and generalization checkpoint.

## Experiment Design
- Task: (a+b) mod 97, 9409 total pairs, 40% train
- Teacher: 1-hidden-layer ReLU, hidden=128, AdamW LR=1e-3 WD=1.0
- Grokking criterion: train acc >99% (memorized), then test acc >80% (grokked)
- Student: hidden=32 (4× compression), distilled 5000 epochs on training set
- Metrics: student test acc pre vs post grokking, region-label purity, pattern count

## Confirmed if
Grokking detected AND post-grokking student test accuracy exceeds pre-grokking student by ≥10%.

## Refuted if
No grokking after 50K epochs, OR grokking detected but compression gain <2%.
