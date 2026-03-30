verdict: APPROVE with advisories
round: 1

## R1: Premise validity
Item 138 is infrastructure (train a model), not a hypothesis test. The premise is sound:
cycle 118 reached PPL 78.6 at 10M tokens with lr=1e-4, and the model was clearly still
improving (not converged). Extending to 30M tokens with an lr sweep is the obvious next
step. The PPL < 50 target is reasonable for a 15.5M param model on WikiText-2 — GPT-2
small (117M params) reaches ~30, and this model is 7.5x smaller, so 40-50 is plausible.
Pass.

## R2: Measurement soundness
Val PPL is computed on held-out WikiText-2 validation split, same as cycle 118. The eval
loop (lines 166-175 of cycle 118) uses up to 256 val blocks — sufficient for stable PPL
estimates. The plan evaluates every 5M tokens during full training, which will show the
learning curve clearly. Checkpoint saved at best val PPL (not final) is correct practice.
Pass.

## R3: Capacity and statistical adequacy
Implementation type — single run per lr is appropriate. The 3-lr sweep is minimal but
adequate for this narrow goal. No multi-seed requirement. Pass.

## R4: Method-architecture alignment

**A1 (lr sweep at 1M vs 5M tokens).** The plan sweeps at 1M tokens to save time, but the
agenda specifies "pick best by val PPL at 5M tokens." At 1M tokens (~977 steps at batch
4 x seq 256), the model has barely exited warmup (5% warmup = ~49 steps at 1M, but cosine
schedule shape depends on total steps). With the cosine schedule set for 1M total, the lr
will have decayed significantly by the end. More critically: lr ranking at 1M tokens may
not predict lr ranking at 30M tokens. Higher lrs (5e-4) may look better early due to
faster initial descent but diverge later, while 1e-4 may look worst at 1M but best at 30M.
This is a real risk.

However: the gap between the agenda's "5M" and the plan's "1M" saves ~20 min. Given that
this is infrastructure (not a hypothesis test), the cost of picking a suboptimal lr is
just slightly higher final PPL, not a wrong scientific conclusion. And the 60M fallback
catches the failure case. Acceptable with advisory.

**A2 (30M tokens sufficiency).** WikiText-2 train has ~2.1M tokens of raw text. With
vocab 4096 and BPE, this yields roughly 2M-2.5M tokens after encoding. At seq_len=256,
that's ~8K-10K training blocks. At 30M tokens with batch 4 and seq 256, the model sees
30M / 1024 = ~29K steps, consuming ~29K x 4 = ~116K block-draws from ~10K unique blocks.
That's ~12 epochs over the data. This is fine for WikiText-2 — overfitting is the risk,
not underfitting. The val PPL eval every 5M tokens will catch overfitting.

Scaling law estimate: at 10M tokens, PPL was 78.6. Language model PPL on WikiText-2
typically follows PPL ~ C * tokens^(-alpha) with alpha ~ 0.05-0.1 for small models.
At 3x the tokens (30M): PPL ~ 78.6 * (30/10)^(-0.076) ~ 78.6 * 0.92 ~ 72. That's
still above 50. However, the lr sweep (cycle 118 used only 1e-4) could yield a large
improvement — 3e-4 or 5e-4 may converge much faster. With a better lr, PPL < 50 is
plausible but not certain. The 60M fallback is essential. Pass with advisory.

**A3 (Tokenizer adequacy).** BPE with vocab 4096 trained on WikiText-2 with Whitespace
pre-tokenizer. Vocab 4096 is small (GPT-2 uses 50K), which means more tokens per word
and potentially harder modeling. But the model was designed for this tokenizer (cycle 118),
and the question is just whether it can reach PPL 50, not compete with GPT-2. The
tokenizer is adequate for the infrastructure goal. Pass.

## R5: Known pitfall scan
- **MP5 (insufficient capacity):** 15.5M params, 4 layers, hidden_size=256, 8 experts
  top-2. Active params per token: ~5.5M (embedding + attention + 2 experts per layer).
  WikiText-2 is a relatively easy benchmark. The model should have sufficient capacity.
  Not triggered.
- **MP3 (underdetermined fitting):** Not applicable — this is standard LM training, not
  a small-sample fitting problem. ~2M unique training tokens >> model params in the
  effective sense (weight sharing across positions). Not triggered.
- **MP1, MP2, MP4, MP6, MP7, MP8:** Not applicable to LM training infrastructure.

## Advisory

**A1 (lr sweep horizon mismatch).** The plan sweeps at 1M tokens but the agenda says 5M.
1M is aggressive — lr ranking may not transfer to 30M. Mitigation: if the three lrs give
very similar PPL at 1M (spread < 5 PPL points), pick the middle value (3e-4) rather than
trusting noisy rankings. If one lr is clearly divergent (PPL > 200), eliminate it but
don't over-index on the ranking of the remaining two.

**A2 (PPL 50 may require 60M).** Back-of-envelope scaling suggests 30M tokens alone may
land at PPL 60-70 even with a better lr. The 60M fallback path is critical and should
be treated as the likely outcome, not an exceptional case. Ensure the cosine schedule
for 60M is recomputed from scratch (fresh model, fresh schedule), not extended from the
30M run.

**A3 (Overfitting risk at 30M+ tokens).** WikiText-2 is small (~2M tokens). At 30M
token-draws the model sees the data ~12x; at 60M it's ~24x. Monitor train loss vs val
PPL divergence. If train loss < 2.0 but val PPL > 60, the model is overfitting and more
tokens won't help — would need data augmentation or regularization. The plan doesn't
mention this failure mode.
