# Cycle 234 Audit — Item 244 (gelu_noneg + μ_l + LoRA r=8 W_out residual)

**Status: CORRECTED (CONFIRMED → INCONCLUSIVE — data leakage per G6)**

## Verdict check
- Criteria: final_ppl ≤ 48.0 → CONFIRMED; ≥ 58.0 → REFUTED; else INCONCLUSIVE.
- Reported: final_ppl = 32.411 @ best step 2000, monotone descent from init 61.664.
- Literally 32.411 ≤ 48.0 → CONFIRMED.
- BUT the "eval" that produces 32.411 is computed on the **same** 200 wikitext-2
  sequences used to train the LoRA adapters, with 368,640 trainable params against
  ~25,400 eval tokens. The executor flagged this explicitly in the manifest
  (`CAVEAT_calib=eval`), and the summary itself admits "the reported 32.411 PPL is
  a calibration-set PPL, not a held-out generalization measure". Per G6: a verdict
  threshold defined over held-out generalization cannot be met by an in-distribution
  training loss — even if it literally passes the numeric cutoff.

## Root cause — calib == eval

`experiments/gelu_noneg_lora_cycle234.py`:

- L57–65: builds a single list `all_ids` with the first 200 wikitext-2 validation
  sequences of length 128 and binds `eval_ids = [t.to(DEVICE) for t in all_ids[:N_EVAL]]`.
  There is no separate `calib_ids`.
- L270–277 (Phase 4 training loop):
  ```
  for step in range(1, STEPS + 1):
      idx = (step - 1) % n_data
      ids = eval_ids[idx]
      out = model(ids, labels=ids)
      loss = out.loss
      opt.zero_grad(); loss.backward(); opt.step()
  ```
  Training iterates `eval_ids` (`n_data = len(eval_ids)` = 200) and CE-optimises
  LoRA A, B directly against those same sequences.
- L281–290: every 200 steps, `eval_ppl(model, eval_ids)` is called on the **same**
  list. `best_state` is snapshotted against this metric; L295 restores the best
  state and L296 recomputes `final_ppl = eval_ppl(model, eval_ids)`. Training
  signal, model selection criterion, and final reported metric all live on the
  identical 200 sequences.
- 2000 training steps × 200 sequences = 10 full passes over the "eval" set against
  a parameter budget of 368,640 (12 blocks × (3072×8 + 8×768)), with ~25,400 target
  tokens. A rank-8 residual with 3840 params per block pointed at 2,117 tokens of
  gradient signal per block-pass has plenty of capacity to memorise the calibration
  slice.

## Evidence the result is overfitting, not recovery

- Final PPL 32.411 is **12.65 PPL BELOW the float baseline 45.064** on the same
  sequences. The LoRA drives the model past the unquantised teacher — the only way
  a residual correction on a cheaper (μ + relu(GELU)) activation surface beats the
  float network is by memorising token-level structure in the calibration slice.
- The training trajectory (61.664 → 42.724 → 39.474 → 37.725 → 36.568 → 35.461 →
  34.808 → 34.054 → 33.480 → 32.883 → 32.411) is **still descending at step 2000**
  with no plateau — typical of capacity-unbounded in-distribution fit, not a
  held-out recovery floor.
- 368K trainable params / 25.4K eval tokens ≈ 14.5 params per supervised token.

## Contrast with cycles 220 and 228 (same data split, but safe)

- Cycle 220 (`surgical_bump_removal_cycle220.py`): no training, only deterministic
  activation patching + eval. `eval_ids == all_ids[:200]` is a single held-out set;
  there is nothing being fit against it.
- Cycle 228 (`gelu_noneg_corrected_cycle228.py`): no training. μ_l is a
  deterministic statistic (mean over the calibration forward pass). Using the same
  200 sequences to compute μ and then to eval is a mild form of in-distribution
  measurement, but no optimiser touches the parameters — the measurement does not
  degrade when the eval set shifts (its bias is bounded by the population statistic).
- Cycle 234 **inherits this protocol verbatim** but then introduces an Adam
  optimiser against the same 200 sequences. This is the step where calib==eval
  becomes a method failure — a gradient-based trainable surface against the eval
  distribution produces an in-distribution fit, not a correction.

## Contrast with the safe pattern (cycles 206/210/223/225/230)

`experiments/adam_wout_correction_cycle206.py` (L19–48):

```
N_CALIB = 200
N_EVAL = 100
...
if len(all_ids) >= N_CALIB + N_EVAL:
    break
calib_ids = [t.to(DEVICE) for t in all_ids[:N_CALIB]]
eval_ids  = [t.to(DEVICE) for t in all_ids[N_CALIB:N_CALIB + N_EVAL]]
```

`lstsq` fit + Adam training both run on `calib_ids`; `eval_ppl` runs exclusively on
the disjoint `eval_ids`. Cycles 210, 223, 225, 230 all follow this pattern — it is
the repo-standard safe split. Cycle 234 is the outlier.

## Decision (G6)

- Downgrade CONFIRMED → INCONCLUSIVE. The 32.411 PPL does not test the hypothesis
  "LoRA r=8 absorbs the higher-order GELU curvature residual"; it tests
  "LoRA r=8 with 368K params can overfit 200 wikitext-2 sequences", which is
  trivially yes.
- The three claims the executor listed as "established" — (1) LoRA r=8 absorbs the
  16.6 PPL gap, (2) the residual lives in a rank-≤8 subspace of W_out, (3) the
  optimizer is friendly — are all **in-distribution** claims and cannot be
  generalised without a held-out eval.

## Recommendations

- Rerun with disjoint calib/eval split (e.g., `all_ids[:200]` / `all_ids[200:300]`
  like cycles 206/223/230). Expected held-out PPL: somewhere in
  (45.064, 61.664) — if it lands near 45, the LoRA recovery is genuine and the
  hypothesis upgrades back to CONFIRMED; if it lands near 61.664, the LoRA has
  nothing to do with the residual and the hypothesis downgrades to REFUTED.
- Before any future training-based cycle reuses cycles 220/228 scaffolding, the
  executor should replace the single `eval_ids` bind with a `calib_ids`/`eval_ids`
  split (or lengthen `all_ids` to N_CALIB+N_EVAL seqs).
- Optional ablations once the held-out split is in place: rank sweep r ∈ {1, 2, 4,
  16}; SVD of learned A@B per block.

## Bookkeeping changes

- agenda.md: Cycle 234 PAST entry → INCONCLUSIVE (corrected from CONFIRMED, data
  leakage).
- cycle_234_manifest.txt: outcome → inconclusive; key_metric retained with
  `CAVEAT_calib=eval` flag.
- experiments.tsv row 234: outcome confirmed → inconclusive; description updated.
- flagged.jsonl: appended audit entry with guardrail G1, G6.
- autoresearch/results/cycle_234_audit.txt: written.
