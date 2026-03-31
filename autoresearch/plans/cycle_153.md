# Cycle 153 Plan: Joint Training + LoRA Corrections (Item 163)

## Task
Train LoRA r=8 corrections on top of a jointly-trained compressed model (cycle 149 protocol). Compare against topk+corrections baseline (cycle 115). The question: does joint training reduce the correction burden?

## Hypothesis
Joint training addresses inter-block composition (it trains all 4 blocks simultaneously under CE). Cycle 151 showed corrections are 93-97% cascade compensation. If joint training already handles cascade, corrections should have less to do — and the gap between joint+corrections and topk+corrections should be small. Alternatively, joint training may provide a better starting point (69.76 vs 80.68 uncorrected) that corrections can exploit further.

## Pre-flight
- P1 (duplicate): not a duplicate. Cycle 115 tested corrections on topk+polish. Cycle 149 tested joint training alone. This is the combination.
- P2 (deps): item 159 (joint training) was already run in cycle 149. We retrain the model (no checkpoint saved) using the same script.
- P3 (scope): seed 42 only, ~20 min for correction training + ~15 min for joint model retraining = ~35 min total.
- P4 (null baseline): float PPL ~57.65. Joint uncorrected ~69.76. Topk+corrections ~59.11. If joint+corrections PPL > 69 (corrections made it worse), training is broken. If < 55 (below float), bug.
- P5 (param accounting): joint model 986K params + corrections 4 blocks × 2 × 768 × 8 = 49,152 params. Total 1,035K. Topk baseline uses 1,576K + 49K corrections = 1,625K. Joint+corrections is actually fewer total params.

## Method

### Step 0: Setup
Fork `experiments/joint_stacking_training_cycle149.py` to `experiments/joint_lora_corrections_cycle153.py`. Add correction training phase from cycle 151's `LowRankCorrection` + `train_corrections()`.

### Step 1: Load model, eval float PPL
- Load GPT-2 from HF. Eval float PPL on 200 OWT sequences. Abort if > 60.
- Tokenize probe corpus (same as cycle 149).

### Step 2: Retrain joint model
- Replace blocks [7, 5, 4, 1] MLPs with `LinearFactorFFN(original_mlp, d_bn=64, init_mode="random")`.
- Train 2000 steps jointly, AdamW lr=1e-4, cosine schedule (exact cycle 149 protocol).
- Eval PPL. Should reproduce ~69.76 ± 1.5.
- Record as `joint_ppl_pre_correction`.

### Step 3: Train LoRA corrections on frozen joint model
- Freeze entire model (including joint-trained encoder/decoder params).
- Add `LowRankCorrection(768, 8)` after each compressed block (4 corrections).
- Hook placement: `block.register_forward_hook` on `model.transformer.h[bid]`, same as cycle 151.
- Train 2000 CE steps, Adam lr=1e-4 (not AdamW — match cycle 115 protocol).
- Log loss every 500 steps.
- Eval PPL with corrections active. Record as `joint_corrected_ppl`.

### Step 4: Topk+corrections control
- Build topk+lstsq compressed model on blocks [7, 5, 4, 1] at 6x. Polish 500 steps.
- Eval uncorrected topk PPL.
- Train LoRA r=8 corrections on the polished topk model, same protocol (2000 CE steps, Adam lr=1e-4).
- Eval corrected topk PPL. Should reproduce cycle 115 ~59 ± 2.

### Step 5: Save results + verdict

Output JSON:
```json
{
  "cycle": 153,
  "item": 163,
  "seed": 42,
  "float_ppl": ...,
  "joint": {
    "uncorrected_ppl": ...,
    "corrected_ppl": ...,
    "recovery_pct": ...,
    "model_params": 986112,
    "correction_params": 49152
  },
  "topk": {
    "uncorrected_ppl": ...,
    "corrected_ppl": ...,
    "recovery_pct": ...,
    "model_params": ...,
    "correction_params": 49152
  },
  "verdict": "...",
  "verdict_detail": "..."
}
```

Recovery computed as: `(uncorrected - corrected) / (uncorrected - float) * 100`.

Verdict logic (from agenda):
- **Confirmed** if joint+corrections beats topk+corrections by >= 3% relative PPL.
- **Refuted** if within 1.5%.
- **Inconclusive** if 1.5-3%.

## Implementation Notes

### LowRankCorrection class
Copy from cycle 151: `C(x) = x + U @ (V @ x)`, U: (768, 8), V: (8, 768). Zero-init U, small random V (0.01 scale). This gives identity at init.

### Hook placement
Corrections go on the transformer block (not the MLP). The hook catches the full residual stream output of the block `model.transformer.h[bid]`. This is the same hook point as cycle 115 and cycle 151 — corrections act on the residual stream after the block's attention + FFN + residual connections.

### Joint model retraining
We retrain from scratch rather than loading a checkpoint because cycle 149 did not save model weights. The script is deterministic (seed 42), so the result should reproduce. If the reproduced joint PPL differs by > 2 from 69.76, flag it but continue.

### No polish for joint model
The joint model is trained end-to-end under CE loss — this IS the polish. Do NOT add a separate polish phase. Corrections train on the joint model as-is. This matches the cycle 142/151 insight: corrections work on unpolished models, not polished ones.

## Output
- `experiments/joint_lora_corrections_cycle153.py`
- `autoresearch/results/cycle_153_joint_lora_corrections.json`

## Validation
- Float PPL: 57.65 ± 1. Abort if > 60.
- Joint uncorrected PPL: 69.76 ± 1.5 (reproduce cycle 149). If > 73, retraining failed.
- Joint corrected PPL: expected range [57, 67]. If > 69 (no improvement), corrections are ineffective.
- Topk corrected PPL: should reproduce cycle 115 ~59 ± 2.
- Correction params: assert 49,152 (4 × 2 × 768 × 8).

## Fallbacks
- If joint model retraining fails to reproduce (PPL > 73): check that init_mode="random" and cosine schedule match cycle 149 exactly. Fall back to SVD init if needed (but expect worse results per cycle 149).
- If corrections on joint model make PPL worse: this would mirror the cycle 142 failure (corrections on polished model). Joint training is effectively a polish — this would challenge the "unpolished only" rule. If this happens, try: (a) lower lr (5e-5), (b) fewer steps (500), (c) corrections on the joint model at step 1000 (before convergence).

## Wall time estimate
- Model load + float eval: ~2 min.
- Joint model retraining (2000 steps): ~12 min.
- Topk baseline + polish: ~8 min.
- Two correction trainings (2 × 2000 steps): ~2 × 10 min = ~20 min.
- Total: ~42 min.
