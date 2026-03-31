# Item
163 — Joint training + LoRA corrections — GPT-2 FFN k=4

# Context
- Cycle 115 (item 118): topk+lstsq+500 polish, then LoRA r=8 corrections (2000 CE steps). Uncorrected PPL 80.68 → corrected PPL 59.11. Recovery 74.3%. 49K correction params.
- Cycle 149 (item 159): joint random rank-64 training. PPL 69.76 (beats topk 70.70 by 1.3%). 986K params.
- Cycle 151 (item 161): corrections are 93-97% cascade compensation (novel subspace). They reroute around damaged composition, not fix local error.
- Cycle 142: corrections on polished model WORSENED PPL (71→88.7). Corrections must be trained on unpolished model.
- Current best with corrections: topk+lstsq+500 polish → LoRA r=8 → PPL ~59.

Joint training already addresses composition (it trains all 4 blocks jointly under CE loss). If cascade compensation is what corrections mostly do (cycle 151), then joint training should pre-solve most of that — leaving corrections less work. But joint training's current PPL (69.76) is still 12 PPL above float (57.65), so there's room for corrections to help.

The question: does joint+corrections beat topk+corrections? Or does joint training already capture what corrections would add?

# Dependencies
- Cycle 149 joint training code (already run). Need to retrain the joint model from scratch (no checkpoint saved).

# Key Files
- `experiments/joint_stacking_training_cycle149.py` — fork for joint model training
- `experiments/correction_subspace_unpolished_cycle151.py` — fork for correction training protocol
- `aleph/lm_geometric_compression.py` — compression pipeline

# Constraints
- BLOCKS = [7, 5, 4, 1] (same as cycle 115/149).
- Joint model: linear factorization rank-64, random init, 2000 steps AdamW lr=1e-4 cosine (replicate cycle 149).
- Corrections: LoRA r=8 on all 4 blocks, 2000 CE steps, Adam lr=1e-4 (replicate cycle 115 protocol).
- Corrections trained on the joint model AFTER joint training completes (frozen joint model, per cycle 142/151 lesson).
- Seed 42 only.
- 200 OWT eval sequences, seq_len=128, batch_size=4.

# Handoff
Fork into `experiments/joint_lora_corrections_cycle153.py`. Two phases: (1) retrain joint random rank-64 model exactly as cycle 149, (2) freeze it, add LoRA r=8 corrections, train 2000 steps. Save JSON to `autoresearch/results/cycle_153_joint_lora_corrections.json`.
