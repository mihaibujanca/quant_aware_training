# Item
225 (P0): Iterative block-by-block Adam correction — sequential refinement.
BLOCKS=[1,4,7], K=512. Per-block: correct W_out via Adam in isolation, with downstream blocks already corrected.
Confirmed ≤46.0 PPL, Refuted ≥60.0, Inconclusive 46-60.

# Context
- Cycle 206 (item 217): joint Adam on W_out for ALL 3 compressed blocks, BLOCKS=[1,4,7] K=512, 1.18M params, 1000 steps → PPL 61.746→48.111 (CONFIRMED, -10.7% vs float 53.895).
- Cycle 210: extended step curve → best 46.556 at step 4000; overfit at step 8000.
- Cycle 211: 6-block joint Adam → 59.68 best at step 1500, then overfit (more params, faster overfit).
- Cycle 212: 10-block joint Adam → REFUTED (98.49 PPL).
- Cycle 213: W_out + W_fc joint Adam → 46.424 (matches W_out-only, no benefit).
- Cycle 214: analytical lstsq replacement → REFUTED (no improvement; need CE gradient).
- Hypothesis: training each block sequentially in isolation reduces effective trainable params per optimization step (only ~390K vs 1.18M), should resist overfitting and may achieve a better minimum.

# Dependencies
- None (item 226, 227, 228, 229 depend on or compare with this).
- Existing infra: experiments/adam_wout_correction_cycle206.py, experiments/adam_step_curve_cycle210.py, experiments/adam_6block_cycle211.py, experiments/adam_full_correction_cycle213.py.

# Key Files
- experiments/adam_wout_correction_cycle206.py — joint Adam baseline (3-block).
- experiments/adam_step_curve_cycle210.py — N_CALIB, eval cadence, early-stop machinery.
- aleph/lm_geometric_compression.py — topk+lstsq init.

# Constraints
- Use existing N_CALIB (200 sequences) and eval set unchanged for comparability.
- 2000 Adam steps **per block** (item 225 specification).
- Per-block: only that block's W_out/b_out is trainable; other compressed blocks frozen at their already-corrected state.
- Order: sequential along the network, block 1 → 4 → 7. Each block uses corrected upstream blocks AS the model context for forward pass; downstream blocks remain at topk+lstsq init when fitting that block (or also pre-corrected? — see plan).
- Eval cadence: every 100 steps; track best PPL per block and final PPL.
- Use ≥3 seeds (42, 137, 2024) — required by audit.
- Compare against cycle 206/210 joint Adam best 46.556.
- Run cap 30 min per cycle.

# Handoff
Plan agent: design the precise sequential update protocol. Decide whether (a) only upstream is corrected when fitting block i, or (b) downstream is also pre-corrected by joint Adam first then refined. Spec (a) is the faithful "block 1 → block 4 → block 7" reading and should be primary. Specify 3 seeds, eval cadence, output JSON schema. Compare PPL against joint baseline (46.556) and topk+lstsq init (61.746).
