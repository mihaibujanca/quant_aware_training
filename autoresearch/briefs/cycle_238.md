# Item
248 (P0): T4 — Activation patching across token pairs (bump-only).
On 100 (A, B) token pairs from held-out wikitext-2 val (same position, distinct top-1, both have logit margin >2), at blocks l ∈ {1,3,5,7,9}: compute Δ̃_l(A), Δ̃_l(B). Bump-only patch: x_l(A) - Δ̃_l(A) + Δ̃_l(B). Full residual patch: x_l(B). Measure how much of the (B-A) prediction shift is captured by the bump-only patch.

# Context
- Cycle 235 (T1, CONFIRMED with G3): top SVs of centred Δ̃ extracted.
- Cycle 236 (T2, INCONCLUSIVE): bump directions are causally active but T1's wte probe is wrong.
- Cycle 237 (T3, CONFIRMED): the bump SUBSPACE is causally critical (23,143× random baseline). Per-block: blocks 6 and 11 catastrophic, blocks 8/9/10 random-like.
- T4: tests whether the bump SPECIFICALLY transports token-dependent information (causal attribution to predictions).

# Dependencies
- 245 / cycle 235 (done) — for centred Δ̃ definition.

# Key Files
- experiments/bump_t1_features_cycle235.py — centring + Δ̃ pipeline.
- experiments/bump_t3_ablation_cycle237.py — eval/hook patterns.
- autoresearch/results/cycle_235_t1_features.pt — mean_delta_l per block (needed for centring).

# Constraints
- Held-out: wikitext-2 val (distinct from cycle 235 train calibration).
- Token pair sampling:
  - Sample 100 (A_seq_id, B_seq_id, position p) tuples such that:
    - both sequences have ≥ p+1 tokens
    - model's top-1 prediction at position p differs (top1(A_seq, p) ≠ top1(B_seq, p))
    - both top-1 predictions have logit margin > 2 over runner-up
- For each pair, at each block l ∈ {1, 3, 5, 7, 9}:
  - Forward A and B → cache residual stream after block l.
  - Compute centred Δ̃_l(A) and Δ̃_l(B) at position p.
  - **Bump-only patch**: x_l(A)_at_position_p ← x_l(A) - Δ̃_l(A) + Δ̃_l(B). Continue forward through blocks l+1..11 + ln_f + lm_head.
  - **Full residual patch**: x_l(A)_at_position_p ← x_l(B). Continue forward.
  - Measure: original_top1_logit (from clean A), B_top1_logit (from clean B at the patch position — actually both top-1 token ids of A and B at that position), patched_top1_logit on the B-token at the patched position.
  - bump_attribution = (patched_logit_on_B_top1 − original_logit_on_B_top1) / (full_patch_logit_on_B_top1 − original_logit_on_B_top1)
- Random direction baseline: replace Δ̃ with random vectors of the same magnitude.
- Confirmed if mean bump-attribution > 0.15 in ≥3 of 5 tested blocks.
- Refuted if bump-attribution < 0.05 in ALL 5 blocks.
- Inconclusive otherwise.

# Output filenames per item 248
- experiments/bump_t4_patching_cycle238.py
- autoresearch/results/cycle_238_t4_patching.json
- autoresearch/results/cycle_238_t4_examples.md
- autoresearch/results/cycle_238_t4_summary.md

# Handoff
Plan agent: design the script. Tricky parts:
1. Pair sampling: need to find 100 valid (A, B, p) tuples with disjoint top-1 predictions and margin > 2. Loop over all (i, j) pairs in the eval set, check positions.
2. For each pair, the patching needs to surgically modify the residual at one specific position p. Use forward_pre_hook on h[l+1] that modifies args[0][:, p, :] only.
3. Δ̃_l for ONE token at one position: redo the centring with the saved mean_delta_l and the per-token Δ at that position.
4. Wall time: 100 pairs × 5 blocks × 3 forward passes (clean, bump-patch, full-patch) = 1500 forward passes. Each takes ~50ms. Total ~75 seconds compute, plus pair sampling. Reasonable.

New script `experiments/bump_t4_patching_cycle238.py`. Output 4 files per the spec.
