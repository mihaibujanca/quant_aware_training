# Task
Cycle 238 / item 248 (P0, T4): activation patching across token pairs (bump-only). Build `experiments/bump_t4_patching_cycle238.py` measuring how much of the (B−A) prediction shift at a patched position is captured by swapping only the centred Δ̃ bump at blocks l ∈ {1,3,5,7,9}, using `cycle_235_t1_features.pt` for `mean_delta_l` centring.

# Hypothesis
Verbatim from agenda item 248:
> T4: Activation patching across token pairs (bump-only). On 100 token pairs (A, B) sampled from held-out wikitext-2 such that: (1) A and B occur at the same sequence position in distinct sequences, (2) the model's top-1 prediction for A ≠ top-1 for B, (3) both predictions have logit margin >2 over the runner-up. For each pair, at each block l ∈ {1,3,5,7,9}: Compute residual stream after block l for both A and B → x_l(A), x_l(B). Compute centred bump residual: Δ̃_l(A) = (GELU_FFN_l(A) − ReLU_FFN_l(A)) − μ_l, similarly for B. Bump-only patch: x_l_patched = x_l(A) − Δ̃_l(A) + Δ̃_l(B); continue forward. Full residual patch: x_l_patched = x_l(B); continue forward. Measure: original_top1_logit, B_top1_logit, patched_top1_logit at the final position. Bump attribution ratio = (patched_logit − original_logit) / (full_patch_logit − original_logit), evaluated on B's top-1 token. Save 20 concrete example records per block. Confirmed if mean bump-attribution ratio across 100 pairs > 0.15 in ≥3 of the 5 tested blocks. Refuted if bump-attribution < 0.05 in ALL 5 blocks. Inconclusive otherwise.

# Pre-flight Results
- P1 deps: clean — cycle 235 done, `cycle_235_t1_features.pt` on disk with per-block `mean_delta` (768,).
- P2 overwrite: clean — `cycle_238_*` filenames fresh.
- P3 patterns: centring from `bump_t1_features_cycle235.py`; block-level hook surgery from `bump_t3_ablation_cycle237.py`.
- P4 budget: 100 pairs × 5 blocks × 3 forwards ≈ 1500 short forwards + per-block x_l cache passes. Est. 2–4 min.

# Steps
1. New `experiments/bump_t4_patching_cycle238.py`. SEED=42, DEVICE=mps/cpu, N_SEQS=200, SEQ_LEN=128, BLOCKS=[1,3,5,7,9], MARGIN_MIN=2.0, N_PAIRS=100, POSITIONS=[32,64,96].
2. Load HF `gpt2` eval + tokenizer; freeze params. Load `cycle_235_t1_features.pt` → `mean_delta[l]=features[f"block_{l}"]["mean_delta"]` for l in BLOCKS → DEVICE.
3. Held-out corpus: wikitext-2 **validation** split; tokenise+concat+chunk → `eval_ids` (200,128), seed 42 (distinct from cycle 235 train calibration).
4. **Pair sampling** (target 100 `(i,j,p,A_top,B_top)`): single batched `no_grad` forward → `logits (200,128,V)`. For each p ∈ POSITIONS: `top2=logits[:,p].topk(2,-1)`, `top1[i]`, `margin[i]=top2.values[:,0]-top2.values[:,1]`. Pool: for `i<j` if `top1_p[i]!=top1_p[j]` and both margins `>MARGIN_MIN`, append `(i,j,p,top1_p[i],top1_p[j])`. Shuffle with seed 42, take first 100. If `<100`, see Fallbacks.
5. **Δ̃ helper** `compute_delta_tilde(seq_ids,l,p)→(768,)`: forward `seq_ids` with `forward_pre_hook` on `h[l].mlp.c_fc` capturing `inputs[0][:,p,:]` (LN2 output @ p). Compute `preact = h @ W_fc + b_fc`; `delta_pre = gelu(preact) - relu(preact)`; `delta = delta_pre @ W_proj` (proj bias cancels); return `delta - mean_delta[l]`. Precompute `(W_fc,b_fc,W_proj)` per l.
6. **x_l cache**: per block l, install `forward_hook` on `h[l]` storing `output[0][:,p,:]` keyed by `(seq_idx,p)`. One batched forward of all 200 eval sequences per block → `cached_x[l]` with all 3 positions. 5 total passes.
7. **Patching** per pair `(A_seq,B_seq,p,A_top,B_top)` × block l:
   a. `clean_logits_Ap = logits[A_seq,p]`, `original_top = A_top`, `orig_Btop = clean_logits_Ap[B_top]`.
   b. `dA = compute_delta_tilde(eval_ids[A_seq:A_seq+1],l,p)`, `dB = compute_delta_tilde(eval_ids[B_seq:B_seq+1],l,p)`. `x_lA=cached_x[l][A_seq,p]`, `x_lB=cached_x[l][B_seq,p]`.
   c. Bump vec `v_bump = x_lA - dA + dB`; full vec `v_full = x_lB`.
   d. For each `v ∈ {v_bump,v_full}`: install `forward_pre_hook` on `h[l+1]` that sets `args[0][:,p,:] = v` for one forward; forward `eval_ids[A_seq:A_seq+1]`; read `logit_on_Btop = out[0,p,B_top]`, `top = out[0,p].argmax()`; remove hook.
   e. `bump_logit_delta = bump_Btop - orig_Btop`; `full_logit_delta = full_Btop - orig_Btop`; `bump_attribution = bump_logit_delta/full_logit_delta` (guard `|full_logit_delta|<1e-6` → record null, skip).
   f. Record `(seq_i,seq_j,p,A_top,B_top,original_top,bump_top,full_top,bump_attribution,bump_logit_delta,full_logit_delta,orig_Btop,bump_Btop,full_Btop)`.
8. Aggregate per block: `mean_bump_attribution[l]` (excluding nulls), `median`, `frac_positive`, `frac_gt_0.15`.
9. Verdict: CONFIRMED if `mean_bump_attribution[l] > 0.15` in ≥3 of 5; REFUTED if `< 0.05` in all 5; else INCONCLUSIVE.
10. Write `cycle_238_t4_patching.json` (config, per-pair records, per-block aggregates, verdict); `cycle_238_t4_examples.md` (20 records/block: `"In ctx (last 50 chars of A): '...' A→tok(A_top); B (last 50): '...' →tok(B_top). Bump(A→B) @ block l: A→<str>, logit on B_top <orig>→<bump>, full <full>. Attribution: <pct>%."`); `cycle_238_t4_summary.md` (verdict + per-block table). Print verdict.

# Validation
- `bump_attribution` should lie in [0,1] for most pairs; record out-of-range fraction.
- `full_Btop > orig_Btop` for majority; else denominator meaningless — flag `frac_full_positive`.
- **Identity sanity**: for 5 random pairs/block, run bump with `v_bump=x_lA` (no swap) → logits must match clean within 1e-4, `bump_attribution≈0`.
- Restore sanity: after each patched forward, `len(h[l+1]._forward_pre_hooks)==0`.
- Shapes: `dA.shape==(768,)`, `x_lA.shape==(768,)`.

# Artifacts
- `experiments/bump_t4_patching_cycle238.py`
- `autoresearch/results/cycle_238_t4_patching.json`
- `autoresearch/results/cycle_238_t4_examples.md`
- `autoresearch/results/cycle_238_t4_summary.md`
- `autoresearch/results/cycle_238_manifest.txt` (execute)

# Fallbacks
- <100 pairs at MARGIN_MIN=2.0: relax to 1.5, then expand POSITIONS to {16,32,48,64,80,96,112}. Record final config.
- Walltime tight: N_PAIRS=50; record `n_pairs` in config.
- `h[l+1]` forward_pre hook: GPT2Block takes `hidden_states` as `args[0]` tensor; verify `type(args[0])` on first call.
- Centring drift: if per-block mean of a few eval seqs differs >5× L2 from `mean_delta[l]`, halt (calibration leak).
- Many `|full_logit_delta|<1e-6`: resample with stricter A_top≠B_top logit separation.

# Update bookkeeping
- Move item 248 from NEXT/P0 to PAST under `### Cycle 238` in `agenda.md` (DELETE from NEXT; short summary in PAST).
- Append row to `autoresearch/logs/experiments.tsv`: cycle 238, item 248, verdict, per-block mean bump_attribution.
- Commit on `geometric_compression`: `[autoresearch] cycle 238: T4 bump-only patching — <verdict>`. Refresh `STATUS.md`.
