# Task
Cycle 239 / item 249 (P0): rerun cycle 234 (`gelu_noneg + μ_l + LoRA r=8` W_out residual) with a **disjoint calib/eval split** (`calib_ids = all_ids[:200]`, `eval_ids = all_ids[200:300]`) to close the cycle 234 data-leakage loop (audit in `autoresearch/audits/cycle_234.md`). NEW correction; no dependencies beyond the cycle 228 μ recipe and the cycle 234 LoRA scaffolding.

# Hypothesis
Item 249 verbatim: "Cycle 234 rerun with disjoint calib/eval split — is the LoRA residual real? Re-execute `experiments/gelu_noneg_lora_cycle234.py` with the cycle 206/210/223/225/230 safe split. Replace the `eval_ids = all_ids[:200]` bind with `calib_ids = all_ids[:200]` and `eval_ids = all_ids[200:300]` (extend the wikitext-2 val loader so `all_ids` has ≥300 sequences at SEQ_LEN=128, seed 42). μ_l, LoRA init, and the Adam training loop (LR=1e-4, 2000 steps, EVAL_EVERY=200, best-state restore) run exclusively against `calib_ids`; every PPL print, the best-state selection metric, and the final `final_ppl` MUST be computed on `eval_ids`. Keep LoRA config identical to cycle 234: rank r=8, all 12 c_proj blocks, A ∈ R^(3072,8) init N(0, 0.01), B ∈ R^(8,768) init zero, 368,640 trainable params. Sanity: LoRA-zero init on eval must reproduce cycle 228 μ-only eval PPL (held-out equivalent ≈ 61.66 or nearby; if it drifts, the splits are wrong)."

Verdict thresholds: CONFIRMED ≤ 48.0, REFUTED ≥ 58.0, else INCONCLUSIVE. Expected landing: (45.064, 61.664) per audit.

# Pre-flight Results
- P1 (novelty): NEW. No prior cycle ran gelu_noneg + μ_l + LoRA r=8 with disjoint calib/eval. Cycle 234 had calib==eval (data leakage).
- P2 (dependencies): clean. Cycle 228 μ recipe on disk; cycle 234 has LoRA scaffolding; cycles 206/210/223/225/230 are the canonical safe-split template.
- P3 (audit): cycle 234 explicitly recommends this follow-up.
- P4 (budget): 2000 Adam steps × ≤3 seeds × 200 calib seqs is within prior cycle 234 wall time.

# Steps
1. Create `experiments/gelu_noneg_lora_disjoint_cycle239.py` by copying `experiments/gelu_noneg_lora_cycle234.py` verbatim, then apply fixes below.
2. **Data split (THE FIX)**: set `N_CALIB=200`, `N_EVAL=100`. In the wikitext-2 val loader, change the stop condition from `len(all_ids) >= N_EVAL` to `len(all_ids) >= N_CALIB + N_EVAL` (i.e. ≥300 seqs at SEQ_LEN=128). Then bind `calib_ids = [t.to(DEVICE) for t in all_ids[:N_CALIB]]` and `eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB:N_CALIB+N_EVAL]]`. Assert `len(calib_ids) == 200 and len(eval_ids) == 100`; assert IDs disjoint by comparing object identity sets (`set(map(id, calib_ids)).isdisjoint(set(map(id, eval_ids)))`).
3. **Fallback inside loader**: if wikitext-2 val cannot yield ≥300 seqs at SEQ_LEN=128, drop SEQ_LEN to 64 (doubles sequence count) and log the fallback in the JSON.
4. **μ precomputation on calib only**: reuse cycle 234 Phase 1 verbatim but loop over `calib_ids` (not `eval_ids`). `mu_l = (gelu(preact) - relu(preact)).mean(dim=0)` computed per block from concatenated calib activations. Assert μ ≤ 0 elementwise.
5. **Patch all 12 blocks** with `GeluNonegCorrected(μ_l)` (cycle 234 Phase 2 class verbatim).
6. **LoRA install**: reuse `LoraConv1D` from cycle 234 Phase 3 verbatim (rank r=8, A init `randn * 0.01`, B init zero, base weight/bias frozen). Wrap all 12 `mlp.c_proj`. Freeze everything; re-enable A, B per block. Assert `n_trainable == 12 * (3072*8 + 8*768) == 368640`.
7. **Baselines**: compute `baseline_ppl_eval = eval_ppl(model, eval_ids)` BEFORE any patching (float ref on eval). After patching + LoRA-zero install, compute `init_ppl_eval = eval_ppl(model, eval_ids)` — this is the μ-only held-out PPL; no assert (audit says it may drift from the 61.664 calib value).
8. **Training loop (Adam on calib)**: 2000 steps, `idx = (step-1) % len(calib_ids)`, `ids = calib_ids[idx]`, standard `out = model(ids, labels=ids); loss.backward(); opt.step()`. LR=1e-4.
9. **Eval every 200 steps on eval_ids**: `ppl = eval_ppl(model, eval_ids)`. Best-state snapshot (per-block (A, B) CPU clones) when `ppl < best_ppl`.
10. **Restore best state, recompute `final_ppl = eval_ppl(model, eval_ids)`**. Assert `abs(final_ppl - best_ppl) < 1e-3`.
11. **Seeds**: run the full Phase-4 loop under 3 seeds {42, 137, 2024} if wall time allows; otherwise SEED=42 only and flag `single_seed=true` in the JSON with justification. Phase 1–3 (μ, patching, LoRA install) can be shared across seeds; only Phase 4 resets A via `torch.manual_seed(s)` before re-initialising A with `randn * 0.01`, B=0, and a fresh optimiser. Report `final_ppl` mean/std/per-seed.
12. **Verdict** per item 249: CONFIRMED if mean `final_ppl` ≤ 48.0, REFUTED if ≥ 58.0, else INCONCLUSIVE.
13. **Sanity**: `init_ppl_eval` is the μ-only held-out baseline; expect > 60 PPL (audit says likely near 61.664 on calib; the held-out equivalent is the honest μ-only number). Report it as the zero-train floor.

# Validation
- `calib_ids ∩ eval_ids = ∅` (assert at load time via object-id disjointness AND tensor equality check on the first calib/eval pair).
- `len(calib_ids) == 200`, `len(eval_ids) == 100`.
- μ ≤ 0 elementwise across all 12 blocks.
- LoRA-zero init eval `init_ppl_eval` reported; no hard equality to 61.664 (held-out ≠ calib).
- `final_ppl ≤ init_ppl_eval` (training on calib should at worst match, at best improve held-out — if `final_ppl > init_ppl_eval` that itself is signal).
- `base_c_proj_0_unchanged == True` (frozen base Conv1D weight unchanged).
- `n_trainable == 368640`.

# Artifacts
- `experiments/gelu_noneg_lora_disjoint_cycle239.py` — the rerun script.
- `autoresearch/results/cycle_239_gelu_noneg_lora_disjoint.json` — full result (config, per-seed `init_ppl_eval`, `best_ppl`, `best_step`, `final_ppl`, ppl history, loss history head/tail, μ stats per block, mean/std across seeds, verdict, cycle 234 leaked ref 32.411, cycle 228 calib ref 61.664, float ref 45.064).
- `autoresearch/results/cycle_239_summary.md` — markdown summary with table (float | cycle 228 calib | cycle 234 leaked | cycle 239 held-out) and verdict narrative.
- `autoresearch/results/cycle_239_manifest.txt` — written by the execute agent.

# Fallbacks
- Wikitext-2 val yields <300 seqs at SEQ_LEN=128 → drop SEQ_LEN=64, re-tokenize, confirm ≥300 seqs, log the fallback in JSON (`fallback_seq_len=64`). All downstream math (μ shape, c_fc input) is independent of SEQ_LEN.
- Wall-time tight for 3 seeds → single SEED=42, set `single_seed=true` and `seed_budget_reason="wall time"` in JSON.
- μ patch init ppl (on calib) drifts from cycle 228's 61.664 by >1 PPL → record but do not abort; the μ recipe is deterministic and any drift is a data-selection artefact of the extended loader.
- If `final_ppl > init_ppl_eval` (training on calib hurts held-out) → still report, verdict per thresholds; note "LoRA training on calib degrades held-out" in summary.

# Update bookkeeping
- Move item 249 from NEXT/P0 to PAST under Cycle 239 in `autoresearch/agenda.md` with verdict and 1-line summary.
- Append row to `autoresearch/experiments.tsv`: cycle 239, item 249, date, script path, `final_ppl` mean, verdict.
- Update `autoresearch/STATUS.md` current-cycle pointer.
- Commit under `[autoresearch] cycle 239: gelu_noneg + μ + LoRA r=8 disjoint split — <VERDICT>`.
- Do NOT delete cycle 234 artefacts; the audit note already links them.
