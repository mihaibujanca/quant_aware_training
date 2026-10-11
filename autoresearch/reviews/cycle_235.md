# Cycle 235 Review — T1 Centred Δ̃ Feature SVD

**Verdict:** APPROVE

**Item:** 245 (P0, T1) — interpretability extraction of top-32 right singular vectors of centred Δ̃ per GPT-2 block, decoded to vocab via W_U and to corpus via |U|.

## Plan summary

Reuses cycle 218 MLP-input hook pipeline. Computes `delta_out = GELU(preact)@W_proj - ReLU(preact)@W_proj` chunked at 4096 rows, centres, SVD, keeps top-32 V/S and U. Decodes each of 384 directions with top-100 corpus tokens + top-50 vocab boost/suppress via `wte @ v`. Applies a small heuristic coherence classifier (punct/stop/cap/num/repeated/mixed/incoherent). Writes features.pt, calibration_meta.pt, decoded.json, examples.md, summary.md matching the notebook loader contract at `notebooks/bump_features_explore.py` L96–104, L296.

## Strengths

- Loader contract is explicit and the filenames use `cycle_235_t1_*` (correct `{cycle:03d}` format for 235).
- `V_top32` stored transposed to (768, 32) as the notebook expects.
- Token-stream grouping avoids per-line padding and gives exact 102400 tokens.
- Chunked `delta_out` reuses the verified cycle 218 pattern.
- SVD explicitly runs on CPU to sidestep MPS memory and device edge cases.
- Variance-explained denominator uses FULL `S`, not just top-32, so fractions are honest.
- Cross-check against cycle 218's `vh_top20` top-20 cos-sum is a soft sanity print, not a hard assert (correct — SEQ_LEN and split differ).
- Fallback tree covers wikitext train unavailability, SVD OOM, runtime, and threshold misfire (raw counts logged).

## Concerns (non-blocking)

1. **Wikitext-2 train split**: the script uses `'wikitext-2-raw-v1'` train. Cycle 218 used val. Train split should be present on HF, but the explicit fallback + warning annotation is correct — no blocker.
2. **Coherence heuristic thresholds** (40/20) are rough. Plan acknowledges T2 is the causal test and logs raw counts on misfire, so a re-labelling pass is cheap. Acceptable.
3. `wte @ v` produces a (50257,) logit-direction score — this is the standard unembedding projection for decoding a residual-space direction. Note this reads `wte` (embed matrix) which is tied to `lm_head` in GPT-2, so it IS W_U. Fine.
4. `signed_score = U_l[idx, i]` is stored but ranking uses `|U_l[:, i]|`. That's correct — a direction's top corpus tokens include both tails, and sign is preserved in the record for inspection.
5. Top-50 suppress list: plan stores `-float(sup_vals[k])` so values are the original (negative) logit impacts. Correct — the notebook can read signed values directly.
6. No `block` field is explicitly added to each decoded record, but the summary counts `rec["block"] == l`. Plan should ensure each record has `block: l` written. Minor — call out during execute.

## Risk assessment

Low. Pipeline mirrors cycle 218, artifacts are strict-contract, verdict thresholds are pre-registered (≥5 coherent in ≥6/12 blocks = CONFIRMED; <2 in ALL 12 = REFUTED). Unblocks items 246–248.

## Decision

**APPROVE.** Proceed to execute. Remind the executor to set `rec["block"] = l` when building decoded_records so the summary count loop works.
