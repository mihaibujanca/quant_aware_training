# Task

Cycle 235 / item 245 (P0, T1): Centred Δ̃ feature SVD with concrete example extraction.
For each of 12 GPT-2 blocks, compute Δ̃_l per calibration token, SVD → top 32 right singular
vectors, decode each direction to vocab via W_U and to corpus via |U_l[:, i]|, label heuristic
coherence, and save artifacts loadable by `notebooks/bump_features_explore.py`. T2/T3/T4
(items 246–248) depend on these outputs.

# Hypothesis

From item 245:
- **Confirmed if**: ≥5 of 32 directions per block have coherence_label ∉ {mixed, incoherent}
  in ≥6 of 12 blocks.
- **Refuted if**: <2 directions per block have a coherent label in ALL 12 blocks (entire
  latent space is noise).
- **Inconclusive otherwise.**

# Pre-flight Results

- P1 (dup-check): NEW item, no prior t1_features.pt in results/.
- P2 (dep-check): none required.
- P3 (calibration): wikitext-2 train split available via HF datasets; same loader pattern
  as cycle 218 (which used val split with SEQ_LEN=128). T1 uses train split, SEQ_LEN=512.
- P4 (loader contract): notebook `bump_features_explore.py` L96–104 expects exact filenames
  `cycle_<N>_t1_features.pt`, `cycle_<N>_t1_decoded.json`, `cycle_<N>_t1_summary.md`,
  `cycle_<N>_t1_examples.md`. L296 expects `features[f"block_{block}"]["V"]` shape (768, 32).

Clean — proceed.

# Steps

1. Create `experiments/bump_t1_features_cycle235.py`.

2. Setup:
   - `from transformers import GPT2LMHeadModel, GPT2Tokenizer`; load `gpt2` on MPS if
     available, fp32, eval mode, `torch.no_grad()` context.
   - `from datasets import load_dataset`; `ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')`.
   - Set `SEED=42`, `N_SEQS=200`, `SEQ_LEN=512`.

3. Calibration batch construction:
   - Tokenize all non-empty lines, then concatenate token ids into one long stream
     (standard LM-style grouping, NOT per-line padding) → reshape into `(N_SEQS, SEQ_LEN)`.
     This avoids padding and gives exactly `200*512 = 102400` real tokens.
   - Save `calib_ids: (N_SEQS, SEQ_LEN)` int64.
   - Build per-token metadata tensors: `seq_index = arange(N_SEQS).unsqueeze(1).expand(-1, SEQ_LEN).flatten()`,
     `pos_in_seq = arange(SEQ_LEN).unsqueeze(0).expand(N_SEQS, -1).flatten()`,
     `token_ids = calib_ids.flatten()`. All shape `(102400,)`.

4. Forward pass + hooks:
   - For each block `l ∈ 0..11`, register a forward_pre_hook on `model.transformer.h[l].mlp.c_fc`
     to capture its INPUT (the post-LN2 residual, shape `(B, T, 768)`).
   - Run the model on `calib_ids` in batches of e.g. 8 sequences at a time to fit in MPS memory.
   - Stash concatenated `(102400, 768)` per block in a dict `mlp_inputs[l]`.

5. Per-block Δ̃ + SVD (loop l = 0..11):
   - `block = model.transformer.h[l]`
   - `W_fc = block.mlp.c_fc.weight.detach().float().cpu()` (768, 3072)
   - `b_fc = block.mlp.c_fc.bias.detach().float().cpu()` (3072,)
   - `W_proj = block.mlp.c_proj.weight.detach().float().cpu()` (3072, 768)
   - `x = mlp_inputs[l].float().cpu()` (102400, 768)
   - Compute `delta_out` in chunks of 4096 rows (same pattern as cycle 218 L133–142):
     ```
     preact = xc @ W_fc + b_fc                 # (chunk, 3072)
     gelu_post = F.gelu(preact, approximate='tanh')
     relu_post = F.relu(preact)
     delta_post = gelu_post - relu_post
     parts.append(delta_post @ W_proj)         # (chunk, 768)
     ```
     `delta_out = torch.cat(parts, dim=0)` (102400, 768)
   - `mean_delta = delta_out.mean(0)` (768,)
   - `delta_centred = delta_out - mean_delta`
   - `U, S, Vh = torch.linalg.svd(delta_centred, full_matrices=False)`
     → U (102400, 768), S (768,), Vh (768, 768)
   - `V_top32 = Vh[:32].T.contiguous()` → shape (768, 32) [matches loader L296]
   - `S_top32 = S[:32].clone()` → (32,)
   - `U_top32 = U[:, :32].clone()` → (102400, 32)
   - Store `features_dict[f"block_{l}"] = {"V": V_top32, "S": S_top32, "mean_delta": mean_delta}`
   - Keep `U_top32` and `S` (for total-energy denominator) in a separate in-memory list for
     decoding step 6. Free `U, Vh, delta_out, delta_centred` immediately.

6. After the block loop, save features:
   - `torch.save(features_dict, "autoresearch/results/cycle_235_t1_features.pt")`
   - `torch.save({"token_ids": token_ids, "seq_index": seq_index, "pos_in_seq": pos_in_seq},
     "autoresearch/results/cycle_235_t1_calibration_meta.pt")`

7. Vocab decoding setup:
   - `wte = model.transformer.wte.weight.detach().float().cpu()` (50257, 768)
   - For direction `v ∈ R^768`: `logit_impact = wte @ v` (50257,).
     top-50 boost = `torch.topk(logit_impact, 50)`;
     top-50 suppress = `torch.topk(-logit_impact, 50)` (then negate scores back for reporting).
   - Build a memoised decoder `tok_cache: dict[int, str]` so `tokenizer.decode([tid])` is
     called at most ~50k times total.

8. Per-direction decoding loop (12 blocks × 32 dirs = 384 records):
   For each block l, let `U_l = U_top32[l]`, `S_l = S_top32[l]`, `total_energy = (S_all[l] ** 2).sum()`:
   - For `i ∈ 0..31`:
     - `score = U_l[:, i].abs()` (102400,)
     - `top_idx = torch.topk(score, 100).indices` (100,)
     - For each `idx` in `top_idx`:
       - `seq_idx = int(seq_index[idx])`; `pos = int(pos_in_seq[idx])`
       - `tid = int(token_ids[idx])`; `tok_str = tok_cache[tid]`
       - `left_ids = calib_ids[seq_idx, max(0, pos-5):pos].tolist()`
       - `right_ids = calib_ids[seq_idx, pos+1:min(SEQ_LEN, pos+6)].tolist()`
       - `context_left = tokenizer.decode(left_ids)`; `context_right = tokenizer.decode(right_ids)`
       - `signed_score = float(U_l[idx, i].item())`
       - record: `{"token_id": tid, "token_str": tok_str, "context_left": context_left,
         "context_right": context_right, "position": pos, "score": signed_score}`
     - `v = V_top32[:, i]` (768,)
     - `logit_impact = wte @ v`
     - `boost_vals, boost_idx = torch.topk(logit_impact, 50)`
     - `sup_vals, sup_idx = torch.topk(-logit_impact, 50)`
     - `top_vocab_boost = [{"token_id": int(boost_idx[k]), "token_str": tok_cache[int(boost_idx[k])],
       "score": float(boost_vals[k])} for k in range(50)]`
     - `top_vocab_suppress = [{..., "score": -float(sup_vals[k])} for k in range(50)]`
     - `variance_explained = float((S_l[i] ** 2) / total_energy)`
     - `singular_value = float(S_l[i])`
     - `coherence_label = classify_coherence(top_corpus_tokens)` (step 9)
     - Append full record dict to `decoded_records`.

9. Coherence classifier `classify_coherence(top_corpus_tokens) -> str`:
   - Extract list of 100 `tok_str` (strip whitespace).
   - Counts:
     - `n_punct` = count where stripped token all in `string.punctuation` and non-empty
     - `n_stop` = count where stripped lowercase in a small fixed set
       `{"the","a","an","of","in","to","and","or","but","is","are","was","were","for","on","at","with","by","as","that","this","it","be","from","which","he","she","they","we","you","i","not","have","has","had","do","does","did"}`
     - `n_cap` = count where first char isupper and len ≥ 2
     - `n_num` = count where stripped.isdigit() or (stripped.replace('.','',1).isdigit())
     - Mode count: `mode_count = Counter(token_ids_in_top100).most_common(1)[0][1]`
   - Rule (priority order):
     - if `mode_count ≥ 30`: return `"repeated"`
     - if `n_punct ≥ 40`: return `"punctuation"`
     - if `n_stop ≥ 40`: return `"stopword"`
     - if `n_cap ≥ 40`: return `"capitalized"`
     - if `n_num ≥ 40`: return `"number"`
     - `max_class = max(n_punct, n_stop, n_cap, n_num, mode_count)`
     - if `max_class ≥ 20`: return `"mixed"`
     - else: return `"incoherent"`

10. Save decoded JSON:
    - `with open("autoresearch/results/cycle_235_t1_decoded.json", "w") as f:
      json.dump(decoded_records, f, indent=2)`

11. Build `cycle_235_t1_examples.md` (human-readable):
    - For each block l ∈ 0..11:
      - `## Block {l}` heading
      - Sort directions by `variance_explained` descending, take top 5
      - Per direction:
        - `### Block {l} dir {i} — var={var:.2%} σ={sigma:.2f} coherence={label}`
        - 3-row table:
          `| context_left | TOKEN | context_right |`
          `|---|---|---|`
          `| ...left... | **tok** | ...right... |` for top-3 corpus tokens
        - `boost: [tok1, tok2, ...]` (comma-joined top-10 boost `token_str`s)
        - `suppress: [...]` (top-10 suppress `token_str`s)
    - Write to `autoresearch/results/cycle_235_t1_examples.md`.

12. Build `cycle_235_t1_summary.md`:
    - `coherent_per_block[l] = sum(1 for rec in decoded_records
       if rec["block"] == l and rec["coherence_label"] not in {"mixed", "incoherent"})`
    - `n_blocks_5plus = sum(1 for l in range(12) if coherent_per_block[l] >= 5)`
    - `all_below_2 = all(c < 2 for c in coherent_per_block)`
    - Verdict:
      - if `n_blocks_5plus >= 6`: `"CONFIRMED"`
      - elif `all_below_2`: `"REFUTED"`
      - else: `"INCONCLUSIVE"`
    - Write a markdown file with:
      - `# Cycle 235 T1 Summary`
      - `**Verdict:** {verdict}`
      - `**Headline:** {n_blocks_5plus}/12 blocks have ≥5 coherent directions.`
      - Per-block count table `| block | coherent | total | labels |`
      - 1-paragraph plain-language interpretation: whether the centred bump space contains
        interpretable latent features or looks like noise, with a caveat that coherence is a
        heuristic (T2 is the causal test).

# Validation

- Assert `n_tokens == 102400` after build; assert no tensor has NaN.
- Assert `V_top32.shape == (768, 32)`, `S_top32.shape == (32,)`, `mean_delta.shape == (768,)`
  for each block.
- Cross-check against cycle 218: reload one block's `V` from features.pt, compute cos-sum
  against cycle 218's `vh_top20[l]` top-20 directions. Top-20 vs top-20 cos-sum should be
  non-trivial (not bounded by random ≈ 2.3); log the value. If zero, centring is off or the
  bases are flipped. Do NOT fail hard on this — it's a sanity print (cycle 218 used SEQ_LEN=128
  val split, whereas this uses SEQ_LEN=512 train split, so exact equality is not expected).
- Print shape assertions for the three saved `.pt` files after `torch.load`.
- Memory check: `delta_centred` is `(102400, 768)` float32 ≈ 314 MB; SVD via
  `torch.linalg.svd(full_matrices=False)` returns U (102400, 768) ≈ 314 MB + Vh (768, 768)
  ≈ 2 MB. Tight on MPS but fine on CPU. Run SVD on CPU to avoid device issues.
- `decoded_records` length must be exactly 384.

# Artifacts (REQUIRED — exact filenames per item 245)

- `experiments/bump_t1_features_cycle235.py`
- `autoresearch/results/cycle_235_t1_features.pt`
- `autoresearch/results/cycle_235_t1_calibration_meta.pt`
- `autoresearch/results/cycle_235_t1_decoded.json`
- `autoresearch/results/cycle_235_t1_examples.md`
- `autoresearch/results/cycle_235_t1_summary.md`
- `autoresearch/results/cycle_235_manifest.txt` (written by execute step)

# Fallbacks

- If wikitext-2 train split is unavailable / tokenizer misbehaves: fall back to wikitext-2
  validation split with an explicit `print("WARNING: using val split as fallback")` and
  annotate the summary. Keep N_SEQS=200, SEQ_LEN=512 (val has ~240 valid sequences at
  SEQ_LEN=512, enough).
- If `(102400, 768)` SVD OOMs on MPS: move `delta_centred` to CPU for SVD
  (`.cpu()` then `torch.linalg.svd`). This is the default path.
- If overall SVD runtime is excessive: chunk U computation by slicing the
  `delta_centred` matrix into 2 halves and SVD each? NO — do not half-SVD; instead reduce
  to `torch.svd_lowrank(delta_centred, q=40, niter=4)` which gives a rank-40 approximation
  sufficient for top-32. Record that fallback in summary.
- If tokenizer.decode is too slow: the `tok_cache` memo already covers it; if still slow,
  cache across all 384 iterations (not just per-direction).
- If coherence classifier labels nothing as coherent due to thresholds: log raw counts per
  direction in the decoded JSON under a `_raw_counts` field (debug only) so thresholds can
  be revisited without re-running.

# Update bookkeeping

- After execute: move item 245 from `NEXT / P0` to `PAST EXPERIMENTS / Cycle 235` in
  `autoresearch/agenda.md`, preserving the full item text and adding the verdict line.
- Append TSV row to `autoresearch/program.md` experiment log.
- Update `autoresearch/STATUS.md` with cycle 235 verdict and headline.
- Commit with message `[autoresearch] cycle 235: T1 bump feature SVD — {VERDICT}`.
- Note: this item unblocks items 246, 247, 248 — do NOT move those, they remain in NEXT.
