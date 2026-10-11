# Item
245 (P0): T1 — Centred Δ̃ feature SVD with concrete example extraction.
For each of 12 GPT-2 blocks, compute Δ̃_l per token on calibration corpus, SVD → top 32 right singular vectors, decode each direction to vocab via W_U projection, find top-100 corpus token examples per direction, label coherence.

# Context
- Cycle 218 (CONFIRMED): centred Δ̃ aligns with c_fc reading subspace +17 to +33σ above random.
- Cycle 227 (INCONCLUSIVE): specificity walked back — alignment is broad across all downstream blocks, not specifically inter-block.
- This item (T1) extracts the LATENT FEATURES of Δ̃ to test whether the directions are interpretable AT ALL.
- T2 (item 246), T3 (247), T4 (248) all depend on T1's outputs.

# Dependencies
None.

# Key Files
- experiments/delta_centred_alignment_cycle218.py — has the Δ̃ pipeline (calibration centring).
- notebooks/bump_features_explore.py — the loader scaffold; defines the EXACT filenames the notebook expects.

# Constraints
- Calibration: wikitext-2 TRAIN split, N_SEQS=200, SEQ_LEN=512 (~100K tokens).
- Centring mean computed on calibration corpus, applied to held-out (T2-T4 will use held-out, T1 only uses calib).
- For each block l (0..11):
  - Compute Δ̃_l per token (n_tokens × 768).
  - SVD → V_l (768, 32), S_l (32,), token coefficients U_l (n_tokens, 32).
  - For each direction i ∈ 0..31:
    - top 100 corpus tokens by |U_l[t, i]| with 5-token contexts
    - top 50 boost vocab via W_U @ v_{l,i}
    - top 50 supp vocab via -W_U @ v_{l,i}
    - variance explained = s_i² / sum(s_j²)
    - heuristic coherence label
- Output filenames (EXACT, three-digit zero-padded cycle):
  - experiments/bump_t1_features_cycle235.py
  - autoresearch/results/cycle_235_t1_features.pt — torch dict {block_l: {V, S, mean_delta}}
  - autoresearch/results/cycle_235_t1_calibration_meta.pt — {token_ids, seq_index, pos_in_seq}
  - autoresearch/results/cycle_235_t1_decoded.json — list of 384 direction records
  - autoresearch/results/cycle_235_t1_examples.md — human-readable per-block top-5 directions
  - autoresearch/results/cycle_235_t1_summary.md — verdict + headline
- Confirmed if: ≥5 of 32 directions per block coherent in ≥6 of 12 blocks.
- Refuted if: <2 coherent per block in ALL 12 blocks.
- Inconclusive otherwise.

# Handoff
Plan agent: design the script. Compute considerations:
- 200 seqs × 512 tokens = ~102K tokens, but seqs may have padding. Use real token count after dropping pad.
- Δ̃ matrix shape (~100K, 768). SVD is fast.
- For each direction, the top-100 corpus token lookup is the slowest part. Use torch.topk on |U_l[:, i]| directly (vectorized).
- Coherence labels: heuristic on the top-100 token strings — count punctuation tokens (string.punctuation), stopwords (small list), capitalized (str.isupper()), numbers (str.isdigit()), single repeat (mode), then label by max class.
- Wall time: ~5-10 min compute, mostly SVD + decoding.
- New script `experiments/bump_t1_features_cycle235.py`. Output all 6 files per the constraint list.
