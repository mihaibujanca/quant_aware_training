# Cycle 332 Brief

## Item
Item 331: Train tiny GPT (4L 256d) on wikitext-2 with three activation variants {GELU, ReLU², LeakyReLU²} — real comparative numbers, not zero-shot patches.

## Context
- No GPT model class exists in `aleph/` — must write from scratch (~150 lines, nanoGPT-shaped)
- wikitext-2 val data exists at `data/wikitext2_valid.txt`; train split loads via HF `datasets`
- Eff_rank census pattern: `experiments/all_blocks_eff_rank_cycle278.py` (hook c_proj output, covariance SVD)
- Curvature framework: `experiments/unified_curvature_cycle302.py` imports `curvature_residual_geometric.py::UnifiedCurvatureFramework`
- GPT-2 BPE tokenizer: `GPT2TokenizerFast.from_pretrained("gpt2")` (cycle 278/302 pattern)
- Previous cycle 331 ran item 345 (PLAN INVALID); item 331 itself is still in the active queue

## Dependencies
- Tiny GPT model: write from scratch in the experiment script
- wikitext-2 BPE: HF datasets + GPT2TokenizerFast (already used in cycles 278, 302)
- Eff_rank census: adapt cycle 278 pattern to the tiny model's 4 blocks
- Curvature framework: `experiments/curvature_residual_geometric.py` (optional, best-effort)

## Key Files
- Data: `data/wikitext2_valid.txt`, HF `wikitext/wikitext-2-raw-v1` for train
- Eff_rank template: `experiments/all_blocks_eff_rank_cycle278.py`
- Curvature template: `experiments/unified_curvature_cycle302.py`, `experiments/curvature_residual_geometric.py`
- Will create: `experiments/tiny_gpt_act_compare_cycle331.py`
- Checkpoints: `experiments/tiny_gpt_act_compare_checkpoints/{gelu,relu2,lrelu2}.pt`

## Constraints
- Run cap: 45 minutes total (3 × ~10 min training + eff_rank census + optional curvature)
- Same seed (42), same optimizer (AdamW lr=3e-4, wd=0.01, cosine decay), 5000 steps each
- Curvature framework is best-effort — if time runs out, eff_rank census alone is the core deliverable

## Handoff
Write tiny GPT model + training loop + eff_rank census in one script. Train 3 variants serially. Run eff_rank census on each checkpoint. Report per-block eff_rank profiles side-by-side. Outcome: CONFIRMED-rank-profile-shape / REFUTED-rank-profile-converges / INCONCLUSIVE per agenda criteria.
