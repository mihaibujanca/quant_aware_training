# Cycle 330 Plan: Selective LeakyReLU² at b4/b5 only

## Step 0 (REQUIRED)
Read `autoresearch/cross_project/parameter_golf.md` for subprocess invocation pattern.
Read `parameter-golf/docs/EFFRANK_CURVATURE_FINDINGS.md` section 6.4 for b4/b5 localization context.

## Step 1: Write pgolf-side script
Path: `/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_selective_lrelu2.py`

Reuse `load_model_v2` from `pgolf_eff_rank_census.py` (handles variable mlp_mult, bigram, skip_weights).

**Critical**: `load_model_v2` always constructs with `mlp_act="relu2"` (lrelu2 not detectable from weights). The script must patch `block.mlp.act` after loading to set the correct activation per config.

Three eval configs, each a full BPB eval:
1. **global-lrelu2**: set `block.mlp.act = "lrelu2"` for ALL 13 blocks
2. **selective**: set `block.mlp.act = "lrelu2"` for blocks 4,5 only; leave rest as `"relu2"`
3. **global-relu2**: leave all blocks as `"relu2"` (model as loaded)

For each config: patch `block.mlp.act`, run `eval_bpb(model, tokens)`, record BPB.

The `lrelu_slope` is already stored per block as `block.lrelu_slope` (loaded from checkpoint weights). The MLP.__call__ reads `slope` from the Block.__call__ caller. So patching `block.mlp.act` is sufficient — the slope flows through automatically.

Print JSON to stdout: `{"global_lrelu2_bpb": float, "selective_bpb": float, "global_relu2_bpb": float, "checkpoint": str, "n_tokens": int, "lrelu_blocks": [4, 5]}`.

## Step 2: Write qat-side wrapper
Path: `experiments/pgolf_selective_lrelu2_cycle330.py`

Subprocess-invoke the pgolf script with:
```
/Users/mihai/Projects/parameter-golf/.venv/bin/python \
  /Users/mihai/Projects/parameter-golf/local_scripts/pgolf_selective_lrelu2.py \
  --checkpoint /Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz \
  --n-tokens 51200
```

Parse JSON stdout, write to `autoresearch/results/cycle_330_pgolf_selective_lrelu2.json`.
Build summary at `autoresearch/results/cycle_330_summary.md`.

Use `timeout=600` (3 forward passes, ~90s expected).

## Step 3: Run and collect
Execute the qat wrapper. Expect 3 BPB numbers.

## Step 4: Verdict
- **CONFIRMED-localized-mechanism**: selective ≈ global-lrelu² (delta < 0.003) AND selective < global-relu² by ≥ 0.003
- **REFUTED-distributed-mechanism**: selective is ≥ 0.003 worse than global-lrelu²
- **INCONCLUSIVE**: deltas below thresholds in both directions

## Step 5: Bookkeeping
Move item 343 to PAST in agenda.md. Commit results + manifest.
