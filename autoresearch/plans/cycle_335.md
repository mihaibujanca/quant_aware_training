# Cycle 335 Plan: Selective LeakyReLU² at b4/b5 — RE-RUN with correct loader

## Step 0 (REQUIRED)
Read `autoresearch/cross_project/parameter_golf.md` for subprocess invocation pattern.

## Step 1: Write pgolf-side script
Path: `/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_selective_lrelu2_v2.py`

Copy `load_model_v2` verbatim from `pgolf_laser_retrain.py` (lines 85-197). This loader:
- Accepts `summary_json: str | None` parameter
- Reads `mlp_act`, `xsa_last_n`, `rope_dims`, `lrelu_slope` from the summary JSON config dict
- Passes all four to the `GPT()` constructor

Add the same summary JSON discovery logic from `pgolf_laser_retrain.py` (lines 301-326):
- Try `<stem>_summary.json` and `<stem>_batch20k_summary.json` next to the checkpoint
- Fallback to `13L_640d_322_7k_wd70_batch20k_summary.json` in the same directory

Keep the same 3-config eval logic from cycle 330's `pgolf_selective_lrelu2.py`:
1. **global-lrelu2**: set `block.mlp.act = "lrelu2"` for ALL 13 blocks → eval BPB (this is the sanity-check baseline, must be ~1.35)
2. **selective**: set `block.mlp.act = "lrelu2"` for blocks 4,5 only; rest `"relu2"` → eval BPB
3. **global-relu2**: set all blocks to `"relu2"` → eval BPB

**Sanity check**: after config 1, assert `global_lrelu2_bpb < 2.0`. If baseline > 2.0, the loader is still broken — abort with error.

Print JSON to stdout: `{"global_lrelu2_bpb": float, "selective_bpb": float, "global_relu2_bpb": float, "checkpoint": str, "n_tokens": int, "lrelu_blocks": [4, 5], "n_blocks": int, "loader_config": {"mlp_act": str, "xsa_last_n": int, "rope_dims": int}}`.

## Step 2: Write qat-side wrapper
Path: `experiments/pgolf_selective_lrelu2_v2_cycle335.py`

Subprocess-invoke the pgolf script:
```
/Users/mihai/Projects/parameter-golf/.venv/bin/python \
  /Users/mihai/Projects/parameter-golf/local_scripts/pgolf_selective_lrelu2_v2.py \
  --checkpoint /Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz \
  --n-tokens 51200
```

Parse JSON stdout, write to `autoresearch/results/cycle_335_pgolf_selective_lrelu2.json`.
Build summary at `autoresearch/results/cycle_335_summary.md` with:
- All 3 BPB values
- Deltas: selective vs global-lrelu2, global-relu2 vs global-lrelu2
- Loader config (mlp_act, xsa_last_n, rope_dims) to confirm correct loading
- Comparison to cycle 330 broken values (12.8 vs ~1.35) as validation

Use `timeout=600` (3 forward passes, ~90s expected).

## Step 3: Run and collect
Execute the qat wrapper. Expect 3 BPB numbers, all near 1.35.

## Step 4: Verdict
- **CONFIRMED-localized-mechanism**: selective ~ global-lrelu2 (delta < 0.003) AND selective < global-relu2 by >= 0.003
- **REFUTED-distributed-mechanism**: selective >= 0.003 worse than global-lrelu2
- **INCONCLUSIVE**: deltas below thresholds in both directions

## Step 5: Bookkeeping
Move item 343b to PAST in agenda.md with verdict and commit hash. Commit results + manifest.

## Validation
- Baseline BPB (global-lrelu2) must be ~1.35, not 12.8 (the cycle 330 broken value)
- Loader must print `mlp_act=lrelu2, xsa_last_n=13, rope_dims=16` to stderr
- All 3 BPB values should be in range [1.2, 2.0] for a valid run

## Artifacts
- `parameter-golf/local_scripts/pgolf_selective_lrelu2_v2.py` (permanent pgolf infrastructure)
- `experiments/pgolf_selective_lrelu2_v2_cycle335.py` (cycle artifact)
- `autoresearch/results/cycle_335_pgolf_selective_lrelu2.json`
- `autoresearch/results/cycle_335_summary.md`
