#!/usr/bin/env python3
"""Cycle 335: Selective LeakyReLU² at b4/b5 — RE-RUN with correct loader.

Subprocess-invokes pgolf_selective_lrelu2_v2.py and writes results.
"""
import json, os, subprocess, sys
from pathlib import Path

PGOLF_PYTHON = "/Users/mihai/Projects/parameter-golf/.venv/bin/python"
PGOLF_SCRIPT = "/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_selective_lrelu2_v2.py"
CHECKPOINT = "/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz"
CYCLE = 335
OUT_DIR = Path("autoresearch/results")


def main():
    print(f"=== Cycle {CYCLE}: Selective LeakyReLU² re-run with correct loader ===")
    print(f"  checkpoint: {CHECKPOINT}")
    print(f"  pgolf python: {PGOLF_PYTHON}")

    pgolf_root = "/Users/mihai/Projects/parameter-golf"
    env = os.environ.copy()
    env["DATA_PATH"] = f"{pgolf_root}/data/datasets/fineweb10B_sp1024"

    proc = subprocess.run(
        [PGOLF_PYTHON, PGOLF_SCRIPT,
         "--checkpoint", CHECKPOINT,
         "--n-tokens", "51200",
         "--lrelu-blocks", "4", "5"],
        capture_output=True, text=True, timeout=600,
        cwd=pgolf_root, env=env,
    )

    # Always print stderr for diagnostics
    if proc.stderr:
        print("=== pgolf stderr ===", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        print(f"ERROR: pgolf script failed with returncode={proc.returncode}", file=sys.stderr)
        raise SystemExit(f"pgolf script failed: {proc.returncode}")

    # Parse JSON from stdout
    result = json.loads(proc.stdout.strip())

    # Write JSON
    json_path = OUT_DIR / f"cycle_{CYCLE}_pgolf_selective_lrelu2.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"  Written: {json_path}")

    # Extract key values
    lrelu2_bpb = result["global_lrelu2_bpb"]
    sel_bpb = result["selective_bpb"]
    relu2_bpb = result["global_relu2_bpb"]
    delta_sel = result["delta_selective_vs_lrelu2"]
    delta_relu2 = result["delta_relu2_vs_lrelu2"]
    lrelu_blocks = result["lrelu_blocks"]
    loader_config = result.get("loader_config", {})

    # Verdict
    if delta_sel < 0.003 and delta_relu2 >= 0.003:
        verdict = "CONFIRMED-localized-mechanism"
    elif delta_sel >= 0.003:
        verdict = "REFUTED-distributed-mechanism"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n=== Results ===")
    print(f"  global-lrelu2 BPB : {lrelu2_bpb:.6f}  (sanity: cycle 330 broken was 12.8)")
    print(f"  selective BPB     : {sel_bpb:.6f}  (lrelu2 at blocks {lrelu_blocks})")
    print(f"  global-relu2 BPB  : {relu2_bpb:.6f}")
    print(f"  delta (sel - lrelu2): {delta_sel:+.6f}")
    print(f"  delta (relu2 - lrelu2): {delta_relu2:+.6f}")
    print(f"  loader_config: {loader_config}")
    print(f"\n  VERDICT: {verdict}")

    # Write summary
    summary_path = OUT_DIR / f"cycle_{CYCLE}_summary.md"
    summary_path.write_text(f"""# Cycle {CYCLE}: Selective LeakyReLU² Re-run

## Verdict: {verdict}

## BPB Results

| Config | BPB |
|---|---|
| global-lrelu2 (baseline) | {lrelu2_bpb:.6f} |
| selective (lrelu2 at b{lrelu_blocks[0]},b{lrelu_blocks[1]}) | {sel_bpb:.6f} |
| global-relu2 | {relu2_bpb:.6f} |

## Deltas

- selective vs global-lrelu2: {delta_sel:+.6f}
- global-relu2 vs global-lrelu2: {delta_relu2:+.6f}

## Loader Config

- mlp_act: {loader_config.get('mlp_act', 'N/A')}
- xsa_last_n: {loader_config.get('xsa_last_n', 'N/A')}
- rope_dims: {loader_config.get('rope_dims', 'N/A')}

## Validation

- Cycle 330 broken value (wrong loader): 12.8 BPB
- This run global-lrelu2 BPB: {lrelu2_bpb:.6f} — {'FIXED' if lrelu2_bpb < 2.0 else 'STILL BROKEN'}

## Thresholds

- CONFIRMED-localized-mechanism: delta_sel < 0.003 AND delta_relu2 >= 0.003
- REFUTED-distributed-mechanism: delta_sel >= 0.003
- INCONCLUSIVE: neither threshold met
""")
    print(f"  Written: {summary_path}")
    return verdict, lrelu2_bpb, sel_bpb, relu2_bpb


if __name__ == "__main__":
    main()
