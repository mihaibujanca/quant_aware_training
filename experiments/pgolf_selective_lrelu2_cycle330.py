#!/usr/bin/env python3
"""Cycle 330: Selective lrelu2 at b4/b5 — qat-side wrapper.

Invokes pgolf_selective_lrelu2.py via subprocess, parses JSON results,
writes cycle_330_pgolf_selective_lrelu2.json and cycle_330_summary.md.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

PGOLF_PYTHON = "/Users/mihai/Projects/parameter-golf/.venv/bin/python"
PGOLF_SCRIPT = "/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_selective_lrelu2.py"
PGOLF_ROOT = "/Users/mihai/Projects/parameter-golf"
CHECKPOINT = "/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz"
CYCLE = 330
OUT_DIR = Path(__file__).resolve().parent.parent / "autoresearch" / "results"


def verdict(global_lrelu2: float, selective: float, global_relu2: float) -> str:
    delta_selective_vs_global_lrelu2 = selective - global_lrelu2
    delta_selective_vs_global_relu2 = global_relu2 - selective
    if abs(delta_selective_vs_global_lrelu2) < 0.003 and delta_selective_vs_global_relu2 >= 0.003:
        return "CONFIRMED-localized-mechanism"
    elif delta_selective_vs_global_lrelu2 >= 0.003:
        return "REFUTED-distributed-mechanism"
    else:
        return "INCONCLUSIVE"


def main():
    proc = subprocess.run(
        [
            PGOLF_PYTHON, PGOLF_SCRIPT,
            "--checkpoint", CHECKPOINT,
            "--n-tokens", "51200",
        ],
        capture_output=True, text=True, timeout=600,
        cwd=PGOLF_ROOT,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"pgolf script failed with returncode {proc.returncode}")

    result = json.loads(proc.stdout)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"cycle_{CYCLE}_pgolf_selective_lrelu2.json"
    json_path.write_text(json.dumps(result, indent=2))

    global_lrelu2 = result["global_lrelu2_bpb"]
    selective = result["selective_bpb"]
    global_relu2 = result["global_relu2_bpb"]
    v = verdict(global_lrelu2, selective, global_relu2)

    delta_sel_vs_lrelu2 = selective - global_lrelu2
    delta_relu2_vs_sel = global_relu2 - selective
    delta_relu2_vs_lrelu2 = global_relu2 - global_lrelu2

    summary = f"""# Cycle {CYCLE} — Selective lrelu2 at b4/b5

## Verdict: {v}

## BPB Results

| Config | BPB |
|---|---|
| global-lrelu2 (as trained) | {global_lrelu2:.6f} |
| selective (b4+b5 only) | {selective:.6f} |
| global-relu2 (full ablation) | {global_relu2:.6f} |

## Deltas

- selective vs global-lrelu2: {delta_sel_vs_lrelu2:+.6f}
- global-relu2 vs selective: {delta_relu2_vs_sel:+.6f}
- global-relu2 vs global-lrelu2: {delta_relu2_vs_lrelu2:+.6f}

## Verdict Criteria

- CONFIRMED-localized-mechanism: |selective - global-lrelu2| < 0.003 AND (global-relu2 - selective) >= 0.003
- REFUTED-distributed-mechanism: selective >= global-lrelu2 + 0.003
- INCONCLUSIVE: otherwise

## Parameters

- Checkpoint: {result["checkpoint"]}
- n_tokens: {result["n_tokens"]}
- lrelu_blocks (selective): {result["lrelu_blocks"]}
- n_blocks: {result["n_blocks"]}
"""

    summary_path = OUT_DIR / f"cycle_{CYCLE}_summary.md"
    summary_path.write_text(summary)

    print(f"VERDICT: {v}")
    print(f"global-lrelu2 BPB: {global_lrelu2:.6f}")
    print(f"selective BPB:      {selective:.6f}")
    print(f"global-relu2 BPB:   {global_relu2:.6f}")
    print(f"Results: {json_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
