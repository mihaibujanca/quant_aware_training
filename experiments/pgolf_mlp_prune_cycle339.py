#!/usr/bin/env python3
"""Cycle 339: mlp_scale-based block pruning — zero b12's MLP contribution.

Subprocess-invokes pgolf_mlp_prune.py and writes results.
"""
import json, os, subprocess, sys
from pathlib import Path

PGOLF_PYTHON = "/Users/mihai/Projects/parameter-golf/.venv/bin/python"
PGOLF_SCRIPT = "/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_mlp_prune.py"
CHECKPOINT = "/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz"
CYCLE = 339
OUT_DIR = Path("autoresearch/results")


def main():
    print(f"=== Cycle {CYCLE}: mlp_scale-based block pruning ===")
    print(f"  checkpoint: {CHECKPOINT}")

    pgolf_root = "/Users/mihai/Projects/parameter-golf"
    env = os.environ.copy()
    env["DATA_PATH"] = f"{pgolf_root}/data/datasets/fineweb10B_sp1024"

    proc = subprocess.run(
        [PGOLF_PYTHON, PGOLF_SCRIPT,
         "--checkpoint", CHECKPOINT,
         "--n-tokens", "51200"],
        capture_output=True, text=True, timeout=600,
        cwd=pgolf_root, env=env,
    )

    if proc.stderr:
        print("=== pgolf stderr ===", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        print(f"ERROR: pgolf script failed with returncode={proc.returncode}", file=sys.stderr)
        raise SystemExit(1)

    result = json.loads(proc.stdout)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"cycle_{CYCLE}_pgolf_mlp_prune.json").write_text(json.dumps(result, indent=2))

    # Print summary
    print(f"\nBaseline BPB: {result['baseline_bpb']:.6f}")
    print(f"Sanity BPB:   {result['sanity_bpb']:.6f} (ok={result['sanity_ok']})")
    print()
    for name, cfg in result["configs"].items():
        print(f"  {name}: blocks={cfg['blocks_zeroed']}, BPB={cfg['bpb']:.6f}, delta={cfg['delta']:+.6f}")

    print(f"\nResults written to {OUT_DIR / f'cycle_{CYCLE}_pgolf_mlp_prune.json'}")


if __name__ == "__main__":
    main()
