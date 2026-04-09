#!/usr/bin/env python3
"""QAT-side wrapper for cycle 323: post-mlp_scale eff_rank census on parameter-golf.

Subprocess-invokes pgolf_eff_rank_census_post_scale.py against the canonical
checkpoint, saves JSON, and writes a side-by-side summary vs cycle 318 baseline.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

PGOLF_ROOT = "/Users/mihai/Projects/parameter-golf"
PGOLF_PYTHON = f"{PGOLF_ROOT}/.venv/bin/python"
PGOLF_SCRIPT = f"{PGOLF_ROOT}/local_scripts/pgolf_eff_rank_census_post_scale.py"
CHECKPOINT = f"{PGOLF_ROOT}/logs/13L_640d_322_7k_wd70_batch20k_best.npz"

CYCLE = 323
OUT_DIR = Path("autoresearch/results")

# Cycle 318 pre-scale baseline
BASELINE = {
    0: 187.61,
    1: 160.08,
    2: 86.05,
    3: 4.89,
    4: 38.30,
    5: 39.96,
    6: 28.21,
    7: 23.23,
    8: 30.78,
    9: 28.85,
    10: 21.55,
    11: 13.40,
    12: 6.94,
}


def write_summary(result: dict, pre: dict[int, float], out_path: Path) -> str:
    n_blocks = result["n_blocks"]
    profile = result["profile"]

    rows = []
    max_rel_change = 0.0
    any_ge50 = False
    all_le20 = True

    for l in range(n_blocks):
        key = str(l)
        er_post = profile[key]["eff_rank"]
        er_pre = pre[l]
        rel = (er_post - er_pre) / er_pre * 100.0
        if abs(rel) > max_rel_change:
            max_rel_change = abs(rel)
        if abs(rel) >= 50:
            any_ge50 = True
        if abs(rel) > 20:
            all_le20 = False
        rows.append((l, er_pre, er_post, rel))

    if all_le20:
        verdict = "CONFIRMED-rank-invariant"
        note = (
            "Post-scale eff_rank agrees within ±20% relative at every block. "
            "The cycle 318 pre-scale measurement is in the correct representation; "
            "the b3 rank-collapse finding is robust to capture point choice."
        )
    elif any_ge50:
        verdict = "REFUTED-scale-matters"
        note = (
            "At least one block shifts by ≥50% relative. "
            "mlp_scale is doing meaningful rank reshaping; "
            "cycle 318 reading should be re-interpreted in the post-scale representation."
        )
    else:
        verdict = "INCONCLUSIVE"
        note = "Some blocks shift >20% but none ≥50%. mlp_scale has a moderate effect on eff_rank."

    lines = [
        f"# Cycle {CYCLE}: Post-mlp_scale eff_rank census — {verdict}",
        "",
        f"**Verdict**: {verdict}",
        "",
        note,
        "",
        f"**Max relative change**: {max_rel_change:.1f}%",
        "",
        "## Per-block comparison",
        "",
        "| Block | Pre-scale (c318) | Post-scale (c323) | Rel change |",
        "|------:|----------------:|------------------:|----------:|",
    ]
    for l, er_pre, er_post, rel in rows:
        flag = " ← >20%" if abs(rel) > 20 else ""
        lines.append(f"| b{l:02d} | {er_pre:.2f} | {er_post:.2f} | {rel:+.1f}%{flag} |")

    lines += [
        "",
        "## Notes",
        "- Pre-scale: `MLP.__call__` return value (cycle 318 measurement)",
        "- Post-scale: `MLP.__call__` output × `block.mlp_scale` per-channel (cycle 323)",
        f"- Checkpoint: `{result['checkpoint']}`",
        f"- n_tokens: {result['n_tokens_requested']}  seq_len: {result['seq_len']}",
    ]

    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    return verdict


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_out = OUT_DIR / f"cycle_{CYCLE}_pgolf_postscale_eff_rank.json"
    summary_out = OUT_DIR / f"cycle_{CYCLE}_summary.md"

    print(f"Running pgolf post-scale eff_rank census …", flush=True)
    proc = subprocess.run(
        [PGOLF_PYTHON, PGOLF_SCRIPT, "--checkpoint", CHECKPOINT, "--n-tokens", "25600"],
        capture_output=True, text=True, timeout=600, cwd=PGOLF_ROOT,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"pgolf script failed (exit {proc.returncode})")

    result = json.loads(proc.stdout)
    json_out.write_text(json.dumps(result, indent=2))
    print(f"JSON saved to {json_out}", flush=True)

    verdict = write_summary(result, BASELINE, summary_out)
    print(f"Summary saved to {summary_out}", flush=True)
    print(f"Verdict: {verdict}", flush=True)

    # Print per-block table to stdout for log
    print("\nPer-block eff_rank (pre vs post scale):")
    print(f"{'Block':>6}  {'Pre(c318)':>10}  {'Post(c323)':>11}  {'Rel%':>8}")
    for l in range(result["n_blocks"]):
        er_post = result["profile"][str(l)]["eff_rank"]
        er_pre = BASELINE[l]
        rel = (er_post - er_pre) / er_pre * 100.0
        print(f"  b{l:02d}   {er_pre:10.2f}   {er_post:11.2f}   {rel:+7.1f}%")


if __name__ == "__main__":
    main()
