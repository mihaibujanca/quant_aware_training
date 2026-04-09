#!/usr/bin/env python3
"""
Cycle 337: Full 13-block weight spectrum census on pgolf checkpoint.
Pure numpy SVD of mlp.proj.weight for all 13 blocks.
No subprocess needed — direct np.load on .npz.
"""
import json
import sys
from pathlib import Path

import numpy as np

CHECKPOINT = "/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz"
OUT_DIR = Path("autoresearch/results")
CYCLE = 337


def energy_fraction(S, k):
    """Fraction of total squared singular value energy in top-k singular values."""
    total = float(np.sum(S ** 2))
    return float(np.sum(S[:k] ** 2) / total)


def rank_for_threshold(S, threshold):
    """Minimum rank to capture >= threshold fraction of energy."""
    total = float(np.sum(S ** 2))
    cumsum = np.cumsum(S ** 2)
    hits = np.where(cumsum / total >= threshold)[0]
    return int(hits[0] + 1) if len(hits) > 0 else len(S)


def main():
    print(f"Loading checkpoint: {CHECKPOINT}", file=sys.stderr)
    data = np.load(CHECKPOINT, allow_pickle=True)

    # List all keys to understand structure
    keys = sorted(data.files)
    proj_keys = [k for k in keys if "mlp.proj.weight" in k]
    print(f"Found {len(proj_keys)} mlp.proj.weight keys", file=sys.stderr)
    for k in proj_keys:
        print(f"  {k}: {data[k].shape}", file=sys.stderr)

    results = {}
    for i in range(13):
        key = f"blocks.{i}.mlp.proj.weight"
        if key not in data:
            print(f"WARNING: key {key} not found", file=sys.stderr)
            continue

        W = data[key].astype(np.float32)
        print(f"Block {i}: W shape = {W.shape}", file=sys.stderr)

        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        n_sv = len(S)

        top1 = energy_fraction(S, 1)
        top5 = energy_fraction(S, min(5, n_sv))
        top13 = energy_fraction(S, min(13, n_sv))

        r50 = rank_for_threshold(S, 0.50)
        r90 = rank_for_threshold(S, 0.90)
        r95 = rank_for_threshold(S, 0.95)

        results[i] = {
            "shape": list(W.shape),
            "n_sv": n_sv,
            "top1": round(top1, 4),
            "top5": round(top5, 4),
            "top13": round(top13, 4),
            "rank50": r50,
            "rank90": r90,
            "rank95": r95,
            "top5_sv": S[:5].tolist(),
        }

        print(
            f"  Block {i:2d}: top-1={top1*100:.2f}%  top-5={top5*100:.2f}%  "
            f"top-13={top13*100:.2f}%  r95={r95}",
            file=sys.stderr,
        )

    # Verdict
    top5_vals = [v["top5"] for v in results.values()]
    max_top5 = max(top5_vals)
    min_top5 = min(top5_vals)

    if max_top5 >= 0.15:
        verdict = "CONFIRMED"
        verdict_detail = f">=1 block has top-5 >= 15% (max={max_top5*100:.2f}%)"
    elif max_top5 < 0.10:
        verdict = "REFUTED"
        verdict_detail = f"All blocks top-5 < 10% (max={max_top5*100:.2f}%)"
    else:
        verdict = "INCONCLUSIVE"
        verdict_detail = f"1+ block in 10-15% range (max={max_top5*100:.2f}%)"

    output = {
        "checkpoint": CHECKPOINT,
        "n_blocks": 13,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "max_top5_pct": round(max_top5 * 100, 4),
        "min_top5_pct": round(min_top5 * 100, 4),
        "per_block": results,
    }

    # Write JSON
    json_path = OUT_DIR / f"cycle_{CYCLE}_spectrum_census.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {json_path}", file=sys.stderr)

    # Build markdown table
    lines = [
        f"# Cycle {CYCLE}: pgolf Weight Spectrum Census",
        "",
        f"**Verdict**: {verdict} — {verdict_detail}",
        "",
        "## Per-block Results",
        "",
        "| Block | Shape | top-1% | top-5% | top-13% | rank50 | rank90 | rank95 |",
        "|-------|-------|--------|--------|---------|--------|--------|--------|",
    ]
    for i in range(13):
        if i not in results:
            continue
        r = results[i]
        shape_str = f"{r['shape'][0]}×{r['shape'][1]}"
        lines.append(
            f"| {i:2d} | {shape_str} | {r['top1']*100:.2f} | {r['top5']*100:.2f} | "
            f"{r['top13']*100:.2f} | {r['rank50']} | {r['rank90']} | {r['rank95']} |"
        )

    lines += [
        "",
        "## LASER Viability",
        "",
        "LASER (Low-Rank Adaptation of SVD) requires top-5 energy >= 15% to be useful.",
        f"Max top-5 across all 13 blocks: {max_top5*100:.2f}%",
        "",
        f"**Conclusion**: {verdict} — {verdict_detail}",
    ]

    summary_path = OUT_DIR / f"cycle_{CYCLE}_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {summary_path}", file=sys.stderr)

    # Write manifest
    manifest_lines = [
        f"cycle: {CYCLE}",
        f"item: 346",
        f"verdict: {verdict}",
        f"verdict_detail: {verdict_detail}",
        f"max_top5_pct: {max_top5*100:.4f}",
        f"min_top5_pct: {min_top5*100:.4f}",
        f"n_blocks_measured: {len(results)}",
        f"files:",
        f"  - autoresearch/results/cycle_{CYCLE}_spectrum_census.json",
        f"  - autoresearch/results/cycle_{CYCLE}_summary.md",
        f"  - autoresearch/results/cycle_{CYCLE}_manifest.txt",
    ]
    manifest_path = OUT_DIR / f"cycle_{CYCLE}_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print(f"Wrote {manifest_path}", file=sys.stderr)

    # Print JSON to stdout
    print(json.dumps(output, indent=2))
    return verdict


if __name__ == "__main__":
    main()
