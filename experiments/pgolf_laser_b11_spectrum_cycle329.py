#!/usr/bin/env python3
"""Cycle 329 — LASER b11 spectrum preflight (numpy-only, no MLX needed).

Loads the checkpoint as raw numpy, computes SVD of blocks.11.mlp.proj.weight,
reports top-13 energy fraction. Gate: if < 10%, REFUTED-flat-spectrum.
"""
import json, sys
import numpy as np
from pathlib import Path

CHECKPOINT = "/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz"
TARGET_BLOCK = 11
RANK = 13
GATE_THRESHOLD = 0.10

OUT_DIR = Path("autoresearch/results")
CYCLE = 329


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {CHECKPOINT}", file=sys.stderr)
    saved = np.load(CHECKPOINT, allow_pickle=True)

    key = f"blocks.{TARGET_BLOCK}.mlp.proj.weight"
    if key not in saved.files:
        print(f"ERROR: key '{key}' not found in checkpoint", file=sys.stderr)
        print(f"Available block keys: {[k for k in saved.files if k.startswith(f'blocks.{TARGET_BLOCK}')]}", file=sys.stderr)
        sys.exit(1)

    W = saved[key].astype(np.float32)
    print(f"b{TARGET_BLOCK}.mlp.proj.weight shape: {W.shape}", file=sys.stderr)

    # Full SVD
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    n_sv = len(S)

    total_energy = float(np.sum(S ** 2))
    frob_norm_sq = float(np.linalg.norm(W, 'fro') ** 2)
    rel_error = abs(total_energy - frob_norm_sq) / frob_norm_sq
    print(f"Frobenius^2: {frob_norm_sq:.6f}  sum(S^2): {total_energy:.6f}  rel_error: {rel_error:.2e}", file=sys.stderr)

    top_r_energy = float(np.sum(S[:RANK] ** 2))
    top_r_fraction = top_r_energy / total_energy

    print(f"n_singular_values: {n_sv}", file=sys.stderr)
    print(f"S[:20] = {S[:20].round(4).tolist()}", file=sys.stderr)
    print(f"top-{RANK} energy fraction: {top_r_fraction:.6f} ({100*top_r_fraction:.2f}%)", file=sys.stderr)

    # Cumulative energy fractions at checkpoints
    cumulative = np.cumsum(S ** 2) / total_energy
    rank_thresholds = {}
    for thresh in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        r = int(np.searchsorted(cumulative, thresh)) + 1
        rank_thresholds[str(thresh)] = r
    print(f"Rank thresholds: {rank_thresholds}", file=sys.stderr)

    gate_pass = bool(top_r_fraction >= GATE_THRESHOLD)

    result = {
        "block": TARGET_BLOCK,
        "rank_checked": RANK,
        "weight_shape": list(W.shape),
        "n_singular_values": n_sv,
        "singular_values_top20": S[:20].tolist(),
        "top_r_energy_fraction": round(top_r_fraction, 6),
        "top_r_energy_pct": round(100.0 * top_r_fraction, 4),
        "total_energy": round(total_energy, 6),
        "frobenius_sq": round(frob_norm_sq, 6),
        "rank_thresholds": rank_thresholds,
        "gate_pass": gate_pass,
    }

    # Save spectrum JSON
    spectrum_path = OUT_DIR / f"cycle_{CYCLE}_spectrum.json"
    spectrum_path.write_text(json.dumps(result, indent=2))
    print(f"\nSpectrum JSON: {spectrum_path}", file=sys.stderr)

    # Determine verdict
    if gate_pass:
        verdict = "GATE-PASS"
        print(f"\nGate PASSED: top-{RANK} energy = {100*top_r_fraction:.2f}% >= {100*GATE_THRESHOLD:.0f}%", file=sys.stderr)
    else:
        verdict = "REFUTED-flat-spectrum"
        print(f"\nGate FAILED: top-{RANK} energy = {100*top_r_fraction:.2f}% < {100*GATE_THRESHOLD:.0f}%", file=sys.stderr)

    print(f"VERDICT: {verdict}", file=sys.stderr)

    # Write summary
    summary_lines = [
        f"# Cycle {CYCLE} — LASER b{TARGET_BLOCK} Spectrum Preflight",
        "",
        f"**Checkpoint**: `13L_640d_322_7k_wd70_batch20k_best.npz`",
        f"**Target block**: b{TARGET_BLOCK} (`mlp.proj.weight` shape {list(W.shape)})",
        f"**Gate threshold**: top-{RANK} energy >= {100*GATE_THRESHOLD:.0f}%",
        "",
        "## Spectrum Results",
        "",
        f"- `proj.weight` shape: {list(W.shape)}",
        f"- n_singular_values: {n_sv}",
        f"- top-{RANK} energy fraction: **{top_r_fraction:.6f} ({100*top_r_fraction:.2f}%)**",
        f"- Gate pass: **{gate_pass}** (threshold: {100*GATE_THRESHOLD:.0f}%)",
        f"- Rank thresholds: {rank_thresholds}",
        f"- S[:10]: {[round(s, 4) for s in S[:10].tolist()]}",
        "",
        "## Verdict",
        "",
        f"**{verdict}**" + (f": top-{RANK} energy = {100*top_r_fraction:.2f}% < 10% gate threshold." if not gate_pass else ""),
        "",
        "## Comparison to cycles 327-328",
        "",
        f"| Block | Hidden | Rank | Energy% | Verdict |",
        f"|-------|--------|------|---------|---------|",
        f"| b3    | 1920   | 5    | 2.60%   | REFUTED-flat-spectrum |",
        f"| b12   | 320    | 7    | 7.07%   | REFUTED-flat-spectrum |",
        f"| b{TARGET_BLOCK}   | 1280   | {RANK}   | {100*top_r_fraction:.2f}%   | {verdict} |",
        "",
    ]
    if not gate_pass:
        summary_lines += [
            f"**Pattern confirmed across 3 blocks**: output eff_rank is NOT a proxy for weight SVD compressibility.",
            "All three blocks (b3, b12, b11) have flat weight spectra regardless of hidden dim (320-1920).",
            "",
        ]

    summary_path = OUT_DIR / f"cycle_{CYCLE}_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Summary: {summary_path}", file=sys.stderr)

    # Write manifest
    manifest_lines = [
        f"cycle: {CYCLE}",
        f"item: 342",
        f"verdict: {verdict}",
        f"target_block: {TARGET_BLOCK}",
        f"weight_shape: {list(W.shape)}",
        f"top_{RANK}_energy_pct: {100*top_r_fraction:.4f}",
        f"gate_pass: {gate_pass}",
        f"n_singular_values: {n_sv}",
        f"rank_thresholds: {rank_thresholds}",
        f"s_top5: {[round(s, 4) for s in S[:5].tolist()]}",
    ]
    manifest_path = OUT_DIR / f"cycle_{CYCLE}_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print(f"Manifest: {manifest_path}", file=sys.stderr)

    # JSON to stdout for automation
    print(json.dumps({"verdict": verdict, "top_r_energy_pct": round(100*top_r_fraction, 4), "gate_pass": gate_pass}))


if __name__ == "__main__":
    main()
