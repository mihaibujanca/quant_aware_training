#!/usr/bin/env python3
"""Cycle 322 / Item 335 — qat-side wrapper.

Runs parameter-golf pgolf_eff_rank_census.py on 3 checkpoints via subprocess
(PyTorch/MLX isolation, see autoresearch/cross_project/parameter_golf.md).
Collects 3 results, writes cycle_335_pgolf_eff_rank_multi.json and
cycle_335_summary.md. Verdict: is the min-eff_rank block stable across
checkpoint variants?

Verdict thresholds (item 335):
- CONFIRMED-stable: min-eff_rank block is same (or within ±1 layer for same
  depth; ±0.1 fractional position for cross-depth) in all 3 checkpoints.
- REFUTED-checkpoint-specific: min-eff_rank block varies by ≥3 layers (same
  depth) or ≥0.3 fractional position (cross-depth).
- INCONCLUSIVE: some shift but within thresholds.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

PGOLF_ROOT = "/Users/mihai/Projects/parameter-golf"
PGOLF_PYTHON = f"{PGOLF_ROOT}/.venv/bin/python"
PGOLF_SCRIPT = f"{PGOLF_ROOT}/local_scripts/pgolf_eff_rank_census.py"
PGOLF_DATA_PATH = f"{PGOLF_ROOT}/data/datasets/fineweb10B_sp1024"

CYCLE = 322
ITEM = 335
N_TOKENS = 25600
SEQ_LEN = 1024

CHECKPOINTS = [
    {
        "tag": "13L_640d_322_7k_wd70_batch20k_best",
        "path": f"{PGOLF_ROOT}/logs/13L_640d_322_7k_wd70_batch20k_best.npz",
        "expected_n_blocks": 13,
        "label": "13L (cycle 318 baseline)",
    },
    {
        "tag": "wd70_11L_5x_polished",
        "path": f"{PGOLF_ROOT}/logs/wd70_11L_5x_polished.npz",
        "expected_n_blocks": 11,
        "label": "11L polished",
    },
    {
        "tag": "10L_640d_multiskip_22k_wd70_polished",
        "path": f"{PGOLF_ROOT}/logs/10L_640d_multiskip_22k_wd70_polished.npz",
        "expected_n_blocks": 10,
        "label": "10L multiskip polished",
    },
]

# Cycle 278 GPT-2 baseline
GPT2_CYCLE_278_EFF_RANK = {
    0: 71.93, 1: 1.94, 2: 1.05, 3: 24.29, 4: 49.04,
    5: 179.19, 6: 290.97, 7: 297.79, 8: 291.41,
    9: 195.50, 10: 16.70, 11: 22.49,
}


def run_checkpoint(ckpt_info: dict) -> dict:
    cmd = [
        PGOLF_PYTHON, PGOLF_SCRIPT,
        "--checkpoint", ckpt_info["path"],
        "--n-tokens", str(N_TOKENS),
        "--seq-len", str(SEQ_LEN),
    ]
    print(f"[cycle {CYCLE}] running {ckpt_info['tag']}: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["DATA_PATH"] = PGOLF_DATA_PATH
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800, env=env, cwd=PGOLF_ROOT
    )
    if proc.returncode != 0:
        sys.stderr.write(f"STDERR ({ckpt_info['tag']}):\n" + proc.stderr + "\n")
        sys.stderr.write(f"STDOUT ({ckpt_info['tag']}):\n" + proc.stdout + "\n")
        raise SystemExit(f"pgolf script failed for {ckpt_info['tag']}: {proc.returncode}")
    result = json.loads(proc.stdout)
    result["tag"] = ckpt_info["tag"]
    result["label"] = ckpt_info["label"]
    result["expected_n_blocks"] = ckpt_info["expected_n_blocks"]
    # Validate n_blocks
    if result["n_blocks"] != ckpt_info["expected_n_blocks"]:
        print(
            f"  WARNING: expected {ckpt_info['expected_n_blocks']} blocks, "
            f"got {result['n_blocks']}",
            flush=True,
        )
    else:
        print(f"  OK: {result['n_blocks']} blocks", flush=True)
    return result


def min_eff_rank_block(result: dict) -> tuple[int, float]:
    profile = {int(k): v for k, v in result["profile"].items()}
    n = result["n_blocks"]
    min_idx = min(range(n), key=lambda l: profile[l]["eff_rank"])
    return min_idx, profile[min_idx]["eff_rank"]


def write_summary(all_results: list[dict], summary_path: Path) -> str:
    lines = []
    lines.append(f"# Cycle {CYCLE} — Item {ITEM} Summary")
    lines.append("")
    lines.append("Multi-checkpoint eff_rank census: is the min-eff_rank block stable?")
    lines.append("")

    # Per-checkpoint min blocks
    min_blocks = []
    for r in all_results:
        idx, er = min_eff_rank_block(r)
        frac = idx / r["n_blocks"]
        min_blocks.append((idx, er, frac, r["n_blocks"], r["label"]))
        lines.append(
            f"**{r['label']}** (`{r['tag']}`): "
            f"min eff_rank block = b{idx} ({er:.2f}), "
            f"frac pos = {frac:.3f}, n_blocks = {r['n_blocks']}"
        )
    lines.append("")

    # Side-by-side table: use max n_blocks as rows
    max_n = max(r["n_blocks"] for r in all_results)
    header = "| block |" + "".join(f" {r['tag'][:20]} eff_rank |" for r in all_results)
    sep = "|---|" + "---|" * len(all_results)
    lines.append("## Per-block eff_rank (side-by-side)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for l in range(max_n):
        row = f"| {l} |"
        for r in all_results:
            profile = {int(k): v for k, v in r["profile"].items()}
            if l < r["n_blocks"]:
                er = profile[l]["eff_rank"]
                marker = " **MIN**" if l == min_eff_rank_block(r)[0] else ""
                row += f" {er:.2f}{marker} |"
            else:
                row += " — |"
        lines.append(row)
    lines.append("")

    # Fractional positions of min blocks
    fracs = [mb[2] for mb in min_blocks]
    frac_spread = max(fracs) - min(fracs)
    abs_indices = [mb[0] for mb in min_blocks]
    abs_spread = max(abs_indices) - min(abs_indices)

    # Verdict logic (item 335)
    if frac_spread <= 0.1 and abs_spread <= 1:
        verdict = "CONFIRMED-stable"
    elif abs_spread >= 3 or frac_spread >= 0.3:
        verdict = "REFUTED-checkpoint-specific"
    else:
        verdict = "INCONCLUSIVE"

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append(
        f"Min-block absolute indices: {abs_indices} (spread={abs_spread})"
    )
    lines.append(
        f"Min-block fractional positions: {[f'{f:.3f}' for f in fracs]} (spread={frac_spread:.3f})"
    )
    lines.append("")
    lines.append("Verdict logic (item 335):")
    lines.append(
        "- CONFIRMED-stable: abs_spread <= 1 AND frac_spread <= 0.1 (b3 is canonical LASER target)"
    )
    lines.append(
        "- REFUTED-checkpoint-specific: abs_spread >= 3 OR frac_spread >= 0.3 (re-target per checkpoint)"
    )
    lines.append("- INCONCLUSIVE: otherwise")
    lines.append("")
    lines.append("## Cycle 318 baseline comparison (13L, b3 eff_rank ~4.89)")
    lines.append("")
    baseline_result = next((r for r in all_results if "13L_640d_322_7k" in r["tag"]), None)
    if baseline_result:
        profile = {int(k): v for k, v in baseline_result["profile"].items()}
        b3_er = profile.get(3, {}).get("eff_rank", "N/A")
        lines.append(
            f"13L checkpoint b3 eff_rank = {b3_er:.2f} (expected ~4.89 from cycle 318)"
            if isinstance(b3_er, float) else f"b3 eff_rank = {b3_er}"
        )
    lines.append("")
    summary_path.write_text("\n".join(lines))
    return verdict


def main() -> int:
    out_dir = Path("autoresearch/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ckpt_info in CHECKPOINTS:
        result = run_checkpoint(ckpt_info)
        all_results.append(result)

    # Write combined JSON
    json_path = out_dir / "cycle_335_pgolf_eff_rank_multi.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"[cycle {CYCLE}] wrote {json_path}", flush=True)

    summary_path = out_dir / "cycle_335_summary.md"
    verdict = write_summary(all_results, summary_path)
    print(f"[cycle {CYCLE}] wrote {summary_path}", flush=True)
    print(f"[cycle {CYCLE}] VERDICT: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
