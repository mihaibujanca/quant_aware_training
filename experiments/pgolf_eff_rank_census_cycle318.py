#!/usr/bin/env python3
"""Cycle 318 / Item 333 — qat-side wrapper.

Runs the parameter-golf pgolf_eff_rank_census.py via subprocess (PyTorch/MLX
isolation, see autoresearch/cross_project/parameter_golf.md), parses the JSON
result, writes cycle_318_pgolf_eff_rank.json + cycle_318_summary.md.

Verdict thresholds (item 333):
- CONFIRMED-collapse-generalizes: any block eff_rank ≤ 20
- CONFIRMED-no-slack: all 13 blocks eff_rank ≥ 100
- CONFIRMED-different-shape: some ≤ 20, some ≥ 100
- INCONCLUSIVE: all in (20, 100)
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
CHECKPOINT = f"{PGOLF_ROOT}/logs/13L_640d_322_7k_wd70_batch20k_best.npz"
PGOLF_DATA_PATH = f"{PGOLF_ROOT}/data/datasets/fineweb10B_sp1024"

CYCLE = 318
N_TOKENS = 25600
SEQ_LEN = 1024

# Cycle 278 GPT-2 baseline (from autoresearch/results/cycle_278_all_blocks_eff_rank.json)
GPT2_CYCLE_278_EFF_RANK = {
    0: 71.93,
    1: 1.94,
    2: 1.05,
    3: 24.29,
    4: 49.04,
    5: 179.19,
    6: 290.97,
    7: 297.79,
    8: 291.41,
    9: 195.50,
    10: 16.70,
    11: 22.49,
}


def write_summary(result: dict, summary_path: Path) -> str:
    profile = result["profile"]
    profile = {int(k): v for k, v in profile.items()}
    n_blocks = result["n_blocks"]

    eff_ranks = [profile[l]["eff_rank"] for l in range(n_blocks)]
    rank_95s = [profile[l]["rank_95"] for l in range(n_blocks)]
    dims = [profile[l]["dim"] for l in range(n_blocks)]

    n_collapsed = sum(1 for er in eff_ranks if er <= 20)
    n_full = sum(1 for er in eff_ranks if er >= 100)

    if n_collapsed > 0 and n_full == n_blocks - n_collapsed and n_full > 0:
        verdict = "CONFIRMED-different-shape"
    elif n_collapsed > 0 and n_full > 0:
        verdict = "CONFIRMED-different-shape"
    elif n_collapsed > 0:
        verdict = "CONFIRMED-collapse-generalizes"
    elif n_full == n_blocks:
        verdict = "CONFIRMED-no-slack"
    else:
        verdict = "INCONCLUSIVE"

    lines = []
    lines.append("# Cycle 318 — Item 333 Summary")
    lines.append("")
    lines.append("Output-covariance eff_rank census on parameter-golf 13L_640d ReLU² checkpoint.")
    lines.append("")
    lines.append(f"**Checkpoint**: `{result['checkpoint']}`")
    lines.append(f"**n_tokens**: {result['n_tokens_requested']} ({SEQ_LEN}-token seqs)")
    lines.append(f"**Activation**: ReLU² (no bias on `mlp.fc` / `mlp.proj`)")
    lines.append("")
    lines.append("## Per-block profile (pre-`mlp_scale` MLP outputs)")
    lines.append("")
    lines.append("| block | dim | eff_rank | rank_95 | rank_99 | top1 / λ_sum |")
    lines.append("|---|---|---|---|---|---|")
    for l in range(n_blocks):
        p = profile[l]
        ratio = p["top5_eigvals"][0] / p["trace"] if p["trace"] > 0 else 0.0
        lines.append(
            f"| {l} | {p['dim']} | {p['eff_rank']:.2f} | {p['rank_95']} | "
            f"{p['rank_99']} | {ratio:.4f} |"
        )
    lines.append("")
    lines.append(f"**Min eff_rank**: {min(eff_ranks):.2f} (block {eff_ranks.index(min(eff_ranks))})")
    lines.append(f"**Max eff_rank**: {max(eff_ranks):.2f} (block {eff_ranks.index(max(eff_ranks))})")
    lines.append(
        f"**# blocks ≤ 20**: {n_collapsed} / {n_blocks}, **# blocks ≥ 100**: {n_full} / {n_blocks}"
    )
    lines.append("")
    lines.append("## Comparison to qat cycle 278 (GPT-2 small, 12 blocks)")
    lines.append("")
    lines.append("GPT-2 small showed a U-shape: extreme collapse at b1/b2 (eff_rank ~1-2), recovery to ~150-200 mid-stack, narrowing again at b11 (~22).")
    lines.append("")
    lines.append("| block | GPT-2 (cycle 278) | parameter-golf 13L_640d |")
    lines.append("|---|---|---|")
    for l in range(max(n_blocks, len(GPT2_CYCLE_278_EFF_RANK))):
        gpt2 = f"{GPT2_CYCLE_278_EFF_RANK[l]:.2f}" if l in GPT2_CYCLE_278_EFF_RANK else "—"
        pgolf = f"{profile[l]['eff_rank']:.2f}" if l < n_blocks else "—"
        lines.append(f"| {l} | {gpt2} | {pgolf} |")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append("Verdict logic (item 333):")
    lines.append("- CONFIRMED-collapse-generalizes: any block eff_rank ≤ 20")
    lines.append("- CONFIRMED-no-slack: all blocks eff_rank ≥ 100")
    lines.append("- CONFIRMED-different-shape: some ≤ 20, some ≥ 100")
    lines.append("- INCONCLUSIVE: all in (20, 100)")
    lines.append("")
    lines.append(f"**Trainable params**: 0 (forward-only measurement, no optimizer step).")
    lines.append("")
    summary_path.write_text("\n".join(lines))
    return verdict


def main() -> int:
    out_dir = Path("autoresearch/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PGOLF_PYTHON,
        PGOLF_SCRIPT,
        "--checkpoint",
        CHECKPOINT,
        "--n-tokens",
        str(N_TOKENS),
        "--seq-len",
        str(SEQ_LEN),
    ]
    print(f"[cycle 318] running: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["DATA_PATH"] = PGOLF_DATA_PATH
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1200, env=env, cwd=PGOLF_ROOT
    )
    if proc.returncode != 0:
        sys.stderr.write("STDERR:\n" + proc.stderr + "\n")
        sys.stderr.write("STDOUT:\n" + proc.stdout + "\n")
        raise SystemExit(f"pgolf script failed: {proc.returncode}")

    result = json.loads(proc.stdout)
    json_path = out_dir / f"cycle_{CYCLE}_pgolf_eff_rank.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"[cycle 318] wrote {json_path}", flush=True)

    summary_path = out_dir / f"cycle_{CYCLE}_summary.md"
    verdict = write_summary(result, summary_path)
    print(f"[cycle 318] wrote {summary_path}", flush=True)
    print(f"[cycle 318] VERDICT: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
