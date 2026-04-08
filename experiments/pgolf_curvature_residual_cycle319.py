#!/usr/bin/env python3
"""Cycle 319 / Item 334 — qat-side wrapper.

Runs `parameter-golf/local_scripts/pgolf_curvature_residual.py` three times
(one per (σ, σ_lin) pair), parses the JSON output, writes a combined result
file and a summary with verdict.

Verdict tree (item 334 verbatim):
- CONFIRMED-leak-localized if any block has (lrelu²,relu²) `rank_95(Δy) ≤ 50`
  AND ‖Δy‖_F at those blocks ≥ 2× the median across blocks.
- CONFIRMED-leak-diffuse if (lrelu²,relu²) `rank_95(Δy) ≥ 200` at every block.
- REFUTED-no-difference if (lrelu²,relu²) ‖Δy‖_F is within 5% of (relu²,relu)
  at every block.
- INCONCLUSIVE otherwise.
"""
from __future__ import annotations
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

PGOLF_ROOT = "/Users/mihai/Projects/parameter-golf"
PGOLF_PYTHON = f"{PGOLF_ROOT}/.venv/bin/python"
PGOLF_SCRIPT = f"{PGOLF_ROOT}/local_scripts/pgolf_curvature_residual.py"
CHECKPOINT = f"{PGOLF_ROOT}/logs/13L_640d_322_7k_wd70_batch20k_best.npz"
PGOLF_DATA_PATH = f"{PGOLF_ROOT}/data/datasets/fineweb10B_sp1024"

CYCLE = 319
N_TOKENS = 25600
SEQ_LEN = 1024

PAIRS = [
    ("relu2", "relu"),
    ("lrelu2", "relu2"),
    ("gelu", "relu"),
]

# Cycle 318 reference: parameter-golf eff_rank profile
PGOLF_CYCLE_318_EFF_RANK = {
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


def run_pair(sigma: str, sigma_lin: str) -> dict:
    cmd = [
        PGOLF_PYTHON, PGOLF_SCRIPT,
        "--checkpoint", CHECKPOINT,
        "--sigma", sigma,
        "--sigma-lin", sigma_lin,
        "--n-tokens", str(N_TOKENS),
        "--seq-len", str(SEQ_LEN),
    ]
    print(f"[cycle 319] running: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["DATA_PATH"] = PGOLF_DATA_PATH
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800, env=env, cwd=PGOLF_ROOT
    )
    if proc.returncode != 0:
        sys.stderr.write(f"STDERR ({sigma}, {sigma_lin}):\n{proc.stderr}\n")
        sys.stderr.write(f"STDOUT ({sigma}, {sigma_lin}):\n{proc.stdout}\n")
        raise SystemExit(f"pgolf script failed for ({sigma}, {sigma_lin}): {proc.returncode}")
    return json.loads(proc.stdout)


def write_summary(combined: dict, summary_path: Path) -> str:
    pairs = combined["pairs"]  # list of {sigma, sigma_lin, profile}
    n_blocks = pairs[0]["n_blocks"]

    def get(pair_name, field, l):
        for p in pairs:
            key = f"{p['sigma']}__vs__{p['sigma_lin']}"
            if key == pair_name:
                return p["profile"][str(l)][field]
        return None

    relu2_relu_key = "relu2__vs__relu"
    lrelu2_relu2_key = "lrelu2__vs__relu2"
    gelu_relu_key = "gelu__vs__relu"

    # Verdict logic
    lrelu_ranks = [get(lrelu2_relu2_key, "rank_95", l) for l in range(n_blocks)]
    lrelu_frobs = [get(lrelu2_relu2_key, "frob_norm", l) for l in range(n_blocks)]
    relu_frobs = [get(relu2_relu_key, "frob_norm", l) for l in range(n_blocks)]

    median_lrelu_frob = statistics.median(lrelu_frobs)
    localized_blocks = [
        l for l in range(n_blocks)
        if lrelu_ranks[l] <= 50 and lrelu_frobs[l] >= 2.0 * median_lrelu_frob
    ]
    all_diffuse = all(r >= 200 for r in lrelu_ranks)
    rel_diffs = [
        abs(lrelu_frobs[l] - relu_frobs[l]) / max(relu_frobs[l], 1e-12)
        for l in range(n_blocks)
    ]
    all_within_5pc = all(d <= 0.05 for d in rel_diffs)

    if localized_blocks:
        verdict = "CONFIRMED-leak-localized"
    elif all_diffuse:
        verdict = "CONFIRMED-leak-diffuse"
    elif all_within_5pc:
        verdict = "REFUTED-no-difference"
    else:
        verdict = "INCONCLUSIVE"

    lines = []
    lines.append(f"# Cycle {CYCLE} — Item 334 Summary")
    lines.append("")
    lines.append("Curvature residual decomposition on parameter-golf 13L_640d ReLU² checkpoint.")
    lines.append("")
    lines.append(f"**Checkpoint**: `{combined['checkpoint']}`")
    lines.append(f"**n_tokens**: {N_TOKENS} ({SEQ_LEN}-token seqs); 3 (σ, σ_lin) pairs.")
    lines.append("")
    lines.append("## (relu2, relu) — relu²'s curvature over relu baseline")
    lines.append("")
    lines.append("| block | hidden | rank_95(Δy) | rank_99 | top1_sv | ‖Δy‖_F | cycle 318 eff_rank |")
    lines.append("|---|---|---|---|---|---|---|")
    for l in range(n_blocks):
        prof = next(p for p in pairs if p["sigma"] == "relu2" and p["sigma_lin"] == "relu")["profile"][str(l)]
        eff = PGOLF_CYCLE_318_EFF_RANK.get(l, "—")
        lines.append(
            f"| {l} | {prof['hidden_dim']} | {prof['rank_95']} | {prof['rank_99']} | "
            f"{prof['top5_sv'][0]:.3f} | {prof['frob_norm']:.3f} | {eff} |"
        )
    lines.append("")
    lines.append("## (lrelu2, relu2) — leaky-relu²'s leak over relu² (the leaderboard candidate)")
    lines.append("")
    lines.append("| block | hidden | rank_95(Δy) | rank_99 | top1_sv | ‖Δy‖_F | rel diff vs (relu²,relu) ‖Δy‖_F |")
    lines.append("|---|---|---|---|---|---|---|")
    for l in range(n_blocks):
        prof = next(p for p in pairs if p["sigma"] == "lrelu2" and p["sigma_lin"] == "relu2")["profile"][str(l)]
        rel = rel_diffs[l]
        lines.append(
            f"| {l} | {prof['hidden_dim']} | {prof['rank_95']} | {prof['rank_99']} | "
            f"{prof['top5_sv'][0]:.3f} | {prof['frob_norm']:.3f} | {rel:.4f} |"
        )
    lines.append("")
    lines.append("## (gelu, relu) — cross-reference, structurally analogous to GPT-2 cycle 302")
    lines.append("")
    lines.append("| block | hidden | rank_95(Δy) | rank_99 | top1_sv | ‖Δy‖_F |")
    lines.append("|---|---|---|---|---|---|")
    for l in range(n_blocks):
        prof = next(p for p in pairs if p["sigma"] == "gelu" and p["sigma_lin"] == "relu")["profile"][str(l)]
        lines.append(
            f"| {l} | {prof['hidden_dim']} | {prof['rank_95']} | {prof['rank_99']} | "
            f"{prof['top5_sv'][0]:.3f} | {prof['frob_norm']:.3f} |"
        )
    lines.append("")
    lines.append(f"**(lrelu², relu²) median ‖Δy‖_F**: {median_lrelu_frob:.3f}")
    lines.append(f"**(lrelu², relu²) blocks with rank_95 ≤ 50 AND ‖Δy‖_F ≥ 2× median**: {localized_blocks}")
    lines.append(f"**(lrelu², relu²) all blocks rank_95 ≥ 200**: {all_diffuse}")
    lines.append(f"**(lrelu², relu²) ‖Δy‖_F within 5% of (relu², relu) at every block**: {all_within_5pc}")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append("Verdict logic (item 334):")
    lines.append("- CONFIRMED-leak-localized: any (lrelu²,relu²) block with rank_95 ≤ 50 AND ‖Δy‖_F ≥ 2× median")
    lines.append("- CONFIRMED-leak-diffuse: all (lrelu²,relu²) blocks have rank_95 ≥ 200")
    lines.append("- REFUTED-no-difference: (lrelu²,relu²) ‖Δy‖_F within 5% of (relu²,relu) at every block")
    lines.append("- INCONCLUSIVE otherwise")
    lines.append("")
    lines.append("**Trainable params**: 0 (forward-only measurement, no optimizer step).")
    lines.append("")
    summary_path.write_text("\n".join(lines))
    return verdict


def main() -> int:
    out_dir = Path("autoresearch/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_results = []
    for sigma, sigma_lin in PAIRS:
        result = run_pair(sigma, sigma_lin)
        pair_results.append(result)

    combined = {
        "checkpoint": CHECKPOINT,
        "n_tokens_requested": N_TOKENS,
        "seq_len": SEQ_LEN,
        "pairs": pair_results,
    }
    json_path = out_dir / f"cycle_{CYCLE}_pgolf_curvature_residual.json"
    json_path.write_text(json.dumps(combined, indent=2))
    print(f"[cycle {CYCLE}] wrote {json_path}", flush=True)

    summary_path = out_dir / f"cycle_{CYCLE}_summary.md"
    verdict = write_summary(combined, summary_path)
    print(f"[cycle {CYCLE}] wrote {summary_path}", flush=True)
    print(f"[cycle {CYCLE}] VERDICT: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
