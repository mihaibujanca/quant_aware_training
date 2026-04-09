#!/usr/bin/env python3
"""Cycle 345: Activation pattern census on pgolf 13L_640d checkpoint.

For each MLP block, count unique ReLU activation patterns across validation
tokens. K* = number of unique patterns = number of distinct piecewise-linear
regions the MLP uses. If K*/width < 0.1 at >= 3 blocks, compression via
piecewise-linear approximation is viable (CONFIRMED).

Subprocess invocation pattern: this script runs from qat venv, calls the
pgolf-side script via pgolf's Python interpreter.
"""
import json, subprocess, sys, datetime
from pathlib import Path

PGOLF_PYTHON = "/Users/mihai/Projects/parameter-golf/.venv/bin/python"
PGOLF_SCRIPT = "/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_activation_pattern_census.py"
CHECKPOINT = "/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz"
CYCLE = 345
N_TOKENS = 25600


def main():
    print(f"[cycle {CYCLE}] Running activation pattern census on pgolf...", file=sys.stderr)
    proc = subprocess.run(
        [PGOLF_PYTHON, PGOLF_SCRIPT,
         "--checkpoint", CHECKPOINT,
         "--n-tokens", str(N_TOKENS),
         "--seq-len", "1024"],
        capture_output=True, text=True, timeout=1800,
        cwd="/Users/mihai/Projects/parameter-golf",
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"pgolf script failed: {proc.returncode}")

    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    result = json.loads(proc.stdout)

    # Save raw JSON
    out_dir = Path("autoresearch/results")
    out_dir.mkdir(exist_ok=True)
    raw_path = out_dir / f"cycle_{CYCLE}_pgolf_activation_patterns.json"
    raw_path.write_text(json.dumps(result, indent=2))
    print(f"[cycle {CYCLE}] Raw results: {raw_path}", file=sys.stderr)

    # Analyze and print summary
    profile = result["profile"]
    n_blocks = result["n_blocks"]

    print(f"\n{'Block':>5} {'Hidden':>6} {'K*':>8} {'K*/width':>10} {'Avg Active':>12} {'Active Frac':>12}")
    print("-" * 60)

    confirmed_blocks = 0
    refuted_blocks = 0
    for l in range(n_blocks):
        p = profile[str(l)]
        K = p["K_star"]
        H = p["hidden_dim"]
        ratio = p["K_star_over_width"]
        avg_a = p["avg_active_neurons"]
        afrac = p["avg_active_frac"]
        print(f"{l:>5} {H:>6} {K:>8} {ratio:>10.4f} {avg_a:>12.1f} {afrac:>12.4f}")

        if ratio < 0.1:
            confirmed_blocks += 1
        if ratio > 0.5:
            refuted_blocks += 1

    # Verdict
    if confirmed_blocks >= 3:
        verdict = "CONFIRMED"
        verdict_detail = f"K*/width < 0.1 at {confirmed_blocks}/{n_blocks} blocks"
    elif refuted_blocks == n_blocks:
        verdict = "REFUTED"
        verdict_detail = f"K*/width > 0.5 at all {n_blocks} blocks"
    else:
        verdict = "INCONCLUSIVE"
        verdict_detail = f"{confirmed_blocks} blocks with K*/width < 0.1, {refuted_blocks} with K*/width > 0.5"

    print(f"\nVerdict: {verdict} — {verdict_detail}")

    # Write manifest
    manifest_lines = [
        f"cycle: {CYCLE}",
        f"item: 354",
        f"verdict: {verdict}",
        f"date: {datetime.date.today().isoformat()}",
        f"",
        f"## Jacobian clustering proxy: activation pattern census on pgolf 13L_640d",
        f"",
        f"{verdict}: {verdict_detail}",
        f"",
        f"Checkpoint: {CHECKPOINT}",
        f"Tokens: {N_TOKENS} (seq_len=1024)",
        f"",
    ]

    # Per-block table
    manifest_lines.append(f"{'Block':>5} {'Hidden':>6} {'K*':>8} {'K*/width':>10} {'AvgActive':>10} {'ActFrac':>8}")
    for l in range(n_blocks):
        p = profile[str(l)]
        manifest_lines.append(
            f"{l:>5} {p['hidden_dim']:>6} {p['K_star']:>8} {p['K_star_over_width']:>10.4f} "
            f"{p['avg_active_neurons']:>10.1f} {p['avg_active_frac']:>8.4f}"
        )

    manifest_lines.extend([
        "",
        "## Interpretation",
        f"K* = unique ReLU activation patterns across {N_TOKENS} tokens.",
        "For ReLU² MLP, each activation pattern defines one local polynomial map.",
        "K*/width < 0.1 means the block uses far fewer distinct behaviors than its width allows.",
        "",
        "## Verdict Criteria",
        "CONFIRMED if K*/width < 0.1 at >= 3 blocks.",
        "REFUTED if K*/width > 0.5 at all blocks.",
        "INCONCLUSIVE otherwise.",
        "",
        "## Artifacts",
        f"experiments/pgolf_activation_pattern_census_cycle345.py",
        f"parameter-golf/local_scripts/pgolf_activation_pattern_census.py",
        f"autoresearch/results/cycle_{CYCLE}_pgolf_activation_patterns.json",
        f"autoresearch/results/cycle_{CYCLE}_manifest.txt",
        "",
    ])

    manifest_path = out_dir / f"cycle_{CYCLE}_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print(f"[cycle {CYCLE}] Manifest: {manifest_path}", file=sys.stderr)

    return verdict


if __name__ == "__main__":
    verdict = main()
    sys.exit(0 if verdict != "ERROR" else 1)
