#!/usr/bin/env python3
"""QAT autoresearch cycle 346 / item 356 — centroid affinity as compression predictor on pgolf.

Replicates cycle 27's finding (centroid affinity rho=0.853 on GPT-2) on the pgolf 13L_640d model.
For each MLP block: compute output centroid, centroid affinity (mean cosine with centroid),
and single-block swap damage (BPB when replacing MLP output with centroid).
Correlate affinity with swap damage via Spearman rho.

Verdict:
  CONFIRMED if |rho| > 0.7
  REFUTED if |rho| < 0.3
  INCONCLUSIVE if 0.3 <= |rho| <= 0.7
"""
import json, subprocess, sys, time
from pathlib import Path
from scipy import stats

PGOLF_PYTHON = "/Users/mihai/Projects/parameter-golf/.venv/bin/python"
PGOLF_SCRIPT = "/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_centroid_affinity.py"
CHECKPOINT = "/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz"
PGOLF_ROOT = "/Users/mihai/Projects/parameter-golf"

N_TOKENS = 25600
SEQ_LEN = 1024

OUT_DIR = Path("autoresearch/results")
CYCLE = 346


def main():
    t0 = time.time()
    print(f"Running centroid affinity measurement on pgolf ({N_TOKENS} tokens)...", flush=True)

    proc = subprocess.run(
        [
            PGOLF_PYTHON, PGOLF_SCRIPT,
            "--checkpoint", CHECKPOINT,
            "--n-tokens", str(N_TOKENS),
            "--seq-len", str(SEQ_LEN),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=PGOLF_ROOT,
    )
    elapsed = time.time() - t0

    if proc.stderr.strip():
        print("--- pgolf stderr ---", flush=True)
        print(proc.stderr, flush=True)

    if proc.returncode != 0:
        print(f"FAILED (exit {proc.returncode}) after {elapsed:.1f}s", flush=True)
        sys.exit(1)

    result = json.loads(proc.stdout)
    result["elapsed_seconds"] = elapsed

    # Save raw JSON
    json_path = OUT_DIR / f"cycle_{CYCLE}_pgolf_centroid_affinity.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"Saved: {json_path}", flush=True)

    # Extract arrays for correlation
    blocks = result["blocks"]
    n_blocks = result["n_blocks"]
    affinities = [blocks[str(l)]["centroid_affinity"] for l in range(n_blocks)]
    swap_deltas = [blocks[str(l)]["swap_delta"] for l in range(n_blocks)]

    # Spearman correlation: affinity vs swap damage
    rho, pval = stats.spearmanr(affinities, swap_deltas)
    result["spearman_rho"] = rho
    result["spearman_pval"] = pval

    # Verdict
    abs_rho = abs(rho)
    if abs_rho > 0.7:
        verdict = "CONFIRMED"
    elif abs_rho < 0.3:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    result["verdict"] = verdict

    # Re-save with correlation
    json_path.write_text(json.dumps(result, indent=2))

    # Build manifest
    manifest_lines = [
        f"cycle: {CYCLE}",
        f"item: 356",
        f"verdict: {verdict}",
        f"date: 2026-04-09",
        "",
        f"## Centroid affinity as compression predictor on pgolf 13L_640d",
        "",
        f"{verdict}: Spearman rho={rho:.3f} (p={pval:.4f}), cycle 27 GPT-2 rho=0.853",
        "",
        f"Checkpoint: {CHECKPOINT}",
        f"Tokens: {N_TOKENS} (seq_len={SEQ_LEN})",
        f"Baseline BPB: {result['baseline_bpb']:.6f}",
        "",
        f"{'Block':>5} {'Affinity':>10} {'CentNorm':>10} {'SwapBPB':>10} {'SwapDelta':>10}",
    ]
    for l in range(n_blocks):
        b = blocks[str(l)]
        manifest_lines.append(
            f"{l:>5} {b['centroid_affinity']:>10.4f} {b['centroid_norm']:>10.4f} "
            f"{b['swap_bpb']:>10.6f} {b['swap_delta']:>+10.6f}"
        )

    manifest_lines.extend([
        "",
        "## Correlation",
        f"Spearman rho = {rho:.4f} (p = {pval:.4f})",
        f"Interpretation: {'strong' if abs_rho > 0.7 else 'weak' if abs_rho < 0.3 else 'moderate'} "
        f"{'negative' if rho < 0 else 'positive'} correlation",
        f"(Higher affinity = tokens more aligned with centroid = MLP doing less unique work)",
        "",
        "## Verdict Criteria",
        "CONFIRMED if |rho| > 0.7",
        "REFUTED if |rho| < 0.3",
        "INCONCLUSIVE if 0.3 <= |rho| <= 0.7",
        "",
        "## Artifacts",
        f"experiments/pgolf_centroid_affinity_cycle{CYCLE}.py",
        f"parameter-golf/local_scripts/pgolf_centroid_affinity.py",
        f"autoresearch/results/cycle_{CYCLE}_pgolf_centroid_affinity.json",
        f"autoresearch/results/cycle_{CYCLE}_manifest.txt",
    ])

    manifest_path = OUT_DIR / f"cycle_{CYCLE}_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print(f"Saved: {manifest_path}", flush=True)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"VERDICT: {verdict} (rho={rho:.3f}, p={pval:.4f})", flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
