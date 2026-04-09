#!/usr/bin/env python3
"""QAT autoresearch cycle 328 / item 341 — LASER spectrum preflight + conditional retrain on pgolf b12.

Step A: Run spectrum check on b12's mlp.proj.weight.
  - If top-7 energy < 10%: REFUTED-flat-spectrum, stop.
  - If top-7 energy >= 10%: proceed to Step B.

Step B: LASER retrain at ranks 7 (primary), 3, 14 (sensitivity).

Writes:
  autoresearch/results/cycle_328_spectrum.json
  autoresearch/results/cycle_328_pgolf_laser_retrain.json  (if gate passes)
  autoresearch/results/cycle_328_summary.md
"""
import json, subprocess, sys, time
from pathlib import Path

PGOLF_PYTHON = "/Users/mihai/Projects/parameter-golf/.venv/bin/python"
PGOLF_SPECTRUM_SCRIPT = "/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_laser_b12_spectrum.py"
PGOLF_RETRAIN_SCRIPT = "/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_laser_retrain.py"
CHECKPOINT = "/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz"
PGOLF_ROOT = "/Users/mihai/Projects/parameter-golf"

TARGET_BLOCK = 12
SPECTRUM_RANK = 7
GATE_THRESHOLD = 0.10  # 10% energy threshold

# Retrain config
RETRAIN_RANKS = [3, 7, 14]  # rank-7 primary; rank-3 and rank-14 for sensitivity
N_TRAIN_STEPS = 500
LR = 1e-4
SEQ_LEN = 1024
N_VAL_TOKENS = 65536

OUT_DIR = Path("autoresearch/results")
CYCLE = 328


def run_spectrum_check() -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"Step A: Spectrum preflight for b{TARGET_BLOCK}", flush=True)
    print(f"{'='*60}", flush=True)
    proc = subprocess.run(
        [
            PGOLF_PYTHON, PGOLF_SPECTRUM_SCRIPT,
            "--checkpoint", CHECKPOINT,
            "--target-block", str(TARGET_BLOCK),
            "--rank", str(SPECTRUM_RANK),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=PGOLF_ROOT,
    )
    if proc.stderr.strip():
        print(proc.stderr, file=sys.stderr, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Spectrum check failed (rc={proc.returncode})\n{proc.stdout}")
    return json.loads(proc.stdout.strip())


def run_one_rank_retrain(rank: int) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"Step B: LASER retrain rank={rank}, block={TARGET_BLOCK}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [
            PGOLF_PYTHON, PGOLF_RETRAIN_SCRIPT,
            "--checkpoint", CHECKPOINT,
            "--target-block", str(TARGET_BLOCK),
            "--rank", str(rank),
            "--n-train-steps", str(N_TRAIN_STEPS),
            "--lr", str(LR),
            "--seq-len", str(SEQ_LEN),
            "--n-val-tokens", str(N_VAL_TOKENS),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=PGOLF_ROOT,
    )
    elapsed = time.time() - t0
    if proc.stderr.strip():
        print(proc.stderr, file=sys.stderr, flush=True)
    if proc.returncode != 0:
        msg = f"Retrain failed for rank={rank} (rc={proc.returncode})"
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)
    result = json.loads(proc.stdout.strip())
    result["wall_seconds"] = round(elapsed, 1)
    return result


def determine_verdict(spectrum: dict, retrain_results: list[dict]) -> str:
    top_r_frac = spectrum["top_r_energy_fraction"]
    if not spectrum["gate_pass"]:
        return "REFUTED-flat-spectrum"
    if not retrain_results:
        return "INCONCLUSIVE"
    rank7 = next((r for r in retrain_results if r.get("rank") == 7), None)
    if rank7 is None:
        return "INCONCLUSIVE"
    delta = rank7["delta_retrained"]
    if delta <= 0.001:
        return "CONFIRMED-LASER-transfers"
    elif delta <= 0.005:
        return "CONFIRMED-partial"
    elif delta >= 0.01:
        return "REFUTED-LASER-fails"
    else:
        return "INCONCLUSIVE"


def write_summary(spectrum: dict, retrain_results: list[dict], verdict: str, out_path: Path) -> None:
    top_r_frac = spectrum["top_r_energy_fraction"]
    lines = [
        f"# Cycle 328 — LASER b12 Spectrum Preflight + Conditional Retrain",
        "",
        f"**Checkpoint**: `13L_640d_322_7k_wd70_batch20k_best.npz`",
        f"**Target block**: b{TARGET_BLOCK} (`mlp.proj.weight` shape {spectrum['weight_shape']})",
        f"**Gate threshold**: top-{SPECTRUM_RANK} energy ≥ {100*GATE_THRESHOLD:.0f}%",
        "",
        "## Spectrum Preflight",
        "",
        f"- `proj.weight` shape: {spectrum['weight_shape']}",
        f"- n_singular_values: {spectrum['n_singular_values']}",
        f"- top-{SPECTRUM_RANK} energy fraction: **{top_r_frac:.6f} ({spectrum['top_r_energy_pct']:.2f}%)**",
        f"- Gate pass: **{spectrum['gate_pass']}** (threshold: {100*GATE_THRESHOLD:.0f}%)",
        f"- Rank thresholds (cumulative energy): {spectrum['rank_thresholds']}",
        f"- S[:10]: {[round(s, 4) for s in spectrum['singular_values_top20'][:10]]}",
        "",
    ]

    if not spectrum["gate_pass"]:
        lines += [
            "## Verdict",
            "",
            f"**{verdict}**: top-7 energy = {spectrum['top_r_energy_pct']:.2f}% < 10% gate threshold.",
            "Same flat-spectrum failure mode as b3 (cycle 327). Retrain skipped.",
            "",
            "## Comparison to cycle 327 (b3)",
            "",
            "- b3: rank-5 energy = 2.6%, retrain delta = +0.038 (REFUTED-flat-spectrum)",
            f"- b12: rank-7 energy = {spectrum['top_r_energy_pct']:.2f}%, retrain SKIPPED",
            "",
        ]
    else:
        lines += [
            "## Retrain Results",
            "",
            f"**Baseline BPB**: {retrain_results[0]['baseline_bpb']:.6f} (cycle 327 baseline: 1.362078)",
            f"**Retrain**: {N_TRAIN_STEPS} Adam steps, lr={LR}, seq_len={SEQ_LEN}",
            "",
            "| rank | baseline_bpb | b_init_bpb | retrained_bpb | Δ_init | Δ_retrained | n_params_B | param_reduction% |",
            "|------|-------------|------------|---------------|--------|-------------|------------|-----------------|",
        ]
        for r in retrain_results:
            if "error" not in r:
                lines.append(
                    f"| {r['rank']} "
                    f"| {r['baseline_bpb']:.6f} "
                    f"| {r['b_init_bpb']:.6f} "
                    f"| {r['retrained_bpb']:.6f} "
                    f"| {r['delta_b_init']:+.6f} "
                    f"| {r['delta_retrained']:+.6f} "
                    f"| {r['n_params_B']:,} "
                    f"| {r['param_reduction_pct']:.1f}% |"
                )
            else:
                lines.append(f"| {r['rank']} | — | — | — | — | ERROR: {r.get('error', '?')} | — | — |")
        lines += [
            "",
            "## Verdict",
            "",
            f"**{verdict}**",
            "",
            "## Comparison to QAT baselines",
            "",
            "- Cycle 280 (GPT-2 LASER): rank-5 retrain recovered float BPB within 0.001",
            "- Cycle 327 (pgolf b3 LASER): REFUTED-flat-spectrum (rank-5 energy=2.6%, delta=+0.038)",
            "",
        ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Summary written to {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step A: Spectrum preflight
    spectrum = run_spectrum_check()
    spectrum_path = OUT_DIR / f"cycle_{CYCLE}_spectrum.json"
    spectrum_path.write_text(json.dumps(spectrum, indent=2))
    print(f"\nSpectrum JSON written to {spectrum_path}")
    print(f"top-{SPECTRUM_RANK} energy fraction: {spectrum['top_r_energy_pct']:.2f}%  "
          f"gate_pass={spectrum['gate_pass']}")

    retrain_results = []
    verdict = determine_verdict(spectrum, retrain_results)

    if not spectrum["gate_pass"]:
        print(f"\nGate FAILED: top-{SPECTRUM_RANK} energy = {spectrum['top_r_energy_pct']:.2f}% < "
              f"{100*GATE_THRESHOLD:.0f}%. Skipping retrain.")
        print(f"VERDICT: {verdict}")
    else:
        # Step B: Retrain at multiple ranks
        print(f"\nGate PASSED: top-{SPECTRUM_RANK} energy = {spectrum['top_r_energy_pct']:.2f}% >= "
              f"{100*GATE_THRESHOLD:.0f}%. Proceeding with retrain.")

        for rank in RETRAIN_RANKS:
            try:
                result = run_one_rank_retrain(rank)
                retrain_results.append(result)
                print(f"rank={rank}: baseline={result['baseline_bpb']:.6f}  "
                      f"b_init={result['b_init_bpb']:.6f}  "
                      f"retrained={result['retrained_bpb']:.6f}  "
                      f"delta={result['delta_retrained']:+.6f}  "
                      f"({result['wall_seconds']:.0f}s)")
            except Exception as e:
                print(f"ERROR for rank={rank}: {e}", file=sys.stderr)
                retrain_results.append({"rank": rank, "error": str(e)})

        json_path = OUT_DIR / f"cycle_{CYCLE}_pgolf_laser_retrain.json"
        json_path.write_text(json.dumps(retrain_results, indent=2))
        print(f"\nRetrain JSON written to {json_path}")

        verdict = determine_verdict(spectrum, retrain_results)
        print(f"\nVERDICT: {verdict}")

    # Write summary
    summary_path = OUT_DIR / f"cycle_{CYCLE}_summary.md"
    write_summary(spectrum, retrain_results, verdict, summary_path)

    # Print final verdict for manifest
    print(f"\n=== FINAL VERDICT: {verdict} ===")

    # Write manifest
    manifest_lines = [
        f"cycle: {CYCLE}",
        f"item: 341",
        f"verdict: {verdict}",
        f"target_block: {TARGET_BLOCK}",
        f"weight_shape: {spectrum['weight_shape']}",
        f"top_{SPECTRUM_RANK}_energy_pct: {spectrum['top_r_energy_pct']:.4f}",
        f"gate_pass: {spectrum['gate_pass']}",
    ]
    if retrain_results:
        good = [r for r in retrain_results if "error" not in r]
        for r in good:
            manifest_lines.append(
                f"rank_{r['rank']}_delta_retrained: {r['delta_retrained']:+.6f}"
            )
    manifest_path = OUT_DIR / f"cycle_{CYCLE}_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
