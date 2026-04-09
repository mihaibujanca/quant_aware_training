#!/usr/bin/env python3
"""QAT autoresearch cycle 327 / item 340 — LASER retrain on pgolf b3.

Invokes parameter-golf/local_scripts/pgolf_laser_retrain.py via subprocess
for ranks {2, 5, 10}.  Writes:
  autoresearch/results/cycle_327_pgolf_laser_retrain.json
  autoresearch/results/cycle_327_summary.md
"""
import json, subprocess, sys, time
from pathlib import Path

PGOLF_PYTHON = "/Users/mihai/Projects/parameter-golf/.venv/bin/python"
PGOLF_SCRIPT = "/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_laser_retrain.py"
CHECKPOINT = "/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz"
PGOLF_ROOT = "/Users/mihai/Projects/parameter-golf"

RANKS = [2, 5, 10]
TARGET_BLOCK = 3
N_TRAIN_STEPS = 500
LR = 1e-4
SEQ_LEN = 1024
N_VAL_TOKENS = 65536

OUT_DIR = Path("autoresearch/results")
CYCLE = 327


def run_one_rank(rank: int) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"Running LASER retrain: rank={rank}, block={TARGET_BLOCK}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [
            PGOLF_PYTHON, PGOLF_SCRIPT,
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

    # Always show stderr (diagnostics)
    if proc.stderr.strip():
        print(proc.stderr, file=sys.stderr, flush=True)

    if proc.returncode != 0:
        msg = f"pgolf_laser_retrain failed for rank={rank} (rc={proc.returncode})"
        print(msg, file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        raise RuntimeError(msg)

    # Parse JSON from stdout
    result = json.loads(proc.stdout.strip())
    result["wall_seconds"] = round(elapsed, 1)
    return result


def write_summary(all_results: list[dict], out_path: Path) -> None:
    if not all_results:
        out_path.write_text("No results.\n")
        return

    baseline = all_results[0]["baseline_bpb"]  # same for all ranks
    lines = [
        "# Cycle 327 — LASER Rank-SVD + B-Retrain on pgolf b3",
        "",
        f"**Checkpoint**: `13L_640d_322_7k_wd70_batch20k_best.npz`",
        f"**Target block**: b{TARGET_BLOCK} (`mlp.proj.weight` shape ~640×1920)",
        f"**Baseline BPB** (from training summary): 1.3474",
        f"**Measured baseline BPB**: {baseline:.6f}",
        f"**Retrain**: {N_TRAIN_STEPS} Adam steps, lr={LR}, seq_len={SEQ_LEN}",
        "",
        "## Results",
        "",
        f"| rank | baseline_bpb | b_init_bpb | retrained_bpb | Δ_init | Δ_retrained | n_params_B | param_reduction% |",
        f"|------|-------------|------------|---------------|--------|-------------|------------|-----------------|",
    ]
    for r in all_results:
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

    lines += [
        "",
        "## Verdict",
        "",
    ]

    # Determine verdict from rank-5 result
    rank5 = next((r for r in all_results if r["rank"] == 5), None)
    if rank5 is None:
        lines.append("No rank-5 result available.")
    else:
        delta5 = rank5["delta_retrained"]
        if delta5 <= 0.001:
            verdict = "CONFIRMED-LASER-transfers"
            lines.append(f"**{verdict}**: rank-5 retrained BPB delta = {delta5:+.6f} ≤ 0.001. "
                         f"LASER recipe transfers to parameter-golf b3. Schedule items 341/342.")
        elif delta5 <= 0.005:
            verdict = "CONFIRMED-partial"
            lines.append(f"**{verdict}**: rank-5 delta = {delta5:+.6f} in (0.001, 0.005). "
                         f"Partial recovery — rank-10 might do better.")
        elif delta5 >= 0.01:
            verdict = "REFUTED-LASER-fails"
            lines.append(f"**{verdict}**: rank-5 delta = {delta5:+.6f} ≥ 0.01. "
                         f"b3's rank-5 capacity insufficient; LASER doesn't transfer.")
        else:
            verdict = "INCONCLUSIVE"
            lines.append(f"**{verdict}**: rank-5 delta = {delta5:+.6f} (0.005–0.01 range).")

    lines += [
        "",
        "## Comparison to QAT baselines",
        "",
        "- Cycle 280 (GPT-2 LASER): rank-5 retrain recovered float BPB within 0.001",
        "- Cycle 318 (pgolf eff_rank census): b3 output eff_rank = 4.89 (pre-scale), 4.01 (post-scale) — lowest in early stack",
        "",
    ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Summary written to {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for rank in RANKS:
        try:
            result = run_one_rank(rank)
            all_results.append(result)
            print(f"rank={rank}: baseline={result['baseline_bpb']:.6f}  "
                  f"b_init={result['b_init_bpb']:.6f}  "
                  f"retrained={result['retrained_bpb']:.6f}  "
                  f"delta={result['delta_retrained']:+.6f}  "
                  f"({result['wall_seconds']:.0f}s)")
        except Exception as e:
            print(f"ERROR for rank={rank}: {e}", file=sys.stderr)
            all_results.append({"rank": rank, "error": str(e)})

    # Write JSON
    json_path = OUT_DIR / f"cycle_{CYCLE}_pgolf_laser_retrain.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nJSON written to {json_path}")

    # Write summary
    good_results = [r for r in all_results if "error" not in r]
    summary_path = OUT_DIR / f"cycle_{CYCLE}_summary.md"
    write_summary(good_results, summary_path)

    # Final status
    print("\n=== FINAL RESULTS ===")
    for r in all_results:
        if "error" in r:
            print(f"  rank={r['rank']}: ERROR — {r['error']}")
        else:
            print(f"  rank={r['rank']}: delta_retrained={r['delta_retrained']:+.6f}  "
                  f"(b_init={r['b_init_bpb']:.4f} → retrained={r['retrained_bpb']:.4f})")


if __name__ == "__main__":
    main()
