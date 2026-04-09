#!/usr/bin/env python3
"""QAT autoresearch cycle 342 / item 351 — LASER retrain on pgolf b3 with extended budget.

Re-evaluates cycle 327's LASER results with 4x more retrain steps and lower lr.
Hypothesis: 500 steps at lr=1e-4 was insufficient for rank-5 factorization.

Configs:
  A: rank=5,  2000 steps, lr=3e-5   (lower lr, more steps)
  B: rank=10, 2000 steps, lr=3e-5   (higher rank, same budget)
  C: rank=5,  2000 steps, lr=1e-4   (same lr as cycle 327, more steps — control)

Invokes parameter-golf/local_scripts/pgolf_laser_retrain.py via subprocess.
"""
import json, subprocess, sys, time
from pathlib import Path

PGOLF_PYTHON = "/Users/mihai/Projects/parameter-golf/.venv/bin/python"
PGOLF_SCRIPT = "/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_laser_retrain.py"
CHECKPOINT = "/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz"
PGOLF_ROOT = "/Users/mihai/Projects/parameter-golf"

TARGET_BLOCK = 3
SEQ_LEN = 1024
N_VAL_TOKENS = 65536

CONFIGS = [
    {"rank": 5,  "n_train_steps": 2000, "lr": 3e-5,  "label": "r5_2000step_lr3e-5"},
    {"rank": 10, "n_train_steps": 2000, "lr": 3e-5,  "label": "r10_2000step_lr3e-5"},
    {"rank": 5,  "n_train_steps": 2000, "lr": 1e-4,  "label": "r5_2000step_lr1e-4"},
]

OUT_DIR = Path("autoresearch/results")
CYCLE = 342


def run_one(cfg: dict) -> dict:
    label = cfg["label"]
    print(f"\n{'='*60}", flush=True)
    print(f"Config {label}: rank={cfg['rank']}, steps={cfg['n_train_steps']}, lr={cfg['lr']}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [
            PGOLF_PYTHON, PGOLF_SCRIPT,
            "--checkpoint", CHECKPOINT,
            "--target-block", str(TARGET_BLOCK),
            "--rank", str(cfg["rank"]),
            "--n-train-steps", str(cfg["n_train_steps"]),
            "--lr", str(cfg["lr"]),
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
        msg = f"pgolf_laser_retrain failed for {label} (rc={proc.returncode})"
        print(msg, file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        raise RuntimeError(msg)

    result = json.loads(proc.stdout.strip())
    result["wall_seconds"] = round(elapsed, 1)
    result["label"] = label
    return result


def write_summary(results: list[dict], out_path: Path) -> None:
    if not results:
        out_path.write_text("No results.\n")
        return

    baseline = results[0]["baseline_bpb"]

    # Cycle 327 comparison data
    c327 = [
        {"rank": 2,  "steps": 500, "lr": 1e-4, "delta": +0.040481},
        {"rank": 5,  "steps": 500, "lr": 1e-4, "delta": +0.037780},
        {"rank": 10, "steps": 500, "lr": 1e-4, "delta": +0.034984},
    ]

    lines = [
        "# Cycle 342 — LASER Retrain on pgolf b3 (Extended Budget)",
        "",
        f"**Checkpoint**: `13L_640d_322_7k_wd70_batch20k_best.npz`",
        f"**Target block**: b{TARGET_BLOCK} (output eff_rank=4.89 from cycle 318)",
        f"**Measured baseline BPB**: {baseline:.6f}",
        "",
        "## Results (this cycle)",
        "",
        "| config | rank | steps | lr | baseline | b_init | retrained | delta | params_B |",
        "|--------|------|-------|----|----------|--------|-----------|-------|----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['label']} | {r['rank']} | {r['n_steps']} | {r['lr']} "
            f"| {r['baseline_bpb']:.6f} | {r['b_init_bpb']:.6f} "
            f"| {r['retrained_bpb']:.6f} | {r['delta_retrained']:+.6f} "
            f"| {r['n_params_B']:,} |"
        )

    lines += [
        "",
        "## Comparison to cycle 327 (500 steps, lr=1e-4)",
        "",
        "| rank | steps | lr | delta_327 | delta_342 | improvement |",
        "|------|-------|----|-----------|-----------|-------------|",
    ]
    for r in results:
        match = next((c for c in c327 if c["rank"] == r["rank"] and c["lr"] == r["lr"]), None)
        if match:
            improvement = match["delta"] - r["delta_retrained"]
            lines.append(
                f"| {r['rank']} | {match['steps']}→{r['n_steps']} | {r['lr']} "
                f"| {match['delta']:+.6f} | {r['delta_retrained']:+.6f} "
                f"| {improvement:+.6f} |"
            )

    lines += ["", "## Verdict", ""]

    # Best delta across all configs
    best = min(results, key=lambda r: r["delta_retrained"])
    best_delta = best["delta_retrained"]

    if best_delta <= 0.005:
        verdict = "CONFIRMED"
        lines.append(
            f"**{verdict}**: best config ({best['label']}) achieves BPB delta "
            f"= {best_delta:+.6f} ≤ 0.005. Extended retrain rescues LASER on pgolf b3."
        )
    elif best_delta > 0.01:
        verdict = "REFUTED"
        lines.append(
            f"**{verdict}**: best config ({best['label']}) achieves BPB delta "
            f"= {best_delta:+.6f} > 0.01. Even 4x budget can't fix LASER on pgolf b3. "
            f"eff_rank 4.89 is not concentrated enough for LASER."
        )
    else:
        verdict = "INCONCLUSIVE"
        lines.append(
            f"**{verdict}**: best config ({best['label']}) achieves BPB delta "
            f"= {best_delta:+.6f} (between 0.005 and 0.01). "
            f"Retrain budget helps but doesn't fully close the gap."
        )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Summary written to {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for cfg in CONFIGS:
        try:
            result = run_one(cfg)
            all_results.append(result)
            print(f"{cfg['label']}: baseline={result['baseline_bpb']:.6f}  "
                  f"b_init={result['b_init_bpb']:.6f}  "
                  f"retrained={result['retrained_bpb']:.6f}  "
                  f"delta={result['delta_retrained']:+.6f}  "
                  f"({result['wall_seconds']:.0f}s)")
        except Exception as e:
            print(f"ERROR for {cfg['label']}: {e}", file=sys.stderr)
            all_results.append({"label": cfg["label"], "error": str(e)})

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
            print(f"  {r['label']}: ERROR — {r['error']}")
        else:
            print(f"  {r['label']}: delta={r['delta_retrained']:+.6f}  "
                  f"(b_init={r['b_init_bpb']:.4f} → retrained={r['retrained_bpb']:.4f})")


if __name__ == "__main__":
    main()
