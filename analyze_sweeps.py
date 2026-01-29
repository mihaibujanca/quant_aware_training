#!/usr/bin/env python3
"""
Analyze hydra sweep + lambda sweep results from runs/sweep_20260120_222535/.

Aggregates individual run logs into dataframes and computes recovery statistics.
"""

import ast
import json
import re
from pathlib import Path

import pandas as pd
import numpy as np


SWEEP_DIR = Path("runs/sweep_20260120_222535")


# ============================================================
# Part 1: Hydra sweep (run_experiment.py via run_sweeps.sh)
# ============================================================

def parse_hydra_run(run_dir: Path) -> dict | None:
    """Parse a single hydra run directory into a result dict."""
    log = run_dir / "run_experiment.log"
    if not log.exists():
        return None

    text = log.read_text()

    # Extract config from directory name
    # Handle nested values like hidden_sizes=[256,128] by parsing carefully
    config = {}
    name = run_dir.name
    # Split on commas that are NOT inside brackets
    parts = re.split(r",(?![^\[]*\])", name)
    for part in parts:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.replace("model.", "")
        try:
            config[key] = int(val)
        except ValueError:
            try:
                config[key] = float(val)
            except ValueError:
                config[key] = val

    # Extract results dict from last line
    match = re.search(r"Results: (\{.*\})", text)
    if not match:
        return None

    results = ast.literal_eval(match.group(1))
    return {**config, **results}


def load_hydra_sweep(task: str) -> pd.DataFrame:
    """Load all runs for a task (baseline + correction)."""
    rows = []

    # Find all subdirectories for this task
    for subdir in sorted(SWEEP_DIR.iterdir()):
        if not subdir.is_dir() or task not in subdir.name:
            continue

        is_baseline = "baseline" in subdir.name
        for run_dir in sorted(subdir.iterdir()):
            if not run_dir.is_dir():
                continue
            result = parse_hydra_run(run_dir)
            if result:
                result["is_baseline"] = is_baseline
                rows.append(result)

    return pd.DataFrame(rows)


def analyze_hydra_classification():
    """Analyze classification sweep results."""
    df = load_hydra_sweep("classification")
    if df.empty:
        print("No classification results found")
        return

    print("=" * 60)
    print("CLASSIFICATION (spirals, 100D)")
    print("=" * 60)
    print(f"Total runs: {len(df)} ({df['is_baseline'].sum()} baseline, {(~df['is_baseline']).sum()} correction)")

    # Baselines
    bl = df[df["is_baseline"]]
    print(f"\nBaseline float acc: {bl['acc_float'].mean():.3f} (std {bl['acc_float'].std():.3f})")
    print(f"Baseline quant acc: {bl['acc_quant'].mean():.3f} (std {bl['acc_quant'].std():.3f})")

    # Correction runs
    corr = df[~df["is_baseline"]]
    if corr.empty:
        return

    print(f"\nCorrection runs recovery: {corr['recovery_pct'].mean():.1f}% (std {corr['recovery_pct'].std():.1f}%)")

    print("\n--- Recovery by bits ---")
    print(corr.groupby("bits")["recovery_pct"].agg(["mean", "std", "count"]).round(1))

    print("\n--- Recovery by correction_every ---")
    print(corr.groupby("correction_every")["recovery_pct"].agg(["mean", "std"]).round(1))

    print("\n--- Recovery by correction_hidden ---")
    print(corr.groupby("correction_hidden")["recovery_pct"].agg(["mean", "std"]).round(1))

    print("\n--- Recovery by bits × correction_hidden ---")
    print(corr.groupby(["bits", "correction_hidden"])["recovery_pct"].agg(["mean", "std"]).round(1))

    print("\n--- Top 5 runs ---")
    top = corr.nlargest(5, "recovery_pct")
    for _, r in top.iterrows():
        print(f"  bits={int(r['bits'])}, every={int(r['correction_every'])}, "
              f"h={int(r['correction_hidden'])}, depth={int(r['depth'])}, "
              f"width={int(r['hidden_size'])}: {r['recovery_pct']:.1f}%")

    return df


def analyze_hydra_autoencoder():
    """Analyze autoencoder sweep results."""
    df = load_hydra_sweep("autoencoder")
    if df.empty:
        print("No autoencoder results found")
        return

    print("\n" + "=" * 60)
    print("AUTOENCODER (MNIST)")
    print("=" * 60)
    print(f"Total runs: {len(df)} ({df['is_baseline'].sum()} baseline, {(~df['is_baseline']).sum()} correction)")

    corr = df[~df["is_baseline"]]
    if corr.empty:
        return

    print(f"\nCorrection runs recovery: {corr['recovery_pct'].mean():.1f}% (std {corr['recovery_pct'].std():.1f}%)")

    print("\n--- Recovery by bits ---")
    print(corr.groupby("bits")["recovery_pct"].agg(["mean", "std", "count"]).round(1))

    print("\n--- Recovery by correction_every ---")
    print(corr.groupby("correction_every")["recovery_pct"].agg(["mean", "std"]).round(1))

    print("\n--- Recovery by correction_hidden ---")
    print(corr.groupby("correction_hidden")["recovery_pct"].agg(["mean", "std"]).round(1))

    print("\n--- Top 5 runs ---")
    top = corr.nlargest(5, "recovery_pct")
    for _, r in top.iterrows():
        print(f"  bits={int(r['bits'])}, every={int(r['correction_every'])}, "
              f"h={int(r['correction_hidden'])}: {r['recovery_pct']:.1f}%")

    return df


def analyze_hydra_transformer():
    """Analyze transformer sweep results."""
    df = load_hydra_sweep("transformer")
    if df.empty:
        print("No transformer results found")
        return

    print("\n" + "=" * 60)
    print("TRANSFORMER (Shakespeare char-LM)")
    print("=" * 60)
    print(f"Total runs: {len(df)} ({df['is_baseline'].sum()} baseline, {(~df['is_baseline']).sum()} correction)")

    corr = df[~df["is_baseline"]]
    if corr.empty:
        return

    print(f"\nCorrection runs recovery: {corr['recovery_pct'].mean():.1f}% (std {corr['recovery_pct'].std():.1f}%)")

    print("\n--- Recovery by bits ---")
    print(corr.groupby("bits")["recovery_pct"].agg(["mean", "std", "count"]).round(1))

    print("\n--- Recovery by correction_every ---")
    print(corr.groupby("correction_every")["recovery_pct"].agg(["mean", "std"]).round(1))

    print("\n--- Recovery by correction_hidden ---")
    print(corr.groupby("correction_hidden")["recovery_pct"].agg(["mean", "std"]).round(1))

    print("\n--- Recovery by d_model ---")
    print(corr.groupby("d_model")["recovery_pct"].agg(["mean", "std"]).round(1))

    print("\n--- Recovery by bits × correction_hidden ---")
    print(corr.groupby(["bits", "correction_hidden"])["recovery_pct"].agg(["mean", "std"]).round(1))

    print("\n--- Top 5 runs ---")
    top = corr.nlargest(5, "recovery_pct")
    for _, r in top.iterrows():
        print(f"  bits={int(r['bits'])}, every={int(r['correction_every'])}, "
              f"h={int(r['correction_hidden'])}, d={int(r['d_model'])}, "
              f"L={int(r['n_layers'])}: {r['recovery_pct']:.1f}% "
              f"(ppl: {r['ppl_float']:.1f}→{r['ppl_quant']:.1f}→{r['ppl_corrected']:.1f})")

    return df


# ============================================================
# Part 2: Lambda sweep (experiments/lambda_sweep.py)
# ============================================================

def load_lambda_sweep() -> dict:
    """Load lambda sweep results from JSON."""
    json_path = SWEEP_DIR / "lambda_sweep_results.json"
    if not json_path.exists():
        print("No lambda sweep JSON found")
        return {}
    with open(json_path) as f:
        return json.load(f)


def analyze_lambda_sweep():
    """Analyze lambda sweep results."""
    data = load_lambda_sweep()
    if not data:
        return

    print("\n" + "=" * 60)
    print("LAMBDA SWEEP (hybrid distillation)")
    print("=" * 60)

    for task_name, task_data in data.items():
        print(f"\n--- {task_name.upper()} ---")

        rows = []
        for run in task_data:
            rows.append(run)

        df = pd.DataFrame(rows)
        if df.empty:
            continue

        # Find the recovery column name
        rec_col = "recovery" if "recovery" in df.columns else "recovery_pct"
        lam_col = "lambda" if "lambda" in df.columns else "layer_loss_weight"

        print(f"Runs: {len(df)}")

        if lam_col in df.columns:
            print(f"\nRecovery by lambda:")
            print(df.groupby(lam_col)[rec_col].agg(["mean", "std"]).round(1))

        if "bits" in df.columns:
            print(f"\nRecovery by bits:")
            print(df.groupby("bits")[rec_col].agg(["mean", "std"]).round(1))

        if "correction_hidden" in df.columns:
            print(f"\nRecovery by correction_hidden:")
            print(df.groupby("correction_hidden")[rec_col].agg(["mean", "std"]).round(1))

        # Best lambda per bits
        if lam_col in df.columns and "bits" in df.columns:
            print("\nBest lambda per bits:")
            for bits in sorted(df["bits"].unique()):
                sub = df[df["bits"] == bits]
                best = sub.groupby(lam_col)[rec_col].mean()
                print(f"  {int(bits)}-bit: best lambda={best.idxmax()}, recovery={best.max():.1f}%")

        # Top 5
        print("\nTop 5 runs:")
        top = df.nlargest(5, rec_col)
        for _, r in top.iterrows():
            lam = r.get(lam_col, "?")
            bits = int(r.get("bits", 0))
            h = int(r.get("correction_hidden", 0))
            seed = int(r.get("seed", 0))
            print(f"  lambda={lam}, bits={bits}, h={h}, seed={seed}: {r[rec_col]:.1f}%")


# ============================================================
# Main
# ============================================================

def main():
    print("SWEEP ANALYSIS")
    print("=" * 60)
    print(f"Sweep dir: {SWEEP_DIR}\n")

    # Hydra sweeps
    analyze_hydra_classification()
    analyze_hydra_autoencoder()
    analyze_hydra_transformer()

    # Lambda sweep
    analyze_lambda_sweep()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
