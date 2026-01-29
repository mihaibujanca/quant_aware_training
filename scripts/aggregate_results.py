"""
Aggregate results from Hydra sweep runs into a summary CSV.

Usage:
    python aggregate_results.py runs/autoencoder_sweep_*
    python aggregate_results.py runs/*_sweep_*
"""

import json
import sys
from pathlib import Path

import pandas as pd


def load_results(sweep_dir: Path) -> list[dict]:
    """Load all results.json files from a sweep directory."""
    results = []
    for results_file in sweep_dir.rglob("results.json"):
        try:
            with open(results_file) as f:
                data = json.load(f)

            # Flatten config and results into single dict
            row = {}
            if "config" in data:
                for k, v in data["config"].items():
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            row[f"{k}.{k2}"] = v2
                    else:
                        row[k] = v
            if "results" in data:
                row.update(data["results"])

            row["_file"] = str(results_file)
            results.append(row)
        except Exception as e:
            print(f"Warning: Failed to load {results_file}: {e}")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_results.py <sweep_dir> [sweep_dir2 ...]")
        sys.exit(1)

    all_results = []
    for pattern in sys.argv[1:]:
        for sweep_dir in Path(".").glob(pattern):
            if sweep_dir.is_dir():
                print(f"Loading results from {sweep_dir}...")
                results = load_results(sweep_dir)
                print(f"  Found {len(results)} runs")
                all_results.extend(results)

    if not all_results:
        print("No results found!")
        sys.exit(1)

    df = pd.DataFrame(all_results)

    # Print summary stats
    print(f"\nTotal runs: {len(df)}")

    if "recovery_pct" in df.columns:
        print(f"\nRecovery stats:")
        print(f"  Mean: {df['recovery_pct'].mean():.1f}%")
        print(f"  Median: {df['recovery_pct'].median():.1f}%")
        print(f"  Min: {df['recovery_pct'].min():.1f}%")
        print(f"  Max: {df['recovery_pct'].max():.1f}%")

        # Group by key hyperparameters
        for col in ["bits", "correction_every", "correction_hidden", "experiment.type"]:
            if col in df.columns:
                print(f"\nRecovery by {col}:")
                grouped = df.groupby(col)["recovery_pct"].agg(["mean", "std", "count"])
                print(grouped.to_string())

    # Save to CSV
    output_file = "sweep_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    return df


if __name__ == "__main__":
    main()
