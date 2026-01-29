#!/usr/bin/env python3
"""
Rounding Mode Analysis for QAT Experiments

Analyzes experiments comparing nearest/floor/ceil rounding modes.
Data: runs/experiment_20260119_201449/summary.csv (630 runs)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_rounding_data():
    """Load the comprehensive 630-run rounding experiment data."""
    csv_path = Path("runs/experiment_20260119_201449/summary.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} runs from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rounding modes: {df['rounding'].unique()}")
    print(f"Bit widths: {df['bits'].unique()}")
    print(f"Widths: {sorted(df['width'].unique())}")
    print(f"Depths: {sorted(df['depth'].unique())}")
    print(f"Seeds: {df['seed'].unique()}")
    return df


def overall_stats(df):
    """Overall accuracy statistics by rounding mode."""
    print("\n" + "="*60)
    print("OVERALL ACCURACY BY ROUNDING MODE")
    print("="*60)

    stats = df.groupby('rounding')['final_test_acc'].agg(['mean', 'std', 'count'])
    stats = stats.sort_values('mean', ascending=False)
    print(stats.round(4))
    return stats


def stats_by_bits(df):
    """Statistics broken down by bit width."""
    print("\n" + "="*60)
    print("ACCURACY BY ROUNDING MODE AND BIT WIDTH")
    print("="*60)

    for bits in sorted(df['bits'].unique()):
        print(f"\n--- {bits}-bit quantization ---")
        subset = df[df['bits'] == bits]
        stats = subset.groupby('rounding')['final_test_acc'].agg(['mean', 'std', 'count'])
        stats = stats.sort_values('mean', ascending=False)
        print(stats.round(4))

    return df.groupby(['bits', 'rounding'])['final_test_acc'].agg(['mean', 'std'])


def gap_vs_nearest(df):
    """Compare floor/ceil performance gap vs nearest rounding."""
    print("\n" + "="*60)
    print("GAP VS NEAREST ROUNDING")
    print("="*60)

    for bits in sorted(df['bits'].unique()):
        print(f"\n--- {bits}-bit ---")
        subset = df[df['bits'] == bits]

        # Get mean accuracy for each mode
        means = subset.groupby('rounding')['final_test_acc'].mean()

        if 'nearest' in means.index:
            nearest_acc = means['nearest']
            for mode in ['floor', 'ceil']:
                if mode in means.index:
                    gap = means[mode] - nearest_acc
                    print(f"{mode} - nearest: {gap:+.4f}")


def depth_effect(df):
    """Analyze how rounding mode effect varies with depth."""
    print("\n" + "="*60)
    print("DEPTH EFFECT ON ROUNDING MODE")
    print("="*60)

    for bits in sorted(df['bits'].unique()):
        print(f"\n--- {bits}-bit ---")
        subset = df[df['bits'] == bits]

        # Get mean accuracy by depth and rounding
        pivot = subset.groupby(['depth', 'rounding'])['final_test_acc'].mean().unstack()

        if 'nearest' in pivot.columns:
            print("\nGap vs nearest by depth:")
            for mode in ['floor', 'ceil']:
                if mode in pivot.columns:
                    gaps = pivot[mode] - pivot['nearest']
                    print(f"\n{mode}:")
                    for depth, gap in gaps.items():
                        print(f"  depth={depth}: {gap:+.4f}")


def width_effect(df):
    """Analyze how rounding mode effect varies with width."""
    print("\n" + "="*60)
    print("WIDTH EFFECT ON ROUNDING MODE")
    print("="*60)

    for bits in sorted(df['bits'].unique()):
        print(f"\n--- {bits}-bit ---")
        subset = df[df['bits'] == bits]

        # Get mean accuracy by width and rounding
        pivot = subset.groupby(['width', 'rounding'])['final_test_acc'].mean().unstack()

        if 'nearest' in pivot.columns:
            print("\nGap vs nearest by width:")
            for mode in ['floor', 'ceil']:
                if mode in pivot.columns:
                    gaps = pivot[mode] - pivot['nearest']
                    print(f"\n{mode}:")
                    for width, gap in gaps.items():
                        print(f"  width={width}: {gap:+.4f}")


def find_floor_ceil_wins(df):
    """Find configurations where floor/ceil beats nearest."""
    print("\n" + "="*60)
    print("CONFIGURATIONS WHERE FLOOR/CEIL BEATS NEAREST")
    print("="*60)

    # Group by configuration (seed, width, depth, bits) and compare modes
    configs = df.groupby(['seed', 'width', 'depth', 'bits', 'rounding'])['final_test_acc'].mean()
    configs = configs.unstack('rounding')

    if 'nearest' not in configs.columns:
        print("No 'nearest' mode found in data")
        return

    wins = {'floor': [], 'ceil': []}

    for mode in ['floor', 'ceil']:
        if mode not in configs.columns:
            continue

        for idx in configs.index:
            nearest_acc = configs.loc[idx, 'nearest']
            mode_acc = configs.loc[idx, mode]

            if pd.notna(mode_acc) and pd.notna(nearest_acc) and mode_acc > nearest_acc:
                gap = mode_acc - nearest_acc
                wins[mode].append({
                    'config': idx,
                    'nearest_acc': nearest_acc,
                    f'{mode}_acc': mode_acc,
                    'gap': gap
                })

    total_configs = len(configs)

    for mode in ['floor', 'ceil']:
        win_list = wins[mode]
        if win_list:
            win_list.sort(key=lambda x: x['gap'], reverse=True)
            print(f"\n{mode.upper()} wins: {len(win_list)}/{total_configs} ({100*len(win_list)/total_configs:.1f}%)")
            print("\nTop 5 wins:")
            for w in win_list[:5]:
                seed, width, depth, bits = w['config']
                print(f"  seed={seed}, w={width}, d={depth}, bits={bits}: "
                      f"{mode}={w[f'{mode}_acc']:.4f} vs nearest={w['nearest_acc']:.4f} "
                      f"(+{w['gap']:.4f})")

    return wins


def seed_stability(df):
    """Analyze variance across seeds for each rounding mode."""
    print("\n" + "="*60)
    print("SEED STABILITY (VARIANCE ACROSS SEEDS)")
    print("="*60)

    for bits in sorted(df['bits'].unique()):
        print(f"\n--- {bits}-bit ---")
        subset = df[df['bits'] == bits]

        # Get variance across seeds for each (width, depth, rounding) config
        variance = subset.groupby(['width', 'depth', 'rounding'])['final_test_acc'].std()

        # Average variance by rounding mode
        avg_var = variance.groupby('rounding').mean()
        print("Mean std across seeds by rounding mode:")
        for mode, std in avg_var.sort_values().items():
            print(f"  {mode}: {std:.4f}")


def summary_table(df):
    """Generate summary table for documentation."""
    print("\n" + "="*60)
    print("SUMMARY TABLE FOR DOCUMENTATION")
    print("="*60)

    print("\n### Overall Accuracy\n")
    print("| Mode     | Mean Acc | Std   | Count |")
    print("|----------|----------|-------|-------|")

    stats = df.groupby('rounding')['final_test_acc'].agg(['mean', 'std', 'count'])
    stats = stats.sort_values('mean', ascending=False)
    for mode, row in stats.iterrows():
        print(f"| {mode:<8} | {row['mean']:.4f}   | {row['std']:.3f} | {int(row['count']):<5} |")

    for bits in sorted(df['bits'].unique()):
        print(f"\n### {bits}-bit Quantization\n")
        print("| Mode    | Mean Acc | Std   |")
        print("|---------|----------|-------|")

        subset = df[df['bits'] == bits]
        stats = subset.groupby('rounding')['final_test_acc'].agg(['mean', 'std'])
        stats = stats.sort_values('mean', ascending=False)
        for mode, row in stats.iterrows():
            print(f"| {mode:<7} | {row['mean']:.4f}   | {row['std']:.3f} |")


def main():
    print("="*60)
    print("QAT ROUNDING MODE ANALYSIS")
    print("="*60)

    df = load_rounding_data()

    # Run all analyses
    overall_stats(df)
    stats_by_bits(df)
    gap_vs_nearest(df)
    depth_effect(df)
    width_effect(df)
    find_floor_ceil_wins(df)
    seed_stability(df)
    summary_table(df)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
