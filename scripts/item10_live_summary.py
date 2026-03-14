#!/usr/bin/env python3
"""Write a live markdown summary for the overnight item 10 sweep."""

from __future__ import annotations

import argparse
import json

from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize live item 10 results")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="runs/nanochat_single_block_localization_overnight_main",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="autoresearch/results/cycle_005_live_summary.md",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    results_path = input_dir / "results.jsonl"
    meta_path = input_dir / "meta.json"
    heartbeat_path = input_dir / "heartbeat.json"
    output_path = Path(args.output)

    meta = _load_json(meta_path)
    heartbeat = _load_json(heartbeat_path)
    records = _load_records(results_path)

    ok_records = [record for record in records if record.get("status") == "ok"]
    latest_by_key: dict[tuple[float, int, str], dict] = {}
    for record in ok_records:
        key = (
            float(record["hidden_fraction"]),
            int(record["layer_id"]),
            str(record["method"]),
        )
        latest_by_key[key] = record

    by_budget_method: dict[tuple[float, str], list[dict]] = defaultdict(list)
    for key, record in latest_by_key.items():
        hidden_fraction, _layer_id, method = key
        by_budget_method[(hidden_fraction, method)].append(record)

    geometric_rows = sorted(
        by_budget_method.get((0.125, "geometric"), []) + by_budget_method.get((0.0625, "geometric"), []),
        key=lambda record: (record["hidden_fraction"], record["relative_perplexity_increase"]),
    )

    paired = []
    for (hidden_fraction, layer_id, method), record in latest_by_key.items():
        if method != "geometric":
            continue
        distilled = latest_by_key.get((hidden_fraction, layer_id, "distilled"))
        paired.append((record, distilled))

    confirmed_count = 0
    total_geometric_ge4x = 0
    for geo, distilled in paired:
        compression_ratio = float(geo["metrics"]["ffn_compression_ratio"])
        if compression_ratio < 4.0:
            continue
        total_geometric_ge4x += 1
        geo_rel = float(geo["relative_perplexity_increase"])
        distilled_ppl = float(distilled["metrics"]["perplexity"]) if distilled is not None else None
        within_distilled = distilled_ppl is not None and float(geo["metrics"]["perplexity"]) <= 1.25 * distilled_ppl
        if geo_rel <= 0.10 and within_distilled:
            confirmed_count += 1

    lines = [
        "# Cycle 005 Live Summary",
        "",
        f"- Input dir: `{input_dir}`",
        f"- Records: `{len(records)}` total, `{len(ok_records)}` successful",
    ]

    if meta is not None:
        lines.extend(
            [
                f"- Teacher: `{meta['teacher']['model_name']}` ({meta['teacher']['model_family']})",
                f"- Proxy dataset label: `{meta['teacher']['dataset_name']}`",
                f"- Teacher perplexity: `{meta['teacher_metrics']['perplexity']:.3f}`",
                f"- Selected layers: `{meta['teacher']['selected_layer_ids']}`",
                f"- Budgets: `{meta['sweep']['hidden_fractions']}` with `max_hidden={meta['sweep']['max_hidden']}`",
            ]
        )

    if heartbeat is not None:
        lines.extend(
            [
                "",
                "## Heartbeat",
                f"- Updated: `{heartbeat.get('updated_at')}`",
                f"- Repeat: `{heartbeat.get('repeat_idx')}`",
                f"- Running: layer `{heartbeat.get('layer_id')}`, fraction `{heartbeat.get('hidden_fraction')}`, method `{heartbeat.get('method')}`",
                f"- Status: `{heartbeat.get('status')}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Item 10 Progress",
            f"- `>=4x` geometric swaps observed: `{total_geometric_ge4x}`",
            f"- Current count meeting the agenda's success condition: `{confirmed_count}`",
        ]
    )

    lines.extend(["", "## Best Geometric Swaps So Far"])
    for record in geometric_rows[:8]:
        lines.append(
            "- layer `{layer}` frac `{frac}` ppl `{ppl:.3f}` rel_ppl `{rel:+.3%}` top1 `{top1:.3f}`".format(
                layer=int(record["layer_id"]),
                frac=float(record["hidden_fraction"]),
                ppl=float(record["metrics"]["perplexity"]),
                rel=float(record["relative_perplexity_increase"]),
                top1=float(record["metrics"]["top1_agreement"]),
            )
        )
    if not geometric_rows:
        lines.append("- No geometric records yet.")

    lines.extend(["", "## Geometric vs Distilled Pairs"])
    pair_rows = []
    for geo, distilled in paired:
        if distilled is None:
            continue
        pair_rows.append(
            (
                float(geo["hidden_fraction"]),
                float(geo["relative_perplexity_increase"]),
                int(geo["layer_id"]),
                geo,
                distilled,
            )
        )
    pair_rows.sort()
    for _frac, _rel, _layer, geo, distilled in pair_rows[:10]:
        lines.append(
            "- layer `{layer}` frac `{frac}` geo_rel `{geo_rel:+.3%}` dist_rel `{dist_rel:+.3%}` geo_ppl `{geo_ppl:.3f}` dist_ppl `{dist_ppl:.3f}`".format(
                layer=int(geo["layer_id"]),
                frac=float(geo["hidden_fraction"]),
                geo_rel=float(geo["relative_perplexity_increase"]),
                dist_rel=float(distilled["relative_perplexity_increase"]),
                geo_ppl=float(geo["metrics"]["perplexity"]),
                dist_ppl=float(distilled["metrics"]["perplexity"]),
            )
        )
    if not pair_rows:
        lines.append("- No geometric/distilled pairs yet.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
