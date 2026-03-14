"""Run continuous single-block FFN swap sweeps for transformer compression."""

from __future__ import annotations

import argparse
import json
import time
import traceback

from datetime import datetime, timezone
from pathlib import Path

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    DistillationSpec,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    build_probe_corpus,
    load_nanochat_teacher,
    run_ffn_distillation_baseline,
    run_ffn_geometric_compression,
)


def _parse_int_list(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or raw.strip() == "":
        return None
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("hidden fractions must not be empty")
    return values


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous single-block FFN localization sweep")
    parser.add_argument("--teacher-source", choices=["hf_pretrained", "hf_random", "nanochat_local"], default="hf_pretrained")
    parser.add_argument("--hf-model", type=str, default="gpt2")
    parser.add_argument("--nanochat-repo", type=str, default=None)
    parser.add_argument("--nanochat-source", type=str, default="base")
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dataset-name", type=str, default="proxy_shakespeare")

    parser.add_argument("--text-path", type=str, default="data/shakespeare.txt")
    parser.add_argument("--parquet-dir", type=str, default=None)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--probe-train-seqs", type=int, default=64)
    parser.add_argument("--probe-holdout-seqs", type=int, default=32)
    parser.add_argument("--eval-val-seqs", type=int, default=32)

    parser.add_argument("--layer-ids", type=str, default=None, help="Comma-separated layer ids; default = all layers")
    parser.add_argument("--hidden-fractions", type=str, default="0.25,0.125")
    parser.add_argument("--min-hidden", type=int, default=64)
    parser.add_argument("--max-hidden", type=int, default=256)
    parser.add_argument("--selection-mode", choices=["topk", "kmeans"], default="topk")
    parser.add_argument("--jacobian-samples", type=int, default=8)
    parser.add_argument("--max-tokens-per-layer", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)

    parser.add_argument("--distill-steps", type=int, default=60)
    parser.add_argument("--distill-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-repeats", type=int, default=0, help="0 means loop forever")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="runs/nanochat_single_block_localization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    heartbeat_path = output_dir / "heartbeat.json"
    meta_path = output_dir / "meta.json"

    teacher = load_nanochat_teacher(
        TeacherLoadSpec(
            source=args.teacher_source,
            hf_model_name=args.hf_model,
            nanochat_repo=args.nanochat_repo,
            nanochat_source=args.nanochat_source,
            nanochat_model_tag=args.model_tag,
            nanochat_step=args.step,
            device=args.device,
            dataset_name=args.dataset_name,
        )
    )
    corpus = build_probe_corpus(
        ProbeCorpusSpec(
            teacher=teacher,
            text_path=args.text_path,
            parquet_dir=args.parquet_dir,
            sequence_length=args.sequence_length,
            probe_train_sequences=args.probe_train_seqs,
            probe_holdout_sequences=args.probe_holdout_seqs,
            eval_val_sequences=args.eval_val_seqs,
            dataset_name=teacher.dataset_name,
        )
    )
    layer_ids = _parse_int_list(args.layer_ids) or teacher.layer_ids
    hidden_fractions = _parse_float_list(args.hidden_fractions)
    teacher_metrics = _evaluate_model(teacher, teacher.model, corpus, batch_size=args.batch_size)

    meta = {
        "created_at": _utc_now(),
        "teacher": {
            "source": teacher.source,
            "model_family": teacher.model_family,
            "model_name": teacher.model_name,
            "dataset_name": teacher.dataset_name,
            "layer_ids": list(teacher.layer_ids),
            "selected_layer_ids": list(layer_ids),
        },
        "corpus": {
            "sequence_length": corpus.sequence_length,
            "probe_train_sequences": int(corpus.probe_train.shape[0]),
            "probe_holdout_sequences": int(corpus.probe_holdout.shape[0]),
            "eval_val_sequences": int(corpus.eval_val.shape[0]),
        },
        "sweep": {
            "hidden_fractions": list(hidden_fractions),
            "min_hidden": args.min_hidden,
            "max_hidden": args.max_hidden,
            "selection_mode": args.selection_mode,
            "jacobian_samples": args.jacobian_samples,
            "max_tokens_per_layer": args.max_tokens_per_layer,
            "distill_steps": args.distill_steps,
            "distill_lr": args.distill_lr,
            "seed": args.seed,
            "max_repeats": args.max_repeats,
        },
        "teacher_metrics": teacher_metrics,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    repeat_idx = 0
    while args.max_repeats == 0 or repeat_idx < args.max_repeats:
        sweep_seed = args.seed + repeat_idx
        for hidden_fraction in hidden_fractions:
            budget = CompressionBudgetSpec(
                hidden_fraction=hidden_fraction,
                min_hidden=args.min_hidden,
                max_hidden=args.max_hidden,
                selection_mode=args.selection_mode,
                jacobian_samples=args.jacobian_samples,
            )
            for layer_id in layer_ids:
                for method in ("geometric", "distilled"):
                    heartbeat = {
                        "updated_at": _utc_now(),
                        "repeat_idx": repeat_idx,
                        "seed": sweep_seed,
                        "layer_id": layer_id,
                        "hidden_fraction": hidden_fraction,
                        "method": method,
                        "status": "running",
                    }
                    heartbeat_path.write_text(json.dumps(heartbeat, indent=2), encoding="utf-8")

                    started = time.time()
                    try:
                        if method == "geometric":
                            result = run_ffn_geometric_compression(
                                teacher,
                                corpus,
                                budget,
                                layer_ids=(layer_id,),
                                batch_size=args.batch_size,
                                max_tokens_per_layer=args.max_tokens_per_layer,
                            )
                        else:
                            result = run_ffn_distillation_baseline(
                                DistillationSpec(
                                    teacher=teacher,
                                    corpus=corpus,
                                    budget=budget,
                                    layer_ids=(layer_id,),
                                    batch_size=args.batch_size,
                                    steps=args.distill_steps,
                                    lr=args.distill_lr,
                                    seed=sweep_seed,
                                )
                            )

                        metrics = dict(result.metrics)
                        layer_metrics = dict(result.layer_metrics[0]) if result.layer_metrics else {}
                        record = {
                            "timestamp": _utc_now(),
                            "status": "ok",
                            "repeat_idx": repeat_idx,
                            "seed": sweep_seed,
                            "method": method,
                            "layer_id": layer_id,
                            "hidden_fraction": hidden_fraction,
                            "runtime_seconds": round(time.time() - started, 3),
                            "teacher_eval_loss": teacher_metrics["eval_loss"],
                            "teacher_perplexity": teacher_metrics["perplexity"],
                            "relative_perplexity_increase": (
                                metrics["perplexity"] / max(teacher_metrics["perplexity"], 1e-12)
                            ) - 1.0,
                            "metrics": metrics,
                            "layer_metrics": layer_metrics,
                        }
                    except Exception as exc:  # pragma: no cover - long-running experiment guardrail
                        record = {
                            "timestamp": _utc_now(),
                            "status": "error",
                            "repeat_idx": repeat_idx,
                            "seed": sweep_seed,
                            "method": method,
                            "layer_id": layer_id,
                            "hidden_fraction": hidden_fraction,
                            "runtime_seconds": round(time.time() - started, 3),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    _append_jsonl(results_path, record)
                    heartbeat_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                    print(
                        f"[{record['timestamp']}] repeat={repeat_idx} seed={sweep_seed} "
                        f"layer={layer_id} frac={hidden_fraction} method={method} status={record['status']}",
                        flush=True,
                    )
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
        repeat_idx += 1


if __name__ == "__main__":
    main()
