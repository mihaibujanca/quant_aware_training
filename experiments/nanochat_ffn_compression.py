"""Run FFN-only geometric compression on GPT-style language models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    DistillationSpec,
    LearnedCompressorSpec,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    build_probe_corpus,
    load_nanochat_teacher,
    result_to_dict,
    run_ffn_distillation_baseline,
    run_ffn_geometric_compression,
    run_ffn_learned_compressor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FFN-only compression for GPT-style models")
    parser.add_argument("--teacher-source", choices=["hf_pretrained", "hf_random", "nanochat_local"], default="hf_pretrained")
    parser.add_argument("--hf-model", type=str, default="gpt2")
    parser.add_argument("--nanochat-repo", type=str, default=None)
    parser.add_argument("--nanochat-source", type=str, default="base")
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--text-path", type=str, default=None)
    parser.add_argument("--parquet-dir", type=str, default=None)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--probe-train-seqs", type=int, default=64)
    parser.add_argument("--probe-holdout-seqs", type=int, default=32)
    parser.add_argument("--eval-val-seqs", type=int, default=32)

    parser.add_argument("--hidden-fraction", type=float, default=0.25)
    parser.add_argument("--min-hidden", type=int, default=32)
    parser.add_argument("--max-hidden", type=int, default=None)
    parser.add_argument("--selection-mode", choices=["topk", "kmeans"], default="topk")

    parser.add_argument("--distill-steps", type=int, default=100)
    parser.add_argument("--distill-lr", type=float, default=1e-3)
    parser.add_argument("--learned-epochs", type=int, default=25)
    parser.add_argument("--learned-lr", type=float, default=1e-3)
    parser.add_argument("--learned-code-dim", type=int, default=32)
    parser.add_argument("--output", type=str, default="runs/nanochat_ffn_compression/result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(
            source=args.teacher_source,
            hf_model_name=args.hf_model,
            nanochat_repo=args.nanochat_repo,
            nanochat_source=args.nanochat_source,
            nanochat_model_tag=args.model_tag,
            nanochat_step=args.step,
            device=args.device,
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
    budget = CompressionBudgetSpec(
        hidden_fraction=args.hidden_fraction,
        min_hidden=args.min_hidden,
        max_hidden=args.max_hidden,
        selection_mode=args.selection_mode,
    )

    geometric = run_ffn_geometric_compression(teacher, corpus, budget)
    distilled = run_ffn_distillation_baseline(
        DistillationSpec(
            teacher=teacher,
            corpus=corpus,
            budget=budget,
            steps=args.distill_steps,
            lr=args.distill_lr,
        )
    )
    learned = run_ffn_learned_compressor(
        LearnedCompressorSpec(
            teacher=teacher,
            corpus=corpus,
            budget=budget,
            epochs=args.learned_epochs,
            lr=args.learned_lr,
            code_dim=args.learned_code_dim,
        )
    )

    payload = {
        "teacher": {
            "model_name": teacher.model_name,
            "model_family": teacher.model_family,
            "dataset_name": teacher.dataset_name,
            "layer_ids": list(teacher.layer_ids),
        },
        "budget": {
            "hidden_fraction": budget.hidden_fraction,
            "min_hidden": budget.min_hidden,
            "max_hidden": budget.max_hidden,
            "selection_mode": budget.selection_mode,
        },
        "results": {
            "geometric": result_to_dict(geometric),
            "distilled": result_to_dict(distilled),
            "learned_autoencoder": result_to_dict(learned),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
