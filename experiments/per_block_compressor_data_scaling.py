"""Cycle 049: Per-block learned compressor vs geometric baseline — GPT-2 FFN.

Tests whether lifting the probe-token budget from 4K to 32K enables per-block
distillation (768->384->768) to beat the geometric (topk+lstsq) baseline.

Cycle 030 Baseline B used max_tokens=4096 and geometric won all tested blocks.
This experiment uses max_tokens=32768 (8x more data) on all 12 blocks.

Criterion:
  CONFIRMED  if 32K-token distillation beats geometric on >=8/12 blocks by >=1 PPL
  REFUTED    if geometric wins >=8/12 blocks even with 32K tokens
  INCONCLUSIVE otherwise
"""

from __future__ import annotations

import copy
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _collect_ffn_outputs,
    _evaluate_model,
    _extract_ffn_handles,
    _forward_logits,
    _iter_lm_batches,
    _parameters_for_target_ffns,
    _random_pack_like,
    _set_seed,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SHAKESPEARE_PATH = str(Path(__file__).parent.parent / "data" / "shakespeare.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"

ALL_BLOCKS = tuple(range(12))
REPRO_BLOCKS = (8, 9, 10, 11)   # blocks used in cycle 030

K_CHANNELS = 384          # 768->384->768  (matches cycle 030 Baseline B)
DISTILL_STEPS = 500       # matches cycle 030 Baseline B
LR = 1e-3                 # matches cycle 030 Baseline B
BATCH_SIZE = 4
SEED = 42

SEQ_LEN = 128

# probe_train_sequences * SEQ_LEN = total train tokens
LARGE_PROBE_SEQS = 256    # 256*128 = 32 768 tokens
SMALL_PROBE_SEQS = 32     # 32*128  =  4 096 tokens  (cycle 030 reproduction)

HOLDOUT_SEQS = 32
EVAL_VAL_SEQS = 32

# Cycle 030 Baseline B numbers for reference (sanity check)
CYCLE_030_REF = {
    8:  {"geometric_ppl": 122.53, "distill_ppl": 122.94},
    9:  {"geometric_ppl": 122.54, "distill_ppl": 122.91},
    10: {"geometric_ppl": 122.14, "distill_ppl": 122.88},
    11: {"geometric_ppl": 126.03, "distill_ppl": 131.86},
}


# ---------------------------------------------------------------------------
# Helper: per-block distillation with explicit probe corpus
# ---------------------------------------------------------------------------

def _run_per_block_distillation(
    teacher,
    corpus,
    block_id: int,
    k: int,
    steps: int,
    lr: float,
    batch_size: int,
    seed: int,
    max_tokens_per_layer: int,
) -> dict:
    """Train a single-block 768->k->768 compressor.

    Returns dict with ppl and block_mse.
    """
    _set_seed(seed)
    budget = CompressionBudgetSpec(
        hidden_fraction=k / 3072.0,
        min_hidden=k,
        max_hidden=k,
        selection_mode="topk",
    )

    # Probe geometry for this block only
    atlas = probe_ffn_geometry(
        teacher,
        corpus,
        layer_ids=(block_id,),
        max_tokens_per_layer=max_tokens_per_layer,
        batch_size=batch_size,
    )

    # Geometric pack (topk + lstsq) — used as initialisation shape, then randomised
    geo_pack = derive_ffn_compression(atlas, budget)
    distilled_pack = _random_pack_like(teacher, geo_pack, seed=seed)

    student_model = swap_compressed_ffns(teacher, distilled_pack, inplace=False).to(teacher.device)

    # Freeze everything except target block
    for param in student_model.parameters():
        param.requires_grad = False
    trainable = _parameters_for_target_ffns(
        student_model, {block_id}, teacher.activation_kind
    )
    for param in trainable:
        param.requires_grad = True

    optimizer = torch.optim.Adam(trainable, lr=lr)

    teacher_handles = [
        h for h in _extract_ffn_handles(teacher.model, teacher.activation_kind)
        if h.layer_id == block_id
    ]
    student_handles = [
        h for h in _extract_ffn_handles(student_model, teacher.activation_kind)
        if h.layer_id == block_id
    ]

    # Training loop
    step = 0
    while step < steps:
        for input_ids, targets in _iter_lm_batches(corpus.probe_train, batch_size, teacher.device):
            optimizer.zero_grad()
            with _collect_ffn_outputs(teacher_handles) as teacher_out, \
                 _collect_ffn_outputs(student_handles) as student_out:
                with torch.no_grad():
                    teacher_logits = _forward_logits(teacher.model, input_ids)
                student_logits = _forward_logits(student_model, input_ids)

            kl = F.kl_div(
                student_logits.log_softmax(dim=-1),
                teacher_logits.softmax(dim=-1),
                reduction="batchmean",
            )
            ce = F.cross_entropy(
                student_logits.reshape(-1, student_logits.shape[-1]),
                targets.reshape(-1),
            )
            layer_losses = []
            for lid in teacher_out:
                if teacher_out[lid] and student_out[lid]:
                    t = torch.cat(teacher_out[lid], dim=0).to(teacher.device)
                    s = torch.cat(student_out[lid], dim=0).to(teacher.device)
                    layer_losses.append(F.mse_loss(s, t))
            layer_loss = (
                torch.stack(layer_losses).mean()
                if layer_losses
                else torch.tensor(0.0, device=teacher.device)
            )

            loss = 1.0 * kl + 0.25 * ce + 0.2 * layer_loss
            loss.backward()
            optimizer.step()
            step += 1
            if step >= steps:
                break

    metrics = _evaluate_model(teacher, student_model, corpus, batch_size=batch_size)

    # Compute block MSE from atlas (input/output samples vs student forward)
    geom = atlas.layer_geometries[0]
    student_model.eval()
    with torch.no_grad():
        # Pass through teacher up to this block's MLP to get block MSE
        # Use the stored input/output samples directly
        student_block_handles = {
            h.layer_id: h
            for h in _extract_ffn_handles(student_model, teacher.activation_kind)
        }
        sh = student_block_handles[block_id]
        x_in = geom.input_sample.to(teacher.device)
        y_out = geom.output_sample.to(teacher.device)
        # Forward through student MLP
        z = sh.c_fc(x_in)
        if teacher.activation_kind == "gelu":
            import torch.nn as nn
            h_act = F.gelu(z)
        elif teacher.activation_kind == "relu":
            h_act = F.relu(z)
        else:
            h_act = F.gelu(z)
        y_pred = sh.c_proj(h_act)
        block_mse = F.mse_loss(y_pred, y_out).item()

    # Clean up student model to free GPU memory
    del student_model
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "ppl": metrics["perplexity"],
        "block_mse": block_mse,
        "metrics": metrics,
    }


def _run_geometric(
    teacher,
    corpus,
    block_id: int,
    k: int,
    batch_size: int,
    max_tokens_per_layer: int,
) -> dict:
    """Run geometric baseline (topk + lstsq) for a single block."""
    budget = CompressionBudgetSpec(
        hidden_fraction=k / 3072.0,
        min_hidden=k,
        max_hidden=k,
        selection_mode="topk",
    )
    atlas = probe_ffn_geometry(
        teacher,
        corpus,
        layer_ids=(block_id,),
        max_tokens_per_layer=max_tokens_per_layer,
        batch_size=batch_size,
    )
    geo_pack = derive_ffn_compression(atlas, budget)
    student = swap_compressed_ffns(teacher, geo_pack, inplace=False)
    metrics = _evaluate_model(teacher, student, corpus, batch_size=batch_size)

    # Block MSE from atlas
    geom = atlas.layer_geometries[0]
    layer = geo_pack.layers[0]
    # Use the fitted student weights directly
    from aleph.lm_geometric_compression import CompressedMLP, _load_compressed_layer
    mlp = CompressedMLP(
        input_dim=layer.c_fc_weight.shape[1],
        hidden_dim=layer.compressed_hidden_dim,
        output_dim=layer.c_proj_weight.shape[0],
        activation_kind=layer.activation_kind,
        c_fc_bias=layer.c_fc_bias is not None,
        c_proj_bias=layer.c_proj_bias is not None,
    )
    _load_compressed_layer(mlp, layer)
    with torch.no_grad():
        block_mse = F.mse_loss(
            mlp(geom.input_sample),
            geom.output_sample,
        ).item()

    del student
    gc.collect()

    return {
        "ppl": metrics["perplexity"],
        "block_mse": block_mse,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _set_seed(SEED)

    shakes_path = SHAKESPEARE_PATH
    if not Path(shakes_path).exists():
        for cand in [
            Path(__file__).parent.parent / "data" / "input.txt",
            Path(__file__).parent.parent / "data" / "tinyshakespeare.txt",
        ]:
            if cand.exists():
                shakes_path = str(cand)
                break
        else:
            raise FileNotFoundError(f"Shakespeare corpus not found at {shakes_path}")

    print("Loading GPT-2 teacher ...", flush=True)
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(
            source="hf_pretrained",
            hf_model_name="gpt2",
            device="auto",
        )
    )

    # --- Build corpora ---
    print("Building large probe corpus (32K tokens) ...", flush=True)
    corpus_large = build_probe_corpus(
        ProbeCorpusSpec(
            sequence_length=SEQ_LEN,
            probe_train_sequences=LARGE_PROBE_SEQS,
            probe_holdout_sequences=HOLDOUT_SEQS,
            eval_val_sequences=EVAL_VAL_SEQS,
            text_path=shakes_path,
            teacher=teacher,
        )
    )
    large_train_tokens = corpus_large.token_counts.get("probe_train", LARGE_PROBE_SEQS * SEQ_LEN)
    print(f"  probe_train tokens: {large_train_tokens}", flush=True)

    print("Building small probe corpus (4K tokens) ...", flush=True)
    corpus_small = build_probe_corpus(
        ProbeCorpusSpec(
            sequence_length=SEQ_LEN,
            probe_train_sequences=SMALL_PROBE_SEQS,
            probe_holdout_sequences=HOLDOUT_SEQS,
            eval_val_sequences=EVAL_VAL_SEQS,
            text_path=shakes_path,
            teacher=teacher,
        )
    )
    small_train_tokens = corpus_small.token_counts.get("probe_train", SMALL_PROBE_SEQS * SEQ_LEN)
    print(f"  probe_train tokens: {small_train_tokens}", flush=True)

    # Baseline PPL
    print("Computing baseline PPL ...", flush=True)
    baseline_metrics = _evaluate_model(teacher, teacher.model, corpus_large, batch_size=BATCH_SIZE)
    baseline_ppl = baseline_metrics["perplexity"]
    print(f"Baseline PPL: {baseline_ppl:.2f}", flush=True)

    results = {
        "baseline_ppl": baseline_ppl,
        "k_channels": K_CHANNELS,
        "distill_steps": DISTILL_STEPS,
        "lr": LR,
        "large_max_tokens": LARGE_PROBE_SEQS * SEQ_LEN,
        "small_max_tokens": SMALL_PROBE_SEQS * SEQ_LEN,
        "all_blocks_32k": {},
        "repro_4k": {},
        "cycle_030_ref": CYCLE_030_REF,
    }

    # -----------------------------------------------------------------------
    # Run A: all 12 blocks, 32K tokens
    # -----------------------------------------------------------------------
    print("\n=== Run A: all 12 blocks, 32K tokens ===", flush=True)
    for block_id in ALL_BLOCKS:
        print(f"\n  Block {block_id} ...", flush=True)

        # Geometric
        print(f"    Geometric ...", flush=True)
        geo = _run_geometric(
            teacher, corpus_large, block_id,
            k=K_CHANNELS, batch_size=BATCH_SIZE,
            max_tokens_per_layer=LARGE_PROBE_SEQS * SEQ_LEN,
        )
        print(f"    Geo   PPL={geo['ppl']:.2f}  blockMSE={geo['block_mse']:.4f}", flush=True)

        # Distillation
        print(f"    Distillation ({DISTILL_STEPS} steps, 32K tokens) ...", flush=True)
        distill = _run_per_block_distillation(
            teacher, corpus_large, block_id,
            k=K_CHANNELS, steps=DISTILL_STEPS, lr=LR,
            batch_size=BATCH_SIZE, seed=SEED,
            max_tokens_per_layer=LARGE_PROBE_SEQS * SEQ_LEN,
        )
        print(f"    Distil PPL={distill['ppl']:.2f}  blockMSE={distill['block_mse']:.4f}", flush=True)

        results["all_blocks_32k"][str(block_id)] = {
            "geometric_ppl": geo["ppl"],
            "geometric_block_mse": geo["block_mse"],
            "distill_ppl": distill["ppl"],
            "distill_block_mse": distill["block_mse"],
            "distill_beats_geo": bool(distill["ppl"] < geo["ppl"] - 1.0),
            "distill_margin": float(geo["ppl"] - distill["ppl"]),
        }

    # -----------------------------------------------------------------------
    # Run B: blocks 8-11, 4K tokens (cycle 030 Baseline B reproduction)
    # -----------------------------------------------------------------------
    print("\n=== Run B: blocks 8-11, 4K tokens (cycle 030 reproduction) ===", flush=True)
    for block_id in REPRO_BLOCKS:
        print(f"\n  Block {block_id} ...", flush=True)

        geo = _run_geometric(
            teacher, corpus_small, block_id,
            k=K_CHANNELS, batch_size=BATCH_SIZE,
            max_tokens_per_layer=SMALL_PROBE_SEQS * SEQ_LEN,
        )
        print(f"    Geo   PPL={geo['ppl']:.2f}", flush=True)

        distill = _run_per_block_distillation(
            teacher, corpus_small, block_id,
            k=K_CHANNELS, steps=DISTILL_STEPS, lr=LR,
            batch_size=BATCH_SIZE, seed=SEED,
            max_tokens_per_layer=SMALL_PROBE_SEQS * SEQ_LEN,
        )
        ref = CYCLE_030_REF.get(block_id, {})
        print(
            f"    Distil PPL={distill['ppl']:.2f}  "
            f"(ref distill={ref.get('distill_ppl','?')} geo={ref.get('geometric_ppl','?')})",
            flush=True,
        )

        results["repro_4k"][str(block_id)] = {
            "geometric_ppl": geo["ppl"],
            "distill_ppl": distill["ppl"],
            "ref_geometric_ppl": ref.get("geometric_ppl"),
            "ref_distill_ppl": ref.get("distill_ppl"),
        }

    # -----------------------------------------------------------------------
    # Evaluate criterion
    # -----------------------------------------------------------------------
    distill_wins = sum(
        1 for v in results["all_blocks_32k"].values()
        if v["distill_beats_geo"]
    )
    total_blocks = len(ALL_BLOCKS)

    if distill_wins >= 8:
        verdict = "CONFIRMED"
    elif (total_blocks - distill_wins) >= 8:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    results["distill_wins_32k"] = distill_wins
    results["total_blocks"] = total_blocks
    results["verdict"] = verdict

    print(f"\n=== VERDICT: {verdict} ===", flush=True)
    print(f"Distillation beats geometric (>=1 PPL): {distill_wins}/{total_blocks} blocks", flush=True)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'Block':>6} {'Geo PPL':>9} {'Dist PPL':>9} {'Margin':>8} {'Wins':>5}")
    print("-" * 45)
    for block_id in ALL_BLOCKS:
        v = results["all_blocks_32k"][str(block_id)]
        win_str = "YES" if v["distill_beats_geo"] else "no"
        print(
            f"{block_id:>6} {v['geometric_ppl']:>9.2f} {v['distill_ppl']:>9.2f} "
            f"{v['distill_margin']:>+8.2f} {win_str:>5}"
        )

    print(f"\nReproduction (4K tokens) vs cycle 030 ref:")
    print(f"{'Block':>6} {'Geo(4K)':>9} {'Dist(4K)':>9} {'RefGeo':>8} {'RefDist':>9}")
    print("-" * 50)
    for block_id in REPRO_BLOCKS:
        v = results["repro_4k"][str(block_id)]
        print(
            f"{block_id:>6} {v['geometric_ppl']:>9.2f} {v['distill_ppl']:>9.2f} "
            f"{v.get('ref_geometric_ppl', '?'):>8} {v.get('ref_distill_ppl', '?'):>9}"
        )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "cycle_049_per_block_data_scaling.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
