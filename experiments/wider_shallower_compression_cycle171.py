"""Cycle 171 / Item 181: Wider-shallower compression — GPT-2 FFN.

Test whether compressing MORE blocks at LIGHTER per-block ratios yields lower stacked PPL
than the current 4-block-at-6x recipe. Since cycle 163 showed error accumulates diffusely,
spreading compression thinly across all 12 blocks may reduce cascade damage.

Configs:
  Baseline: 4 blocks [1,4,7,11], hidden=512 (6x)   [established recipe]
  A: 12 blocks [0..11],   hidden=1024 (3x per block)
  B: 12 blocks [0..11],   hidden=1536 (2x per block)
  C: 8 blocks  [0..7],    hidden=768  (4x per block)

All: topk+lstsq, 500 joint Adam polish, 3 seeds, 200 OWT eval seqs.

Verdict:
  CONFIRMED    if mean PPL(A) < mean PPL(baseline) - 2
  REFUTED      if mean PPL(A) >= mean PPL(baseline)
  INCONCLUSIVE if within 2 PPL
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnLayer,
    CompressedFfnPack,
    CompressedMLP,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _layer_target_hidden,
    _select_hidden_units,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [42, 137, 7]
EVAL_SEQS = 200
SEQ_LEN = 128
BATCH_SIZE = 4
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200
POLISH_STEPS = 500
POLISH_LR = 1e-4

GPT2_FFN_HIDDEN = 3072

# Block configurations
CONFIGS = {
    "baseline_4x6x": {
        "blocks": [1, 4, 7, 11],
        "hidden": 512,
        "label": "4 blocks @ 6x",
    },
    "A_12x3x": {
        "blocks": list(range(12)),
        "hidden": 1024,
        "label": "12 blocks @ 3x",
    },
    "B_12x2x": {
        "blocks": list(range(12)),
        "hidden": 1536,
        "label": "12 blocks @ 2x",
    },
    "C_8x4x": {
        "blocks": list(range(8)),
        "hidden": 768,
        "label": "8 blocks @ 4x",
    },
}

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _param_count(w: torch.Tensor, b: Optional[torch.Tensor]) -> int:
    return w.numel() + (0 if b is None else b.numel())


def _find_compressed_mlp(model: nn.Module, block_id: int):
    try:
        block = model.transformer.h[block_id]
    except (AttributeError, IndexError):
        return None
    return getattr(block, "mlp", None)


def _derive_block_pack(
    atlas: FfnGeometryAtlas,
    block_ids: list[int],
    hidden_dim: int,
) -> CompressedFfnPack:
    """Derive compressed pack for specified blocks at specified hidden dim."""
    budget = CompressionBudgetSpec(
        hidden_fraction=hidden_dim / GPT2_FFN_HIDDEN,
        min_hidden=32,
        selection_mode="topk",
        per_layer_hidden={bid: hidden_dim for bid in block_ids},
    )
    compressed_layers = []
    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
            continue
        k = hidden_dim
        selected = _select_hidden_units(layer, budget, k)
        source_w_in = _standard_weight(layer.handle.c_fc).to(dtype=torch.float32)
        source_b_in = _standard_bias(layer.handle.c_fc)
        use_output_bias = _standard_bias(layer.handle.c_proj) is not None
        student_w_in = source_w_in[selected].contiguous()
        student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()
        hidden = layer.hidden_sample[:, selected].to(dtype=torch.float32)
        outputs = layer.output_sample.to(dtype=torch.float32)
        student_w_out, student_b_out = _fit_output_projection(hidden, outputs, use_bias=use_output_bias)
        source_param_count = _param_count(source_w_in, source_b_in) + _param_count(
            _standard_weight(layer.handle.c_proj), _standard_bias(layer.handle.c_proj),
        )
        compressed_param_count = _param_count(student_w_in, student_b_in) + _param_count(
            student_w_out, student_b_out,
        )
        compressed_layers.append(CompressedFfnLayer(
            layer_id=layer.layer_id,
            activation_kind=layer.activation_kind,
            source_hidden_dim=layer.hidden_dim,
            compressed_hidden_dim=k,
            c_fc_weight=student_w_in,
            c_fc_bias=student_b_in,
            c_proj_weight=student_w_out,
            c_proj_bias=student_b_out,
            selected_indices=selected,
            source_param_count=source_param_count,
            compressed_param_count=compressed_param_count,
            notes={"selection_mode": "topk"},
        ))
    return CompressedFfnPack(
        method="geometric",
        model_family="gpt2",
        dataset_name="openwebtext",
        layers=tuple(compressed_layers),
        metadata={"sequence_length": SEQ_LEN},
    )


def _joint_polish(
    student_model: nn.Module,
    block_ids: list[int],
    polish_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> list[float]:
    """Joint polish: optimize ALL compressed blocks' c_proj weights simultaneously.

    This is O(n_steps) forward passes total, regardless of block count.
    """
    if n_steps == 0:
        return []

    student_model.requires_grad_(False)

    params = []
    for bid in block_ids:
        mlp = _find_compressed_mlp(student_model, bid)
        if mlp is None or not hasattr(mlp, "c_proj"):
            continue
        w = mlp.c_proj.weight
        w.requires_grad_(True)
        params.append(w)
        b = getattr(mlp.c_proj, "bias", None)
        if b is not None:
            b.requires_grad_(True)
            params.append(b)

    if not params:
        return []

    optimizer = torch.optim.Adam(params, lr=lr)
    student_model.train()
    losses = []
    n_seqs = polish_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(seed)

    for step in range(n_steps):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = polish_tokens[idx].to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        optimizer.zero_grad()
        raw = student_model(inputs)
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    student_model.eval()
    student_model.requires_grad_(False)
    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2 teacher...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()
device = teacher.device
print(f"  Teacher on device={device}")

# Print param budget comparison
print("\nParam budget comparison (compressed FFN params per config):")
for name, cfg in CONFIGS.items():
    n_blocks = len(cfg["blocks"])
    hidden = cfg["hidden"]
    params_per_block = hidden * 768 + hidden + 768 * hidden + 768
    total = n_blocks * params_per_block
    ratio = GPT2_FFN_HIDDEN / hidden
    print(f"  {name}: {n_blocks} blocks x {hidden} hidden ({ratio:.1f}x) = {total:,} params")

# ---------------------------------------------------------------------------
# Per-seed loop
# ---------------------------------------------------------------------------
all_seed_results: dict[int, dict] = {}

for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
    _set_seed(seed)

    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher,
        text_path=str(OWT_PATH),
        sequence_length=SEQ_LEN,
        probe_train_sequences=PROBE_TRAIN_SEQS,
        probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
        eval_val_sequences=EVAL_SEQS,
        dataset_name="openwebtext",
        shuffle_seed=seed,
    ))

    bl = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = float(bl["perplexity"])
    print(f"  Teacher baseline PPL: {baseline_ppl:.3f}")

    if not (25.0 <= baseline_ppl <= 65.0):
        print(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [25, 65]")
        sys.exit(1)

    # Probe ALL 12 blocks
    print("  Probing FFN geometry for all 12 blocks...")
    _set_seed(seed)
    atlas = probe_ffn_geometry(
        teacher, corpus,
        layer_ids=tuple(range(12)),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )

    seed_results = {"baseline_ppl": baseline_ppl, "configs": {}}

    for config_name, cfg in CONFIGS.items():
        blocks = cfg["blocks"]
        hidden = cfg["hidden"]
        ratio = GPT2_FFN_HIDDEN / hidden
        print(f"\n  Config {config_name}: {len(blocks)} blocks @ {ratio:.0f}x (hidden={hidden})...")

        _set_seed(seed)
        pack = _derive_block_pack(atlas, blocks, hidden)

        # Unpolished eval
        student_raw = swap_compressed_ffns(teacher, pack, inplace=False)
        student_raw.eval()
        m_raw = _evaluate_model(teacher, student_raw, corpus, batch_size=BATCH_SIZE)
        ppl_raw = float(m_raw["perplexity"])
        print(f"    Unpolished PPL: {ppl_raw:.3f}")
        del student_raw

        # Joint polish (all blocks simultaneously, 500 steps total)
        student_polished = swap_compressed_ffns(teacher, pack, inplace=False)
        print(f"    Joint polish ({POLISH_STEPS} steps, {len(blocks)} blocks)...")
        losses = _joint_polish(
            student_polished, blocks, corpus.probe_holdout,
            POLISH_STEPS, POLISH_LR, device, seed,
        )
        if losses:
            print(f"      loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

        student_polished.eval()
        m_polished = _evaluate_model(teacher, student_polished, corpus, batch_size=BATCH_SIZE)
        ppl_polished = float(m_polished["perplexity"])
        print(f"    Polished PPL: {ppl_polished:.3f}")

        compressed_params = sum(l.compressed_param_count for l in pack.layers)
        source_params = sum(l.source_param_count for l in pack.layers)

        seed_results["configs"][config_name] = {
            "blocks": blocks,
            "hidden": hidden,
            "ratio": ratio,
            "n_blocks": len(blocks),
            "ppl_unpolished": ppl_raw,
            "ppl_polished": ppl_polished,
            "compressed_params": compressed_params,
            "source_params": source_params,
        }
        del student_polished

    all_seed_results[seed] = seed_results


# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATE RESULTS\n{'='*60}")

aggregated = {}
for config_name in CONFIGS:
    ppls_raw = [all_seed_results[s]["configs"][config_name]["ppl_unpolished"] for s in SEEDS]
    ppls_pol = [all_seed_results[s]["configs"][config_name]["ppl_polished"] for s in SEEDS]
    cfg = CONFIGS[config_name]
    mean_raw = sum(ppls_raw) / len(ppls_raw)
    std_raw = (sum((p - mean_raw)**2 for p in ppls_raw) / len(ppls_raw)) ** 0.5
    mean_pol = sum(ppls_pol) / len(ppls_pol)
    std_pol = (sum((p - mean_pol)**2 for p in ppls_pol) / len(ppls_pol)) ** 0.5

    aggregated[config_name] = {
        "label": cfg["label"],
        "n_blocks": len(cfg["blocks"]),
        "hidden": cfg["hidden"],
        "mean_ppl_unpolished": mean_raw,
        "std_ppl_unpolished": std_raw,
        "mean_ppl_polished": mean_pol,
        "std_ppl_polished": std_pol,
        "per_seed_unpolished": ppls_raw,
        "per_seed_polished": ppls_pol,
        "compressed_params": all_seed_results[SEEDS[0]]["configs"][config_name]["compressed_params"],
    }

# Print summary table
baseline_agg = aggregated["baseline_4x6x"]
print(f"\n{'Config':<25} {'Blk':<5} {'Hid':<6} {'PPL(raw)':<14} {'PPL(polish)':<14} {'Delta':<8} {'Params':>10}")
print("-" * 90)
for config_name, agg in aggregated.items():
    delta = agg["mean_ppl_polished"] - baseline_agg["mean_ppl_polished"]
    print(f"{agg['label']:<25} {agg['n_blocks']:<5} {agg['hidden']:<6} "
          f"{agg['mean_ppl_unpolished']:.2f}+/-{agg['std_ppl_unpolished']:.1f}  "
          f"{agg['mean_ppl_polished']:.2f}+/-{agg['std_ppl_polished']:.1f}  "
          f"{delta:+.2f}   "
          f"{agg['compressed_params']:>10,}")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
delta_A = aggregated["A_12x3x"]["mean_ppl_polished"] - baseline_agg["mean_ppl_polished"]
if delta_A <= -2.0:
    verdict = "CONFIRMED"
    verdict_note = (f"12@3x PPL ({aggregated['A_12x3x']['mean_ppl_polished']:.2f}) < "
                    f"4@6x PPL ({baseline_agg['mean_ppl_polished']:.2f}) by {-delta_A:.1f} points")
elif delta_A >= 0.0:
    verdict = "REFUTED"
    verdict_note = (f"12@3x PPL ({aggregated['A_12x3x']['mean_ppl_polished']:.2f}) >= "
                    f"4@6x PPL ({baseline_agg['mean_ppl_polished']:.2f})")
else:
    verdict = "INCONCLUSIVE"
    verdict_note = (f"12@3x PPL ({aggregated['A_12x3x']['mean_ppl_polished']:.2f}) vs "
                    f"4@6x PPL ({baseline_agg['mean_ppl_polished']:.2f}): delta={delta_A:.2f}")

print(f"\nVerdict: {verdict}")
print(f"  {verdict_note}")

delta_B = aggregated["B_12x2x"]["mean_ppl_polished"] - baseline_agg["mean_ppl_polished"]
delta_C = aggregated["C_8x4x"]["mean_ppl_polished"] - baseline_agg["mean_ppl_polished"]
print(f"\n  Config B (12@2x) delta: {delta_B:+.2f}")
print(f"  Config C (8@4x) delta: {delta_C:+.2f}")

best_name = min(
    [n for n in aggregated if n != "baseline_4x6x"],
    key=lambda n: aggregated[n]["mean_ppl_polished"],
)
best_delta = aggregated[best_name]["mean_ppl_polished"] - baseline_agg["mean_ppl_polished"]
print(f"  Best non-baseline: {best_name} ({aggregated[best_name]['label']}), delta={best_delta:+.2f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
baseline_ppls = [all_seed_results[s]["baseline_ppl"] for s in SEEDS]
result = {
    "cycle": 171,
    "item": 181,
    "verdict": verdict,
    "verdict_note": verdict_note,
    "seeds": SEEDS,
    "polish_steps": POLISH_STEPS,
    "polish_lr": POLISH_LR,
    "polish_mode": "joint",
    "eval_sequences": EVAL_SEQS,
    "teacher_ppl_mean": sum(baseline_ppls) / len(baseline_ppls),
    "teacher_ppl_per_seed": {str(s): all_seed_results[s]["baseline_ppl"] for s in SEEDS},
    "configs": aggregated,
    "per_seed_detail": {str(s): all_seed_results[s] for s in SEEDS},
    "delta_A_vs_baseline": float(delta_A),
    "delta_B_vs_baseline": float(delta_B),
    "delta_C_vs_baseline": float(delta_C),
    "best_nonbaseline": best_name,
    "best_nonbaseline_delta": float(best_delta),
}

out_path = RESULTS_DIR / "cycle_171_wider_shallower.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
