"""Cycle 163 / Item 173: Per-block error contribution profiling — stacked topk+lstsq — GPT-2 FFN k=4.

Phase 1: Progressive stacking profile. Compress blocks {0..i} for i=0..11, record PPL at each step.
Phase 2: Correction concentration test. Identify top-3 blocks by marginal PPL, compare:
  - Condition A: LoRA r=8 on top-3 blocks only (~37K params)
  - Condition B: LoRA r=11 on top-3 blocks only (~49K params, budget-matched)
  - Uniform baseline: LoRA r=8 on standard blocks [1,4,7,11] (~49K params)

Verdict:
  CONFIRMED    if top-3 contribute >60% of total degradation AND (A or B) improves >= 1 PPL over uniform
  REFUTED      if no Pareto structure (no block >15%) OR concentrated corrections >= 1 PPL worse
  INCONCLUSIVE if Pareto structure exists but correction neither helps nor hurts (within 1 PPL)
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnLayer,
    CompressedFfnPack,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _iter_lm_batches,
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

ALL_BLOCKS = list(range(12))         # blocks 0..11 for Phase 1
UNIFORM_BLOCKS = [1, 4, 7, 11]       # standard 4 blocks for baseline
HIDDEN_FRACTION = 1 / 6
POLISH_STEPS = 500
POLISH_LR = 1e-4
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200

# Phase 2 correction config
CORRECTION_TRAIN_STEPS = 2000
CORRECTION_LR = 1e-4
D_MODEL = 768
RANK_A = 8    # condition A: r=8 on top-3
RANK_B = 11   # condition B: r=11 on top-3 (budget-match: 3*2*768*11 ≈ 4*2*768*8)
RANK_UNIFORM = 8  # uniform: r=8 on 4 blocks

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _param_count(w: torch.Tensor, b: Optional[torch.Tensor]) -> int:
    return w.numel() + (0 if b is None else b.numel())


def _derive_block_pack_from_atlas(
    atlas: FfnGeometryAtlas,
    budget: CompressionBudgetSpec,
    block_ids: list[int],
) -> CompressedFfnPack:
    compressed_layers = []
    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
            continue
        k = _layer_target_hidden(layer, budget)
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
            _standard_weight(layer.handle.c_proj), _standard_bias(layer.handle.c_proj)
        )
        compressed_param_count = _param_count(student_w_in, student_b_in) + _param_count(
            student_w_out, student_b_out
        )
        compressed_layers.append(
            CompressedFfnLayer(
                layer_id=layer.layer_id,
                activation_kind=layer.activation_kind,
                source_hidden_dim=layer.hidden_dim,
                compressed_hidden_dim=int(k),
                c_fc_weight=student_w_in,
                c_fc_bias=student_b_in,
                c_proj_weight=student_w_out,
                c_proj_bias=student_b_out,
                selected_indices=selected,
                source_param_count=source_param_count,
                compressed_param_count=compressed_param_count,
                notes={"selection_mode": budget.selection_mode},
            )
        )
    return CompressedFfnPack(
        method="geometric",
        model_family=atlas.model_family,
        dataset_name=atlas.dataset_name,
        layers=tuple(compressed_layers),
        metadata={"sequence_length": atlas.sequence_length},
    )


# ---------------------------------------------------------------------------
# Low-rank correction module (identical to cascade_correction_stacking_gpt2)
# ---------------------------------------------------------------------------
class LowRankCorrection(nn.Module):
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(d_model, rank))
        self.V = nn.Parameter(torch.randn(rank, d_model) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (x @ self.V.t()) @ self.U.t()


class CorrectedModel(nn.Module):
    def __init__(self, base_model: nn.Module, block_ids: list[int], rank: int, d_model: int):
        super().__init__()
        self.base_model = base_model
        self.block_ids = sorted(block_ids)
        self.corrections = nn.ModuleDict({
            str(bid): LowRankCorrection(d_model, rank) for bid in block_ids
        })
        self._hooks: list = []

    def _install_hooks(self) -> None:
        self._remove_hooks()
        for bid in self.block_ids:
            correction = self.corrections[str(bid)]
            block = self.base_model.transformer.h[bid]

            def make_hook(corr):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        h_corrected = corr(h)
                        return (h_corrected,) + output[1:]
                    else:
                        return corr(output)
                return hook

            h = block.register_forward_hook(make_hook(correction))
            self._hooks.append(h)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, input_ids: torch.Tensor):
        return self.base_model(input_ids)

    def __del__(self):
        self._remove_hooks()


def _train_corrections(
    corrected_model: CorrectedModel,
    train_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> list[float]:
    corrected_model.base_model.requires_grad_(False)
    corrected_model.corrections.requires_grad_(True)

    params = list(corrected_model.corrections.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    corrected_model.base_model.eval()
    corrected_model.train()

    losses = []
    n_seqs = train_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(seed)

    for step in range(n_steps):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = train_tokens[idx].to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        optimizer.zero_grad()
        raw = corrected_model(inputs)
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{n_steps}: loss={loss.item():.4f}")

    corrected_model.eval()
    return losses


def _eval_corrected(
    corrected_model: CorrectedModel,
    teacher: TeacherBundle,
    corpus,
) -> float:
    corrected_model._install_hooks()
    corrected_model.eval()
    try:
        metrics = _evaluate_model(teacher, corrected_model.base_model, corpus, batch_size=BATCH_SIZE)
    finally:
        corrected_model._remove_hooks()
    return float(metrics["perplexity"])


# ---------------------------------------------------------------------------
# Load teacher
# ---------------------------------------------------------------------------
print("Loading GPT-2 teacher...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()
print(f"  Teacher loaded on device={teacher.device}")

budget = CompressionBudgetSpec(
    hidden_fraction=HIDDEN_FRACTION,
    min_hidden=32,
    selection_mode="topk",
)

# Param count verification
params_A = len([0, 1, 2]) * 2 * D_MODEL * RANK_A
params_B = len([0, 1, 2]) * 2 * D_MODEL * RANK_B
params_uniform = len(UNIFORM_BLOCKS) * 2 * D_MODEL * RANK_UNIFORM
print(f"Param counts: A(top3 r=8)={params_A:,}, B(top3 r=11)={params_B:,}, uniform(4 blocks r=8)={params_uniform:,}")

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

    # Baseline
    bl_metrics = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = float(bl_metrics["perplexity"])
    print(f"  Baseline PPL: {baseline_ppl:.3f}")

    if not (25.0 <= baseline_ppl <= 65.0):
        print(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [25, 65]")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # PHASE 1: Probe geometry for ALL 12 blocks
    # -----------------------------------------------------------------------
    print(f"  Probing FFN geometry for all 12 blocks...")
    _set_seed(seed)
    atlas = probe_ffn_geometry(
        teacher,
        corpus,
        layer_ids=tuple(ALL_BLOCKS),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )

    # Progressive stacking: compress {0..i} for i in 0..11
    stacking_ppls: list[float] = []
    prev_ppl = baseline_ppl

    print("  Phase 1: Progressive stacking profile...")
    for i in ALL_BLOCKS:
        active_blocks = list(range(i + 1))
        _set_seed(seed)
        pack = _derive_block_pack_from_atlas(atlas, budget, active_blocks)
        student = swap_compressed_ffns(teacher, pack, inplace=False)
        student.eval()
        metrics = _evaluate_model(teacher, student, corpus, batch_size=BATCH_SIZE)
        ppl_i = float(metrics["perplexity"])
        marginal = ppl_i - prev_ppl
        stacking_ppls.append(ppl_i)
        print(f"    blocks {{0..{i}}}: PPL={ppl_i:.3f}  marginal={marginal:+.3f}")
        prev_ppl = ppl_i
        del student

    # Compute marginals
    marginals = []
    for i, ppl in enumerate(stacking_ppls):
        prev = baseline_ppl if i == 0 else stacking_ppls[i - 1]
        marginals.append(float(ppl - prev))

    total_degradation = stacking_ppls[-1] - baseline_ppl
    print(f"  Total degradation: {total_degradation:.3f}")
    print(f"  Per-block marginals: {[f'{m:+.3f}' for m in marginals]}")

    # Identify top-3 blocks by marginal contribution
    sorted_blocks = sorted(range(12), key=lambda i: marginals[i], reverse=True)
    top3_blocks = sorted_blocks[:3]
    top3_fraction = sum(marginals[i] for i in top3_blocks) / total_degradation if total_degradation > 0 else 0.0
    print(f"  Top-3 blocks (highest marginal): {top3_blocks}")
    print(f"  Top-3 fraction of total degradation: {top3_fraction:.3f}")

    # -----------------------------------------------------------------------
    # PHASE 2: Correction concentration test
    # -----------------------------------------------------------------------
    print("\n  Phase 2: Correction concentration test...")

    # Build compressed model with ALL 12 blocks (no polish for Phase 2 comparisons)
    _set_seed(seed)
    full_pack = _derive_block_pack_from_atlas(atlas, budget, ALL_BLOCKS)
    full_compressed = swap_compressed_ffns(teacher, full_pack, inplace=False)
    full_compressed.eval()

    all_blocks_uncorrected_ppl = float(_evaluate_model(teacher, full_compressed, corpus, batch_size=BATCH_SIZE)["perplexity"])
    print(f"  All-12 blocks compressed PPL (uncorrected): {all_blocks_uncorrected_ppl:.3f}")

    # Also build the standard uniform baseline model (4 blocks)
    _set_seed(seed)
    uniform_pack = _derive_block_pack_from_atlas(atlas, budget, UNIFORM_BLOCKS)
    uniform_compressed = swap_compressed_ffns(teacher, uniform_pack, inplace=False)
    uniform_compressed.eval()
    uniform_uncorrected_ppl = float(_evaluate_model(teacher, uniform_compressed, corpus, batch_size=BATCH_SIZE)["perplexity"])
    print(f"  Uniform 4-block compressed PPL (uncorrected): {uniform_uncorrected_ppl:.3f}")

    correction_results = {}

    # Condition A: top-3 blocks, r=8
    print(f"  Condition A: LoRA r={RANK_A} on top-3 blocks {sorted(top3_blocks)}...")
    _set_seed(seed)
    # Build compressed model with top-3 blocks for focused correction
    top3_pack = _derive_block_pack_from_atlas(atlas, budget, top3_blocks)
    top3_compressed = swap_compressed_ffns(teacher, top3_pack, inplace=False)
    top3_compressed.eval()
    corr_A = CorrectedModel(copy.deepcopy(top3_compressed), top3_blocks, RANK_A, D_MODEL)
    corr_A._install_hooks()
    corr_A = corr_A.to(teacher.device)
    loss_curve_A = _train_corrections(corr_A, corpus.probe_holdout, CORRECTION_TRAIN_STEPS, CORRECTION_LR, teacher.device, seed)
    ppl_A = _eval_corrected(corr_A, teacher, corpus)
    params_A_actual = sum(p.numel() for p in corr_A.corrections.parameters())
    print(f"  Condition A PPL: {ppl_A:.3f}  params={params_A_actual:,}")
    correction_results["condition_A"] = {
        "blocks": sorted(top3_blocks),
        "rank": RANK_A,
        "params": params_A_actual,
        "ppl": ppl_A,
        "final_loss": float(loss_curve_A[-1]) if loss_curve_A else None,
    }
    del corr_A, top3_compressed

    # Condition B: top-3 blocks, r=11
    print(f"  Condition B: LoRA r={RANK_B} on top-3 blocks {sorted(top3_blocks)}...")
    _set_seed(seed)
    top3_pack_B = _derive_block_pack_from_atlas(atlas, budget, top3_blocks)
    top3_compressed_B = swap_compressed_ffns(teacher, top3_pack_B, inplace=False)
    top3_compressed_B.eval()
    corr_B = CorrectedModel(copy.deepcopy(top3_compressed_B), top3_blocks, RANK_B, D_MODEL)
    corr_B._install_hooks()
    corr_B = corr_B.to(teacher.device)
    loss_curve_B = _train_corrections(corr_B, corpus.probe_holdout, CORRECTION_TRAIN_STEPS, CORRECTION_LR, teacher.device, seed)
    ppl_B = _eval_corrected(corr_B, teacher, corpus)
    params_B_actual = sum(p.numel() for p in corr_B.corrections.parameters())
    print(f"  Condition B PPL: {ppl_B:.3f}  params={params_B_actual:,}")
    correction_results["condition_B"] = {
        "blocks": sorted(top3_blocks),
        "rank": RANK_B,
        "params": params_B_actual,
        "ppl": ppl_B,
        "final_loss": float(loss_curve_B[-1]) if loss_curve_B else None,
    }
    del corr_B, top3_compressed_B

    # Uniform baseline: 4 standard blocks, r=8
    print(f"  Uniform baseline: LoRA r={RANK_UNIFORM} on blocks {UNIFORM_BLOCKS}...")
    _set_seed(seed)
    corr_uniform = CorrectedModel(copy.deepcopy(uniform_compressed), UNIFORM_BLOCKS, RANK_UNIFORM, D_MODEL)
    corr_uniform._install_hooks()
    corr_uniform = corr_uniform.to(teacher.device)
    loss_curve_uniform = _train_corrections(corr_uniform, corpus.probe_holdout, CORRECTION_TRAIN_STEPS, CORRECTION_LR, teacher.device, seed)
    ppl_uniform = _eval_corrected(corr_uniform, teacher, corpus)
    params_uniform_actual = sum(p.numel() for p in corr_uniform.corrections.parameters())
    print(f"  Uniform baseline PPL: {ppl_uniform:.3f}  params={params_uniform_actual:,}")
    correction_results["uniform"] = {
        "blocks": UNIFORM_BLOCKS,
        "rank": RANK_UNIFORM,
        "params": params_uniform_actual,
        "ppl": ppl_uniform,
        "final_loss": float(loss_curve_uniform[-1]) if loss_curve_uniform else None,
    }
    del corr_uniform, uniform_compressed

    all_seed_results[seed] = {
        "baseline_ppl": baseline_ppl,
        "stacking_ppls": stacking_ppls,
        "marginals": marginals,
        "total_degradation": total_degradation,
        "top3_blocks": top3_blocks,
        "top3_fraction": top3_fraction,
        "all_blocks_uncorrected_ppl": all_blocks_uncorrected_ppl,
        "uniform_uncorrected_ppl": uniform_uncorrected_ppl,
        "correction_results": correction_results,
    }

# ---------------------------------------------------------------------------
# Aggregate across seeds and determine verdict
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATING RESULTS\n{'='*60}")

mean_top3_fraction = float(np.mean([r["top3_fraction"] for r in all_seed_results.values()]))
ppl_A_vals = [r["correction_results"]["condition_A"]["ppl"] for r in all_seed_results.values()]
ppl_B_vals = [r["correction_results"]["condition_B"]["ppl"] for r in all_seed_results.values()]
ppl_uniform_vals = [r["correction_results"]["uniform"]["ppl"] for r in all_seed_results.values()]
baseline_vals = [r["baseline_ppl"] for r in all_seed_results.values()]

mean_ppl_A = float(np.mean(ppl_A_vals))
std_ppl_A = float(np.std(ppl_A_vals))
mean_ppl_B = float(np.mean(ppl_B_vals))
std_ppl_B = float(np.std(ppl_B_vals))
mean_ppl_uniform = float(np.mean(ppl_uniform_vals))
std_ppl_uniform = float(np.std(ppl_uniform_vals))
mean_baseline = float(np.mean(baseline_vals))

# Per-block max marginal as fraction across seeds
max_single_fractions = []
for r in all_seed_results.values():
    td = r["total_degradation"]
    if td > 0:
        max_single_fractions.append(max(m / td for m in r["marginals"]))
mean_max_single_fraction = float(np.mean(max_single_fractions)) if max_single_fractions else 0.0

best_concentrated_ppl = min(mean_ppl_A, mean_ppl_B)
improvement_over_uniform = mean_ppl_uniform - best_concentrated_ppl

pareto_exists = mean_top3_fraction > 0.60
has_dominant_block = mean_max_single_fraction > 0.15

print(f"Mean baseline PPL: {mean_baseline:.3f}")
print(f"Mean top-3 fraction: {mean_top3_fraction:.3f}")
print(f"Mean max single-block fraction: {mean_max_single_fraction:.3f}")
print(f"Pareto structure (top3 > 60%): {pareto_exists}")
print(f"Has dominant block (> 15%): {has_dominant_block}")
print(f"Condition A (top-3 r=8): {mean_ppl_A:.3f} ± {std_ppl_A:.3f}")
print(f"Condition B (top-3 r=11): {mean_ppl_B:.3f} ± {std_ppl_B:.3f}")
print(f"Uniform (4 blocks r=8): {mean_ppl_uniform:.3f} ± {std_ppl_uniform:.3f}")
print(f"Best concentrated vs uniform: {improvement_over_uniform:+.3f} PPL improvement")

# Verdict logic
if not pareto_exists or not has_dominant_block:
    verdict = "REFUTED"
    verdict_reason = (
        f"No Pareto structure: top-3 fraction={mean_top3_fraction:.2f} "
        f"(need >0.60), max single-block fraction={mean_max_single_fraction:.2f} (need >0.15)"
    )
elif improvement_over_uniform >= 1.0:
    verdict = "CONFIRMED"
    verdict_reason = (
        f"Top-3 fraction={mean_top3_fraction:.2f} >60% AND "
        f"best concentrated PPL improves {improvement_over_uniform:.2f} over uniform (need >=1.0)"
    )
elif improvement_over_uniform <= -1.0:
    verdict = "REFUTED"
    verdict_reason = (
        f"Pareto exists (top-3={mean_top3_fraction:.2f}) but concentrated corrections "
        f"{-improvement_over_uniform:.2f} PPL worse than uniform (need not <=-1.0)"
    )
else:
    verdict = "INCONCLUSIVE"
    verdict_reason = (
        f"Pareto structure exists (top-3={mean_top3_fraction:.2f}) "
        f"but concentrated corrections within 1 PPL of uniform (delta={improvement_over_uniform:+.2f})"
    )

print(f"\nVERDICT: {verdict}")
print(f"Reason: {verdict_reason}")

# ---------------------------------------------------------------------------
# Build per-block marginal summary (averaged across seeds)
# ---------------------------------------------------------------------------
marginal_means = []
for block_idx in range(12):
    vals = [r["marginals"][block_idx] for r in all_seed_results.values()]
    marginal_means.append(float(np.mean(vals)))

total_degradation_mean = float(np.mean([r["total_degradation"] for r in all_seed_results.values()]))
marginal_fractions = [m / total_degradation_mean for m in marginal_means] if total_degradation_mean > 0 else [0.0] * 12

print(f"\nPer-block marginal contributions (mean across seeds):")
for i, (m, f) in enumerate(zip(marginal_means, marginal_fractions)):
    bar = "#" * int(f * 40)
    print(f"  Block {i:2d}: {m:+.3f} PPL  ({f*100:5.1f}%) {bar}")

# ---------------------------------------------------------------------------
# Plot: per-block marginal contributions
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Progressive stacking PPL
for seed in SEEDS:
    ppls = all_seed_results[seed]["stacking_ppls"]
    axes[0].plot(ALL_BLOCKS, ppls, alpha=0.4, marker="o", markersize=4, label=f"seed={seed}")
mean_stacking = [float(np.mean([all_seed_results[s]["stacking_ppls"][i] for s in SEEDS])) for i in range(12)]
axes[0].plot(ALL_BLOCKS, mean_stacking, "k-o", linewidth=2, markersize=6, label="mean")
axes[0].axhline(mean_baseline, color="gray", linestyle="--", label=f"baseline={mean_baseline:.1f}")
axes[0].set_xlabel("Blocks compressed (0..i)")
axes[0].set_ylabel("PPL")
axes[0].set_title("Progressive Stacking PPL")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Per-block marginal contributions
colors = ["#d62728" if i in sorted(all_seed_results[SEEDS[0]]["top3_blocks"]) else "#1f77b4" for i in range(12)]
axes[1].bar(ALL_BLOCKS, [f * 100 for f in marginal_fractions], color=colors, alpha=0.8)
axes[1].axhline(15, color="orange", linestyle="--", label="15% threshold")
axes[1].axhline(20, color="red", linestyle="--", label="20%")
axes[1].set_xlabel("Block index")
axes[1].set_ylabel("% of total degradation")
axes[1].set_title("Per-block marginal contribution (mean across seeds)\nRed = top-3 blocks")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis="y")
axes[1].set_xticks(ALL_BLOCKS)

plt.tight_layout()
plot_path = ROOT / "autoresearch" / "results" / "cycle_163_per_block_profile.png"
plt.savefig(plot_path, dpi=120)
plt.close()
print(f"\nPlot saved: {plot_path}")

# ---------------------------------------------------------------------------
# Save JSON results
# ---------------------------------------------------------------------------
output = {
    "cycle": 163,
    "item": 173,
    "verdict": verdict,
    "verdict_reason": verdict_reason,
    "summary": {
        "mean_baseline_ppl": mean_baseline,
        "mean_top3_fraction": mean_top3_fraction,
        "mean_max_single_fraction": mean_max_single_fraction,
        "pareto_exists": pareto_exists,
        "has_dominant_block": has_dominant_block,
        "best_concentrated_ppl": best_concentrated_ppl,
        "mean_ppl_A": mean_ppl_A,
        "std_ppl_A": std_ppl_A,
        "mean_ppl_B": mean_ppl_B,
        "std_ppl_B": std_ppl_B,
        "mean_ppl_uniform": mean_ppl_uniform,
        "std_ppl_uniform": std_ppl_uniform,
        "improvement_over_uniform": improvement_over_uniform,
        "per_block_marginal_means": marginal_means,
        "per_block_marginal_fractions": marginal_fractions,
        "total_degradation_mean": total_degradation_mean,
    },
    "per_seed": {
        str(s): {
            **all_seed_results[s],
            "top3_blocks": all_seed_results[s]["top3_blocks"],
        }
        for s in SEEDS
    },
    "config": {
        "seeds": SEEDS,
        "all_blocks": ALL_BLOCKS,
        "uniform_blocks": UNIFORM_BLOCKS,
        "hidden_fraction": HIDDEN_FRACTION,
        "polish_steps": POLISH_STEPS,
        "correction_train_steps": CORRECTION_TRAIN_STEPS,
        "correction_lr": CORRECTION_LR,
        "rank_A": RANK_A,
        "rank_B": RANK_B,
        "rank_uniform": RANK_UNIFORM,
    },
}

out_path = RESULTS_DIR / "cycle_163_per_block_profile.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Results saved: {out_path}")

print(f"\n{'='*60}")
print(f"FINAL VERDICT: {verdict}")
print(f"Top-3 fraction: {mean_top3_fraction:.3f}")
print(f"Condition A (top-3 r=8): {mean_ppl_A:.3f}")
print(f"Condition B (top-3 r=11): {mean_ppl_B:.3f}")
print(f"Uniform (4 blocks r=8): {mean_ppl_uniform:.3f}")
print(f"Improvement over uniform: {improvement_over_uniform:+.3f}")
print(f"{'='*60}")
