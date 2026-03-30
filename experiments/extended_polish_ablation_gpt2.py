"""Cycle 116: Extended-polish ablation — GPT-2 FFN (item 132).

Compress k=4 FFN blocks [7,5,4,1] at 6x with topk+lstsq. Test 4 conditions:
  A: 500 Adam steps on W_down only (baseline)
  B: 2500 Adam steps on W_down only (extended polish)
  C: 500 Adam on W_down, freeze, insert r=8 corrections, 2000 steps CE (cycle 115 recipe)
  D: 500 Adam on W_down, continue 2000 MORE Adam steps on W_down (matched budget, no corrections)

For B and D, log eval PPL at steps {500, 1000, 1500, 2000, 2500}.
3 seeds [42, 123, 7], 200 eval seqs OWT.

Criteria:
  CONFIRMED    if C outperforms B and D by >=5% relative PPL
  REFUTED      if B or D within 2% of C
  INCONCLUSIVE if 2-5% gap, or D converges to C
"""
from __future__ import annotations

import copy
import json
import math
import os
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
SEEDS = [42, 123, 7]
EVAL_SEQS = 200
SEQ_LEN = 128
BATCH_SIZE = 4

FFN_BLOCKS = [7, 5, 4, 1]
HIDDEN_FRACTION = 1 / 6
POLISH_STEPS = 500
POLISH_LR = 1e-4
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200

CORRECTION_RANK = 8
CORRECTION_TRAIN_STEPS = 2000
CORRECTION_LR = 1e-4
EXTENDED_POLISH_STEPS = 2500          # for condition B
EXTRA_POLISH_STEPS = 2000             # extra steps for condition D (total = 500+2000=2500)
D_MODEL = 768

CHECKPOINT_STEPS = [500, 1000, 1500, 2000, 2500]  # for B and D learning curves

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


def _find_compressed_mlp(model: nn.Module, block_id: int):
    try:
        block = model.transformer.h[block_id]
    except (AttributeError, IndexError):
        return None
    return getattr(block, "mlp", None)


def _get_wdown_params(model: nn.Module, block_ids: list[int]) -> list[nn.Parameter]:
    """Get W_down (c_proj.weight and bias) parameters for specified blocks."""
    params = []
    for block_id in block_ids:
        mlp = _find_compressed_mlp(model, block_id)
        if mlp is None or not hasattr(mlp, "c_proj"):
            continue
        w_down = mlp.c_proj.weight
        w_down.requires_grad_(True)
        params.append(w_down)
        b_down = getattr(mlp.c_proj, "bias", None)
        if b_down is not None:
            b_down.requires_grad_(True)
            params.append(b_down)
    return params


def _polish_wdown(
    student_model: nn.Module,
    block_ids: list[int],
    polish_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
    seed: int,
    checkpoint_steps: Optional[list[int]] = None,
    teacher: Optional[TeacherBundle] = None,
    corpus=None,
) -> tuple[list[float], dict[int, float]]:
    """Polish W_down for all specified blocks jointly.

    Returns (loss_curve, {step: ppl}) where ppl is evaluated at checkpoint_steps.
    Checkpoints require teacher and corpus.
    """
    student_model.requires_grad_(False)
    params_to_optimize = _get_wdown_params(student_model, block_ids)
    if not params_to_optimize:
        print("  WARNING: no W_down params found, skipping polish")
        return [], {}

    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    student_model.train()
    losses = []
    checkpoint_ppls: dict[int, float] = {}
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

        current_step = step + 1
        if checkpoint_steps is not None and current_step in checkpoint_steps:
            if teacher is not None and corpus is not None:
                student_model.eval()
                student_model.requires_grad_(False)
                metrics = _evaluate_model(teacher, student_model, corpus, batch_size=BATCH_SIZE)
                ppl = float(metrics["perplexity"])
                checkpoint_ppls[current_step] = ppl
                print(f"    step {current_step}: loss={losses[-1]:.4f}, eval_ppl={ppl:.3f}")
                student_model.train()
                # Re-enable gradients for params
                for p in params_to_optimize:
                    p.requires_grad_(True)
            else:
                print(f"    step {current_step}: loss={losses[-1]:.4f}")

        if (step + 1) % 500 == 0 and (checkpoint_steps is None or current_step not in checkpoint_steps):
            print(f"    step {current_step}/{n_steps}: loss={losses[-1]:.4f}")

    student_model.eval()
    student_model.requires_grad_(False)
    return losses, checkpoint_ppls


def _continue_polish_wdown(
    student_model: nn.Module,
    block_ids: list[int],
    polish_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
    seed: int,
    step_offset: int,
    checkpoint_steps: Optional[list[int]] = None,
    teacher: Optional[TeacherBundle] = None,
    corpus=None,
) -> tuple[list[float], dict[int, float]]:
    """Continue polishing W_down after initial 500 steps (for condition D).

    Creates a fresh optimizer (as specified in plan for D).
    """
    student_model.requires_grad_(False)
    params_to_optimize = _get_wdown_params(student_model, block_ids)
    if not params_to_optimize:
        return [], {}

    # Fresh optimizer (per plan: reset optimizer for D's second phase)
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    student_model.train()
    losses = []
    checkpoint_ppls: dict[int, float] = {}
    n_seqs = polish_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(seed + 1000)  # different seed for second phase

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

        current_step = step_offset + step + 1
        if checkpoint_steps is not None and current_step in checkpoint_steps:
            if teacher is not None and corpus is not None:
                student_model.eval()
                student_model.requires_grad_(False)
                metrics = _evaluate_model(teacher, student_model, corpus, batch_size=BATCH_SIZE)
                ppl = float(metrics["perplexity"])
                checkpoint_ppls[current_step] = ppl
                print(f"    step {current_step}: loss={losses[-1]:.4f}, eval_ppl={ppl:.3f}")
                student_model.train()
                for p in params_to_optimize:
                    p.requires_grad_(True)
            else:
                print(f"    step {current_step}: loss={losses[-1]:.4f}")

        if (step + 1) % 500 == 0 and (checkpoint_steps is None or current_step not in checkpoint_steps):
            print(f"    step {current_step}/{step_offset + n_steps}: loss={losses[-1]:.4f}")

    student_model.eval()
    student_model.requires_grad_(False)
    return losses, checkpoint_ppls


def _extract_wdown_weights(model: nn.Module, block_ids: list[int]) -> dict[int, dict]:
    """Extract W_down weights for all specified blocks."""
    weights = {}
    for bid in block_ids:
        mlp = _find_compressed_mlp(model, bid)
        if mlp is None or not hasattr(mlp, "c_proj"):
            continue
        weights[bid] = {
            "c_proj_weight": mlp.c_proj.weight.detach().clone(),
            "c_proj_bias": (
                mlp.c_proj.bias.detach().clone()
                if hasattr(mlp.c_proj, "bias") and mlp.c_proj.bias is not None
                else None
            ),
        }
    return weights


def _inject_wdown_weights(model: nn.Module, weights: dict[int, dict]) -> None:
    """Inject W_down weights for all blocks."""
    for bid, w in weights.items():
        mlp = _find_compressed_mlp(model, bid)
        if mlp is None or not hasattr(mlp, "c_proj"):
            continue
        with torch.no_grad():
            mlp.c_proj.weight.copy_(w["c_proj_weight"])
            if w["c_proj_bias"] is not None and hasattr(mlp.c_proj, "bias") and mlp.c_proj.bias is not None:
                mlp.c_proj.bias.copy_(w["c_proj_bias"])


# ---------------------------------------------------------------------------
# Low-rank correction module (same as cycle 115)
# ---------------------------------------------------------------------------
class LowRankCorrection(nn.Module):
    """Additive low-rank perturbation: x + U @ (V @ x).
    U: (d, r), V: (r, d). Initialized with U=0 so identity at start.
    """
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(d_model, rank))
        self.V = nn.Parameter(torch.randn(rank, d_model) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (x @ self.V.t()) @ self.U.t()


class CorrectedModel(nn.Module):
    """Wraps compressed GPT-2 with per-block low-rank corrections."""
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
    """Train only the correction parameters under next-token CE loss."""
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


def _eval_corrected(corrected_model: CorrectedModel, teacher: TeacherBundle, corpus) -> float:
    """Evaluate corrected model PPL with hooks installed."""
    corrected_model._install_hooks()
    corrected_model.eval()
    try:
        metrics = _evaluate_model(teacher, corrected_model.base_model, corpus, batch_size=BATCH_SIZE)
    finally:
        corrected_model._remove_hooks()
    return float(metrics["perplexity"])


# ---------------------------------------------------------------------------
# Main
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

# Verify correction param count
expected_correction_params = len(FFN_BLOCKS) * 2 * D_MODEL * CORRECTION_RANK
print(f"Condition C correction params: {expected_correction_params:,}")
assert expected_correction_params == 49152, f"Unexpected: {expected_correction_params}"
print("Param count check passed.")

# ---------------------------------------------------------------------------
# Per-seed loop
# ---------------------------------------------------------------------------
per_seed: dict[int, dict] = {}

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

    # Baseline float PPL
    bl_metrics = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = float(bl_metrics["perplexity"])
    print(f"  Baseline PPL: {baseline_ppl:.3f}")

    if not (25.0 <= baseline_ppl <= 65.0):
        print(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [25, 65]")
        sys.exit(1)

    # Probe FFN geometry
    print(f"  Probing FFN geometry for blocks {sorted(FFN_BLOCKS)}...")
    _set_seed(seed)
    atlas = probe_ffn_geometry(
        teacher,
        corpus,
        layer_ids=tuple(sorted(FFN_BLOCKS)),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )
    polish_tokens = corpus.probe_holdout

    # Build the compressed model (lstsq init, no polish yet)
    print("  Building k=4 compressed model (lstsq init)...")
    _set_seed(seed)
    pack = _derive_block_pack_from_atlas(atlas, budget, FFN_BLOCKS)
    # We build one reference compressed model and deep-copy for each condition
    ref_compressed_model = swap_compressed_ffns(teacher, pack, inplace=False)
    ref_compressed_model.eval()

    # Eval uncorrected (pre-polish) — just for reference
    unc_metrics = _evaluate_model(teacher, ref_compressed_model, corpus, batch_size=BATCH_SIZE)
    uncorrected_ppl = float(unc_metrics["perplexity"])
    print(f"  Uncorrected PPL (pre-polish): {uncorrected_ppl:.3f}")

    if uncorrected_ppl > 3 * baseline_ppl:
        print(f"ABORT: uncorrected PPL {uncorrected_ppl:.2f} > 3x baseline")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Condition A: 500 Adam steps on W_down
    # -----------------------------------------------------------------------
    print("\n  [Condition A] 500 Adam steps on W_down...")
    model_A = copy.deepcopy(ref_compressed_model)
    _set_seed(seed)
    losses_A, _ = _polish_wdown(
        model_A, FFN_BLOCKS, polish_tokens, POLISH_STEPS, POLISH_LR,
        teacher.device, seed,
    )
    metrics_A = _evaluate_model(teacher, model_A, corpus, batch_size=BATCH_SIZE)
    ppl_A = float(metrics_A["perplexity"])
    print(f"  Condition A PPL: {ppl_A:.3f}")

    # Save polished weights for conditions C and D (both start from 500-step checkpoint)
    polished_500_weights = _extract_wdown_weights(model_A, FFN_BLOCKS)

    # -----------------------------------------------------------------------
    # Condition B: 2500 Adam steps on W_down (extended polish)
    # -----------------------------------------------------------------------
    print("\n  [Condition B] 2500 Adam steps on W_down (extended polish)...")
    model_B = copy.deepcopy(ref_compressed_model)
    _set_seed(seed)
    losses_B, checkpoint_ppls_B = _polish_wdown(
        model_B, FFN_BLOCKS, polish_tokens, EXTENDED_POLISH_STEPS, POLISH_LR,
        teacher.device, seed,
        checkpoint_steps=CHECKPOINT_STEPS,
        teacher=teacher,
        corpus=corpus,
    )
    metrics_B = _evaluate_model(teacher, model_B, corpus, batch_size=BATCH_SIZE)
    ppl_B = float(metrics_B["perplexity"])
    print(f"  Condition B PPL (2500 steps): {ppl_B:.3f}")
    print(f"  B learning curve: {checkpoint_ppls_B}")

    # -----------------------------------------------------------------------
    # Condition C: 500 Adam + r=8 corrections 2000 steps
    # -----------------------------------------------------------------------
    print(f"\n  [Condition C] 500 Adam + r={CORRECTION_RANK} corrections {CORRECTION_TRAIN_STEPS} steps...")
    model_C_base = copy.deepcopy(ref_compressed_model)
    _inject_wdown_weights(model_C_base, polished_500_weights)
    model_C_base.eval()
    model_C_base.requires_grad_(False)

    _set_seed(seed)
    corrected_C = CorrectedModel(model_C_base, FFN_BLOCKS, CORRECTION_RANK, D_MODEL)
    corrected_C._install_hooks()
    corrected_C = corrected_C.to(teacher.device)

    losses_C = _train_corrections(
        corrected_C,
        polish_tokens,
        n_steps=CORRECTION_TRAIN_STEPS,
        lr=CORRECTION_LR,
        device=teacher.device,
        seed=seed,
    )
    ppl_C = _eval_corrected(corrected_C, teacher, corpus)
    print(f"  Condition C PPL: {ppl_C:.3f}  (correction loss: {losses_C[0]:.4f} → {losses_C[-1]:.4f})")

    # -----------------------------------------------------------------------
    # Condition D: 500 Adam + 2000 MORE Adam on W_down (matched budget)
    # -----------------------------------------------------------------------
    print(f"\n  [Condition D] 500 Adam + 2000 more Adam on W_down (matched budget)...")
    model_D = copy.deepcopy(ref_compressed_model)
    _inject_wdown_weights(model_D, polished_500_weights)

    # Checkpoint at step 500 (already done via A)
    checkpoint_ppls_D = {500: ppl_A}

    _set_seed(seed)
    losses_D_extra, extra_checkpoint_ppls_D = _continue_polish_wdown(
        model_D, FFN_BLOCKS, polish_tokens, EXTRA_POLISH_STEPS, POLISH_LR,
        teacher.device, seed,
        step_offset=POLISH_STEPS,
        checkpoint_steps=CHECKPOINT_STEPS,
        teacher=teacher,
        corpus=corpus,
    )
    checkpoint_ppls_D.update(extra_checkpoint_ppls_D)

    metrics_D = _evaluate_model(teacher, model_D, corpus, batch_size=BATCH_SIZE)
    ppl_D = float(metrics_D["perplexity"])
    print(f"  Condition D PPL (500+2000 steps): {ppl_D:.3f}")
    print(f"  D learning curve: {checkpoint_ppls_D}")

    # -----------------------------------------------------------------------
    # Store seed results
    # -----------------------------------------------------------------------
    per_seed[seed] = {
        "baseline": baseline_ppl,
        "uncorrected": uncorrected_ppl,
        "ppl_A": ppl_A,
        "ppl_B": ppl_B,
        "ppl_C": ppl_C,
        "ppl_D": ppl_D,
        "checkpoint_ppls_B": {str(k): v for k, v in checkpoint_ppls_B.items()},
        "checkpoint_ppls_D": {str(k): v for k, v in checkpoint_ppls_D.items()},
        "losses_C": losses_C,
    }

    # Cleanup
    del model_A, model_B, model_C_base, corrected_C, model_D, ref_compressed_model

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATING\n{'='*60}")

mean_baseline = float(np.mean([per_seed[s]["baseline"] for s in SEEDS]))
std_baseline = float(np.std([per_seed[s]["baseline"] for s in SEEDS]))
mean_uncorrected = float(np.mean([per_seed[s]["uncorrected"] for s in SEEDS]))
std_uncorrected = float(np.std([per_seed[s]["uncorrected"] for s in SEEDS]))

mean_A = float(np.mean([per_seed[s]["ppl_A"] for s in SEEDS]))
std_A = float(np.std([per_seed[s]["ppl_A"] for s in SEEDS]))
mean_B = float(np.mean([per_seed[s]["ppl_B"] for s in SEEDS]))
std_B = float(np.std([per_seed[s]["ppl_B"] for s in SEEDS]))
mean_C = float(np.mean([per_seed[s]["ppl_C"] for s in SEEDS]))
std_C = float(np.std([per_seed[s]["ppl_C"] for s in SEEDS]))
mean_D = float(np.mean([per_seed[s]["ppl_D"] for s in SEEDS]))
std_D = float(np.std([per_seed[s]["ppl_D"] for s in SEEDS]))

print(f"Baseline:     PPL={mean_baseline:.3f} ± {std_baseline:.3f}")
print(f"Uncorrected:  PPL={mean_uncorrected:.3f} ± {std_uncorrected:.3f}")
print(f"Condition A:  PPL={mean_A:.3f} ± {std_A:.3f}  (500 Adam steps)")
print(f"Condition B:  PPL={mean_B:.3f} ± {std_B:.3f}  (2500 Adam steps)")
print(f"Condition C:  PPL={mean_C:.3f} ± {std_C:.3f}  (500 Adam + r=8 corrections)")
print(f"Condition D:  PPL={mean_D:.3f} ± {std_D:.3f}  (500+2000 Adam steps)")

# Verdict logic from plan
gap_B_pct = abs(mean_C - mean_B) / mean_C * 100.0
gap_D_pct = abs(mean_C - mean_D) / mean_C * 100.0

print(f"\nRelative gap C vs B: {gap_B_pct:.2f}%")
print(f"Relative gap C vs D: {gap_D_pct:.2f}%")

# C outperforms = C has lower PPL
c_better_than_B = mean_C < mean_B
c_better_than_D = mean_C < mean_D

if c_better_than_B and c_better_than_D and gap_B_pct >= 5.0 and gap_D_pct >= 5.0:
    verdict = "CONFIRMED"
elif (not c_better_than_B and gap_B_pct <= 2.0) or (not c_better_than_D and gap_D_pct <= 2.0):
    verdict = "REFUTED"
elif gap_B_pct <= 2.0 or gap_D_pct <= 2.0:
    # One condition within 2% of C
    verdict = "REFUTED"
elif gap_B_pct < 5.0 or gap_D_pct < 5.0:
    verdict = "INCONCLUSIVE"
else:
    # Both gaps >= 5% but C might not be strictly better
    verdict = "INCONCLUSIVE"

print(f"Verdict: {verdict}")

# ---------------------------------------------------------------------------
# Learning curve averages for B and D
# ---------------------------------------------------------------------------
def _avg_checkpoint_ppls(per_seed_dict: dict, cond_key: str) -> dict[str, float]:
    """Average checkpoint PPLs across seeds."""
    all_keys = set()
    for s in SEEDS:
        all_keys.update(per_seed_dict[s][cond_key].keys())
    avg = {}
    for k in sorted(all_keys):
        vals = [per_seed_dict[s][cond_key].get(k, float("nan")) for s in SEEDS]
        valid = [v for v in vals if not math.isnan(v)]
        avg[k] = float(np.mean(valid)) if valid else float("nan")
    return avg

avg_checkpoint_B = _avg_checkpoint_ppls(per_seed, "checkpoint_ppls_B")
avg_checkpoint_D = _avg_checkpoint_ppls(per_seed, "checkpoint_ppls_D")

print(f"\nB learning curve (avg): {avg_checkpoint_B}")
print(f"D learning curve (avg): {avg_checkpoint_D}")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
results = {
    "cycle": 116,
    "item": 132,
    "seeds": SEEDS,
    "ffn_blocks": FFN_BLOCKS,
    "hidden_fraction": HIDDEN_FRACTION,
    "polish_steps": POLISH_STEPS,
    "extended_polish_steps": EXTENDED_POLISH_STEPS,
    "extra_polish_steps": EXTRA_POLISH_STEPS,
    "polish_lr": POLISH_LR,
    "correction_rank": CORRECTION_RANK,
    "correction_train_steps": CORRECTION_TRAIN_STEPS,
    "correction_lr": CORRECTION_LR,
    "correction_params": expected_correction_params,
    "d_model": D_MODEL,
    "baseline_ppl": {"mean": mean_baseline, "std": std_baseline},
    "uncorrected_ppl": {"mean": mean_uncorrected, "std": std_uncorrected},
    "condition_A": {"mean_ppl": mean_A, "std_ppl": std_A, "description": "500 Adam on W_down"},
    "condition_B": {
        "mean_ppl": mean_B, "std_ppl": std_B, "description": "2500 Adam on W_down",
        "checkpoint_ppls_avg": avg_checkpoint_B,
    },
    "condition_C": {
        "mean_ppl": mean_C, "std_ppl": std_C,
        "description": f"500 Adam + r={CORRECTION_RANK} corrections {CORRECTION_TRAIN_STEPS} steps",
        "correction_params": expected_correction_params,
    },
    "condition_D": {
        "mean_ppl": mean_D, "std_ppl": std_D, "description": "500+2000 Adam on W_down",
        "checkpoint_ppls_avg": avg_checkpoint_D,
    },
    "gap_C_vs_B_pct": gap_B_pct,
    "gap_C_vs_D_pct": gap_D_pct,
    "c_better_than_B": c_better_than_B,
    "c_better_than_D": c_better_than_D,
    "verdict": verdict,
    "per_seed": {
        str(s): {
            "baseline": per_seed[s]["baseline"],
            "uncorrected": per_seed[s]["uncorrected"],
            "ppl_A": per_seed[s]["ppl_A"],
            "ppl_B": per_seed[s]["ppl_B"],
            "ppl_C": per_seed[s]["ppl_C"],
            "ppl_D": per_seed[s]["ppl_D"],
            "checkpoint_ppls_B": per_seed[s]["checkpoint_ppls_B"],
            "checkpoint_ppls_D": per_seed[s]["checkpoint_ppls_D"],
        }
        for s in SEEDS
    },
}

json_path = RESULTS_DIR / "cycle_116_extended_polish_ablation.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
# 1. Bar chart: final PPL for A/B/C/D with error bars
fig, ax = plt.subplots(figsize=(9, 5))
labels = ["float", "uncorr", "A (500)", "B (2500)", "C (500+corr)", "D (500+2000)"]
means_ = [mean_baseline, mean_uncorrected, mean_A, mean_B, mean_C, mean_D]
stds_ = [std_baseline, std_uncorrected, std_A, std_B, std_C, std_D]
colors_ = ["#4CAF50", "#F44336", "#FF9800", "#FF5722", "#2196F3", "#9C27B0"]
x_ = np.arange(len(labels))
bars_ = ax.bar(x_, means_, yerr=stds_, color=colors_, capsize=5, alpha=0.85)
ax.set_xticks(x_)
ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("Perplexity")
ax.set_title(
    f"Extended-Polish Ablation — GPT-2 k=4 [7,5,4,1] 6x\n"
    f"Verdict: {verdict}  (C vs B: {gap_B_pct:.1f}%, C vs D: {gap_D_pct:.1f}%)"
)
ax.axhline(mean_baseline, color="#4CAF50", linestyle="--", alpha=0.5, label="float baseline")
for bar_, m_ in zip(bars_, means_):
    ax.text(bar_.get_x() + bar_.get_width() / 2, m_ + 0.3, f"{m_:.2f}",
            ha="center", va="bottom", fontsize=8)
ax.legend()
plt.tight_layout()
bar_path = RESULTS_DIR / "cycle_116_ppl_bar.png"
plt.savefig(bar_path, dpi=120)
plt.close()
print(f"Saved bar chart: {bar_path}")

# 2. Learning curves for B and D
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

# Condition B
ax_B = axes[0]
for s in SEEDS:
    ckpt_B = per_seed[s]["checkpoint_ppls_B"]
    steps_B = sorted(int(k) for k in ckpt_B.keys())
    ppls_B = [ckpt_B[str(st)] for st in steps_B]
    ax_B.plot(steps_B, ppls_B, marker="o", alpha=0.6, label=f"seed={s}")
# Plot average
avg_steps_B = sorted(int(k) for k in avg_checkpoint_B.keys())
avg_ppls_B = [avg_checkpoint_B[str(st)] for st in avg_steps_B]
ax_B.plot(avg_steps_B, avg_ppls_B, "k-", linewidth=2, marker="s", label="avg")
ax_B.axhline(mean_C, color="#2196F3", linestyle="--", alpha=0.7, label=f"C PPL={mean_C:.2f}")
ax_B.set_title(f"Condition B: Extended Adam (avg final={mean_B:.2f})")
ax_B.set_xlabel("Adam steps")
ax_B.set_ylabel("Eval PPL")
ax_B.legend(fontsize=7)
ax_B.grid(alpha=0.3)

# Condition D
ax_D = axes[1]
for s in SEEDS:
    ckpt_D = per_seed[s]["checkpoint_ppls_D"]
    steps_D = sorted(int(k) for k in ckpt_D.keys())
    ppls_D = [ckpt_D[str(st)] for st in steps_D]
    ax_D.plot(steps_D, ppls_D, marker="o", alpha=0.6, label=f"seed={s}")
avg_steps_D = sorted(int(k) for k in avg_checkpoint_D.keys())
avg_ppls_D = [avg_checkpoint_D[str(st)] for st in avg_steps_D]
ax_D.plot(avg_steps_D, avg_ppls_D, "k-", linewidth=2, marker="s", label="avg")
ax_D.axhline(mean_C, color="#2196F3", linestyle="--", alpha=0.7, label=f"C PPL={mean_C:.2f}")
ax_D.set_title(f"Condition D: Adam-only matched budget (avg final={mean_D:.2f})")
ax_D.set_xlabel("Adam steps (cumulative)")
ax_D.set_ylabel("Eval PPL")
ax_D.legend(fontsize=7)
ax_D.grid(alpha=0.3)

plt.suptitle("Learning Curves: B vs D (dashed = C correction target)")
plt.tight_layout()
curves_path = RESULTS_DIR / "cycle_116_learning_curves.png"
plt.savefig(curves_path, dpi=120)
plt.close()
print(f"Saved learning curves: {curves_path}")

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
manifest_path = RESULTS_DIR / "cycle_116_manifest.txt"
with open(manifest_path, "w") as f:
    f.write("Cycle 116: Extended-polish ablation — GPT-2 FFN\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Item: 132\n")
    f.write(f"Blocks: {FFN_BLOCKS} at 6x (topk+lstsq)\n")
    f.write(f"Seeds: {SEEDS}, Eval seqs: {EVAL_SEQS}\n\n")
    f.write(f"Conditions:\n")
    f.write(f"  A: 500 Adam steps on W_down (baseline)\n")
    f.write(f"  B: 2500 Adam steps on W_down (extended polish)\n")
    f.write(f"  C: 500 Adam + r={CORRECTION_RANK} corrections {CORRECTION_TRAIN_STEPS} steps ({expected_correction_params:,} params)\n")
    f.write(f"  D: 500+2000 Adam steps on W_down (matched budget)\n\n")
    f.write(f"Results:\n")
    f.write(f"  Float baseline: {mean_baseline:.3f} ± {std_baseline:.3f}\n")
    f.write(f"  Uncorrected:    {mean_uncorrected:.3f} ± {std_uncorrected:.3f}\n")
    f.write(f"  A (500 Adam):   {mean_A:.3f} ± {std_A:.3f}\n")
    f.write(f"  B (2500 Adam):  {mean_B:.3f} ± {std_B:.3f}\n")
    f.write(f"  C (500+corr):   {mean_C:.3f} ± {std_C:.3f}\n")
    f.write(f"  D (500+2000):   {mean_D:.3f} ± {std_D:.3f}\n\n")
    f.write(f"Gaps:\n")
    f.write(f"  C vs B: {gap_B_pct:.2f}% relative  (C {'better' if c_better_than_B else 'worse'} than B)\n")
    f.write(f"  C vs D: {gap_D_pct:.2f}% relative  (C {'better' if c_better_than_D else 'worse'} than D)\n\n")
    f.write(f"B learning curve (avg): {avg_checkpoint_B}\n")
    f.write(f"D learning curve (avg): {avg_checkpoint_D}\n\n")
    f.write(f"Verdict: {verdict}\n\n")
    f.write(f"Criteria:\n")
    f.write(f"  CONFIRMED if C outperforms B and D by >=5% relative PPL\n")
    f.write(f"  REFUTED if B or D within 2% of C\n")
    f.write(f"  INCONCLUSIVE if 2-5% gap\n\n")
    f.write(f"Files:\n")
    f.write(f"  {json_path}\n")
    f.write(f"  {bar_path}\n")
    f.write(f"  {curves_path}\n")
    f.write(f"  {manifest_path}\n")
    f.write(f"\nTSV row:\n")
    f.write(
        f"116\t132\textended_polish_ablation_gpt2\t{verdict}\t"
        f"A={mean_A:.2f},B={mean_B:.2f},C={mean_C:.2f},D={mean_D:.2f},"
        f"gap_B={gap_B_pct:.1f}%,gap_D={gap_D_pct:.1f}%,baseline={mean_baseline:.2f}\n"
    )

print(f"Saved manifest: {manifest_path}")
print("\nDone.")
