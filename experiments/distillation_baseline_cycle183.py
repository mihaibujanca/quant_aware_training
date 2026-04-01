"""Cycle 183 / Item 193: Matched-budget distillation control baseline — GPT-2 FFN k=4.

Train a student FFN (768→512→768, GELU, ~787K params) using local Adam+MSE distillation
from the float teacher, for blocks [1,4,7,11]. Uses pre-cached teacher block I/O to
avoid full-model forward per training step (feasible on CPU).

"Local distillation" = train student MLP to match teacher block output given teacher
block input via Adam+MSE. This is identical to the topk+lstsq objective but uses an
iterative optimizer instead of closed-form least-squares.

Comparison:
  - topk+lstsq: closed-form least-squares fit of subsampled teacher I/O
  - local distill: Adam+MSE on full teacher I/O, 500 steps (matched budget)
  - local distill converged: Adam+MSE on full teacher I/O, up to 2000 steps

Verdict (based on converged distillation):
  CONFIRMED    if topk+lstsq+polish beats distillation by ≥3 PPL (geometric justified)
  REFUTED      if distillation matches or beats topk+lstsq+polish (no geometric value)
  INCONCLUSIVE if within ±3 PPL
"""
from __future__ import annotations

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
DISTILL_STEPS_SNAPSHOT = 500     # matched-budget snapshot
DISTILL_STEPS_CONVERGED = 2000   # convergence run
LOG_AT_STEPS = {0, 100, 200, 500, 1000, 1500, 2000}
PLATEAU_WINDOW = 50
PLATEAU_THRESHOLD = 0.05
LR = 1e-4
BLOCKS = [1, 4, 7, 11]
STUDENT_HIDDEN = 512  # k=4 channel groups of 128 each
GPT2_FFN_HIDDEN = 3072
GPT2_HIDDEN = 768

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


def _find_mlp(model: nn.Module, block_id: int):
    return model.transformer.h[block_id].mlp


def _swap_mlp(model: nn.Module, block_id: int, new_mlp: nn.Module):
    model.transformer.h[block_id].mlp = new_mlp


def _make_student(device: torch.device) -> CompressedMLP:
    """Create randomly initialized student MLP: 768→512→768 with GELU."""
    student = CompressedMLP(
        input_dim=GPT2_HIDDEN,
        hidden_dim=STUDENT_HIDDEN,
        output_dim=GPT2_HIDDEN,
        activation_kind="gelu",
        c_fc_bias=True,
        c_proj_bias=True,
    )
    return student.to(device)


# ---------------------------------------------------------------------------
# Cache teacher block I/O
# ---------------------------------------------------------------------------
@torch.no_grad()
def _cache_block_io(
    teacher_model: nn.Module,
    block_id: int,
    tokens: torch.Tensor,
    device: torch.device,
    batch_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect (block_input, block_output) pairs for all tokens.

    Returns:
        X_in: (N_tokens, 768) block input hidden states
        X_out: (N_tokens, 768) block output (residual delta = mlp(ln(x)))
    """
    mlp = _find_mlp(teacher_model, block_id)
    ln = teacher_model.transformer.h[block_id].ln_2

    X_ins, X_outs = [], []

    teacher_model.eval()
    n_seqs = tokens.shape[0]
    for i in range(0, n_seqs, batch_size):
        batch = tokens[i : i + batch_size].to(device)
        inputs = batch[:, :-1].contiguous()  # (B, T-1)

        # Run model up to block output manually via hooks
        # Simpler: run full model but grab hidden state at block via hook
        hidden_in = []
        hidden_out = []

        def hook_in(module, input, output):
            # ln_2 output = normalized hidden; we want the pre-ln hidden
            # Actually we want the MLP input = ln_2(x), not x
            # So capture ln_2 output
            hidden_in.append(input[0].detach().cpu())  # input to ln_2 = residual stream before block

        def hook_mlp_out(module, input, output):
            hidden_out.append(output.detach().cpu())

        h1 = ln.register_forward_hook(hook_in)
        h2 = mlp.register_forward_hook(hook_mlp_out)
        try:
            teacher_model(inputs)
        finally:
            h1.remove()
            h2.remove()

        # hidden_in[0]: (B, T-1, 768) — residual stream at block input
        # hidden_out[0]: (B, T-1, 768) — MLP output (residual delta)
        X_ins.append(hidden_in[0].reshape(-1, GPT2_HIDDEN))
        X_outs.append(hidden_out[0].reshape(-1, GPT2_HIDDEN))

    return torch.cat(X_ins, dim=0), torch.cat(X_outs, dim=0)


# ---------------------------------------------------------------------------
# Local Adam+MSE distillation
# ---------------------------------------------------------------------------
def _distill_block_local(
    student_mlp: CompressedMLP,
    X_in: torch.Tensor,
    X_out: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
    seed: int,
    ln: nn.Module,
    log_at_steps: set | None = None,
    plateau_window: int = 50,
    plateau_threshold: float = 0.05,
) -> tuple[list[float], dict[int, float], int, bool]:
    """Train student_mlp to match teacher block output via Adam+MSE.

    The objective is: minimize ||student(ln(X_in)) - X_out||^2

    This is the same objective as lstsq, but solved with Adam.

    Returns:
        losses, logged_losses, actual_steps, converged_early
    """
    student_mlp = student_mlp.to(device)
    ln = ln.to(device)
    X_in = X_in.to(device)
    X_out = X_out.to(device)

    # Precompute ln(X_in) once (ln is frozen teacher norm)
    with torch.no_grad():
        ln_X_in = ln(X_in)  # (N, 768)

    optimizer = torch.optim.Adam(student_mlp.parameters(), lr=lr)
    student_mlp.train()

    N = X_in.shape[0]
    losses: list[float] = []
    logged: dict[int, float] = {}
    converged_early = False
    actual_steps = n_steps

    if log_at_steps is None:
        log_at_steps = set()

    rng = torch.Generator()
    rng.manual_seed(seed)

    for step in range(n_steps):
        # Mini-batch from cached data
        idx = torch.randperm(N, generator=rng)[:BATCH_SIZE * SEQ_LEN]
        x = ln_X_in[idx]    # (B*T, 768)
        y = X_out[idx]      # (B*T, 768)

        optimizer.zero_grad()
        pred = student_mlp(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        cur_step = step + 1
        if cur_step in log_at_steps:
            logged[cur_step] = losses[-1]

        # Plateau detection
        if cur_step >= 2 * plateau_window and cur_step % plateau_window == 0:
            prev = losses[cur_step - 2*plateau_window : cur_step - plateau_window]
            curr = losses[cur_step - plateau_window : cur_step]
            pm = sum(prev) / len(prev)
            cm = sum(curr) / len(curr)
            if pm > 0 and abs(pm - cm) / pm < plateau_threshold:
                converged_early = True
                actual_steps = cur_step
                break

    student_mlp.eval()
    return losses, logged, actual_steps, converged_early


# ---------------------------------------------------------------------------
# topk+lstsq+polish (reference)
# ---------------------------------------------------------------------------
def _derive_topk_pack(
    atlas: FfnGeometryAtlas,
    block_ids: list[int],
    hidden_dim: int,
) -> CompressedFfnPack:
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
    if n_steps == 0:
        return []
    student_model.requires_grad_(False)
    params = []
    for bid in block_ids:
        mlp = _find_mlp(student_model, bid)
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
print("Loading GPT-2 teacher...", flush=True)
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()
device = teacher.device
print(f"  Teacher on device={device}", flush=True)

params_per_block = STUDENT_HIDDEN * GPT2_HIDDEN + STUDENT_HIDDEN + GPT2_HIDDEN * STUDENT_HIDDEN + GPT2_HIDDEN
total_params = len(BLOCKS) * params_per_block
print(f"\nStudent: 768→{STUDENT_HIDDEN}→768 GELU, {params_per_block:,} params/block, {total_params:,} total (4 blocks)", flush=True)

# ---------------------------------------------------------------------------
# Per-seed loop
# ---------------------------------------------------------------------------
all_seed_results: dict[int, dict] = {}

for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
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
    print(f"  Teacher baseline PPL: {baseline_ppl:.3f}", flush=True)

    if not (15.0 <= baseline_ppl <= 65.0):
        print(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [15, 65]", flush=True)
        sys.exit(1)

    seed_results = {"baseline_ppl": baseline_ppl}

    # ===========================================================================
    # A. topk+lstsq+polish (reference)
    # ===========================================================================
    print("\n  --- topk+lstsq+polish (reference) ---", flush=True)
    _set_seed(seed)
    atlas = probe_ffn_geometry(
        teacher, corpus,
        layer_ids=tuple(BLOCKS),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )
    pack = _derive_topk_pack(atlas, BLOCKS, STUDENT_HIDDEN)

    # Single-block evals (topk, unpolished)
    topk_single_ppls = {}
    import copy
    for layer in pack.layers:
        single_pack = CompressedFfnPack(
            method="geometric", model_family="gpt2", dataset_name="openwebtext",
            layers=(layer,), metadata={"sequence_length": SEQ_LEN},
        )
        single_model = swap_compressed_ffns(teacher, single_pack, inplace=False)
        single_model.eval()
        m = _evaluate_model(teacher, single_model, corpus, batch_size=BATCH_SIZE)
        topk_single_ppls[layer.layer_id] = float(m["perplexity"])
        print(f"    topk single block {layer.layer_id}: PPL={topk_single_ppls[layer.layer_id]:.3f}", flush=True)
        del single_model

    # Stacked unpolished
    student_topk = swap_compressed_ffns(teacher, pack, inplace=False)
    student_topk.eval()
    m = _evaluate_model(teacher, student_topk, corpus, batch_size=BATCH_SIZE)
    topk_stacked_raw = float(m["perplexity"])
    print(f"    topk stacked unpolished: PPL={topk_stacked_raw:.3f}", flush=True)

    # Polish
    _set_seed(seed)
    losses = _joint_polish(student_topk, BLOCKS, corpus.probe_holdout, POLISH_STEPS, LR, device, seed)
    if losses:
        print(f"    polish loss: {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
    student_topk.eval()
    m = _evaluate_model(teacher, student_topk, corpus, batch_size=BATCH_SIZE)
    topk_stacked_polished = float(m["perplexity"])
    print(f"    topk stacked polished: PPL={topk_stacked_polished:.3f}", flush=True)
    del student_topk

    seed_results["topk"] = {
        "single_ppls": topk_single_ppls,
        "stacked_raw": topk_stacked_raw,
        "stacked_polished": topk_stacked_polished,
    }

    # ===========================================================================
    # B. Local Adam+MSE distillation baseline
    # Cache teacher block I/O once per block, then train student with Adam
    # ===========================================================================
    print("\n  --- Local Adam+MSE distillation (cached block I/O) ---", flush=True)

    distill_single_ppls_snapshot: dict[int, float] = {}
    distill_single_ppls_final: dict[int, float] = {}
    distill_loss_trajectories: dict[int, dict] = {}
    distill_convergence_steps: dict[int, int] = {}
    trained_students_snapshot: dict[int, CompressedMLP] = {}
    trained_students_final: dict[int, CompressedMLP] = {}

    for bid in BLOCKS:
        print(f"\n    Block {bid}:", flush=True)
        _set_seed(seed + bid)

        # Cache teacher I/O for this block (one full-model forward pass per sequence)
        print(f"      Caching teacher block {bid} I/O...", flush=True)
        ln_layer = teacher.model.transformer.h[bid].ln_2
        X_in, X_out = _cache_block_io(
            teacher.model, bid, corpus.probe_holdout, device, batch_size=BATCH_SIZE
        )
        print(f"      Cached {X_in.shape[0]} token vectors", flush=True)

        # --- Snapshot run (500 steps) ---
        _set_seed(seed + bid)
        snap_student = _make_student(device)
        snap_losses, snap_logged, snap_steps, snap_conv = _distill_block_local(
            snap_student, X_in, X_out,
            n_steps=DISTILL_STEPS_SNAPSHOT,
            lr=LR, device=device, seed=seed + bid,
            ln=ln_layer,
            log_at_steps=LOG_AT_STEPS,
            plateau_window=PLATEAU_WINDOW,
            plateau_threshold=PLATEAU_THRESHOLD,
        )
        print(f"      snapshot ({snap_steps} steps): loss {snap_losses[0]:.5f} -> {snap_losses[-1]:.5f}", flush=True)

        # Eval single-block PPL for snapshot
        snap_model = copy.deepcopy(teacher.model)
        _swap_mlp(snap_model, bid, snap_student)
        snap_model.eval()
        m_snap = _evaluate_model(teacher, snap_model, corpus, batch_size=BATCH_SIZE)
        distill_single_ppls_snapshot[bid] = float(m_snap["perplexity"])
        print(f"      snapshot PPL: {distill_single_ppls_snapshot[bid]:.3f}", flush=True)
        trained_students_snapshot[bid] = copy.deepcopy(snap_student).cpu()
        del snap_model

        # --- Converged run (up to 2000 steps) ---
        _set_seed(seed + bid)
        final_student = _make_student(device)
        final_losses, final_logged, final_steps, final_conv = _distill_block_local(
            final_student, X_in, X_out,
            n_steps=DISTILL_STEPS_CONVERGED,
            lr=LR, device=device, seed=seed + bid,
            ln=ln_layer,
            log_at_steps=LOG_AT_STEPS,
            plateau_window=PLATEAU_WINDOW,
            plateau_threshold=PLATEAU_THRESHOLD,
        )
        print(f"      converged ({final_steps} steps, early={final_conv}): loss {final_losses[0]:.5f} -> {final_losses[-1]:.5f}", flush=True)

        # Eval single-block PPL for converged
        final_model = copy.deepcopy(teacher.model)
        _swap_mlp(final_model, bid, final_student)
        final_model.eval()
        m_final = _evaluate_model(teacher, final_model, corpus, batch_size=BATCH_SIZE)
        distill_single_ppls_final[bid] = float(m_final["perplexity"])
        print(f"      converged PPL: {distill_single_ppls_final[bid]:.3f}", flush=True)

        distill_loss_trajectories[bid] = {str(k): v for k, v in final_logged.items()}
        distill_convergence_steps[bid] = final_steps
        trained_students_final[bid] = copy.deepcopy(final_student).cpu()
        del final_model

    # Stacked eval — snapshot students
    stacked_snap = copy.deepcopy(teacher.model)
    stacked_snap.eval()
    for bid in BLOCKS:
        _swap_mlp(stacked_snap, bid, trained_students_snapshot[bid].to(device))
    stacked_snap.eval()
    m_snap2 = _evaluate_model(teacher, stacked_snap, corpus, batch_size=BATCH_SIZE)
    distill_stacked_snapshot = float(m_snap2["perplexity"])
    print(f"\n    Distillation stacked PPL (step-{DISTILL_STEPS_SNAPSHOT}): {distill_stacked_snapshot:.3f}", flush=True)
    del stacked_snap

    # Stacked eval — converged students
    stacked_final = copy.deepcopy(teacher.model)
    stacked_final.eval()
    for bid in BLOCKS:
        _swap_mlp(stacked_final, bid, trained_students_final[bid].to(device))
    stacked_final.eval()
    m_final2 = _evaluate_model(teacher, stacked_final, corpus, batch_size=BATCH_SIZE)
    distill_stacked_final = float(m_final2["perplexity"])
    print(f"    Distillation stacked PPL (converged): {distill_stacked_final:.3f}", flush=True)
    del stacked_final

    seed_results["distillation"] = {
        "single_ppls_snapshot": distill_single_ppls_snapshot,
        "single_ppls_final": distill_single_ppls_final,
        "stacked_snapshot": distill_stacked_snapshot,
        "stacked_final": distill_stacked_final,
        "loss_trajectories": distill_loss_trajectories,
        "convergence_steps": {str(k): v for k, v in distill_convergence_steps.items()},
    }

    all_seed_results[seed] = seed_results


# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATE RESULTS\n{'='*60}", flush=True)

topk_ppls = [all_seed_results[s]["topk"]["stacked_polished"] for s in SEEDS]
distill_ppls_final = [all_seed_results[s]["distillation"]["stacked_final"] for s in SEEDS]
distill_ppls_snap = [all_seed_results[s]["distillation"]["stacked_snapshot"] for s in SEEDS]

mean_topk = sum(topk_ppls) / len(topk_ppls)
std_topk = (sum((p - mean_topk)**2 for p in topk_ppls) / len(topk_ppls)) ** 0.5
mean_distill = sum(distill_ppls_final) / len(distill_ppls_final)
std_distill = (sum((p - mean_distill)**2 for p in distill_ppls_final) / len(distill_ppls_final)) ** 0.5
mean_distill_snap = sum(distill_ppls_snap) / len(distill_ppls_snap)

print(f"\n  topk+lstsq+polish stacked:          {mean_topk:.2f} ± {std_topk:.2f}  per-seed: {[f'{p:.1f}' for p in topk_ppls]}", flush=True)
print(f"  distill (converged) stacked:        {mean_distill:.2f} ± {std_distill:.2f}  per-seed: {[f'{p:.1f}' for p in distill_ppls_final]}", flush=True)
print(f"  distill (step-{DISTILL_STEPS_SNAPSHOT}) stacked:       {mean_distill_snap:.2f}  per-seed: {[f'{p:.1f}' for p in distill_ppls_snap]}", flush=True)

delta = mean_distill - mean_topk
print(f"\n  delta (distill_converged - topk+polish): {delta:+.2f}", flush=True)

# Single-block comparison
print(f"\n  Single-block PPLs (mean over seeds):", flush=True)
print(f"  {'Block':<8} {'topk(lstsq)':<18} {'distill(snap)':<18} {'distill(conv)':<18}", flush=True)
print(f"  {'-'*62}", flush=True)
for bid in BLOCKS:
    t_ppls = [all_seed_results[s]["topk"]["single_ppls"][bid] for s in SEEDS]
    d_snap = [all_seed_results[s]["distillation"]["single_ppls_snapshot"][bid] for s in SEEDS]
    d_final = [all_seed_results[s]["distillation"]["single_ppls_final"][bid] for s in SEEDS]
    print(f"  {bid:<8} {sum(t_ppls)/len(t_ppls):<18.2f} {sum(d_snap)/len(d_snap):<18.2f} {sum(d_final)/len(d_final):<18.2f}", flush=True)

# ---------------------------------------------------------------------------
# Verdict (based on CONVERGED distillation vs topk+polish)
# ---------------------------------------------------------------------------
if delta > 3.0:
    verdict = "CONFIRMED"
    verdict_note = (f"distill(converged) stacked PPL ({mean_distill:.2f}) > topk+polish ({mean_topk:.2f}) "
                    f"by {delta:.1f} PPL — geometric compression justified")
elif delta <= 0.0:
    verdict = "REFUTED"
    verdict_note = (f"distill(converged) stacked PPL ({mean_distill:.2f}) <= topk+polish ({mean_topk:.2f}) "
                    f"— distillation matches or beats geometric approach")
else:
    verdict = "INCONCLUSIVE"
    verdict_note = (f"distill(converged) ({mean_distill:.2f}) vs topk+polish ({mean_topk:.2f}) "
                    f"delta={delta:.1f} PPL within ±3 threshold")

print(f"\n  VERDICT: {verdict}", flush=True)
print(f"  {verdict_note}", flush=True)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = {
    "cycle": 183,
    "item": 193,
    "verdict": verdict,
    "verdict_note": verdict_note,
    "method_note": "local Adam+MSE distillation (cached block I/O) vs topk+lstsq+polish",
    "mean_topk_stacked_polished": mean_topk,
    "std_topk_stacked_polished": std_topk,
    "mean_distillation_stacked_converged": mean_distill,
    "std_distillation_stacked_converged": std_distill,
    "mean_distillation_stacked_snapshot": mean_distill_snap,
    "delta_distill_converged_minus_topk": delta,
    "per_seed": {str(s): all_seed_results[s] for s in SEEDS},
    "config": {
        "blocks": BLOCKS,
        "student_hidden": STUDENT_HIDDEN,
        "polish_steps": POLISH_STEPS,
        "distill_steps_snapshot": DISTILL_STEPS_SNAPSHOT,
        "distill_steps_converged": DISTILL_STEPS_CONVERGED,
        "plateau_window": PLATEAU_WINDOW,
        "plateau_threshold": PLATEAU_THRESHOLD,
        "lr": LR,
        "seeds": SEEDS,
        "eval_seqs": EVAL_SEQS,
        "seq_len": SEQ_LEN,
        "probe_holdout_seqs": PROBE_HOLDOUT_SEQS,
    },
}

out_path = RESULTS_DIR / "cycle_183_distillation_baseline.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to {out_path}", flush=True)
