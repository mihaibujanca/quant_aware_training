"""Cycle 178 / Item 188: Negative-regime bias absorption + positive-only compression.

Hypothesis: absorb constant negative GELU bias per block, then compress only
the positive-regime channels (~9-13% active). This reduces the effective
compression problem from 3072 channels to ~270-400.

Three conditions (4 blocks each: [1,4,7,11]):
  A) Standard topk+lstsq (GELU, 512 channels) — control
  B) Bias-absorbed + positive-only topk+lstsq (ReLU on positive channels)
  C) Bias-absorbed only (no compression, just GELU->ReLU+b_leak in 4 blocks)

Verdict:
  CONFIRMED    if positive-only PPL <= 68 and < standard topk - 2
  REFUTED      if condition C > baseline*1.05, or positive-only > standard topk
  INCONCLUSIVE if positive-only ~ standard topk (within +/-2)
"""
from __future__ import annotations

import copy
import json
import math
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
    CompressedMLP,
    CompressionBudgetSpec,
    FfnGeometryAtlas,
    FfnLayerGeometry,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _apply_activation,
    _evaluate_model,
    _fit_output_projection,
    _select_hidden_units,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    load_nanochat_teacher,
    probe_ffn_geometry,
)

P = lambda *a, **kw: print(*a, **kw, flush=True)

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
GPT2_MODEL_DIM = 768
BLOCKS = [1, 4, 7, 11]
TOPK_HIDDEN = 512  # standard topk keeps 512 channels

N_TOKENS_BLEAK = 4096  # tokens for b_leak calibration
POSITIVE_THRESHOLD = 0.01  # channel must be positive in >= 1% of tokens

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# b_leak computation (from cycle 166, adapted)
# ---------------------------------------------------------------------------
def compute_b_leak_and_positive_mask(model, corpus_seqs, device, block_ids,
                                      n_tokens=N_TOKENS_BLEAK):
    """Compute b_leak and positive-regime mask for specified blocks."""
    accum_neg = {i: None for i in block_ids}
    accum_pos_count = {i: None for i in block_ids}
    counts = {i: 0 for i in block_ids}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            z = inp[0]  # pre-activation input to GELU
            activated = out  # gelu(z)
            z_flat = z.reshape(-1, z.shape[-1])
            act_flat = activated.reshape(-1, activated.shape[-1])
            mask_neg = (z_flat <= 0)
            mask_pos = (z_flat > 0)
            y_neg = torch.where(mask_neg, act_flat, torch.zeros_like(act_flat))
            neg_sum = y_neg.sum(dim=0).cpu().float()
            pos_count = mask_pos.sum(dim=0).cpu().float()
            n = z_flat.shape[0]
            if accum_neg[idx] is None:
                accum_neg[idx] = neg_sum
                accum_pos_count[idx] = pos_count
            else:
                accum_neg[idx] += neg_sum
                accum_pos_count[idx] += pos_count
            counts[idx] += n
        return hook_fn

    for i in block_ids:
        h = model.transformer.h[i].mlp.act.register_forward_hook(make_hook(i))
        hooks.append(h)

    n_seqs_needed = max(1, n_tokens // SEQ_LEN + 1)
    model.eval()
    with torch.no_grad():
        for i in range(0, min(n_seqs_needed, len(corpus_seqs)), BATCH_SIZE):
            batch = corpus_seqs[i:i + BATCH_SIZE, :-1].to(device)
            model(batch)

    for h in hooks:
        h.remove()

    b_leak = {}
    frac_positive = {}
    for i in block_ids:
        b_leak[i] = (accum_neg[i] / counts[i]).to(device)
        frac_positive[i] = (accum_pos_count[i] / counts[i])  # (3072,) on CPU
        n_pos = (frac_positive[i] > POSITIVE_THRESHOLD).sum().item()
        P(f"  Block {i}: b_leak norm={b_leak[i].norm().item():.4f}, "
          f"n_positive_channels={n_pos}/3072 ({100*n_pos/3072:.1f}%), "
          f"tokens={counts[i]}")

    return b_leak, frac_positive


# ---------------------------------------------------------------------------
# Bias-absorbed MLP (ReLU + constant bias folded into output)
# ---------------------------------------------------------------------------
class BiasAbsorbedMLP(nn.Module):
    """Replace GELU with ReLU; fold b_leak contribution into a constant output bias.

    output = c_proj(ReLU(c_fc(x))) + c_proj(b_leak)
           = W_out @ ReLU(W_in @ x + b_in) + b_out + W_out @ b_leak
    """

    def __init__(self, orig_mlp, b_leak_vec):
        super().__init__()
        self.c_fc = orig_mlp.c_fc
        # Compute the constant output contribution of b_leak
        W_out = _standard_weight(orig_mlp.c_proj)  # (768, 3072)
        b_out = _standard_bias(orig_mlp.c_proj)
        output_bias = W_out @ b_leak_vec  # (768,)
        if b_out is not None:
            output_bias = output_bias + b_out
        self.c_proj = nn.Linear(W_out.shape[1], W_out.shape[0], bias=True)
        with torch.no_grad():
            self.c_proj.weight.copy_(W_out)
            self.c_proj.bias.copy_(output_bias)

    def forward(self, x):
        z = self.c_fc(x)
        h = F.relu(z)
        return self.c_proj(h)


# ---------------------------------------------------------------------------
# Positive-only compressed MLP
# ---------------------------------------------------------------------------
class PositiveOnlyCompressedMLP(nn.Module):
    """Compressed MLP operating only on positive-regime channels with ReLU.

    The b_leak contribution is folded into a constant output bias.
    c_fc maps input to selected positive channels only.
    c_proj maps from selected channels to output dim (fit by lstsq).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, *, c_fc_bias, c_proj_bias):
        super().__init__()
        self.c_fc = nn.Linear(input_dim, hidden_dim, bias=c_fc_bias)
        self.c_proj = nn.Linear(hidden_dim, output_dim, bias=c_proj_bias)

    def forward(self, x):
        z = self.c_fc(x)
        h = F.relu(z)
        return self.c_proj(h)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_owt_seqs(tokenizer, n_seqs, seq_len, seed):
    """Load OWT evaluation sequences (for b_leak calibration)."""
    with open(OWT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    rng = torch.Generator()
    rng.manual_seed(seed)
    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    max_start = len(tokens) - seq_len - 1
    indices = torch.randperm(max_start, generator=rng)[:n_seqs]
    seqs = torch.stack([tokens[i:i + seq_len + 1] for i in sorted(indices.tolist())])
    return seqs


def _find_mlp(model, block_id):
    return model.transformer.h[block_id].mlp


def _joint_polish(student_model, block_ids, polish_tokens, n_steps, lr, device, seed):
    """Fine-tune c_proj weights of compressed blocks on LM loss."""
    if n_steps == 0:
        return []

    student_model.requires_grad_(False)
    params = []
    for bid in block_ids:
        mlp = _find_mlp(student_model, bid)
        if hasattr(mlp, "c_proj"):
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
        logits = raw.logits if hasattr(raw, "logits") else raw[0]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    student_model.eval()
    student_model.requires_grad_(False)
    return losses


# ---------------------------------------------------------------------------
# Build conditions
# ---------------------------------------------------------------------------
def build_condition_A(teacher, atlas, block_ids):
    """Standard topk+lstsq with GELU (control). Returns (model, per_block_stats)."""
    model = copy.deepcopy(teacher.model)
    model.eval()
    stats = {}

    budget = CompressionBudgetSpec(
        hidden_fraction=TOPK_HIDDEN / GPT2_FFN_HIDDEN,
        min_hidden=32,
        selection_mode="topk",
        per_layer_hidden={bid: TOPK_HIDDEN for bid in block_ids},
    )

    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
            continue
        bid = layer.layer_id
        selected = _select_hidden_units(layer, budget, TOPK_HIDDEN)
        source_w_in = _standard_weight(layer.handle.c_fc).float()
        source_b_in = _standard_bias(layer.handle.c_fc)
        use_output_bias = _standard_bias(layer.handle.c_proj) is not None

        hidden = layer.hidden_sample[:, selected].float()
        outputs = layer.output_sample.float()
        w_out, b_out = _fit_output_projection(hidden, outputs, use_bias=use_output_bias)

        new_mlp = CompressedMLP(
            input_dim=source_w_in.shape[1],
            hidden_dim=TOPK_HIDDEN,
            output_dim=w_out.shape[0],
            activation_kind=layer.activation_kind,
            c_fc_bias=source_b_in is not None,
            c_proj_bias=b_out is not None,
        )
        with torch.no_grad():
            new_mlp.c_fc.weight.copy_(source_w_in[selected])
            if source_b_in is not None:
                new_mlp.c_fc.bias.copy_(source_b_in[selected])
            new_mlp.c_proj.weight.copy_(w_out)
            if b_out is not None:
                new_mlp.c_proj.bias.copy_(b_out)

        device = next(model.transformer.h[bid].mlp.parameters()).device
        new_mlp.to(device)
        model.transformer.h[bid].mlp = new_mlp

        # lstsq residual
        with torch.no_grad():
            pred = hidden @ w_out.t()
            if b_out is not None:
                pred = pred + b_out
            mse = F.mse_loss(pred, outputs).item()

        n_params = TOPK_HIDDEN * GPT2_MODEL_DIM + TOPK_HIDDEN + GPT2_MODEL_DIM * TOPK_HIDDEN + GPT2_MODEL_DIM
        stats[bid] = {
            "n_channels": TOPK_HIDDEN,
            "lstsq_mse": round(mse, 6),
            "params": n_params,
        }
        P(f"    Block {bid}: standard topk {TOPK_HIDDEN} channels, "
          f"lstsq_mse={mse:.6f}, params={n_params:,}")

    return model, stats


def build_condition_B(teacher, atlas, block_ids, b_leak, frac_positive):
    """Positive-only topk+lstsq with ReLU + absorbed bias. Returns (model, stats)."""
    model = copy.deepcopy(teacher.model)
    model.eval()
    stats = {}

    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
            continue
        bid = layer.layer_id
        source_w_in = _standard_weight(layer.handle.c_fc).float()
        source_b_in = _standard_bias(layer.handle.c_fc)
        source_w_out = _standard_weight(layer.handle.c_proj).float()
        source_b_out = _standard_bias(layer.handle.c_proj)

        # Identify positive-regime channels
        pos_mask = frac_positive[bid] > POSITIVE_THRESHOLD
        pos_indices = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
        n_pos = pos_indices.shape[0]

        # Select topk within positive channels
        k = min(TOPK_HIDDEN, n_pos)
        # Use channel_scores restricted to positive channels
        pos_scores = layer.channel_scores[pos_indices]
        topk_within = torch.topk(pos_scores, k=k, largest=True).indices
        selected = pos_indices[topk_within].sort().values

        # Collect hidden activations with ReLU (not GELU)
        # Recompute: hidden_sample used GELU, we need ReLU
        # hidden_sample = GELU(z), we need ReLU(z)
        # preact = input_sample @ W_in.T + b_in
        preact = layer.input_sample.float() @ source_w_in.t()
        if source_b_in is not None:
            preact = preact + source_b_in
        hidden_relu = F.relu(preact)  # (N, 3072)
        hidden_relu_selected = hidden_relu[:, selected]  # (N, k)

        # Target: original output = W_out @ GELU(z) + b_out
        # Our model: W_new @ ReLU(z_selected) + bias_absorbed
        # bias_absorbed includes W_out @ b_leak + b_out
        # So lstsq target = original_output - W_out @ b_leak (- b_out handled by lstsq bias)
        # Actually: target for lstsq is the output minus the constant b_leak contribution
        output_bias_from_bleak = source_w_out @ b_leak[bid]  # (768,)
        targets = layer.output_sample.float()  # (N, 768) — original GELU output
        # The compressed model will output: W_new @ ReLU(z_selected) + b_new
        # We want this to equal: targets
        # But targets = W_out @ GELU(z) + b_out
        #             = W_out @ (ReLU(z) + neg_contribution) + b_out
        #             ~ W_out @ ReLU(z) + W_out @ b_leak + b_out  (if neg ~ b_leak)
        # So: W_new @ ReLU(z_selected) + b_new ~ W_out @ ReLU(z) + W_out @ b_leak + b_out
        # Rearranging: W_new @ ReLU(z_selected) ~ targets - W_out @ b_leak - b_out + b_out
        #            = targets - output_bias_from_bleak (if lstsq includes bias)
        # Wait — targets already includes b_out. Let lstsq handle the bias.
        # lstsq target = targets (original output)
        # The lstsq will fit: hidden_relu_selected @ W.T + bias = targets
        # The bias will absorb the constant b_leak contribution automatically.

        w_out, b_out_fitted = _fit_output_projection(
            hidden_relu_selected, targets, use_bias=True
        )

        new_mlp = PositiveOnlyCompressedMLP(
            input_dim=source_w_in.shape[1],
            hidden_dim=k,
            output_dim=w_out.shape[0],
            c_fc_bias=source_b_in is not None,
            c_proj_bias=True,
        )
        with torch.no_grad():
            new_mlp.c_fc.weight.copy_(source_w_in[selected])
            if source_b_in is not None:
                new_mlp.c_fc.bias.copy_(source_b_in[selected])
            new_mlp.c_proj.weight.copy_(w_out)
            new_mlp.c_proj.bias.copy_(b_out_fitted)

        device = next(model.transformer.h[bid].mlp.parameters()).device
        new_mlp.to(device)
        model.transformer.h[bid].mlp = new_mlp

        # lstsq residual
        with torch.no_grad():
            pred = hidden_relu_selected @ w_out.t() + b_out_fitted
            mse = F.mse_loss(pred, targets).item()

        n_params = k * GPT2_MODEL_DIM + k + GPT2_MODEL_DIM * k + GPT2_MODEL_DIM
        stats[bid] = {
            "n_positive_channels": n_pos,
            "n_selected": k,
            "lstsq_mse": round(mse, 6),
            "params": n_params,
            "frac_positive_mean": round(frac_positive[bid][pos_indices].mean().item(), 4),
        }
        P(f"    Block {bid}: positive={n_pos}, selected={k}, "
          f"lstsq_mse={mse:.6f}, params={n_params:,}")

    return model, stats


def build_condition_C(teacher, block_ids, b_leak):
    """Bias absorption only (GELU->ReLU+b_leak) in specified blocks, no compression."""
    model = copy.deepcopy(teacher.model)
    model.eval()

    for bid in block_ids:
        orig_mlp = model.transformer.h[bid].mlp
        new_mlp = BiasAbsorbedMLP(orig_mlp, b_leak[bid])
        device = next(orig_mlp.parameters()).device
        new_mlp.to(device)
        model.transformer.h[bid].mlp = new_mlp

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
P("Loading GPT-2 teacher...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()
device = teacher.device
P(f"  Teacher on device={device}")

# Load calibration sequences for b_leak
P("\nLoading calibration sequences...")
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
calib_seqs = _load_owt_seqs(tokenizer, n_seqs=50, seq_len=SEQ_LEN, seed=42)

P("\nComputing b_leak and positive-regime masks for blocks", BLOCKS, "...")
b_leak, frac_positive = compute_b_leak_and_positive_mask(
    teacher.model, calib_seqs, device, BLOCKS
)

# ---------------------------------------------------------------------------
# Per-seed loop
# ---------------------------------------------------------------------------
all_seed_results = {}

for seed in SEEDS:
    P(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
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
    P(f"  Baseline PPL: {baseline_ppl:.3f}")

    if not (25.0 <= baseline_ppl <= 65.0):
        P(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [25, 65]")
        sys.exit(1)

    # Probe FFN geometry
    P(f"  Probing FFN geometry for blocks {BLOCKS}...")
    _set_seed(seed)
    atlas = probe_ffn_geometry(
        teacher, corpus,
        layer_ids=tuple(BLOCKS),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )

    # --- Condition C: Bias absorption only (4 blocks) ---
    P("\n  --- Condition C: Bias absorption only (4 blocks) ---")
    model_C = build_condition_C(teacher, BLOCKS, b_leak)
    res_C = _evaluate_model(teacher, model_C, corpus, batch_size=BATCH_SIZE)
    ppl_C = float(res_C["perplexity"])
    P(f"  Condition C PPL (absorbed only): {ppl_C:.3f} "
      f"(delta: {100*(ppl_C-baseline_ppl)/baseline_ppl:.1f}%)")
    del model_C

    # --- Condition A: Standard topk+lstsq (control) ---
    P("\n  --- Condition A: Standard topk+lstsq ---")
    model_A, stats_A = build_condition_A(teacher, atlas, BLOCKS)
    res_A_pre = _evaluate_model(teacher, model_A, corpus, batch_size=BATCH_SIZE)
    ppl_A_pre = float(res_A_pre["perplexity"])
    P(f"  Condition A PPL (pre-polish): {ppl_A_pre:.3f}")

    P(f"  Polishing condition A ({POLISH_STEPS} steps)...")
    polish_losses_A = _joint_polish(
        model_A, BLOCKS, corpus.eval_val, POLISH_STEPS, POLISH_LR, device, seed
    )
    res_A = _evaluate_model(teacher, model_A, corpus, batch_size=BATCH_SIZE)
    ppl_A = float(res_A["perplexity"])
    P(f"  Condition A PPL (post-polish): {ppl_A:.3f}")
    del model_A

    # --- Condition B: Positive-only compression ---
    P("\n  --- Condition B: Positive-only topk+lstsq ---")
    model_B, stats_B = build_condition_B(teacher, atlas, BLOCKS, b_leak, frac_positive)
    res_B_pre = _evaluate_model(teacher, model_B, corpus, batch_size=BATCH_SIZE)
    ppl_B_pre = float(res_B_pre["perplexity"])
    P(f"  Condition B PPL (pre-polish): {ppl_B_pre:.3f}")

    P(f"  Polishing condition B ({POLISH_STEPS} steps)...")
    polish_losses_B = _joint_polish(
        model_B, BLOCKS, corpus.eval_val, POLISH_STEPS, POLISH_LR, device, seed
    )
    res_B = _evaluate_model(teacher, model_B, corpus, batch_size=BATCH_SIZE)
    ppl_B = float(res_B["perplexity"])
    P(f"  Condition B PPL (post-polish): {ppl_B:.3f}")
    del model_B

    all_seed_results[seed] = {
        "baseline_ppl": baseline_ppl,
        "ppl_C_absorbed": ppl_C,
        "ppl_A_pre_polish": ppl_A_pre,
        "ppl_A_post_polish": ppl_A,
        "ppl_B_pre_polish": ppl_B_pre,
        "ppl_B_post_polish": ppl_B,
        "stats_A": {str(k): v for k, v in stats_A.items()},
        "stats_B": {str(k): v for k, v in stats_B.items()},
        "delta_C_pct": round(100 * (ppl_C - baseline_ppl) / baseline_ppl, 2),
        "delta_A_pct": round(100 * (ppl_A - baseline_ppl) / baseline_ppl, 2),
        "delta_B_pct": round(100 * (ppl_B - baseline_ppl) / baseline_ppl, 2),
    }

    P(f"\n  Summary seed {seed}:")
    P(f"    Baseline:        {baseline_ppl:.3f}")
    P(f"    C (absorbed):    {ppl_C:.3f} ({all_seed_results[seed]['delta_C_pct']:+.1f}%)")
    P(f"    A (std topk):    {ppl_A:.3f} ({all_seed_results[seed]['delta_A_pct']:+.1f}%)")
    P(f"    B (pos-only):    {ppl_B:.3f} ({all_seed_results[seed]['delta_B_pct']:+.1f}%)")

# ---------------------------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------------------------
P(f"\n{'='*60}\nAGGREGATE ACROSS {len(SEEDS)} SEEDS\n{'='*60}")

mean_base = sum(r["baseline_ppl"] for r in all_seed_results.values()) / len(SEEDS)
mean_C = sum(r["ppl_C_absorbed"] for r in all_seed_results.values()) / len(SEEDS)
mean_A = sum(r["ppl_A_post_polish"] for r in all_seed_results.values()) / len(SEEDS)
mean_B = sum(r["ppl_B_post_polish"] for r in all_seed_results.values()) / len(SEEDS)

P(f"  Mean baseline:     {mean_base:.3f}")
P(f"  Mean C (absorbed): {mean_C:.3f} (delta: {100*(mean_C-mean_base)/mean_base:+.1f}%)")
P(f"  Mean A (std topk): {mean_A:.3f} (delta: {100*(mean_A-mean_base)/mean_base:+.1f}%)")
P(f"  Mean B (pos-only): {mean_B:.3f} (delta: {100*(mean_B-mean_base)/mean_base:+.1f}%)")

# Verdict
absorption_ok = mean_C < mean_base * 1.05
B_better = mean_B < mean_A - 2
B_confirmed = mean_B <= 68 and B_better
B_refuted = (not absorption_ok) or (mean_B > mean_A)
if B_confirmed:
    verdict = "CONFIRMED"
elif B_refuted:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

P(f"\n  VERDICT: {verdict}")
P(f"    absorption_ok={absorption_ok} (C < base*1.05: {mean_C:.1f} < {mean_base*1.05:.1f})")
P(f"    B_better={B_better} (B < A-2: {mean_B:.1f} < {mean_A-2:.1f})")
P(f"    B_confirmed={B_confirmed} (B<=68 and B<A-2)")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
out = {
    "cycle": 178,
    "item": 188,
    "seeds": SEEDS,
    "blocks": BLOCKS,
    "topk_hidden": TOPK_HIDDEN,
    "positive_threshold": POSITIVE_THRESHOLD,
    "polish_steps": POLISH_STEPS,
    "verdict": verdict,
    "mean_baseline_ppl": round(mean_base, 3),
    "mean_ppl_C_absorbed": round(mean_C, 3),
    "mean_ppl_A_std_topk": round(mean_A, 3),
    "mean_ppl_B_pos_only": round(mean_B, 3),
    "per_seed": {str(k): v for k, v in all_seed_results.items()},
}

out_json = RESULTS_DIR / "cycle_178_neg_absorption.json"
with open(out_json, "w") as f:
    json.dump(out, f, indent=2)
P(f"\nResults saved to {out_json}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: PPL by condition across seeds
ax = axes[0]
seeds_list = list(all_seed_results.keys())
x = range(len(seeds_list))
width = 0.2
ppls_base = [all_seed_results[s]["baseline_ppl"] for s in seeds_list]
ppls_C = [all_seed_results[s]["ppl_C_absorbed"] for s in seeds_list]
ppls_A = [all_seed_results[s]["ppl_A_post_polish"] for s in seeds_list]
ppls_B = [all_seed_results[s]["ppl_B_post_polish"] for s in seeds_list]

bars1 = ax.bar([i - 1.5*width for i in x], ppls_base, width, label="Float baseline", color="grey")
bars2 = ax.bar([i - 0.5*width for i in x], ppls_C, width, label="C: absorbed only", color="skyblue")
bars3 = ax.bar([i + 0.5*width for i in x], ppls_A, width, label="A: std topk+lstsq", color="salmon")
bars4 = ax.bar([i + 1.5*width for i in x], ppls_B, width, label="B: pos-only topk", color="mediumseagreen")

ax.set_xticks(list(x))
ax.set_xticklabels([f"seed {s}" for s in seeds_list])
ax.set_ylabel("PPL")
ax.set_title("PPL by condition")
ax.legend(fontsize=8)

# Panel 2: Per-block positive channel counts (seed 42)
ax = axes[1]
first_seed = SEEDS[0]
stats_B_first = all_seed_results[first_seed]["stats_B"]
block_labels = [f"Block {b}" for b in BLOCKS]
n_pos_list = [stats_B_first[str(b)]["n_positive_channels"] for b in BLOCKS]
n_sel_list = [stats_B_first[str(b)]["n_selected"] for b in BLOCKS]

x2 = range(len(BLOCKS))
ax.bar([i - 0.2 for i in x2], n_pos_list, 0.35, label="Positive channels", color="steelblue")
ax.bar([i + 0.2 for i in x2], n_sel_list, 0.35, label="Selected (topk)", color="darkorange")
ax.axhline(y=TOPK_HIDDEN, color="red", linestyle="--", alpha=0.5, label=f"Std topk={TOPK_HIDDEN}")
ax.set_xticks(list(x2))
ax.set_xticklabels(block_labels)
ax.set_ylabel("Channels")
ax.set_title("Positive-regime channels per block")
ax.legend(fontsize=8)

plt.tight_layout()
out_png = RESULTS_DIR / "cycle_178_neg_absorption.png"
fig.savefig(out_png, dpi=150)
P(f"Plot saved to {out_png}")
plt.close(fig)

P("\n=== DONE ===")
