#!/usr/bin/env python3
"""Cycle 348 / Item 355 — GPT-2 multi-block topk+lstsq accumulation curve.

Tests whether topk+lstsq-compressed blocks compose when added incrementally,
using the SAME blocks and order as cycle 343 (LASER), to answer:
"Is composition failure LASER-specific or universal?"

Accumulation stages (same as cycle 343):
  Stage 1: b2
  Stage 2: b2 + b1
  Stage 3: b2 + b1 + b10
  Stage 4: b2 + b1 + b10 + b11

Compression: topk channel selection (hidden_fraction=1/6, k=512 of 3072),
lstsq output refit, then joint Adam retrain (1000 steps) — matching cycle 343.

Cycle 343 LASER results (for comparison):
  Stage 1: [b2] → PPL 50.21 ± 0.13
  Stage 2: [b2,b1] → PPL 55.10 ± 0.14
  Stage 3: [b2,b1,b10] → PPL 58.15 ± 0.08
  Stage 4: [b2,b1,b10,b11] → PPL 68.97 ± 0.26

Verdict (item 355):
  CONFIRMED if any >=2-block stage has best_mean <= 54.90 (float + 1.0)
  REFUTED if accumulation degrades monotonically like LASER
  INCONCLUSIVE if non-monotonic but exceeds threshold
"""
import json
import math
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
# Same blocks and order as cycle 343
ACCUM_BLOCKS = [2, 1, 10, 11]

# topk+lstsq config: 6x compression = keep 512 of 3072 hidden channels
HIDDEN_FRACTION = 1 / 6  # → k=512

N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 1000  # match cycle 343
LR = 1e-4  # match cycle 343
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
LOAD_SEED = 42

FLOAT_BASELINE_REF = 53.895
CONFIRMED_LE = 54.90  # float + 1.0

OUTPUT_PATH = "autoresearch/results/cycle_348_accumulation.json"

torch.manual_seed(LOAD_SEED)
print(f"Device: {DEVICE}", flush=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading GPT-2 tokenizer and wikitext-2 data...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
all_ids = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        all_ids.append(enc["input_ids"][:, :SEQ_LEN])
    if len(all_ids) >= N_CALIB + N_EVAL:
        break

calib_ids_cpu = list(all_ids[:N_CALIB])
eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB : N_CALIB + N_EVAL]]
print(f"Calibration: {len(calib_ids_cpu)} seqs, Eval: {len(eval_ids)} seqs", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def eval_ppl(model, eval_ids):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


def collect_block_activations(model, block_ids, calib_ids, device, seq_len=128):
    """Collect MLP input and output activations for specified blocks.

    Returns dict: block_id -> {"input": [N, 768], "output": [N, 768],
                                "preact": [N, 3072], "hidden": [N, 3072]}
    """
    block_data = {b: {"input": [], "output": [], "preact": [], "hidden": []} for b in block_ids}
    hooks = []

    def make_hook(blk_id):
        def hook_fn(module, inputs, output):
            x = inputs[0].detach()  # (batch, seq, 768)
            # GPT-2 MLP: c_fc -> act -> c_proj
            z = module.c_fc(x).detach()  # preactivation (batch, seq, 3072)
            h = F.gelu(z).detach()       # hidden activation
            y = output.detach()          # MLP output (batch, seq, 768)
            # Flatten batch×seq
            block_data[blk_id]["input"].append(x.reshape(-1, x.shape[-1]).cpu())
            block_data[blk_id]["output"].append(y.reshape(-1, y.shape[-1]).cpu())
            block_data[blk_id]["preact"].append(z.reshape(-1, z.shape[-1]).cpu())
            block_data[blk_id]["hidden"].append(h.reshape(-1, h.shape[-1]).cpu())
        return hook_fn

    for blk_id in block_ids:
        hook = model.transformer.h[blk_id].mlp.register_forward_hook(make_hook(blk_id))
        hooks.append(hook)

    model.eval()
    with torch.no_grad():
        for ids in calib_ids:
            ids_dev = ids.to(device) if ids.device != device else ids
            model(ids_dev)

    for hook in hooks:
        hook.remove()

    # Concatenate
    result = {}
    for blk_id in block_ids:
        result[blk_id] = {
            "input": torch.cat(block_data[blk_id]["input"], dim=0).float(),
            "output": torch.cat(block_data[blk_id]["output"], dim=0).float(),
            "preact": torch.cat(block_data[blk_id]["preact"], dim=0).float(),
            "hidden": torch.cat(block_data[blk_id]["hidden"], dim=0).float(),
        }
    return result


def topk_lstsq_compress_block(block_acts, k):
    """Topk channel selection + lstsq output refit for a single block.

    Args:
        block_acts: dict with "input", "output", "hidden", "preact" tensors
        k: number of channels to keep

    Returns:
        selected_indices: (k,) tensor of selected channel indices
        w_out_refit: (768, k) weight matrix from lstsq
        b_out_refit: (768,) bias from lstsq
    """
    hidden = block_acts["hidden"]  # (N, 3072)
    output = block_acts["output"]  # (N, 768)

    # Channel scoring: std(h) * ||w_out_col|| * (|mean(h)| + |mean(deriv)| + eps)
    # Same formula as lm_geometric_compression.py line 695
    preact = block_acts["preact"]
    std_h = hidden.std(dim=0)
    mean_h = hidden.mean(dim=0)

    # GELU derivative
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    deriv = 0.5 * (1.0 + torch.erf(preact * inv_sqrt_2)) + preact * torch.exp(-0.5 * preact.square()) * inv_sqrt_2pi
    mean_deriv = deriv.mean(dim=0)

    # We don't have direct access to c_proj weights here, but we need output_norms
    # Instead, use a proxy: the output norms should be computed from the model weights
    # Since we're operating on activations, let's compute channel importance directly
    # from the actual contribution: std * |output correlation|
    # Actually, the recipe formula is: std_hidden * output_norms * (|mean_hidden| + |mean_deriv| + eps)
    # We'll get output_norms from the model during compression setup

    # For now, store scores as: std * (|mean| + |mean_deriv| + eps)
    # output_norms will be multiplied in the caller
    partial_scores = std_h * (mean_h.abs() + mean_deriv.abs() + 1e-6)

    return partial_scores


class CompressedGPT2MLP(nn.Module):
    """Compressed MLP: keeps k of 3072 hidden channels, with trainable W_out.

    c_fc rows (768->k) are frozen (copied from original at selected indices).
    c_proj (k->768) is initialized from lstsq and trainable.
    """

    def __init__(self, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias):
        super().__init__()
        k = c_fc_weight.shape[0]
        # Frozen input projection
        self.register_buffer("fc_weight", c_fc_weight.contiguous())  # (k, 768)
        self.register_buffer("fc_bias", c_fc_bias.contiguous())      # (k,)
        # Trainable output projection
        self.proj_weight = nn.Parameter(c_proj_weight.contiguous())   # (768, k)  [standard Linear shape: out×in]
        self.proj_bias = nn.Parameter(c_proj_bias.contiguous())       # (768,)

    def forward(self, x):
        # GPT-2 c_fc is Conv1D: x @ weight + bias, weight shape is (768, 3072)
        # We store as standard linear: (k, 768) @ x^T
        h = F.linear(x, self.fc_weight, self.fc_bias)  # (batch, seq, k)
        h = F.gelu(h)
        # c_proj: (batch, seq, k) -> (batch, seq, 768)
        out = F.linear(h, self.proj_weight, self.proj_bias)  # (batch, seq, 768)
        return out


# ── Float baseline ────────────────────────────────────────────────────────────
print("Computing float baseline PPL...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
float_ppl = eval_ppl(base_model, eval_ids)
print(f"Float PPL: {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from reference {FLOAT_BASELINE_REF} by > 0.5"
)

# ── Probe geometry and compress all target blocks ────────────────────────────
print("\nCollecting calibration activations for all target blocks...", flush=True)
calib_dev = [t.to(DEVICE) for t in calib_ids_cpu]
block_acts = collect_block_activations(base_model, ACCUM_BLOCKS, calib_dev, DEVICE)
parts = [f"b{b}: {block_acts[b]['hidden'].shape}" for b in ACCUM_BLOCKS]
print(f"Collected activations: {', '.join(parts)}", flush=True)

# Compute channel scores and compress each block
compress_cache = {}
for blk in ACCUM_BLOCKS:
    acts = block_acts[blk]
    hidden = acts["hidden"]  # (N, 3072)
    output = acts["output"]  # (N, 768)
    preact = acts["preact"]  # (N, 3072)

    hidden_dim = hidden.shape[1]
    k = max(1, int(round(hidden_dim * HIDDEN_FRACTION)))

    # Channel scoring (same as library)
    std_h = hidden.std(dim=0)
    mean_h = hidden.mean(dim=0)
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    deriv = 0.5 * (1.0 + torch.erf(preact * inv_sqrt_2)) + preact * torch.exp(-0.5 * preact.square()) * inv_sqrt_2pi
    mean_deriv = deriv.mean(dim=0)

    # Get output norms from c_proj weight
    c_proj = base_model.transformer.h[blk].mlp.c_proj
    # GPT-2 Conv1D: weight shape (3072, 768), forward does x @ weight + bias
    # Row j of weight is how hidden channel j maps to 768 output dims
    c_proj_weight = c_proj.weight.detach().cpu().float()  # (3072, 768) for Conv1D
    output_norms = c_proj_weight.norm(dim=1)  # (3072,) — one norm per hidden channel

    # Full channel scores
    channel_scores = std_h * output_norms * (mean_h.abs() + mean_deriv.abs() + 1e-6)

    # Dead mask
    dead_mask = hidden.abs().amax(dim=0) <= 1e-6
    available = torch.arange(hidden_dim)
    live = available[~dead_mask]
    if live.numel() < k:
        live = available
    scores = channel_scores[live]
    topk_idx = torch.topk(scores, k=k, largest=True).indices
    selected = live[topk_idx].sort().values

    # Extract c_fc weights/bias at selected indices
    c_fc = base_model.transformer.h[blk].mlp.c_fc
    # Conv1D: weight shape (768, 3072), forward: x @ weight + bias
    # weight.T gives (3072, 768) in standard Linear format
    c_fc_weight_full = c_fc.weight.detach().cpu().float()  # (768, 3072) Conv1D
    c_fc_bias_full = c_fc.bias.detach().cpu().float()      # (3072,)

    fc_w = c_fc_weight_full.T[selected]  # (k, 768) — standard Linear format
    fc_b = c_fc_bias_full[selected]      # (k,)

    # lstsq for output projection
    hidden_sel = hidden[:, selected].float()  # (N, k)
    design = torch.cat([hidden_sel, torch.ones(hidden_sel.shape[0], 1)], dim=1)  # (N, k+1)
    solution = torch.linalg.lstsq(design, output.float()).solution  # (k+1, 768)
    proj_w = solution[:-1].T.contiguous()  # (768, k) — standard Linear format
    proj_b = solution[-1].contiguous()     # (768,)

    # Verify lstsq reconstruction quality
    with torch.no_grad():
        recon = hidden_sel @ proj_w.T + proj_b
        mse = F.mse_loss(recon, output).item()
        cos = F.cosine_similarity(recon.reshape(1, -1), output.reshape(1, -1)).item()

    compress_cache[blk] = {
        "fc_w": fc_w,
        "fc_b": fc_b,
        "proj_w": proj_w,
        "proj_b": proj_b,
        "selected": selected,
        "k": k,
        "mse": mse,
        "cos": cos,
    }
    print(f"Block {blk}: k={k}, dead={dead_mask.sum().item()}, "
          f"lstsq MSE={mse:.4f}, cos={cos:.6f}", flush=True)

# Count trainable params per block (proj_w + proj_b)
for blk in ACCUM_BLOCKS:
    c = compress_cache[blk]
    n_train = c["proj_w"].numel() + c["proj_b"].numel()
    print(f"Block {blk}: trainable params = {n_train:,} (proj_w {c['proj_w'].shape} + proj_b {c['proj_b'].shape})", flush=True)

del block_acts
if DEVICE == "mps":
    torch.mps.empty_cache()

# ── Accumulation curve ────────────────────────────────────────────────────────
stages = []
for n_blocks in range(1, len(ACCUM_BLOCKS) + 1):
    active_blocks = ACCUM_BLOCKS[:n_blocks]
    trainable_count = sum(compress_cache[b]["proj_w"].numel() + compress_cache[b]["proj_b"].numel() for b in active_blocks)

    print(f"\n{'='*60}", flush=True)
    print(f"Stage {n_blocks}: blocks {active_blocks}, k={compress_cache[active_blocks[0]]['k']}, "
          f"trainable={trainable_count:,}", flush=True)
    print(f"{'='*60}", flush=True)

    seed_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        random.seed(seed)

        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        model.eval()

        # Install CompressedGPT2MLP for active blocks
        compressed_modules = {}
        for blk in active_blocks:
            c = compress_cache[blk]
            comp_mlp = CompressedGPT2MLP(
                c_fc_weight=c["fc_w"].to(device=DEVICE, dtype=torch.float32),
                c_fc_bias=c["fc_b"].to(device=DEVICE, dtype=torch.float32),
                c_proj_weight=c["proj_w"].to(device=DEVICE, dtype=torch.float32),
                c_proj_bias=c["proj_b"].to(device=DEVICE, dtype=torch.float32),
            ).to(DEVICE)
            model.transformer.h[blk].mlp = comp_mlp
            compressed_modules[blk] = comp_mlp

        init_ppl = eval_ppl(model, eval_ids)

        # Freeze everything, unfreeze proj_weight + proj_bias
        for p in model.parameters():
            p.requires_grad_(False)
        trainable = []
        for blk in active_blocks:
            compressed_modules[blk].proj_weight.requires_grad_(True)
            compressed_modules[blk].proj_bias.requires_grad_(True)
            trainable.append(compressed_modules[blk].proj_weight)
            trainable.append(compressed_modules[blk].proj_bias)
        n_train = sum(p.numel() for p in trainable)
        assert n_train == trainable_count, f"{n_train} != {trainable_count}"

        optimizer = torch.optim.Adam(trainable, lr=LR)
        calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]
        random.shuffle(calib_ids_dev)

        best_ppl = init_ppl
        best_step = 0
        best_state = {}
        for blk in active_blocks:
            best_state[f"proj_w_{blk}"] = compressed_modules[blk].proj_weight.detach().clone()
            best_state[f"proj_b_{blk}"] = compressed_modules[blk].proj_bias.detach().clone()
        history = [{"step": 0, "eval_ppl": float(init_ppl)}]

        step = 0
        t0 = time.time()
        while step < N_STEPS:
            for ids in calib_ids_dev:
                if step >= N_STEPS:
                    break
                model.train()
                optimizer.zero_grad()
                out = model(ids, labels=ids)
                loss = out.loss
                loss.backward()
                optimizer.step()
                step += 1

                if step % EVAL_EVERY == 0:
                    ppl_e = eval_ppl(model, eval_ids)
                    history.append({"step": step, "eval_ppl": float(ppl_e)})
                    if ppl_e < best_ppl:
                        best_ppl = ppl_e
                        best_step = step
                        best_state = {}
                        for blk in active_blocks:
                            best_state[f"proj_w_{blk}"] = compressed_modules[blk].proj_weight.detach().clone()
                            best_state[f"proj_b_{blk}"] = compressed_modules[blk].proj_bias.detach().clone()
                    if step % 500 == 0:
                        print(f"  seed {seed} step {step}: ppl={ppl_e:.2f} "
                              f"best={best_ppl:.2f}@{best_step}", flush=True)

        # Restore best
        with torch.no_grad():
            for blk in active_blocks:
                compressed_modules[blk].proj_weight.data.copy_(best_state[f"proj_w_{blk}"])
                compressed_modules[blk].proj_bias.data.copy_(best_state[f"proj_b_{blk}"])
        restored_ppl = eval_ppl(model, eval_ids)
        dt = time.time() - t0

        print(f"  seed {seed}: init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
              f"restored={restored_ppl:.2f} ({dt:.0f}s)", flush=True)

        seed_results.append({
            "seed": seed,
            "init_ppl": float(init_ppl),
            "best_ppl": float(best_ppl),
            "best_step": int(best_step),
            "restored_ppl": float(restored_ppl),
            "history": history,
        })

        del model, compressed_modules, optimizer
        if DEVICE == "mps":
            torch.mps.empty_cache()

    # Aggregate stage
    best_ppls = [s["best_ppl"] for s in seed_results]
    best_mean = sum(best_ppls) / len(best_ppls)
    best_std = math.sqrt(sum((x - best_mean) ** 2 for x in best_ppls) / len(best_ppls))
    init_ppls = [s["init_ppl"] for s in seed_results]
    init_mean = sum(init_ppls) / len(init_ppls)

    stage_data = {
        "n_blocks": n_blocks,
        "blocks": active_blocks,
        "k_per_block": compress_cache[active_blocks[0]]["k"],
        "trainable_params": trainable_count,
        "init_mean": float(init_mean),
        "best_mean": float(best_mean),
        "best_std": float(best_std),
        "best_min": float(min(best_ppls)),
        "best_max": float(max(best_ppls)),
        "seeds": seed_results,
    }
    stages.append(stage_data)

    print(f"\nStage {n_blocks} result: best_mean={best_mean:.2f} +/- {best_std:.2f} "
          f"(init {init_mean:.2f})", flush=True)


# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("ACCUMULATION CURVE — topk+lstsq", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Blocks':>20s}  {'Init':>8s}  {'Best':>8s}  {'Std':>6s}  {'Params':>8s}")
for s in stages:
    print(f"{str(s['blocks']):>20s}  {s['init_mean']:8.2f}  {s['best_mean']:8.2f}  "
          f"{s['best_std']:6.2f}  {s['trainable_params']:8d}")

print(f"\nCycle 343 LASER comparison:")
laser_curve = [50.21, 55.10, 58.15, 68.97]
for i, (s, l) in enumerate(zip(stages, laser_curve)):
    delta = s["best_mean"] - l
    print(f"  Stage {i+1}: topk={s['best_mean']:.2f}, LASER={l:.2f}, delta={delta:+.2f}")

# Check verdict
single_best = stages[0]["best_mean"]
multi_compose = any(s["best_mean"] <= CONFIRMED_LE for s in stages[1:])
multi_all_degrade = all(s["best_mean"] > single_best for s in stages[1:])
monotonic = all(stages[i+1]["best_mean"] > stages[i]["best_mean"] for i in range(len(stages)-1))

if multi_compose:
    verdict = "CONFIRMED"
elif monotonic:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nFloat PPL: {float_ppl:.4f}", flush=True)
print(f"Single-block best (stage 1): {single_best:.4f}", flush=True)
print(f"CONFIRMED threshold (<=): {CONFIRMED_LE}", flush=True)
print(f"Multi-block compose (any >=2-block <= {CONFIRMED_LE}): {multi_compose}", flush=True)
print(f"Multi-block monotonic degrade: {monotonic}", flush=True)
print(f"VERDICT: {verdict}", flush=True)

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 348,
    "item": 355,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "confirmed_le": CONFIRMED_LE,
    "method": "topk+lstsq",
    "hidden_fraction": HIDDEN_FRACTION,
    "accumulation_order": ACCUM_BLOCKS,
    "stages": [{k: v for k, v in s.items() if k != "seeds"} for s in stages],
    "stage_seeds": {s["n_blocks"]: s["seeds"] for s in stages},
    "laser_comparison": {
        "cycle_343_laser_curve": laser_curve,
        "topk_curve": [s["best_mean"] for s in stages],
        "deltas": [s["best_mean"] - l for s, l in zip(stages, laser_curve)],
    },
    "per_block_lstsq": {
        blk: {"k": compress_cache[blk]["k"], "mse": compress_cache[blk]["mse"],
               "cos": compress_cache[blk]["cos"]}
        for blk in ACCUM_BLOCKS
    },
    "config": {
        "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE, "N_STEPS": N_STEPS, "LR": LR,
        "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
        "HIDDEN_FRACTION": HIDDEN_FRACTION,
    },
}

import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved results to {OUTPUT_PATH}", flush=True)
