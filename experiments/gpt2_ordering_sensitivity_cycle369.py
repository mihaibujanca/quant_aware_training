#!/usr/bin/env python3
"""Cycle 369 / Item 360 — Accumulation order sensitivity for GPT-2.

Test whether the order in which blocks are compressed affects final PPL.
Sequential cumulative compression: each block is compressed from the current
modified model state (not all from float), so ordering can matter.

Three orderings (blocks [1,4,7,11]):
  - easiest_first: [1, 7, 4, 11]  (ascending importance delta from cycle 367)
  - block_index:   [1, 4, 7, 11]  (natural block index order)
  - hardest_first: [11, 4, 7, 1]  (descending importance delta)

Block importance deltas from cycle 367:
  block 1: +1.60, block 7: +2.35, block 4: +6.85, block 11: +9.12

Config: k=512, adapter_hidden=64, 500 steps/stage, lr=1e-4, seeds=[42,137,2024]

Verdict:
  CONFIRMED    if max(mean_ppl)/min(mean_ppl) > 1.05 (>5% gap)
  REFUTED      if gap < 3% (orderings essentially equivalent)
  INCONCLUSIVE if gap is 3-5%
"""
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
# Orderings of 4 blocks to test
ORDERINGS = {
    "easiest_first": [1, 7, 4, 11],
    "block_index":   [1, 4, 7, 11],
    "hardest_first": [11, 4, 7, 1],
}

K = 512
ADAPTER_HIDDEN = 64
STEPS_PER_STAGE = 500
LR = 1e-4
SEEDS = [42, 137, 2024]

N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
EVAL_EVERY = 100

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

FLOAT_BASELINE_REF = 53.895

OUTPUT_PATH = "autoresearch/results/cycle_369_ordering_sensitivity.json"


def eval_ppl(model, eval_ids_dev):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ids in eval_ids_dev:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


def collect_single_block_activations(model, block_id, calib_ids, device):
    """Collect MLP activations for a single block from current model state."""
    data = {"input": [], "output": [], "preact": [], "hidden": []}

    def hook_fn(module, inputs, output):
        x = inputs[0].detach()
        z = module.c_fc(x).detach()
        h = F.gelu(z).detach()
        y = output.detach()
        data["input"].append(x.reshape(-1, x.shape[-1]).cpu())
        data["output"].append(y.reshape(-1, y.shape[-1]).cpu())
        data["preact"].append(z.reshape(-1, z.shape[-1]).cpu())
        data["hidden"].append(h.reshape(-1, h.shape[-1]).cpu())

    hook = model.transformer.h[block_id].mlp.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for ids in calib_ids:
            ids_dev = ids.to(device) if ids.device.type != device else ids
            model(ids_dev)

    hook.remove()

    return {
        "input": torch.cat(data["input"], dim=0).float(),
        "output": torch.cat(data["output"], dim=0).float(),
        "preact": torch.cat(data["preact"], dim=0).float(),
        "hidden": torch.cat(data["hidden"], dim=0).float(),
    }


class CompressedGPT2MLPWithAdapter(nn.Module):
    """Compressed MLP with MLP adapter."""

    def __init__(self, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias,
                 adapter_hidden=64):
        super().__init__()
        k = c_fc_weight.shape[0]
        self.register_buffer("fc_weight", c_fc_weight.contiguous())
        self.register_buffer("fc_bias", c_fc_bias.contiguous())
        self.proj_weight = nn.Parameter(c_proj_weight.contiguous())
        self.proj_bias = nn.Parameter(c_proj_bias.contiguous())

        scale1 = 1.0 / math.sqrt(k)
        scale2 = 1.0 / math.sqrt(adapter_hidden)
        self.adapter_W1 = nn.Parameter(torch.randn(adapter_hidden, k) * scale1)
        self.adapter_b1 = nn.Parameter(torch.zeros(adapter_hidden))
        self.adapter_W2 = nn.Parameter(torch.randn(768, adapter_hidden) * scale2)
        self.adapter_b2 = nn.Parameter(torch.zeros(768))

    def forward(self, x):
        h = F.linear(x, self.fc_weight, self.fc_bias)
        h = F.gelu(h)
        out = F.linear(h, self.proj_weight, self.proj_bias)
        a = F.relu(F.linear(h, self.adapter_W1, self.adapter_b1))
        out = out + F.linear(a, self.adapter_W2, self.adapter_b2)
        return out


def compress_single_block(model, block_id, k, calib_ids_dev, device):
    """Compress one block from current model state using topk+lstsq. Returns compressed module."""
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    acts = collect_single_block_activations(model, block_id, calib_ids_dev, device)
    hidden = acts["hidden"]
    output = acts["output"]
    preact = acts["preact"]

    hidden_dim = hidden.shape[1]

    std_h = hidden.std(dim=0)
    mean_h = hidden.mean(dim=0)
    deriv = (0.5 * (1.0 + torch.erf(preact * inv_sqrt_2))
             + preact * torch.exp(-0.5 * preact.square()) * inv_sqrt_2pi)
    mean_deriv = deriv.mean(dim=0)

    # GPT2 uses Conv1D: c_fc.weight is (768, 3072) i.e. (in, out) convention.
    # .T gives (3072, 768); [selected] selects k rows -> (k, 768). Correct for CompressedGPT2MLP.
    # c_proj.weight is (3072, 768) i.e. (in, out) — norm(dim=1) gives norms per hidden neuron.
    mlp = model.transformer.h[block_id].mlp
    assert not isinstance(mlp, CompressedGPT2MLPWithAdapter), \
        f"Block {block_id} is already compressed — ordering cannot re-compress a block"

    c_proj = mlp.c_proj
    c_proj_weight = c_proj.weight.detach().cpu().float()   # (3072, 768)
    output_norms = c_proj_weight.norm(dim=1)               # (3072,)
    c_fc = mlp.c_fc
    c_fc_weight_full = c_fc.weight.detach().cpu().float()  # (768, 3072)
    c_fc_bias_full = c_fc.bias.detach().cpu().float()      # (3072,)

    channel_scores = std_h * output_norms * (mean_h.abs() + mean_deriv.abs() + 1e-6)

    dead_mask = hidden.abs().amax(dim=0) <= 1e-6
    available = torch.arange(hidden_dim)
    live = available[~dead_mask]
    if live.numel() < k:
        live = available
    scores = channel_scores[live]
    topk_idx = torch.topk(scores, k=k, largest=True).indices
    selected = live[topk_idx].sort().values

    # c_fc_weight_full.T is (3072, 768); [selected] picks k rows -> (k, 768)
    fc_w = c_fc_weight_full.T[selected]    # (k, 768)
    fc_b = c_fc_bias_full[selected]        # (k,)

    hidden_sel = hidden[:, selected].float()
    design = torch.cat([hidden_sel, torch.ones(hidden_sel.shape[0], 1)], dim=1)
    solution = torch.linalg.lstsq(design, output.float()).solution
    proj_w = solution[:-1].T.contiguous()   # (768, k)
    proj_b = solution[-1].contiguous()      # (768,)

    with torch.no_grad():
        recon = hidden_sel @ proj_w.T + proj_b
        mse = F.mse_loss(recon, output.float()).item()
        frob_recon = (recon - output.float()).norm().item()

    print(f"  Block {block_id}: k={k} (frac={k/hidden_dim:.3f}), "
          f"MSE={mse:.4f}, frob_recon={frob_recon:.2f}", flush=True)

    return CompressedGPT2MLPWithAdapter(
        c_fc_weight=fc_w.to(device=device, dtype=torch.float32),
        c_fc_bias=fc_b.to(device=device, dtype=torch.float32),
        c_proj_weight=proj_w.to(device=device, dtype=torch.float32),
        c_proj_bias=proj_b.to(device=device, dtype=torch.float32),
        adapter_hidden=ADAPTER_HIDDEN,
    ).to(device), {"mse": mse, "frob_recon": frob_recon}


def train_joint_adapters(model, compressed_modules, n_steps, lr,
                         calib_ids_dev, eval_ids_dev, phase_name="joint"):
    """Train proj_weight + adapter params for all compressed_modules jointly."""
    for p in model.parameters():
        p.requires_grad_(False)

    trainable = []
    for mod in compressed_modules.values():
        for pname in ["proj_weight", "proj_bias",
                      "adapter_W1", "adapter_b1", "adapter_W2", "adapter_b2"]:
            p = getattr(mod, pname)
            p.requires_grad_(True)
            trainable.append(p)

    n_train = sum(p.numel() for p in trainable)
    print(f"  [{phase_name}] trainable params: {n_train:,}", flush=True)

    init_ppl = eval_ppl(model, eval_ids_dev)
    print(f"  [{phase_name}] init PPL: {init_ppl:.4f}", flush=True)

    optimizer = torch.optim.Adam(trainable, lr=lr)
    ids_shuffled = list(calib_ids_dev)
    random.shuffle(ids_shuffled)

    best_ppl = init_ppl
    best_step = 0
    best_state = {}
    for blk, mod in compressed_modules.items():
        for pname in ["proj_weight", "proj_bias",
                      "adapter_W1", "adapter_b1", "adapter_W2", "adapter_b2"]:
            best_state[f"{pname}_{blk}"] = getattr(mod, pname).detach().clone()

    step = 0
    t0 = time.time()
    while step < n_steps:
        for ids in ids_shuffled:
            if step >= n_steps:
                break
            model.train()
            optimizer.zero_grad()
            out = model(ids, labels=ids)
            out.loss.backward()
            optimizer.step()
            step += 1
            if step % EVAL_EVERY == 0:
                ppl_e = eval_ppl(model, eval_ids_dev)
                if ppl_e < best_ppl:
                    best_ppl = ppl_e
                    best_step = step
                    for blk, mod in compressed_modules.items():
                        for pname in ["proj_weight", "proj_bias",
                                      "adapter_W1", "adapter_b1", "adapter_W2", "adapter_b2"]:
                            best_state[f"{pname}_{blk}"] = getattr(mod, pname).detach().clone()
                if step % 500 == 0:
                    print(f"  [{phase_name}] step {step}: ppl={ppl_e:.2f} "
                          f"best={best_ppl:.2f}@{best_step}", flush=True)

    # Restore best
    with torch.no_grad():
        for blk, mod in compressed_modules.items():
            for pname in ["proj_weight", "proj_bias",
                          "adapter_W1", "adapter_b1", "adapter_W2", "adapter_b2"]:
                getattr(mod, pname).data.copy_(best_state[f"{pname}_{blk}"])

    restored_ppl = eval_ppl(model, eval_ids_dev)
    dt = time.time() - t0
    print(f"  [{phase_name}] init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
          f"restored={restored_ppl:.2f} ({dt:.0f}s)", flush=True)
    return best_ppl, restored_ppl


def run_ordering(ordering_name, ordering, seed, calib_ids_cpu, eval_ids_dev):
    """Run one (ordering, seed) pair. Returns list of PPL at each stage."""
    print(f"\n{'='*60}", flush=True)
    print(f"Ordering: {ordering_name} {ordering}  Seed: {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    torch.manual_seed(seed)
    random.seed(seed)

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]

    compressed_modules = {}  # blk -> CompressedGPT2MLPWithAdapter
    stage_ppls = []
    stage_stats = []

    for stage_idx, block_id in enumerate(ordering):
        stage_num = stage_idx + 1
        print(f"\n-- Stage {stage_num}/4: compressing block {block_id} --", flush=True)

        # Compress single block from current model state
        comp_mlp, stats = compress_single_block(
            model, block_id, K, calib_ids_dev, DEVICE
        )
        model.transformer.h[block_id].mlp = comp_mlp
        compressed_modules[block_id] = comp_mlp

        # Joint-train all currently compressed modules for 500 steps
        best_ppl, restored_ppl = train_joint_adapters(
            model, compressed_modules, STEPS_PER_STAGE, LR,
            calib_ids_dev, eval_ids_dev,
            phase_name=f"{ordering_name}-s{seed}-stage{stage_num}"
        )

        stage_ppls.append(float(restored_ppl))
        stage_stats.append({
            "block_id": block_id,
            "stage": stage_num,
            "mse": stats["mse"],
            "frob_recon": stats["frob_recon"],
            "best_ppl": float(best_ppl),
            "restored_ppl": float(restored_ppl),
        })
        print(f"  Stage {stage_num} PPL: {restored_ppl:.4f}", flush=True)

    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return stage_ppls, stage_stats


# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}", flush=True)
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
eval_ids_dev = [t.to(DEVICE) for t in all_ids[N_CALIB: N_CALIB + N_EVAL]]
print(f"Calibration: {len(calib_ids_cpu)} seqs, Eval: {len(eval_ids_dev)} seqs", flush=True)

# ── Float baseline ─────────────────────────────────────────────────────────────
print("Computing float baseline PPL...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
float_ppl = eval_ppl(base_model, eval_ids_dev)
print(f"Float PPL: {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from reference by > 0.5"
)
del base_model
if DEVICE == "mps":
    torch.mps.empty_cache()

# ── Run experiments ────────────────────────────────────────────────────────────
all_results = {
    "float_ppl": float(float_ppl),
    "config": {
        "k": K,
        "adapter_hidden": ADAPTER_HIDDEN,
        "steps_per_stage": STEPS_PER_STAGE,
        "lr": LR,
        "seeds": SEEDS,
        "n_calib": N_CALIB,
        "n_eval": N_EVAL,
        "seq_len": SEQ_LEN,
        "orderings": ORDERINGS,
    },
    "orderings": {},
}

for ordering_name, ordering in ORDERINGS.items():
    ordering_results = {
        "ordering": ordering,
        "seeds": {},
        "stage_ppls_per_seed": [],
    }

    for seed in SEEDS:
        stage_ppls, stage_stats = run_ordering(
            ordering_name, ordering, seed, calib_ids_cpu, eval_ids_dev
        )
        ordering_results["seeds"][str(seed)] = {
            "stage_ppls": stage_ppls,
            "stage_stats": stage_stats,
            "final_ppl": stage_ppls[-1],
        }
        ordering_results["stage_ppls_per_seed"].append(stage_ppls)

    # Compute mean PPL at each stage
    n_seeds = len(SEEDS)
    mean_stage_ppls = [
        sum(ordering_results["stage_ppls_per_seed"][s][st] for s in range(n_seeds)) / n_seeds
        for st in range(4)
    ]
    final_ppls = [ordering_results["seeds"][str(s)]["final_ppl"] for s in SEEDS]
    mean_final = sum(final_ppls) / len(final_ppls)
    std_final = (sum((p - mean_final) ** 2 for p in final_ppls) / len(final_ppls)) ** 0.5

    ordering_results["mean_stage_ppls"] = mean_stage_ppls
    ordering_results["mean_final_ppl"] = float(mean_final)
    ordering_results["std_final_ppl"] = float(std_final)

    print(f"\n{ordering_name}: mean stage PPLs = {[f'{p:.2f}' for p in mean_stage_ppls]}", flush=True)
    print(f"  Final: {mean_final:.4f} ± {std_final:.4f}", flush=True)

    all_results["orderings"][ordering_name] = ordering_results

# ── Compute verdict ────────────────────────────────────────────────────────────
mean_finals = {name: res["mean_final_ppl"] for name, res in all_results["orderings"].items()}
best_ordering = min(mean_finals, key=mean_finals.get)
worst_ordering = max(mean_finals, key=mean_finals.get)
best_ppl_final = mean_finals[best_ordering]
worst_ppl_final = mean_finals[worst_ordering]
ratio = worst_ppl_final / best_ppl_final
gap_pct = (ratio - 1.0) * 100.0

print(f"\n{'='*60}", flush=True)
print(f"ORDERING SENSITIVITY SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
for name, ppl in sorted(mean_finals.items(), key=lambda x: x[1]):
    print(f"  {name}: mean_final_ppl={ppl:.4f}", flush=True)
print(f"  Best: {best_ordering} ({best_ppl_final:.4f})", flush=True)
print(f"  Worst: {worst_ordering} ({worst_ppl_final:.4f})", flush=True)
print(f"  Ratio worst/best: {ratio:.4f}  Gap: {gap_pct:.1f}%", flush=True)

if ratio > 1.05:
    verdict = "CONFIRMED"
elif ratio < 1.03:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nVERDICT: {verdict}", flush=True)
print(f"  Threshold: >5% gap CONFIRMED, <3% REFUTED, 3-5% INCONCLUSIVE", flush=True)
print(f"{'='*60}", flush=True)

all_results["verdict"] = verdict
all_results["mean_finals"] = mean_finals
all_results["best_ordering"] = best_ordering
all_results["worst_ordering"] = worst_ordering
all_results["ratio_worst_best"] = float(ratio)
all_results["gap_pct"] = float(gap_pct)

# ── Save results ───────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {OUTPUT_PATH}", flush=True)
