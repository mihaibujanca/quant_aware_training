#!/usr/bin/env python3
"""Cycle 351 / Item 360 — GPT-2 topk+lstsq accumulation ORDER sensitivity.

Question: does the ORDER in which blocks are added affect final PPL?

Blocks under test: [b1, b4, b7, b11] — spanning early/mid/late GPT-2.

Protocol:
  1. Measure single-block swap damage for each block (compress alone, measure PPL).
  2. Sort blocks by damage (lowest = "easiest", highest = "hardest").
  3. Run 3 orderings × 4 accumulation stages = 12 (ordering, stage) pairs.
     - easiest-first: lowest damage first
     - block-index:   [b1, b4, b7, b11] in numeric order
     - hardest-first: highest damage first (reverse of easiest-first)
  4. Each (ordering, stage) measured with 3 seeds.

Verdict:
  CONFIRMED  if >5% PPL gap between best and worst ordering at stage 4
  REFUTED    if <2% PPL gap at stage 4 (ordering doesn't matter)
  INCONCLUSIVE if 2-5% gap

Reference (cycle 348): same blocks [b2,b1,b10,b11], stage-4 PPL=55.99.
Float baseline: ~53.895
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
TARGET_BLOCKS = [1, 4, 7, 11]   # blocks to test; numeric order = block-index ordering

# topk+lstsq config: 6x compression = keep 512 of 3072 hidden channels
HIDDEN_FRACTION = 1 / 6  # → k=512

N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 1000
LR = 1e-4
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
LOAD_SEED = 42

FLOAT_BASELINE_REF = 53.895

OUTPUT_PATH = "autoresearch/results/cycle_351_order_sensitivity.json"

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


def collect_block_activations(model, block_ids, calib_ids, device):
    """Collect MLP activations for specified blocks."""
    block_data = {b: {"input": [], "output": [], "preact": [], "hidden": []} for b in block_ids}
    hooks = []

    def make_hook(blk_id):
        def hook_fn(module, inputs, output):
            x = inputs[0].detach()
            z = module.c_fc(x).detach()
            h = F.gelu(z).detach()
            y = output.detach()
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

    result = {}
    for blk_id in block_ids:
        result[blk_id] = {
            "input": torch.cat(block_data[blk_id]["input"], dim=0).float(),
            "output": torch.cat(block_data[blk_id]["output"], dim=0).float(),
            "preact": torch.cat(block_data[blk_id]["preact"], dim=0).float(),
            "hidden": torch.cat(block_data[blk_id]["hidden"], dim=0).float(),
        }
    return result


class CompressedGPT2MLP(nn.Module):
    """Compressed MLP: keeps k of 3072 hidden channels, with trainable W_out."""

    def __init__(self, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias):
        super().__init__()
        # Frozen input projection
        self.register_buffer("fc_weight", c_fc_weight.contiguous())  # (k, 768)
        self.register_buffer("fc_bias", c_fc_bias.contiguous())      # (k,)
        # Trainable output projection
        self.proj_weight = nn.Parameter(c_proj_weight.contiguous())   # (768, k)
        self.proj_bias = nn.Parameter(c_proj_bias.contiguous())       # (768,)

    def forward(self, x):
        h = F.linear(x, self.fc_weight, self.fc_bias)
        h = F.gelu(h)
        out = F.linear(h, self.proj_weight, self.proj_bias)
        return out


def compute_compress_cache(base_model, block_acts, blocks, device):
    """Compute topk+lstsq compression parameters for each block."""
    cache = {}
    for blk in blocks:
        acts = block_acts[blk]
        hidden = acts["hidden"]   # (N, 3072)
        output = acts["output"]   # (N, 768)
        preact = acts["preact"]   # (N, 3072)

        hidden_dim = hidden.shape[1]
        k = max(1, int(round(hidden_dim * HIDDEN_FRACTION)))

        std_h = hidden.std(dim=0)
        mean_h = hidden.mean(dim=0)
        inv_sqrt_2 = 1.0 / math.sqrt(2.0)
        inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
        deriv = 0.5 * (1.0 + torch.erf(preact * inv_sqrt_2)) + preact * torch.exp(-0.5 * preact.square()) * inv_sqrt_2pi
        mean_deriv = deriv.mean(dim=0)

        c_proj = base_model.transformer.h[blk].mlp.c_proj
        c_proj_weight = c_proj.weight.detach().cpu().float()  # Conv1D: (3072, 768)
        output_norms = c_proj_weight.norm(dim=1)              # (3072,)

        channel_scores = std_h * output_norms * (mean_h.abs() + mean_deriv.abs() + 1e-6)

        dead_mask = hidden.abs().amax(dim=0) <= 1e-6
        available = torch.arange(hidden_dim)
        live = available[~dead_mask]
        if live.numel() < k:
            live = available
        scores = channel_scores[live]
        topk_idx = torch.topk(scores, k=k, largest=True).indices
        selected = live[topk_idx].sort().values

        c_fc = base_model.transformer.h[blk].mlp.c_fc
        c_fc_weight_full = c_fc.weight.detach().cpu().float()  # Conv1D: (768, 3072)
        c_fc_bias_full = c_fc.bias.detach().cpu().float()      # (3072,)

        fc_w = c_fc_weight_full.T[selected]   # (k, 768)
        fc_b = c_fc_bias_full[selected]       # (k,)

        hidden_sel = hidden[:, selected].float()
        design = torch.cat([hidden_sel, torch.ones(hidden_sel.shape[0], 1)], dim=1)
        solution = torch.linalg.lstsq(design, output.float()).solution
        proj_w = solution[:-1].T.contiguous()  # (768, k)
        proj_b = solution[-1].contiguous()     # (768,)

        with torch.no_grad():
            recon = hidden_sel @ proj_w.T + proj_b
            mse = F.mse_loss(recon, output).item()
            cos = F.cosine_similarity(recon.reshape(1, -1), output.reshape(1, -1)).item()

        cache[blk] = {
            "fc_w": fc_w, "fc_b": fc_b, "proj_w": proj_w, "proj_b": proj_b,
            "selected": selected, "k": k, "mse": mse, "cos": cos,
        }
        print(f"  Block {blk}: k={k}, dead={dead_mask.sum().item()}, "
              f"MSE={mse:.4f}, cos={cos:.6f}", flush=True)
    return cache


def run_accumulation_stage(base_model, active_blocks, compress_cache, seeds, device,
                            calib_ids_cpu, eval_ids, n_steps=N_STEPS, lr=LR):
    """Run one accumulation stage (list of active blocks) with multiple seeds.

    Returns list of per-seed dicts with init_ppl, best_ppl, best_step, restored_ppl.
    """
    trainable_count = sum(
        compress_cache[b]["proj_w"].numel() + compress_cache[b]["proj_b"].numel()
        for b in active_blocks
    )
    seed_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)

        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        model.eval()

        compressed_modules = {}
        for blk in active_blocks:
            c = compress_cache[blk]
            comp_mlp = CompressedGPT2MLP(
                c_fc_weight=c["fc_w"].to(device=device, dtype=torch.float32),
                c_fc_bias=c["fc_b"].to(device=device, dtype=torch.float32),
                c_proj_weight=c["proj_w"].to(device=device, dtype=torch.float32),
                c_proj_bias=c["proj_b"].to(device=device, dtype=torch.float32),
            ).to(device)
            model.transformer.h[blk].mlp = comp_mlp
            compressed_modules[blk] = comp_mlp

        init_ppl = eval_ppl(model, eval_ids)

        for p in model.parameters():
            p.requires_grad_(False)
        trainable = []
        for blk in active_blocks:
            compressed_modules[blk].proj_weight.requires_grad_(True)
            compressed_modules[blk].proj_bias.requires_grad_(True)
            trainable.extend([compressed_modules[blk].proj_weight,
                               compressed_modules[blk].proj_bias])

        optimizer = torch.optim.Adam(trainable, lr=lr)
        calib_ids_dev = [t.to(device) for t in calib_ids_cpu]
        random.shuffle(calib_ids_dev)

        best_ppl = init_ppl
        best_step = 0
        best_state = {
            f"proj_w_{blk}": compressed_modules[blk].proj_weight.detach().clone()
            for blk in active_blocks
        }
        best_state.update({
            f"proj_b_{blk}": compressed_modules[blk].proj_bias.detach().clone()
            for blk in active_blocks
        })

        step = 0
        t0 = time.time()
        while step < n_steps:
            for ids in calib_ids_dev:
                if step >= n_steps:
                    break
                model.train()
                optimizer.zero_grad()
                out = model(ids, labels=ids)
                out.loss.backward()
                optimizer.step()
                step += 1

                if step % EVAL_EVERY == 0:
                    ppl_e = eval_ppl(model, eval_ids)
                    if ppl_e < best_ppl:
                        best_ppl = ppl_e
                        best_step = step
                        for blk in active_blocks:
                            best_state[f"proj_w_{blk}"] = compressed_modules[blk].proj_weight.detach().clone()
                            best_state[f"proj_b_{blk}"] = compressed_modules[blk].proj_bias.detach().clone()

        with torch.no_grad():
            for blk in active_blocks:
                compressed_modules[blk].proj_weight.data.copy_(best_state[f"proj_w_{blk}"])
                compressed_modules[blk].proj_bias.data.copy_(best_state[f"proj_b_{blk}"])
        restored_ppl = eval_ppl(model, eval_ids)
        dt = time.time() - t0

        print(f"    seed {seed}: init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
              f"restored={restored_ppl:.2f} ({dt:.0f}s)", flush=True)

        seed_results.append({
            "seed": seed,
            "init_ppl": float(init_ppl),
            "best_ppl": float(best_ppl),
            "best_step": int(best_step),
            "restored_ppl": float(restored_ppl),
            "trainable_params": trainable_count,
        })

        del model, compressed_modules, optimizer
        if device == "mps":
            torch.mps.empty_cache()

    return seed_results


def aggregate(seed_results):
    ppls = [s["best_ppl"] for s in seed_results]
    mean = sum(ppls) / len(ppls)
    std = math.sqrt(sum((x - mean) ** 2 for x in ppls) / len(ppls))
    return mean, std


# ── Float baseline ────────────────────────────────────────────────────────────
print("Computing float baseline PPL...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
float_ppl = eval_ppl(base_model, eval_ids)
print(f"Float PPL: {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from reference {FLOAT_BASELINE_REF} by > 0.5"
)

# ── Collect activations for all target blocks ─────────────────────────────────
print("\nCollecting calibration activations...", flush=True)
calib_dev = [t.to(DEVICE) for t in calib_ids_cpu]
block_acts = collect_block_activations(base_model, TARGET_BLOCKS, calib_dev, DEVICE)

# ── Compute compression cache ─────────────────────────────────────────────────
print("\nComputing topk+lstsq compression for each block...", flush=True)
compress_cache = compute_compress_cache(base_model, block_acts, TARGET_BLOCKS, DEVICE)
del block_acts
if DEVICE == "mps":
    torch.mps.empty_cache()

# ── Phase 1: single-block swap damage ─────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("Phase 1: Single-block swap damage (no Adam fine-tuning)", flush=True)
print("="*60, flush=True)

single_block_ppl = {}
for blk in TARGET_BLOCKS:
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    c = compress_cache[blk]
    comp_mlp = CompressedGPT2MLP(
        c_fc_weight=c["fc_w"].to(device=DEVICE, dtype=torch.float32),
        c_fc_bias=c["fc_b"].to(device=DEVICE, dtype=torch.float32),
        c_proj_weight=c["proj_w"].to(device=DEVICE, dtype=torch.float32),
        c_proj_bias=c["proj_b"].to(device=DEVICE, dtype=torch.float32),
    ).to(DEVICE)
    model.transformer.h[blk].mlp = comp_mlp
    ppl = eval_ppl(model, eval_ids)
    damage = ppl - float_ppl
    single_block_ppl[blk] = {"ppl": float(ppl), "damage": float(damage)}
    print(f"  Block {blk}: PPL={ppl:.2f}  damage={damage:+.2f}", flush=True)
    del model, comp_mlp
    if DEVICE == "mps":
        torch.mps.empty_cache()

# ── Determine orderings ───────────────────────────────────────────────────────
# easiest-first: sort ascending by damage (lowest damage = easiest to compress)
easiest_first = sorted(TARGET_BLOCKS, key=lambda b: single_block_ppl[b]["damage"])
# block-index: numeric order
block_index_order = sorted(TARGET_BLOCKS)
# hardest-first: reverse of easiest-first
hardest_first = list(reversed(easiest_first))

orderings = {
    "easiest_first": easiest_first,
    "block_index": block_index_order,
    "hardest_first": hardest_first,
}

print("\nDerived orderings:", flush=True)
for name, order in orderings.items():
    damages = [f"b{b}({single_block_ppl[b]['damage']:+.2f})" for b in order]
    print(f"  {name:16s}: {order}  damages: {damages}", flush=True)

# ── Phase 2: run 3 orderings × 4 stages ──────────────────────────────────────
results_by_ordering = {}

for ordering_name, block_order in orderings.items():
    print(f"\n{'='*60}", flush=True)
    print(f"Ordering: {ordering_name}  →  {block_order}", flush=True)
    print(f"{'='*60}", flush=True)

    stages = []
    for n_blocks in range(1, len(block_order) + 1):
        active = block_order[:n_blocks]
        trainable_count = sum(
            compress_cache[b]["proj_w"].numel() + compress_cache[b]["proj_b"].numel()
            for b in active
        )
        print(f"\n  Stage {n_blocks}: {active}  trainable={trainable_count:,}", flush=True)

        seed_results = run_accumulation_stage(
            base_model=base_model,
            active_blocks=active,
            compress_cache=compress_cache,
            seeds=SEEDS,
            device=DEVICE,
            calib_ids_cpu=calib_ids_cpu,
            eval_ids=eval_ids,
        )
        mean_ppl, std_ppl = aggregate(seed_results)
        print(f"  Stage {n_blocks} result: PPL={mean_ppl:.2f} ± {std_ppl:.2f}", flush=True)

        stages.append({
            "stage": n_blocks,
            "blocks": active,
            "trainable_params": trainable_count,
            "best_mean": float(mean_ppl),
            "best_std": float(std_ppl),
            "seeds": seed_results,
        })

    results_by_ordering[ordering_name] = {
        "order": block_order,
        "stages": stages,
    }

# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("ORDERING SENSITIVITY RESULTS", flush=True)
print(f"{'='*60}", flush=True)
print(f"Float baseline PPL: {float_ppl:.2f}", flush=True)
print(f"{'Ordering':16s}  Stage1  Stage2  Stage3  Stage4")

stage4_ppls = {}
for name, res in results_by_ordering.items():
    ppls = [s["best_mean"] for s in res["stages"]]
    stage4_ppls[name] = ppls[-1]
    ppl_str = "  ".join(f"{p:6.2f}" for p in ppls)
    print(f"{name:16s}  {ppl_str}", flush=True)

print("\nSingle-block swap damage (no Adam):", flush=True)
for blk in TARGET_BLOCKS:
    print(f"  Block {blk}: PPL={single_block_ppl[blk]['ppl']:.2f}  "
          f"damage={single_block_ppl[blk]['damage']:+.2f}", flush=True)

best_ordering = min(stage4_ppls, key=stage4_ppls.get)
worst_ordering = max(stage4_ppls, key=stage4_ppls.get)
gap_pct = 100.0 * (stage4_ppls[worst_ordering] - stage4_ppls[best_ordering]) / stage4_ppls[worst_ordering]

print(f"\nStage-4 comparison:", flush=True)
for name, ppl in stage4_ppls.items():
    tag = " ← best" if name == best_ordering else (" ← worst" if name == worst_ordering else "")
    print(f"  {name:16s}: {ppl:.2f}{tag}", flush=True)
print(f"Gap: {stage4_ppls[worst_ordering]:.2f} - {stage4_ppls[best_ordering]:.2f} "
      f"= {stage4_ppls[worst_ordering] - stage4_ppls[best_ordering]:.2f} PPL "
      f"({gap_pct:.1f}%)", flush=True)

if gap_pct > 5.0:
    verdict = "CONFIRMED"
    verdict_reason = f"Stage-4 gap {gap_pct:.1f}% > 5% threshold: ordering matters."
elif gap_pct < 2.0:
    verdict = "REFUTED"
    verdict_reason = f"Stage-4 gap {gap_pct:.1f}% < 2% threshold: ordering doesn't matter."
else:
    verdict = "INCONCLUSIVE"
    verdict_reason = f"Stage-4 gap {gap_pct:.1f}% in 2-5% zone."

print(f"\nVERDICT: {verdict}  —  {verdict_reason}", flush=True)

# ── Save JSON ─────────────────────────────────────────────────────────────────
import os
result = {
    "cycle": 351,
    "item": 360,
    "verdict": verdict,
    "verdict_reason": verdict_reason,
    "float_ppl": float(float_ppl),
    "gap_pct": float(gap_pct),
    "stage4_ppls": stage4_ppls,
    "best_ordering": best_ordering,
    "worst_ordering": worst_ordering,
    "orderings": {
        name: {
            "order": res["order"],
            "stage_ppls": [s["best_mean"] for s in res["stages"]],
            "stage_stds": [s["best_std"] for s in res["stages"]],
            "stages": [{k: v for k, v in s.items() if k != "seeds"} for s in res["stages"]],
        }
        for name, res in results_by_ordering.items()
    },
    "single_block_damage": single_block_ppl,
    "config": {
        "TARGET_BLOCKS": TARGET_BLOCKS,
        "HIDDEN_FRACTION": HIDDEN_FRACTION,
        "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE, "N_STEPS": N_STEPS, "LR": LR,
        "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
    },
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved results to {OUTPUT_PATH}", flush=True)
