#!/usr/bin/env python3
"""Cycle 359 / Item 367 — Adapter training schedule: joint vs sequential vs curriculum.

Tests whether training schedule is the multi-block bottleneck.
Cycle 354 baseline: 3-block h=128 adapters joint training → PPL 48.53.
Hypothesis: sequential/curriculum training may compose better.

Schedules (all 3-block [b2, b1, b10], h=128, Adam lr=1e-4, 3 seeds):
  Joint:      all 3 adapters trained simultaneously, 2000 steps
  Sequential: train b2 (1000 steps), freeze, add b1 (1000 steps), freeze, add b10 (1000 steps)
  Curriculum: b2 alone (500 steps) → b2+b1 (500 steps) → b2+b1+b10 (1000 steps), unfrozen

Verdict:
  CONFIRMED if any schedule gives PPL < 47.0 (beating joint baseline by >1.5 PPL)
  REFUTED if all schedules within ±0.5 of joint (48.53 ± 0.5 = 48.03-49.03)
  INCONCLUSIVE otherwise
"""
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS = [2, 1, 10]  # same 3-block combo as c354
HIDDEN_FRACTION = 1 / 6  # → k=512

N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

LR = 1e-4
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
LOAD_SEED = 42
ADAPTER_HIDDEN = 128

# Step budgets per schedule
JOINT_STEPS = 2000                       # same as c354
SEQUENTIAL_STEPS_PER_BLOCK = 1000       # 1000 × 3 = 3000 total
CURRICULUM_STEPS = [500, 500, 1000]     # stage1, stage2, stage3 = 2000 total

FLOAT_BASELINE_REF = 53.895
JOINT_BASELINE_REF = 48.53              # cycle 354 result

CONFIRMED_LT = 47.0
REFUTED_BAND = 0.5                      # within ±0.5 of joint = same schedule

OUTPUT_PATH = "autoresearch/results/cycle_359_adapter_schedule.json"

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
eval_ids_list = [t.to(DEVICE) for t in all_ids[N_CALIB: N_CALIB + N_EVAL]]
print(f"Calibration: {len(calib_ids_cpu)} seqs, Eval: {len(eval_ids_list)} seqs", flush=True)


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


class CompressedGPT2MLPWithAdapter(nn.Module):
    """Compressed MLP (topk+lstsq) with optional MLP adapter on GELU hidden.

    output = compressed(x) + adapter(gelu_hidden)
    adapter: W2 @ relu(W1 @ h + b1) + b2  where h = gelu(c_fc(x))
    """

    def __init__(self, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias,
                 adapter_hidden=0):
        super().__init__()
        k = c_fc_weight.shape[0]
        self.register_buffer("fc_weight", c_fc_weight.contiguous())   # (k, 768)
        self.register_buffer("fc_bias", c_fc_bias.contiguous())       # (k,)
        self.proj_weight = nn.Parameter(c_proj_weight.contiguous())   # (768, k)
        self.proj_bias = nn.Parameter(c_proj_bias.contiguous())       # (768,)

        self.adapter_active = adapter_hidden > 0
        if self.adapter_active:
            scale1 = 1.0 / math.sqrt(k)
            scale2 = 1.0 / math.sqrt(adapter_hidden)
            self.adapter_W1 = nn.Parameter(torch.randn(adapter_hidden, k) * scale1)
            self.adapter_b1 = nn.Parameter(torch.zeros(adapter_hidden))
            self.adapter_W2 = nn.Parameter(torch.randn(768, adapter_hidden) * scale2)
            self.adapter_b2 = nn.Parameter(torch.zeros(768))

    def activate_adapter(self, hidden_size: int):
        """Add adapter to an already-constructed module (used by sequential/curriculum)."""
        k = self.fc_weight.shape[0]
        scale1 = 1.0 / math.sqrt(k)
        scale2 = 1.0 / math.sqrt(hidden_size)
        self.adapter_W1 = nn.Parameter(torch.randn(hidden_size, k) * scale1,
                                       requires_grad=True)
        self.adapter_b1 = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
        self.adapter_W2 = nn.Parameter(torch.randn(768, hidden_size) * scale2,
                                       requires_grad=True)
        self.adapter_b2 = nn.Parameter(torch.zeros(768), requires_grad=True)
        self.adapter_active = True

    def adapter_params(self):
        if not self.adapter_active:
            return []
        return [self.adapter_W1, self.adapter_b1, self.adapter_W2, self.adapter_b2]

    def freeze_adapter(self):
        if self.adapter_active:
            for p in self.adapter_params():
                p.requires_grad_(False)

    def forward(self, x):
        h = F.linear(x, self.fc_weight, self.fc_bias)
        h = F.gelu(h)
        out = F.linear(h, self.proj_weight, self.proj_bias)
        if self.adapter_active:
            a = F.relu(F.linear(h, self.adapter_W1, self.adapter_b1))
            out = out + F.linear(a, self.adapter_W2, self.adapter_b2)
        return out


# ── Float baseline ────────────────────────────────────────────────────────────
print("Computing float baseline PPL...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
float_ppl = eval_ppl(base_model, eval_ids_list)
print(f"Float PPL: {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from reference {FLOAT_BASELINE_REF} by > 0.5"
)

# ── Compress all blocks (once, reuse across schedules) ────────────────────────
print("\nCollecting calibration activations for all target blocks...", flush=True)
calib_dev = [t.to(DEVICE) for t in calib_ids_cpu]
block_acts = collect_block_activations(base_model, BLOCKS, calib_dev, DEVICE)

compress_cache = {}
for blk in BLOCKS:
    acts = block_acts[blk]
    hidden = acts["hidden"]
    output = acts["output"]
    preact = acts["preact"]

    hidden_dim = hidden.shape[1]
    k = max(1, int(round(hidden_dim * HIDDEN_FRACTION)))

    std_h = hidden.std(dim=0)
    mean_h = hidden.mean(dim=0)
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    deriv = (0.5 * (1.0 + torch.erf(preact * inv_sqrt_2))
             + preact * torch.exp(-0.5 * preact.square()) * inv_sqrt_2pi)
    mean_deriv = deriv.mean(dim=0)

    c_proj = base_model.transformer.h[blk].mlp.c_proj
    c_proj_weight = c_proj.weight.detach().cpu().float()   # Conv1D: (3072, 768)
    output_norms = c_proj_weight.norm(dim=1)               # (3072,)

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
    c_fc_bias_full = c_fc.bias.detach().cpu().float()

    fc_w = c_fc_weight_full.T[selected]   # (k, 768)
    fc_b = c_fc_bias_full[selected]       # (k,)

    hidden_sel = hidden[:, selected].float()
    design = torch.cat([hidden_sel, torch.ones(hidden_sel.shape[0], 1)], dim=1)
    solution = torch.linalg.lstsq(design, output.float()).solution
    proj_w = solution[:-1].T.contiguous()   # (768, k)
    proj_b = solution[-1].contiguous()      # (768,)

    with torch.no_grad():
        recon = hidden_sel @ proj_w.T + proj_b
        mse = F.mse_loss(recon, output).item()
        cos = F.cosine_similarity(recon.reshape(1, -1), output.reshape(1, -1)).item()

    compress_cache[blk] = {
        "fc_w": fc_w, "fc_b": fc_b,
        "proj_w": proj_w, "proj_b": proj_b,
        "selected": selected, "k": k,
        "mse": mse, "cos": cos,
    }
    print(f"Block {blk}: k={k}, dead={dead_mask.sum().item()}, "
          f"lstsq MSE={mse:.4f}, cos={cos:.6f}", flush=True)

del block_acts, base_model
if DEVICE == "mps":
    torch.mps.empty_cache()

# ── Phase 1 retrain (reproduce topk+lstsq baseline, shared across schedules) ──
print(f"\n{'='*60}", flush=True)
print("Phase 1: topk+lstsq 3-block proj retrain (shared baseline)", flush=True)
print(f"{'='*60}", flush=True)

torch.manual_seed(42)
random.seed(42)

model_p1 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_p1.eval()

comp_mods_p1 = {}
for blk in BLOCKS:
    c = compress_cache[blk]
    m = CompressedGPT2MLPWithAdapter(
        c_fc_weight=c["fc_w"].to(DEVICE, torch.float32),
        c_fc_bias=c["fc_b"].to(DEVICE, torch.float32),
        c_proj_weight=c["proj_w"].to(DEVICE, torch.float32),
        c_proj_bias=c["proj_b"].to(DEVICE, torch.float32),
        adapter_hidden=0,
    ).to(DEVICE)
    model_p1.transformer.h[blk].mlp = m
    comp_mods_p1[blk] = m

init_ppl_p1 = eval_ppl(model_p1, eval_ids_list)
print(f"Phase 1 init PPL: {init_ppl_p1:.4f}", flush=True)

for p in model_p1.parameters():
    p.requires_grad_(False)
trainable_p1 = []
for blk in BLOCKS:
    comp_mods_p1[blk].proj_weight.requires_grad_(True)
    comp_mods_p1[blk].proj_bias.requires_grad_(True)
    trainable_p1.extend([comp_mods_p1[blk].proj_weight, comp_mods_p1[blk].proj_bias])

print(f"Phase 1 trainable params: {sum(p.numel() for p in trainable_p1):,}", flush=True)

opt_p1 = torch.optim.Adam(trainable_p1, lr=LR)
calib_dev = [t.to(DEVICE) for t in calib_ids_cpu]
random.shuffle(calib_dev)

best_ppl_p1 = init_ppl_p1
best_state_p1 = {}
for blk in BLOCKS:
    best_state_p1[blk] = {
        "proj_w": comp_mods_p1[blk].proj_weight.detach().clone(),
        "proj_b": comp_mods_p1[blk].proj_bias.detach().clone(),
    }

step = 0
t0 = time.time()
while step < 1000:
    for ids in calib_dev:
        if step >= 1000:
            break
        model_p1.train()
        opt_p1.zero_grad()
        model_p1(ids, labels=ids).loss.backward()
        opt_p1.step()
        step += 1
        if step % 100 == 0:
            ppl_e = eval_ppl(model_p1, eval_ids_list)
            if ppl_e < best_ppl_p1:
                best_ppl_p1 = ppl_e
                for blk in BLOCKS:
                    best_state_p1[blk] = {
                        "proj_w": comp_mods_p1[blk].proj_weight.detach().clone(),
                        "proj_b": comp_mods_p1[blk].proj_bias.detach().clone(),
                    }
            if step % 500 == 0:
                print(f"  Phase1 step {step}: ppl={ppl_e:.2f} best={best_ppl_p1:.2f}", flush=True)

# restore best proj weights
with torch.no_grad():
    for blk in BLOCKS:
        comp_mods_p1[blk].proj_weight.data.copy_(best_state_p1[blk]["proj_w"])
        comp_mods_p1[blk].proj_bias.data.copy_(best_state_p1[blk]["proj_b"])
phase1_ppl = eval_ppl(model_p1, eval_ids_list)
print(f"Phase 1 baseline PPL: {phase1_ppl:.4f} ({time.time()-t0:.0f}s)", flush=True)

# Save retrained proj params
retrained_proj = {}
for blk in BLOCKS:
    retrained_proj[blk] = {
        "proj_w": best_state_p1[blk]["proj_w"].cpu().clone(),
        "proj_b": best_state_p1[blk]["proj_b"].cpu().clone(),
    }

del model_p1, comp_mods_p1, opt_p1
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Training loop helper ──────────────────────────────────────────────────────
def run_steps(model, trainable_params, n_steps, calib_dev, eval_ids,
              eval_every=100, label=""):
    """Run n_steps of Adam training, return (best_ppl, history)."""
    opt = torch.optim.Adam(trainable_params, lr=LR)
    shuffled = list(calib_dev)
    random.shuffle(shuffled)

    init_ppl = eval_ppl(model, eval_ids)
    best_ppl = init_ppl
    best_state = {id(p): p.detach().clone() for p in trainable_params}
    history = [{"step": 0, "eval_ppl": float(init_ppl)}]

    step = 0
    while step < n_steps:
        for ids in shuffled:
            if step >= n_steps:
                break
            model.train()
            opt.zero_grad()
            model(ids, labels=ids).loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0:
                ppl_e = eval_ppl(model, eval_ids)
                history.append({"step": step, "eval_ppl": float(ppl_e)})
                if ppl_e < best_ppl:
                    best_ppl = ppl_e
                    best_state = {id(p): p.detach().clone() for p in trainable_params}
                if step % 500 == 0:
                    print(f"  {label} step {step}/{n_steps}: ppl={ppl_e:.2f} best={best_ppl:.2f}",
                          flush=True)

    # restore best
    with torch.no_grad():
        for p in trainable_params:
            p.data.copy_(best_state[id(p)])

    final_ppl = eval_ppl(model, eval_ids)
    return best_ppl, final_ppl, history


# ═══════════════════════════════════════════════════════════════════════════════
# Schedule 1: JOINT — train all 3 adapters simultaneously (2000 steps)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print("SCHEDULE: JOINT (2000 steps, all 3 adapters simultaneously)", flush=True)
print(f"{'='*60}", flush=True)

joint_results = []
for seed in SEEDS:
    torch.manual_seed(seed)
    random.seed(seed)

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()

    comp_mods = {}
    for blk in BLOCKS:
        c = compress_cache[blk]
        m = CompressedGPT2MLPWithAdapter(
            c_fc_weight=c["fc_w"].to(DEVICE, torch.float32),
            c_fc_bias=c["fc_b"].to(DEVICE, torch.float32),
            c_proj_weight=retrained_proj[blk]["proj_w"].to(DEVICE, torch.float32),
            c_proj_bias=retrained_proj[blk]["proj_b"].to(DEVICE, torch.float32),
            adapter_hidden=ADAPTER_HIDDEN,
        ).to(DEVICE)
        model.transformer.h[blk].mlp = m
        comp_mods[blk] = m

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for blk in BLOCKS:
        comp_mods[blk].proj_weight.requires_grad_(True)
        comp_mods[blk].proj_bias.requires_grad_(True)
        for p in comp_mods[blk].adapter_params():
            p.requires_grad_(True)
        trainable.extend([comp_mods[blk].proj_weight, comp_mods[blk].proj_bias]
                         + comp_mods[blk].adapter_params())

    calib_dev_s = [t.to(DEVICE) for t in calib_ids_cpu]
    t0 = time.time()
    best_ppl, final_ppl, history = run_steps(
        model, trainable, JOINT_STEPS, calib_dev_s, eval_ids_list,
        label=f"joint seed={seed}"
    )
    dt = time.time() - t0
    print(f"  joint seed={seed}: best={best_ppl:.2f} final={final_ppl:.2f} ({dt:.0f}s)", flush=True)
    joint_results.append({
        "seed": seed, "best_ppl": float(best_ppl), "final_ppl": float(final_ppl),
        "history": history,
    })

    del model, comp_mods
    if DEVICE == "mps":
        torch.mps.empty_cache()

joint_ppls = [r["best_ppl"] for r in joint_results]
joint_mean = sum(joint_ppls) / len(joint_ppls)
joint_std = math.sqrt(sum((x - joint_mean)**2 for x in joint_ppls) / len(joint_ppls))
print(f"\nJOINT: mean={joint_mean:.2f} ± {joint_std:.2f} "
      f"(min={min(joint_ppls):.2f}, max={max(joint_ppls):.2f})", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Schedule 2: SEQUENTIAL — train blocks one at a time, freezing previous
# Order: b2 (1000 steps) → freeze b2 → b1 (1000 steps) → freeze b1 → b10 (1000 steps)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print("SCHEDULE: SEQUENTIAL (1000 steps per block, freeze previous)", flush=True)
print(f"{'='*60}", flush=True)

sequential_results = []
for seed in SEEDS:
    torch.manual_seed(seed)
    random.seed(seed)

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()

    comp_mods = {}
    # Start with just compressed mlp (no adapters yet)
    for blk in BLOCKS:
        c = compress_cache[blk]
        m = CompressedGPT2MLPWithAdapter(
            c_fc_weight=c["fc_w"].to(DEVICE, torch.float32),
            c_fc_bias=c["fc_b"].to(DEVICE, torch.float32),
            c_proj_weight=retrained_proj[blk]["proj_w"].to(DEVICE, torch.float32),
            c_proj_bias=retrained_proj[blk]["proj_b"].to(DEVICE, torch.float32),
            adapter_hidden=0,
        ).to(DEVICE)
        model.transformer.h[blk].mlp = m
        comp_mods[blk] = m

    for p in model.parameters():
        p.requires_grad_(False)

    stage_histories = []
    calib_dev_s = [t.to(DEVICE) for t in calib_ids_cpu]
    t0 = time.time()

    for i, blk in enumerate(BLOCKS):
        # activate adapter for this block
        comp_mods[blk].activate_adapter(ADAPTER_HIDDEN)
        comp_mods[blk] = comp_mods[blk].to(DEVICE)

        # trainable: only this block's adapter params (proj already frozen)
        trainable = comp_mods[blk].adapter_params()
        for p in trainable:
            p.requires_grad_(True)

        init_stage_ppl = eval_ppl(model, eval_ids_list)
        best_ppl_s, final_ppl_s, hist_s = run_steps(
            model, trainable, SEQUENTIAL_STEPS_PER_BLOCK, calib_dev_s, eval_ids_list,
            label=f"seq seed={seed} b{blk}"
        )
        print(f"  seq seed={seed} stage {i+1} (b{blk}): "
              f"init={init_stage_ppl:.2f} best={best_ppl_s:.2f}", flush=True)
        stage_histories.append({
            "block": blk, "stage": i + 1,
            "init_ppl": float(init_stage_ppl),
            "best_ppl": float(best_ppl_s),
            "history": hist_s,
        })

        # freeze this block's adapter before moving on
        comp_mods[blk].freeze_adapter()

    final_ppl = eval_ppl(model, eval_ids_list)
    dt = time.time() - t0
    print(f"  seq seed={seed}: final PPL={final_ppl:.2f} ({dt:.0f}s)", flush=True)
    sequential_results.append({
        "seed": seed,
        "best_ppl": float(stage_histories[-1]["best_ppl"]),
        "final_ppl": float(final_ppl),
        "stages": stage_histories,
    })

    del model, comp_mods
    if DEVICE == "mps":
        torch.mps.empty_cache()

seq_ppls = [r["final_ppl"] for r in sequential_results]
seq_mean = sum(seq_ppls) / len(seq_ppls)
seq_std = math.sqrt(sum((x - seq_mean)**2 for x in seq_ppls) / len(seq_ppls))
print(f"\nSEQUENTIAL: mean={seq_mean:.2f} ± {seq_std:.2f} "
      f"(min={min(seq_ppls):.2f}, max={max(seq_ppls):.2f})", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Schedule 3: CURRICULUM — start with 1 block, add more gradually, all unfrozen
# Stage 1: b2 only (500 steps)
# Stage 2: b2+b1 (500 steps, both unfrozen)
# Stage 3: b2+b1+b10 (1000 steps, all 3 unfrozen)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print("SCHEDULE: CURRICULUM (500+500+1000 steps, gradually add blocks, all unfrozen)", flush=True)
print(f"{'='*60}", flush=True)

curriculum_results = []
for seed in SEEDS:
    torch.manual_seed(seed)
    random.seed(seed)

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()

    comp_mods = {}
    for blk in BLOCKS:
        c = compress_cache[blk]
        m = CompressedGPT2MLPWithAdapter(
            c_fc_weight=c["fc_w"].to(DEVICE, torch.float32),
            c_fc_bias=c["fc_b"].to(DEVICE, torch.float32),
            c_proj_weight=retrained_proj[blk]["proj_w"].to(DEVICE, torch.float32),
            c_proj_bias=retrained_proj[blk]["proj_b"].to(DEVICE, torch.float32),
            adapter_hidden=0,
        ).to(DEVICE)
        model.transformer.h[blk].mlp = m
        comp_mods[blk] = m

    for p in model.parameters():
        p.requires_grad_(False)

    stage_histories = []
    calib_dev_s = [t.to(DEVICE) for t in calib_ids_cpu]
    active_blocks = []
    t0 = time.time()

    for i, (blk, n_stage_steps) in enumerate(zip(BLOCKS, CURRICULUM_STEPS)):
        # activate adapter for newly added block
        comp_mods[blk].activate_adapter(ADAPTER_HIDDEN)
        comp_mods[blk] = comp_mods[blk].to(DEVICE)
        active_blocks.append(blk)

        # trainable: ALL active adapters (none frozen)
        trainable = []
        for ab in active_blocks:
            for p in comp_mods[ab].adapter_params():
                p.requires_grad_(True)
            trainable.extend(comp_mods[ab].adapter_params())

        init_stage_ppl = eval_ppl(model, eval_ids_list)
        best_ppl_s, final_ppl_s, hist_s = run_steps(
            model, trainable, n_stage_steps, calib_dev_s, eval_ids_list,
            label=f"curr seed={seed} stage{i+1}(b{blk})"
        )
        print(f"  curr seed={seed} stage {i+1} (add b{blk}, {len(active_blocks)} blocks active): "
              f"init={init_stage_ppl:.2f} best={best_ppl_s:.2f}", flush=True)
        stage_histories.append({
            "block_added": blk, "stage": i + 1,
            "active_blocks": list(active_blocks),
            "n_steps": n_stage_steps,
            "init_ppl": float(init_stage_ppl),
            "best_ppl": float(best_ppl_s),
            "history": hist_s,
        })

    final_ppl = eval_ppl(model, eval_ids_list)
    dt = time.time() - t0
    print(f"  curr seed={seed}: final PPL={final_ppl:.2f} ({dt:.0f}s)", flush=True)
    curriculum_results.append({
        "seed": seed,
        "best_ppl": float(stage_histories[-1]["best_ppl"]),
        "final_ppl": float(final_ppl),
        "stages": stage_histories,
    })

    del model, comp_mods
    if DEVICE == "mps":
        torch.mps.empty_cache()

curr_ppls = [r["final_ppl"] for r in curriculum_results]
curr_mean = sum(curr_ppls) / len(curr_ppls)
curr_std = math.sqrt(sum((x - curr_mean)**2 for x in curr_ppls) / len(curr_ppls))
print(f"\nCURRICULUM: mean={curr_mean:.2f} ± {curr_std:.2f} "
      f"(min={min(curr_ppls):.2f}, max={max(curr_ppls):.2f})", flush=True)


# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("ADAPTER TRAINING SCHEDULE COMPARISON", flush=True)
print(f"{'='*60}", flush=True)

schedule_means = {
    "joint": joint_mean,
    "sequential": seq_mean,
    "curriculum": curr_mean,
}
best_schedule = min(schedule_means, key=lambda k: schedule_means[k])
best_mean = schedule_means[best_schedule]
worst_mean = max(schedule_means.values())
ppl_gap = worst_mean - best_mean

print(f"{'Schedule':>12s}  {'mean PPL':>10s}  {'std':>6s}", flush=True)
print(f"{'joint':>12s}  {joint_mean:>10.2f}  {joint_std:>6.2f}", flush=True)
print(f"{'sequential':>12s}  {seq_mean:>10.2f}  {seq_std:>6.2f}", flush=True)
print(f"{'curriculum':>12s}  {curr_mean:>10.2f}  {curr_std:>6.2f}", flush=True)
print(f"\nBest schedule: {best_schedule} (PPL={best_mean:.2f})", flush=True)
print(f"PPL gap (worst-best): {ppl_gap:.2f}", flush=True)
print(f"c354 joint ref: {JOINT_BASELINE_REF}", flush=True)
print(f"Float PPL: {float_ppl:.4f}", flush=True)

if best_mean < CONFIRMED_LT:
    verdict = "CONFIRMED"
elif abs(joint_mean - seq_mean) <= REFUTED_BAND and abs(joint_mean - curr_mean) <= REFUTED_BAND:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nVERDICT: {verdict}", flush=True)

# ── Save JSON ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
result = {
    "cycle": 359,
    "item": 367,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "phase1_ppl": float(phase1_ppl),
    "joint_baseline_ref": JOINT_BASELINE_REF,
    "confirmed_lt": CONFIRMED_LT,
    "best_schedule": best_schedule,
    "best_mean_ppl": float(best_mean),
    "ppl_gap": float(ppl_gap),
    "schedules": {
        "joint": {
            "mean": float(joint_mean), "std": float(joint_std),
            "min": float(min(joint_ppls)), "max": float(max(joint_ppls)),
            "seeds": joint_results,
        },
        "sequential": {
            "mean": float(seq_mean), "std": float(seq_std),
            "min": float(min(seq_ppls)), "max": float(max(seq_ppls)),
            "seeds": sequential_results,
        },
        "curriculum": {
            "mean": float(curr_mean), "std": float(curr_std),
            "min": float(min(curr_ppls)), "max": float(max(curr_ppls)),
            "seeds": curriculum_results,
        },
    },
    "config": {
        "BLOCKS": BLOCKS, "ADAPTER_HIDDEN": ADAPTER_HIDDEN,
        "HIDDEN_FRACTION": HIDDEN_FRACTION, "LR": LR,
        "SEEDS": SEEDS, "DEVICE": DEVICE,
        "JOINT_STEPS": JOINT_STEPS,
        "SEQUENTIAL_STEPS_PER_BLOCK": SEQUENTIAL_STEPS_PER_BLOCK,
        "CURRICULUM_STEPS": CURRICULUM_STEPS,
        "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved results to {OUTPUT_PATH}", flush=True)
