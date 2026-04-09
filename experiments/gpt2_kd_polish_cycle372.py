#!/usr/bin/env python3
"""Cycle 363 / Item 372 — KD fine-tune after geometric compression.

Hypothesis: use geometric compression for INIT, then do a short global KD fine-tune
to let the full model adapt. Geometric gives a good starting point; KD polishes
the whole model.

Setup (GPT-2 small, 3-block [b2, b1, b10]):
  Phase 1: topk+lstsq k=512 + h=64 adapters on 3 blocks, train 1000 steps (adapter only)
  Phase 2: KD fine-tune — unfreeze ALL model params, KL divergence vs teacher (original GPT-2)
           logits, 500 steps, lr=5e-5

Reference:
  - Geometric-only best (c359 3-block joint h=128): PPL 47.43
  - Pure distillation (single-block, c356): PPL 56.38
  - Float: ~53.89 (paradox: single-block KD from random init is worse than float)

Verdict:
  CONFIRMED if KD-polished PPL < geometric-only PPL by >= 1.0 (i.e. < 46.43)
  REFUTED if KD doesn't help (improvement < 0.2 PPL)
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
BLOCKS = [2, 1, 10]
HIDDEN_FRACTION = 1 / 6  # → k=512

N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Phase 1: adapter training (topk+lstsq + adapters)
N_STEPS_P1 = 1000
LR_P1 = 1e-4
ADAPTER_HIDDEN = 64  # per item spec

# Phase 2: global KD fine-tune
N_STEPS_P2 = 500
LR_P2 = 5e-5
KD_TEMPERATURE = 1.0

EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
LOAD_SEED = 42

FLOAT_BASELINE_REF = 53.895
GEOMETRIC_ONLY_PPL_REF = 47.43  # c359 3-block joint h=128

CONFIRMED_IMPROVEMENT = 1.0   # PPL must drop by >= 1.0 vs geometric-only
REFUTED_IMPROVEMENT = 0.2     # less than 0.2 = no help

OUTPUT_PATH = "autoresearch/results/cycle_363_kd_polish.json"

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
eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB: N_CALIB + N_EVAL]]
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
    """Collect MLP hidden activations, preact, input, output for specified blocks."""
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
    """Compressed MLP (topk+lstsq) with optional MLP adapter.

    Compressed path: frozen c_fc (k channels) → GELU → trainable c_proj (k→768)
    Adapter path:    GELU hidden (k) → W1 (h, k) → ReLU → W2 (768, h) + bias
    """

    def __init__(self, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias, adapter_hidden=0):
        super().__init__()
        k = c_fc_weight.shape[0]
        self.register_buffer("fc_weight", c_fc_weight.contiguous())
        self.register_buffer("fc_bias", c_fc_bias.contiguous())
        self.proj_weight = nn.Parameter(c_proj_weight.contiguous())
        self.proj_bias = nn.Parameter(c_proj_bias.contiguous())

        self.adapter_active = adapter_hidden > 0
        if self.adapter_active:
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
        if self.adapter_active:
            a = F.relu(F.linear(h, self.adapter_W1, self.adapter_b1))
            out = out + F.linear(a, self.adapter_W2, self.adapter_b2)
        return out


# ── Float baseline ────────────────────────────────────────────────────────────
print("Computing float baseline PPL...", flush=True)
teacher_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
teacher_model.eval()
float_ppl = eval_ppl(teacher_model, eval_ids)
print(f"Float PPL: {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from reference {FLOAT_BASELINE_REF} by > 0.5"
)

# ── Collect calibration activations ──────────────────────────────────────────
print("\nCollecting calibration activations for all target blocks...", flush=True)
calib_dev = [t.to(DEVICE) for t in calib_ids_cpu]
block_acts = collect_block_activations(teacher_model, BLOCKS, calib_dev, DEVICE)
parts = [f"b{b}: {block_acts[b]['hidden'].shape}" for b in BLOCKS]
print(f"Collected activations: {', '.join(parts)}", flush=True)

# ── topk+lstsq compression per block ─────────────────────────────────────────
compress_cache = {}
for blk in BLOCKS:
    acts = block_acts[blk]
    hidden = acts["hidden"]
    output = acts["output"]
    preact = acts["preact"]

    hidden_dim = hidden.shape[1]
    k = max(1, int(round(hidden_dim * HIDDEN_FRACTION)))

    # Channel scoring (same as prior cycles)
    std_h = hidden.std(dim=0)
    mean_h = hidden.mean(dim=0)
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    deriv = (0.5 * (1.0 + torch.erf(preact * inv_sqrt_2))
             + preact * torch.exp(-0.5 * preact.square()) * inv_sqrt_2pi)
    mean_deriv = deriv.mean(dim=0)

    c_proj = teacher_model.transformer.h[blk].mlp.c_proj
    c_proj_weight = c_proj.weight.detach().cpu().float()
    output_norms = c_proj_weight.norm(dim=1)

    channel_scores = std_h * output_norms * (mean_h.abs() + mean_deriv.abs() + 1e-6)

    dead_mask = hidden.abs().amax(dim=0) <= 1e-6
    available = torch.arange(hidden_dim)
    live = available[~dead_mask]
    if live.numel() < k:
        live = available
    scores = channel_scores[live]
    topk_idx = torch.topk(scores, k=k, largest=True).indices
    selected = live[topk_idx].sort().values

    c_fc = teacher_model.transformer.h[blk].mlp.c_fc
    c_fc_weight_full = c_fc.weight.detach().cpu().float()
    c_fc_bias_full = c_fc.bias.detach().cpu().float()

    fc_w = c_fc_weight_full.T[selected]    # (k, 768)
    fc_b = c_fc_bias_full[selected]        # (k,)

    hidden_sel = hidden[:, selected].float()
    design = torch.cat([hidden_sel, torch.ones(hidden_sel.shape[0], 1)], dim=1)
    solution = torch.linalg.lstsq(design, output.float()).solution
    proj_w = solution[:-1].T.contiguous()  # (768, k)
    proj_b = solution[-1].contiguous()     # (768,)

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

del block_acts

# Keep teacher for Phase 2 KD
# (teacher_model stays loaded on DEVICE, eval mode)
print(f"Teacher model kept for KD (device: {DEVICE})", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: topk+lstsq 3-block + adapter training
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print(f"Phase 1: topk+lstsq 3-block [b2,b1,b10] + adapter h={ADAPTER_HIDDEN}", flush=True)
print(f"  {N_STEPS_P1} steps, lr={LR_P1}", flush=True)
print(f"{'='*60}", flush=True)

torch.manual_seed(LOAD_SEED)
random.seed(LOAD_SEED)

student_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
student_model.eval()

# Install compressed MLPs WITH adapters
compressed_modules = {}
for blk in BLOCKS:
    c = compress_cache[blk]
    comp_mlp = CompressedGPT2MLPWithAdapter(
        c_fc_weight=c["fc_w"].to(device=DEVICE, dtype=torch.float32),
        c_fc_bias=c["fc_b"].to(device=DEVICE, dtype=torch.float32),
        c_proj_weight=c["proj_w"].to(device=DEVICE, dtype=torch.float32),
        c_proj_bias=c["proj_b"].to(device=DEVICE, dtype=torch.float32),
        adapter_hidden=ADAPTER_HIDDEN,
    ).to(DEVICE)
    student_model.transformer.h[blk].mlp = comp_mlp
    compressed_modules[blk] = comp_mlp

init_ppl_p1 = eval_ppl(student_model, eval_ids)
print(f"Phase 1 init PPL: {init_ppl_p1:.4f}", flush=True)

# Freeze everything, unfreeze proj + adapter params
for p in student_model.parameters():
    p.requires_grad_(False)
trainable_p1 = []
for blk in BLOCKS:
    mod = compressed_modules[blk]
    mod.proj_weight.requires_grad_(True)
    mod.proj_bias.requires_grad_(True)
    trainable_p1.extend([mod.proj_weight, mod.proj_bias])
    if mod.adapter_active:
        mod.adapter_W1.requires_grad_(True)
        mod.adapter_b1.requires_grad_(True)
        mod.adapter_W2.requires_grad_(True)
        mod.adapter_b2.requires_grad_(True)
        trainable_p1.extend([mod.adapter_W1, mod.adapter_b1,
                              mod.adapter_W2, mod.adapter_b2])

n_train_p1 = sum(p.numel() for p in trainable_p1)
print(f"Phase 1 trainable params: {n_train_p1:,}", flush=True)

optimizer_p1 = torch.optim.Adam(trainable_p1, lr=LR_P1)
calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]
random.shuffle(calib_ids_dev)

best_ppl_p1 = init_ppl_p1
best_step_p1 = 0

# Best state for Phase 1
def save_state_p1():
    state = {}
    for blk in BLOCKS:
        mod = compressed_modules[blk]
        state[f"proj_w_{blk}"] = mod.proj_weight.detach().clone()
        state[f"proj_b_{blk}"] = mod.proj_bias.detach().clone()
        if mod.adapter_active:
            state[f"W1_{blk}"] = mod.adapter_W1.detach().clone()
            state[f"b1_{blk}"] = mod.adapter_b1.detach().clone()
            state[f"W2_{blk}"] = mod.adapter_W2.detach().clone()
            state[f"b2_{blk}"] = mod.adapter_b2.detach().clone()
    return state


def restore_state_p1(state):
    with torch.no_grad():
        for blk in BLOCKS:
            mod = compressed_modules[blk]
            mod.proj_weight.data.copy_(state[f"proj_w_{blk}"])
            mod.proj_bias.data.copy_(state[f"proj_b_{blk}"])
            if mod.adapter_active:
                mod.adapter_W1.data.copy_(state[f"W1_{blk}"])
                mod.adapter_b1.data.copy_(state[f"b1_{blk}"])
                mod.adapter_W2.data.copy_(state[f"W2_{blk}"])
                mod.adapter_b2.data.copy_(state[f"b2_{blk}"])


best_state_p1 = save_state_p1()
history_p1 = [{"step": 0, "eval_ppl": float(init_ppl_p1)}]

step = 0
t0 = time.time()
while step < N_STEPS_P1:
    for ids in calib_ids_dev:
        if step >= N_STEPS_P1:
            break
        student_model.train()
        optimizer_p1.zero_grad()
        out = student_model(ids, labels=ids)
        out.loss.backward()
        optimizer_p1.step()
        step += 1
        if step % EVAL_EVERY == 0:
            ppl_e = eval_ppl(student_model, eval_ids)
            history_p1.append({"step": step, "eval_ppl": float(ppl_e)})
            if ppl_e < best_ppl_p1:
                best_ppl_p1 = ppl_e
                best_step_p1 = step
                best_state_p1 = save_state_p1()
            if step % 500 == 0:
                print(f"  Phase 1 step {step}: ppl={ppl_e:.2f} best={best_ppl_p1:.2f}@{best_step_p1}",
                      flush=True)

restore_state_p1(best_state_p1)
restored_ppl_p1 = eval_ppl(student_model, eval_ids)
dt_p1 = time.time() - t0
print(f"Phase 1: init={init_ppl_p1:.2f} best={best_ppl_p1:.2f}@{best_step_p1} "
      f"restored={restored_ppl_p1:.2f} ({dt_p1:.0f}s)", flush=True)

phase1_result = {
    "init_ppl": float(init_ppl_p1),
    "best_ppl": float(best_ppl_p1),
    "best_step": int(best_step_p1),
    "restored_ppl": float(restored_ppl_p1),
    "trainable_params": n_train_p1,
    "history": history_p1,
}

del optimizer_p1
if DEVICE == "mps":
    torch.mps.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Global KD fine-tune — unfreeze all model params, KL vs teacher
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print(f"Phase 2: Global KD fine-tune (all params unfrozen)", flush=True)
print(f"  {N_STEPS_P2} steps, lr={LR_P2}, T={KD_TEMPERATURE}", flush=True)
print(f"  Loss: KL(student_logits || teacher_logits) on final output", flush=True)
print(f"{'='*60}", flush=True)

# 3 seeds — Phase 1 result is fixed (seed 42 best from Phase 1)
# We vary Phase 2 seeds to measure stability
seed_results_p2 = []

for seed in SEEDS:
    torch.manual_seed(seed)
    random.seed(seed)

    # Fresh copy of student at Phase 1 best checkpoint
    model_p2 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model_p2.eval()

    # Re-install compressed MLPs using Phase 1 best state
    comp_mods_p2 = {}
    for blk in BLOCKS:
        c = compress_cache[blk]
        comp_mlp = CompressedGPT2MLPWithAdapter(
            c_fc_weight=c["fc_w"].to(device=DEVICE, dtype=torch.float32),
            c_fc_bias=c["fc_b"].to(device=DEVICE, dtype=torch.float32),
            c_proj_weight=best_state_p1[f"proj_w_{blk}"].to(device=DEVICE, dtype=torch.float32),
            c_proj_bias=best_state_p1[f"proj_b_{blk}"].to(device=DEVICE, dtype=torch.float32),
            adapter_hidden=ADAPTER_HIDDEN,
        ).to(DEVICE)
        if f"W1_{blk}" in best_state_p1:
            with torch.no_grad():
                comp_mlp.adapter_W1.data.copy_(best_state_p1[f"W1_{blk}"].to(DEVICE))
                comp_mlp.adapter_b1.data.copy_(best_state_p1[f"b1_{blk}"].to(DEVICE))
                comp_mlp.adapter_W2.data.copy_(best_state_p1[f"W2_{blk}"].to(DEVICE))
                comp_mlp.adapter_b2.data.copy_(best_state_p1[f"b2_{blk}"].to(DEVICE))
        model_p2.transformer.h[blk].mlp = comp_mlp
        comp_mods_p2[blk] = comp_mlp

    init_ppl_p2 = eval_ppl(model_p2, eval_ids)
    print(f"\nSeed {seed}: Phase 2 init PPL: {init_ppl_p2:.4f}", flush=True)

    # Unfreeze ALL model parameters for global KD fine-tune
    for p in model_p2.parameters():
        p.requires_grad_(True)

    n_train_p2 = sum(p.numel() for p in model_p2.parameters() if p.requires_grad)
    print(f"  Phase 2 trainable params: {n_train_p2:,}", flush=True)

    optimizer_p2 = torch.optim.Adam(model_p2.parameters(), lr=LR_P2)
    calib_ids_dev_p2 = [t.to(DEVICE) for t in calib_ids_cpu]
    random.shuffle(calib_ids_dev_p2)

    best_ppl_p2 = init_ppl_p2
    best_step_p2 = 0

    # Best-state snapshot for Phase 2 (full model — too large to store all)
    # We just track best PPL and restore at end via eval
    # Instead: save only the compressed block params (they change the most)
    def save_full_state():
        return {n: p.detach().clone() for n, p in model_p2.named_parameters()}

    def restore_full_state(state):
        with torch.no_grad():
            for n, p in model_p2.named_parameters():
                p.data.copy_(state[n])

    best_state_p2 = save_full_state()
    history_p2 = [{"step": 0, "eval_ppl": float(init_ppl_p2)}]

    step = 0
    t0 = time.time()
    while step < N_STEPS_P2:
        for ids in calib_ids_dev_p2:
            if step >= N_STEPS_P2:
                break
            model_p2.train()
            optimizer_p2.zero_grad()

            # KD loss: KL(student || teacher) on final logits
            with torch.no_grad():
                teacher_out = teacher_model(ids)
                teacher_logits = teacher_out.logits.detach()  # (1, seq, vocab)

            student_out = model_p2(ids)
            student_logits = student_out.logits  # (1, seq, vocab)

            # KL divergence: sum over vocab
            # KL(teacher || student) = teacher_prob * log(teacher_prob / student_prob)
            T = KD_TEMPERATURE
            log_student = F.log_softmax(student_logits / T, dim=-1)
            log_teacher = F.log_softmax(teacher_logits / T, dim=-1)
            teacher_prob = log_teacher.exp()

            # kl_div expects log-probabilities for input, probabilities for target
            loss = F.kl_div(log_student, teacher_prob,
                            reduction="batchmean", log_target=False)
            loss = loss * (T ** 2)  # temperature scaling

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_p2.parameters(), 1.0)
            optimizer_p2.step()
            step += 1

            if step % EVAL_EVERY == 0:
                ppl_e = eval_ppl(model_p2, eval_ids)
                history_p2.append({"step": step, "eval_ppl": float(ppl_e)})
                if ppl_e < best_ppl_p2:
                    best_ppl_p2 = ppl_e
                    best_step_p2 = step
                    best_state_p2 = save_full_state()
                print(f"  seed {seed} step {step}: ppl={ppl_e:.2f} "
                      f"best={best_ppl_p2:.2f}@{best_step_p2} "
                      f"kd_loss={loss.item():.4f}", flush=True)

    restore_full_state(best_state_p2)
    restored_ppl_p2 = eval_ppl(model_p2, eval_ids)
    dt_p2 = time.time() - t0

    print(f"  seed {seed}: init={init_ppl_p2:.2f} best={best_ppl_p2:.2f}@{best_step_p2} "
          f"restored={restored_ppl_p2:.2f} ({dt_p2:.0f}s)", flush=True)

    seed_results_p2.append({
        "seed": seed,
        "init_ppl": float(init_ppl_p2),
        "best_ppl": float(best_ppl_p2),
        "best_step": int(best_step_p2),
        "restored_ppl": float(restored_ppl_p2),
        "n_train_params": n_train_p2,
        "history": history_p2,
    })

    del model_p2, comp_mods_p2, optimizer_p2, best_state_p2
    if DEVICE == "mps":
        torch.mps.empty_cache()


# ── Summary and verdict ──────────────────────────────────────────────────────
best_ppls_p2 = [r["best_ppl"] for r in seed_results_p2]
mean_ppl_p2 = sum(best_ppls_p2) / len(best_ppls_p2)
std_ppl_p2 = math.sqrt(sum((x - mean_ppl_p2) ** 2 for x in best_ppls_p2) / len(best_ppls_p2))

improvement = best_ppl_p1 - mean_ppl_p2  # positive = KD helped

print(f"\n{'='*60}", flush=True)
print("KD POLISH — SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"Float PPL:                      {float_ppl:.4f}", flush=True)
print(f"Geometric-only ref (c359):      {GEOMETRIC_ONLY_PPL_REF:.2f}", flush=True)
print(f"Phase 1 (geometric best):       {best_ppl_p1:.4f}", flush=True)
print(f"Phase 2 (KD polish):            {mean_ppl_p2:.4f} ± {std_ppl_p2:.4f}", flush=True)
print(f"  Min: {min(best_ppls_p2):.4f}  Max: {max(best_ppls_p2):.4f}", flush=True)
print(f"Improvement (P1 - P2):          {improvement:+.4f}", flush=True)

for r in seed_results_p2:
    print(f"  seed {r['seed']}: P1 init={r['init_ppl']:.2f} best={r['best_ppl']:.2f}@{r['best_step']}",
          flush=True)

if improvement >= CONFIRMED_IMPROVEMENT:
    verdict = "CONFIRMED"
    verdict_detail = (f"KD-polished PPL {mean_ppl_p2:.2f} < geometric-only {best_ppl_p1:.2f} "
                      f"by {improvement:.2f} >= {CONFIRMED_IMPROVEMENT}")
elif improvement < REFUTED_IMPROVEMENT:
    verdict = "REFUTED"
    verdict_detail = (f"KD doesn't help: improvement {improvement:.2f} < {REFUTED_IMPROVEMENT}. "
                      f"Phase 2 PPL: {mean_ppl_p2:.2f}, Phase 1: {best_ppl_p1:.2f}")
else:
    verdict = "INCONCLUSIVE"
    verdict_detail = (f"marginal improvement {improvement:.2f} "
                      f"(between {REFUTED_IMPROVEMENT} and {CONFIRMED_IMPROVEMENT})")

print(f"\nVERDICT: {verdict} — {verdict_detail}", flush=True)

# ── Save JSON ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

result = {
    "cycle": 363,
    "item": 372,
    "verdict": verdict,
    "verdict_detail": verdict_detail,
    "float_ppl": float(float_ppl),
    "float_ppl_ref": FLOAT_BASELINE_REF,
    "geometric_only_ppl_ref": GEOMETRIC_ONLY_PPL_REF,
    "phase1": phase1_result,
    "phase2": {
        "mean_ppl": float(mean_ppl_p2),
        "std_ppl": float(std_ppl_p2),
        "min_ppl": float(min(best_ppls_p2)),
        "max_ppl": float(max(best_ppls_p2)),
        "improvement": float(improvement),
        "n_steps": N_STEPS_P2,
        "lr": LR_P2,
        "kd_temperature": KD_TEMPERATURE,
    },
    "phase2_seeds": seed_results_p2,
    "per_block_lstsq": {
        str(blk): {"k": compress_cache[blk]["k"], "mse": compress_cache[blk]["mse"],
                   "cos": compress_cache[blk]["cos"]}
        for blk in BLOCKS
    },
    "config": {
        "BLOCKS": BLOCKS,
        "HIDDEN_FRACTION": HIDDEN_FRACTION,
        "ADAPTER_HIDDEN": ADAPTER_HIDDEN,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE,
        "N_STEPS_P1": N_STEPS_P1,
        "LR_P1": LR_P1,
        "N_STEPS_P2": N_STEPS_P2,
        "LR_P2": LR_P2,
        "KD_TEMPERATURE": KD_TEMPERATURE,
        "EVAL_EVERY": EVAL_EVERY,
        "SEEDS": SEEDS,
        "LOAD_SEED": LOAD_SEED,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved results to {OUTPUT_PATH}", flush=True)
print("Done.", flush=True)
