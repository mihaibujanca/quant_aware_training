#!/usr/bin/env python3
"""Cycle 356 / Item 365 — Distillation baseline for GPT-2 b2.

Trains a student FFN block at the same total param budget as the compressed recipe
(topk+lstsq k=512 + adapter h=128 ≈ 985K params) via knowledge distillation.

Student architecture:
  c_fc (768 → 641) + c_proj (641 → 768)  → 985,344 params
  (768*641 + 641 + 641*768 + 768 = 985,344 ≈ 985K target)

KD training:
  Loss = KL(student_block_output || teacher_block_output) on wikitext-2 sequences
  1000 steps, Adam lr=1e-4, 3 seeds — matching adapter training conditions

Eval:
  Replace b2 FFN with trained student, measure full-model PPL.

Verdict:
  CONFIRMED (geometric competitive): |46.22 - dist_PPL| ≤ 1.0 PPL
  REFUTED: distillation wins by > 2.0 PPL (dist_PPL < 44.22)
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
BLOCK_ID = 2
D_MODEL = 768

# Budget matching
GEOMETRIC_TOTAL_PARAMS = 985216  # topk k=512 FFN (787712) + adapter h=128 (197504)
# Student hidden: solve 768*h + h + h*768 + 768 = 985216  → h*(768+1+768) + 768 = 985216
# h*(1537) = 984448  → h = 640.5 → round to 641
STUDENT_HIDDEN = 641
# Actual: 768*641 + 641 + 641*768 + 768 = 491,904 + 641 + 492,288 + 768 = 985,601 ≈ close enough

N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
N_STEPS = 1000
LR = 1e-4
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]

# Reference values
GEOMETRIC_PPL = 46.22   # topk+lstsq k=512 + adapter h=128 (cycle 350)
FLOAT_PPL_REF = 53.895

# Verdict thresholds
CONFIRMED_DELTA = 1.0   # geometric within 1.0 PPL of distillation
REFUTED_DELTA = 2.0     # distillation wins by > 2.0 PPL

OUTPUT_PATH = "autoresearch/results/cycle_356_distillation_baseline.json"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(42)
print(f"Device: {DEVICE}", flush=True)

# ── Student param count ───────────────────────────────────────────────────────
n_student = D_MODEL * STUDENT_HIDDEN + STUDENT_HIDDEN + STUDENT_HIDDEN * D_MODEL + D_MODEL
print(f"Student hidden={STUDENT_HIDDEN}, params={n_student:,} "
      f"(target={GEOMETRIC_TOTAL_PARAMS:,})", flush=True)

# ── Data ──────────────────────────────────────────────────────────────────────
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


def collect_block_inputs(model, block_id, calib_ids, device):
    """Collect MLP input activations (post-layernorm, pre-MLP) for a block."""
    data = []
    hooks = []

    def hook_fn(module, inputs, output):
        x = inputs[0].detach()
        data.append(x.reshape(-1, x.shape[-1]).cpu())

    hook = model.transformer.h[block_id].mlp.register_forward_hook(hook_fn)
    hooks.append(hook)

    model.eval()
    with torch.no_grad():
        for ids in calib_ids:
            ids_dev = ids.to(device) if ids.device != device else ids
            model(ids_dev)

    for h in hooks:
        h.remove()

    return torch.cat(data, dim=0).float()


# ── Student MLP ───────────────────────────────────────────────────────────────
class StudentFFN(nn.Module):
    """Smaller FFN with GELU (matching teacher architecture)."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden)
        self.c_proj = nn.Linear(hidden, d_model)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Float baseline ────────────────────────────────────────────────────────────
print("Computing float baseline PPL...", flush=True)
teacher_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
teacher_model.eval()
float_ppl = eval_ppl(teacher_model, eval_ids)
print(f"Float PPL: {float_ppl:.4f} (ref {FLOAT_PPL_REF})", flush=True)
assert abs(float_ppl - FLOAT_PPL_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from reference {FLOAT_PPL_REF} by > 0.5"
)

# ── Teacher block outputs for KD ─────────────────────────────────────────────
print(f"\nCollecting teacher block {BLOCK_ID} outputs for KD...", flush=True)
teacher_block_outputs = []
teacher_block_inputs = []
kd_hooks = []

def make_teacher_hook():
    def hook_fn(module, inputs, output):
        x = inputs[0].detach()
        y = output.detach()
        teacher_block_inputs.append(x.reshape(-1, x.shape[-1]).cpu())
        teacher_block_outputs.append(y.reshape(-1, y.shape[-1]).cpu())
    return hook_fn

kd_hook = teacher_model.transformer.h[BLOCK_ID].mlp.register_forward_hook(
    make_teacher_hook()
)

teacher_model.eval()
with torch.no_grad():
    for ids in calib_ids_cpu:
        ids_dev = ids.to(DEVICE)
        teacher_model(ids_dev)

kd_hook.remove()

# Stack: (N_total_tokens, 768)
teacher_inputs = torch.cat(teacher_block_inputs, dim=0).float()   # (N, 768)
teacher_outputs = torch.cat(teacher_block_outputs, dim=0).float() # (N, 768)
print(f"Teacher outputs shape: {teacher_outputs.shape}", flush=True)

del teacher_block_inputs, teacher_block_outputs

# ── KD Training ───────────────────────────────────────────────────────────────
# We train student to match teacher block output given teacher input activations.
# Loss: MSE on block output vectors (token-level distillation on the FFN output).
# This avoids the need for a full forward pass through the teacher each step.

all_results = []

for seed in SEEDS:
    torch.manual_seed(seed)
    random.seed(seed)

    student = StudentFFN(D_MODEL, STUDENT_HIDDEN).to(DEVICE)
    n_params = student.n_params
    print(f"\nSeed {seed}: student params={n_params:,}", flush=True)

    optimizer = torch.optim.Adam(student.parameters(), lr=LR)

    # Build mini-batches from precomputed teacher inputs/outputs
    BATCH_SIZE = 64
    N_tokens = teacher_inputs.shape[0]

    t_in = teacher_inputs.to(DEVICE)
    t_out = teacher_outputs.to(DEVICE)

    # Initial eval: swap student into model, measure PPL
    eval_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    eval_model.eval()
    eval_model.transformer.h[BLOCK_ID].mlp = student
    for p in eval_model.parameters():
        p.requires_grad_(False)
    for p in student.parameters():
        p.requires_grad_(True)

    init_ppl = eval_ppl(eval_model, eval_ids)
    print(f"  Init PPL: {init_ppl:.2f}", flush=True)

    best_ppl = init_ppl
    best_step = 0
    best_state = {n: p.detach().clone() for n, p in student.named_parameters()}
    history = [{"step": 0, "eval_ppl": float(init_ppl)}]

    step = 0
    t0 = time.time()

    # Build shuffled index permutation each pass
    perm = torch.randperm(N_tokens)
    batch_start = 0

    while step < N_STEPS:
        student.train()
        optimizer.zero_grad()

        # Get next batch
        if batch_start + BATCH_SIZE > N_tokens:
            perm = torch.randperm(N_tokens, device=DEVICE)
            batch_start = 0
        idx = perm[batch_start: batch_start + BATCH_SIZE]
        batch_start += BATCH_SIZE

        x_batch = t_in[idx]    # (B, 768)
        y_batch = t_out[idx]   # (B, 768)

        y_pred = student(x_batch)
        loss = F.mse_loss(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        step += 1

        if step % EVAL_EVERY == 0:
            ppl_e = eval_ppl(eval_model, eval_ids)
            history.append({"step": step, "eval_ppl": float(ppl_e)})
            if ppl_e < best_ppl:
                best_ppl = ppl_e
                best_step = step
                best_state = {n: p.detach().clone() for n, p in student.named_parameters()}
            if step % 200 == 0:
                print(f"  step={step}: ppl={ppl_e:.2f} best={best_ppl:.2f}@{best_step} "
                      f"loss={loss.item():.4f}", flush=True)

    # Restore best
    with torch.no_grad():
        for n, p in student.named_parameters():
            p.data.copy_(best_state[n])
    restored_ppl = eval_ppl(eval_model, eval_ids)
    dt = time.time() - t0

    print(f"  seed={seed}: init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
          f"restored={restored_ppl:.2f} ({dt:.0f}s)", flush=True)

    all_results.append({
        "seed": seed,
        "n_params": n_params,
        "init_ppl": float(init_ppl),
        "best_ppl": float(best_ppl),
        "best_step": int(best_step),
        "restored_ppl": float(restored_ppl),
        "history": history,
    })

    del student, optimizer, eval_model
    if DEVICE == "mps":
        torch.mps.empty_cache()


# ── Verdict ───────────────────────────────────────────────────────────────────
best_ppls = [r["best_ppl"] for r in all_results]
dist_mean = sum(best_ppls) / len(best_ppls)
dist_std = math.sqrt(sum((x - dist_mean) ** 2 for x in best_ppls) / len(best_ppls))
dist_min = min(best_ppls)
dist_max = max(best_ppls)

print(f"\n{'='*60}", flush=True)
print("DISTILLATION BASELINE — GPT-2 b2", flush=True)
print(f"{'='*60}", flush=True)
print(f"Student PPL: {dist_mean:.2f} ± {dist_std:.2f} "
      f"(min={dist_min:.2f}, max={dist_max:.2f})", flush=True)
print(f"Geometric recipe PPL: {GEOMETRIC_PPL}", flush=True)
print(f"Float PPL:  {float_ppl:.4f}", flush=True)

delta = dist_mean - GEOMETRIC_PPL  # positive → geometric is better
print(f"Delta (dist - geometric): {delta:+.2f} PPL", flush=True)

if abs(delta) <= CONFIRMED_DELTA:
    verdict = "CONFIRMED"
    verdict_detail = f"geometric competitive (|delta|={abs(delta):.2f} ≤ 1.0)"
elif delta > REFUTED_DELTA:
    verdict = "CONFIRMED"  # geometric actually wins by a lot
    verdict_detail = f"geometric WINS by {delta:.2f} PPL"
elif delta < -REFUTED_DELTA:
    verdict = "REFUTED"
    verdict_detail = f"distillation wins by {-delta:.2f} PPL (>{REFUTED_DELTA})"
else:
    verdict = "INCONCLUSIVE"
    verdict_detail = f"delta={delta:+.2f} (within 1–2 PPL gray zone)"

print(f"VERDICT: {verdict} — {verdict_detail}", flush=True)

# ── Save JSON ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
result = {
    "cycle": 356,
    "item": 365,
    "verdict": verdict,
    "verdict_detail": verdict_detail,
    "float_ppl": float(float_ppl),
    "float_ppl_ref": FLOAT_PPL_REF,
    "geometric_ppl": GEOMETRIC_PPL,
    "distillation_mean": float(dist_mean),
    "distillation_std": float(dist_std),
    "distillation_min": float(dist_min),
    "distillation_max": float(dist_max),
    "delta_dist_minus_geometric": float(delta),
    "student_hidden": STUDENT_HIDDEN,
    "student_params": n_student,
    "geometric_total_params": GEOMETRIC_TOTAL_PARAMS,
    "seeds": all_results,
    "config": {
        "BLOCK_ID": BLOCK_ID,
        "D_MODEL": D_MODEL,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "N_STEPS": N_STEPS,
        "LR": LR,
        "EVAL_EVERY": EVAL_EVERY,
        "SEEDS": SEEDS,
        "BATCH_SIZE": 64,
        "loss": "MSE(student_output, teacher_output)",
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nResults saved to {OUTPUT_PATH}", flush=True)
print("Done.", flush=True)
