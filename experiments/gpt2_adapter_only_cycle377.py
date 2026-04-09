#!/usr/bin/env python3
"""Cycle 377 / Item 385 — Adapter-only baseline: does compression or the adapter cause the
"beat float" anomaly?

Attaches an MLP adapter (h=64, 768→64→768) to UNCOMPRESSED GPT-2 blocks [b2, b1, b10].
All model params frozen; only adapter params trained (2000 steps, Adam lr=1e-4, 3 seeds).

Verdict thresholds:
  CONFIRMED: mean PPL < 50.0  (adapter alone explains beating float)
  REFUTED:   mean PPL > 53.0  (compression necessary for the gain)
  INCONCLUSIVE: [50, 53]

References:
  Float baseline PPL:              ~53.895
  Compressed+adapter ref PPL:     ~47.43  (cycle 367, h=128, 595K adapter params)
  This experiment:                  h=64, 297K adapter params (conservative)
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
ADAPTER_HIDDEN = 64
LR = 1e-4
N_STEPS = 2000
SEEDS = [42, 137, 2024]
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
EVAL_EVERY = 100
FLOAT_BASELINE_REF = 53.895

CONFIRMED_LT = 50.0
REFUTED_GT = 53.0

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_377_adapter_only.json"

torch.manual_seed(42)
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


# ── FloatGPT2MLPWithAdapter ───────────────────────────────────────────────────
class FloatGPT2MLPWithAdapter(nn.Module):
    """Wraps the original (uncompressed) GPT-2 MLP with a residual adapter.

    forward(x):
        out = c_proj(gelu(c_fc(x)))          # original float FFN
        a   = relu(W1 @ out + b1)            # adapter bottleneck (on post-FFN output)
        out = out + W2 @ a + b2              # residual correction
        return out

    Adapter takes 768-dim post-FFN output (not GELU hidden).
    W1: (h, 768), b1: (h,), W2: (768, h), b2: (768,)
    Params per block: h*768 + h + 768*h + 768 = 2*h*768 + h + 768

    All GPT-2 weights are registered as frozen buffers.
    Only W1, b1, W2, b2 are trainable nn.Parameters.
    """

    def __init__(self, fc_weight, fc_bias, proj_weight, proj_bias, h=64):
        """
        fc_weight:   (3072, 768)  — c_fc.weight from Conv1D (already transposed)
        fc_bias:     (3072,)
        proj_weight: (768, 3072)  — c_proj.weight from Conv1D (already transposed)
        proj_bias:   (768,)
        """
        super().__init__()
        # Register original weights as frozen buffers
        self.register_buffer("fc_weight", fc_weight.contiguous())    # (3072, 768)
        self.register_buffer("fc_bias", fc_bias.contiguous())        # (3072,)
        self.register_buffer("proj_weight", proj_weight.contiguous())  # (768, 3072)
        self.register_buffer("proj_bias", proj_bias.contiguous())    # (768,)

        # Adapter parameters (trainable)
        d_model = proj_weight.shape[0]  # 768
        self.adapter_W1 = nn.Parameter(torch.randn(h, d_model) * 0.01)
        self.adapter_b1 = nn.Parameter(torch.zeros(h))
        self.adapter_W2 = nn.Parameter(torch.randn(d_model, h) * 0.01)
        self.adapter_b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # Original float FFN
        h = F.linear(x, self.fc_weight, self.fc_bias)   # (*, 3072)
        h = F.gelu(h)
        out = F.linear(h, self.proj_weight, self.proj_bias)  # (*, 768)

        # Adapter on post-FFN output
        a = F.relu(F.linear(out, self.adapter_W1, self.adapter_b1))  # (*, h)
        out = out + F.linear(a, self.adapter_W2, self.adapter_b2)    # (*, 768)
        return out

    def adapter_params(self):
        return [self.adapter_W1, self.adapter_b1, self.adapter_W2, self.adapter_b2]


# ── Training helper ───────────────────────────────────────────────────────────
def run_steps(model, trainable_params, n_steps, calib_dev, eval_ids, label=""):
    opt = torch.optim.Adam(trainable_params, lr=LR)
    shuffled = list(calib_dev)
    random.shuffle(shuffled)

    init_ppl = eval_ppl(model, eval_ids)
    best_ppl = init_ppl
    best_state = {id(p): p.detach().clone() for p in trainable_params}
    history = [{"step": 0, "eval_ppl": float(init_ppl)}]
    print(f"  {label} init PPL: {init_ppl:.4f}", flush=True)

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
            if step % EVAL_EVERY == 0:
                ppl_e = eval_ppl(model, eval_ids)
                history.append({"step": step, "eval_ppl": float(ppl_e)})
                if ppl_e < best_ppl:
                    best_ppl = ppl_e
                    best_state = {id(p): p.detach().clone() for p in trainable_params}
                if step % 500 == 0:
                    print(f"  {label} step {step}/{n_steps}: ppl={ppl_e:.2f} best={best_ppl:.2f}",
                          flush=True)

    # Restore best checkpoint
    with torch.no_grad():
        for p in trainable_params:
            p.data.copy_(best_state[id(p)])

    final_ppl = eval_ppl(model, eval_ids)
    return best_ppl, final_ppl, history


# ── Float baseline ────────────────────────────────────────────────────────────
print("Computing float baseline PPL...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
float_ppl = eval_ppl(base_model, eval_ids_list)
print(f"Float PPL: {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl:.4f} differs from reference {FLOAT_BASELINE_REF} by > 0.5"
)
del base_model
if DEVICE == "mps":
    torch.mps.empty_cache()

# ── Per-seed training ─────────────────────────────────────────────────────────
# Expected trainable params: 3 blocks × (2*64*768 + 64 + 768) = 3 × 99_136 = 297_408
EXPECTED_ADAPTER_PARAMS = 3 * (2 * ADAPTER_HIDDEN * 768 + ADAPTER_HIDDEN + 768)

adapter_results = []

for seed in SEEDS:
    print(f"\n{'='*60}", flush=True)
    print(f"Seed {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    torch.manual_seed(seed)
    random.seed(seed)

    # Fresh GPT-2
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()

    # Replace target MLPs with FloatGPT2MLPWithAdapter
    for blk in BLOCKS:
        mlp = model.transformer.h[blk].mlp
        # Conv1D weight layout: weight shape is (in_features, out_features) in PyTorch terms
        # GPT-2 Conv1D: c_fc.weight is (768, 3072), c_proj.weight is (3072, 768)
        # To use F.linear we need (out, in), so transpose
        fc_w = mlp.c_fc.weight.detach().float().T.contiguous()    # (3072, 768)
        fc_b = mlp.c_fc.bias.detach().float()                     # (3072,)
        proj_w = mlp.c_proj.weight.detach().float().T.contiguous()  # (768, 3072)
        proj_b = mlp.c_proj.bias.detach().float()                 # (768,)

        new_mlp = FloatGPT2MLPWithAdapter(
            fc_weight=fc_w, fc_bias=fc_b,
            proj_weight=proj_w, proj_bias=proj_b,
            h=ADAPTER_HIDDEN
        ).to(DEVICE)
        model.transformer.h[blk].mlp = new_mlp

    # Freeze all params, then unfreeze only adapter params
    for p in model.parameters():
        p.requires_grad_(False)

    trainable = []
    for blk in BLOCKS:
        adapter_ps = model.transformer.h[blk].mlp.adapter_params()
        for p in adapter_ps:
            p.requires_grad_(True)
        trainable.extend(adapter_ps)

    total_trainable = sum(p.numel() for p in trainable)
    print(f"Trainable params: {total_trainable} (expected {EXPECTED_ADAPTER_PARAMS})", flush=True)
    assert total_trainable == EXPECTED_ADAPTER_PARAMS, (
        f"Param count mismatch: {total_trainable} != {EXPECTED_ADAPTER_PARAMS}"
    )

    calib_dev = [t.to(DEVICE) for t in calib_ids_cpu]
    t0 = time.time()
    best_ppl, final_ppl, history = run_steps(
        model, trainable, N_STEPS, calib_dev, eval_ids_list,
        label=f"seed={seed}"
    )
    elapsed = time.time() - t0
    print(f"Seed {seed}: best_ppl={best_ppl:.4f}, final_ppl={final_ppl:.4f}, "
          f"time={elapsed:.1f}s", flush=True)

    adapter_results.append({
        "seed": seed,
        "best_ppl": float(best_ppl),
        "final_ppl": float(final_ppl),
        "history": history,
    })

    del model, trainable, calib_dev
    if DEVICE == "mps":
        torch.mps.empty_cache()

# ── Verdict ───────────────────────────────────────────────────────────────────
ppls = [r["best_ppl"] for r in adapter_results]
mean_ppl = sum(ppls) / len(ppls)
std_ppl = math.sqrt(sum((p - mean_ppl) ** 2 for p in ppls) / len(ppls))

if mean_ppl < CONFIRMED_LT:
    verdict = "CONFIRMED"
elif mean_ppl > REFUTED_GT:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}", flush=True)
print(f"VERDICT: {verdict}", flush=True)
print(f"Mean PPL: {mean_ppl:.4f} ± {std_ppl:.4f}", flush=True)
print(f"Individual: {[f'{p:.2f}' for p in ppls]}", flush=True)
print(f"Float baseline: {float_ppl:.4f}", flush=True)
print(f"Compressed+adapter ref: 47.43 (h=128, 595K params)", flush=True)

result = {
    "float_ppl": float(float_ppl),
    "compressed_adapter_ref_ppl": 47.43,
    "compressed_adapter_ref_h": 128,
    "compressed_adapter_ref_params": 595200,
    "adapter_only_h": ADAPTER_HIDDEN,
    "adapter_params_per_block": 2 * ADAPTER_HIDDEN * 768 + ADAPTER_HIDDEN + 768,
    "total_adapter_params": EXPECTED_ADAPTER_PARAMS,
    "adapter_only_results": adapter_results,
    "mean_ppl": float(mean_ppl),
    "std_ppl": float(std_ppl),
    "verdict": verdict,
    "confirmed_lt": CONFIRMED_LT,
    "refuted_gt": REFUTED_GT,
    "note": (
        f"Float adapter uses h={ADAPTER_HIDDEN} ({EXPECTED_ADAPTER_PARAMS} params) "
        f"vs compressed ref h=128 (595K params). Conservative comparison."
    ),
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to {OUTPUT_PATH}", flush=True)
