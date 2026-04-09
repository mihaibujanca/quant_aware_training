#!/usr/bin/env python3
"""Cycle 380 / Item 388 — Full-model float+adapter ceiling: adapters on all 12 GPT-2 blocks.

Attaches an MLP adapter (h=64, 768→64→768) to ALL 12 UNCOMPRESSED GPT-2 blocks.
All model params frozen; only adapter params trained (2000 steps, Adam lr=1e-4, 3 seeds).

Data protocol (c378):
  Train: wikitext-2-val (N_CALIB=200 seqs)
  Val eval (checkpoint selection): 50 seqs from val
  Test eval (final, touched once per seed): wikitext-2-test (N_EVAL=200 seqs)

Verdict thresholds:
  CONFIRMED: mean test PPL < 36.0
  REFUTED:   mean test PPL > 38.0
  INCONCLUSIVE: [36, 38]

References:
  Float baseline PPL (c378):           ~56.52
  3-block float+adapter ref (c377):    val=38.94 / test=39.73
  Expected params: 12 × (2*64*768 + 64 + 768) = 12 × 99_136 = 1_189_632
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
BLOCKS = list(range(12))   # all 12 blocks
ADAPTER_HIDDEN = 64
LR = 1e-4
N_STEPS = 2000
SEEDS = [42, 137, 2024]
N_CALIB = 200
N_VAL_EVAL = 50    # sequences from val used for checkpoint selection
N_EVAL = 200       # sequences from test for final eval
SEQ_LEN = 128
EVAL_EVERY = 100
FLOAT_BASELINE_REF = 56.52  # c378 test PPL
THREE_BLOCK_REF = {"val": 38.94, "test": 39.73}  # c377

CONFIRMED_LT = 36.0
REFUTED_GT = 38.0

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_380_adapter_12block.json"

torch.manual_seed(42)
print(f"Device: {DEVICE}", flush=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading GPT-2 tokenizer and wikitext-2 data...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Training + val-eval data: from wikitext-2-val
ds_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts_val = [r["text"] for r in ds_val if len(r["text"].strip()) > 50]
all_val_ids = []
for text in texts_val:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        all_val_ids.append(enc["input_ids"][:, :SEQ_LEN])
    if len(all_val_ids) >= N_CALIB + N_VAL_EVAL:
        break

calib_ids_cpu = list(all_val_ids[:N_CALIB])
val_eval_ids = [t.to(DEVICE) for t in all_val_ids[N_CALIB: N_CALIB + N_VAL_EVAL]]
print(f"Calibration: {len(calib_ids_cpu)} seqs, Val eval: {len(val_eval_ids)} seqs", flush=True)

# Test data: from wikitext-2-test (touched once per seed, after checkpoint selection)
ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts_test = [r["text"] for r in ds_test if len(r["text"].strip()) > 50]
test_ids_cpu = []
for text in texts_test:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        test_ids_cpu.append(enc["input_ids"][:, :SEQ_LEN])
    if len(test_ids_cpu) >= N_EVAL:
        break

test_ids = [t.to(DEVICE) for t in test_ids_cpu]
print(f"Test: {len(test_ids)} seqs", flush=True)


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

    W1: (h, 768), b1: (h,), W2: (768, h), b2: (768,)
    Params per block: h*768 + h + 768*h + 768 = 2*h*768 + h + 768
    """

    def __init__(self, fc_weight, fc_bias, proj_weight, proj_bias, h=64):
        super().__init__()
        self.register_buffer("fc_weight", fc_weight.contiguous())      # (3072, 768)
        self.register_buffer("fc_bias", fc_bias.contiguous())          # (3072,)
        self.register_buffer("proj_weight", proj_weight.contiguous())  # (768, 3072)
        self.register_buffer("proj_bias", proj_bias.contiguous())      # (768,)

        d_model = proj_weight.shape[0]  # 768
        self.adapter_W1 = nn.Parameter(torch.randn(h, d_model) * 0.01)
        self.adapter_b1 = nn.Parameter(torch.zeros(h))
        self.adapter_W2 = nn.Parameter(torch.randn(d_model, h) * 0.01)
        self.adapter_b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        h = F.linear(x, self.fc_weight, self.fc_bias)
        h = F.gelu(h)
        out = F.linear(h, self.proj_weight, self.proj_bias)
        a = F.relu(F.linear(out, self.adapter_W1, self.adapter_b1))
        out = out + F.linear(a, self.adapter_W2, self.adapter_b2)
        return out

    def adapter_params(self):
        return [self.adapter_W1, self.adapter_b1, self.adapter_W2, self.adapter_b2]


# ── Training helper ───────────────────────────────────────────────────────────
def run_steps(model, trainable_params, n_steps, calib_dev, val_ids, label=""):
    """Train on calib_dev, select checkpoint by val_ids. Returns best_ppl on val."""
    opt = torch.optim.Adam(trainable_params, lr=LR)
    shuffled = list(calib_dev)
    random.shuffle(shuffled)

    init_ppl = eval_ppl(model, val_ids)
    best_ppl = init_ppl
    best_state = {id(p): p.detach().clone() for p in trainable_params}
    history = [{"step": 0, "val_ppl": float(init_ppl)}]
    print(f"  {label} init val PPL: {init_ppl:.4f}", flush=True)

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
                ppl_e = eval_ppl(model, val_ids)
                history.append({"step": step, "val_ppl": float(ppl_e)})
                if ppl_e < best_ppl:
                    best_ppl = ppl_e
                    best_state = {id(p): p.detach().clone() for p in trainable_params}
                if step % 500 == 0:
                    print(f"  {label} step {step}/{n_steps}: val_ppl={ppl_e:.2f} best={best_ppl:.2f}",
                          flush=True)

    # Check convergence: is val_ppl still decreasing?
    if len(history) >= 3:
        last3 = [h["val_ppl"] for h in history[-3:]]
        converged = last3[-1] < last3[0]
    else:
        converged = None

    # Restore best checkpoint
    with torch.no_grad():
        for p in trainable_params:
            p.data.copy_(best_state[id(p)])

    return best_ppl, history, converged


# ── Float baseline on test ─────────────────────────────────────────────────────
print("Computing float baseline PPL on test...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
float_test_ppl = eval_ppl(base_model, test_ids)
print(f"Float test PPL: {float_test_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
assert abs(float_test_ppl - FLOAT_BASELINE_REF) < 1.0, (
    f"Float test PPL {float_test_ppl:.4f} differs from reference {FLOAT_BASELINE_REF} by > 1.0"
)
del base_model
if DEVICE == "mps":
    torch.mps.empty_cache()

# ── Expected params ───────────────────────────────────────────────────────────
EXPECTED_ADAPTER_PARAMS = len(BLOCKS) * (2 * ADAPTER_HIDDEN * 768 + ADAPTER_HIDDEN + 768)
print(f"Expected trainable params: {EXPECTED_ADAPTER_PARAMS} "
      f"({len(BLOCKS)} blocks × {2 * ADAPTER_HIDDEN * 768 + ADAPTER_HIDDEN + 768} per block)",
      flush=True)

# ── Per-seed training ─────────────────────────────────────────────────────────
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

    # Replace all 12 MLPs with FloatGPT2MLPWithAdapter
    for blk in BLOCKS:
        mlp = model.transformer.h[blk].mlp
        fc_w = mlp.c_fc.weight.detach().float().T.contiguous()      # (3072, 768)
        fc_b = mlp.c_fc.bias.detach().float()                       # (3072,)
        proj_w = mlp.c_proj.weight.detach().float().T.contiguous()  # (768, 3072)
        proj_b = mlp.c_proj.bias.detach().float()                   # (768,)

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

    # Train with val checkpoint selection
    best_val_ppl, history, converged = run_steps(
        model, trainable, N_STEPS, calib_dev, val_eval_ids,
        label=f"seed={seed}"
    )

    # Final test eval (touched once, after checkpoint selection)
    test_ppl = eval_ppl(model, test_ids)

    elapsed = time.time() - t0
    print(f"Seed {seed}: best_val_ppl={best_val_ppl:.4f}, test_ppl={test_ppl:.4f}, "
          f"converged={converged}, time={elapsed:.1f}s", flush=True)

    adapter_results.append({
        "seed": seed,
        "best_val_ppl": float(best_val_ppl),
        "test_ppl": float(test_ppl),
        "converged": converged,
        "history": history,
    })

    del model, trainable, calib_dev
    if DEVICE == "mps":
        torch.mps.empty_cache()

# ── Verdict ───────────────────────────────────────────────────────────────────
test_ppls = [r["test_ppl"] for r in adapter_results]
mean_test_ppl = sum(test_ppls) / len(test_ppls)
std_test_ppl = math.sqrt(sum((p - mean_test_ppl) ** 2 for p in test_ppls) / len(test_ppls))

if mean_test_ppl < CONFIRMED_LT:
    verdict = "CONFIRMED"
elif mean_test_ppl > REFUTED_GT:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}", flush=True)
print(f"VERDICT: {verdict}", flush=True)
print(f"Mean test PPL: {mean_test_ppl:.4f} ± {std_test_ppl:.4f}", flush=True)
print(f"Individual: {[f'{p:.2f}' for p in test_ppls]}", flush=True)
print(f"Float test PPL: {float_test_ppl:.4f}", flush=True)
print(f"3-block ref: val={THREE_BLOCK_REF['val']}, test={THREE_BLOCK_REF['test']}", flush=True)
print(f"Improvement vs 3-block test: {THREE_BLOCK_REF['test'] - mean_test_ppl:.2f} PPL", flush=True)

result = {
    "float_test_ppl": float(float_test_ppl),
    "three_block_ref": THREE_BLOCK_REF,
    "twelve_block": {
        "mean_test_ppl": float(mean_test_ppl),
        "std": float(std_test_ppl),
        "seeds": test_ppls,
    },
    "verdict": verdict,
    "confirmed_lt": CONFIRMED_LT,
    "refuted_gt": REFUTED_GT,
    "n_blocks": len(BLOCKS),
    "adapter_hidden": ADAPTER_HIDDEN,
    "total_adapter_params": EXPECTED_ADAPTER_PARAMS,
    "per_seed_results": adapter_results,
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to {OUTPUT_PATH}", flush=True)
