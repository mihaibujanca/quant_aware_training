"""
Cycle 212 — Item 222: 10-block full compression with Adam — BLOCKS=[1..10] K=512.
Maximum FFN compression: all 10 compressible blocks (exclude 0 and 11 bookends).
Pattern from prior cycles: more trainable params → faster overfitting:
  3-block (1.18M params): optimal ~4000 steps
  6-block (2.36M params): optimal ~1500 steps
  → 10-block (3.93M params): estimate ~500-800 steps before overfitting

Use eval-based early stopping: check every 100 steps; stop if PPL increases 2 consecutive times.
Max steps: 2000 to catch early overfitting.
Confirmed if: 10-block Adam PPL ≤65.0 (+20.6% vs float) with early stopping.
"""
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

BLOCKS = list(range(1, 11))  # [1,2,3,4,5,6,7,8,9,10]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
MAX_STEPS = 1000
EVAL_EVERY = 200
OUTPUT_PATH = "autoresearch/results/cycle_212_adam_10block.json"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

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

calib_ids = [t.to(DEVICE) for t in all_ids[:N_CALIB]]
eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB:N_CALIB + N_EVAL]]
print(f"Calibration: {len(calib_ids)} seqs, Eval: {len(eval_ids)} seqs")


class CompressedMLP(nn.Module):
    def __init__(self, fc_weight, fc_bias, W_out, b_out):
        super().__init__()
        self.register_buffer("fc_weight", fc_weight)
        self.register_buffer("fc_bias", fc_bias)
        self.W_out = nn.Parameter(W_out.clone())
        self.b_out = nn.Parameter(b_out.clone())
    def forward(self, x):
        return F.gelu(x @ self.fc_weight + self.fc_bias) @ self.W_out.T + self.b_out


def collect_mlp_io(model, calib_ids, block_ids):
    ln_outs = {b: [] for b in block_ids}
    proj_outs = {b: [] for b in block_ids}
    hooks = []
    for b in block_ids:
        def mk_hook(bb):
            def fn(m, inp, out):
                ln_outs[bb].append(inp[0].detach().float().cpu().reshape(-1, 768))
                proj_outs[bb].append(out.detach().float().cpu().reshape(-1, 768))
            return fn
        hooks.append(model.transformer.h[b].mlp.register_forward_hook(mk_hook(b)))
    with torch.no_grad():
        for ids in calib_ids:
            model(ids)
    for h in hooks:
        h.remove()
    for b in block_ids:
        ln_outs[b] = torch.cat(ln_outs[b])
        proj_outs[b] = torch.cat(proj_outs[b])
    return ln_outs, proj_outs


def fit_block_lstsq(block, ln_out_cpu, proj_out_cpu, K):
    W_fc = block.mlp.c_fc.weight.data.float().cpu()
    b_fc = block.mlp.c_fc.bias.data.float().cpu()
    W_proj = block.mlp.c_proj.weight.data.float().cpu()
    act_all = F.gelu(ln_out_cpu @ W_fc + b_fc)
    scores = act_all.std(0) * W_proj.norm(dim=1)
    topk_idx = scores.topk(K).indices
    act_sub = act_all[:, topk_idx]
    ones = torch.ones(act_sub.shape[0], 1)
    sol = torch.linalg.lstsq(torch.cat([act_sub, ones], 1), proj_out_cpu, rcond=None).solution
    return CompressedMLP(W_fc[:, topk_idx], b_fc[topk_idx], sol[:K].T, sol[K])


def eval_ppl(model, eval_ids):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


# Float baseline
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()
float_ppl = eval_ppl(model_float, eval_ids)
print(f"Float PPL: {float_ppl:.3f}")
del model_float

# 10-block compressed model
print("Building 10-block compressed model...")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
ln_outs, proj_outs = collect_mlp_io(model, calib_ids, BLOCKS)
for b in BLOCKS:
    model.transformer.h[b].mlp = fit_block_lstsq(
        model.transformer.h[b], ln_outs[b], proj_outs[b], K).to(DEVICE)
ppl_init = eval_ppl(model, eval_ids)
print(f"10-block init PPL: {ppl_init:.3f}")

# Freeze and train
for param in model.parameters():
    param.requires_grad_(False)
trainable_params = []
for b in BLOCKS:
    model.transformer.h[b].mlp.W_out.requires_grad_(True)
    model.transformer.h[b].mlp.b_out.requires_grad_(True)
    trainable_params.extend([model.transformer.h[b].mlp.W_out,
                              model.transformer.h[b].mlp.b_out])
n_trainable = sum(p.numel() for p in trainable_params)
print(f"Trainable params: {n_trainable:,} ({n_trainable/1e6:.2f}M)")

optimizer = torch.optim.Adam(trainable_params, lr=LR)
random.seed(SEED)
calib_shuffled = list(calib_ids)
random.shuffle(calib_shuffled)

ppl_history = []
best_ppl = float('inf')
best_step = 0
consecutive_worse = 0
prev_ppl = float('inf')

step = 0
early_stopped = False
print(f"\nTraining up to {MAX_STEPS} steps (early stopping on 2 consecutive increases)...")
while step < MAX_STEPS:
    for ids in calib_shuffled:
        if step >= MAX_STEPS or early_stopped:
            break
        model.train()
        optimizer.zero_grad()
        model(ids, labels=ids).loss.backward()
        optimizer.step()
        step += 1
        if step % EVAL_EVERY == 0:
            ppl_e = eval_ppl(model, eval_ids)
            ppl_history.append({"step": step, "eval_ppl": float(ppl_e)})
            print(f"  Step {step:5d}: eval_ppl={ppl_e:.3f}", flush=True)
            if ppl_e < best_ppl:
                best_ppl = ppl_e
                best_step = step
                consecutive_worse = 0
            else:
                consecutive_worse += 1
                if consecutive_worse >= 2:
                    print(f"  Early stopping at step {step} (2 consecutive increases)")
                    early_stopped = True
            prev_ppl = ppl_e

print(f"\nBest PPL: {best_ppl:.3f} at step {best_step}")

# Verdict
if best_ppl <= 65.0:
    verdict = "CONFIRMED"
elif best_ppl > 75.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nResults:")
print(f"  Float PPL:     {float_ppl:.3f}")
print(f"  10-block init: {ppl_init:.3f}")
print(f"  Best Adam:     {best_ppl:.3f} at step {best_step} (+{(best_ppl/float_ppl-1)*100:.1f}% vs float)")
print(f"  Early stopped: {early_stopped}")
print(f"  Verdict: {verdict}")

result = {
    "cycle": 212,
    "item": 222,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "ppl_10b_init": float(ppl_init),
    "best_ppl": float(best_ppl),
    "best_step": best_step,
    "early_stopped": early_stopped,
    "pct_vs_float": float((best_ppl / float_ppl - 1) * 100),
    "ppl_history": ppl_history,
    "config": {"BLOCKS": BLOCKS, "K": K, "MAX_STEPS": MAX_STEPS, "LR": LR,
               "EVAL_EVERY": EVAL_EVERY,
               "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN, "SEED": SEED},
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
