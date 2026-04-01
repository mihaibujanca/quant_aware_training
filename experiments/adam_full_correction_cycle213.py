"""
Cycle 213 — Item 223: Adam with trainable W_fc — does unfreezing fc improve?
BLOCKS=[1,4,7] K=512. Compare:
  (a) W_out-only: cycle 206 baseline = 48.111 (1000 steps), cycle 210 optimal = 46.556 (4000 steps)
  (b) W_out+W_fc: both trainable (adds 3×768×512 = 1.18M more params; total 2.36M)
Run W_out+W_fc with early stopping (eval every 500 steps, stop on 2 consecutive worsening).
Confirmed if: W_out+W_fc achieves PPL ≤46.1 (≥2 PPL improvement over W_out-only 48.1@1000 steps).
Refuted if: <0.5 PPL improvement.
"""
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

BLOCKS = [1, 4, 7]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
MAX_STEPS = 2000
EVAL_EVERY = 500
OUTPUT_PATH = "autoresearch/results/cycle_213_adam_full_correction.json"

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


class FullCompressedMLP(nn.Module):
    """Both W_fc and W_out are trainable."""
    def __init__(self, fc_weight, fc_bias, W_out, b_out):
        super().__init__()
        self.W_fc = nn.Parameter(fc_weight.clone())   # (768, K) — trainable
        self.b_fc = nn.Parameter(fc_bias.clone())     # (K,)
        self.W_out = nn.Parameter(W_out.clone())      # (768, K)
        self.b_out = nn.Parameter(b_out.clone())      # (768,)
    def forward(self, x):
        return F.gelu(x @ self.W_fc + self.b_fc) @ self.W_out.T + self.b_out


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
    return FullCompressedMLP(W_fc[:, topk_idx], b_fc[topk_idx], sol[:K].T, sol[K])


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

# Build init model
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
ln_outs, proj_outs = collect_mlp_io(model, calib_ids, BLOCKS)
for b in BLOCKS:
    model.transformer.h[b].mlp = fit_block_lstsq(
        model.transformer.h[b], ln_outs[b], proj_outs[b], K).to(DEVICE)
ppl_init = eval_ppl(model, eval_ids)
print(f"Init PPL (topk+lstsq): {ppl_init:.3f}")

# Freeze all except W_fc+W_out+biases of compressed blocks
for param in model.parameters():
    param.requires_grad_(False)
trainable_params = []
for b in BLOCKS:
    mlp = model.transformer.h[b].mlp
    mlp.W_fc.requires_grad_(True)
    mlp.b_fc.requires_grad_(True)
    mlp.W_out.requires_grad_(True)
    mlp.b_out.requires_grad_(True)
    trainable_params.extend([mlp.W_fc, mlp.b_fc, mlp.W_out, mlp.b_out])
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

step = 0
early_stopped = False
print(f"\nTraining W_out+W_fc up to {MAX_STEPS} steps...")
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
                    print(f"  Early stopping at step {step}")
                    early_stopped = True

# W_out-only baseline for comparison (cycle 210 result)
wout_only_best = 46.556
wout_only_at_1000 = 48.173

improvement_vs_wout_1000 = wout_only_at_1000 - best_ppl
improvement_vs_wout_best = wout_only_best - best_ppl

if improvement_vs_wout_1000 >= 2.0:
    verdict = "CONFIRMED"
elif improvement_vs_wout_1000 < 0.5:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nResults:")
print(f"  Float PPL:           {float_ppl:.3f}")
print(f"  Init PPL:            {ppl_init:.3f}")
print(f"  W_out+W_fc best:     {best_ppl:.3f} at step {best_step}")
print(f"  W_out-only @1000:    {wout_only_at_1000:.3f} (cycle 210 ref)")
print(f"  W_out-only best:     {wout_only_best:.3f} @4000 steps (cycle 210)")
print(f"  Improvement vs @1000: {improvement_vs_wout_1000:+.3f}")
print(f"  Verdict: {verdict}")

result = {
    "cycle": 213,
    "item": 223,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "ppl_init": float(ppl_init),
    "best_ppl_wfc_wout": float(best_ppl),
    "best_step": best_step,
    "early_stopped": early_stopped,
    "wout_only_ref_1000steps": wout_only_at_1000,
    "wout_only_ref_best": wout_only_best,
    "improvement_vs_wout_1000": float(improvement_vs_wout_1000),
    "improvement_vs_wout_best": float(improvement_vs_wout_best),
    "ppl_history": ppl_history,
    "config": {"BLOCKS": BLOCKS, "K": K, "MAX_STEPS": MAX_STEPS, "LR": LR,
               "EVAL_EVERY": EVAL_EVERY, "N_CALIB": N_CALIB, "N_EVAL": N_EVAL,
               "SEQ_LEN": SEQ_LEN, "SEED": SEED},
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
