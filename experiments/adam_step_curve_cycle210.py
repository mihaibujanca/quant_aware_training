"""
Cycle 210 — Item 220: Adam W_out correction step-count curve.
BLOCKS=[1,4,7] K=512. Test {500,1000,2000,4000,8000} steps.
Find where PPL plateaus and whether overfitting occurs.
"""
import json
import math
import copy
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
EVAL_STEPS = [500, 1000, 2000, 4000, 8000]
MAX_STEPS = max(EVAL_STEPS)
OUTPUT_PATH = "autoresearch/results/cycle_210_adam_step_curve.json"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load ──────────────────────────────────────────────────────────────────────
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

# ── Helpers ───────────────────────────────────────────────────────────────────
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
        ln_outs[b] = torch.cat(ln_outs[b], dim=0)
        proj_outs[b] = torch.cat(proj_outs[b], dim=0)
    return ln_outs, proj_outs


def fit_block_lstsq(block, ln_out_cpu, proj_out_cpu, K):
    W_fc = block.mlp.c_fc.weight.data.float().cpu()
    b_fc = block.mlp.c_fc.bias.data.float().cpu()
    W_proj = block.mlp.c_proj.weight.data.float().cpu()
    act_all = F.gelu(ln_out_cpu @ W_fc + b_fc)
    scores = act_all.std(0) * W_proj.norm(dim=1)
    topk_idx = scores.topk(K).indices
    W_fc_sub = W_fc[:, topk_idx]
    b_fc_sub = b_fc[topk_idx]
    act_sub = act_all[:, topk_idx]
    ones = torch.ones(act_sub.shape[0], 1)
    sol = torch.linalg.lstsq(torch.cat([act_sub, ones], 1), proj_out_cpu, rcond=None).solution
    return CompressedMLP(W_fc_sub, b_fc_sub, sol[:K].T, sol[K])


def eval_ppl(model, eval_ids):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


# ── Float baseline ────────────────────────────────────────────────────────────
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()
float_ppl = eval_ppl(model_float, eval_ids)
print(f"Float PPL: {float_ppl:.3f}")
del model_float

# ── Build topk+lstsq init ─────────────────────────────────────────────────────
print("Building compressed model...")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
ln_outs, proj_outs = collect_mlp_io(model, calib_ids, BLOCKS)
for b in BLOCKS:
    c = fit_block_lstsq(model.transformer.h[b], ln_outs[b], proj_outs[b], K)
    model.transformer.h[b].mlp = c.to(DEVICE)
ppl_init = eval_ppl(model, eval_ids)
print(f"PPL after init: {ppl_init:.3f}")

# ── Freeze and set up Adam ────────────────────────────────────────────────────
for param in model.parameters():
    param.requires_grad_(False)
trainable_params = []
for b in BLOCKS:
    model.transformer.h[b].mlp.W_out.requires_grad_(True)
    model.transformer.h[b].mlp.b_out.requires_grad_(True)
    trainable_params.extend([model.transformer.h[b].mlp.W_out,
                              model.transformer.h[b].mlp.b_out])
print(f"Trainable: {sum(p.numel() for p in trainable_params):,}")

optimizer = torch.optim.Adam(trainable_params, lr=LR)
random.seed(SEED)
calib_shuffled = list(calib_ids)
random.shuffle(calib_shuffled)

# ── Training loop ─────────────────────────────────────────────────────────────
step_results = {}
step = 0
print(f"\nTraining {MAX_STEPS} steps...")
while step < MAX_STEPS:
    for ids in calib_shuffled:
        if step >= MAX_STEPS:
            break
        model.train()
        optimizer.zero_grad()
        out = model(ids, labels=ids)
        out.loss.backward()
        optimizer.step()
        step += 1

        if step in EVAL_STEPS:
            ppl_e = eval_ppl(model, eval_ids)
            step_results[step] = float(ppl_e)
            print(f"  Step {step:5d}: eval_ppl={ppl_e:.3f}", flush=True)

# ── Verdict ───────────────────────────────────────────────────────────────────
ppls = [step_results[s] for s in EVAL_STEPS]
# Plateau: last 2 eval points differ by <0.5 PPL
last_two = ppls[-1] - ppls[-2] if len(ppls) >= 2 else 999
plateau_found = abs(last_two) < 0.5
# Overfitting: PPL increases after midpoint
mid_idx = len(ppls) // 2
overfitting = ppls[-1] > ppls[mid_idx] + 2.0

if plateau_found and not overfitting:
    verdict = "CONFIRMED"
elif not plateau_found:
    verdict = "REFUTED"  # still decreasing
else:
    verdict = "INCONCLUSIVE"

print(f"\nResults:")
print(f"  Float PPL:  {float_ppl:.3f}")
print(f"  Init PPL:   {ppl_init:.3f}")
for s in EVAL_STEPS:
    print(f"  Step {s:5d}: {step_results[s]:.3f}")
print(f"  Plateau: {plateau_found}  Overfitting: {overfitting}")
print(f"  Verdict: {verdict}")

result = {
    "cycle": 210,
    "item": 220,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "ppl_init": float(ppl_init),
    "step_results": {str(s): step_results[s] for s in EVAL_STEPS},
    "plateau_found": plateau_found,
    "overfitting": overfitting,
    "last_two_delta": float(last_two),
    "config": {"BLOCKS": BLOCKS, "K": K, "LR": LR, "EVAL_STEPS": EVAL_STEPS,
               "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN, "SEED": SEED},
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
