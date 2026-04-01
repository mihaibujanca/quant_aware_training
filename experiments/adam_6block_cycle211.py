"""
Cycle 211 — Item 221: 6-block Adam correction — BLOCKS=[1,2,4,5,7,8] K=512.
Cycle 203 showed any superset of [1,4,7] causes cascade to 75+ PPL with topk+lstsq.
Cycle 206 dissolved the [1,4,7] cascade with Adam correction.
Cycle 210 found optimal is ~4000 steps before overfitting.
Test whether Adam can dissolve cascade for 6 blocks.
Trainable: W_out/b_out of 6 compressed MLPs (2×1.18M=2.36M params).
Run 4000 Adam steps with eval every 500 steps.
Confirmed if: 6-block PPL ≤60.0 (+11.3% vs float 53.895).
"""
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

BLOCKS = [1, 2, 4, 5, 7, 8]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 4000
EVAL_EVERY = 500
OUTPUT_PATH = "autoresearch/results/cycle_211_adam_6block.json"

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

# 3-block baseline (cycle 206 reference)
print("Building 3-block baseline [1,4,7]...")
model_3b = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_3b.eval()
ln_3b, proj_3b = collect_mlp_io(model_3b, calib_ids, [1, 4, 7])
for b in [1, 4, 7]:
    model_3b.transformer.h[b].mlp = fit_block_lstsq(
        model_3b.transformer.h[b], ln_3b[b], proj_3b[b], K).to(DEVICE)
ppl_3b_init = eval_ppl(model_3b, eval_ids)
print(f"3-block init PPL: {ppl_3b_init:.3f}")
del model_3b

# 6-block compressed model
print("Building 6-block compressed model...")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
ln_outs, proj_outs = collect_mlp_io(model, calib_ids, BLOCKS)
for b in BLOCKS:
    model.transformer.h[b].mlp = fit_block_lstsq(
        model.transformer.h[b], ln_outs[b], proj_outs[b], K).to(DEVICE)
    print(f"  Compressed block {b}", flush=True)

ppl_init = eval_ppl(model, eval_ids)
print(f"6-block init PPL: {ppl_init:.3f}")

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
step = 0
print(f"\nTraining {N_STEPS} steps...")
while step < N_STEPS:
    for ids in calib_shuffled:
        if step >= N_STEPS:
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

ppl_final = eval_ppl(model, eval_ids)
print(f"\nFinal 6-block PPL: {ppl_final:.3f}")

# Verdict
if ppl_final <= 60.0:
    verdict = "CONFIRMED"
elif ppl_final > 65.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nResults:")
print(f"  Float PPL:     {float_ppl:.3f}")
print(f"  3-block init:  {ppl_3b_init:.3f} (reference)")
print(f"  6-block init:  {ppl_init:.3f}")
print(f"  6-block Adam:  {ppl_final:.3f} (+{(ppl_final/float_ppl-1)*100:.1f}% vs float)")
print(f"  Verdict: {verdict}")

result = {
    "cycle": 211,
    "item": 221,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "ppl_3b_init": float(ppl_3b_init),
    "ppl_6b_init": float(ppl_init),
    "ppl_6b_adam": float(ppl_final),
    "pct_vs_float": float((ppl_final / float_ppl - 1) * 100),
    "ppl_history": ppl_history,
    "config": {"BLOCKS": BLOCKS, "K": K, "N_STEPS": N_STEPS, "LR": LR,
               "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN, "SEED": SEED},
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
