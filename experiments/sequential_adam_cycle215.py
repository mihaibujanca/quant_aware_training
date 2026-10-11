"""
Cycle 215 — Item 225: Sequential Adam correction — correct blocks one at a time.
Joint correction (cycle 206): all 3 blocks trained simultaneously.
Sequential: correct block 1 W_out first (freeze rest), then block 4, then block 7.
Each block corrected for 2000 steps with early stopping.
BLOCKS=[1,4,7] K=512.
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
STEPS_PER_BLOCK = 2000
EVAL_EVERY = 500
OUTPUT_PATH = "autoresearch/results/cycle_215_sequential_adam.json"

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


def train_block_wout(model, block_id, calib_ids, eval_ids, n_steps, eval_every):
    """Train only W_out/b_out of block_id; all other params frozen."""
    for param in model.parameters():
        param.requires_grad_(False)
    mlp = model.transformer.h[block_id].mlp
    mlp.W_out.requires_grad_(True)
    mlp.b_out.requires_grad_(True)

    optimizer = torch.optim.Adam([mlp.W_out, mlp.b_out], lr=LR)
    random.seed(SEED + block_id)
    calib_shuffled = list(calib_ids)
    random.shuffle(calib_shuffled)

    best_ppl = float('inf')
    best_step = 0
    consecutive_worse = 0
    history = []

    step = 0
    early_stopped = False
    while step < n_steps:
        for ids in calib_shuffled:
            if step >= n_steps or early_stopped:
                break
            model.train()
            optimizer.zero_grad()
            model(ids, labels=ids).loss.backward()
            optimizer.step()
            step += 1
            if step % eval_every == 0:
                ppl_e = eval_ppl(model, eval_ids)
                history.append({"step": step, "eval_ppl": float(ppl_e)})
                print(f"    step {step}: ppl={ppl_e:.3f}", flush=True)
                if ppl_e < best_ppl:
                    best_ppl = ppl_e
                    best_step = step
                    consecutive_worse = 0
                else:
                    consecutive_worse += 1
                    if consecutive_worse >= 2:
                        early_stopped = True

    return best_ppl, best_step, history


# Float baseline
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()
float_ppl = eval_ppl(model_float, eval_ids)
print(f"Float PPL: {float_ppl:.3f}")
del model_float

# Build init
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
ln_outs, proj_outs = collect_mlp_io(model, calib_ids, BLOCKS)
for b in BLOCKS:
    model.transformer.h[b].mlp = fit_block_lstsq(
        model.transformer.h[b], ln_outs[b], proj_outs[b], K).to(DEVICE)
ppl_init = eval_ppl(model, eval_ids)
print(f"Init PPL: {ppl_init:.3f}")

# Sequential correction
block_results = {}
for b in BLOCKS:
    print(f"\nBlock {b} sequential correction ({STEPS_PER_BLOCK} steps)...")
    best_ppl, best_step, history = train_block_wout(
        model, b, calib_ids, eval_ids, STEPS_PER_BLOCK, EVAL_EVERY)
    block_results[b] = {"best_ppl": best_ppl, "best_step": best_step, "history": history}
    ppl_current = eval_ppl(model, eval_ids)
    print(f"  Block {b} done: best={best_ppl:.3f} at step {best_step}, full-model PPL={ppl_current:.3f}")

ppl_final = eval_ppl(model, eval_ids)
print(f"\nFinal sequential PPL: {ppl_final:.3f}")

# Reference values
joint_adam_1000 = 48.111  # cycle 206
joint_adam_best = 46.556  # cycle 210

if ppl_final <= 46.0:
    verdict = "CONFIRMED"
elif ppl_final >= 60.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nResults:")
print(f"  Float PPL:           {float_ppl:.3f}")
print(f"  Init PPL:            {ppl_init:.3f}")
print(f"  Sequential final:    {ppl_final:.3f}")
print(f"  Joint Adam @1000:    {joint_adam_1000:.3f}")
print(f"  Joint Adam best:     {joint_adam_best:.3f}")
print(f"  Verdict: {verdict}")

result = {
    "cycle": 215,
    "item": 225,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "ppl_init": float(ppl_init),
    "ppl_sequential_final": float(ppl_final),
    "block_results": {str(b): block_results[b] for b in BLOCKS},
    "joint_adam_1000_ref": joint_adam_1000,
    "joint_adam_best_ref": joint_adam_best,
    "config": {"BLOCKS": BLOCKS, "K": K, "STEPS_PER_BLOCK": STEPS_PER_BLOCK,
               "LR": LR, "EVAL_EVERY": EVAL_EVERY,
               "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN, "SEED": SEED},
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
