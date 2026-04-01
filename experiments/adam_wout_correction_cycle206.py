"""
Cycle 206 — Item 217: Adam W_out correction on BLOCKS=[1,4,7] K=512 model.
Start from topk+lstsq init (PPL=61.75), freeze everything except W_out of 3
compressed MLPs (3 × 768 × K = 1.18M params), run Adam on full-model CE.
Target: reduce cascade from +14.6% to below +11.4% (PPL ≤59.0).
"""
import json
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS = [1, 4, 7]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 1000
LR = 1e-4
EVAL_EVERY = 100
OUTPUT_PATH = "autoresearch/results/cycle_206_adam_wout_correction.json"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading GPT-2 and data...")
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
        # W_out is trainable — register as parameter
        self.W_out = nn.Parameter(W_out.clone())   # (768, K)
        self.b_out = nn.Parameter(b_out.clone())   # (768,)
        self.register_buffer("fc_bias_buf", fc_bias)  # unused, but kept for shape

    def forward(self, x):
        act = F.gelu(x @ self.fc_weight + self.fc_bias)
        return act @ self.W_out.T + self.b_out


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
    """topk+lstsq fit; return frozen fc weights and trainable W_out."""
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
    design = torch.cat([act_sub, ones], dim=1)
    sol = torch.linalg.lstsq(design, proj_out_cpu, rcond=None).solution
    W_out = sol[:K].T   # (768, K)
    b_out = sol[K]      # (768,)
    return CompressedMLP(W_fc_sub, b_fc_sub, W_out, b_out)


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


# ── Float baseline ────────────────────────────────────────────────────────────
print("Float baseline PPL...")
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()
float_ppl = eval_ppl(model_float, eval_ids)
print(f"Float PPL: {float_ppl:.3f}")
del model_float

# ── Build topk+lstsq compressed model ────────────────────────────────────────
print("\nBuilding topk+lstsq compressed model...")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

ln_outs, proj_outs = collect_mlp_io(model, calib_ids, BLOCKS)
for b in BLOCKS:
    compressed = fit_block_lstsq(model.transformer.h[b], ln_outs[b], proj_outs[b], K)
    model.transformer.h[b].mlp = compressed.to(DEVICE)
    print(f"  Compressed block {b}", flush=True)

ppl_before = eval_ppl(model, eval_ids)
print(f"PPL after topk+lstsq init: {ppl_before:.3f}")

# ── Freeze everything except W_out and b_out of compressed blocks ─────────────
print("\nFreezing all params except W_out/b_out of compressed blocks...")
trainable_params = []
for param in model.parameters():
    param.requires_grad_(False)

for b in BLOCKS:
    mlp = model.transformer.h[b].mlp
    mlp.W_out.requires_grad_(True)
    mlp.b_out.requires_grad_(True)
    trainable_params.extend([mlp.W_out, mlp.b_out])

n_trainable = sum(p.numel() for p in trainable_params)
print(f"Trainable params: {n_trainable:,} ({n_trainable / 1e6:.2f}M)")

# ── Adam training ─────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(trainable_params, lr=LR)
import random
random.shuffle(calib_ids)

training_log = []
ppl_eval_history = []

print(f"\nTraining {N_STEPS} steps, LR={LR}...")
step = 0
while step < N_STEPS:
    for ids in calib_ids:
        if step >= N_STEPS:
            break
        model.train()
        # Only compressed MLPs have requires_grad=True
        optimizer.zero_grad()
        out = model(ids, labels=ids)
        loss = out.loss
        loss.backward()
        optimizer.step()
        step += 1

        if step % EVAL_EVERY == 0:
            ppl_e = eval_ppl(model, eval_ids)
            ppl_c = math.exp(loss.item())
            ppl_eval_history.append({"step": step, "eval_ppl": float(ppl_e), "calib_loss": float(loss.item())})
            print(f"  Step {step:4d}: eval_ppl={ppl_e:.3f}, calib_loss={loss.item():.4f}", flush=True)

ppl_after = eval_ppl(model, eval_ids)
print(f"\nPPL after Adam correction: {ppl_after:.3f}")

# ── Verdict ───────────────────────────────────────────────────────────────────
improvement = ppl_before - ppl_after
eval_ppls = [r["eval_ppl"] for r in ppl_eval_history]
calib_losses_final = ppl_eval_history[-1]["calib_loss"] if ppl_eval_history else None

# Check for overfitting: last eval_ppl vs second-to-last
if len(eval_ppls) >= 2:
    overfitting = eval_ppls[-1] - eval_ppls[max(0, len(eval_ppls)//2)] > 2.0
else:
    overfitting = False

if improvement >= 2.75 and not overfitting:
    verdict = "CONFIRMED"
elif improvement < 0.5:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nResults:")
print(f"  Float PPL:    {float_ppl:.3f}")
print(f"  Before Adam:  {ppl_before:.3f} (+{(ppl_before/float_ppl-1)*100:.1f}%)")
print(f"  After Adam:   {ppl_after:.3f} (+{(ppl_after/float_ppl-1)*100:.1f}%)")
print(f"  Improvement:  {improvement:.3f} PPL")
print(f"  Overfitting:  {overfitting}")
print(f"  Verdict: {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "cycle": 206,
    "item": 217,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "ppl_before_adam": float(ppl_before),
    "ppl_after_adam": float(ppl_after),
    "improvement": float(improvement),
    "overfitting": overfitting,
    "ppl_eval_history": ppl_eval_history,
    "config": {
        "BLOCKS": BLOCKS,
        "K": K,
        "N_STEPS": N_STEPS,
        "LR": LR,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "SEED": SEED,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
