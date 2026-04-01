"""
Cycle 204 — Item 215: 3-block selection sweep.
Is [1,4,7] optimal among all evenly-spaced 3-block selections from {1,...,10}?
Anchor: BLOCKS=[1,4,7] K=512 → 61.75 PPL (+14.6% above float=53.895).
"""
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_204_block3_selection.json"

# All reasonable 3-block selections from {1,...,10} (excluding 0 and 11)
# Include various strides and starting points
BLOCK_SETS = {
    "147":   [1, 4, 7],   # anchor
    "258":   [2, 5, 8],   # stride-3, offset-2
    "369":   [3, 6, 9],   # stride-3, offset-3
    "1_5_9": [1, 5, 9],   # stride-4
    "2_6_10":[2, 6, 10],  # stride-4, offset-2
    "3_6_9": [3, 6, 9],   # same as 369
    "1_4_10":[1, 4, 10],  # dense early, late
    "1_7_10":[1, 7, 10],  # early, late-heavy
    "2_5_10":[2, 5, 10],  # mixed
    "1_3_8": [1, 3, 8],   # irregular
    "3_7_10":[3, 7, 10],  # later blocks
    "2_7_10":[2, 7, 10],  # mixed
    "1_6_10":[1, 6, 10],  # stride-5
}

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
        self.register_buffer("W_out", W_out)
        self.register_buffer("b_out", b_out)

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


def fit_block(block, ln_out_cpu, proj_out_cpu, K):
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
    W_out = sol[:K].T
    b_out = sol[K]
    return CompressedMLP(W_fc_sub, b_fc_sub, W_out, b_out)


def eval_ppl(model, eval_ids):
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

# ── Sweep ─────────────────────────────────────────────────────────────────────
# Deduplicate block sets
seen = {}
unique_sets = {}
for name, blist in BLOCK_SETS.items():
    key = tuple(sorted(blist))
    if key not in seen:
        seen[key] = name
        unique_sets[name] = blist

results = []
for name, block_list in unique_sets.items():
    print(f"\nBLOCKS={block_list}...", flush=True)
    model_k = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model_k.eval()
    ln_outs, proj_outs = collect_mlp_io(model_k, calib_ids, block_list)
    for b in block_list:
        compressed = fit_block(model_k.transformer.h[b], ln_outs[b], proj_outs[b], K)
        model_k.transformer.h[b].mlp = compressed.to(DEVICE)
    ppl = eval_ppl(model_k, eval_ids)
    pct = (ppl / float_ppl - 1) * 100
    print(f"  PPL={ppl:.3f} (+{pct:.1f}%)", flush=True)
    results.append({"name": name, "blocks": block_list, "ppl": float(ppl), "pct_above_float": float(pct)})
    del model_k

# ── Verdict ───────────────────────────────────────────────────────────────────
anchor_ppl = next(r["ppl"] for r in results if r["name"] == "147")
best = min(results, key=lambda r: r["ppl"])
any_better = [r for r in results if r["ppl"] < anchor_ppl - 1.0]

if any_better:
    verdict = "CONFIRMED"   # found a better 3-block selection
elif all(abs(r["ppl"] - anchor_ppl) <= 1.0 for r in results):
    verdict = "INCONCLUSIVE"  # all within noise
else:
    verdict = "REFUTED"     # [1,4,7] is clearly best

print(f"\nResults (sorted by PPL):")
for r in sorted(results, key=lambda x: x["ppl"]):
    marker = " ← BEST" if r["name"] == best["name"] else ("  ← anchor" if r["name"] == "147" else "")
    print(f"  {str(r['blocks']):20s}: PPL={r['ppl']:.3f} (+{r['pct_above_float']:.1f}%){marker}")
print(f"\nAnchor [1,4,7] PPL: {anchor_ppl:.3f}")
print(f"Best: {best['blocks']} PPL={best['ppl']:.3f}")
print(f"Verdict: {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "cycle": 204,
    "item": 215,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "anchor_ppl_147": float(anchor_ppl),
    "best_blocks": best["blocks"],
    "best_ppl": best["ppl"],
    "improvement_over_anchor": float(anchor_ppl - best["ppl"]),
    "results": results,
    "config": {"K": K, "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN, "SEED": SEED},
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
