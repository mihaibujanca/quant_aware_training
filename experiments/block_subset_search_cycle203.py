"""
Cycle 203 — Item 214: Block subset search — more compressible blocks from {2,3,5,6,8,9,10}.
Anchor: BLOCKS=[1,4,7] K=512 → 61.746 PPL (+14.6% above float).
Test supersets by adding middle blocks to see if cascade is acceptable.
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
OUTPUT_PATH = "autoresearch/results/cycle_203_block_subset_search.json"

# Block sets to test — all are supersets of [1,4,7] excluding blocks 0 and 11
BLOCK_SETS = {
    "147":       [1, 4, 7],           # anchor (cycle 202)
    "12347":     [1, 2, 3, 4, 7],     # add 2,3
    "14567":     [1, 4, 5, 6, 7],     # add 5,6
    "14789":     [1, 4, 7, 8, 9],     # add 8,9
    "1234567":   [1, 2, 3, 4, 5, 6, 7],  # 7 blocks, all 1-7
    "1234789":   [1, 2, 3, 4, 7, 8, 9],  # 7 blocks, 1-4 + 7-9
    "147_8_9_10":[1, 4, 7, 8, 9, 10],   # add 8,9,10
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
print("Computing float baseline PPL...")
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()
float_ppl = eval_ppl(model_float, eval_ids)
print(f"Float PPL: {float_ppl:.3f}")
del model_float

# ── Sweep block sets ──────────────────────────────────────────────────────────
results = []
for name, block_list in BLOCK_SETS.items():
    print(f"\nBLOCKS={block_list}...", flush=True)
    model_k = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model_k.eval()
    ln_outs, proj_outs = collect_mlp_io(model_k, calib_ids, block_list)
    for b in block_list:
        compressed = fit_block(model_k.transformer.h[b], ln_outs[b], proj_outs[b], K)
        model_k.transformer.h[b].mlp = compressed.to(DEVICE)
    ppl = eval_ppl(model_k, eval_ids)
    pct = (ppl / float_ppl - 1) * 100
    print(f"  BLOCKS={block_list}: PPL={ppl:.3f} (+{pct:.1f}% above float)", flush=True)
    results.append({
        "name": name,
        "blocks": block_list,
        "n_blocks": len(block_list),
        "ppl": float(ppl),
        "pct_above_float": float(pct),
    })
    del model_k

# ── Verdict ───────────────────────────────────────────────────────────────────
anchor_ppl = next(r["ppl"] for r in results if r["name"] == "147")
threshold_ppl = anchor_ppl + 2.0  # within 2 PPL of anchor

better_sets = [r for r in results if r["n_blocks"] > 3 and r["ppl"] <= threshold_ppl]
any_improvement = [r for r in results if r["ppl"] < anchor_ppl]

if any_improvement:
    verdict = "CONFIRMED"
elif all(r["ppl"] > anchor_ppl + 5.0 for r in results if r["n_blocks"] > 3):
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nResults summary:")
for r in sorted(results, key=lambda x: x["ppl"]):
    marker = " ← BETTER than anchor" if r["ppl"] < anchor_ppl else ""
    print(f"  {r['name']:15s}: PPL={r['ppl']:.3f} (+{r['pct_above_float']:.1f}%, {r['n_blocks']} blocks){marker}")
print(f"\nAnchor [1,4,7] PPL: {anchor_ppl:.3f}")
print(f"Within-2PPL supersets: {[r['name'] for r in better_sets]}")
print(f"Any improvement: {[r['name'] for r in any_improvement]}")
print(f"Verdict: {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "cycle": 203,
    "item": 214,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "anchor_ppl_147": float(anchor_ppl),
    "results": results,
    "better_than_anchor": [r["name"] for r in any_improvement],
    "within_2ppl_of_anchor": [r["name"] for r in better_sets],
    "config": {
        "K": K,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "SEED": SEED,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
