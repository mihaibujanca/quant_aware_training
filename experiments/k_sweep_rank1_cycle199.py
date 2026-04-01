"""
Cycle 199 — Item 209: K-sweep for near-rank-1-input blocks.
Blocks 4,7,11 have input_rank_90=1 (cycle 198); block 1 has rank=207.
Test K={8,16,32,64,128,256,512} for BLOCKS=[1,4,7,11] with topk+lstsq.
"""
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
K_VALUES = [8, 16, 32, 64, 128, 256, 512]
BLOCKS = [1, 4, 7, 11]
N_CALIB = 200    # sequences for fitting
N_EVAL = 100     # separate sequences for PPL evaluation
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_199_k_sweep.json"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading GPT-2...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ── Load calibration + eval data ──────────────────────────────────────────────
print("Loading data...")
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

# ── CompressedMLP ─────────────────────────────────────────────────────────────

class CompressedMLP(nn.Module):
    def __init__(self, fc_weight, fc_bias, W_out, b_out):
        super().__init__()
        self.register_buffer("fc_weight", fc_weight)  # (768, K)
        self.register_buffer("fc_bias", fc_bias)       # (K,)
        self.register_buffer("W_out", W_out)           # (768, K)
        self.register_buffer("b_out", b_out)           # (768,)

    def forward(self, x):  # x: (B, T, 768) — LN-2 output
        act = F.gelu(x @ self.fc_weight + self.fc_bias)  # (B, T, K)
        return act @ self.W_out.T + self.b_out            # (B, T, 768)


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_mlp_io(model, calib_ids, block_ids):
    """Collect (ln_out, proj_out) for each block via hooks on block.mlp."""
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
        ln_outs[b] = torch.cat(ln_outs[b], dim=0)    # (N_tokens, 768)
        proj_outs[b] = torch.cat(proj_outs[b], dim=0) # (N_tokens, 768)
    return ln_outs, proj_outs


def fit_block(block, ln_out_cpu, proj_out_cpu, K):
    """Fit topk+lstsq compressed MLP for one block at given K."""
    # Conv1D layout: c_fc.weight=(768, 3072), c_proj.weight=(3072, 768)
    W_fc = block.mlp.c_fc.weight.data.float().cpu()    # (768, 3072)
    b_fc = block.mlp.c_fc.bias.data.float().cpu()      # (3072,)
    W_proj = block.mlp.c_proj.weight.data.float().cpu() # (3072, 768)

    # Score channels by activation std × c_proj output norm
    act_all = F.gelu(ln_out_cpu @ W_fc + b_fc)  # (N_tokens, 3072)
    scores = act_all.std(0) * W_proj.norm(dim=1)  # (3072,)
    topk_idx = scores.topk(K).indices             # (K,)

    W_fc_sub = W_fc[:, topk_idx]   # (768, K)
    b_fc_sub = b_fc[topk_idx]      # (K,)
    act_sub = act_all[:, topk_idx] # (N_tokens, K)

    # lstsq with bias: design=[act_sub | ones] → (N, K+1)
    ones = torch.ones(act_sub.shape[0], 1)
    design = torch.cat([act_sub, ones], dim=1)  # (N, K+1)
    sol = torch.linalg.lstsq(design, proj_out_cpu, rcond=None).solution  # (K+1, 768)
    W_out = sol[:K].T   # (768, K)
    b_out = sol[K]      # (768,)

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

# ── K-sweep ───────────────────────────────────────────────────────────────────
k_sweep = []
for K in K_VALUES:
    print(f"\nK={K}...", flush=True)
    model_k = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model_k.eval()

    # Collect I/O from float model
    ln_outs, proj_outs = collect_mlp_io(model_k, calib_ids, BLOCKS)

    # Fit and replace each block
    for b in BLOCKS:
        compressed = fit_block(model_k.transformer.h[b], ln_outs[b], proj_outs[b], K)
        model_k.transformer.h[b].mlp = compressed.to(DEVICE)

    ppl = eval_ppl(model_k, eval_ids)
    print(f"  K={K:4d}: PPL={ppl:.3f}", flush=True)
    k_sweep.append({"K": K, "ppl": float(ppl)})
    del model_k

# ── Verdict ───────────────────────────────────────────────────────────────────
baseline_ppl = k_sweep[-1]["ppl"]  # K=512 from this run
threshold_ppl = baseline_ppl * 1.05
within_5pct = [r for r in k_sweep if r["ppl"] <= threshold_ppl]
min_k_within = min(r["K"] for r in within_5pct) if within_5pct else None

if min_k_within is not None and min_k_within <= 64:
    verdict = "CONFIRMED"
elif min_k_within is None or min_k_within >= 256:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nK-sweep results:")
for r in k_sweep:
    marker = " ← within 5% of K=512" if r["ppl"] <= threshold_ppl else ""
    print(f"  K={r['K']:4d}: PPL={r['ppl']:.3f}{marker}")
print(f"\nBaseline (K=512): {baseline_ppl:.3f}")
print(f"5% threshold: {threshold_ppl:.3f}")
print(f"Min K within 5%: {min_k_within}")
print(f"Verdict: {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "cycle": 199,
    "item": 209,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "baseline_k512_ppl": float(baseline_ppl),
    "threshold_ppl_5pct": float(threshold_ppl),
    "min_k_within_5pct": min_k_within,
    "k_sweep": k_sweep,
    "config": {
        "K_VALUES": K_VALUES,
        "BLOCKS": BLOCKS,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "SEED": SEED,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
