"""
Cycle 205 — Item 216: Block 1 single-block K-sweep.
Block 1 (input_rank=192) is the highest-value compressible block.
Cycle 199 K-swept all 4 blocks aggregate; this isolates block 1.
Test K={256,512,768,1024,1536} — does PPL keep improving beyond K=512?
"""
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
K_VALUES = [256, 512, 768, 1024, 1536]
BLOCK = 1   # single block to compress
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_205_block1_k_sweep.json"

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


def collect_mlp_io_single(model, calib_ids, block_id):
    ln_outs = []
    proj_outs = []
    def fn(m, inp, out):
        ln_outs.append(inp[0].detach().float().cpu().reshape(-1, 768))
        proj_outs.append(out.detach().float().cpu().reshape(-1, 768))
    hook = model.transformer.h[block_id].mlp.register_forward_hook(fn)
    with torch.no_grad():
        for ids in calib_ids:
            model(ids)
    hook.remove()
    return torch.cat(ln_outs, dim=0), torch.cat(proj_outs, dim=0)


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

# Collect once from float model
ln_out_cpu, proj_out_cpu = collect_mlp_io_single(model_float, calib_ids, BLOCK)
del model_float
print(f"Calibration I/O collected: {ln_out_cpu.shape}")

# ── K-sweep ───────────────────────────────────────────────────────────────────
k_sweep = []
for K in K_VALUES:
    print(f"\nK={K}...", flush=True)
    model_k = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model_k.eval()

    compressed = fit_block(model_k.transformer.h[BLOCK], ln_out_cpu, proj_out_cpu, K)
    model_k.transformer.h[BLOCK].mlp = compressed.to(DEVICE)

    ppl = eval_ppl(model_k, eval_ids)
    pct = (ppl / float_ppl - 1) * 100
    print(f"  K={K:5d}: PPL={ppl:.3f} (+{pct:.1f}%)", flush=True)
    k_sweep.append({"K": K, "ppl": float(ppl), "pct_above_float": float(pct)})
    del model_k

# ── Verdict ───────────────────────────────────────────────────────────────────
ppl_512 = next(r["ppl"] for r in k_sweep if r["K"] == 512)
# Check if any K>512 improves by >=0.5 PPL over K=512
improvement_beyond_512 = [r for r in k_sweep if r["K"] > 512 and r["ppl"] < ppl_512 - 0.5]
max_improvement = ppl_512 - min(r["ppl"] for r in k_sweep if r["K"] > 512)

if improvement_beyond_512:
    verdict = "CONFIRMED"   # headroom beyond K=512
elif max_improvement < 0.3:
    verdict = "REFUTED"     # K=512 already at ceiling
else:
    verdict = "INCONCLUSIVE"

print(f"\nBlock 1 K-sweep:")
for r in k_sweep:
    print(f"  K={r['K']:5d}: PPL={r['ppl']:.3f} (+{r['pct_above_float']:.1f}%)")
print(f"\nK=512 PPL: {ppl_512:.3f}")
print(f"Max improvement beyond K=512: {max_improvement:.3f}")
print(f"Verdict: {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "cycle": 205,
    "item": 216,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "ppl_at_k512": float(ppl_512),
    "max_improvement_beyond_512": float(max_improvement),
    "k_sweep": k_sweep,
    "config": {
        "K_VALUES": K_VALUES,
        "BLOCK": BLOCK,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "SEED": SEED,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
