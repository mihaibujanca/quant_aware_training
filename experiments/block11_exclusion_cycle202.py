"""
Cycle 202 — Item 213: BLOCKS=[1,4,7] vs [1,4,7,11] compression.
Cycle 201 found block 11 is output decompressor (in_rank=2→out_rank=231).
Hypothesis: excluding block 11 from compression improves PPL.
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
OUTPUT_PATH = "autoresearch/results/cycle_202_block11_exclusion.json"

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
        self.register_buffer("fc_weight", fc_weight)  # (768, K)
        self.register_buffer("fc_bias", fc_bias)       # (K,)
        self.register_buffer("W_out", W_out)           # (768, K)
        self.register_buffer("b_out", b_out)           # (768,)

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

# ── Experiment: BLOCKS=[1,4,7] (exclude block 11) ────────────────────────────
print("\nCompressing BLOCKS=[1,4,7] (excluding block 11)...")
blocks_147 = [1, 4, 7]
model_147 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_147.eval()
ln_outs, proj_outs = collect_mlp_io(model_147, calib_ids, blocks_147)
for b in blocks_147:
    compressed = fit_block(model_147.transformer.h[b], ln_outs[b], proj_outs[b], K)
    model_147.transformer.h[b].mlp = compressed.to(DEVICE)
    print(f"  Compressed block {b}", flush=True)
ppl_147 = eval_ppl(model_147, eval_ids)
print(f"BLOCKS=[1,4,7] PPL: {ppl_147:.3f}", flush=True)
del model_147

# ── Experiment: BLOCKS=[1,4,7,11] (current recipe) ───────────────────────────
print("\nCompressing BLOCKS=[1,4,7,11] (current recipe)...")
blocks_14711 = [1, 4, 7, 11]
model_14711 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_14711.eval()
ln_outs, proj_outs = collect_mlp_io(model_14711, calib_ids, blocks_14711)
for b in blocks_14711:
    compressed = fit_block(model_14711.transformer.h[b], ln_outs[b], proj_outs[b], K)
    model_14711.transformer.h[b].mlp = compressed.to(DEVICE)
    print(f"  Compressed block {b}", flush=True)
ppl_14711 = eval_ppl(model_14711, eval_ids)
print(f"BLOCKS=[1,4,7,11] PPL: {ppl_14711:.3f}", flush=True)
del model_14711

# ── Verdict ───────────────────────────────────────────────────────────────────
improvement = ppl_14711 - ppl_147  # positive = [1,4,7] is better
if improvement > 1.0:
    verdict = "CONFIRMED"   # excluding block 11 improves PPL
elif improvement < -1.0:
    verdict = "REFUTED"     # including block 11 in compression is better (unexpected)
else:
    verdict = "INCONCLUSIVE"  # difference within noise

print(f"\nResults:")
print(f"  Float PPL:             {float_ppl:.3f}")
print(f"  BLOCKS=[1,4,7] PPL:    {ppl_147:.3f}  (+{(ppl_147/float_ppl-1)*100:.1f}% above float)")
print(f"  BLOCKS=[1,4,7,11] PPL: {ppl_14711:.3f} (+{(ppl_14711/float_ppl-1)*100:.1f}% above float)")
print(f"  Improvement from excluding block 11: {improvement:.3f} PPL")
print(f"  Verdict: {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "cycle": 202,
    "item": 213,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "ppl_blocks_147": float(ppl_147),
    "ppl_blocks_14711": float(ppl_14711),
    "improvement_from_exclusion": float(improvement),
    "pct_above_float_147": float((ppl_147 / float_ppl - 1) * 100),
    "pct_above_float_14711": float((ppl_14711 / float_ppl - 1) * 100),
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
