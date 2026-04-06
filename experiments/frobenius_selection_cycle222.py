"""
Cycle 222 — Item 227: Frobenius-weighted channel selection on BLOCKS=[1,4,7] K=512.

Compare two selection rules for the topk+lstsq init pipeline used in cycle 206:
  Variant A (BASELINE): score[c] = act_std[c] * ||W_proj[c]||
  Variant B (Frobenius-importance): score[c] = ||W_proj[c]||^2 * sum_t gelu(z[t,c])^2

For each variant we build CompressedMLPs for blocks [1,4,7], fit W_out via lstsq,
patch all three blocks into the float GPT-2, and report full-model init PPL.

Verdict (item 227 thresholds):
  Confirmed   : best_alt init PPL <= baseline_mean - 1.0 (i.e. <=60.75 with baseline 61.746)
  Refuted     : best_alt improvement < 0.3 PPL
  Inconclusive: 0.3 <= improvement < 1.0
"""

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# Match cycle 206 exactly
BLOCKS = [1, 4, 7]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
SEEDS = [42, 137, 2024]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_222_frobenius_selection.json"

print(f"Device: {DEVICE}")


def load_data():
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
    return calib_ids, eval_ids


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


def select_channels(W_fc, b_fc, W_proj, ln_out_cpu, variant: str, K: int):
    """Return topk_idx (LongTensor of size K) according to variant rule.
    W_fc: (768, 3072), b_fc: (3072,), W_proj: (3072, 768)."""
    act_all = F.gelu(ln_out_cpu @ W_fc + b_fc)  # (N, 3072)
    if variant == "topk_act_std":
        scores = act_all.std(0) * W_proj.norm(dim=1)
    elif variant == "frobenius_importance":
        # ||W_proj_row||^2 * sum_t gelu(z[t,c])^2
        proj_norm_sq = (W_proj.norm(dim=1)) ** 2  # (3072,)
        act_sq_sum = (act_all ** 2).sum(dim=0)    # (3072,)
        scores = proj_norm_sq * act_sq_sum
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return scores.topk(K).indices


def fit_block_lstsq_with_idx(W_fc, b_fc, ln_out_cpu, proj_out_cpu, topk_idx, K):
    """Build CompressedMLP for fixed channel set."""
    W_fc_sub = W_fc[:, topk_idx]
    b_fc_sub = b_fc[topk_idx]
    act_sub = F.gelu(ln_out_cpu @ W_fc_sub + b_fc_sub)
    ones = torch.ones(act_sub.shape[0], 1)
    design = torch.cat([act_sub, ones], dim=1)
    sol = torch.linalg.lstsq(design, proj_out_cpu, rcond=None).solution
    W_out = sol[:K].T  # (768, K)
    b_out = sol[K]     # (768,)
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


def build_and_eval(variant, calib_ids, eval_ids):
    """Build a fresh GPT-2, patch BLOCKS with the variant's CompressedMLPs, eval PPL.
    Returns (ppl, idx_per_block)."""
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    ln_outs, proj_outs = collect_mlp_io(model, calib_ids, BLOCKS)

    idx_per_block = {}
    for b in BLOCKS:
        block = model.transformer.h[b]
        W_fc = block.mlp.c_fc.weight.data.float().cpu()
        b_fc = block.mlp.c_fc.bias.data.float().cpu()
        W_proj = block.mlp.c_proj.weight.data.float().cpu()

        topk_idx = select_channels(W_fc, b_fc, W_proj, ln_outs[b], variant, K)
        assert topk_idx.numel() == K, f"selected {topk_idx.numel()} != K={K}"
        idx_per_block[b] = topk_idx.clone()

        compressed = fit_block_lstsq_with_idx(
            W_fc, b_fc, ln_outs[b], proj_outs[b], topk_idx, K
        )
        model.transformer.h[b].mlp = compressed.to(DEVICE)

    ppl = eval_ppl(model, eval_ids)
    del model
    return ppl, idx_per_block


# ── Main ──────────────────────────────────────────────────────────────────────
calib_ids, eval_ids = load_data()
print(f"Calibration: {len(calib_ids)} seqs, Eval: {len(eval_ids)} seqs")

# Float baseline (single)
print("\nFloat baseline PPL...")
mfloat = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
mfloat.eval()
float_ppl = eval_ppl(mfloat, eval_ids)
print(f"Float PPL: {float_ppl:.3f}")
del mfloat

# Variant A: standard topk over 3 seeds
print("\n=== Variant A: topk_act_std (cycle 206 baseline) ===")
ppl_per_seed_A = []
idx_A_first = None
for seed in SEEDS:
    torch.manual_seed(seed)
    ppl, idx_per_block = build_and_eval("topk_act_std", calib_ids, eval_ids)
    print(f"  seed={seed}: init PPL = {ppl:.4f}")
    ppl_per_seed_A.append(float(ppl))
    if idx_A_first is None:
        idx_A_first = idx_per_block

ppl_mean_A = sum(ppl_per_seed_A) / len(ppl_per_seed_A)
ppl_std_A = (
    sum((p - ppl_mean_A) ** 2 for p in ppl_per_seed_A) / len(ppl_per_seed_A)
) ** 0.5
print(f"  Mean: {ppl_mean_A:.4f}  Std: {ppl_std_A:.4f}")

# Variant B: Frobenius-importance (deterministic given calibration)
print("\n=== Variant B: frobenius_importance ===")
torch.manual_seed(SEEDS[0])
ppl_B, idx_B = build_and_eval("frobenius_importance", calib_ids, eval_ids)
print(f"  init PPL = {ppl_B:.4f}")

# Sanity check: assert at least one block has different selection between A and B
diff_any = False
diff_counts = {}
for b in BLOCKS:
    set_A = set(idx_A_first[b].tolist())
    set_B = set(idx_B[b].tolist())
    n_diff = len(set_A.symmetric_difference(set_B)) // 2
    diff_counts[b] = n_diff
    if n_diff > 0:
        diff_any = True
print(f"  Channel-set differences (B vs A) per block: {diff_counts}")
assert diff_any, "Frobenius scoring did not change any selection — bug?"

# Sanity check on cycle 206 reproduction
delta_repro = abs(ppl_mean_A - 61.746)
print(f"\nReproduction check vs cycle 206 init PPL 61.746: delta={delta_repro:.4f}")
if delta_repro > 0.5:
    print("WARNING: standard topk does not reproduce cycle 206 within 0.5 PPL.")

# ── Verdict ───────────────────────────────────────────────────────────────────
improvement_B = ppl_mean_A - ppl_B
print("\n=== Results ===")
print(f"  Float PPL                : {float_ppl:.4f}")
print(f"  Variant A (topk_act_std) : {ppl_mean_A:.4f} ± {ppl_std_A:.4f}")
print(f"  Variant B (frobenius)    : {ppl_B:.4f}")
print(f"  Δ (A_mean - B)           : {improvement_B:+.4f} PPL  (positive = B is better)")

if improvement_B >= 1.0:
    verdict = "Confirmed"
elif improvement_B < 0.3:
    verdict = "Refuted"
else:
    verdict = "Inconclusive"
print(f"  Verdict: {verdict}")

result = {
    "cycle": 222,
    "item": 227,
    "verdict": verdict,
    "config": {
        "blocks": BLOCKS,
        "K": K,
        "n_calib": N_CALIB,
        "n_eval": N_EVAL,
        "seq_len": SEQ_LEN,
        "seeds": SEEDS,
    },
    "float_ppl": float(float_ppl),
    "variants": [
        {
            "name": "topk_act_std",
            "init_ppl_mean": float(ppl_mean_A),
            "init_ppl_std": float(ppl_std_A),
            "ppl_per_seed": ppl_per_seed_A,
        },
        {
            "name": "frobenius_importance",
            "init_ppl": float(ppl_B),
        },
    ],
    "improvement_vs_baseline": {
        "frobenius_importance": float(improvement_B),
    },
    "channel_set_diff_per_block_B_vs_A": {str(k): int(v) for k, v in diff_counts.items()},
    "cycle_206_reproduction_delta": float(delta_repro),
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {OUTPUT_PATH}")
