"""
Cycle 209 — Item 212: Rank-1 input FFN compression — explicit 1D subspace projection.
Blocks 3-11 have input_rank_90=1. Test rank-1 parameterized FFN vs topk+lstsq.

Rank-1 FFN forward:
  s = (x - mu) @ v1          # scalar projection onto top PCA direction
  pre_act = w_1d * s + b_fc  # w_1d ∈ ℝ^K (scalars, one per neuron)
  act = gelu(pre_act)
  output = act @ W_out.T + b_out

w_1d[k] = (W_fc[:, k] @ v1) — project each neuron weight onto top PCA direction.
W_out fit via lstsq on the rank-1 activations.

Compare at:
  - BLOCK=4 (representative rank-1 block, mid-network)
  - K=512 (rank-1 vs standard at same K — ~50% fewer parameters)
  - K=256 (rank-1 at K=256 ≈ same param count as standard K=131)

Confirmed if: rank-1 K=512 PPL within 1 PPL of standard K=512 (54.689 for block 1;
              need to measure standard K=512 for block 4 fresh).
Refuted if: rank-1 PPL worse by >2 PPL vs standard at same K.
Inconclusive if: gap 1-2 PPL.
"""
import json
import math
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import torch.nn as nn
import copy

# ── Config ────────────────────────────────────────────────────────────────────
BLOCK = 4      # representative rank-1 block
K_VALUES = [256, 512, 1024]   # test at multiple K for rank-1
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_209_rank1_ffn.json"

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

def collect_block_io(model, calib_ids, block):
    ln_outs, proj_outs = [], []
    def fn(m, inp, out):
        ln_outs.append(inp[0].detach().float().cpu().reshape(-1, 768))
        proj_outs.append(out.detach().float().cpu().reshape(-1, 768))
    h = model.transformer.h[block].mlp.register_forward_hook(fn)
    with torch.no_grad():
        for ids in calib_ids:
            model(ids)
    h.remove()
    return torch.cat(ln_outs, 0), torch.cat(proj_outs, 0)


def eval_ppl_patched(model_orig, eval_ids, block, new_mlp):
    m = copy.deepcopy(model_orig)
    m.transformer.h[block].mlp = new_mlp
    m.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for ids in eval_ids:
            out = m(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    del m
    return math.exp(total_loss / total_tokens)


class StandardMLP(nn.Module):
    def __init__(self, W_fc, b_fc, W_out, b_out):
        super().__init__()
        self.register_buffer("W_fc", W_fc)
        self.register_buffer("b_fc", b_fc)
        self.register_buffer("W_out", W_out)
        self.register_buffer("b_out", b_out)
    def forward(self, x):
        return F.gelu(x @ self.W_fc + self.b_fc) @ self.W_out.T + self.b_out


class Rank1MLP(nn.Module):
    """Uses only the top-1 PCA component of the input."""
    def __init__(self, w_1d, b_fc, W_out, b_out, v1, mu):
        super().__init__()
        self.register_buffer("w_1d", w_1d)   # (K,) scalar projections
        self.register_buffer("b_fc", b_fc)   # (K,)
        self.register_buffer("W_out", W_out) # (768, K)
        self.register_buffer("b_out", b_out) # (768,)
        self.register_buffer("v1", v1)       # (768,) top PCA direction
        self.register_buffer("mu", mu)       # (768,) mean
    def forward(self, x):
        s = (x - self.mu) @ self.v1          # (..., 1) scalar
        act = F.gelu(s.unsqueeze(-1) * self.w_1d + self.b_fc)  # (..., K)
        return act @ self.W_out.T + self.b_out


def fit_standard(W_fc, b_fc, W_proj, ln_out_cpu, proj_out_cpu, K):
    act_all = F.gelu(ln_out_cpu @ W_fc + b_fc)
    scores = act_all.std(0) * W_proj.norm(dim=1)
    topk_idx = scores.topk(K).indices
    W_fc_sub = W_fc[:, topk_idx]
    b_fc_sub = b_fc[topk_idx]
    act_sub = act_all[:, topk_idx]
    ones = torch.ones(act_sub.shape[0], 1)
    sol = torch.linalg.lstsq(torch.cat([act_sub, ones], 1), proj_out_cpu, rcond=None).solution
    return StandardMLP(W_fc_sub, b_fc_sub, sol[:K].T, sol[K])


def fit_rank1(W_fc, b_fc, W_proj, ln_out_cpu, proj_out_cpu, K, v1, mu):
    """Rank-1 FFN: use only scalar projection onto top PCA direction."""
    # Compute rank-1 activations
    s_all = (ln_out_cpu - mu) @ v1           # (N,) scalars
    # w_1d[k] = W_fc[:, k] @ v1
    w_1d_full = W_fc.T @ v1                  # (3072,) — scalar weight per neuron
    b_fc_full = b_fc + mu @ W_fc             # (3072,) — absorb mean contribution
    # Note: (x-mu)@v1 * (W_fc[:,k]@v1) + (mu@W_fc[k]+b_fc[k])
    #     = x@W_fc[:,k] when x is exactly mu + s*v1

    act_full = F.gelu(s_all.unsqueeze(1) * w_1d_full + b_fc_full)  # (N, 3072)
    scores = act_full.std(0) * W_proj.norm(dim=1)
    topk_idx = scores.topk(K).indices

    w_1d_sub = w_1d_full[topk_idx]           # (K,)
    b_fc_sub = b_fc_full[topk_idx]           # (K,)
    act_sub = act_full[:, topk_idx]          # (N, K)

    ones = torch.ones(act_sub.shape[0], 1)
    sol = torch.linalg.lstsq(torch.cat([act_sub, ones], 1), proj_out_cpu, rcond=None).solution
    return Rank1MLP(w_1d_sub, b_fc_sub, sol[:K].T, sol[K], v1, mu)


# ── Float baseline ────────────────────────────────────────────────────────────
print("Float baseline PPL...")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
total_loss, total_tokens = 0.0, 0
with torch.no_grad():
    for ids in eval_ids:
        out = model(ids, labels=ids)
        total_loss += out.loss.item() * (ids.shape[1] - 1)
        total_tokens += ids.shape[1] - 1
float_ppl = math.exp(total_loss / total_tokens)
print(f"Float PPL: {float_ppl:.3f}")

# ── Collect block I/O ─────────────────────────────────────────────────────────
print(f"Collecting block {BLOCK} I/O...")
ln_out_cpu, proj_out_cpu = collect_block_io(model, calib_ids, BLOCK)
print(f"  ln_out shape: {ln_out_cpu.shape}")

W_fc = model.transformer.h[BLOCK].mlp.c_fc.weight.data.float().cpu()
b_fc = model.transformer.h[BLOCK].mlp.c_fc.bias.data.float().cpu()
W_proj = model.transformer.h[BLOCK].mlp.c_proj.weight.data.float().cpu()

# Compute top PCA direction of block inputs
mu = ln_out_cpu.mean(0)
X_centered = ln_out_cpu - mu
_, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
v1 = Vh[0]   # (768,) top direction

# Measure input rank
sv = torch.linalg.svdvals(X_centered)
energy = (sv**2).cumsum(0) / (sv**2).sum()
rank_90 = int((energy < 0.90).sum()) + 1
print(f"  Block {BLOCK} input_rank_90={rank_90}")

# ── Run experiments ────────────────────────────────────────────────────────────
results = {}

for K in K_VALUES:
    print(f"\nK={K}:")
    # Standard
    mlp_std = fit_standard(W_fc, b_fc, W_proj, ln_out_cpu, proj_out_cpu, K)
    ppl_std = eval_ppl_patched(model, eval_ids, BLOCK, mlp_std.to(DEVICE))
    params_std = K*768 + K + K*768 + 768
    print(f"  Standard K={K}: PPL={ppl_std:.3f}  params={params_std:,}")

    # Rank-1
    mlp_r1 = fit_rank1(W_fc, b_fc, W_proj, ln_out_cpu, proj_out_cpu, K, v1, mu)
    ppl_r1 = eval_ppl_patched(model, eval_ids, BLOCK, mlp_r1.to(DEVICE))
    params_r1 = K + K + K*768 + 768 + 768 + 768  # w_1d, b_fc, W_out, b_out, v1, mu
    print(f"  Rank-1  K={K}: PPL={ppl_r1:.3f}  params={params_r1:,}")

    results[f"standard_k{K}"] = {"ppl": float(ppl_std), "params": params_std}
    results[f"rank1_k{K}"] = {"ppl": float(ppl_r1), "params": params_r1}

# ── Verdict ───────────────────────────────────────────────────────────────────
# Primary comparison: rank-1 K=512 vs standard K=512
r1_512 = results["rank1_k512"]["ppl"]
std_512 = results["standard_k512"]["ppl"]
gap_512 = r1_512 - std_512

# Secondary: rank-1 K=512 vs standard K=256 (comparable params)
r1_256 = results["rank1_k256"]["ppl"]
std_256 = results["standard_k256"]["ppl"]

if gap_512 <= 1.0:
    verdict = "CONFIRMED"
elif gap_512 > 2.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nResults (block {BLOCK}):")
print(f"  Float PPL: {float_ppl:.3f}")
for k, v in results.items():
    print(f"  {k}: PPL={v['ppl']:.3f}  params={v['params']:,}")
print(f"  Gap rank-1 vs standard at K=512: {gap_512:+.3f}")
print(f"  Verdict: {verdict}")

result = {
    "cycle": 209,
    "item": 212,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "block": BLOCK,
    "input_rank_90": int(rank_90),
    "results": results,
    "gap_rank1_vs_standard_k512": float(gap_512),
    "config": {
        "BLOCK": BLOCK,
        "K_VALUES": K_VALUES,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "SEED": SEED,
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
