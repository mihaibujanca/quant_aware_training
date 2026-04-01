"""
Cycle 200 — Item 210: Residual stream rank trajectory under compression.
Compare input_rank_90 per block for float vs compressed GPT-2 (BLOCKS=[1,4,7,11], K=512).
Cycle 198: float input ranks 269→207→25→1 by block 3. Does compression disrupt this?
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS_COMPRESSED = [1, 4, 7, 11]
K = 512
N_CALIB = 200    # sequences for fitting
N_RANK = 50      # sequences for rank measurement (same config as cycle 198)
SEQ_LEN = 64     # same as cycle 198
SEED = 42
ENERGY_THRESH = 0.90
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_200_rank_trajectory.json"

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
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max(SEQ_LEN, 128))
    if enc["input_ids"].shape[1] >= 128:
        all_ids.append(enc["input_ids"])
    if len(all_ids) >= N_CALIB + N_RANK:
        break

# First N_CALIB for fitting (SEQ_LEN=128 to match cycle 196/199)
calib_ids = [t[:, :128].to(DEVICE) for t in all_ids[:N_CALIB]]
# Next N_RANK for rank measurement (SEQ_LEN=64 to match cycle 198)
rank_ids = [t[:, :SEQ_LEN].to(DEVICE) for t in all_ids[N_CALIB:N_CALIB + N_RANK]]
print(f"Calibration: {len(calib_ids)} seqs @ 128 tokens, Rank measurement: {len(rank_ids)} seqs @ {SEQ_LEN} tokens")

# ── CompressedMLP ─────────────────────────────────────────────────────────────

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


# ── Collect residual stream states per block ──────────────────────────────────

def collect_residual_states(model, rank_ids, n_blocks=12):
    """Hook on each block's forward to capture residual stream input (inp[0])."""
    states = {b: [] for b in range(n_blocks)}
    hooks = []
    for b in range(n_blocks):
        def mk_hook(bb):
            def fn(m, inp, out):
                states[bb].append(inp[0].detach().float().cpu().reshape(-1, 768))
            return fn
        hooks.append(model.transformer.h[b].register_forward_hook(mk_hook(b)))
    with torch.no_grad():
        for ids in rank_ids:
            model(ids)
    for h in hooks:
        h.remove()
    for b in range(n_blocks):
        states[b] = torch.cat(states[b], dim=0)  # (N_tokens, 768)
    return states


def effective_rank(X, threshold=ENERGY_THRESH):
    X_c = X - X.mean(0)
    S = torch.linalg.svdvals(X_c)
    energy = (S ** 2).cumsum(0) / ((S ** 2).sum() + 1e-12)
    return int((energy < threshold).sum()) + 1


# ── Build compressed model ────────────────────────────────────────────────────
print("Building compressed model...")
model_comp = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_comp.eval()

ln_outs, proj_outs = collect_mlp_io(model_comp, calib_ids, BLOCKS_COMPRESSED)
for b in BLOCKS_COMPRESSED:
    compressed = fit_block(model_comp.transformer.h[b], ln_outs[b], proj_outs[b], K)
    model_comp.transformer.h[b].mlp = compressed.to(DEVICE)
    print(f"  Compressed block {b}", flush=True)

# ── Float model ───────────────────────────────────────────────────────────────
print("Loading float model...")
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()

# ── Collect rank trajectories ─────────────────────────────────────────────────
print("Collecting residual states...")
float_states = collect_residual_states(model_float, rank_ids)
comp_states = collect_residual_states(model_comp, rank_ids)

print("Computing ranks...")
float_ranks = []
comp_ranks = []
for b in range(12):
    fr = effective_rank(float_states[b])
    cr = effective_rank(comp_states[b])
    float_ranks.append(fr)
    comp_ranks.append(cr)
    print(f"  Block {b:2d}: float_rank={fr:4d}, comp_rank={cr:4d}, ratio={cr/max(fr,1):.2f}", flush=True)

# ── Frobenius divergence ──────────────────────────────────────────────────────
print("Computing Frobenius divergence...")
frob_div = []
for b in range(12):
    f_norms = float_states[b].norm(dim=1)
    diff_norms = (comp_states[b] - float_states[b]).norm(dim=1)
    frob_div.append(float((diff_norms / (f_norms + 1e-8)).mean().item()))

# ── Verdict ───────────────────────────────────────────────────────────────────
rank_ratios = [comp_ranks[b] / max(float_ranks[b], 1) for b in range(12)]
max_rank_disruption = max(rank_ratios)
# Key check: does compression disrupt rank at block 3 (first rank-1 block in float)?
rank_ratio_b3 = rank_ratios[3]

if rank_ratio_b3 >= 10:
    verdict = "CONFIRMED"
elif rank_ratio_b3 <= 2:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nRank trajectories:")
print(f"  Float:      {float_ranks}")
print(f"  Compressed: {comp_ranks}")
print(f"  Ratios:     {[f'{r:.1f}' for r in rank_ratios]}")
print(f"  Frob div:   {[f'{d:.3f}' for d in frob_div]}")
print(f"\nRank ratio at block 3: {rank_ratio_b3:.1f}")
print(f"Max rank disruption: {max_rank_disruption:.1f}")
print(f"Verdict: {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "cycle": 200,
    "item": 210,
    "verdict": verdict,
    "float_ranks": float_ranks,
    "compressed_ranks": comp_ranks,
    "rank_ratios": rank_ratios,
    "frob_divergence": frob_div,
    "rank_ratio_b3": float(rank_ratio_b3),
    "max_rank_disruption": float(max_rank_disruption),
    "config": {
        "BLOCKS_COMPRESSED": BLOCKS_COMPRESSED,
        "K": K,
        "N_CALIB": N_CALIB,
        "N_RANK": N_RANK,
        "SEQ_LEN": SEQ_LEN,
        "ENERGY_THRESH": ENERGY_THRESH,
        "SEED": SEED,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
