"""
Cycle 196 — Item 207: LayerNorm recalibration for stacked compressed blocks
Test if recalibrating LN γ/β after topk+lstsq compression on 4 blocks reduces cascade PPL.
Uses blocks [1,4,7,11] at k=512 channels (6x compression) — same config as cycle 185.
"""
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS = [1, 4, 7, 11]       # same as cycle 185 working recipe
K = 512                        # channels (3072/6 = 512, 6x compression)
N_SEQS_CALIB = 200
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_196_ln_recal.json"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading GPT-2...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# ── Calibration and evaluation data ──────────────────────────────────────────
print("Loading data...")
train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [r["text"] for r in train_ds if len(r["text"].strip()) > 50]
calib_ids = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        calib_ids.append(enc["input_ids"][:, :SEQ_LEN].to(DEVICE))
    if len(calib_ids) >= N_SEQS_CALIB:
        break

val_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
val_text = "\n\n".join([r["text"] for r in val_ds])
val_ids = tokenizer(val_text, return_tensors="pt")["input_ids"].to(DEVICE)

print(f"Calibration: {len(calib_ids)} seqs × {SEQ_LEN} = {len(calib_ids) * SEQ_LEN} tokens")


def compute_ppl(m, max_tokens=50000, stride=512, max_length=1024):
    m.eval()
    ids = val_ids[:, :max_tokens]
    T = ids.shape[1]
    nll_sum, n_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, T - 1, stride):
            begin = max(0, i - max_length + stride)
            end = min(T, i + stride)
            chunk = ids[:, begin:end + 1]
            tl = end - i
            out = m(chunk)
            logits = out.logits[:, -tl - 1:-1, :]
            targets = chunk[:, -tl:]
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                  targets.reshape(-1), reduction="sum")
            nll_sum += nll.item()
            n_tokens += tl
    return float(torch.exp(torch.tensor(nll_sum / n_tokens)))


print("Computing float PPL...")
float_ppl = compute_ppl(model)
print(f"Float PPL: {float_ppl:.3f}")

# ── Collect MLP I/O for selected blocks ──────────────────────────────────────
print(f"Collecting MLP I/O for blocks {BLOCKS} via hooks...")
mlp_in = {b: [] for b in BLOCKS}
mlp_out = {b: [] for b in BLOCKS}
hooks = []

for b in BLOCKS:
    def make_hook(bb):
        def hook_fn(m, inp, out):
            mlp_in[bb].append(inp[0].detach().float().cpu().reshape(-1, 768))
            mlp_out[bb].append(out.detach().float().cpu().reshape(-1, 768))
        return hook_fn
    hooks.append(model.transformer.h[b].mlp.register_forward_hook(make_hook(b)))

with torch.no_grad():
    for ids in calib_ids:
        model(ids)

for h in hooks:
    h.remove()

for b in BLOCKS:
    mlp_in[b] = torch.cat(mlp_in[b], dim=0)   # (N_tokens, 768)
    mlp_out[b] = torch.cat(mlp_out[b], dim=0)  # (N_tokens, 768)

N_tokens = mlp_in[BLOCKS[0]].shape[0]
print(f"Tokens per block: {N_tokens}")

# ── Fit topk+lstsq for each selected block ────────────────────────────────────
print(f"Fitting topk+lstsq (K={K}) for blocks {BLOCKS}...")

class CompressedMLP(nn.Module):
    """Topk+lstsq compressed FFN block.
    Keeps the original c_fc input projection, selects top-K channels,
    and replaces c_proj with a fitted lstsq projection.
    """
    def __init__(self, fc_weight, fc_bias, topk_idx, W_out, b_out):
        super().__init__()
        # fc_weight: (768, K) — subsetted input projection (K selected channels)
        # W_out: (768, K) — fitted output projection
        # b_out: (768,) — fitted output bias
        self.register_buffer("fc_weight", fc_weight)  # (768, K)
        self.register_buffer("fc_bias", fc_bias)       # (K,)
        self.register_buffer("W_out", W_out)           # (768, K)
        self.register_buffer("b_out", b_out)           # (768,)

    def forward(self, x):  # x: (B, T, 768)
        fc = x @ self.fc_weight + self.fc_bias   # (B, T, K)
        act = F.gelu(fc)                          # (B, T, K)
        return act @ self.W_out.T + self.b_out   # (B, T, 768)


topk_fits = {}

with torch.no_grad():
    for b in BLOCKS:
        block = model.transformer.h[b]
        ln_out = mlp_in[b].to(DEVICE)      # (N_tokens, 768) — actual MLP input (ln_2 output)
        proj_out = mlp_out[b].to(DEVICE)   # (N_tokens, 768) — actual MLP output

        # Full hidden activations for channel selection
        fc_full = ln_out @ block.mlp.c_fc.weight + block.mlp.c_fc.bias  # (N_tokens, 3072)
        act_full = F.gelu(fc_full)

        # Channel importance: std × output_norm × (mean_act + 1e-6)
        c_proj_w = block.mlp.c_proj.weight  # (3072, 768) Conv1D — c_proj(x)=x@w+b
        output_norms = c_proj_w.norm(dim=1)  # (3072,) — norm of each row = output contribution per channel
        std_h = act_full.std(dim=0)
        mean_h = act_full.mean(dim=0)
        scores = std_h * output_norms * (mean_h.abs() + 1e-6)
        topk_idx = scores.topk(K).indices  # (K,)

        # Subsetted input weights
        fc_w_sub = block.mlp.c_fc.weight[:, topk_idx].contiguous().cpu()  # (768, K)
        fc_b_sub = block.mlp.c_fc.bias[topk_idx].contiguous().cpu()       # (K,)

        # Hidden activations for selected channels
        act_sub = act_full[:, topk_idx]  # (N_tokens, K)

        # lstsq: proj_out ≈ act_sub @ W_out.T + b_out
        # Include bias term: design = [act_sub, ones]
        ones = torch.ones(N_tokens, 1, device=DEVICE)
        design = torch.cat([act_sub.float(), ones], dim=1)  # (N_tokens, K+1)
        solution, _, _, _ = torch.linalg.lstsq(
            design.cpu(), proj_out.float().cpu(), rcond=None
        )  # (K+1, 768)
        W_out = solution[:K].T.contiguous()  # (768, K)
        b_out = solution[K].contiguous()      # (768,)

        topk_fits[b] = (fc_w_sub, fc_b_sub, topk_idx.cpu(), W_out, b_out)
        print(f"  Block {b:2d}: fitted, W_out range [{W_out.min():.3f}, {W_out.max():.3f}]", flush=True)

# ── Build compressed model ────────────────────────────────────────────────────
print("Building compressed model...")
comp_model = copy.deepcopy(model)
comp_model.eval()

for b in BLOCKS:
    fc_w, fc_b, idx, W_out, b_out = topk_fits[b]
    comp_model.transformer.h[b].mlp = CompressedMLP(
        fc_w.to(DEVICE), fc_b.to(DEVICE), idx.to(DEVICE), W_out.to(DEVICE), b_out.to(DEVICE)
    )

print("Computing compressed PPL (before recalibration)...")
comp_ppl_before = compute_ppl(comp_model)
print(f"Compressed PPL (before recal): {comp_ppl_before:.3f}")

# ── Collect LN input statistics (float vs compressed) ────────────────────────
print("Collecting LN-2 statistics (float vs compressed)...")


def collect_ln2_stats(m, calib_ids):
    """Per-channel mean and std of ln_2 inputs for all 12 blocks."""
    stats = {}
    inputs_per_block = {b: [] for b in range(12)}
    hooks = []

    for b in range(12):
        def mk(bb):
            def fn(module, inp, out):
                inputs_per_block[bb].append(inp[0].detach().float().cpu().reshape(-1, 768))
            return fn
        hooks.append(m.transformer.h[b].ln_2.register_forward_hook(mk(b)))

    with torch.no_grad():
        for ids in calib_ids:
            m(ids)

    for h in hooks:
        h.remove()

    for b in range(12):
        data = torch.cat(inputs_per_block[b], dim=0)  # (N_tokens, 768)
        stats[b] = (data.mean(0), data.std(0) + 1e-6)

    return stats


float_ln2 = collect_ln2_stats(model, calib_ids)
comp_ln2 = collect_ln2_stats(comp_model, calib_ids)

# ── Recalibrate LN-2 for compressed blocks only ───────────────────────────────
print("Recalibrating LN-2 for compressed blocks...")
comp_model_recal = copy.deepcopy(comp_model)

with torch.no_grad():
    for b in BLOCKS:
        block = comp_model_recal.transformer.h[b]
        gamma = block.ln_2.weight.data.float()
        beta = block.ln_2.bias.data.float()

        float_mean, float_std = float_ln2[b]
        comp_mean, comp_std = comp_ln2[b]

        float_mean = float_mean.to(DEVICE)
        float_std = float_std.to(DEVICE)
        comp_mean = comp_mean.to(DEVICE)
        comp_std = comp_std.to(DEVICE)

        # Adjust so compressed distribution matches float after LN
        gamma_new = gamma * (float_std / comp_std)
        beta_new = beta + gamma * (float_mean - comp_mean) / comp_std

        block.ln_2.weight.data = gamma_new
        block.ln_2.bias.data = beta_new

print("Computing PPL (LN-2 recalibrated for compressed blocks)...")
comp_ppl_recal = compute_ppl(comp_model_recal)
print(f"Compressed PPL (LN-2 recal): {comp_ppl_recal:.3f}")

# ── Recalibrate all 12 blocks' LN-2 (downstream shift) ───────────────────────
print("Recalibrating LN-2 for all 12 blocks (downstream shift correction)...")
comp_model_all_recal = copy.deepcopy(comp_model)

with torch.no_grad():
    for b in range(12):
        block = comp_model_all_recal.transformer.h[b]
        gamma = block.ln_2.weight.data.float()
        beta = block.ln_2.bias.data.float()

        float_mean, float_std = float_ln2[b]
        comp_mean, comp_std = comp_ln2[b]

        float_mean = float_mean.to(DEVICE)
        float_std = float_std.to(DEVICE)
        comp_mean = comp_mean.to(DEVICE)
        comp_std = comp_std.to(DEVICE)

        gamma_new = gamma * (float_std / comp_std)
        beta_new = beta + gamma * (float_mean - comp_mean) / comp_std

        block.ln_2.weight.data = gamma_new
        block.ln_2.bias.data = beta_new

print("Computing PPL (all-12 LN-2 recalibrated)...")
comp_ppl_all_recal = compute_ppl(comp_model_all_recal)
print(f"Compressed PPL (all-12 LN-2 recal): {comp_ppl_all_recal:.3f}")

# ── Verdict ───────────────────────────────────────────────────────────────────
best_recal_ppl = min(comp_ppl_recal, comp_ppl_all_recal)
improvement = comp_ppl_before - best_recal_ppl

if improvement > 9:
    verdict = "CONFIRMED"
elif improvement < 2:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nFloat PPL:          {float_ppl:.3f}")
print(f"Compressed (before): {comp_ppl_before:.3f}")
print(f"LN-2 recal (compressed blocks): {comp_ppl_recal:.3f}")
print(f"LN-2 recal (all 12 blocks):     {comp_ppl_all_recal:.3f}")
print(f"Best improvement:    {improvement:.3f}")
print(f"Verdict: {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "cycle": 196,
    "item": 207,
    "verdict": verdict,
    "float_ppl": float_ppl,
    "compressed_ppl_before_recal": comp_ppl_before,
    "compressed_ppl_ln2_recal_blocks_only": comp_ppl_recal,
    "compressed_ppl_ln2_recal_all12": comp_ppl_all_recal,
    "best_improvement": improvement,
    "n_recal_params_blocks_only": 2 * 768 * len(BLOCKS),
    "n_recal_params_all12": 2 * 768 * 12,
    "config": {
        "BLOCKS": BLOCKS,
        "K": K,
        "N_SEQS_CALIB": N_SEQS_CALIB,
        "SEQ_LEN": SEQ_LEN,
        "SEED": SEED,
        "DEVICE": DEVICE,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
