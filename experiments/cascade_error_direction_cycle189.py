"""Cycle 189 / Item 199: Cascade error direction analysis — GELU vs ReLU+bias residual stream.

Build a full-swap ReLU+bias GPT-2 model (all 12 blocks), collect residual stream
from both float and swapped models, and analyze per-layer error directionality.

Error at layer l: e_l = r_float[l] - r_swapped[l]   shape: (N_tokens, 768)
Metrics per layer:
  cos(e_l, e_{l-1}) — directional alignment with previous layer's error
  ||e_l||_F         — error magnitude
  rank_90           — effective rank at 90% SVD energy

Verdict:
  CONFIRMED    if mean cos(e_l, e_{l-1}) > 0.3 in >=8/12 layers
  REFUTED      if mean cos(e_l, e_{l-1}) < 0.1 in all layers (diffuse)
  INCONCLUSIVE if 0.1-0.3 range
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
N_SEQS_CALIB = 200
N_SEQS_ANALYSIS = 100
BATCH_SIZE = 4
SEQ_LEN = 128
SEED_CALIB = 1337
SEED_ANALYSIS = 42
N_BLOCKS = 12
N_CHANNELS = 3072

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}", flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tokens(tokenizer, n_seqs: int, seed: int) -> torch.Tensor:
    text = OWT_PATH.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    tokens_t = torch.tensor(tokens, dtype=torch.long)
    rng = torch.Generator()
    rng.manual_seed(seed)
    max_start = len(tokens_t) - SEQ_LEN - 1
    indices = torch.randperm(max_start, generator=rng)[:n_seqs]
    seqs = torch.stack([tokens_t[i:i + SEQ_LEN + 1] for i in sorted(indices.tolist())])
    return seqs  # (n_seqs, SEQ_LEN+1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_ppl(model, seqs: torch.Tensor) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, len(seqs), BATCH_SIZE):
        batch = seqs[i:i + BATCH_SIZE].to(DEVICE)
        inp, tgt = batch[:, :-1], batch[:, 1:]
        out = model(inp)
        logits = out.logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_tokens += tgt.numel()
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# Calibration: compute per-block expected (GELU(z) - ReLU(z)) delta
# projected to output space (768-dim bias to add to c_proj.bias)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_bias_vectors(model, seqs: torch.Tensor) -> dict[int, torch.Tensor]:
    """Returns bias_to_add[b]: shape (768,) = E[GELU(z)-ReLU(z)] @ c_proj.weight"""
    sum_delta = {b: torch.zeros(N_CHANNELS, dtype=torch.float64) for b in range(N_BLOCKS)}
    count = {b: 0 for b in range(N_BLOCKS)}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            z = inp[0].reshape(-1, N_CHANNELS).float()  # (B*S, 3072)
            gelu_out = out.reshape(-1, N_CHANNELS).float()
            relu_out = F.relu(z)
            delta = (gelu_out - relu_out)               # (B*S, 3072)
            sum_delta[idx] += delta.sum(dim=0).cpu().double()
            count[idx] += delta.shape[0]
        return hook_fn

    for b in range(N_BLOCKS):
        h = model.transformer.h[b].mlp.act.register_forward_hook(make_hook(b))
        hooks.append(h)

    model.eval()
    for i in range(0, len(seqs), BATCH_SIZE):
        batch = seqs[i:i + BATCH_SIZE, :-1].to(DEVICE)
        model(batch)

    for h in hooks:
        h.remove()

    bias_vectors = {}
    for b in range(N_BLOCKS):
        mean_delta = (sum_delta[b] / count[b]).float()  # (3072,)
        # Project to output space: mean_delta @ c_proj.weight
        # c_proj: Conv1D mapping (B,S,3072) -> (B,S,768), weight shape (3072, 768)
        w = model.transformer.h[b].mlp.c_proj.weight.data.float().cpu()  # (3072, 768)
        bias_to_add = mean_delta @ w  # (768,)
        bias_vectors[b] = bias_to_add.cpu()
        print(f"  Block {b}: ||bias||={bias_to_add.norm().item():.4f}  "
              f"||mean_delta||={mean_delta.norm().item():.4f}", flush=True)

    return bias_vectors


# ---------------------------------------------------------------------------
# Build full ReLU+bias model (all 12 blocks swapped)
# ---------------------------------------------------------------------------
def build_relu_bias_model(float_model, bias_vectors: dict[int, torch.Tensor]) -> nn.Module:
    swapped = copy.deepcopy(float_model)
    for b in range(N_BLOCKS):
        swapped.transformer.h[b].mlp.act = nn.ReLU()
        swapped.transformer.h[b].mlp.c_proj.bias.data += bias_vectors[b].to(DEVICE)
    swapped.eval()
    return swapped


# ---------------------------------------------------------------------------
# Residual collection
# ---------------------------------------------------------------------------
class ResidualCollector:
    def __init__(self, model):
        self.residuals: dict[int, list[torch.Tensor]] = {l: [] for l in range(N_BLOCKS)}
        self._hooks = []
        for l in range(N_BLOCKS):
            def make_hook(layer_idx):
                def hook_fn(module, inp, out):
                    # GPT-2 block returns tuple (hidden_state, ...) or tensor
                    h = out[0] if isinstance(out, tuple) else out
                    self.residuals[layer_idx].append(h.detach().cpu().float())
                return hook_fn
            self._hooks.append(model.transformer.h[l].register_forward_hook(make_hook(l)))

    def remove(self):
        for h in self._hooks:
            h.remove()

    def get(self) -> dict[int, torch.Tensor]:
        # (N_seqs, seq_len, 768) -> (N_tokens, 768)
        return {l: torch.cat([r.reshape(-1, 768) for r in self.residuals[l]], dim=0)
                for l in self.residuals}


@torch.no_grad()
def collect_residuals(model, seqs: torch.Tensor) -> dict[int, torch.Tensor]:
    model.eval()
    collector = ResidualCollector(model)
    for i in range(0, len(seqs), BATCH_SIZE):
        batch = seqs[i:i + BATCH_SIZE, :-1].to(DEVICE)
        model(batch)
    collector.remove()
    return collector.get()  # {layer_id: (N_tokens, 768)}


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------
def mean_cos_similarity(e_l: torch.Tensor, e_prev: torch.Tensor) -> float:
    """Mean per-token cosine similarity between e_l and e_prev."""
    norm_l = e_l.norm(dim=1, keepdim=True).clamp(min=1e-8)
    norm_p = e_prev.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos = ((e_l / norm_l) * (e_prev / norm_p)).sum(dim=1)
    return float(cos.mean().item())


def frobenius_norm(e: torch.Tensor) -> float:
    return float(e.norm().item())


def effective_rank(e: torch.Tensor, threshold: float = 0.90) -> int:
    """SVD-based effective rank at energy threshold."""
    try:
        S = torch.linalg.svdvals(e.float())
        energy = (S ** 2).cumsum(0) / (S ** 2).sum().clamp(min=1e-12)
        rank = int((energy < threshold).sum().item()) + 1
        return min(rank, e.shape[1])
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
float_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
float_model.eval()

print("Loading data...", flush=True)
calib_seqs = load_tokens(tokenizer, N_SEQS_CALIB, SEED_CALIB)
analysis_seqs = load_tokens(tokenizer, N_SEQS_ANALYSIS, SEED_ANALYSIS)
print(f"  Calib: {len(calib_seqs)} seqs | Analysis: {len(analysis_seqs)} seqs", flush=True)

# Baseline PPL
print("Baseline PPL...", flush=True)
baseline_ppl = eval_ppl(float_model, analysis_seqs)
print(f"  Baseline PPL: {baseline_ppl:.3f}", flush=True)

if not (15.0 <= baseline_ppl <= 65.0):
    print(f"ABORT: baseline {baseline_ppl:.2f} outside [15,65]", flush=True)
    sys.exit(1)

# Calibration
print("\nComputing ReLU→GELU bias vectors...", flush=True)
bias_vectors = compute_bias_vectors(float_model, calib_seqs)

# Build ReLU+bias model (all 12 blocks)
print("\nBuilding full-swap ReLU+bias model...", flush=True)
swapped_model = build_relu_bias_model(float_model, bias_vectors)
swapped_ppl = eval_ppl(swapped_model, analysis_seqs)
print(f"  Full-swap PPL: {swapped_ppl:.3f}  (Δ={swapped_ppl-baseline_ppl:+.3f})", flush=True)

# Collect residuals
print("\nCollecting residual streams...", flush=True)
res_float = collect_residuals(float_model, analysis_seqs)
res_swapped = collect_residuals(swapped_model, analysis_seqs)
print(f"  Float residuals: {len(res_float)} layers × {res_float[0].shape[0]} tokens × 768", flush=True)

# Compute per-layer error metrics
print("\nComputing error metrics...", flush=True)
layer_metrics = []
errors = {}
for l in range(N_BLOCKS):
    e_l = res_float[l] - res_swapped[l]  # (N_tokens, 768)
    errors[l] = e_l
    cos_prev = mean_cos_similarity(e_l, errors[l - 1]) if l > 0 else None
    frob = frobenius_norm(e_l)
    rank90 = effective_rank(e_l)
    layer_metrics.append({
        "layer": l,
        "error_frob": frob,
        "cos_prev_layer": cos_prev,
        "rank_90": rank90,
    })
    cos_str = f"cos(e_{l},e_{l-1})={cos_prev:.4f}" if cos_prev is not None else "cos=N/A"
    print(f"  Layer {l:2d}: ||e||_F={frob:.2f}  {cos_str}  rank_90={rank90}", flush=True)

# Aggregate verdict
cos_values = [m["cos_prev_layer"] for m in layer_metrics if m["cos_prev_layer"] is not None]
mean_cos = sum(cos_values) / len(cos_values)
n_high = sum(1 for c in cos_values if c > 0.3)
n_low = sum(1 for c in cos_values if c < 0.1)

if n_high >= 8:
    verdict = "CONFIRMED"
    verdict_note = f"Mean cos={mean_cos:.4f}; {n_high}/11 layers have cos>0.3 (directionally aligned)"
elif n_low == 11:
    verdict = "REFUTED"
    verdict_note = f"Mean cos={mean_cos:.4f}; all 11 layers have cos<0.1 (diffuse)"
else:
    verdict = "INCONCLUSIVE"
    verdict_note = f"Mean cos={mean_cos:.4f}; {n_high}/11 layers with cos>0.3, {n_low}/11 with cos<0.1"

print(f"\n{'='*60}", flush=True)
print(f"Mean cos(e_l, e_{{l-1}}): {mean_cos:.4f}", flush=True)
print(f"Layers with cos>0.3: {n_high}/11", flush=True)
print(f"VERDICT: {verdict}", flush=True)
print(f"{verdict_note}", flush=True)

# Save
results = {
    "cycle": 189, "item": 199,
    "verdict": verdict, "verdict_note": verdict_note,
    "baseline_ppl": baseline_ppl,
    "swapped_ppl": swapped_ppl,
    "mean_cos": mean_cos,
    "n_high_cos": n_high,
    "n_low_cos": n_low,
    "layer_metrics": layer_metrics,
    "config": {
        "n_seqs_calib": N_SEQS_CALIB, "n_seqs_analysis": N_SEQS_ANALYSIS,
        "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN,
        "seed_calib": SEED_CALIB, "seed_analysis": SEED_ANALYSIS,
        "device": str(DEVICE),
    },
}

out_path = RESULTS_DIR / "cycle_189_cascade_error_direction.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}", flush=True)
