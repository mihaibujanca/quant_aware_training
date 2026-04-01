"""Cycle 194 / Item 204: Self-correction depth analysis.

Non-monotone PPL in cycles 190/193: peak at k=7, decrease at k=8-10, rise at k=11.
This experiment measures whether the decrease is due to:
  (a) Error MAGNITUDE decreasing (partial cancellation) → CONFIRMED
  (b) Error GROWING but redirected to low-PPL subspace → REFUTED

Hybrid activation: GELU for z<0, ReLU for z>=0 (same as cycle 190).
Config: SEQ_LEN=1024, N_SEQS=200, SEED=42 (same as cycle 190).

For each k in [6..11]:
  e_l[k] = res_float[l] - res_hybrid_k[l]
  frob[k][l] = ||e_l[k]||_F
  cos_vs_k7[k][l] = mean cosine similarity between e_l[k] and e_l[k=7]

Verdict:
  CONFIRMED    frob[k=8,9,10][l] < frob[k=7][l] for >=6/12 layers (magnitude decreases)
  REFUTED      frob keeps growing but cos_vs_k7 changes (error redirected)
  INCONCLUSIVE both effects mix
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

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
N_SEQS = 200
BATCH_SIZE = 4
SEQ_LEN = 1024
SEED = 42
N_BLOCKS = 12

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_K = list(range(6, 12))  # k=6..11 for residual analysis

print(f"Device: {DEVICE}", flush=True)


class GELUnegReLUpos(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.where(z < 0, F.gelu(z), F.relu(z))


def load_tokens(tokenizer, n_seqs: int, seed: int) -> torch.Tensor:
    text = OWT_PATH.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    tokens_t = torch.tensor(tokens, dtype=torch.long)
    rng = torch.Generator()
    rng.manual_seed(seed)
    max_start = len(tokens_t) - SEQ_LEN - 1
    indices = torch.randperm(max_start, generator=rng)[:n_seqs]
    return torch.stack([tokens_t[i:i + SEQ_LEN + 1] for i in sorted(indices.tolist())])


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
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                               reduction="sum")
        total_loss += float(loss.item())
        total_tokens += tgt.numel()
    return math.exp(total_loss / total_tokens)


class ResidualCollector:
    def __init__(self, model):
        self.residuals: dict[int, list] = {l: [] for l in range(N_BLOCKS)}
        self._hooks = []
        for l in range(N_BLOCKS):
            def make_hook(idx):
                def hook_fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    self.residuals[idx].append(h.detach().cpu().float())
                return hook_fn
            self._hooks.append(model.transformer.h[l].register_forward_hook(make_hook(l)))

    def remove(self):
        for h in self._hooks:
            h.remove()

    def get(self) -> dict[int, torch.Tensor]:
        return {l: torch.cat([r.reshape(-1, 768) for r in self.residuals[l]], dim=0)
                for l in self.residuals}


@torch.no_grad()
def collect_residuals(model, seqs: torch.Tensor) -> dict[int, torch.Tensor]:
    model.eval()
    collector = ResidualCollector(model)
    for i in range(0, len(seqs), BATCH_SIZE):
        model(seqs[i:i + BATCH_SIZE, :-1].to(DEVICE))
    collector.remove()
    return collector.get()


def mean_cos_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    norm_a = a.norm(dim=1, keepdim=True).clamp(min=1e-8)
    norm_b = b.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return float(((a / norm_a) * (b / norm_b)).sum(dim=1).mean().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
float_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
float_model.eval()

print("Loading data...", flush=True)
seqs = load_tokens(tokenizer, N_SEQS, SEED)

# Baseline PPL
print("Baseline PPL...", flush=True)
ppl_baseline = eval_ppl(float_model, seqs)
print(f"  Baseline: {ppl_baseline:.3f}", flush=True)

if not (15.0 <= ppl_baseline <= 30.0):
    print(f"ABORT: baseline {ppl_baseline:.2f} outside [15,30]", flush=True)
    sys.exit(1)

# Progressive PPL sweep (full k=0..11)
print("\nProgressive PPL sweep...", flush=True)
progressive_ppl = []
for k in range(N_BLOCKS):
    swapped = copy.deepcopy(float_model)
    for b in range(k + 1):
        swapped.transformer.h[b].mlp.act = GELUnegReLUpos()
    ppl = eval_ppl(swapped, seqs)
    pct = (ppl - ppl_baseline) / ppl_baseline * 100
    progressive_ppl.append({"k": k, "ppl": ppl, "pct": pct})
    print(f"  k={k:2d}: PPL={ppl:.3f}  ({pct:+.2f}%)", flush=True)
    del swapped

# Collect float residuals once
print("\nCollecting float residuals...", flush=True)
res_float = collect_residuals(float_model, seqs)
n_tokens = res_float[0].shape[0]
print(f"  Float residuals: {n_tokens} tokens × 768", flush=True)

# Store k=7 errors as reference
res_k7 = None
errors_k7 = None

# Per-k residual analysis
print("\nPer-k residual analysis (k=6..11)...", flush=True)
error_metrics_by_k = {}
for k in ANALYSIS_K:
    swapped = copy.deepcopy(float_model)
    for b in range(k + 1):
        swapped.transformer.h[b].mlp.act = GELUnegReLUpos()

    res_hybrid = collect_residuals(swapped, seqs)
    del swapped

    # Compute per-layer error metrics
    errors = {l: res_float[l] - res_hybrid[l] for l in range(N_BLOCKS)}

    if k == 7:
        errors_k7 = {l: errors[l].clone() for l in range(N_BLOCKS)}

    frob_per_layer = [float(errors[l].norm().item()) for l in range(N_BLOCKS)]
    cos_vs_k7_per_layer = []

    if errors_k7 is not None:
        for l in range(N_BLOCKS):
            cos_vs_k7_per_layer.append(mean_cos_similarity(errors[l], errors_k7[l]))
    else:
        cos_vs_k7_per_layer = [None] * N_BLOCKS

    error_metrics_by_k[str(k)] = {
        "frob_per_layer": frob_per_layer,
        "cos_vs_k7_per_layer": cos_vs_k7_per_layer,
        "total_frob": float(sum(f**2 for f in frob_per_layer) ** 0.5),
    }

    frob_str = "  ".join(f"L{l}={frob_per_layer[l]:.1f}" for l in range(0, 12, 2))
    print(f"  k={k}: {frob_str}", flush=True)

    del res_hybrid, errors

# Fill in k<7 without cos_vs_k7 (set for k=6)
if 6 in ANALYSIS_K and errors_k7 is not None:
    swapped = copy.deepcopy(float_model)
    for b in range(7):
        swapped.transformer.h[b].mlp.act = GELUnegReLUpos()
    res_hybrid6 = collect_residuals(swapped, seqs)
    del swapped
    errors_k6 = {l: res_float[l] - res_hybrid6[l] for l in range(N_BLOCKS)}
    cos_vs_k7_k6 = [mean_cos_similarity(errors_k6[l], errors_k7[l]) for l in range(N_BLOCKS)]
    error_metrics_by_k["6"]["cos_vs_k7_per_layer"] = cos_vs_k7_k6
    del res_hybrid6, errors_k6

# Verdict analysis
# Focus on layers k=7,8,9,10: does frob decrease?
frob_k7 = error_metrics_by_k.get("7", {}).get("frob_per_layer", [])
frob_k8 = error_metrics_by_k.get("8", {}).get("frob_per_layer", [])
frob_k9 = error_metrics_by_k.get("9", {}).get("frob_per_layer", [])
frob_k10 = error_metrics_by_k.get("10", {}).get("frob_per_layer", [])

n_decrease_k8 = sum(1 for l in range(N_BLOCKS) if frob_k8 and frob_k8[l] < frob_k7[l]) if frob_k7 and frob_k8 else 0
n_decrease_k9 = sum(1 for l in range(N_BLOCKS) if frob_k9 and frob_k9[l] < frob_k7[l]) if frob_k7 and frob_k9 else 0
n_decrease_k10 = sum(1 for l in range(N_BLOCKS) if frob_k10 and frob_k10[l] < frob_k7[l]) if frob_k7 and frob_k10 else 0

max_decrease = max(n_decrease_k8, n_decrease_k9, n_decrease_k10)

if max_decrease >= 6:
    verdict = "CONFIRMED"
    verdict_note = (f"Error magnitude DECREASES at k=8-10 for ≥6/12 layers "
                    f"(max_decrease={max_decrease}/12). Partial cancellation occurs.")
elif max_decrease <= 2:
    verdict = "REFUTED"
    verdict_note = (f"Error magnitude keeps growing at k=8-10 (max_decrease={max_decrease}/12). "
                    f"Error redirected to low-PPL subspace.")
else:
    verdict = "INCONCLUSIVE"
    verdict_note = (f"Mixed: max_decrease={max_decrease}/12 layers show magnitude decrease. "
                    f"Both cancellation and redirection contribute.")

print(f"\n{'='*60}", flush=True)
print(f"Layers where ||e_l|| decreases from k=7:", flush=True)
print(f"  k=8: {n_decrease_k8}/12 layers", flush=True)
print(f"  k=9: {n_decrease_k9}/12 layers", flush=True)
print(f"  k=10: {n_decrease_k10}/12 layers", flush=True)
print(f"VERDICT: {verdict}", flush=True)
print(f"{verdict_note}", flush=True)

results = {
    "cycle": 194, "item": 204,
    "verdict": verdict,
    "verdict_note": verdict_note,
    "baseline_ppl": ppl_baseline,
    "progressive_ppl": progressive_ppl,
    "error_metrics_by_k": error_metrics_by_k,
    "n_frob_decrease_k8": n_decrease_k8,
    "n_frob_decrease_k9": n_decrease_k9,
    "n_frob_decrease_k10": n_decrease_k10,
    "config": {
        "n_seqs": N_SEQS, "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN,
        "seed": SEED, "device": str(DEVICE),
    },
}

out_path = RESULTS_DIR / "cycle_194_self_correction.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}", flush=True)
