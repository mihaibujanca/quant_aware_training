"""Cycle 191 / Item 201: Mean residual error correction.

Build full-swap ReLU+bias GPT-2 model (all 12 blocks).
Calibrate per-layer mean correction: mean_correction[l] = E[r_float[l] - r_swapped[l]].
Apply as a constant bias to each block's residual stream output (9216 params total).
Measure PPL improvement over uncorrected full-swap.

Verdict:
  CONFIRMED    corrected PPL < 38 (saves >6 PPL from uncorrected ~44.8, >15% improvement)
  REFUTED      corrected PPL within 1 PPL of uncorrected
  INCONCLUSIVE 1-6 PPL improvement
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
# Config — match cycle 189
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
# Data
# ---------------------------------------------------------------------------
def load_tokens(tokenizer, n_seqs: int, seed: int) -> torch.Tensor:
    text = OWT_PATH.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    tokens_t = torch.tensor(tokens, dtype=torch.long)
    rng = torch.Generator()
    rng.manual_seed(seed)
    max_start = len(tokens_t) - SEQ_LEN - 1
    indices = torch.randperm(max_start, generator=rng)[:n_seqs]
    return torch.stack([tokens_t[i:i + SEQ_LEN + 1] for i in sorted(indices.tolist())])


# ---------------------------------------------------------------------------
# PPL
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
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                               reduction="sum")
        total_loss += float(loss.item())
        total_tokens += tgt.numel()
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# Bias vectors (same as cycle 189)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_bias_vectors(model, seqs: torch.Tensor) -> dict[int, torch.Tensor]:
    sum_delta = {b: torch.zeros(N_CHANNELS, dtype=torch.float64) for b in range(N_BLOCKS)}
    count = {b: 0 for b in range(N_BLOCKS)}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            z = inp[0].reshape(-1, N_CHANNELS).float()
            gelu_out = out.reshape(-1, N_CHANNELS).float()
            relu_out = F.relu(z)
            delta = gelu_out - relu_out
            sum_delta[idx] += delta.sum(dim=0).cpu().double()
            count[idx] += delta.shape[0]
        return hook_fn

    for b in range(N_BLOCKS):
        h = model.transformer.h[b].mlp.act.register_forward_hook(make_hook(b))
        hooks.append(h)

    model.eval()
    for i in range(0, len(seqs), BATCH_SIZE):
        model(seqs[i:i + BATCH_SIZE, :-1].to(DEVICE))

    for h in hooks:
        h.remove()

    bias_vectors = {}
    for b in range(N_BLOCKS):
        mean_delta = (sum_delta[b] / count[b]).float()
        w = model.transformer.h[b].mlp.c_proj.weight.data.float().cpu()  # (3072, 768)
        bias_vectors[b] = (mean_delta @ w).cpu()
    return bias_vectors


# ---------------------------------------------------------------------------
# Build swapped model
# ---------------------------------------------------------------------------
def build_relu_bias_model(float_model, bias_vectors) -> nn.Module:
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


# ---------------------------------------------------------------------------
# Apply correction hooks
# ---------------------------------------------------------------------------
def apply_correction_hooks(model, mean_corrections: dict[int, torch.Tensor]) -> list:
    hooks = []
    for l in range(N_BLOCKS):
        corr = mean_corrections[l].to(DEVICE)

        def make_hook(c):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                corrected = h + c
                return (corrected,) + out[1:] if isinstance(out, tuple) else corrected
            return hook_fn

        hooks.append(model.transformer.h[l].register_forward_hook(make_hook(corr)))
    return hooks


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
print(f"  Calib: {len(calib_seqs)} | Analysis: {len(analysis_seqs)}", flush=True)

# Baseline
print("Baseline PPL...", flush=True)
ppl_baseline = eval_ppl(float_model, analysis_seqs)
print(f"  Baseline PPL: {ppl_baseline:.3f}", flush=True)

if not (15.0 <= ppl_baseline <= 65.0):
    print(f"ABORT: baseline {ppl_baseline:.2f} outside [15,65]", flush=True)
    sys.exit(1)

# Build swapped model
print("\nComputing bias vectors...", flush=True)
bias_vectors = compute_bias_vectors(float_model, calib_seqs)
print("\nBuilding full-swap ReLU+bias model...", flush=True)
swapped_model = build_relu_bias_model(float_model, bias_vectors)

# Uncorrected PPL
print("Uncorrected PPL...", flush=True)
ppl_uncorrected = eval_ppl(swapped_model, analysis_seqs)
print(f"  Uncorrected PPL: {ppl_uncorrected:.3f}  (Δ={ppl_uncorrected-ppl_baseline:+.3f})",
      flush=True)

# Calibrate mean corrections
print("\nCalibrating mean corrections...", flush=True)
res_float = collect_residuals(float_model, calib_seqs)
res_swapped = collect_residuals(swapped_model, calib_seqs)

mean_corrections = {}
correction_norms = []
fraction_of_frob = []
for l in range(N_BLOCKS):
    e_l = res_float[l] - res_swapped[l]  # (N_tokens, 768)
    mc = e_l.mean(dim=0)                 # (768,)
    mean_corrections[l] = mc
    norm = float(mc.norm().item())
    frob = float(e_l.norm().item())
    n_tokens = e_l.shape[0]
    frac = norm / (frob / (n_tokens ** 0.5)) if frob > 0 else 0.0
    correction_norms.append(norm)
    fraction_of_frob.append(frac)
    print(f"  Layer {l:2d}: ||mean_corr||={norm:.4f}  frac_frob={frac:.4f}", flush=True)

# Corrected PPL
print("\nApplying corrections...", flush=True)
correction_hooks = apply_correction_hooks(swapped_model, mean_corrections)
ppl_corrected = eval_ppl(swapped_model, analysis_seqs)
for h in correction_hooks:
    h.remove()
print(f"  Corrected PPL: {ppl_corrected:.3f}", flush=True)

# Verdict
delta = ppl_uncorrected - ppl_corrected
if delta > 6.0:
    verdict = "CONFIRMED"
    verdict_note = (f"Correction saves {delta:.3f} PPL ({ppl_uncorrected:.3f}→{ppl_corrected:.3f}). "
                    f"First-moment correction is useful.")
elif delta < 1.0:
    verdict = "REFUTED"
    verdict_note = (f"Correction saves only {delta:.3f} PPL ({ppl_uncorrected:.3f}→{ppl_corrected:.3f}). "
                    f"First moment is not useful.")
else:
    verdict = "INCONCLUSIVE"
    verdict_note = (f"Correction saves {delta:.3f} PPL ({ppl_uncorrected:.3f}→{ppl_corrected:.3f}). "
                    f"Partial benefit.")

print(f"\n{'='*60}", flush=True)
print(f"Baseline:    {ppl_baseline:.3f}", flush=True)
print(f"Uncorrected: {ppl_uncorrected:.3f}", flush=True)
print(f"Corrected:   {ppl_corrected:.3f}  (saves {delta:+.3f})", flush=True)
print(f"VERDICT: {verdict}", flush=True)
print(f"{verdict_note}", flush=True)

results = {
    "cycle": 191, "item": 201,
    "verdict": verdict,
    "verdict_note": verdict_note,
    "baseline_ppl": ppl_baseline,
    "uncorrected_ppl": ppl_uncorrected,
    "corrected_ppl": ppl_corrected,
    "ppl_improvement": delta,
    "correction_norms": correction_norms,
    "fraction_of_frobenius": fraction_of_frob,
    "config": {
        "n_seqs_calib": N_SEQS_CALIB, "n_seqs_analysis": N_SEQS_ANALYSIS,
        "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN,
        "seed_calib": SEED_CALIB, "seed_analysis": SEED_ANALYSIS,
        "device": str(DEVICE),
    },
}

out_path = RESULTS_DIR / "cycle_191_mean_error_correction.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}", flush=True)
