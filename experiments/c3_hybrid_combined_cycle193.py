"""Cycle 193 / Item 203: C3+Hybrid combined activation — superadditivity isolation.

Combined activation:
  z < 0: mu_neg[b][c]  (per-channel mean of GELU(z<0), same as C3)
  z >= 0: ReLU(z) = z  (shape change, same as Hybrid)

Apply to all 12 blocks. Compare to:
  Baseline         ~22.700
  C3 alone         27.314 (+20.33%)  [cycle 188]
  Hybrid alone     28.980 (+27.67%)  [cycle 190]
  Simple sum pred  ~33.6  (+48%)
  ReLU+bias        ~42.6  (+88%)     [cycle 181]

If C3+Hybrid >> 33.6 → superadditivity is general (not specific to ReLU zero-point).

Verdict (baseline=22.700):
  CONFIRMED    C3+Hybrid PPL > 45 (+98%): superadditivity is general
  REFUTED      C3+Hybrid PPL < 34 (+50%): no superadditivity, close to simple sum
  INCONCLUSIVE 34-45 range
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
# Config — match cycle 188
# ---------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
N_SEQS = 200
BATCH_SIZE = 4
SEQ_LEN = 1024
SEED = 42
N_BLOCKS = 12
N_CHANNELS = 3072

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_C3 = 27.314      # cycle 188
REFERENCE_HYBRID = 28.980  # cycle 190
REFERENCE_RELU_BIAS = 42.6 # cycle 181 approximate
REFERENCE_BASELINE = 22.700

print(f"Device: {DEVICE}", flush=True)


# ---------------------------------------------------------------------------
# Combined activation
# ---------------------------------------------------------------------------
class C3HybridActivation(nn.Module):
    """z<0: constant mu_neg (C3); z>=0: ReLU(z) (Hybrid)."""
    def __init__(self, mu_neg: torch.Tensor):
        super().__init__()
        self.register_buffer('mu_neg', mu_neg)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mu = self.mu_neg.expand_as(z)
        return torch.where(z < 0, mu, F.relu(z))


# ---------------------------------------------------------------------------
# Data / PPL (same as cycle 188)
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
# Calibrate mu_neg (same as cycle 188 C3)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_mu_neg(model, seqs: torch.Tensor) -> dict[int, torch.Tensor]:
    sum_neg = {b: torch.zeros(N_CHANNELS, dtype=torch.float64) for b in range(N_BLOCKS)}
    count_neg = {b: 0 for b in range(N_BLOCKS)}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            z = inp[0].reshape(-1, N_CHANNELS).float()
            gelu_out = out.reshape(-1, N_CHANNELS).float()
            neg_mask = (z < 0)
            for c in range(0, N_CHANNELS, 256):  # chunked for memory
                end = min(c + 256, N_CHANNELS)
                m = neg_mask[:, c:end]
                g = gelu_out[:, c:end]
                sum_neg[idx][c:end] += g[m].sum(dim=0).double() if m.any() else torch.zeros(end - c, dtype=torch.float64)
                # count per channel
                count_neg[idx][c:end] += m.sum(dim=0).cpu().long().double()
        return hook_fn

    # Simpler: collect all at once
    sum_neg2 = {b: torch.zeros(N_CHANNELS, dtype=torch.float64) for b in range(N_BLOCKS)}
    count_neg2 = {b: torch.zeros(N_CHANNELS, dtype=torch.float64) for b in range(N_BLOCKS)}
    hooks2 = []

    def make_hook2(idx):
        def hook_fn2(module, inp, out):
            z = inp[0].reshape(-1, N_CHANNELS).float().cpu()
            gelu_out = out.reshape(-1, N_CHANNELS).float().cpu()
            neg_mask = (z < 0)
            sum_neg2[idx] += (gelu_out * neg_mask.float()).sum(dim=0).double()
            count_neg2[idx] += neg_mask.float().sum(dim=0).double()
        return hook_fn2

    for b in range(N_BLOCKS):
        h = model.transformer.h[b].mlp.act.register_forward_hook(make_hook2(b))
        hooks2.append(h)

    model.eval()
    for i in range(0, len(seqs), BATCH_SIZE):
        model(seqs[i:i + BATCH_SIZE, :-1].to(DEVICE))

    for h in hooks2:
        h.remove()

    mu_neg = {}
    for b in range(N_BLOCKS):
        cnt = count_neg2[b].clamp(min=1.0)
        mu_neg[b] = (sum_neg2[b] / cnt).float()
        print(f"  Block {b}: ||mu_neg||={mu_neg[b].norm():.4f}  "
              f"neg_fraction={float((count_neg2[b] / (len(seqs)*(SEQ_LEN)*1.0)).mean()):.3f}",
              flush=True)
    return mu_neg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
float_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
float_model.eval()

print("Loading data...", flush=True)
seqs = load_tokens(tokenizer, N_SEQS, SEED)
print(f"  {len(seqs)} seqs × {SEQ_LEN} tokens", flush=True)

# Baseline
print("Baseline PPL...", flush=True)
ppl_baseline = eval_ppl(float_model, seqs)
print(f"  Baseline PPL: {ppl_baseline:.3f}", flush=True)

if not (15.0 <= ppl_baseline <= 30.0):
    print(f"ABORT: baseline {ppl_baseline:.2f} outside [15,30]", flush=True)
    sys.exit(1)

# Calibrate mu_neg
print("\nCalibrating mu_neg...", flush=True)
mu_neg = compute_mu_neg(float_model, seqs)

# Progressive sweep
print("\nProgressive C3+Hybrid sweep...", flush=True)
progressive = []
for k in range(N_BLOCKS):
    swapped = copy.deepcopy(float_model)
    for b in range(k + 1):
        swapped.transformer.h[b].mlp.act = C3HybridActivation(mu_neg[b].to(DEVICE))
    swapped.eval()
    ppl = eval_ppl(swapped, seqs)
    pct = (ppl - ppl_baseline) / ppl_baseline * 100
    progressive.append({"k": k, "blocks_swapped": k + 1, "ppl": ppl, "pct_increase": pct})
    print(f"  k={k:2d} ({k+1:2d} blocks): PPL={ppl:.3f}  ({pct:+.2f}%)", flush=True)
    del swapped

all12_ppl = progressive[-1]["ppl"]
all12_pct = progressive[-1]["pct_increase"]

# Verdict
if all12_ppl > 45.0:
    verdict = "CONFIRMED"
    verdict_note = (f"C3+Hybrid all-12 PPL={all12_ppl:.3f} ({all12_pct:+.2f}%) > 45. "
                    f"Superadditivity is general — not specific to ReLU zero-point.")
elif all12_ppl < 34.0:
    verdict = "REFUTED"
    verdict_note = (f"C3+Hybrid all-12 PPL={all12_ppl:.3f} ({all12_pct:+.2f}%) < 34. "
                    f"No superadditivity. Both-regime modification cost ≈ simple sum.")
else:
    verdict = "INCONCLUSIVE"
    verdict_note = (f"C3+Hybrid all-12 PPL={all12_ppl:.3f} ({all12_pct:+.2f}%) in 34-45 range. "
                    f"Partial superadditivity.")

print(f"\n{'='*60}", flush=True)
print(f"Baseline:                {ppl_baseline:.3f}", flush=True)
print(f"C3+Hybrid all-12:        {all12_ppl:.3f}  ({all12_pct:+.2f}%)", flush=True)
print(f"C3 alone (cycle 188):    {REFERENCE_C3:.3f} (+20.33%)", flush=True)
print(f"Hybrid alone (cycle 190): {REFERENCE_HYBRID:.3f} (+27.67%)", flush=True)
print(f"Simple sum prediction:   ~{ppl_baseline * 1.48:.3f} (+48%)", flush=True)
print(f"ReLU+bias (cycle 181):   ~{REFERENCE_RELU_BIAS:.1f} (+88%)", flush=True)
print(f"VERDICT: {verdict}", flush=True)
print(f"{verdict_note}", flush=True)

results = {
    "cycle": 193, "item": 203,
    "verdict": verdict,
    "verdict_note": verdict_note,
    "baseline_ppl": ppl_baseline,
    "c3_hybrid_all12_ppl": all12_ppl,
    "c3_hybrid_all12_pct": all12_pct,
    "simple_sum_prediction_ppl": ppl_baseline * 1.48,
    "reference_c3_ppl": REFERENCE_C3,
    "reference_hybrid_ppl": REFERENCE_HYBRID,
    "reference_relu_bias_ppl": REFERENCE_RELU_BIAS,
    "progressive": progressive,
    "config": {
        "n_seqs": N_SEQS, "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN,
        "seed": SEED, "device": str(DEVICE),
    },
}

out_path = RESULTS_DIR / "cycle_193_c3_hybrid_combined.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}", flush=True)
