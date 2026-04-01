"""Cycle 190 / Item 200: GELU-neg + ReLU-pos hybrid activation.

Hybrid act: GELU(z) for z < 0 (preserves negative regime exactly),
            ReLU(z) for z >= 0 (changes positive-regime shape).

Apply to all 12 GPT-2 blocks simultaneously + progressive sweep.
No bias correction needed — negative regime is preserved exactly.

Compare:
  baseline     ~22.7 PPL
  this hybrid   ???
  C3 all-12    27.314 (cycle 188, +20.33%)
  ReLU+bias    ~42.6  (cycle 181, +88%)

Verdict:
  CONFIRMED    hybrid PPL > 35 (+54%) — positive shape IS the dominant cost
  REFUTED      hybrid PPL < 28 (+23%) — negative regime loss dominates
  INCONCLUSIVE 28-35 range
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
# Config — match cycle 188 exactly for fair comparison
# ---------------------------------------------------------------------------
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

print(f"Device: {DEVICE}", flush=True)


# ---------------------------------------------------------------------------
# Hybrid activation
# ---------------------------------------------------------------------------
class GELUnegReLUpos(nn.Module):
    """GELU for z<0, ReLU for z>=0. Preserves negative regime exactly."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.where(z < 0, F.gelu(z), F.relu(z))


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
# PPL evaluation
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
baseline_ppl = eval_ppl(float_model, seqs)
print(f"  Baseline PPL: {baseline_ppl:.3f}", flush=True)

if not (15.0 <= baseline_ppl <= 30.0):
    print(f"ABORT: baseline {baseline_ppl:.2f} outside [15,30] — check config", flush=True)
    sys.exit(1)

# Progressive sweep: k=0..11 blocks swapped
print("\nProgressive hybrid sweep...", flush=True)
progressive = []
for k in range(N_BLOCKS):
    swapped = copy.deepcopy(float_model)
    for b in range(k + 1):
        swapped.transformer.h[b].mlp.act = GELUnegReLUpos()
    swapped.eval()
    ppl = eval_ppl(swapped, seqs)
    pct = (ppl - baseline_ppl) / baseline_ppl * 100
    progressive.append({"k": k, "blocks_swapped": k + 1, "ppl": ppl, "pct_increase": pct})
    print(f"  k={k:2d} ({k+1:2d} blocks): PPL={ppl:.3f}  ({pct:+.2f}%)", flush=True)
    del swapped

hybrid_ppl = progressive[-1]["ppl"]
hybrid_pct = progressive[-1]["pct_increase"]

# Verdict
C3_ALL12_PPL = 27.314  # cycle 188
RELU_BIAS_ALL12_PPL = 42.6  # cycle 181 approximate

if hybrid_ppl > 35.0:
    verdict = "CONFIRMED"
    verdict_note = (f"Hybrid all-12 PPL={hybrid_ppl:.3f} ({hybrid_pct:+.2f}%) > 35. "
                    f"Positive-regime shape change IS the dominant composition error.")
elif hybrid_ppl < 28.0:
    verdict = "REFUTED"
    verdict_note = (f"Hybrid all-12 PPL={hybrid_ppl:.3f} ({hybrid_pct:+.2f}%) < 28. "
                    f"Negative-regime loss dominates.")
else:
    verdict = "INCONCLUSIVE"
    verdict_note = (f"Hybrid all-12 PPL={hybrid_ppl:.3f} ({hybrid_pct:+.2f}%) in 28-35 range.")

print(f"\n{'='*60}", flush=True)
print(f"Baseline:         {baseline_ppl:.3f}", flush=True)
print(f"Hybrid all-12:    {hybrid_ppl:.3f}  ({hybrid_pct:+.2f}%)", flush=True)
print(f"C3 all-12:        {C3_ALL12_PPL:.3f} (+20.33%)  [cycle 188]", flush=True)
print(f"ReLU+bias all-12: {RELU_BIAS_ALL12_PPL:.1f}  (+88%)    [cycle 181 approx]", flush=True)
print(f"VERDICT: {verdict}", flush=True)
print(f"{verdict_note}", flush=True)

results = {
    "cycle": 190, "item": 200,
    "verdict": verdict,
    "verdict_note": verdict_note,
    "baseline_ppl": baseline_ppl,
    "hybrid_all12_ppl": hybrid_ppl,
    "hybrid_pct_increase": hybrid_pct,
    "reference_c3_all12_ppl": C3_ALL12_PPL,
    "reference_relu_bias_all12_ppl": RELU_BIAS_ALL12_PPL,
    "progressive": progressive,
    "config": {
        "n_seqs": N_SEQS, "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN,
        "seed": SEED, "device": str(DEVICE),
    },
}

out_path = RESULTS_DIR / "cycle_190_gelu_neg_relu_pos.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}", flush=True)
