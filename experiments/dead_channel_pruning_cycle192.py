"""Cycle 192 / Item 202: Exact dead-channel pruning baseline.

For each GPT-2 FFN block: identify channels where max |GELU(z_c)| < epsilon
over all calibration tokens. Prune by zeroing their c_fc and c_proj weights.
Test multiple thresholds (0.001, 0.01, 0.1, 1.0).

This is the Serra/Berzins exact white-box pruning baseline (required by program R2).

Verdict:
  CONFIRMED    near-dead (eps=0.01) channels > 5% AND pruned PPL < 38 (+6% of 35.9 baseline)
  REFUTED      near-dead (eps=0.01) < 1% of channels
  INCONCLUSIVE otherwise
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
N_SEQS_CALIB = 500
BATCH_SIZE = 4
SEQ_LEN = 128
SEED = 1337
N_BLOCKS = 12
N_CHANNELS = 3072
EPSILONS = [0.001, 0.01, 0.1, 1.0]

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
# Collect per-channel max GELU activation
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_max_gelu(model, seqs: torch.Tensor) -> dict[int, torch.Tensor]:
    """max_gelu[b] shape: (N_CHANNELS,) — max |GELU(z_c)| over all tokens."""
    max_gelu = {b: torch.zeros(N_CHANNELS) for b in range(N_BLOCKS)}
    min_z = {b: torch.full((N_CHANNELS,), float('inf')) for b in range(N_BLOCKS)}
    max_z = {b: torch.full((N_CHANNELS,), float('-inf')) for b in range(N_BLOCKS)}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            gelu_vals = out.reshape(-1, N_CHANNELS).abs().float().cpu()
            z_vals = inp[0].reshape(-1, N_CHANNELS).float().cpu()
            max_gelu[idx] = torch.max(max_gelu[idx], gelu_vals.max(dim=0).values)
            min_z[idx] = torch.min(min_z[idx], z_vals.min(dim=0).values)
            max_z[idx] = torch.max(max_z[idx], z_vals.max(dim=0).values)
        return hook_fn

    for b in range(N_BLOCKS):
        h = model.transformer.h[b].mlp.act.register_forward_hook(make_hook(b))
        hooks.append(h)

    model.eval()
    for i in range(0, len(seqs), BATCH_SIZE):
        model(seqs[i:i + BATCH_SIZE, :-1].to(DEVICE))

    for h in hooks:
        h.remove()

    return max_gelu, min_z, max_z


# ---------------------------------------------------------------------------
# Prune dead channels
# ---------------------------------------------------------------------------
def prune_dead_channels(float_model, max_gelu: dict, epsilon: float) -> nn.Module:
    """Zero out c_fc and c_proj weights for channels where max |GELU| < epsilon."""
    pruned = copy.deepcopy(float_model)
    n_dead_per_block = []

    for b in range(N_BLOCKS):
        dead_mask = (max_gelu[b] < epsilon)  # (3072,) bool
        n_dead = int(dead_mask.sum().item())
        n_dead_per_block.append(n_dead)

        if n_dead > 0:
            # c_fc: Conv1D weight shape (768, 3072). Zero columns for dead channels.
            pruned.transformer.h[b].mlp.c_fc.weight.data[:, dead_mask] = 0.0
            pruned.transformer.h[b].mlp.c_fc.bias.data[dead_mask] = 0.0
            # c_proj: Conv1D weight shape (3072, 768). Zero rows for dead channels.
            pruned.transformer.h[b].mlp.c_proj.weight.data[dead_mask, :] = 0.0

    pruned.eval()
    return pruned, n_dead_per_block


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
float_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
float_model.eval()

print("Loading data...", flush=True)
seqs = load_tokens(tokenizer, N_SEQS_CALIB, SEED)
print(f"  {len(seqs)} seqs × {SEQ_LEN} tokens", flush=True)

# Baseline
print("Baseline PPL...", flush=True)
ppl_baseline = eval_ppl(float_model, seqs)
print(f"  Baseline PPL: {ppl_baseline:.3f}", flush=True)

if not (15.0 <= ppl_baseline <= 65.0):
    print(f"ABORT: baseline {ppl_baseline:.2f} outside [15,65]", flush=True)
    sys.exit(1)

# Collect activation statistics
print("\nCollecting per-channel max GELU activations...", flush=True)
max_gelu, min_z, max_z = collect_max_gelu(float_model, seqs)

# Print statistics
print("\nMax GELU per-block summary:", flush=True)
for b in range(N_BLOCKS):
    mg = max_gelu[b]
    n_lt_001 = int((mg < 0.001).sum())
    n_lt_01 = int((mg < 0.01).sum())
    n_lt_1 = int((mg < 0.1).sum())
    n_lt_10 = int((mg < 1.0).sum())
    print(f"  Block {b:2d}: <0.001={n_lt_001:4d}  <0.01={n_lt_01:4d}  "
          f"<0.1={n_lt_1:4d}  <1.0={n_lt_10:4d} (of {N_CHANNELS})", flush=True)

# Always-active: channels where z > 0 always (min_z > 0)
print("\nAlways-active channels (min_z > 0):", flush=True)
always_active_per_block = []
for b in range(N_BLOCKS):
    n_always = int((min_z[b] > 0).sum())
    always_active_per_block.append(n_always)
    print(f"  Block {b:2d}: {n_always:4d} always-active channels", flush=True)

# Threshold sweep
print("\nPruning sweep...", flush=True)
threshold_results = []
for eps in EPSILONS:
    pruned, n_dead = prune_dead_channels(float_model, max_gelu, eps)
    ppl = eval_ppl(pruned, seqs)
    mean_pct_dead = sum(n_dead) / (N_BLOCKS * N_CHANNELS) * 100
    print(f"  eps={eps:.3f}: mean_dead={mean_pct_dead:.2f}%  PPL={ppl:.3f}", flush=True)
    threshold_results.append({
        "epsilon": eps,
        "n_dead_per_block": n_dead,
        "total_dead": sum(n_dead),
        "mean_pct_dead": mean_pct_dead,
        "ppl": ppl,
        "ppl_delta": ppl - ppl_baseline,
    })
    del pruned

# Verdict (based on eps=0.01 result)
res_001 = next(r for r in threshold_results if r["epsilon"] == 0.01)
pct_001 = res_001["mean_pct_dead"]
ppl_001 = res_001["ppl"]

if pct_001 > 5.0 and ppl_001 < 38.0:
    verdict = "CONFIRMED"
    verdict_note = (f"eps=0.01: {pct_001:.2f}% dead channels, PPL={ppl_001:.3f} (<38). "
                    f"Exact pruning viable.")
elif pct_001 < 1.0:
    verdict = "REFUTED"
    verdict_note = (f"eps=0.01: only {pct_001:.2f}% near-dead channels. "
                    f"GPT-2 FFN has near-zero dead channels at this threshold.")
else:
    verdict = "INCONCLUSIVE"
    verdict_note = (f"eps=0.01: {pct_001:.2f}% dead channels, PPL={ppl_001:.3f}. "
                    f"Some dead channels but unclear compression benefit.")

print(f"\n{'='*60}", flush=True)
print(f"Baseline PPL: {ppl_baseline:.3f}", flush=True)
for r in threshold_results:
    print(f"  eps={r['epsilon']:.3f}: {r['mean_pct_dead']:.2f}% dead  PPL={r['ppl']:.3f}  "
          f"(Δ={r['ppl_delta']:+.3f})", flush=True)
print(f"VERDICT: {verdict}", flush=True)
print(f"{verdict_note}", flush=True)

results = {
    "cycle": 192, "item": 202,
    "verdict": verdict,
    "verdict_note": verdict_note,
    "baseline_ppl": ppl_baseline,
    "thresholds": threshold_results,
    "always_active": {
        "n_per_block": always_active_per_block,
        "mean_pct": sum(always_active_per_block) / (N_BLOCKS * N_CHANNELS) * 100,
    },
    "max_gelu_per_block_median": [float(max_gelu[b].median().item()) for b in range(N_BLOCKS)],
    "max_gelu_per_block_min": [float(max_gelu[b].min().item()) for b in range(N_BLOCKS)],
    "config": {
        "n_seqs_calib": N_SEQS_CALIB, "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN,
        "seed": SEED, "device": str(DEVICE), "n_channels": N_CHANNELS,
    },
}

out_path = RESULTS_DIR / "cycle_192_dead_channel_pruning.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}", flush=True)
