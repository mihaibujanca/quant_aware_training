"""Cycle 188 / Item 198: Progressive C3 composition — per-channel mean replacement
across all 12 GPT-2 FFN blocks.

C3 = replace GELU(z_c) with mu_neg[c] = E[GELU(z_c)|z_c<0] when z_c<0,
     keep GELU(z_c) unchanged when z_c>=0.

Progressively install C3 on blocks 0..k for k=0..11 and measure PPL at each step.
Tests whether per-channel replacement composes without shape change confound
(unlike cycles 181/182 which also changed GELU→ReLU).

Verdict:
  CONFIRMED    if all-12 C3 PPL < baseline * 1.15  (+15%)
  REFUTED      if all-12 C3 PPL > baseline * 1.40  (+40%)
  INCONCLUSIVE if between +15% and +40%

Forked from experiments/regime_ablation_table_cycle173.py.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Config — match cycle 173 for comparable baseline
# ---------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
N_SEQS = 200
SEQ_LEN = 1024
BATCH_SIZE = 4
SEED = 42
N_BLOCKS = 12
N_CHANNELS = 3072  # GPT-2 FFN intermediate size

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading (verbatim from cycle 173)
# ---------------------------------------------------------------------------
def load_owt_tokens(tokenizer):
    text = OWT_PATH.read_text(encoding="utf-8")
    rng = torch.Generator()
    rng.manual_seed(SEED)
    tokens = tokenizer.encode(text)
    tokens_t = torch.tensor(tokens, dtype=torch.long)
    max_start = len(tokens_t) - SEQ_LEN - 1
    indices = torch.randperm(max_start, generator=rng)[:N_SEQS]
    seqs = torch.stack([tokens_t[i:i + SEQ_LEN + 1] for i in sorted(indices.tolist())])
    return seqs  # (N_SEQS, SEQ_LEN+1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def eval_ppl(model, seqs) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH_SIZE):
            batch = seqs[i:i + BATCH_SIZE].to(DEVICE)
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            out = model(inp)
            logits = out.logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += tgt.numel()
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# Calibration — collect per-block per-channel conditional means (verbatim cycle 173)
# ---------------------------------------------------------------------------
def collect_gelu_stats(model, seqs: torch.Tensor) -> dict:
    sum_neg = {b: torch.zeros(N_CHANNELS, dtype=torch.float64) for b in range(N_BLOCKS)}
    cnt_neg = {b: torch.zeros(N_CHANNELS, dtype=torch.float64) for b in range(N_BLOCKS)}

    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            z = inp[0]
            y = out
            z_flat = z.reshape(-1, N_CHANNELS)
            y_flat = y.reshape(-1, N_CHANNELS).float()
            mask_neg = (z_flat < 0)
            y_neg = torch.where(mask_neg, y_flat, torch.zeros_like(y_flat))
            sum_neg[idx] += y_neg.sum(dim=0).cpu().double()
            cnt_neg[idx] += mask_neg.sum(dim=0).cpu().double()
        return hook_fn

    for b in range(N_BLOCKS):
        h = model.transformer.h[b].mlp.act.register_forward_hook(make_hook(b))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH_SIZE):
            batch = seqs[i:i + BATCH_SIZE, :-1].to(DEVICE)
            model(batch)

    for h in hooks:
        h.remove()

    stats = {}
    for b in range(N_BLOCKS):
        mu_neg = (sum_neg[b] / cnt_neg[b].clamp(min=1)).float()
        stats[b] = {"mu_neg": mu_neg}
        print(f"  Block {b}: ||mu_neg||={mu_neg.norm().item():.4f}", flush=True)

    return stats


# ---------------------------------------------------------------------------
# C3 hook
# ---------------------------------------------------------------------------
class C3Hook:
    """Install a C3 forward hook on model.transformer.h[block_idx].mlp.act."""

    def __init__(self, model, block_idx: int, mu_neg: torch.Tensor):
        self.block_idx = block_idx
        self._mu = mu_neg.to(DEVICE)
        self._handle = None
        self._install(model)

    def _install(self, model):
        mu = self._mu

        def hook_fn(module, inp, out):
            z = inp[0]
            mask_neg = (z < 0)
            return torch.where(mask_neg, mu.expand_as(out), out)

        self._handle = model.transformer.h[self.block_idx].mlp.act.register_forward_hook(hook_fn)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
print(f"  Device: {DEVICE}", flush=True)
print(f"  Activation: {type(model.transformer.h[0].mlp.act).__name__}", flush=True)

print("\nLoading eval tokens...", flush=True)
seqs = load_owt_tokens(tokenizer)
print(f"  {len(seqs)} sequences of length {SEQ_LEN}", flush=True)

# Baseline
print("\nBaseline PPL...", flush=True)
baseline_ppl = eval_ppl(model, seqs)
print(f"  Baseline PPL: {baseline_ppl:.3f}", flush=True)

if not (15.0 <= baseline_ppl <= 65.0):
    print(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [15, 65]", flush=True)
    sys.exit(1)

# Calibration
print("\nCalibrating C3 means...", flush=True)
stats = collect_gelu_stats(model, seqs)

# Progressive C3 sweep
print("\nProgressive C3 sweep...", flush=True)
active_hooks: list[C3Hook] = []
progressive_ppls: list[dict] = []

for k in range(N_BLOCKS):
    hook = C3Hook(model, k, stats[k]["mu_neg"])
    active_hooks.append(hook)
    ppl_k = eval_ppl(model, seqs)
    delta = ppl_k - baseline_ppl
    pct = 100.0 * delta / baseline_ppl
    progressive_ppls.append({
        "n_blocks": k + 1,
        "last_block_added": k,
        "ppl": ppl_k,
        "delta": delta,
        "pct_increase": pct,
    })
    print(f"  C3 blocks 0..{k:2d}: PPL={ppl_k:.3f}  Δ={delta:+.3f}  ({pct:+.2f}%)", flush=True)

# Verdict on k=11 (all-12 blocks)
all12_ppl = progressive_ppls[-1]["ppl"]
threshold_confirmed = baseline_ppl * 1.15
threshold_refuted = baseline_ppl * 1.40

if all12_ppl < threshold_confirmed:
    verdict = "CONFIRMED"
    verdict_note = (f"C3 all-12 blocks PPL={all12_ppl:.2f} < baseline*1.15={threshold_confirmed:.2f} "
                    f"({progressive_ppls[-1]['pct_increase']:+.2f}%)")
elif all12_ppl > threshold_refuted:
    verdict = "REFUTED"
    verdict_note = (f"C3 all-12 blocks PPL={all12_ppl:.2f} > baseline*1.40={threshold_refuted:.2f} "
                    f"({progressive_ppls[-1]['pct_increase']:+.2f}%)")
else:
    verdict = "INCONCLUSIVE"
    verdict_note = (f"C3 all-12 blocks PPL={all12_ppl:.2f} in [{threshold_confirmed:.2f}, {threshold_refuted:.2f}] "
                    f"({progressive_ppls[-1]['pct_increase']:+.2f}%)")

print(f"\n{'='*60}", flush=True)
print(f"VERDICT: {verdict}", flush=True)
print(f"{verdict_note}", flush=True)

# Cleanup
for hook in active_hooks:
    hook.remove()

# Save
results = {
    "cycle": 188, "item": 198,
    "verdict": verdict, "verdict_note": verdict_note,
    "baseline_ppl": baseline_ppl,
    "all12_ppl": all12_ppl,
    "threshold_confirmed": threshold_confirmed,
    "threshold_refuted": threshold_refuted,
    "progressive": progressive_ppls,
    "config": {
        "n_seqs": N_SEQS, "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
        "seed": SEED, "device": str(DEVICE), "n_blocks": N_BLOCKS,
    },
}

out_path = RESULTS_DIR / "cycle_188_c3_composition.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}", flush=True)
