"""Cycle 187 / Item 196: Layer-drop baseline — GPT-2 FFN skip blocks [1,4,7,11].

Zero out each block's FFN contribution one at a time (residual passes unchanged).
Then drop all TARGET_BLOCKS simultaneously. Compare against float baseline.

Verdict:
  CONFIRMED    if ≥3 blocks have single-drop PPL <40
  REFUTED      if all single-drop PPL >60
  INCONCLUSIVE if 1-2 blocks have single-drop PPL <40, or best PPL in 40-60 range
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EVAL_SEQS = 200
SEQ_LEN = 128
BATCH_SIZE = 4
SEED = 42
TARGET_BLOCKS = [1, 4, 7, 11]

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_eval_tokens() -> torch.Tensor:
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    text = OWT_PATH.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    seqs = []
    stride = max(1, len(tokens) // EVAL_SEQS)
    rng = random.Random(SEED)
    offsets = list(range(0, len(tokens) - SEQ_LEN, stride))
    rng.shuffle(offsets)
    for off in offsets[:EVAL_SEQS]:
        seqs.append(tokens[off:off + SEQ_LEN])
    if len(seqs) < EVAL_SEQS:
        print(f"WARNING: only {len(seqs)} sequences available", flush=True)
    return torch.tensor(seqs, dtype=torch.long)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_ppl(model, tokens: torch.Tensor, device) -> float:
    """Compute PPL as exp(mean cross-entropy over all tokens)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, len(tokens), BATCH_SIZE):
        batch = tokens[i:i + BATCH_SIZE].to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        outputs = model(inputs)
        logits = outputs[0] if isinstance(outputs, tuple) else (
            outputs.logits if hasattr(outputs, "logits") else outputs)
        n = targets.numel()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_tokens += n
    return float(torch.exp(torch.tensor(total_loss / total_tokens)).item())


# ---------------------------------------------------------------------------
# Block drop context
# ---------------------------------------------------------------------------
class _ZeroForward:
    """Replaces an MLP's forward with a zero pass. Restores on exit."""
    def __init__(self, mlp):
        self.mlp = mlp
        self._orig = None

    def __enter__(self):
        self._orig = self.mlp.forward
        self.mlp.forward = lambda x: torch.zeros_like(x)
        return self

    def __exit__(self, *args):
        self.mlp.forward = self._orig
        self._orig = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2...", flush=True)
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else
                      ("mps" if torch.backends.mps.is_available() else "cpu"))
model = model.to(device)
print(f"  Device: {device}", flush=True)

print("Loading eval tokens...", flush=True)
tokens = load_eval_tokens()
print(f"  {len(tokens)} sequences of length {SEQ_LEN}", flush=True)

# Baseline
print("\nEvaluating baseline...", flush=True)
baseline_ppl = evaluate_ppl(model, tokens, device)
print(f"  Baseline PPL: {baseline_ppl:.3f}", flush=True)

if not (15.0 <= baseline_ppl <= 65.0):
    print(f"ABORT: baseline PPL {baseline_ppl:.2f} outside sanity range [15, 65]", flush=True)
    sys.exit(1)

# Single-block drops
print("\nSingle-block drop sweep...", flush=True)
n_blocks = len(model.transformer.h)
single_block_results = {}

for block_id in range(n_blocks):
    with _ZeroForward(model.transformer.h[block_id].mlp):
        ppl = evaluate_ppl(model, tokens, device)
    damage = ppl - baseline_ppl
    single_block_results[block_id] = {"ppl": ppl, "damage": damage}
    droppable = ppl < 40.0
    print(f"  block {block_id:2d}: PPL={ppl:.3f}  damage=+{damage:.3f}"
          f"{'  <-- droppable' if droppable else ''}", flush=True)

# Multi-block drop
print(f"\nMulti-block drop {TARGET_BLOCKS}...", flush=True)
ctxs = [_ZeroForward(model.transformer.h[b].mlp) for b in TARGET_BLOCKS]
try:
    for ctx in ctxs:
        ctx.__enter__()
    multi_ppl = evaluate_ppl(model, tokens, device)
finally:
    for ctx in reversed(ctxs):
        ctx.__exit__(None, None, None)
multi_damage = multi_ppl - baseline_ppl
print(f"  Multi-drop PPL: {multi_ppl:.3f}  damage=+{multi_damage:.3f}", flush=True)

# Ranking
sorted_blocks = sorted(single_block_results.keys(),
                       key=lambda b: single_block_results[b]["ppl"])
block_ranking = [{"block": b, "ppl": single_block_results[b]["ppl"],
                  "damage": single_block_results[b]["damage"]}
                 for b in sorted_blocks]

# Droppable = absolute PPL < 40 (verbatim from agenda)
droppable_blocks = [b for b in range(n_blocks) if single_block_results[b]["ppl"] < 40.0]
n_droppable = len(droppable_blocks)

# Verdict
if n_droppable >= 3:
    verdict = "CONFIRMED"
    verdict_note = f"{n_droppable} blocks with single-drop PPL <40: {droppable_blocks}"
elif all(single_block_results[b]["ppl"] > 60.0 for b in range(n_blocks)):
    verdict = "REFUTED"
    verdict_note = f"All {n_blocks} blocks have single-drop PPL >60 (min={min(v['ppl'] for v in single_block_results.values()):.1f})"
else:
    verdict = "INCONCLUSIVE"
    verdict_note = f"{n_droppable} blocks droppable (<40 PPL); best single-drop PPL={min(v['ppl'] for v in single_block_results.values()):.2f}"

print(f"\n{'='*60}", flush=True)
print(f"Droppable blocks (PPL <40): {droppable_blocks} ({n_droppable})", flush=True)
print(f"VERDICT: {verdict}", flush=True)
print(f"{verdict_note}", flush=True)

# Save
results = {
    "cycle": 187, "item": 196,
    "verdict": verdict, "verdict_note": verdict_note,
    "model": "gpt2",
    "eval_config": {
        "num_sequences": len(tokens), "sequence_length": SEQ_LEN,
        "batch_size": BATCH_SIZE, "seed": SEED, "device": str(device),
    },
    "baseline_ppl": baseline_ppl,
    "single_block_drops": {str(b): v for b, v in single_block_results.items()},
    "multi_block_drop": {"blocks": TARGET_BLOCKS, "ppl": multi_ppl, "damage": multi_damage},
    "block_ranking": block_ranking,
    "droppable_blocks": droppable_blocks,
}

out_path = RESULTS_DIR / "cycle_187_layer_drop.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}", flush=True)
