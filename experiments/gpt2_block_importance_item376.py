"""
Cycle 367 / Item 376: Block importance pruning for GPT-2.
Zero out FFN contribution per block, measure PPL delta.
Blocks with mean |delta| < 0.5 are "prunable".
"""
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ──────────────────────────────────────────────────────────────────
SEQ_LEN = 128
SEQS_PER_BATCH = 100
BATCH_OFFSETS = [0, 12800, 25600]   # token offsets for 3 disjoint batches
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
THRESHOLD = 0.5   # PPL delta below which a block is "prunable"
RESULT_JSON = "autoresearch/results/cycle_367_block_importance.json"
RESULT_PNG  = "autoresearch/results/cycle_367_block_importance.png"

# ── Load model & tokenizer ───────────────────────────────────────────────────
print("Loading GPT-2 …")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# ── Prepare token corpus ─────────────────────────────────────────────────────
print("Tokenizing wikitext-2 test split …")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(dataset["text"])
tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)  # (N_tokens,)
print(f"  Total tokens: {len(tokens)}")

def make_batch(offset: int) -> torch.Tensor:
    """Return (SEQS_PER_BATCH, SEQ_LEN) token tensor from corpus."""
    needed = SEQS_PER_BATCH * SEQ_LEN
    chunk = tokens[offset : offset + needed]
    if len(chunk) < needed:
        # wrap around
        chunk = torch.cat([chunk, tokens[:needed - len(chunk)]])
    return chunk.view(SEQS_PER_BATCH, SEQ_LEN).to(DEVICE)

batches = [make_batch(off) for off in BATCH_OFFSETS]

# ── PPL computation ──────────────────────────────────────────────────────────
@torch.no_grad()
def compute_ppl(batch: torch.Tensor) -> float:
    """Cross-entropy PPL on batch (seqs, seq_len)."""
    input_ids  = batch[:, :-1]
    target_ids = batch[:, 1:]
    logits = model(input_ids).logits          # (B, T-1, V)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
    )
    return torch.exp(loss).item()

# ── Float baseline ────────────────────────────────────────────────────────────
print("\nFloat baseline PPL:")
baseline_ppls = [compute_ppl(b) for b in batches]
print(f"  Batch PPLs: {[f'{p:.2f}' for p in baseline_ppls]}")
assert all(50 <= p <= 80 for p in baseline_ppls), \
    f"Sanity check failed — expected PPL in [50,80], got {baseline_ppls}"
print(f"  (Note: reference PPL 53.89 uses sliding-window; fixed-window gives ~64-66 — consistent)")

# ── Block ablation ────────────────────────────────────────────────────────────
n_blocks = model.config.n_layer   # 12
print(f"\nAblating {n_blocks} blocks …")

results = {}    # block_idx -> {"deltas": [...], "mean": ..., "std": ...}

for block_idx in range(n_blocks):
    # Register hook that zeros FFN output
    def make_hook():
        def hook(module, input, output):
            # GPT-2 MLP output is a single tensor
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        return hook

    handle = model.transformer.h[block_idx].mlp.register_forward_hook(make_hook())

    ppls = [compute_ppl(b) for b in batches]
    handle.remove()

    deltas = [ppl - base for ppl, base in zip(ppls, baseline_ppls)]
    mean_d = sum(deltas) / len(deltas)
    std_d  = (sum((d - mean_d)**2 for d in deltas) / len(deltas))**0.5

    results[block_idx] = {
        "ppls":   ppls,
        "deltas": deltas,
        "mean":   mean_d,
        "std":    std_d,
        "prunable": abs(mean_d) < THRESHOLD,
    }
    prunable_str = "PRUNABLE" if abs(mean_d) < THRESHOLD else "important"
    print(f"  Block {block_idx:2d}: mean_delta={mean_d:+.2f}  std={std_d:.2f}  [{prunable_str}]")

# ── Ranking ───────────────────────────────────────────────────────────────────
ranking = sorted(results.keys(), key=lambda i: abs(results[i]["mean"]))
prunable_blocks = [i for i in ranking if results[i]["prunable"]]
print(f"\nRanking (most prunable first): {ranking}")
print(f"Prunable blocks (|delta|<{THRESHOLD}): {prunable_blocks}")

# ── Cross-reference with cycle 366 ────────────────────────────────────────────
suspicious = []
for b in [10, 11]:
    if results[b]["prunable"]:
        suspicious.append(b)
        print(f"  WARNING: block {b} appears prunable but cycle-366 showed high error growth — suspicious!")

# ── Verdict ───────────────────────────────────────────────────────────────────
n_prunable = len(prunable_blocks)
# INCONCLUSIVE if std > 0.5 for any borderline block (|mean| in [0, 1.0])
borderline = [i for i in range(n_blocks)
              if abs(results[i]["mean"]) < 1.0 and results[i]["std"] > 0.5]
if borderline:
    verdict = "INCONCLUSIVE"
elif n_prunable >= 2:
    verdict = "CONFIRMED"
else:
    verdict = "REFUTED"

print(f"\nVERDICT: {verdict}  (prunable={n_prunable}, borderline_high_variance={borderline})")

# ── Save JSON ─────────────────────────────────────────────────────────────────
output = {
    "baseline_ppls": baseline_ppls,
    "blocks": {str(k): v for k, v in results.items()},
    "ranking": ranking,
    "prunable_blocks": prunable_blocks,
    "suspicious_blocks": suspicious,
    "verdict": verdict,
    "threshold": THRESHOLD,
}
with open(RESULT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved JSON → {RESULT_JSON}")

# ── Bar chart ─────────────────────────────────────────────────────────────────
means = [results[i]["mean"] for i in range(n_blocks)]
stds  = [results[i]["std"]  for i in range(n_blocks)]
colors = ["tab:green" if abs(m) < THRESHOLD else "tab:blue" for m in means]

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(n_blocks), means, yerr=stds, color=colors, capsize=4)
ax.axhline(THRESHOLD,  color="red", linestyle="--", label=f"+{THRESHOLD} threshold")
ax.axhline(-THRESHOLD, color="red", linestyle="--")
ax.axhline(0, color="black", linewidth=0.7)
ax.set_xlabel("Block index")
ax.set_ylabel("PPL delta (zeroed FFN − float)")
ax.set_title(f"GPT-2 FFN block importance  [verdict: {verdict}]")
ax.set_xticks(range(n_blocks))
ax.legend()
plt.tight_layout()
plt.savefig(RESULT_PNG, dpi=120)
print(f"Saved chart → {RESULT_PNG}")
