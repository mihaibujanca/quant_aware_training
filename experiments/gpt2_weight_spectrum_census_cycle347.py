"""
Cycle 338: GPT-2 full 12-block weight spectrum census
Why does LASER work on GPT-2?
"""
import json
import torch
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

results = []

print(f"{'Block':>5} | {'top-1':>6} | {'top-5':>6} | {'top-13':>7} | {'r50':>4} | {'r90':>4} | {'r95':>4}")
print("-" * 55)

for i in range(12):
    # Conv1D weight shape is (n_input, n_output) = (3072, 768)
    W = model.transformer.h[i].mlp.c_proj.weight.data.float()
    # SVD — energy fractions are transpose-invariant
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    energy = S ** 2
    total = energy.sum().item()
    cumsum = energy.cumsum(0)

    def efrac(k):
        return (energy[:k].sum() / total).item() * 100.0

    def rank_for(pct):
        threshold = pct * total
        return int((cumsum < threshold).sum().item()) + 1

    t1 = efrac(1)
    t5 = efrac(5)
    t13 = efrac(13)
    r50 = rank_for(0.50)
    r90 = rank_for(0.90)
    r95 = rank_for(0.95)

    results.append({
        "block": i,
        "top1_pct": round(t1, 2),
        "top5_pct": round(t5, 2),
        "top13_pct": round(t13, 2),
        "r50": r50,
        "r90": r90,
        "r95": r95,
    })

    print(f"{i:>5} | {t1:>5.2f}% | {t5:>5.2f}% | {t13:>6.2f}% | {r50:>4} | {r90:>4} | {r95:>4}")

# Verdict
top5_vals = [r["top5_pct"] for r in results]
blocks_ge15 = sum(1 for v in top5_vals if v >= 15.0)
blocks_ge8 = sum(1 for v in top5_vals if v >= 8.0)
median_top5 = sorted(top5_vals)[len(top5_vals) // 2]

print()
print(f"Blocks with top-5 >= 15%: {blocks_ge15}")
print(f"Blocks with top-5 >= 8%: {blocks_ge8}")
print(f"Median top-5: {median_top5:.2f}%")

# Verdict logic from execution instructions:
# CONFIRMED if >=4 blocks have top-5 >= 15%
# REFUTED if only b1 has top-5 >= 15%
# INCONCLUSIVE if 2-3 blocks pass
if blocks_ge15 >= 4:
    verdict = "CONFIRMED"
elif blocks_ge15 <= 1:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"VERDICT: {verdict}")

# Save JSON
out = {
    "verdict": verdict,
    "blocks_ge15_top5": blocks_ge15,
    "blocks_ge8_top5": blocks_ge8,
    "median_top5_pct": round(median_top5, 2),
    "per_block": results,
}
with open("autoresearch/results/cycle_338_spectrum_census.json", "w") as f:
    json.dump(out, f, indent=2)

# Write manifest
manifest = f"""cycle_338_manifest
verdict: {verdict}
blocks_ge15_top5: {blocks_ge15}
blocks_ge8_top5: {blocks_ge8}
median_top5_pct: {median_top5:.2f}%
files:
  autoresearch/results/cycle_338_spectrum_census.json
  autoresearch/results/cycle_338_manifest.txt
"""
with open("autoresearch/results/cycle_338_manifest.txt", "w") as f:
    f.write(manifest)

print("Manifest written.")
