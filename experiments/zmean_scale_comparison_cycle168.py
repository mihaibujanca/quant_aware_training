"""
Cycle 168 / Item 178: z_mean scale comparison across GPT-2 small/medium/large.
Cycle 165 proved negativity is learned (random-init ≈ 0, trained ≈ -1) for GPT-2 small.
This cycle tests whether z_mean ≈ -1 is universal across model scales.
"""

import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# --- Config ---
N_TOKENS = 1024  # GPT-2 max context
DEVICE = "cpu"
RESULTS_DIR = "autoresearch/results"
JSON_PATH = f"{RESULTS_DIR}/cycle_168_scale_comparison.json"
PLOT_PATH = f"{RESULTS_DIR}/cycle_168_scale_comparison.png"

MODELS = {
    "small": {"name": "gpt2", "n_blocks": 12, "d_model": 768, "d_ff": 3072},
    "medium": {"name": "gpt2-medium", "n_blocks": 24, "d_model": 1024, "d_ff": 4096},
    "large": {"name": "gpt2-large", "n_blocks": 36, "d_model": 1280, "d_ff": 5120},
}

# --- Load data (shared tokenizer) ---
print("Loading wikitext-2...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text = "\n".join([t for t in ds["text"] if t.strip()])
tokens = tokenizer(text, return_tensors="pt")["input_ids"][0][:N_TOKENS]
input_ids = tokens.unsqueeze(0)

# --- Analyze each model ---
all_results = {}

for scale, cfg in MODELS.items():
    print(f"\n=== Loading {scale} ({cfg['name']}) ===")
    model = GPT2LMHeadModel.from_pretrained(cfg["name"])
    model.eval()

    # Sanity: PPL on the tokens
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
        ppl = float(torch.exp(out.loss))
    print(f"  PPL: {ppl:.1f}")

    # Hook storage
    hooks_data = {}

    def make_hook(block_idx):
        def hook_fn(module, inp, out):
            hooks_data[block_idx] = {
                "x": inp[0].detach().squeeze(0),  # (T, d_model)
                "z": out.detach().squeeze(0),      # (T, d_ff)
            }
        return hook_fn

    handles = []
    for i, block in enumerate(model.transformer.h):
        h = block.mlp.c_fc.register_forward_hook(make_hook(i))
        handles.append(h)

    with torch.no_grad():
        _ = model(input_ids)

    for h in handles:
        h.remove()

    # Per-block z_mean decomposition
    per_block = []
    for i in range(cfg["n_blocks"]):
        x = hooks_data[i]["x"]  # (T, d_model)
        z = hooks_data[i]["z"]  # (T, d_ff)

        # Bias
        bias = model.transformer.h[i].mlp.c_fc.bias.detach()  # (d_ff,)
        b_mean = float(bias.mean())

        # z_mean and Wx_mean
        z_mean = float(z.mean())
        Wx_mean = z_mean - b_mean

        # Fraction of negative z
        frac_neg = float((z < 0).float().mean())

        per_block.append({
            "block": i,
            "z_mean": round(z_mean, 4),
            "b_mean": round(b_mean, 4),
            "Wx_mean": round(Wx_mean, 4),
            "frac_neg": round(frac_neg, 4),
        })

    # Summary
    z_means = [b["z_mean"] for b in per_block]
    Wx_means = [b["Wx_mean"] for b in per_block]

    all_results[scale] = {
        "model": cfg["name"],
        "n_blocks": cfg["n_blocks"],
        "d_model": cfg["d_model"],
        "d_ff": cfg["d_ff"],
        "ppl": round(ppl, 2),
        "z_mean_avg": round(float(np.mean(z_means)), 4),
        "z_mean_std": round(float(np.std(z_means)), 4),
        "Wx_mean_avg": round(float(np.mean(Wx_means)), 4),
        "frac_neg_avg": round(float(np.mean([b["frac_neg"] for b in per_block])), 4),
        "per_block": per_block,
    }

    print(f"  z_mean avg: {all_results[scale]['z_mean_avg']:.3f} "
          f"(std: {all_results[scale]['z_mean_std']:.3f})")
    print(f"  Wx_mean avg: {all_results[scale]['Wx_mean_avg']:.3f}")
    print(f"  frac_neg avg: {all_results[scale]['frac_neg_avg']:.3f}")

    # Free memory
    del model, hooks_data
    import gc; gc.collect()

# --- Verdict ---
z_avgs = {s: abs(all_results[s]["z_mean_avg"]) for s in MODELS}
max_ratio = max(z_avgs.values()) / min(z_avgs.values())
all_negative = all(all_results[s]["z_mean_avg"] < -0.3 for s in MODELS)

if max_ratio < 2.0 and all_negative:
    verdict = "CONFIRMED"
elif max_ratio > 2.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n=== VERDICT: {verdict} ===")
print(f"  z_mean averages: small={all_results['small']['z_mean_avg']:.3f}, "
      f"medium={all_results['medium']['z_mean_avg']:.3f}, "
      f"large={all_results['large']['z_mean_avg']:.3f}")
print(f"  max_ratio={max_ratio:.2f}, all_negative={all_negative}")

# --- Save JSON ---
output = {
    "verdict": verdict,
    "max_ratio": round(max_ratio, 4),
    "all_negative": all_negative,
    "n_tokens": N_TOKENS,
    "per_scale": all_results,
}

with open(JSON_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved JSON to {JSON_PATH}")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: z_mean per block for all 3 models
colors = {"small": "#1f77b4", "medium": "#ff7f0e", "large": "#2ca02c"}
for scale in MODELS:
    blocks = [b["block"] for b in all_results[scale]["per_block"]]
    z_means_plot = [b["z_mean"] for b in all_results[scale]["per_block"]]
    label = (f"{scale} ({MODELS[scale]['d_model']}d, {MODELS[scale]['n_blocks']}L, "
             f"avg={all_results[scale]['z_mean_avg']:.2f})")
    axes[0].plot(blocks, z_means_plot, "o-", color=colors[scale], label=label,
                 markersize=4, alpha=0.8)

axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[0].axhline(y=-1.0, color="red", linestyle=":", alpha=0.3, label="z=-1 reference")
axes[0].set_xlabel("Block index")
axes[0].set_ylabel("z_mean")
axes[0].set_title(f"z_mean per block across GPT-2 scales (verdict: {verdict})")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Right: Wx_mean per block (same layout)
for scale in MODELS:
    blocks = [b["block"] for b in all_results[scale]["per_block"]]
    Wx_means_plot = [b["Wx_mean"] for b in all_results[scale]["per_block"]]
    label = f"{scale} (avg={all_results[scale]['Wx_mean_avg']:.2f})"
    axes[1].plot(blocks, Wx_means_plot, "o-", color=colors[scale], label=label,
                 markersize=4, alpha=0.8)

axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[1].set_xlabel("Block index")
axes[1].set_ylabel("Wx_mean (weight contribution only)")
axes[1].set_title("Wx_mean per block (bias removed)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved plot to {PLOT_PATH}")
print("Done.")
