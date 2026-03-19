"""Cycle 050: Oracle-guided adaptive compression ratio — GPT-2 FFN.

Per-block ratios derived from P1 Frobenius std (cycle 048):
  block 4: 10.0x  (P1=5.79, easiest)
  block 1:  9.8x  (P1=5.89)
  block 5:  9.0x  (P1=6.46)
  block 7:  7.9x  (P1=7.34)

Also runs a uniform control at the mean ratio (~9.2x).
3 seeds, 200 eval sequences, lstsq boundary correction.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnPack,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ── Config ─────────────────────────────────────────────────────────────────────
SHAKESPEARE_PATH = str(Path(__file__).parent.parent / "data" / "shakespeare.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BLOCKS = [7, 5, 4, 1]
SEEDS = [42, 123, 7]
PROBE_TRAIN_SEQ = 200
PROBE_EVAL_SEQ = 200
BATCH_SIZE = 4
CORRECTION_LAYER = max(TARGET_BLOCKS) + 1  # block 8

# P1 values from cycle 048 (proxies_by_layer keys 1,4,5,7)
P1 = {1: 5.894309739602049, 4: 5.788395826727694, 5: 6.455819952102833, 7: 7.338698831589167}

# Compute oracle per-block ratios: ratio_i = 10 * (P1_min / P1_i), clamp [4, 10]
P1_MIN = min(P1.values())
HIDDEN_DIM = 3072  # GPT-2 FFN hidden
oracle_ratios: dict[int, float] = {}
oracle_hidden: dict[int, int] = {}
for block_id, p1 in P1.items():
    ratio = 10.0 * (P1_MIN / p1)
    ratio = max(4.0, min(10.0, ratio))
    oracle_ratios[block_id] = ratio
    oracle_hidden[block_id] = int(HIDDEN_DIM / ratio)

mean_oracle_ratio = sum(oracle_ratios.values()) / len(oracle_ratios)
# Uniform control at the mean oracle ratio
uniform_hidden_k = int(HIDDEN_DIM / mean_oracle_ratio)

print("Oracle ratios per block:")
for b in TARGET_BLOCKS:
    print(f"  block {b}: ratio={oracle_ratios[b]:.2f}x  →  hidden={oracle_hidden[b]}")
print(f"Mean oracle ratio: {mean_oracle_ratio:.2f}x  (uniform control hidden={uniform_hidden_k})")
print(f"Correction at block {CORRECTION_LAYER}")


def eval_ppl(model, teacher, corpus) -> float:
    metrics = _evaluate_model(teacher, model, corpus, batch_size=BATCH_SIZE)
    return float(metrics["perplexity"])


def collect_hidden_states(model, input_ids, hook_layer_idx):
    collected = []

    def hook_fn(module, args):
        collected.append(args[0].detach().clone())

    handle = model.transformer.h[hook_layer_idx].register_forward_pre_hook(hook_fn)
    with torch.no_grad():
        for i in range(0, len(input_ids), BATCH_SIZE):
            model(input_ids[i:i + BATCH_SIZE])
    handle.remove()
    return torch.cat(collected, dim=0)


def fit_correction(h_float, h_comp):
    h_f = h_float.reshape(-1, h_float.shape[-1]).float()
    h_c = h_comp.reshape(-1, h_comp.shape[-1]).float()
    H = torch.cat([h_c, torch.ones(h_c.shape[0], 1)], dim=1)
    sol = torch.linalg.lstsq(H, h_f).solution
    W = sol[:h_f.shape[1]].T
    b = sol[h_f.shape[1]]
    return W, b


def make_correction_hook(W, b):
    def hook_fn(module, args):
        h = args[0]
        return (h @ W.T + b,) + args[1:]
    return hook_fn


def run_config(label: str, budget: CompressionBudgetSpec, teacher, corpus, atlas):
    """Run k=4 accumulation + correction for a given budget. Returns dict."""
    full_pack = derive_ffn_compression(atlas, budget)
    compressed_by_id = {l.layer_id: l for l in full_pack.layers}

    baseline_ppl = eval_ppl(teacher.model, teacher, corpus)

    print(f"\n  [{label}] accumulation")
    acc_results = []
    for k in range(1, len(TARGET_BLOCKS) + 1):
        active = TARGET_BLOCKS[:k]
        sub_layers = tuple(compressed_by_id[b] for b in active)
        sub_pack = CompressedFfnPack(
            method=full_pack.method, model_family=full_pack.model_family,
            dataset_name=full_pack.dataset_name, layers=sub_layers, metadata=full_pack.metadata,
        )
        student = swap_compressed_ffns(teacher, sub_pack, inplace=False)
        ppl = eval_ppl(student, teacher, corpus)
        rel = (ppl - baseline_ppl) / baseline_ppl * 100
        print(f"    k={k} blocks={active} PPL={ppl:.2f} ({rel:+.2f}%)")
        acc_results.append({"k": k, "blocks": active, "ppl": float(ppl), "rel_pct": float(rel)})
        if k == len(TARGET_BLOCKS):
            k4_model = student
        else:
            del student

    # lstsq boundary correction
    probe_ids = corpus.probe_train[:, :-1]
    h_float = collect_hidden_states(teacher.model, probe_ids, CORRECTION_LAYER)
    h_comp = collect_hidden_states(k4_model, probe_ids, CORRECTION_LAYER)
    W_corr, b_corr = fit_correction(h_float, h_comp)

    mse_before = F.mse_loss(h_comp.reshape(-1, 768).float(), h_float.reshape(-1, 768).float()).item()
    h_fixed = h_comp.reshape(-1, 768).float() @ W_corr.T + b_corr
    mse_after = F.mse_loss(h_fixed, h_float.reshape(-1, 768).float()).item()

    hook = k4_model.transformer.h[CORRECTION_LAYER].register_forward_pre_hook(
        make_correction_hook(W_corr, b_corr)
    )
    corr_ppl = eval_ppl(k4_model, teacher, corpus)
    corr_rel = (corr_ppl - baseline_ppl) / baseline_ppl * 100
    hook.remove()

    k4_uncorr_rel = acc_results[-1]["rel_pct"]
    recovery = k4_uncorr_rel - corr_rel
    print(f"  [{label}] k=4 corrected: {corr_ppl:.2f} ({corr_rel:+.2f}%) recovery={recovery:+.2f}pp")
    print(f"  [{label}] MSE: {mse_before:.4f} → {mse_after:.4f}")

    del k4_model
    return {
        "accumulation": acc_results,
        "corrected_ppl": float(corr_ppl),
        "corrected_rel_pct": float(corr_rel),
        "recovery_pp": float(recovery),
        "mse_before": float(mse_before),
        "mse_after": float(mse_after),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
print("\nLoading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)

all_results: dict = {
    "oracle": [],
    "uniform_control": [],
    "oracle_ratios": oracle_ratios,
    "oracle_hidden": oracle_hidden,
    "mean_oracle_ratio": float(mean_oracle_ratio),
    "uniform_control_hidden": uniform_hidden_k,
    "baselines": {},
}

oracle_budget = CompressionBudgetSpec(
    hidden_fraction=0.1,  # fallback (not used since per_layer_hidden overrides)
    per_layer_hidden=oracle_hidden,
    selection_mode="topk",
)
uniform_budget = CompressionBudgetSpec(
    hidden_fraction=float(uniform_hidden_k) / HIDDEN_DIM,
    selection_mode="topk",
)

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    corpus_spec = ProbeCorpusSpec(
        teacher=teacher,
        text_path=SHAKESPEARE_PATH,
        sequence_length=128,
        probe_train_sequences=PROBE_TRAIN_SEQ,
        probe_holdout_sequences=PROBE_EVAL_SEQ,
        eval_val_sequences=PROBE_EVAL_SEQ,
        dataset_name="proxy_shakespeare",
        shuffle_seed=seed,
    )
    corpus = build_probe_corpus(corpus_spec)
    baseline_ppl = eval_ppl(teacher.model, teacher, corpus)
    all_results["baselines"][str(seed)] = float(baseline_ppl)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    print("  Probing FFN geometry...")
    atlas = probe_ffn_geometry(
        teacher, corpus, layer_ids=tuple(TARGET_BLOCKS),
        max_tokens_per_layer=2048, batch_size=BATCH_SIZE, jacobian_samples=8,
    )

    # Oracle adaptive
    oracle_res = run_config("oracle-adaptive", oracle_budget, teacher, corpus, atlas)
    oracle_res["seed"] = seed
    oracle_res["baseline_ppl"] = float(baseline_ppl)
    all_results["oracle"].append(oracle_res)

    # Uniform control
    uniform_res = run_config("uniform-control", uniform_budget, teacher, corpus, atlas)
    uniform_res["seed"] = seed
    uniform_res["baseline_ppl"] = float(baseline_ppl)
    all_results["uniform_control"].append(uniform_res)

# ── Aggregate ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("AGGREGATE RESULTS")
print(f"{'='*60}")

def summarize(entries, label):
    rels = [e["corrected_rel_pct"] for e in entries]
    m, s = float(np.mean(rels)), float(np.std(rels, ddof=1) if len(rels) > 1 else 0.0)
    recs = [e["recovery_pp"] for e in entries]
    print(f"  {label}: mean={m:+.2f}% ±{s:.2f}  recovery={np.mean(recs):+.2f}pp")
    return m, s

oracle_mean, oracle_std = summarize(all_results["oracle"], f"oracle-adaptive ({mean_oracle_ratio:.1f}x mean)")
uniform_mean, uniform_std = summarize(all_results["uniform_control"], f"uniform-control ({mean_oracle_ratio:.1f}x)")

# Reference baselines from prior cycles
print(f"\n  Reference baselines:")
print(f"    Uniform 6x uncorrected (c036, k=4): +9.59%")
print(f"    Uniform 6x corrected   (c039, k=4): +8.35%")
print(f"    Uniform 10x corrected  (c042, k=4): +8.45% mean")

# Verdict
print(f"\n{'='*60}")
if oracle_mean <= 8.0:
    verdict = "confirmed"
    print(f"Verdict: CONFIRMED — oracle k=4 corrected mean={oracle_mean:+.2f}% (≤8.0%)")
elif oracle_mean > 8.45:
    verdict = "refuted"
    print(f"Verdict: REFUTED — oracle mean={oracle_mean:+.2f}% > uniform 10x baseline (+8.45%)")
else:
    verdict = "inconclusive"
    print(f"Verdict: INCONCLUSIVE — oracle mean={oracle_mean:+.2f}% (8.0–8.45%)")

all_results["summary"] = {
    "oracle_corrected_mean": oracle_mean,
    "oracle_corrected_std": oracle_std,
    "uniform_control_mean": uniform_mean,
    "uniform_control_std": uniform_std,
    "mean_oracle_ratio": float(mean_oracle_ratio),
    "delta_vs_uniform_control_pp": float(uniform_mean - oracle_mean),
    "verdict": verdict,
}
all_results["verdict"] = verdict

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ks = list(range(1, 5))

# Panel 1: accumulation curves (oracle vs uniform control)
for label, entries, color in [
    ("oracle-adaptive", all_results["oracle"], "darkorange"),
    ("uniform-control", all_results["uniform_control"], "steelblue"),
]:
    for k in ks:
        vals = [e["accumulation"][k - 1]["rel_pct"] for e in entries]
        axes[0].errorbar(
            k, np.mean(vals), yerr=(np.std(vals, ddof=1) if len(vals) > 1 else 0),
            marker="o", color=color, capsize=4,
        )
    means = [np.mean([e["accumulation"][k - 1]["rel_pct"] for e in entries]) for k in ks]
    stds = [np.std([e["accumulation"][k - 1]["rel_pct"] for e in entries], ddof=1) if len(entries) > 1 else 0 for k in ks]
    axes[0].errorbar(ks, means, yerr=stds, marker="o", color=color, lw=2, capsize=4, label=label)

# k=4 corrected
axes[0].errorbar([4.1], [oracle_mean], yerr=[oracle_std], marker="s", color="darkorange",
                 ms=10, capsize=4, label=f"oracle corrected ({oracle_mean:+.2f}%)")
axes[0].errorbar([3.9], [uniform_mean], yerr=[uniform_std], marker="D", color="steelblue",
                 ms=10, capsize=4, label=f"uniform corrected ({uniform_mean:+.2f}%)")

axes[0].axhline(8.45, color="gray", ls="--", alpha=0.7, label="c042 10x corrected (+8.45%)")
axes[0].axhline(8.35, color="lightblue", ls="--", alpha=0.7, label="c039 6x corrected (+8.35%)")
axes[0].set_xlabel("k (blocks compressed)")
axes[0].set_ylabel("Relative PPL change (%)")
axes[0].set_title(f"Cycle 050: Oracle vs Uniform Accumulation\n(mean ratio={mean_oracle_ratio:.1f}x)")
axes[0].legend(fontsize=8, loc="upper left")
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(ks)

# Panel 2: per-block ratio bar chart
blocks_sorted = sorted(oracle_ratios.keys())
ratios = [oracle_ratios[b] for b in blocks_sorted]
bars = axes[1].bar([str(b) for b in blocks_sorted], ratios, color="darkorange", alpha=0.7, label="oracle ratio")
axes[1].axhline(mean_oracle_ratio, color="gray", ls="--", label=f"mean {mean_oracle_ratio:.2f}x")
axes[1].axhline(10.0, color="lightgray", ls=":", label="max 10x")
axes[1].axhline(4.0, color="lightgray", ls=":", label="min 4x")
for bar, b in zip(bars, blocks_sorted):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{oracle_ratios[b]:.2f}x\n(k={oracle_hidden[b]})", ha="center", fontsize=9)
axes[1].set_xlabel("Block ID")
axes[1].set_ylabel("Compression ratio")
axes[1].set_title("Oracle per-block ratios (P1-guided)")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis="y")
axes[1].set_ylim(0, 12)

plt.suptitle(f"Cycle 050 — Oracle Adaptive Compression | Verdict: {verdict.upper()}", fontweight="bold")
plt.tight_layout()

out_png = RESULTS_DIR / "cycle_050_oracle_adaptive.png"
plt.savefig(out_png, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved plot: {out_png}")

out_json = RESULTS_DIR / "cycle_050_oracle_adaptive.json"
with open(out_json, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Saved data: {out_json}")
