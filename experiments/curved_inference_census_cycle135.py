"""Cycle 135: Curved inference census — GPT-2 residual stream (item 145).

Compute salience and curvature under pullback metric G = U^T U.
Correlate curvature with single-block swap PPL damage at 6x.
3 seeds [42, 123, 7].

Verdict:
  CONFIRMED    if |salience-curvature rho| >= 0.7 AND |curvature-damage rho| >= 0.5
  REFUTED      if |salience-curvature rho| < 0.3
  INCONCLUSIVE otherwise
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.pullback_geometry import (
    collect_residual_trajectory,
    compute_curvature,
    compute_salience,
    get_unembedding,
)
from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    build_probe_corpus,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

SEEDS = [42, 123, 7]
N_BLOCKS = 12
D_MODEL = 768
SEQ_LEN = 128
BATCH_SIZE = 8
HIDDEN_FRACTION = 1 / 6  # 6x = 512/3072
EVAL_SEQS = 200
PROBE_SEQS = 200
MAX_TOKENS = 4096

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


if __name__ == "__main__":
    print("Loading GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    model = teacher.model
    model.eval()
    device = teacher.device
    U = get_unembedding(model)  # (50257, 768)
    print(f"  U shape: {U.shape}", flush=True)

    all_salience = []
    all_curvature = []
    all_ppl_deltas = []

    for seed in SEEDS:
        print(f"\n{'='*50}\nSEED {seed}\n{'='*50}", flush=True)
        _set_seed(seed)

        corpus = build_probe_corpus(ProbeCorpusSpec(
            teacher=teacher, text_path=str(OWT_PATH), sequence_length=SEQ_LEN,
            probe_train_sequences=PROBE_SEQS, probe_holdout_sequences=50,
            eval_val_sequences=EVAL_SEQS, dataset_name="openwebtext", shuffle_seed=seed,
        ))

        # Part A: Residual stream geometry
        print(f"  Computing trajectory geometry...", flush=True)
        salience_accum = torch.zeros(N_BLOCKS)
        curvature_accum = torch.zeros(N_BLOCKS - 1)
        n_batches = 0

        tokens = corpus.eval_val
        for i in range(0, tokens.shape[0], BATCH_SIZE):
            batch = tokens[i:i+BATCH_SIZE].to(device)
            inp = batch[:, :-1]
            trajectory = collect_residual_trajectory(model, inp)
            s = compute_salience(trajectory, U, reduce="mean")   # (12,)
            k = compute_curvature(trajectory, U, reduce="mean")  # (11,)
            salience_accum += s.cpu()
            curvature_accum += k.cpu()
            n_batches += 1

        salience_mean = (salience_accum / n_batches).numpy()  # (12,)
        curvature_mean = (curvature_accum / n_batches).numpy()  # (11,)

        print(f"  Salience profile: {' '.join(f'{s:.2f}' for s in salience_mean)}", flush=True)
        print(f"  Curvature profile: {' '.join(f'{k:.4f}' for k in curvature_mean)}", flush=True)

        # Salience-curvature correlation (align: salience[1:12] with curvature[0:11])
        rho_sk, p_sk = stats.spearmanr(salience_mean[1:], curvature_mean)
        print(f"  Salience-curvature Spearman: rho={rho_sk:.4f}, p={p_sk:.4f}", flush=True)

        # Part B: Single-block swap PPL at 6x
        print(f"  Computing per-block PPL deltas...", flush=True)
        bl = _evaluate_model(teacher, model, corpus, batch_size=BATCH_SIZE)
        baseline_ppl = bl["perplexity"]
        print(f"  Baseline PPL: {baseline_ppl:.3f}", flush=True)

        budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION)
        ppl_deltas = []

        for bid in range(N_BLOCKS):
            _set_seed(seed)
            atlas = probe_ffn_geometry(teacher, corpus, layer_ids=(bid,),
                                        max_tokens_per_layer=MAX_TOKENS, batch_size=BATCH_SIZE)
            from aleph.lm_geometric_compression import derive_ffn_compression
            pack = derive_ffn_compression(atlas, budget)
            compressed = swap_compressed_ffns(teacher, pack, inplace=False)
            compressed.eval()
            m = _evaluate_model(teacher, compressed, corpus, batch_size=BATCH_SIZE)
            delta = m["perplexity"] - baseline_ppl
            ppl_deltas.append(delta)
            print(f"    Block {bid:2d}: PPL={m['perplexity']:.3f} (delta={delta:+.3f})", flush=True)
            del compressed

        ppl_deltas = np.array(ppl_deltas)

        # Curvature-damage correlation (align: curvature[0:11] with ppl_deltas[1:12])
        rho_kd, p_kd = stats.spearmanr(curvature_mean, ppl_deltas[1:])
        print(f"  Curvature-damage Spearman: rho={rho_kd:.4f}, p={p_kd:.4f}", flush=True)

        all_salience.append(salience_mean)
        all_curvature.append(curvature_mean)
        all_ppl_deltas.append(ppl_deltas)

    # Cross-seed summary
    print(f"\n{'='*50}\nCROSS-SEED SUMMARY\n{'='*50}", flush=True)
    mean_s = np.mean(all_salience, axis=0)
    mean_k = np.mean(all_curvature, axis=0)
    mean_d = np.mean(all_ppl_deltas, axis=0)

    rho_sk_final, p_sk_final = stats.spearmanr(mean_s[1:], mean_k)
    rho_kd_final, p_kd_final = stats.spearmanr(mean_k, mean_d[1:])
    print(f"  Mean salience-curvature Spearman: rho={rho_sk_final:.4f}, p={p_sk_final:.4f}", flush=True)
    print(f"  Mean curvature-damage Spearman: rho={rho_kd_final:.4f}, p={p_kd_final:.4f}", flush=True)

    # Verdict
    abs_rho_sk = abs(rho_sk_final)
    abs_rho_kd = abs(rho_kd_final)

    if abs_rho_sk >= 0.7 and abs_rho_kd >= 0.5:
        verdict = "CONFIRMED"
        detail = f"|rho_sk|={abs_rho_sk:.3f}>=0.7, |rho_kd|={abs_rho_kd:.3f}>=0.5"
    elif abs_rho_sk < 0.3:
        verdict = "REFUTED"
        detail = f"|rho_sk|={abs_rho_sk:.3f}<0.3, no anticorrelation"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"|rho_sk|={abs_rho_sk:.3f}, |rho_kd|={abs_rho_kd:.3f}"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  Detail: {detail}", flush=True)

    # Save JSON
    output = {
        "cycle": 135, "item": 145, "seeds": SEEDS,
        "mean_salience": mean_s.tolist(),
        "mean_curvature": mean_k.tolist(),
        "mean_ppl_deltas": mean_d.tolist(),
        "rho_salience_curvature": float(rho_sk_final),
        "p_salience_curvature": float(p_sk_final),
        "rho_curvature_damage": float(rho_kd_final),
        "p_curvature_damage": float(p_kd_final),
        "per_seed": [{"salience": s.tolist(), "curvature": k.tolist(), "ppl_deltas": d.tolist()}
                     for s, k, d in zip(all_salience, all_curvature, all_ppl_deltas)],
        "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_135_curved_inference.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON", flush=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Salience + Curvature profiles
    ax = axes[0]
    ax2 = ax.twinx()
    layers_s = np.arange(12)
    layers_k = np.arange(1, 12)
    ax.plot(layers_s, mean_s, 'b-o', label=f"Salience S_l", markersize=4)
    ax2.plot(layers_k, mean_k, 'r-s', label=f"Curvature κ_l", markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Salience (G-norm)", color='b')
    ax2.set_ylabel("Curvature", color='r')
    ax.set_title(f"Residual Stream Geometry\nrho(S,κ)={rho_sk_final:.3f}")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")

    # Panel 2: Curvature vs PPL-delta scatter
    ax3 = axes[1]
    ax3.scatter(mean_k, mean_d[1:], c=layers_k, cmap='viridis', s=60, edgecolors='k', zorder=3)
    for i, lid in enumerate(layers_k):
        ax3.annotate(f"L{lid}", (mean_k[i], mean_d[1:][i]), fontsize=7, ha='center', va='bottom')
    ax3.set_xlabel("Curvature κ_l")
    ax3.set_ylabel("PPL delta (single-block 6x)")
    ax3.set_title(f"Curvature vs Compression Damage\nrho={rho_kd_final:.3f}")
    ax3.grid(alpha=0.3)

    fig.suptitle(f"Curved Inference Census — Verdict: {verdict}", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cycle_135_curved_inference.png", dpi=120)
    plt.close()
    print("Saved plot", flush=True)
    print("\nDone.", flush=True)
