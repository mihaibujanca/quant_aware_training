"""Cycle 136: Pullback-metric error measurement — GPT-2 FFN (item 146).

Measure per-block FFN compression error in Frobenius and G-weighted norms at 6x.
Correlate with PPL deltas from cycle 135.

Verdict:
  CONFIRMED    if G-weighted |rho| exceeds Frobenius |rho| by >=0.15 AND G |rho| >=0.6
  REFUTED      if Frobenius |rho| >= G-weighted |rho|
  INCONCLUSIVE otherwise
"""
from __future__ import annotations

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

from aleph.pullback_geometry import get_unembedding, pullback_norm
from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

SEEDS = [42, 123, 7]
N_BLOCKS = 12
D_MODEL = 768
SEQ_LEN = 128
BATCH_SIZE = 8
HIDDEN_FRACTION = 1 / 6
EVAL_SEQS = 200
MAX_TOKENS = 4096

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
CYCLE135_JSON = RESULTS_DIR / "cycle_135_curved_inference.json"


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def collect_ffn_outputs(model, tokens, block_id, device, batch_size=8):
    """Collect FFN output activations for a specific block."""
    outputs = []

    def hook_fn(mod, inp, out):
        outputs.append(out.detach().cpu())

    h = model.transformer.h[block_id].mlp.register_forward_hook(hook_fn)
    for i in range(0, tokens.shape[0], batch_size):
        batch = tokens[i:i+batch_size].to(device)
        model(batch[:, :-1])
    h.remove()

    return torch.cat(outputs, dim=0)  # (n_batches*batch, seq, 768)


if __name__ == "__main__":
    # Load cycle 135 PPL deltas
    with open(CYCLE135_JSON) as f:
        c135 = json.load(f)
    ppl_deltas_135 = np.array(c135["mean_ppl_deltas"])  # (12,)
    print(f"Cycle 135 PPL deltas: {ppl_deltas_135.tolist()}", flush=True)

    print("Loading GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    model = teacher.model
    model.eval()
    device = teacher.device
    U = get_unembedding(model)  # (50257, 768)

    all_frob = []
    all_gw = []

    for seed in SEEDS:
        print(f"\n{'='*50}\nSEED {seed}\n{'='*50}", flush=True)
        _set_seed(seed)

        corpus = build_probe_corpus(ProbeCorpusSpec(
            teacher=teacher, text_path=str(OWT_PATH), sequence_length=SEQ_LEN,
            probe_train_sequences=200, probe_holdout_sequences=50,
            eval_val_sequences=EVAL_SEQS, dataset_name="openwebtext", shuffle_seed=seed,
        ))
        tokens = corpus.eval_val
        budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION)

        frob_errors = []
        gw_errors = []

        for bid in range(N_BLOCKS):
            # Collect float FFN outputs
            float_out = collect_ffn_outputs(model, tokens, bid, device, BATCH_SIZE)

            # Compress block
            _set_seed(seed)
            atlas = probe_ffn_geometry(teacher, corpus, layer_ids=(bid,),
                                        max_tokens_per_layer=MAX_TOKENS, batch_size=BATCH_SIZE)
            pack = derive_ffn_compression(atlas, budget)
            comp_model = swap_compressed_ffns(teacher, pack, inplace=False)
            comp_model.eval()

            # Collect compressed FFN outputs
            comp_out = collect_ffn_outputs(comp_model, tokens, bid, device, BATCH_SIZE)

            # Error
            error = comp_out - float_out  # (n, seq, 768)

            # Frobenius: mean per-token L2 norm
            frob = error.norm(dim=-1).mean().item()  # scalar

            # G-weighted: mean per-token pullback norm
            # Reshape to (n*seq, 768) for pullback_norm
            e_flat = error.reshape(-1, D_MODEL)
            gw = pullback_norm(e_flat, U).mean().item()

            frob_errors.append(frob)
            gw_errors.append(gw)
            ratio = gw / (frob + 1e-12)
            print(f"  Block {bid:2d}: frob={frob:.4f}, G-weighted={gw:.4f}, ratio={ratio:.4f}", flush=True)
            del comp_model, float_out, comp_out, error

        frob_errors = np.array(frob_errors)
        gw_errors = np.array(gw_errors)

        # Correlations with PPL delta
        rho_frob, p_frob = stats.spearmanr(frob_errors, ppl_deltas_135)
        rho_gw, p_gw = stats.spearmanr(gw_errors, ppl_deltas_135)
        print(f"\n  Spearman(Frob, PPL_delta): rho={rho_frob:.4f}, p={p_frob:.4f}", flush=True)
        print(f"  Spearman(G-wt, PPL_delta): rho={rho_gw:.4f}, p={p_gw:.4f}", flush=True)
        print(f"  G advantage: {abs(rho_gw) - abs(rho_frob):+.4f}", flush=True)

        all_frob.append(frob_errors)
        all_gw.append(gw_errors)

    # Cross-seed summary
    print(f"\n{'='*50}\nCROSS-SEED SUMMARY\n{'='*50}", flush=True)
    mean_frob = np.mean(all_frob, axis=0)
    mean_gw = np.mean(all_gw, axis=0)

    rho_frob_final, p_frob_final = stats.spearmanr(mean_frob, ppl_deltas_135)
    rho_gw_final, p_gw_final = stats.spearmanr(mean_gw, ppl_deltas_135)
    gap = abs(rho_gw_final) - abs(rho_frob_final)

    print(f"  Mean Frob Spearman: rho={rho_frob_final:.4f}", flush=True)
    print(f"  Mean G-wt Spearman: rho={rho_gw_final:.4f}", flush=True)
    print(f"  Gap: {gap:+.4f}", flush=True)

    # Per-block ratio
    ratios = mean_gw / (mean_frob + 1e-12)
    print(f"  G/Frob ratios: {' '.join(f'{r:.3f}' for r in ratios)}", flush=True)

    # Verdict
    if gap >= 0.15 and abs(rho_gw_final) >= 0.6:
        verdict = "CONFIRMED"
        detail = f"|rho_g|={abs(rho_gw_final):.3f}, |rho_f|={abs(rho_frob_final):.3f}, gap={gap:.3f}>=0.15"
    elif abs(rho_frob_final) >= abs(rho_gw_final):
        verdict = "REFUTED"
        detail = f"Frobenius |rho|={abs(rho_frob_final):.3f} >= G-weighted |rho|={abs(rho_gw_final):.3f}"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"|rho_g|={abs(rho_gw_final):.3f}, |rho_f|={abs(rho_frob_final):.3f}, gap={gap:.3f}<0.15"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  Detail: {detail}", flush=True)

    # Save
    output = {
        "cycle": 136, "item": 146, "seeds": SEEDS,
        "mean_frob_errors": mean_frob.tolist(),
        "mean_gw_errors": mean_gw.tolist(),
        "ppl_deltas": ppl_deltas_135.tolist(),
        "rho_frob": float(rho_frob_final), "p_frob": float(p_frob_final),
        "rho_gw": float(rho_gw_final), "p_gw": float(p_gw_final),
        "gap": float(gap),
        "gw_frob_ratios": ratios.tolist(),
        "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_136_pullback_error.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON", flush=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    blocks = np.arange(N_BLOCKS)

    ax1 = axes[0]
    ax1.scatter(mean_frob, ppl_deltas_135, c=blocks, cmap='viridis', s=60, edgecolors='k')
    for i in range(N_BLOCKS):
        ax1.annotate(f"B{i}", (mean_frob[i], ppl_deltas_135[i]), fontsize=7)
    ax1.set_xlabel("Frobenius error")
    ax1.set_ylabel("PPL delta")
    ax1.set_title(f"Frobenius (rho={rho_frob_final:.3f})")
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(mean_gw, ppl_deltas_135, c=blocks, cmap='viridis', s=60, edgecolors='k')
    for i in range(N_BLOCKS):
        ax2.annotate(f"B{i}", (mean_gw[i], ppl_deltas_135[i]), fontsize=7)
    ax2.set_xlabel("G-weighted error")
    ax2.set_ylabel("PPL delta")
    ax2.set_title(f"G-weighted (rho={rho_gw_final:.3f})")
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Error Metric vs PPL Damage — Verdict: {verdict}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cycle_136_scatter.png", dpi=120)
    plt.close()
    print("Saved plot", flush=True)
    print("\nDone.", flush=True)
