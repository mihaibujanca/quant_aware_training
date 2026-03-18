"""Cycle 009: Jacobian variance atlas for GPT-2 FFN blocks.

Compute per-block linearity index L = 1 - mean_x(||J(x) - J̄||_F / ||J̄||_F)
and correlate with single-block swap damage from cycle 005.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from aleph.lm_geometric_compression import (
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _activation_derivative,
    _extract_ffn_handles,
    _standard_weight,
    build_probe_corpus,
    load_nanochat_teacher,
)


# ── Cycle 005 per-layer 16x relative perplexity increase ──
CYCLE_005_DAMAGE = {
    0: 0.3312, 1: 0.0074, 2: 0.0114, 3: 0.0100,
    4: 0.0064, 5: 0.0012, 6: 0.0100, 7: -0.0063,
    8: 0.0195, 9: 0.0163, 10: 0.0181, 11: 0.0652,
}


def compute_linearity_index(
    weight_in: torch.Tensor,
    weight_out: torch.Tensor,
    preactivations: torch.Tensor,
    activation_kind: str,
    max_tokens: int = 256,
) -> dict:
    """Compute linearity index L for one FFN block.

    L = 1 - mean_x(||J(x) - J̄||_F / ||J̄||_F)

    Uses the efficient factored form:
      J(x) = (W_out * d(x)) @ W_in
      J̄   = (W_out * d̄) @ W_in
      J(x) - J̄ = (W_out * (d(x) - d̄)) @ W_in
    """
    N = min(max_tokens, preactivations.shape[0])
    z = preactivations[:N].to(dtype=torch.float32)

    # Compute activation derivatives for all tokens
    deriv = _activation_derivative(z, activation_kind)  # [N, H]
    d_bar = deriv.mean(dim=0)  # [H]

    # Mean Jacobian norm: ||J̄||_F = ||(W_out * d̄) @ W_in||_F
    J_bar = (weight_out * d_bar.unsqueeze(0)) @ weight_in  # [D, D]
    J_bar_norm = J_bar.norm().item()

    if J_bar_norm < 1e-12:
        return {"L": 0.0, "J_bar_norm": 0.0, "mean_ratio": 1.0, "std_ratio": 0.0}

    # Per-token deviation ratios
    ratios = []
    for i in range(N):
        delta = deriv[i] - d_bar  # [H]
        J_diff = (weight_out * delta.unsqueeze(0)) @ weight_in  # [D, D]
        ratio = J_diff.norm().item() / J_bar_norm
        ratios.append(ratio)

    ratios_t = torch.tensor(ratios)
    mean_ratio = ratios_t.mean().item()
    std_ratio = ratios_t.std().item() if N > 1 else 0.0
    L = 1.0 - mean_ratio

    return {
        "L": L,
        "J_bar_norm": J_bar_norm,
        "mean_ratio": mean_ratio,
        "std_ratio": std_ratio,
        "n_tokens": N,
    }


def main():
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load teacher and corpus
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2", device=str(device))
    )
    corpus = build_probe_corpus(
        ProbeCorpusSpec(
            teacher=teacher,
            text_path="data/shakespeare.txt",
            sequence_length=128,
            probe_train_sequences=64,
            probe_holdout_sequences=32,
            eval_val_sequences=32,
            dataset_name="proxy_shakespeare",
        )
    )

    handles = _extract_ffn_handles(teacher.model, teacher.activation_kind)
    teacher.model.eval()

    results = {}
    print(f"{'Layer':>5} | {'L':>8} | {'mean_ratio':>10} | {'J_bar_norm':>10} | {'damage':>8}")
    print("-" * 60)

    for handle in handles:
        # Collect preactivations via hook
        preact_chunks: list[torch.Tensor] = []
        max_tokens = 256

        def hook_fn(module, inputs, output):
            x = inputs[0].detach()
            z = module.c_fc(x).detach()
            flat = z.reshape(-1, z.shape[-1])
            remaining = max_tokens - sum(c.shape[0] for c in preact_chunks)
            if remaining > 0:
                take = min(remaining, flat.shape[0])
                preact_chunks.append(flat[:take].cpu())

        hook = handle.mlp.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                for start in range(0, corpus.probe_holdout.shape[0], 4):
                    batch = corpus.probe_holdout[start:start+4].to(device)
                    teacher.model(batch[:, :-1])
                    if sum(c.shape[0] for c in preact_chunks) >= max_tokens:
                        break
        finally:
            hook.remove()

        z = torch.cat(preact_chunks, dim=0).to(dtype=torch.float32)
        w_in = _standard_weight(handle.c_fc).to(dtype=torch.float32, device="cpu")
        w_out = _standard_weight(handle.c_proj).to(dtype=torch.float32, device="cpu")

        info = compute_linearity_index(w_in, w_out, z, handle.activation_kind, max_tokens=256)
        info["layer_id"] = handle.layer_id
        info["damage"] = CYCLE_005_DAMAGE.get(handle.layer_id, None)
        results[handle.layer_id] = info

        damage = info["damage"] if info["damage"] is not None else float("nan")
        print(f"{handle.layer_id:5d} | {info['L']:8.4f} | {info['mean_ratio']:10.4f} | {info['J_bar_norm']:10.2f} | {damage:8.4f}")

    # ── Correlation analysis ──
    layers = sorted(results.keys())
    L_values = [results[l]["L"] for l in layers]
    damage_values = [CYCLE_005_DAMAGE[l] for l in layers]
    # Compressibility = 1 - damage (low damage = easy to compress)
    compressibility = [1.0 - d for d in damage_values]

    rho, pval = spearmanr(L_values, compressibility)
    print(f"\nSpearman correlation (L vs compressibility): rho={rho:.4f}, p={pval:.4f}")

    # ── Criteria evaluation ──
    n_high_L = sum(1 for v in L_values if v >= 0.85)
    print(f"Blocks with L >= 0.85: {n_high_L}/12 (need >=2 for 'confirmed')")
    print(f"All blocks L < 0.6? {all(v < 0.6 for v in L_values)} (if true → 'refuted')")

    # ── Save results ──
    out = {
        "per_block": {str(l): results[l] for l in layers},
        "spearman_rho": rho,
        "spearman_pval": pval,
        "n_high_L": n_high_L,
        "all_below_0.6": all(v < 0.6 for v in L_values),
    }
    out_path = Path("autoresearch/results/cycle_009_jacobian_atlas.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.bar(layers, L_values, color="steelblue")
    ax.axhline(0.85, color="red", linestyle="--", label="L=0.85 threshold")
    ax.axhline(0.6, color="orange", linestyle="--", label="L=0.6 threshold")
    ax.set_xlabel("FFN Block")
    ax.set_ylabel("Linearity Index L")
    ax.set_title("Per-Block Linearity Index")
    ax.legend()
    ax.set_xticks(layers)

    ax = axes[1]
    ax.scatter(L_values, compressibility, c="steelblue", s=60)
    for i, l in enumerate(layers):
        ax.annotate(str(l), (L_values[i], compressibility[i]), fontsize=8,
                     xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Linearity Index L")
    ax.set_ylabel("Compressibility (1 - rel_ppl_increase)")
    ax.set_title(f"L vs Compressibility (Spearman ρ={rho:.3f})")

    plt.tight_layout()
    plot_path = Path("autoresearch/results/cycle_009_jacobian_atlas.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
