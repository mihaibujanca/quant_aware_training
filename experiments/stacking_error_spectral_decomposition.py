"""
Cycle 072: Stacking error spectral decomposition — GPT-2 FFN.

For k=4 blocks [7,5,4,1] at 6x (HIDDEN_FRACTION=1/6), compute per-block
reconstruction error and project onto Jacobian SVD basis.

Compare single-block vs stacked spectral profiles:
  - single: only one block compressed, local compression error
  - stacked: all 4 blocks compressed, accumulated error at each block's output

Confirmed if >=70% of stacking error energy is in discarded spectral directions
AND stacking shifts error into retained spectrum vs single-block.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    build_probe_corpus,
    load_nanochat_teacher,
    probe_ffn_geometry,
    derive_ffn_compression,
    swap_compressed_ffns,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_BLOCKS = [7, 5, 4, 1]
HIDDEN_FRACTION = 1 / 6
SEEDS = [42, 123, 7]
SEQ_LEN = 128
PROBE_TRAIN_SEQS = 8   # 8×128 = 1024 tokens for probe
PROBE_HOLDOUT_SEQS = 4
EVAL_SEQS = 4
DEVICE = "cpu"

DATA_PATH = ROOT / "data" / "shakespeare.txt"
OUTPUT_DIR = ROOT / "autoresearch" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cumulative energy sample points (k=1,16,32,...,768)
_SAMPLE_K = list(range(1, 769, 16))
if _SAMPLE_K[-1] != 768:
    _SAMPLE_K.append(768)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _std_weight(module: nn.Module) -> torch.Tensor:
    """Return (out_features, in_features) weight."""
    w = module.weight.detach().float()
    if module.__class__.__name__ == "Conv1D":
        return w.t().contiguous()
    return w.contiguous()


def _gelu_grad(z: torch.Tensor) -> torch.Tensor:
    """Element-wise GELU'(z) via autograd (tanh approximation)."""
    z2 = z.detach().requires_grad_(True)
    F.gelu(z2, approximate="tanh").sum().backward()
    return z2.grad.detach()


def _mean_jacobian_svd(block_idx: int, model: nn.Module, x_probe: torch.Tensor):
    """
    J̄ = mean_t( W_out @ diag(GELU'(z_t)) @ W_in ), shape (768,768).
    Returns U (768,768), S (768,), rank_99.
    """
    mlp = model.transformer.h[block_idx].mlp
    W_in = _std_weight(mlp.c_fc)    # (3072, 768)
    W_out = _std_weight(mlp.c_proj) # (768, 3072)

    b_fc = mlp.c_fc.bias.detach().float() if mlp.c_fc.bias is not None else None
    z = x_probe @ W_in.t()  # (T, 3072)
    if b_fc is not None:
        z = z + b_fc

    T = x_probe.shape[0]
    J_sum = torch.zeros(768, 768, dtype=torch.float32)
    for t in range(T):
        g_t = _gelu_grad(z[t])  # (3072,)
        J_sum.add_((W_out * g_t.unsqueeze(0)) @ W_in)

    J_mean = J_sum / T
    U, S, Vh = torch.linalg.svd(J_mean, full_matrices=True)  # U: (768,768)

    S2 = S ** 2
    cumS2 = torch.cumsum(S2, dim=0)
    total_S2 = S2.sum()
    rank_99 = int((cumS2 / total_S2 < 0.99).sum().item()) + 1
    rank_99 = min(rank_99, 768)

    return U, S, rank_99


def _random_orthonormal(seed_key: int, n: int = 768) -> torch.Tensor:
    """Random orthonormal (768, 768) basis."""
    gen = torch.Generator()
    gen.manual_seed(seed_key)
    A = torch.randn(n, n, generator=gen)
    Q, _ = torch.linalg.qr(A)
    return Q.contiguous()


def _spectral_fractions(e: torch.Tensor, U: torch.Tensor, rank_99: int):
    """
    e: (T, 768), U: (768, 768) orthonormal columns.
    Returns frac_retained, frac_discarded, cumulative energy at _SAMPLE_K points.
    """
    proj = e @ U  # (T, 768)
    per_dir = (proj ** 2).sum(dim=0)  # energy per direction
    total = (e ** 2).sum().item()

    cum_energy = torch.cumsum(per_dir, dim=0)
    cum_frac = (cum_energy / total).tolist()

    frac_retained = float(per_dir[:rank_99].sum().item() / total)
    frac_discarded = float(per_dir[rank_99:].sum().item() / total)
    cum_at_samples = [cum_frac[min(k - 1, 767)] for k in _SAMPLE_K]
    return frac_retained, frac_discarded, cum_at_samples


def _collect_ffn_errors(
    float_model: nn.Module,
    compressed_model: nn.Module,
    block_idx: int,
    token_ids: torch.Tensor,
    batch_size: int = 4,
) -> torch.Tensor:
    """
    Hook block_idx mlp output in BOTH models on SAME inputs.
    Returns error = comp_output - float_output, shape (T, 768).
    Both models are run in separate forward passes on same token_ids.
    """
    float_outs = []
    comp_outs = []

    def _hook_float(module, inp, out):
        float_outs.append(out.detach().float().reshape(-1, out.shape[-1]))

    def _hook_comp(module, inp, out):
        comp_outs.append(out.detach().float().reshape(-1, out.shape[-1]))

    h_float = float_model.transformer.h[block_idx].mlp.register_forward_hook(_hook_float)
    h_comp = compressed_model.transformer.h[block_idx].mlp.register_forward_hook(_hook_comp)
    try:
        N = token_ids.shape[0]
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = token_ids[i:i + batch_size].to(DEVICE)
                float_model(batch, use_cache=False)
                compressed_model(batch, use_cache=False)
    finally:
        h_float.remove()
        h_comp.remove()

    e_float = torch.cat(float_outs, dim=0)
    e_comp = torch.cat(comp_outs, dim=0)
    return e_comp - e_float


def _collect_ffn_inputs(
    model: nn.Module,
    block_idx: int,
    token_ids: torch.Tensor,
    batch_size: int = 4,
) -> torch.Tensor:
    """Collect FFN inputs (pre-MLP) for Jacobian computation."""
    x_list = []

    def _hook(module, inp, out):
        x_list.append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]))

    h = model.transformer.h[block_idx].mlp.register_forward_hook(_hook)
    try:
        N = token_ids.shape[0]
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = token_ids[i:i + batch_size].to(DEVICE)
                model(batch, use_cache=False)
    finally:
        h.remove()

    return torch.cat(x_list, dim=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_seed(seed: int, teacher_bundle, probe_tokens: torch.Tensor):
    """Run all analysis for one seed. Returns per-block result dicts."""
    print(f"\n=== Seed {seed} ===")

    float_model = teacher_bundle.model
    float_model.eval()

    corpus = build_probe_corpus(
        ProbeCorpusSpec(
            text_path=str(DATA_PATH),
            sequence_length=SEQ_LEN,
            probe_train_sequences=PROBE_TRAIN_SEQS,
            probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
            eval_val_sequences=EVAL_SEQS,
            shuffle_seed=seed,
            teacher=teacher_bundle,
        )
    )

    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        selection_mode="topk",
    )

    # Build all-4-compressed model
    atlas_all4 = probe_ffn_geometry(teacher_bundle, corpus, layer_ids=tuple(TARGET_BLOCKS))
    pack_all4 = derive_ffn_compression(atlas_all4, budget)
    model_all4 = swap_compressed_ffns(teacher_bundle, pack_all4, inplace=False)
    model_all4.eval()

    block_results = []
    for block_idx in TARGET_BLOCKS:
        print(f"  Block {block_idx}...")

        # --- Jacobian SVD from float model ---
        x_float = _collect_ffn_inputs(float_model, block_idx, probe_tokens)
        U, S, rank_99 = _mean_jacobian_svd(block_idx, float_model, x_float)

        print(f"    rank_99={rank_99}, S[0]={S[0]:.4f}, S[-1]={S[-1]:.6f}")
        if rank_99 > 700:
            print(f"    WARNING: rank_99={rank_99} > 700 — near full rank, degenerate")

        # --- Single-block error ---
        atlas_single = probe_ffn_geometry(teacher_bundle, corpus, layer_ids=(block_idx,))
        pack_single = derive_ffn_compression(atlas_single, budget)
        model_single = swap_compressed_ffns(teacher_bundle, pack_single, inplace=False)
        model_single.eval()

        e_single = _collect_ffn_errors(float_model, model_single, block_idx, probe_tokens)
        norm_single = float(e_single.norm().item())
        print(f"    single error norm={norm_single:.4f}")

        # --- Stacked error at this block ---
        e_stacked = _collect_ffn_errors(float_model, model_all4, block_idx, probe_tokens)
        norm_stacked = float(e_stacked.norm().item())
        print(f"    stacked error norm={norm_stacked:.4f}")

        # --- Spectral projection ---
        frac_ret_single, frac_disc_single, cum_single = _spectral_fractions(e_single, U, rank_99)
        frac_ret_stacked, frac_disc_stacked, cum_stacked = _spectral_fractions(e_stacked, U, rank_99)

        # Null baseline: random orthonormal basis
        Q_rand = _random_orthonormal(seed_key=seed * 100 + block_idx)
        frac_ret_rand, frac_disc_rand, cum_rand = _spectral_fractions(e_single, Q_rand, rank_99)
        expected_null = rank_99 / 768

        print(f"    frac_retained: single={frac_ret_single:.3f}, stacked={frac_ret_stacked:.3f}, "
              f"null={frac_ret_rand:.3f} (expected~{expected_null:.3f})")

        # Validation: energy conservation
        for label, fr, fd in [("single", frac_ret_single, frac_disc_single),
                               ("stacked", frac_ret_stacked, frac_disc_stacked)]:
            total = fr + fd
            if not (0.999 <= total <= 1.001):
                print(f"    WARNING energy conservation FAIL for {label}: {total:.6f}")

        block_results.append({
            "block_idx": block_idx,
            "rank_99": rank_99,
            "degenerate": rank_99 > 700,
            "S_top5": S[:5].tolist(),
            "frac_retained_single": frac_ret_single,
            "frac_discarded_single": frac_disc_single,
            "frac_retained_stacked": frac_ret_stacked,
            "frac_discarded_stacked": frac_disc_stacked,
            "frac_retained_random": frac_ret_rand,
            "total_error_norm_single": norm_single,
            "total_error_norm_stacked": norm_stacked,
            "cum_single": cum_single,
            "cum_stacked": cum_stacked,
            "cum_random": cum_rand,
            "expected_null_frac": expected_null,
        })

    return block_results


def main():
    print("Loading GPT-2 teacher...")
    load_spec = TeacherLoadSpec(hf_model_name="gpt2", device=DEVICE)
    teacher_bundle = load_nanochat_teacher(load_spec)
    float_model = teacher_bundle.model
    float_model.eval()
    tokenizer = teacher_bundle.tokenizer

    # Build fixed probe tokens (same across seeds) for Jacobian + error measurement
    text = DATA_PATH.read_text(encoding="utf-8")
    enc = tokenizer(text[:50000], return_tensors="pt")["input_ids"][0]
    n_seqs = PROBE_TRAIN_SEQS
    probe_tokens_list = []
    for i in range(n_seqs):
        start = i * SEQ_LEN
        end = start + SEQ_LEN
        if end > len(enc):
            break
        probe_tokens_list.append(enc[start:end].unsqueeze(0))
    probe_tokens = torch.cat(probe_tokens_list, dim=0)  # (N, SEQ_LEN)
    print(f"Probe tokens: {probe_tokens.shape}")

    # Run all seeds
    all_seed_results = []
    for seed in SEEDS:
        seed_results = run_seed(seed, teacher_bundle, probe_tokens)
        all_seed_results.append({"seed": seed, "blocks": seed_results})

    # ---------------------------------------------------------------------------
    # Aggregate across seeds
    # ---------------------------------------------------------------------------
    aggregated = {}
    for i, block_idx in enumerate(TARGET_BLOCKS):
        rows = [r["blocks"][i] for r in all_seed_results]
        rank_99_vals = [r["rank_99"] for r in rows]

        def _mean_std(key):
            vals = [r[key] for r in rows]
            t = torch.tensor(vals, dtype=torch.float32)
            return float(t.mean()), float(t.std())

        fr_s_mean, fr_s_std = _mean_std("frac_retained_single")
        fr_st_mean, fr_st_std = _mean_std("frac_retained_stacked")
        fr_r_mean, fr_r_std = _mean_std("frac_retained_random")
        norm_s_mean, norm_s_std = _mean_std("total_error_norm_single")
        norm_st_mean, norm_st_std = _mean_std("total_error_norm_stacked")

        aggregated[block_idx] = {
            "rank_99_mean": float(torch.tensor(rank_99_vals, dtype=torch.float32).mean()),
            "frac_retained_single_mean": fr_s_mean,
            "frac_retained_single_std": fr_s_std,
            "frac_retained_stacked_mean": fr_st_mean,
            "frac_retained_stacked_std": fr_st_std,
            "frac_retained_random_mean": fr_r_mean,
            "frac_retained_random_std": fr_r_std,
            "norm_single_mean": norm_s_mean,
            "norm_single_std": norm_s_std,
            "norm_stacked_mean": norm_st_mean,
            "norm_stacked_std": norm_st_std,
        }

    print("\n=== Aggregated results ===")
    for block_idx in TARGET_BLOCKS:
        a = aggregated[block_idx]
        print(f"Block {block_idx}: rank_99={a['rank_99_mean']:.0f}, "
              f"frac_ret_single={a['frac_retained_single_mean']:.3f}±{a['frac_retained_single_std']:.3f}, "
              f"frac_ret_stacked={a['frac_retained_stacked_mean']:.3f}±{a['frac_retained_stacked_std']:.3f}, "
              f"null={a['frac_retained_random_mean']:.3f}±{a['frac_retained_random_std']:.3f}")

    # ---------------------------------------------------------------------------
    # Classify outcome
    # ---------------------------------------------------------------------------
    all_disc_above_70 = all(
        (1 - aggregated[b]["frac_retained_stacked_mean"]) >= 0.70
        for b in TARGET_BLOCKS
    )
    stacking_shifts_retained = all(
        aggregated[b]["frac_retained_stacked_mean"] > aggregated[b]["frac_retained_single_mean"]
        for b in TARGET_BLOCKS
    )

    if all_disc_above_70 and stacking_shifts_retained:
        outcome = "confirmed"
    elif all_disc_above_70 and not stacking_shifts_retained:
        outcome = "inconclusive"
    else:
        outcome = "refuted"

    print(f"\nOutcome: {outcome}")
    print(f"  all_disc_above_70={all_disc_above_70}, stacking_shifts_retained={stacking_shifts_retained}")

    # ---------------------------------------------------------------------------
    # Plot: 2x2 grid, one per block
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Stacking Error Spectral Decomposition — GPT-2 FFN\n"
                 f"Blocks {TARGET_BLOCKS}, HIDDEN_FRACTION=1/6 (6x), Seeds {SEEDS}", fontsize=11)

    for ax, block_idx in zip(axes.flat, TARGET_BLOCKS):
        a = aggregated[block_idx]
        rank_99 = int(round(a["rank_99_mean"]))
        b0 = next(b for b in all_seed_results[0]["blocks"] if b["block_idx"] == block_idx)

        xs = _SAMPLE_K
        ax.plot(xs, b0["cum_single"], color="royalblue", linewidth=1.8, label="single-block error")
        ax.plot(xs, b0["cum_stacked"], color="tomato", linewidth=1.8, label="stacked error")
        ax.plot(xs, b0["cum_random"], color="grey", linewidth=1.2, linestyle="--", label="random null")
        ax.axvline(rank_99, color="black", linewidth=1.0, linestyle=":", alpha=0.8)

        frs = a["frac_retained_single_mean"]
        frst = a["frac_retained_stacked_mean"]
        frr = a["frac_retained_random_mean"]
        ax.set_title(f"Block {block_idx}  rank₉₉={rank_99}\n"
                     f"frac_ret: single={frs:.2f}  stacked={frst:.2f}  null={frr:.2f}",
                     fontsize=9)
        ax.set_xlabel("SVD rank k")
        ax.set_ylabel("cumulative energy fraction")
        ax.set_xlim(0, 768)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "cycle_072_stacking_error_spectrum.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")

    # ---------------------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------------------
    json_data = {
        "config": {
            "blocks": TARGET_BLOCKS,
            "hidden_fraction": HIDDEN_FRACTION,
            "seeds": SEEDS,
            "probe_train_seqs": PROBE_TRAIN_SEQS,
            "seq_len": SEQ_LEN,
            "sample_k_points": _SAMPLE_K,
        },
        "outcome": outcome,
        "aggregated": {str(k): v for k, v in aggregated.items()},
        "per_seed": all_seed_results,
    }
    json_path = OUTPUT_DIR / "cycle_072_stacking_error_spectrum.json"
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"JSON saved: {json_path}")

    return outcome, aggregated


if __name__ == "__main__":
    main()
