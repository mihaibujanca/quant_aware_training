"""Cycle 015 — Item 26: Intrinsic dimension trajectory through GPT-2 depth.

TwoNN estimator (Facco et al. 2017) applied at each of the 13 residual-stream
checkpoints (embedding + 12 blocks). Correlates per-block ID change with
cycle-005 compression damage (Spearman).
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy import stats

from aleph.lm_geometric_compression import (
    TeacherLoadSpec,
    ProbeCorpusSpec,
    load_nanochat_teacher,
    build_probe_corpus,
)

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_HOLDOUT = 32
SEQ_LEN = 128
RESULTS_DIR = Path("autoresearch/results")
JSONL_PATH = Path("runs/nanochat_single_block_localization_overnight_main/results.jsonl")


# ── Helpers ──────────────────────────────────────────────────────────────────

def twonn_id(X: torch.Tensor) -> float:
    """TwoNN MLE intrinsic dimension (Facco et al. 2017).

    Args:
        X: (N, D) tensor of representations.
    Returns:
        Scalar intrinsic dimension estimate.
    """
    X = X.float()
    # Pairwise L2 distances: (N, N)
    D = torch.cdist(X, X)
    # Sort each row; index 0 is self (distance 0), 1 is nearest, 2 is 2nd nearest
    sorted_d, _ = torch.sort(D, dim=1)
    d1 = sorted_d[:, 1]  # nearest neighbour distances
    d2 = sorted_d[:, 2]  # 2nd nearest neighbour distances
    # Remove any points where d1 == 0 (duplicates)
    mask = d1 > 0
    d1 = d1[mask]
    d2 = d2[mask]
    mu = d2 / d1  # ratio >= 1
    mu = torch.clamp(mu, min=1.0 + 1e-8)
    N = mu.shape[0]
    id_estimate = float(N / torch.sum(torch.log(mu)).item())
    return id_estimate


def load_cycle005_damage() -> list[float]:
    """Return per-layer mean relative perplexity increase for geometric 16× swap."""
    records = [json.loads(l) for l in JSONL_PATH.read_text().splitlines() if l.strip()]
    filtered = [
        r for r in records
        if r["method"] == "geometric" and abs(r["hidden_fraction"] - 0.0625) < 1e-9
    ]
    by_layer: dict[int, list[float]] = defaultdict(list)
    for r in filtered:
        by_layer[r["layer_id"]].append(r["relative_perplexity_increase"])
    return [
        sum(by_layer[lid]) / len(by_layer[lid])
        for lid in sorted(by_layer.keys())
    ]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    print(f"Device: {DEVICE}")

    # Load model + corpus
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2", device=str(DEVICE))
    )
    corpus = build_probe_corpus(
        ProbeCorpusSpec(
            teacher=teacher,
            text_path="data/shakespeare.txt",
            sequence_length=SEQ_LEN,
            probe_train_sequences=64,
            probe_holdout_sequences=N_HOLDOUT,
            eval_val_sequences=32,
            dataset_name="proxy_shakespeare",
        )
    )

    model = teacher.model
    model.eval()

    # ── Register hooks on each transformer block ─────────────────────────────
    # GPT-2 HuggingFace: model.transformer.h[i] are the blocks
    # We capture residual-stream output (hidden_states) after each block.
    activations: dict[int, list[torch.Tensor]] = defaultdict(list)

    hooks = []
    for i, block in enumerate(model.transformer.h):
        def make_hook(layer_idx):
            def hook(module, inputs, output):
                # output is a tuple; first element is hidden_states (B, T, D)
                h = output[0].detach().cpu()  # (B, T, D)
                activations[layer_idx].append(h)
            return hook
        hooks.append(block.register_forward_hook(make_hook(i)))

    # Also capture embedding output (before block 0)
    emb_activations: list[torch.Tensor] = []

    def emb_hook(module, inputs, output):
        # output shape: (B, T, D)
        emb_activations.append(output.detach().cpu())

    emb_hook_handle = model.transformer.drop.register_forward_hook(emb_hook)

    # ── Forward pass over holdout sequences ─────────────────────────────────
    holdout_ids = corpus.probe_holdout  # (N_holdout, T) on device
    print(f"Running forward pass over {holdout_ids.shape[0]} holdout sequences ×{SEQ_LEN} tokens...")
    with torch.no_grad():
        # Process in batches to avoid OOM on pairwise distance later
        BATCH = 8
        for start in range(0, holdout_ids.shape[0], BATCH):
            batch = holdout_ids[start:start + BATCH].to(DEVICE)
            model(batch)

    # Remove hooks
    for h in hooks:
        h.remove()
    emb_hook_handle.remove()

    # ── Collect representations at each checkpoint ───────────────────────────
    # Checkpoint 0 = embedding, checkpoints 1-12 = after blocks 0-11
    checkpoints: list[torch.Tensor] = []

    # Embedding: stack all batches → (N_seq, T, D) → flatten to (N_tokens, D)
    emb_all = torch.cat(emb_activations, dim=0)  # (N_seq, T, D)
    N_seq, T, D = emb_all.shape
    checkpoints.append(emb_all.reshape(-1, D))  # (N_tokens, D)

    for i in range(12):
        block_act = torch.cat(activations[i], dim=0)  # (N_seq, T, D)
        checkpoints.append(block_act.reshape(-1, D))

    N_tokens = checkpoints[0].shape[0]
    print(f"N_tokens={N_tokens}, D={D}, n_checkpoints={len(checkpoints)}")

    # ── TwoNN at each checkpoint ─────────────────────────────────────────────
    # With N_tokens=4096, D=768 on CPU: pairwise distance is 4096×4096 float32 = 64MB, fine
    # But TwoNN only needs k=2; no need for full NxN. Use torch.topk to be memory-efficient.
    print("Computing TwoNN intrinsic dimensions...")

    def twonn_id_efficient(X: torch.Tensor) -> float:
        """Memory-efficient TwoNN: uses topk instead of full NxN matrix."""
        X = X.float()
        N = X.shape[0]
        # Chunk to avoid OOM with large N
        chunk = 512
        d1_list, d2_list = [], []
        for start in range(0, N, chunk):
            Xc = X[start:start + chunk]  # (C, D)
            dists = torch.cdist(Xc, X)   # (C, N)
            top3, _ = torch.topk(dists, k=3, dim=1, largest=False)  # (C, 3)
            # index 0 = self (0), index 1 = nearest, index 2 = 2nd nearest
            d1_list.append(top3[:, 1])
            d2_list.append(top3[:, 2])
        d1 = torch.cat(d1_list)
        d2 = torch.cat(d2_list)
        mask = d1 > 0
        d1 = d1[mask]; d2 = d2[mask]
        mu = torch.clamp(d2 / d1, min=1.0 + 1e-8)
        return float(mu.shape[0] / torch.sum(torch.log(mu)).item())

    ids = []
    for ck_idx, X in enumerate(checkpoints):
        id_val = twonn_id_efficient(X)
        print(f"  checkpoint {ck_idx:2d}: ID={id_val:.2f}  (shape {tuple(X.shape)})")
        ids.append(id_val)

    # ── Per-block delta ID ───────────────────────────────────────────────────
    delta_ids = [abs(ids[i] - ids[i - 1]) for i in range(1, 13)]  # 12 values, aligning with blocks 0-11

    # ── Cycle 005 damage ─────────────────────────────────────────────────────
    damage = load_cycle005_damage()
    assert len(damage) == 12, f"Expected 12 damage values, got {len(damage)}"
    print(f"Damage: {[f'{d:.4f}' for d in damage]}")

    # ── Spearman correlation ──────────────────────────────────────────────────
    rho, p_val = stats.spearmanr(delta_ids, damage)
    print(f"Spearman rho={rho:.4f}, p={p_val:.4f}")

    # ── Final-third drop ──────────────────────────────────────────────────────
    # Checkpoints 0-12 correspond to: 0=embed, 1=after block0, ..., 12=after block11
    # "First third" = layers 0-4 → checkpoints 0-4
    # "Final third" = layers 9-12 → checkpoints 9-12
    initial_mean = sum(ids[0:5]) / 5
    final_mean = sum(ids[9:13]) / 4
    drop_frac = 1.0 - final_mean / initial_mean
    print(f"Initial mean ID (ck 0-4): {initial_mean:.2f}")
    print(f"Final mean ID (ck 9-12): {final_mean:.2f}")
    print(f"Drop fraction: {drop_frac:.4f} ({drop_frac*100:.1f}%)")

    # ── Verdict ──────────────────────────────────────────────────────────────
    confirmed = drop_frac >= 0.30 and rho >= 0.4
    refuted = max(ids) - min(ids) < 0.1 * sum(ids) / len(ids)
    if confirmed:
        verdict = "CONFIRMED"
    elif refuted:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\nVerdict: {verdict}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {
        "layer_ids": list(range(13)),
        "intrinsic_dims": ids,
        "delta_ids": delta_ids,
        "cycle005_damage": damage,
        "spearman_rho": rho,
        "spearman_p": p_val,
        "final_third_drop_frac": drop_frac,
        "initial_mean_id": initial_mean,
        "final_mean_id": final_mean,
        "verdict": verdict,
        "n_tokens": N_tokens,
    }
    json_path = RESULTS_DIR / "cycle_015_intrinsic_dim.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(range(13), ids, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.axvspan(9, 12, alpha=0.12, color="salmon", label="Final third (blocks 9-11)")
    ax.axvspan(0, 4, alpha=0.12, color="lightgreen", label="First third (ck 0-4)")
    ax.set_xlabel("Checkpoint (0=embed, 1-12=after block)")
    ax.set_ylabel("TwoNN Intrinsic Dimension")
    ax.set_title(f"GPT-2 Intrinsic Dimension Trajectory\n(N={N_tokens} tokens, D=768)")
    ax.legend(fontsize=8)
    ax.text(
        0.97, 0.05,
        f"Drop: {drop_frac*100:.1f}%",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.8),
    )

    ax = axes[1]
    ax.scatter(delta_ids, damage, color="steelblue", s=60, zorder=3)
    for i, (dx, dy) in enumerate(zip(delta_ids, damage)):
        ax.annotate(f"B{i}", (dx, dy), textcoords="offset points", xytext=(4, 4), fontsize=7)
    # Fit line
    m, b = float(torch.tensor(delta_ids).float() @ torch.tensor(damage).float() / (torch.tensor(delta_ids).float() ** 2).sum()), 0
    xs = sorted(delta_ids)
    rho_s, p_s = stats.spearmanr(delta_ids, damage)
    ax.set_xlabel("|ΔID| per block")
    ax.set_ylabel("Cycle-005 damage (rel. PPL increase)")
    ax.set_title(f"Per-block |ΔID| vs Compression Damage\nSpearman ρ={rho_s:.3f}, p={p_s:.3f}")
    ax.text(
        0.97, 0.95,
        f"ρ = {rho_s:.3f}\np = {p_s:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_015_intrinsic_dim.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")

    return verdict, rho, drop_frac


if __name__ == "__main__":
    main()
