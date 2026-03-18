"""Cycle 018: Procrustes linearity profile + identity-layer pruning — GPT-2.

For each consecutive layer pair (i, i+1), compute:
  linearity[i] = ||X_i^T X_{i+1}||_* / (||X_i||_F * ||X_{i+1}||_F)
where X_i is the residual stream after block i (or after embedding for i=0).
||·||_* is the nuclear norm (sum of singular values).

Then replace top-2 most linear blocks with identity (zero residual contribution)
and measure perplexity. Correlate linearity ranking with cycle 005 damage.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from aleph.lm_geometric_compression import (
    ProbeCorpusSpec,
    TeacherLoadSpec,
    build_probe_corpus,
    load_nanochat_teacher,
)


# ── Cycle 005 per-layer 16x relative perplexity increase ──
CYCLE_005_DAMAGE = {
    0: 0.3312, 1: 0.0074, 2: 0.0114, 3: 0.0100,
    4: 0.0064, 5: 0.0012, 6: 0.0100, 7: -0.0063,
    8: 0.0195, 9: 0.0163, 10: 0.0181, 11: 0.0652,
}


def compute_perplexity(model, tokens: torch.Tensor, device: torch.device, batch_size: int = 4) -> float:
    """Compute average per-token cross-entropy loss (perplexity = exp(loss))."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            batch = tokens[start:start + batch_size].to(device)
            # Shift: input = batch[:, :-1], targets = batch[:, 1:]
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            out = model(input_ids, use_cache=False)
            logits = out.logits  # [B, T-1, V]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += target_ids.numel()
    avg_loss = total_loss / total_tokens
    return float(torch.exp(torch.tensor(avg_loss)).item())


def collect_residual_streams(model, tokens: torch.Tensor, device: torch.device, batch_size: int = 4) -> list[torch.Tensor]:
    """Collect residual stream representations at 13 checkpoints.

    Checkpoints:
      [0]  = output of embedding (wte + wpe)
      [i+1] = output of transformer.h[i] for i in 0..11

    Returns list of 13 tensors, each [N*T, 768].
    """
    n_layers = len(model.transformer.h)
    # Storage: list of chunk lists per checkpoint
    chunks: list[list[torch.Tensor]] = [[] for _ in range(n_layers + 1)]

    # Hook for embedding output (after wte + wpe, before any block)
    # In GPT-2, the forward flow is:
    #   hidden = wte(input_ids) + wpe(position_ids)   <- checkpoint 0
    #   for each block: hidden = block(hidden)         <- checkpoint i+1

    handles = []

    # We'll use a custom forward pass approach: run with output_hidden_states=True
    model.eval()
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            batch = tokens[start:start + batch_size].to(device)
            input_ids = batch[:, :-1]  # [B, T-1]
            out = model(input_ids, use_cache=False, output_hidden_states=True)
            # out.hidden_states: tuple of (n_layers + 1) tensors, each [B, T-1, 768]
            # hidden_states[0] = after embedding
            # hidden_states[i+1] = after block i
            for ck, hs in enumerate(out.hidden_states):
                flat = hs.detach().reshape(-1, hs.shape[-1]).cpu()
                chunks[ck].append(flat)

    # Concatenate
    return [torch.cat(chunks[ck], dim=0).float() for ck in range(n_layers + 1)]


def procrustes_linearity(X: torch.Tensor, Y: torch.Tensor) -> dict:
    """Compute Procrustes linearity score between consecutive residual streams.

    From "Your Transformer is Secretly Linear" (2405.12250):
      linearity = ||X^T Y||_* / (||X||_F * ||Y||_F)

    where ||·||_* is nuclear norm (sum of singular values) of the [D, D] cross-covariance.

    Score = 1 when the mapping X→Y is a perfect scaled orthogonal transformation.
    Score ∈ [0, 1] in general (bounded by Cauchy-Schwarz for nuclear norm).

    Returns dict with score and diagnostics.
    """
    N, D = X.shape
    norm_X = X.norm(p="fro")
    norm_Y = Y.norm(p="fro")
    if norm_X < 1e-12 or norm_Y < 1e-12:
        return {"linearity": 0.0, "norm_X": norm_X.item(), "norm_Y": norm_Y.item()}

    # M = X^T Y is (D, D) cross-covariance
    M = X.T @ Y  # [D, D]
    sv = torch.linalg.svdvals(M)
    nuclear_norm = sv.sum().item()

    # Denominator: ||X||_F * ||Y||_F
    denom = norm_X.item() * norm_Y.item()
    score = nuclear_norm / denom

    return {
        "linearity": score,
        "nuclear_norm": nuclear_norm,
        "denom": denom,
        "norm_X": norm_X.item(),
        "norm_Y": norm_Y.item(),
        "sv_max": sv[0].item(),
        "sv_min": sv[-1].item(),
    }


def evaluate_with_identity_blocks(model, tokens: torch.Tensor, device: torch.device,
                                   identity_block_ids: list[int], batch_size: int = 4) -> float:
    """Evaluate perplexity with specified blocks zeroed out.

    For each block in identity_block_ids: the block's residual contribution is zeroed.
    In GPT-2's residual stream: hidden = hidden + block(hidden).
    Setting block to identity means: hidden = hidden + 0 = hidden.
    We achieve this by hooking block output and returning zeros.
    """
    hooks = []
    for block_id in identity_block_ids:
        block = model.transformer.h[block_id]

        def make_hook(bid):
            def hook_fn(module, inputs, output):
                # output is a tuple: (hidden_states, present, ...) or just hidden_states
                # In GPT-2 forward: attn_output is returned as tuple[0]
                # But the block forward returns: (hidden_states, ...) where hidden_states
                # has already added the residual. We need to intercept BEFORE residual add.
                # Actually GPT-2 block forward does:
                #   residual = hidden_states
                #   hidden_states = self.ln_1(hidden_states)
                #   attn_output = self.attn(...)
                #   hidden_states = attn_output[0] + residual   <- residual connection
                #   ...
                # The block returns the full output (after residual). We can't cleanly
                # zero only the "increment" without patching internals.
                # Instead: output[0] = input (pass through unchanged).
                # inputs[0] is the hidden state going INTO the block.
                # We return inputs[0] as the block output (pure identity).
                if isinstance(output, tuple):
                    return (inputs[0],) + output[1:]
                else:
                    return inputs[0]
            return hook_fn

        h = block.register_forward_hook(make_hook(block_id))
        hooks.append(h)

    try:
        ppl = compute_perplexity(model, tokens, device, batch_size=batch_size)
    finally:
        for h in hooks:
            h.remove()

    return ppl


def main():
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load teacher and corpus ──
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

    model = teacher.model
    model.eval()
    n_layers = len(model.transformer.h)
    print(f"GPT-2 layers: {n_layers}")

    # ── P1: Baseline perplexity ──
    baseline_ppl = compute_perplexity(model, corpus.probe_holdout, device)
    print(f"Baseline perplexity: {baseline_ppl:.3f}")

    # ── P2-P3: Collect residual streams ──
    print("Collecting residual streams (output_hidden_states=True)...")
    streams = collect_residual_streams(model, corpus.probe_holdout, device)
    print(f"Collected {len(streams)} checkpoints, each {streams[0].shape}")

    # ── Compute Procrustes linearity for each consecutive pair ──
    linearity_results = {}
    print(f"\n{'Layer pair':>12} | {'linearity':>10} | {'nuclear_norm':>12} | {'damage':>8}")
    print("-" * 55)

    for i in range(n_layers):
        X = streams[i]    # after block i-1 (or embedding if i=0)
        Y = streams[i+1]  # after block i
        info = procrustes_linearity(X, Y)
        info["layer_id"] = i
        info["damage"] = CYCLE_005_DAMAGE.get(i)
        linearity_results[i] = info
        damage = info["damage"] if info["damage"] is not None else float("nan")
        print(f"  {i:2d}→{i+1:2d}       | {info['linearity']:10.6f} | {info['nuclear_norm']:12.4f} | {damage:8.4f}")

    # ── Rank by linearity (descending = most linear first) ──
    linearity_scores = [linearity_results[i]["linearity"] for i in range(n_layers)]
    ranked_layers = sorted(range(n_layers), key=lambda i: linearity_scores[i], reverse=True)
    top2 = ranked_layers[:2]
    print(f"\nLinearity ranking (highest first): {ranked_layers}")
    print(f"Top-2 most linear blocks: {top2}")
    n_above_threshold = sum(1 for s in linearity_scores if s >= 0.98)
    print(f"Layers with linearity >= 0.98: {n_above_threshold}/12")

    # ── P4: Identity replacement intervention ──
    print(f"\nRunning identity intervention on blocks {top2}...")
    identity_ppl = evaluate_with_identity_blocks(model, corpus.probe_holdout, device, top2)
    ppl_increase = (identity_ppl - baseline_ppl) / baseline_ppl
    print(f"Identity PPL: {identity_ppl:.3f}, increase: {ppl_increase:.1%}")

    # Also test each block individually
    single_identity_ppls = {}
    for bid in range(n_layers):
        ppl_i = evaluate_with_identity_blocks(model, corpus.probe_holdout, device, [bid])
        rel_increase = (ppl_i - baseline_ppl) / baseline_ppl
        single_identity_ppls[bid] = {"ppl": ppl_i, "rel_increase": rel_increase}
        print(f"  Block {bid:2d} identity: PPL={ppl_i:.3f}, rel_increase={rel_increase:.1%}")

    # ── Spearman correlation ──
    layers = list(range(n_layers))
    damage_vals = [CYCLE_005_DAMAGE[l] for l in layers]
    lin_vals = linearity_scores

    # High linearity ↔ low damage (easy to compress)
    rho, pval = spearmanr(lin_vals, [-d for d in damage_vals])
    print(f"\nSpearman(linearity, -damage): rho={rho:.4f}, p={pval:.4f}")

    # Also check: linearity vs single_identity_ppl increase (lower identity cost = more linear?)
    identity_increases = [single_identity_ppls[l]["rel_increase"] for l in layers]
    rho2, pval2 = spearmanr(lin_vals, [-x for x in identity_increases])
    print(f"Spearman(linearity, -identity_ppl_increase): rho={rho2:.4f}, p={pval2:.4f}")

    # ── Criteria evaluation ──
    print("\n── Criteria ──")
    crit_1 = n_above_threshold >= 4
    crit_2 = abs(ppl_increase) <= 0.10
    crit_3 = rho >= 0.4
    print(f"C1: >=4 layers with linearity>=0.98: {n_above_threshold} -> {'PASS' if crit_1 else 'FAIL'}")
    print(f"C2: top-2 identity <= 10% ppl increase: {ppl_increase:.1%} -> {'PASS' if crit_2 else 'FAIL'}")
    print(f"C3: Spearman(linearity, -damage) >= 0.4: {rho:.3f} -> {'PASS' if crit_3 else 'FAIL'}")
    all_below_090 = all(s < 0.90 for s in lin_vals)
    print(f"Refuted: all linearity < 0.90: {all_below_090}")

    if crit_1 and crit_2 and crit_3:
        verdict = "CONFIRMED"
    elif all_below_090:
        verdict = "REFUTED"
    elif not crit_1 and not all_below_090:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PARTIAL"
    print(f"Verdict: {verdict}")

    # ── Save JSON ──
    out = {
        "baseline_ppl": baseline_ppl,
        "identity_ppl_top2": identity_ppl,
        "identity_blocks": top2,
        "ppl_increase_top2": ppl_increase,
        "linearity_scores": {str(i): linearity_results[i] for i in range(n_layers)},
        "linearity_ranking": ranked_layers,
        "n_above_098": n_above_threshold,
        "spearman_linearity_vs_neg_damage": {"rho": rho, "pval": pval},
        "spearman_linearity_vs_neg_identity_increase": {"rho": rho2, "pval": pval2},
        "single_identity_ppls": {str(k): v for k, v in single_identity_ppls.items()},
        "verdict": verdict,
        "criteria": {"C1": bool(crit_1), "C2": bool(crit_2), "C3": bool(crit_3)},
    }
    out_path = Path("autoresearch/results/cycle_018_linearity.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Linearity bar chart
    ax = axes[0]
    colors = ["firebrick" if i in top2 else "steelblue" for i in layers]
    bars = ax.bar(layers, lin_vals, color=colors)
    ax.axhline(0.98, color="red", linestyle="--", linewidth=1.5, label="0.98 threshold")
    ax.axhline(0.90, color="orange", linestyle="--", linewidth=1.0, label="0.90 threshold")
    ax.set_xlabel("Block Index")
    ax.set_ylabel("Procrustes Linearity Score")
    ax.set_title("Procrustes Linearity per Block\n(red = identity replacement)")
    ax.set_xticks(layers)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # Panel 2: Linearity vs damage
    ax = axes[1]
    ax.scatter(lin_vals, damage_vals, c="steelblue", s=60)
    for i in layers:
        ax.annotate(str(i), (lin_vals[i], damage_vals[i]), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Procrustes Linearity")
    ax.set_ylabel("Cycle 005 Damage (rel ppl increase)")
    ax.set_title(f"Linearity vs Compression Damage\n(Spearman rho={rho:.3f}, p={pval:.3f})")

    # Panel 3: Single-block identity ppl increase
    ax = axes[2]
    id_increases = [single_identity_ppls[i]["rel_increase"] for i in layers]
    colors2 = ["firebrick" if i in top2 else "steelblue" for i in layers]
    ax.bar(layers, id_increases, color=colors2)
    ax.axhline(0.10, color="red", linestyle="--", linewidth=1.5, label="10% threshold")
    ax.set_xlabel("Block Index")
    ax.set_ylabel("Rel. Perplexity Increase")
    ax.set_title("Single-Block Identity PPL Cost\n(red = top-2 linearity blocks)")
    ax.set_xticks(layers)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = Path("autoresearch/results/cycle_018_linearity.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
