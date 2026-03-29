"""
Cycle 103: GELU-ReLU Disagreement Anatomy — GPT-2 FFN

Dissects WHERE GELU and ReLU disagree in GPT-2 FFN blocks:
- Threshold sweep over bump zones |z| < T
- Signed-side tracking (neg leak vs pos attenuation)
- Token and channel anatomy
- Masking experiment: replace GELU(z) with ReLU(z) in bump zone only
"""

import json
import math
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ─────────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_TOKENS = 4096
SEQ_LEN = 128
THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5]
PRIMARY_T = 2.5
SEEDS = [42, 123, 7]
N_BLOCKS = 12
RESULTS_DIR = "autoresearch/results"
ABORT_PPL_LO, ABORT_PPL_HI = 15.0, 65.0

torch.manual_seed(0)
print(f"Device: {DEVICE}")


# ── Data loading ────────────────────────────────────────────────────────────
def load_owt_tokens(tokenizer, n_tokens, seq_len, seed=42):
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    rng = torch.Generator()
    rng.manual_seed(seed)
    tokens = []
    for item in dataset:
        enc = tokenizer(item["text"], truncation=False, return_tensors="pt")
        tokens.append(enc["input_ids"][0])
        if sum(t.numel() for t in tokens) >= n_tokens + seq_len * 20:
            break
    all_tokens = torch.cat(tokens)
    # Build fixed sequences
    seqs = []
    for i in range(0, len(all_tokens) - seq_len, seq_len):
        seqs.append(all_tokens[i : i + seq_len])
        if len(seqs) * seq_len >= n_tokens + seq_len:
            break
    return torch.stack(seqs)  # [N, seq_len]


# ── Baseline PPL ────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_ppl(model, seqs, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for seq in seqs:
        seq = seq.unsqueeze(0).to(device)
        out = model(seq, labels=seq)
        loss = out.loss.item()
        n = seq.shape[1] - 1
        total_loss += loss * n
        total_tokens += n
    return math.exp(total_loss / total_tokens)


# ── PPL with top-1 change tracking ─────────────────────────────────────────
@torch.no_grad()
def compute_ppl_and_top1(model, seqs, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_preds = []
    for seq in seqs:
        seq = seq.unsqueeze(0).to(device)
        out = model(seq, labels=seq)
        total_loss += out.loss.item() * (seq.shape[1] - 1)
        total_tokens += seq.shape[1] - 1
        preds = out.logits[0, :-1].argmax(dim=-1)  # [L-1]
        all_preds.append(preds.cpu())
    ppl = math.exp(total_loss / total_tokens)
    all_preds = torch.cat(all_preds)
    return ppl, all_preds


# ── Hook-based pre-activation collection ────────────────────────────────────
def collect_preactivations(model, seqs, device, n_blocks=12):
    """Collect FFN pre-activations z = W_in @ x + b_in from each block."""
    z_store = {}

    def make_hook(block_idx):
        def hook(module, input, output):
            z_store[block_idx] = input[0].detach().cpu()  # pre-activation
        return hook

    hooks = []
    for b in range(n_blocks):
        ffn = model.transformer.h[b].mlp
        # Hook the activation function (new_gelu / gelu)
        act = ffn.act
        h = act.register_forward_hook(make_hook(b))
        hooks.append(h)

    with torch.no_grad():
        for seq in seqs:
            seq = seq.unsqueeze(0).to(device)
            model(seq)

    for h in hooks:
        h.remove()

    return z_store  # dict: block -> [N*L, hidden_dim] (after concat across seqs)


def collect_all_preactivations(model, seqs, device, n_blocks=12):
    """Collect and concatenate pre-activations across all sequences."""
    all_z = defaultdict(list)

    def make_hook(block_idx):
        def hook(module, input, output):
            # input[0] shape: [1, L, 4H]
            all_z[block_idx].append(input[0].detach().cpu().squeeze(0))
        return hook

    hooks = []
    for b in range(n_blocks):
        ffn = model.transformer.h[b].mlp
        act = ffn.act
        h = act.register_forward_hook(make_hook(b))
        hooks.append(h)

    with torch.no_grad():
        for seq in seqs:
            seq = seq.unsqueeze(0).to(device)
            model(seq)

    for h in hooks:
        h.remove()

    # Concatenate: [total_tokens, 4H]
    result = {}
    for b in range(n_blocks):
        result[b] = torch.cat(all_z[b], dim=0)  # [N*L, 4H]
    return result


# ── Threshold analysis ──────────────────────────────────────────────────────
def analyze_threshold(z_dict, thresholds, n_blocks):
    """
    For each threshold T: compute per-block zone fraction and energy fraction.
    z_dict: {block: tensor [tokens, channels]}
    Returns: {threshold: {block: {zone_frac, energy_frac, neg_frac, pos_frac, neg_energy_frac, pos_energy_frac}}}
    """
    results = {}
    for T in thresholds:
        results[T] = {}
        for b in range(n_blocks):
            z = z_dict[b].float()  # [tokens, channels]
            gelu_z = F.gelu(z)
            relu_z = F.relu(z)
            diff = gelu_z - relu_z  # disagreement

            bump_mask = z.abs() < T
            zone_frac = bump_mask.float().mean().item()

            # Energy = sum(|diff|^2) in bump zone / sum(|gelu_z|^2)
            diff_sq = diff ** 2
            gelu_sq = gelu_z ** 2
            energy_frac = (diff_sq * bump_mask.float()).sum().item() / (gelu_sq.sum().item() + 1e-10)

            # Signed sides: z<0 (neg leak) vs z>0 (pos attenuation)
            neg_mask = bump_mask & (z < 0)
            pos_mask = bump_mask & (z >= 0)
            total_bump = bump_mask.float().sum().item() + 1e-10
            neg_frac = neg_mask.float().sum().item() / total_bump
            pos_frac = pos_mask.float().sum().item() / total_bump

            total_diff_sq = diff_sq.sum().item() + 1e-10
            neg_energy_frac = (diff_sq * neg_mask.float()).sum().item() / total_diff_sq
            pos_energy_frac = (diff_sq * pos_mask.float()).sum().item() / total_diff_sq

            results[T][b] = dict(
                zone_frac=zone_frac,
                energy_frac=energy_frac,
                neg_frac=neg_frac,
                pos_frac=pos_frac,
                neg_energy_frac=neg_energy_frac,
                pos_energy_frac=pos_energy_frac,
            )
    return results


# ── Token and channel anatomy ────────────────────────────────────────────────
def token_channel_anatomy(z_dict, seqs, T, n_blocks, tokenizer, top_k=20):
    """
    For bump zone at threshold T:
    - Top tokens by bump fraction (aggregated over blocks and channels)
    - Top channels by bump fraction (aggregated over blocks)
    seqs: [N, L] token tensor
    """
    flat_tokens = seqs.reshape(-1)  # [N*L]
    all_token_bump = defaultdict(lambda: [0, 0])  # token_id -> [bump_count, total]
    all_channel_bump = defaultdict(lambda: [0, 0])  # channel -> [bump_count, total]

    for b in range(n_blocks):
        z = z_dict[b].float()  # [N*L, 4H]
        bump_mask = z.abs() < T  # [N*L, 4H]
        # Per-token: sum over channels
        tok_bump = bump_mask.float().sum(dim=1)  # [N*L]
        tok_total = torch.full_like(tok_bump, z.shape[1])
        for i, tid in enumerate(flat_tokens.tolist()):
            all_token_bump[tid][0] += tok_bump[i].item()
            all_token_bump[tid][1] += tok_total[i].item()
        # Per-channel: sum over tokens
        ch_bump = bump_mask.float().sum(dim=0)  # [4H]
        ch_total = torch.full_like(ch_bump, z.shape[0])
        for c in range(z.shape[1]):
            all_channel_bump[c][0] += ch_bump[c].item()
            all_channel_bump[c][1] += ch_total[c].item()

    # Token anatomy
    token_fracs = {tid: v[0] / (v[1] + 1e-10) for tid, v in all_token_bump.items()}
    top_tokens = sorted(token_fracs.items(), key=lambda x: -x[1])[:top_k]
    top_tokens_decoded = [(tokenizer.decode([tid]), frac) for tid, frac in top_tokens]

    # Channel anatomy
    channel_fracs = {c: v[0] / (v[1] + 1e-10) for c, v in all_channel_bump.items()}
    top_channels = sorted(channel_fracs.items(), key=lambda x: -x[1])[:top_k]

    return top_tokens_decoded, top_channels, channel_fracs


# ── Masking experiment ──────────────────────────────────────────────────────
def make_masked_model(model, block_idx, T):
    """Return a copy of model with GELU replaced by masked GELU for one block."""
    import copy
    m = copy.deepcopy(model)

    class MaskedGELU(torch.nn.Module):
        def forward(self, z):
            gelu_z = F.gelu(z)
            relu_z = F.relu(z)
            mask = z.abs() < T
            return torch.where(mask, relu_z, gelu_z)

    m.transformer.h[block_idx].mlp.act = MaskedGELU()
    return m


def make_all_masked_model(model, T):
    import copy
    m = copy.deepcopy(model)

    class MaskedGELU(torch.nn.Module):
        def forward(self, z):
            gelu_z = F.gelu(z)
            relu_z = F.relu(z)
            mask = z.abs() < T
            return torch.where(mask, relu_z, gelu_z)

    for b in range(N_BLOCKS):
        m.transformer.h[b].mlp.act = MaskedGELU()
    return m


def masking_experiment(model, seqs, T, seeds, n_blocks, device):
    """
    For each block and all-blocks: compute PPL delta and top-1 change fraction.
    Returns: {block: {ppl_delta_mean, ppl_delta_std, pred_change_mean, pred_change_std}}
    """
    # Get baseline predictions (seed-independent, use seed=0 model)
    base_ppl, base_preds = compute_ppl_and_top1(model, seqs, device)
    print(f"  Masking baseline PPL: {base_ppl:.4f}")

    block_results = {}
    for b in range(n_blocks):
        print(f"  Block {b:2d}...", end=" ", flush=True)
        ppls, pred_changes = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            m = make_masked_model(model, b, T)
            m.to(device)
            ppl, preds = compute_ppl_and_top1(m, seqs, device)
            pred_change = (preds != base_preds).float().mean().item()
            ppls.append(ppl)
            pred_changes.append(pred_change)
            del m
        ppl_deltas = [p - base_ppl for p in ppls]
        print(f"PPL delta: {sum(ppl_deltas)/len(ppl_deltas):+.4f}, pred_change: {sum(pred_changes)/len(pred_changes):.4f}")
        block_results[b] = dict(
            ppl_delta_mean=sum(ppl_deltas) / len(ppl_deltas),
            ppl_delta_std=(sum((x - sum(ppl_deltas)/len(ppl_deltas))**2 for x in ppl_deltas) / len(ppl_deltas)) ** 0.5,
            pred_change_mean=sum(pred_changes) / len(pred_changes),
            pred_change_std=(sum((x - sum(pred_changes)/len(pred_changes))**2 for x in pred_changes) / len(pred_changes)) ** 0.5,
        )

    # All blocks masked
    print("  All blocks...", end=" ", flush=True)
    ppls_all, pred_changes_all = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        m = make_all_masked_model(model, T)
        m.to(device)
        ppl, preds = compute_ppl_and_top1(m, seqs, device)
        pred_change = (preds != base_preds).float().mean().item()
        ppls_all.append(ppl)
        pred_changes_all.append(pred_change)
        del m
    ppl_deltas_all = [p - base_ppl for p in ppls_all]
    print(f"PPL delta: {sum(ppl_deltas_all)/len(ppl_deltas_all):+.4f}, pred_change: {sum(pred_changes_all)/len(pred_changes_all):.4f}")
    block_results["all"] = dict(
        ppl_delta_mean=sum(ppl_deltas_all) / len(ppl_deltas_all),
        ppl_delta_std=(sum((x - sum(ppl_deltas_all)/len(ppl_deltas_all))**2 for x in ppl_deltas_all) / len(ppl_deltas_all)) ** 0.5,
        pred_change_mean=sum(pred_changes_all) / len(pred_changes_all),
        pred_change_std=(sum((x - sum(pred_changes_all)/len(pred_changes_all))**2 for x in pred_changes_all) / len(pred_changes_all)) ** 0.5,
    )
    return base_ppl, block_results


# ── Plotting ─────────────────────────────────────────────────────────────────
def make_plots(threshold_results, masking_results, base_ppl, thresholds, n_blocks, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Cycle 103: GELU-ReLU Disagreement Anatomy — GPT-2 FFN\n(baseline PPL={base_ppl:.2f})", fontsize=13)

    # Panel 1: Threshold sweep — mean energy fraction across blocks
    ax = axes[0, 0]
    for b in range(n_blocks):
        energies = [threshold_results[T][b]["energy_frac"] for T in thresholds]
        ax.plot(thresholds, energies, alpha=0.4, color="steelblue", linewidth=0.8)
    mean_energies = [sum(threshold_results[T][b]["energy_frac"] for b in range(n_blocks)) / n_blocks for T in thresholds]
    ax.plot(thresholds, mean_energies, "k-", linewidth=2, label="mean")
    ax.axhline(0.05, color="green", linestyle="--", label="5% threshold (confirmed)")
    ax.axhline(0.20, color="red", linestyle="--", label="20% threshold (refuted)")
    ax.set_xlabel("Threshold T")
    ax.set_ylabel("Disagreement energy fraction")
    ax.set_title("Threshold sweep — disagreement energy")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # Panel 2: Zone fraction heatmap at T=2.5
    ax = axes[0, 1]
    T = PRIMARY_T
    zone_fracs = [[threshold_results[T][b]["zone_frac"]] for b in range(n_blocks)]
    neg_fracs = [threshold_results[T][b]["neg_frac"] for b in range(n_blocks)]
    pos_fracs = [threshold_results[T][b]["pos_frac"] for b in range(n_blocks)]
    x = list(range(n_blocks))
    bars = ax.bar(x, [threshold_results[T][b]["zone_frac"] for b in x], color="steelblue", label="bump zone")
    ax.set_xlabel("Block")
    ax.set_ylabel("Fraction of activations in bump zone")
    ax.set_title(f"Zone fraction per block (T={T})")
    ax.set_xticks(x)

    # Panel 3: Neg/pos energy split at T=2.5
    ax = axes[1, 0]
    neg_e = [threshold_results[PRIMARY_T][b]["neg_energy_frac"] for b in range(n_blocks)]
    pos_e = [threshold_results[PRIMARY_T][b]["pos_energy_frac"] for b in range(n_blocks)]
    ax.bar(x, neg_e, label="neg (z<0, leak)", color="tomato")
    ax.bar(x, pos_e, bottom=neg_e, label="pos (z>0, attenuation)", color="steelblue")
    ax.set_xlabel("Block")
    ax.set_ylabel("Fraction of bump-zone energy")
    ax.set_title(f"Signed-side decomposition (T={T})")
    ax.legend(fontsize=9)
    ax.set_xticks(x)

    # Panel 4: Prediction change fraction per block
    ax = axes[1, 1]
    pred_changes = [masking_results[b]["pred_change_mean"] for b in range(n_blocks)]
    pred_stds = [masking_results[b]["pred_change_std"] for b in range(n_blocks)]
    ax.bar(x, pred_changes, color="orange", label="pred change fraction")
    ax.errorbar(x, pred_changes, yerr=pred_stds, fmt="none", color="black", capsize=3)
    ax.axhline(0.01, color="green", linestyle="--", label="1% (confirmed)")
    ax.axhline(0.10, color="red", linestyle="--", label="10% (refuted)")
    # All-blocks
    all_change = masking_results["all"]["pred_change_mean"]
    ax.axhline(all_change, color="purple", linestyle=":", label=f"all blocks: {all_change:.3f}")
    ax.set_xlabel("Block")
    ax.set_ylabel("Top-1 prediction change fraction")
    ax.set_title(f"Masking impact per block (T={T})")
    ax.legend(fontsize=8)
    ax.set_xticks(x)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Cycle 103: GELU-ReLU Disagreement Anatomy")
    print("=" * 60)

    # Load model and tokenizer
    print("\n[1] Loading GPT-2...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(DEVICE)
    model.eval()

    # Load data
    print("\n[2] Loading OWT tokens...")
    seqs = load_owt_tokens(tokenizer, N_TOKENS, SEQ_LEN)
    print(f"  Loaded {len(seqs)} sequences × {SEQ_LEN} tokens = {len(seqs)*SEQ_LEN} total tokens")
    seqs_dev = seqs  # keep on CPU, push per-batch

    # Baseline PPL
    print("\n[3] Baseline PPL...")
    base_ppl = compute_ppl(model, seqs, DEVICE)
    print(f"  Baseline PPL: {base_ppl:.4f}")
    if not (ABORT_PPL_LO <= base_ppl <= ABORT_PPL_HI):
        raise RuntimeError(f"ABORT: baseline PPL {base_ppl:.2f} outside [{ABORT_PPL_LO}, {ABORT_PPL_HI}]")

    # Collect pre-activations
    print("\n[4] Collecting FFN pre-activations...")
    z_dict = collect_all_preactivations(model, seqs, DEVICE)
    total_pairs = sum(z_dict[b].numel() for b in range(N_BLOCKS))
    print(f"  Collected {total_pairs:,} pre-activation values across {N_BLOCKS} blocks")
    print(f"  Shape per block: {z_dict[0].shape}")

    # Abort check at T=1.0
    bump_frac_t1 = (z_dict[0].abs() < 1.0).float().mean().item()
    print(f"  Bump fraction at T=1.0 (block 0): {bump_frac_t1:.4f}")
    if bump_frac_t1 < 0.001:
        raise RuntimeError(f"ABORT: bump fraction {bump_frac_t1:.4f} < 0.001 (threshold too tight)")
    if bump_frac_t1 > 0.80:
        raise RuntimeError(f"ABORT: bump fraction {bump_frac_t1:.4f} > 0.80 (threshold too loose)")

    # Threshold analysis
    print("\n[5] Threshold sweep analysis...")
    threshold_results = analyze_threshold(z_dict, THRESHOLDS, N_BLOCKS)
    for T in THRESHOLDS:
        mean_ef = sum(threshold_results[T][b]["energy_frac"] for b in range(N_BLOCKS)) / N_BLOCKS
        mean_zf = sum(threshold_results[T][b]["zone_frac"] for b in range(N_BLOCKS)) / N_BLOCKS
        print(f"  T={T:.1f}: mean zone_frac={mean_zf:.4f}, mean energy_frac={mean_ef:.6f}")

    # Token and channel anatomy
    print("\n[6] Token/channel anatomy at T=2.5...")
    top_tokens, top_channels, channel_fracs = token_channel_anatomy(
        z_dict, seqs, PRIMARY_T, N_BLOCKS, tokenizer
    )
    print("  Top-10 tokens by bump fraction:")
    for tok, frac in top_tokens[:10]:
        print(f"    '{tok}': {frac:.4f}")
    print("  Top-10 channels by bump fraction:")
    for ch, frac in top_channels[:10]:
        print(f"    ch {ch}: {frac:.4f}")

    # Masking experiment
    print(f"\n[7] Masking experiment (T={PRIMARY_T}, seeds={SEEDS})...")
    print("  This will run 12 blocks × 3 seeds + 3 all-block evals = 39 evals...")
    base_ppl_mask, masking_results = masking_experiment(model, seqs, PRIMARY_T, SEEDS, N_BLOCKS, DEVICE)

    # Verdict
    print("\n[8] Verdict computation...")
    max_energy = max(threshold_results[PRIMARY_T][b]["energy_frac"] for b in range(N_BLOCKS))
    max_pred_change = max(masking_results[b]["pred_change_mean"] for b in range(N_BLOCKS))
    all_ppl_delta = masking_results["all"]["ppl_delta_mean"]
    print(f"  Max energy fraction (T={PRIMARY_T}): {max_energy:.6f}")
    print(f"  Max pred change: {max_pred_change:.6f}")
    print(f"  All-blocks PPL delta: {all_ppl_delta:+.4f}")

    if max_energy < 0.05 and max_pred_change < 0.01:
        verdict = "CONFIRMED"
        verdict_reason = f"bump_energy={max_energy:.4f}<0.05, pred_change={max_pred_change:.4f}<0.01"
    elif max_energy > 0.20 or max_pred_change > 0.10:
        verdict = "REFUTED"
        verdict_reason = f"bump_energy={max_energy:.4f}>0.20 OR pred_change={max_pred_change:.4f}>0.10"
    else:
        verdict = "INCONCLUSIVE"
        verdict_reason = f"bump_energy={max_energy:.4f} (5-20%), pred_change={max_pred_change:.4f}"
    print(f"  Verdict: {verdict} — {verdict_reason}")

    # Serialize results
    print("\n[9] Saving results...")

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, float):
            return round(obj, 8)
        else:
            return obj

    results = {
        "cycle": 103,
        "item": 108,
        "baseline_ppl": base_ppl,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "max_energy_frac": max_energy,
        "max_pred_change": max_pred_change,
        "all_blocks_ppl_delta": all_ppl_delta,
        "threshold_sweep": make_serializable(threshold_results),
        "masking_results": make_serializable(masking_results),
        "top_tokens": [(tok, round(frac, 6)) for tok, frac in top_tokens],
        "top_channels": [(ch, round(frac, 6)) for ch, frac in top_channels],
    }

    json_path = f"{RESULTS_DIR}/cycle_103_gelu_relu_anatomy.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # Plot
    plot_path = f"{RESULTS_DIR}/cycle_103_gelu_relu_anatomy.png"
    make_plots(threshold_results, masking_results, base_ppl, THRESHOLDS, N_BLOCKS, plot_path)

    print(f"\n{'='*60}")
    print(f"DONE. Verdict: {verdict}")
    print(f"  Baseline PPL: {base_ppl:.4f}")
    print(f"  Max energy frac (T=2.5): {max_energy:.6f}")
    print(f"  Max pred change: {max_pred_change:.6f}")
    print(f"  All-blocks PPL delta: {all_ppl_delta:+.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
