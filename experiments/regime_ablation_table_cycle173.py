"""
Cycle 173 / Item 184: Unified per-block regime ablation table — GPT-2 FFN

Reconciles contradictory claims from cycles 102, 109, 164.
For each of 12 blocks independently (all others float), measure PPL under
6 conditions:
  C0: float baseline
  C1: kill negative (set gelu output to 0 where z<0, keeping gelu(z) where z>=0)
  C2: kill positive (set gelu output to 0 where z>=0, keeping gelu(z) where z<0)
  C3: replace negative with conditional mean E[gelu(z)|z<0] per channel
  C4: replace positive with conditional mean E[gelu(z)|z>=0] per channel
  C5: replace all with unconditional mean E[gelu(z)] per channel (constant output)

Confirmed if: C3 within +2% PPL of C0 in >=8/12 blocks.
Refuted if:   C3 >+10% PPL in >=8/12 blocks.
"""

import json
import math
import os
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
N_SEQS = 200
SEQ_LEN = 1024
BATCH_SIZE = 4
SEED = 42
OWT_PATH = "data/openwebtext_sample.txt"
OUT_JSON = "autoresearch/results/cycle_173_regime_ablation.json"
OUT_PNG = "autoresearch/results/cycle_173_regime_ablation.png"
BASELINE_PPL_LO, BASELINE_PPL_HI = 15.0, 65.0
N_BLOCKS = 12

print(f"Device: {DEVICE}", flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_owt_tokens(tokenizer, n_seqs=N_SEQS, seq_len=SEQ_LEN, seed=SEED):
    with open(OWT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    rng = torch.Generator()
    rng.manual_seed(seed)
    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    max_start = len(tokens) - seq_len - 1
    indices = torch.randperm(max_start, generator=rng)[:n_seqs]
    seqs = torch.stack([tokens[i:i + seq_len + 1] for i in sorted(indices.tolist())])
    return seqs  # (n_seqs, seq_len+1)


def eval_ppl(model, seqs, batch_size=BATCH_SIZE):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size].to(DEVICE)
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            out = model(inp)
            logits = out.logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += tgt.numel()
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# Collect per-block GELU stats (conditional means)
# ---------------------------------------------------------------------------
def collect_gelu_stats(model, seqs):
    """
    For each block, collect:
      mu_neg[c] = E[gelu(z_c) | z_c < 0]   (conditional mean, negative regime)
      mu_pos[c] = E[gelu(z_c) | z_c >= 0]   (conditional mean, positive regime)
      mu_all[c] = E[gelu(z_c)]              (unconditional mean)
      norms: ||y_pos||, ||y_neg||, etc.
    """
    N_CHANNELS = 3072  # GPT-2 intermediate size

    # Accumulators
    sum_neg = {b: torch.zeros(N_CHANNELS, dtype=torch.float64) for b in range(N_BLOCKS)}
    sum_pos = {b: torch.zeros(N_CHANNELS, dtype=torch.float64) for b in range(N_BLOCKS)}
    sum_all = {b: torch.zeros(N_CHANNELS, dtype=torch.float64) for b in range(N_BLOCKS)}
    cnt_neg = {b: torch.zeros(N_CHANNELS, dtype=torch.float64) for b in range(N_BLOCKS)}
    cnt_pos = {b: torch.zeros(N_CHANNELS, dtype=torch.float64) for b in range(N_BLOCKS)}
    cnt_all = {b: torch.zeros(N_CHANNELS, dtype=torch.float64) for b in range(N_BLOCKS)}
    # For norms: accumulate sum of squared norms
    sqnorm_ypos = {b: 0.0 for b in range(N_BLOCKS)}
    sqnorm_yneg = {b: 0.0 for b in range(N_BLOCKS)}
    n_tokens_total = {b: 0 for b in range(N_BLOCKS)}

    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            z = inp[0]  # pre-activation (batch, seq, 3072)
            y = out      # gelu(z)
            z_flat = z.reshape(-1, N_CHANNELS)  # (B*S, 3072)
            y_flat = y.reshape(-1, N_CHANNELS).float()

            mask_neg = (z_flat < 0)
            mask_pos = ~mask_neg

            y_neg = torch.where(mask_neg, y_flat, torch.zeros_like(y_flat))
            y_pos = torch.where(mask_pos, y_flat, torch.zeros_like(y_flat))

            sum_neg[idx] += y_neg.sum(dim=0).cpu().double()
            sum_pos[idx] += y_pos.sum(dim=0).cpu().double()
            sum_all[idx] += y_flat.sum(dim=0).cpu().double()
            cnt_neg[idx] += mask_neg.sum(dim=0).cpu().double()
            cnt_pos[idx] += mask_pos.sum(dim=0).cpu().double()
            cnt_all[idx] += torch.tensor(y_flat.shape[0], dtype=torch.float64)

            # Per-token norms: ||y_pos(x)||^2 and ||y_neg(x)||^2 summed over tokens
            sqnorm_ypos[idx] += y_pos.pow(2).sum().item()
            sqnorm_yneg[idx] += y_neg.pow(2).sum().item()
            n_tokens_total[idx] += y_flat.shape[0]

        return hook_fn

    for b in range(N_BLOCKS):
        h = model.transformer.h[b].mlp.act.register_forward_hook(make_hook(b))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH_SIZE):
            batch = seqs[i:i + BATCH_SIZE, :-1].to(DEVICE)
            model(batch)

    for h in hooks:
        h.remove()

    # Compute means and norms
    stats = {}
    for b in range(N_BLOCKS):
        mu_neg = (sum_neg[b] / cnt_neg[b].clamp(min=1)).float()
        mu_pos = (sum_pos[b] / cnt_pos[b].clamp(min=1)).float()
        mu_all = (sum_all[b] / cnt_all[b].clamp(min=1)).float()

        # s_neg = y_neg - mu_neg (token-dependent part) — we need variance for norm
        # ||s_neg|| = sqrt(E[||y_neg - mu_neg||^2]) but we don't have per-token data
        # Instead report RMS norms
        n = n_tokens_total[b]
        rms_ypos = math.sqrt(sqnorm_ypos[b] / n) if n > 0 else 0.0
        rms_yneg = math.sqrt(sqnorm_yneg[b] / n) if n > 0 else 0.0

        stats[b] = {
            "mu_neg": mu_neg,       # (3072,) tensor
            "mu_pos": mu_pos,       # (3072,) tensor
            "mu_all": mu_all,       # (3072,) tensor
            "norm_mu_neg": mu_neg.norm().item(),
            "norm_mu_pos": mu_pos.norm().item(),
            "norm_mu_all": mu_all.norm().item(),
            "rms_ypos": rms_ypos,
            "rms_yneg": rms_yneg,
            "n_tokens": n,
        }
        print(f"  Block {b}: ||mu_neg||={stats[b]['norm_mu_neg']:.4f}, "
              f"||mu_pos||={stats[b]['norm_mu_pos']:.4f}, "
              f"rms_ypos={rms_ypos:.4f}, rms_yneg={rms_yneg:.4f}", flush=True)

    return stats


# ---------------------------------------------------------------------------
# Condition wrappers — modify GELU activation in-place via hook
# ---------------------------------------------------------------------------
class ConditionHook:
    """Registers a forward hook on model.transformer.h[block].mlp.act that
    modifies the GELU output according to the specified condition."""

    def __init__(self, model, block_idx, condition, stats):
        self.block_idx = block_idx
        self.condition = condition
        self.stats = stats
        self.handle = None
        self._install(model)

    def _install(self, model):
        mu_neg = self.stats[self.block_idx]["mu_neg"].to(DEVICE)
        mu_pos = self.stats[self.block_idx]["mu_pos"].to(DEVICE)
        mu_all = self.stats[self.block_idx]["mu_all"].to(DEVICE)
        cond = self.condition

        def hook_fn(module, inp, out):
            z = inp[0]  # pre-activation
            y = out      # gelu(z)
            z_flat = z  # keep original shape
            mask_neg = (z_flat < 0)
            mask_pos = ~mask_neg

            if cond == "C1":  # kill negative → 0 where z<0
                return torch.where(mask_neg, torch.zeros_like(y), y)
            elif cond == "C2":  # kill positive → 0 where z>=0
                return torch.where(mask_pos, torch.zeros_like(y), y)
            elif cond == "C3":  # replace negative with conditional mean
                return torch.where(mask_neg, mu_neg.expand_as(y), y)
            elif cond == "C4":  # replace positive with conditional mean
                return torch.where(mask_pos, mu_pos.expand_as(y), y)
            elif cond == "C5":  # replace all with unconditional mean
                return mu_all.expand_as(y)
            else:
                return y  # C0 = no modification

        self.handle = model.transformer.h[self.block_idx].mlp.act.register_forward_hook(hook_fn)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading model and tokenizer...", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    print(f"Activation: {type(model.transformer.h[0].mlp.act).__name__}", flush=True)

    # Load data
    seqs = load_owt_tokens(tokenizer)

    # -------------------------------------------------------------------
    # Baseline PPL
    # -------------------------------------------------------------------
    print("\n=== Baseline PPL ===", flush=True)
    ppl_base = eval_ppl(model, seqs)
    print(f"  Baseline PPL: {ppl_base:.3f}", flush=True)

    if not (BASELINE_PPL_LO <= ppl_base <= BASELINE_PPL_HI):
        print(f"ABORT: Baseline PPL {ppl_base:.3f} outside [{BASELINE_PPL_LO}, {BASELINE_PPL_HI}]")
        return

    # -------------------------------------------------------------------
    # Collect GELU stats
    # -------------------------------------------------------------------
    print("\n=== Collecting GELU stats ===", flush=True)
    stats = collect_gelu_stats(model, seqs)

    # -------------------------------------------------------------------
    # Run 12 blocks x 5 conditions (C1-C5), plus C0 is the baseline
    # -------------------------------------------------------------------
    conditions = ["C1", "C2", "C3", "C4", "C5"]
    results_table = {}  # results_table[block][condition] = ppl

    for b in range(N_BLOCKS):
        results_table[b] = {"C0": ppl_base}
        for cond in conditions:
            print(f"\n  Block {b}, {cond}...", end=" ", flush=True)
            hook = ConditionHook(model, b, cond, stats)
            ppl = eval_ppl(model, seqs)
            hook.remove()
            results_table[b][cond] = ppl
            delta_pct = 100.0 * (ppl - ppl_base) / ppl_base
            print(f"PPL={ppl:.2f} (delta={delta_pct:+.1f}%)", flush=True)

    # -------------------------------------------------------------------
    # Compute norms table and interaction check
    # -------------------------------------------------------------------
    norms_table = {}
    for b in range(N_BLOCKS):
        norms_table[b] = {
            "norm_mu_neg": stats[b]["norm_mu_neg"],
            "norm_mu_pos": stats[b]["norm_mu_pos"],
            "norm_mu_all": stats[b]["norm_mu_all"],
            "rms_ypos": stats[b]["rms_ypos"],
            "rms_yneg": stats[b]["rms_yneg"],
        }

    interaction_check = {}
    for b in range(N_BLOCKS):
        additive = results_table[b]["C1"] + results_table[b]["C2"] - ppl_base
        actual_c5 = results_table[b]["C5"]
        interaction_check[b] = {
            "additive_C1_C2": additive,
            "actual_C5": actual_c5,
            "interaction_gap": actual_c5 - additive,
        }

    # -------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------
    c3_within_2pct = 0
    c3_above_10pct = 0
    for b in range(N_BLOCKS):
        delta = 100.0 * (results_table[b]["C3"] - ppl_base) / ppl_base
        if delta <= 2.0:
            c3_within_2pct += 1
        if delta > 10.0:
            c3_above_10pct += 1

    if c3_within_2pct >= 8:
        verdict = "CONFIRMED"
    elif c3_above_10pct >= 8:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n=== VERDICT: {verdict} ===")
    print(f"  C3 within +2%: {c3_within_2pct}/12 blocks")
    print(f"  C3 above +10%: {c3_above_10pct}/12 blocks")

    # -------------------------------------------------------------------
    # Print table
    # -------------------------------------------------------------------
    print("\n=== PPL Table (delta-% from baseline) ===")
    header = f"{'Block':>6}" + "".join(f"{c:>10}" for c in ["C0", "C1", "C2", "C3", "C4", "C5"])
    print(header)
    for b in range(N_BLOCKS):
        row = f"{b:>6}"
        for c in ["C0", "C1", "C2", "C3", "C4", "C5"]:
            ppl = results_table[b][c]
            if c == "C0":
                row += f"{'---':>10}"
            else:
                delta = 100.0 * (ppl - ppl_base) / ppl_base
                row += f"{delta:>+9.1f}%"
        print(row)

    # -------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------
    out = {
        "cycle": 173,
        "item": 184,
        "device": DEVICE,
        "n_seqs": N_SEQS,
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "ppl_baseline": ppl_base,
        "verdict": verdict,
        "c3_within_2pct": c3_within_2pct,
        "c3_above_10pct": c3_above_10pct,
        "ppl_table": {str(b): {c: results_table[b][c] for c in ["C0", "C1", "C2", "C3", "C4", "C5"]}
                      for b in range(N_BLOCKS)},
        "delta_pct_table": {str(b): {c: 100.0 * (results_table[b][c] - ppl_base) / ppl_base
                                      for c in ["C1", "C2", "C3", "C4", "C5"]}
                            for b in range(N_BLOCKS)},
        "norms": {str(b): norms_table[b] for b in range(N_BLOCKS)},
        "interaction_check": {str(b): interaction_check[b] for b in range(N_BLOCKS)},
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_JSON}", flush=True)

    # -------------------------------------------------------------------
    # Plot: heatmap of delta-PPL%
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    conds = ["C1", "C2", "C3", "C4", "C5"]
    data = np.zeros((N_BLOCKS, len(conds)))
    for b in range(N_BLOCKS):
        for j, c in enumerate(conds):
            data[b, j] = 100.0 * (results_table[b][c] - ppl_base) / ppl_base

    # Use log scale for better visualization (some values are huge)
    data_clipped = np.clip(data, -100, 10000)
    # Use symlog-like: sign(x) * log10(1 + |x|) for display
    data_display = np.sign(data_clipped) * np.log10(1 + np.abs(data_clipped))

    im = ax.imshow(data_display, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds)
    ax.set_yticks(range(N_BLOCKS))
    ax.set_yticklabels([f"Block {b}" for b in range(N_BLOCKS)])
    ax.set_xlabel("Condition")
    ax.set_ylabel("Block")
    ax.set_title(f"Cycle 173: Per-block regime ablation (symlog delta-PPL%)\n"
                 f"Verdict: {verdict} | Baseline PPL: {ppl_base:.1f}")

    # Annotate cells with actual delta-PPL%
    for b in range(N_BLOCKS):
        for j, c in enumerate(conds):
            val = data[b, j]
            if abs(val) > 100:
                txt = f"{val:.0f}%"
            else:
                txt = f"{val:+.1f}%"
            ax.text(j, b, txt, ha="center", va="center", fontsize=7,
                    color="white" if abs(data_display[b, j]) > 1.5 else "black")

    plt.colorbar(im, ax=ax, label="symlog(delta-PPL%)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved: {OUT_PNG}", flush=True)
    plt.close()


if __name__ == "__main__":
    main()
