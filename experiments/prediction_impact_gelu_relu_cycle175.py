"""
Cycle 175 / Item 186: Per-token prediction impact of GELU→ReLU — GPT-2 FFN

For each of 12 blocks independently, replace GELU with ReLU and measure:
  (a) P_float = softmax(logits_float)
  (b) P_relu = softmax(logits_modified)
  (c) per-token KL(P_float || P_relu)
  (d) top-50 tokens by KL per block — actual tokens and contexts
  (e) fraction where P_relu(correct) > P_float(correct)
  (f) correlate mean KL with ||y_neg|| and ||s_neg||

Confirmed if: mean KL > 0.1 nats in ≥8/12 blocks AND top-50 tokens show semantic pattern.
Refuted if: mean KL < 0.01 nats in ≥8/12 blocks — negligible per-block prediction impact.
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
N_SEQS = 16
SEQ_LEN = 128
N_TOKENS = N_SEQS * SEQ_LEN  # 2048
BATCH_SIZE = 4
SEED = 42
OWT_PATH = "data/openwebtext_sample.txt"
OUT_JSON = "autoresearch/results/cycle_175_prediction_impact.json"
OUT_PNG = "autoresearch/results/cycle_175_prediction_impact.png"
BASELINE_PPL_LO, BASELINE_PPL_HI = 15.0, 65.0
N_BLOCKS = 12
TOP_K = 50

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


@torch.no_grad()
def eval_ppl(model, seqs, batch_size=BATCH_SIZE):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size].to(DEVICE)
        inp, tgt = batch[:, :-1], batch[:, 1:]
        logits = model(inp).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += tgt.numel()
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# Hook: replace GELU with ReLU in one block's MLP
# ---------------------------------------------------------------------------
class GeluToReluHook:
    """Hook c_fc to capture pre-activation, then hook the MLP forward to replace GELU→ReLU."""

    def __init__(self, block):
        self.block = block
        self.handles = []
        self._c_fc_out = None

    def _hook_c_fc(self, module, input, output):
        """Capture c_fc output (pre-activation z)."""
        self._c_fc_out = output

    def _hook_c_proj(self, module, input, output):
        """Replace c_proj input: instead of gelu(z), use relu(z)."""
        z = self._c_fc_out
        relu_z = F.relu(z)
        # c_proj is Conv1D: output = input @ weight + bias
        new_output = relu_z @ module.weight + module.bias
        return new_output

    def install(self):
        h1 = self.block.mlp.c_fc.register_forward_hook(self._hook_c_fc)
        h2 = self.block.mlp.c_proj.register_forward_hook(self._hook_c_proj)
        self.handles = [h1, h2]

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------------------------------------------------------
# Collect signal norms (y_neg, s_neg) for correlation with KL
# ---------------------------------------------------------------------------
class SignalNormCollector:
    """Collect ||y_neg|| and ||s_neg|| per block, following cycle-174 methodology."""

    def __init__(self, block):
        self.block = block
        self.handle = None
        self._gelu_out = None

    def _hook(self, module, input, output):
        self._gelu_out = input[0].detach()  # pre c_proj = post-GELU

    def install(self):
        self.handle = self.block.mlp.c_proj.register_forward_hook(self._hook)

    def remove(self):
        if self.handle:
            self.handle.remove()

    def compute_norms(self):
        """Return mean ||y_neg||, mean ||s_neg|| across tokens."""
        h = self._gelu_out  # [batch, seq_len, 3072]
        W_out = self.block.mlp.c_proj.weight.detach()  # Conv1D: [3072, 768]

        h_flat = h.reshape(-1, h.shape[-1])  # [N_TOKENS, 3072]

        neg_mask = (h_flat < 0).float()
        y_neg = (h_flat * neg_mask) @ W_out  # [N_TOKENS, 768]
        mu_neg = y_neg.mean(dim=0)
        s_neg = y_neg - mu_neg

        y_neg_norm = y_neg.norm(dim=1).mean().item()
        s_neg_norm = s_neg.norm(dim=1).mean().item()
        return y_neg_norm, s_neg_norm


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()

    # Load data
    seqs = load_owt_tokens(tokenizer)
    print(f"Loaded {len(seqs)} sequences of length {SEQ_LEN+1}", flush=True)

    # Baseline PPL sanity
    eval_seqs = load_owt_tokens(tokenizer, n_seqs=200, seq_len=1024, seed=99)
    ppl_base = eval_ppl(model, eval_seqs)
    print(f"Baseline PPL: {ppl_base:.1f}", flush=True)
    assert BASELINE_PPL_LO <= ppl_base <= BASELINE_PPL_HI, \
        f"Baseline PPL {ppl_base:.1f} outside [{BASELINE_PPL_LO}, {BASELINE_PPL_HI}]"

    # Get float logits (P_float) for all tokens
    print("Computing float logits...", flush=True)
    all_log_p_float = []
    all_targets = []
    input_ids_list = []
    for i in range(0, len(seqs), BATCH_SIZE):
        batch = seqs[i:i + BATCH_SIZE].to(DEVICE)
        inp, tgt = batch[:, :-1], batch[:, 1:]
        with torch.no_grad():
            logits = model(inp).logits  # [B, SEQ_LEN, vocab]
        log_p = F.log_softmax(logits, dim=-1)  # [B, SEQ_LEN, vocab]
        all_log_p_float.append(log_p.cpu())
        all_targets.append(tgt.cpu())
        input_ids_list.append(inp.cpu())

    log_p_float = torch.cat(all_log_p_float, dim=0)  # [N_SEQS, SEQ_LEN, vocab]
    targets = torch.cat(all_targets, dim=0)           # [N_SEQS, SEQ_LEN]
    input_ids_all = torch.cat(input_ids_list, dim=0)  # [N_SEQS, SEQ_LEN]

    log_p_float_flat = log_p_float.reshape(-1, log_p_float.shape[-1])  # [N_TOKENS, vocab]
    targets_flat = targets.reshape(-1)                                  # [N_TOKENS]
    p_float_flat = log_p_float_flat.exp()  # [N_TOKENS, vocab]

    # P_float(correct) per token
    p_float_correct = p_float_flat[torch.arange(N_TOKENS), targets_flat]  # [N_TOKENS]

    results = {"baseline_ppl": ppl_base, "n_tokens": N_TOKENS, "blocks": []}
    all_mean_kl = []
    all_y_neg_norm = []
    all_s_neg_norm = []

    for b in range(N_BLOCKS):
        print(f"\n--- Block {b} ---", flush=True)

        # Collect signal norms first (unmodified model)
        snc = SignalNormCollector(model.transformer.h[b])
        snc.install()
        with torch.no_grad():
            for i in range(0, len(seqs), BATCH_SIZE):
                batch = seqs[i:i + BATCH_SIZE].to(DEVICE)
                _ = model(batch[:, :-1])
        y_neg_norm, s_neg_norm = snc.compute_norms()
        snc.remove()
        print(f"  ||y_neg|| = {y_neg_norm:.4f}, ||s_neg|| = {s_neg_norm:.4f}", flush=True)

        # Install GELU→ReLU hook
        hook = GeluToReluHook(model.transformer.h[b])
        hook.install()

        # Get modified logits
        all_log_p_relu = []
        for i in range(0, len(seqs), BATCH_SIZE):
            batch = seqs[i:i + BATCH_SIZE].to(DEVICE)
            inp = batch[:, :-1]
            with torch.no_grad():
                logits_mod = model(inp).logits
            log_p = F.log_softmax(logits_mod, dim=-1)
            all_log_p_relu.append(log_p.cpu())

        hook.remove()

        log_p_relu = torch.cat(all_log_p_relu, dim=0)  # [N_SEQS, SEQ_LEN, vocab]
        log_p_relu_flat = log_p_relu.reshape(-1, log_p_relu.shape[-1])  # [N_TOKENS, vocab]
        p_relu_flat = log_p_relu_flat.exp()

        # Sanity: verify P_relu ≠ P_float
        diff_check = (log_p_float_flat - log_p_relu_flat).abs().max().item()
        print(f"  max |log_p_float - log_p_relu| = {diff_check:.6f}", flush=True)
        assert diff_check > 1e-6, f"Block {b}: hook didn't change logits!"

        # KL(P_float || P_relu) per token
        # = sum_v P_float(v) * (log P_float(v) - log P_relu(v))
        kl_per_token = (p_float_flat * (log_p_float_flat - log_p_relu_flat)).sum(dim=-1)  # [N_TOKENS]

        mean_kl = kl_per_token.mean().item()
        median_kl = kl_per_token.median().item()
        max_kl = kl_per_token.max().item()
        print(f"  KL: mean={mean_kl:.6f}, median={median_kl:.6f}, max={max_kl:.4f}", flush=True)

        # P_relu(correct) per token
        p_relu_correct = p_relu_flat[torch.arange(N_TOKENS), targets_flat]
        frac_relu_better = (p_relu_correct > p_float_correct).float().mean().item()
        print(f"  Frac P_relu(correct) > P_float(correct): {frac_relu_better:.4f}", flush=True)

        # Top-50 tokens by KL
        top_indices = kl_per_token.topk(TOP_K).indices.tolist()
        top_tokens_info = []
        for idx in top_indices:
            seq_idx = idx // SEQ_LEN
            pos_idx = idx % SEQ_LEN
            # Context: 5 tokens before the target token
            ctx_start = max(0, pos_idx - 4)
            ctx_ids = input_ids_all[seq_idx, ctx_start:pos_idx + 1].tolist()
            target_id = targets_flat[idx].item()
            ctx_text = tokenizer.decode(ctx_ids)
            target_text = tokenizer.decode([target_id])
            top_tokens_info.append({
                "kl": kl_per_token[idx].item(),
                "context": ctx_text,
                "target": target_text,
                "p_float": p_float_correct[idx].item(),
                "p_relu": p_relu_correct[idx].item(),
            })

        # Print top-10 for console inspection
        print(f"  Top-10 by KL:", flush=True)
        for t in top_tokens_info[:10]:
            print(f"    KL={t['kl']:.4f} ctx='{t['context'][:40]}' → '{t['target']}' "
                  f"p_f={t['p_float']:.4f} p_r={t['p_relu']:.4f}", flush=True)

        block_result = {
            "block": b,
            "mean_kl": mean_kl,
            "median_kl": median_kl,
            "max_kl": max_kl,
            "frac_relu_better": frac_relu_better,
            "y_neg_norm": y_neg_norm,
            "s_neg_norm": s_neg_norm,
            "top_50": top_tokens_info,
        }
        results["blocks"].append(block_result)
        all_mean_kl.append(mean_kl)
        all_y_neg_norm.append(y_neg_norm)
        all_s_neg_norm.append(s_neg_norm)

    # ---------------------------------------------------------------------------
    # Correlation: mean_KL vs ||y_neg||, mean_KL vs ||s_neg||
    # ---------------------------------------------------------------------------
    kl_arr = np.array(all_mean_kl)
    y_neg_arr = np.array(all_y_neg_norm)
    s_neg_arr = np.array(all_s_neg_norm)

    corr_kl_yneg = np.corrcoef(kl_arr, y_neg_arr)[0, 1] if kl_arr.std() > 0 else 0.0
    corr_kl_sneg = np.corrcoef(kl_arr, s_neg_arr)[0, 1] if kl_arr.std() > 0 else 0.0
    print(f"\nCorrelation mean_KL vs ||y_neg||: {corr_kl_yneg:.4f}", flush=True)
    print(f"Correlation mean_KL vs ||s_neg||: {corr_kl_sneg:.4f}", flush=True)

    results["corr_kl_yneg"] = float(corr_kl_yneg)
    results["corr_kl_sneg"] = float(corr_kl_sneg)

    # ---------------------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------------------
    n_above_01 = sum(1 for kl in all_mean_kl if kl > 0.1)
    n_below_001 = sum(1 for kl in all_mean_kl if kl < 0.01)

    # Semantic pattern check: see if top-50 tokens across blocks share common
    # target tokens (same token appearing in top-50 of ≥4 blocks)
    from collections import Counter
    all_top_targets = []
    for br in results["blocks"]:
        targets_set = set(t["target"].strip() for t in br["top_50"])
        all_top_targets.extend(targets_set)
    target_counts = Counter(all_top_targets)
    repeated_targets = {t: c for t, c in target_counts.items() if c >= 4}
    has_semantic_pattern = len(repeated_targets) >= 3
    results["repeated_top_targets"] = {t: c for t, c in sorted(repeated_targets.items(),
                                                                 key=lambda x: -x[1])[:20]}
    results["has_semantic_pattern"] = has_semantic_pattern

    if n_above_01 >= 8 and has_semantic_pattern:
        verdict = "confirmed"
    elif n_below_001 >= 8:
        verdict = "refuted"
    else:
        verdict = "inconclusive"

    results["verdict"] = verdict
    results["n_above_01"] = n_above_01
    results["n_below_001"] = n_below_001

    print(f"\n=== VERDICT: {verdict.upper()} ===", flush=True)
    print(f"  Blocks with mean KL > 0.1: {n_above_01}/12", flush=True)
    print(f"  Blocks with mean KL < 0.01: {n_below_001}/12", flush=True)
    print(f"  Semantic pattern: {has_semantic_pattern}", flush=True)

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Mean KL per block
    ax = axes[0]
    blocks_x = list(range(N_BLOCKS))
    ax.bar(blocks_x, all_mean_kl, color="steelblue")
    ax.axhline(0.1, color="green", linestyle="--", alpha=0.7, label="confirmed threshold")
    ax.axhline(0.01, color="red", linestyle="--", alpha=0.7, label="refuted threshold")
    ax.set_xlabel("Block")
    ax.set_ylabel("Mean KL (nats)")
    ax.set_title("Per-block KL(P_float || P_relu)")
    ax.legend(fontsize=8)

    # Panel 2: Correlation scatter — KL vs ||y_neg||
    ax = axes[1]
    ax.scatter(y_neg_arr, kl_arr, c="steelblue", s=40)
    for i in range(N_BLOCKS):
        ax.annotate(str(i), (y_neg_arr[i], kl_arr[i]), fontsize=8)
    ax.set_xlabel("||y_neg|| (mean)")
    ax.set_ylabel("Mean KL (nats)")
    ax.set_title(f"KL vs ||y_neg|| (r={corr_kl_yneg:.3f})")

    # Panel 3: Correlation scatter — KL vs ||s_neg||
    ax = axes[2]
    ax.scatter(s_neg_arr, kl_arr, c="coral", s=40)
    for i in range(N_BLOCKS):
        ax.annotate(str(i), (s_neg_arr[i], kl_arr[i]), fontsize=8)
    ax.set_xlabel("||s_neg|| (mean)")
    ax.set_ylabel("Mean KL (nats)")
    ax.set_title(f"KL vs ||s_neg|| (r={corr_kl_sneg:.3f})")

    fig.suptitle(f"Cycle 175: GELU→ReLU prediction impact — {verdict.upper()}", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot to {OUT_PNG}", flush=True)

    # Save JSON
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
