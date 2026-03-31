"""
Cycle 169 / Item 179: Does constant bias shift under compression? — GPT-2 FFN

Cycle 166 showed b_leak = E[y_neg] is functionally a constant bias per block
on the float model (11/12 within +5% PPL). But under topk+lstsq compression
(k=4 blocks at 6x), upstream blocks produce different activations, so b_leak
may shift. If the shift is large, fixing it is a cheap cascade correction.

Steps:
  (a) Compute b_leak_float in MLP output space (768-dim) per block on float model
  (b) Compress k=4 blocks at 6x, compute b_leak_compressed on compressed model
  (c) Measure relative shift per block
  (d) If shift > 20%: fix b_leak to float value, measure PPL improvement

Confirmed if: shift > 20% AND fixing reduces cascade by >= 10%.
Refuted if: shift < 5%.
Inconclusive if: shift exists but fixing doesn't help.
"""

import copy
import json
import math
import os
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
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedMLP,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
N_SEQS = 200
SEQ_LEN = 128
BATCH_SIZE = 4
SEED = 42
N_BLOCKS = 12
N_TOKENS_BLEAK = 4096
COMPRESS_BLOCKS = [7, 5, 4, 1]  # k=4 blocks, standard order
HIDDEN_FRACTION = 1 / 6  # 6x compression
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
OUT_JSON = ROOT / "autoresearch" / "results" / "cycle_169_bias_shift.json"
OUT_PNG = ROOT / "autoresearch" / "results" / "cycle_169_bias_shift.png"
BASELINE_PPL_LO, BASELINE_PPL_HI = 15.0, 65.0

print(f"Device: {DEVICE}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
# b_leak computation in OUTPUT space (768-dim)
# ---------------------------------------------------------------------------
def compute_b_leak_output_space(model, seqs, n_tokens=N_TOKENS_BLEAK):
    """
    Compute b_leak in MLP output space (768-dim) per block.

    For each block, we hook the MLP to:
    1. Get the pre-activation z (input to activation function)
    2. Get the post-activation h (output of activation function)
    3. Compute h_neg = h where z <= 0 (contribution from negative channels)
    4. Project through W_down: output_neg = h_neg @ W_down.T (+ bias contribution)

    This gives b_leak_output = E[W_down @ h_neg] in 768-dim, comparable across
    compressed (512 hidden) and uncompressed (3072 hidden) blocks.
    """
    b_leak = {}
    accum = {}  # block_idx -> accumulated sum in output space (768,)
    counts = {}  # block_idx -> token count

    # We need to track pre-activation z to know which channels are negative,
    # and then compute the output-space contribution of those channels.
    # Strategy: hook the activation to capture z and h, then project h_neg through W_down.
    act_data = {}  # block_idx -> (z, h) from most recent forward

    hooks = []

    def make_act_hook(idx):
        """Hook on the activation module to capture z (input) and h (output)."""
        def hook_fn(module, inp, out):
            z = inp[0].detach()  # pre-activation, (batch, seq, hidden_dim)
            h = out.detach()     # post-activation
            act_data[idx] = (z, h)
        return hook_fn

    def make_mlp_hook(idx):
        """Hook on the full MLP to project h_neg through W_down after activation hook runs."""
        def hook_fn(module, inp, out):
            if idx not in act_data:
                return
            z, h = act_data[idx]
            # Mask: where z <= 0
            mask_neg = (z <= 0).float()
            h_neg = h * mask_neg  # zero out positive channels
            # Project through W_down (c_proj)
            # For both GPT2MLP and CompressedMLP, c_proj is the down-projection
            if hasattr(module, 'c_proj'):
                w_down = module.c_proj.weight.detach()  # Conv1D: (hidden, output) or Linear: (output, hidden)
                b_down = module.c_proj.bias.detach() if module.c_proj.bias is not None else None

                # HuggingFace Conv1D: weight is (in_features, out_features), forward = x @ weight + bias
                # nn.Linear: weight is (out_features, in_features), forward = x @ weight.T + bias
                if type(module.c_proj).__name__ == "Conv1D":
                    # Conv1D: x @ weight, weight shape = (hidden_dim, output_dim)
                    output_neg = h_neg @ w_down
                else:
                    # nn.Linear: x @ weight.T
                    output_neg = h_neg @ w_down.t()

                # Note: we DON'T add b_down here — the bias is shared between pos/neg channels
                # and is not part of the "negative regime contribution". We want ONLY the
                # contribution of h_neg going through W_down.

                # Sum over batch and seq dims -> (768,)
                batch_sum = output_neg.sum(dim=0)
                if batch_sum.dim() > 1:
                    batch_sum = batch_sum.sum(dim=0)
                n = z.shape[0] * (z.shape[1] if z.dim() == 3 else 1)

                if idx not in accum:
                    accum[idx] = batch_sum.cpu().float()
                    counts[idx] = n
                else:
                    accum[idx] += batch_sum.cpu().float()
                    counts[idx] += n
        return hook_fn

    # Register hooks on each block's MLP
    for i in range(N_BLOCKS):
        block = model.transformer.h[i]
        mlp = block.mlp
        # Hook the activation
        if isinstance(mlp, CompressedMLP):
            # CompressedMLP doesn't have .act attribute — activation is inline in forward
            # We need a different approach: hook c_fc output and c_proj input
            # Actually, let's just wrap the forward to capture internals
            pass
        else:
            h_act = mlp.act.register_forward_hook(make_act_hook(i))
            hooks.append(h_act)
        h_mlp = mlp.register_forward_hook(make_mlp_hook(i))
        hooks.append(h_mlp)

    # For CompressedMLP blocks, we need a special hook since there's no .act module
    # Solution: hook c_fc (output = pre-activation z) and use a wrapper
    compressed_block_ids = set()
    for i in range(N_BLOCKS):
        mlp = model.transformer.h[i].mlp
        if isinstance(mlp, CompressedMLP):
            compressed_block_ids.add(i)

            def make_cfc_hook(idx):
                def hook_fn(module, inp, out):
                    # out = z (pre-activation), shape (batch, seq, hidden)
                    # We'll compute h = gelu(z) manually and store
                    z = out.detach()
                    h = F.gelu(z)  # CompressedMLP uses gelu
                    act_data[idx] = (z, h)
                return hook_fn

            h_cfc = mlp.c_fc.register_forward_hook(make_cfc_hook(i))
            hooks.append(h_cfc)

    # Run enough sequences
    n_seqs_needed = max(1, n_tokens // SEQ_LEN + 1)
    model.eval()
    with torch.no_grad():
        for i in range(0, min(n_seqs_needed, len(seqs)), BATCH_SIZE):
            batch = seqs[i:i + BATCH_SIZE, :-1].to(DEVICE)
            model(batch)

    for h in hooks:
        h.remove()

    for i in range(N_BLOCKS):
        if i in accum:
            b_leak[i] = (accum[i] / counts[i]).to(DEVICE)
            print(f"  Block {i}: b_leak_output mean={b_leak[i].mean().item():.6f}, "
                  f"norm={b_leak[i].norm().item():.4f}, tokens={counts[i]}", flush=True)
        else:
            print(f"  Block {i}: NO DATA (hook didn't fire)", flush=True)

    return b_leak


# ---------------------------------------------------------------------------
# Bias correction wrapper
# ---------------------------------------------------------------------------
class BiasFixWrapper(nn.Module):
    """Wraps an MLP to add a constant correction to its output."""
    def __init__(self, original_mlp, correction_vec):
        super().__init__()
        self.original_mlp = original_mlp
        self.register_buffer("correction", correction_vec)

    def forward(self, x):
        return self.original_mlp(x) + self.correction


# ---------------------------------------------------------------------------
# Compression pipeline
# ---------------------------------------------------------------------------
def compress_model(teacher_bundle, corpus):
    """Compress k=4 blocks at 6x using topk+lstsq."""
    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        selection_mode="topk",
    )
    atlas = probe_ffn_geometry(
        teacher_bundle,
        corpus,
        layer_ids=tuple(COMPRESS_BLOCKS),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )
    pack = derive_ffn_compression(atlas, budget)
    compressed_model = swap_compressed_ffns(teacher_bundle, pack, inplace=False)
    return compressed_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed()
    print("Loading model and tokenizer...", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()

    print("Loading eval data...", flush=True)
    seqs = load_owt_tokens(tokenizer)

    # Step 1: Baseline PPL
    print("\n--- Step 1: Baseline PPL ---", flush=True)
    ppl_base = eval_ppl(model, seqs)
    print(f"  Baseline PPL: {ppl_base:.3f}", flush=True)
    assert BASELINE_PPL_LO <= ppl_base <= BASELINE_PPL_HI, \
        f"Baseline PPL {ppl_base:.3f} outside [{BASELINE_PPL_LO}, {BASELINE_PPL_HI}]"

    # Step 2: Compute b_leak_float per block
    print("\n--- Step 2: b_leak on FLOAT model ---", flush=True)
    b_leak_float = compute_b_leak_output_space(model, seqs)

    # Step 3: Compress k=4 blocks
    print("\n--- Step 3: Compressing k=4 blocks ---", flush=True)
    teacher_spec = TeacherLoadSpec(
        source="hf_pretrained",
        hf_model_name="gpt2",
        dataset_name="openwebtext_sample",
        device=DEVICE,
    )
    teacher = load_nanochat_teacher(teacher_spec)
    corpus = build_probe_corpus(
        ProbeCorpusSpec(
            probe_train_sequences=PROBE_TRAIN_SEQS,
            probe_holdout_sequences=PROBE_TRAIN_SEQS,
            eval_val_sequences=N_SEQS,
            sequence_length=SEQ_LEN,
            text_path=str(OWT_PATH),
            teacher=teacher,
            shuffle_seed=SEED,
        ),
    )
    compressed_model = compress_model(teacher, corpus)
    compressed_model = compressed_model.to(DEVICE).eval()

    ppl_compressed = eval_ppl(compressed_model, seqs)
    print(f"  Compressed PPL: {ppl_compressed:.3f}", flush=True)

    # Step 4: Compute b_leak_compressed per block
    print("\n--- Step 4: b_leak on COMPRESSED model ---", flush=True)
    b_leak_compressed = compute_b_leak_output_space(compressed_model, seqs)

    # Step 5: Measure shift
    print("\n--- Step 5: Relative shift ---", flush=True)
    per_block = {}
    shifts = []
    for i in range(N_BLOCKS):
        if i not in b_leak_float or i not in b_leak_compressed:
            print(f"  Block {i}: MISSING DATA", flush=True)
            continue
        bf = b_leak_float[i]
        bc = b_leak_compressed[i]
        delta = (bc - bf).norm().item()
        ref = bf.norm().item()
        rel_shift = delta / ref if ref > 1e-8 else float("inf")
        per_block[i] = {
            "b_leak_float_norm": ref,
            "b_leak_compressed_norm": bc.norm().item(),
            "delta_norm": delta,
            "relative_shift": rel_shift,
            "is_compressed_block": i in COMPRESS_BLOCKS,
        }
        shifts.append(rel_shift)
        tag = " [COMPRESSED]" if i in COMPRESS_BLOCKS else ""
        print(f"  Block {i}: shift={rel_shift:.4f} ({rel_shift*100:.1f}%){tag}", flush=True)

    max_shift = max(shifts) if shifts else 0
    mean_shift = np.mean(shifts) if shifts else 0
    # Separate compressed vs uncompressed blocks
    compressed_shifts = [per_block[i]["relative_shift"] for i in COMPRESS_BLOCKS if i in per_block]
    uncompressed_shifts = [per_block[i]["relative_shift"] for i in range(N_BLOCKS)
                          if i not in COMPRESS_BLOCKS and i in per_block]
    mean_compressed_shift = np.mean(compressed_shifts) if compressed_shifts else 0
    mean_uncompressed_shift = np.mean(uncompressed_shifts) if uncompressed_shifts else 0

    print(f"\n  Max shift: {max_shift:.4f} ({max_shift*100:.1f}%)")
    print(f"  Mean shift (all): {mean_shift:.4f} ({mean_shift*100:.1f}%)")
    print(f"  Mean shift (compressed blocks): {mean_compressed_shift:.4f} ({mean_compressed_shift*100:.1f}%)")
    print(f"  Mean shift (uncompressed blocks): {mean_uncompressed_shift:.4f} ({mean_uncompressed_shift*100:.1f}%)")

    # Step 6: If shift > 20%, try fixing b_leak
    ppl_fixed = None
    cascade_reduction = None
    any_above_20 = any(s > 0.20 for s in shifts)

    if any_above_20:
        print("\n--- Step 6: Fixing b_leak to float values ---", flush=True)
        # Apply correction to all blocks with shift > 20%
        fixed_model = copy.deepcopy(compressed_model)
        n_fixed = 0
        for i in range(N_BLOCKS):
            if i not in per_block or per_block[i]["relative_shift"] <= 0.20:
                continue
            correction = b_leak_float[i] - b_leak_compressed[i]
            fixed_model.transformer.h[i].mlp = BiasFixWrapper(
                fixed_model.transformer.h[i].mlp, correction
            )
            n_fixed += 1
            print(f"  Fixed block {i} (shift={per_block[i]['relative_shift']*100:.1f}%)", flush=True)

        fixed_model = fixed_model.to(DEVICE).eval()
        ppl_fixed = eval_ppl(fixed_model, seqs)
        ppl_delta_compressed = ppl_compressed - ppl_base
        ppl_delta_fixed = ppl_fixed - ppl_base
        cascade_reduction = (ppl_delta_compressed - ppl_delta_fixed) / ppl_delta_compressed * 100 \
            if ppl_delta_compressed > 0 else 0

        print(f"  PPL base: {ppl_base:.3f}")
        print(f"  PPL compressed: {ppl_compressed:.3f} (delta={ppl_delta_compressed:.3f})")
        print(f"  PPL fixed: {ppl_fixed:.3f} (delta={ppl_delta_fixed:.3f})")
        print(f"  Cascade reduction: {cascade_reduction:.1f}%")
        print(f"  Blocks fixed: {n_fixed}")
    else:
        print("\n--- Step 6: SKIPPED (no block has shift > 20%) ---", flush=True)

    # Verdict
    if max_shift < 0.05:
        verdict = "REFUTED"
    elif any_above_20 and cascade_reduction is not None and cascade_reduction >= 10:
        verdict = "CONFIRMED"
    elif any_above_20 and cascade_reduction is not None and cascade_reduction < 10:
        verdict = "INCONCLUSIVE"
    elif mean_shift >= 0.05 and max_shift < 0.20:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n=== VERDICT: {verdict} ===", flush=True)

    # Save results
    results = {
        "cycle": 169,
        "item": 179,
        "verdict": verdict,
        "ppl_base": round(ppl_base, 3),
        "ppl_compressed": round(ppl_compressed, 3),
        "ppl_fixed": round(ppl_fixed, 3) if ppl_fixed is not None else None,
        "cascade_reduction_pct": round(cascade_reduction, 2) if cascade_reduction is not None else None,
        "max_shift": round(max_shift, 4),
        "mean_shift": round(float(mean_shift), 4),
        "mean_compressed_block_shift": round(float(mean_compressed_shift), 4),
        "mean_uncompressed_block_shift": round(float(mean_uncompressed_shift), 4),
        "compress_blocks": COMPRESS_BLOCKS,
        "per_block": {str(k): {kk: round(vv, 4) if isinstance(vv, float) else vv
                                for kk, vv in v.items()}
                      for k, v in per_block.items()},
    }

    os.makedirs(OUT_JSON.parent, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_JSON}", flush=True)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    blocks = sorted(per_block.keys())
    shift_vals = [per_block[b]["relative_shift"] * 100 for b in blocks]
    colors = ["#d62728" if b in COMPRESS_BLOCKS else "#1f77b4" for b in blocks]
    bars = ax.bar(blocks, shift_vals, color=colors)
    ax.axhline(y=20, color="red", linestyle="--", alpha=0.5, label="20% threshold")
    ax.axhline(y=5, color="orange", linestyle="--", alpha=0.5, label="5% threshold")
    ax.set_xlabel("Block")
    ax.set_ylabel("Relative b_leak shift (%)")
    ax.set_title(f"b_leak shift: float vs compressed (k=4@6x) — {verdict}")
    ax.set_xticks(blocks)
    ax.legend()
    # Add compressed/uncompressed legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="Compressed block"),
        Patch(facecolor="#1f77b4", label="Uncompressed block"),
    ]
    ax.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color="red", linestyle="--", alpha=0.5, label="20% threshold"),
        plt.Line2D([0], [0], color="orange", linestyle="--", alpha=0.5, label="5% threshold"),
    ], loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"Saved: {OUT_PNG}", flush=True)

    return results


if __name__ == "__main__":
    main()
