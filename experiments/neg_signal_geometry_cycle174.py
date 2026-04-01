"""
Cycle 174 — Geometry of token-dependent negative signal in GPT-2 FFN blocks.

For each of 12 blocks on 2048 tokens:
  (a) SVD of S_neg = y_neg - mean(y_neg) → effective rank at 95% energy
  (b) SVD of S_pos = y_pos - mean(y_pos) → effective rank, compare
  (c) Gradient J(x) = d(CE_loss)/d(y_block) per token
  (d) Cosine alignment cos(s_neg, J) and cos(s_pos, J)
  (f) Report per-block metrics

Confirmed if: |cos(s_neg, J)| > 0.2 in ≥8/12 blocks AND rank(S_neg) < rank(S_pos)/4
Refuted if: |cos(s_neg, J)| < 0.05
"""

import json
import math
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_SEQS = 16
SEQ_LEN = 128
N_TOKENS = N_SEQS * SEQ_LEN  # 2048
SEED = 42
ENERGY_THRESHOLD = 0.95
OWT_PATH = "data/openwebtext_sample.txt"
OUT_JSON = Path("autoresearch/results/cycle_174_neg_signal_geometry.json")
EVAL_N_SEQS = 200
BATCH_SIZE = 4


def load_owt_tokens(tokenizer, n_seqs, seq_len, seed=SEED):
    """Load OWT tokens from local text file."""
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
    """Evaluate perplexity."""
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


def effective_rank(S_matrix, threshold=ENERGY_THRESHOLD):
    """Compute effective rank at given energy threshold via SVD."""
    # S_matrix: [n_tokens, d_model]
    # Use float32 for SVD stability
    U, s, Vh = torch.linalg.svd(S_matrix.float(), full_matrices=False)
    energy = (s ** 2).cumsum(0) / (s ** 2).sum()
    rank = (energy < threshold).sum().item() + 1
    return rank, s


def cosine_alignment(A, B):
    """Per-row cosine similarity between A and B, return mean|cos|."""
    # A, B: [n_tokens, d_model]
    A_norm = F.normalize(A.float(), dim=1)
    B_norm = F.normalize(B.float(), dim=1)
    cos = (A_norm * B_norm).sum(dim=1)  # [n_tokens]
    return cos.abs().mean().item(), cos


def analyze_block(model, input_ids, target_ids, block_idx):
    """
    For one block:
    1. Forward pass to get post-GELU activations → y_neg, y_pos, s_neg, s_pos, ranks
    2. Backward pass to get J(x) = d(CE)/d(block_output) per token
    3. Compute cosine alignments
    """
    n_blocks = len(model.transformer.h)
    block = model.transformer.h[block_idx]

    # ---- Phase 1: Signal decomposition (no grad needed) ----
    gelu_output = {}

    def hook_gelu(module, input, output):
        # GPT-2 MLP: c_fc → gelu → c_proj
        # We hook the activation function output (post-GELU, pre-c_proj)
        gelu_output["val"] = output.detach()

    # The GELU activation is applied inside the MLP forward. We need to hook
    # the intermediate activation. In GPT-2, MLP is:
    #   h = act(c_fc(x))
    #   output = c_proj(h)
    # We want h (post-GELU). Hook c_proj input.
    handle = block.mlp.c_proj.register_forward_hook(
        lambda m, inp, out: gelu_output.update({"val": inp[0].detach()})
    )

    with torch.no_grad():
        _ = model(input_ids)

    handle.remove()
    h_gelu = gelu_output["val"]  # [N_SEQS, SEQ_LEN, 3072]

    # Get c_proj weight: W_out is [768, 3072] for Conv1D or [3072, 768] for Linear
    # GPT-2 uses Conv1D where weight is [in_features, out_features]
    W_out = block.mlp.c_proj.weight.detach()  # Conv1D: [3072, 768]

    # Flatten to [N_TOKENS, 3072]
    h_flat = h_gelu.reshape(-1, h_gelu.shape[-1])  # [2048, 3072]

    # Negative/positive masks based on pre-GELU sign
    # Actually, GELU(z) < 0 when z is in a small negative range. But the convention
    # from prior cycles (109, 173) is: negative regime = channels where GELU output < 0
    neg_mask = (h_flat < 0).float()  # [2048, 3072]
    pos_mask = (h_flat >= 0).float()

    # y_neg, y_pos per token: contribution through c_proj
    # Conv1D: output = input @ weight, so y = h @ W_out  (W_out is [3072, 768])
    y_neg = (h_flat * neg_mask) @ W_out  # [2048, 768]
    y_pos = (h_flat * pos_mask) @ W_out  # [2048, 768]

    # Token-dependent signals
    mu_neg = y_neg.mean(dim=0)  # [768]
    mu_pos = y_pos.mean(dim=0)
    s_neg = y_neg - mu_neg  # [2048, 768]
    s_pos = y_pos - mu_pos

    # SVD for effective rank
    rank_neg, sv_neg = effective_rank(s_neg)
    rank_pos, sv_pos = effective_rank(s_pos)

    # L2 norms
    l2_neg = s_neg.norm(dim=1).mean().item()
    l2_pos = s_pos.norm(dim=1).mean().item()

    # ---- Phase 2: Gradient computation ----
    # We need d(CE_loss)/d(block_l_output) per token.
    # block_l_output is the residual addition: x + attn_out + mlp_out
    # We want d(loss)/d(mlp_out) specifically, since that's the FFN contribution.
    # Actually the item says d(CE)/d(y_block) where y_block is the block output
    # in the residual stream. Let's capture gradient at the block output.

    block_output_tensor = {}

    def hook_save_output(module, input, output):
        # GPT-2 block output is a tuple: (hidden_states, ...)
        # output[0] is the hidden states
        out_tensor = output[0]
        out_tensor.requires_grad_(True)
        out_tensor.retain_grad()
        block_output_tensor["val"] = out_tensor
        # Return modified output so autograd can track it
        if len(output) > 1:
            return (out_tensor,) + output[1:]
        return (out_tensor,)

    handle = model.transformer.h[block_idx].register_forward_hook(hook_save_output)

    model.zero_grad()
    logits = model(input_ids).logits  # [N_SEQS, SEQ_LEN, vocab]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           target_ids.reshape(-1))
    loss.backward()

    handle.remove()

    J = block_output_tensor["val"].grad  # [N_SEQS, SEQ_LEN, 768]
    if J is None:
        print(f"  WARNING: gradient is None for block {block_idx}")
        return None

    J_flat = J.reshape(-1, J.shape[-1])  # [2048, 768]

    # ---- Phase 3: Alignment ----
    mean_cos_neg_J, cos_neg = cosine_alignment(s_neg, J_flat)
    mean_cos_pos_J, cos_pos = cosine_alignment(s_pos, J_flat)

    result = {
        "block": block_idx,
        "rank_neg_95": rank_neg,
        "rank_pos_95": rank_pos,
        "rank_ratio": rank_pos / max(rank_neg, 1),
        "l2_s_neg": l2_neg,
        "l2_s_pos": l2_pos,
        "l2_ratio": l2_pos / max(l2_neg, 1e-8),
        "mean_abs_cos_neg_J": mean_cos_neg_J,
        "mean_abs_cos_pos_J": mean_cos_pos_J,
        "cos_ratio": mean_cos_neg_J / max(mean_cos_pos_J, 1e-8),
        "top10_sv_neg": sv_neg[:10].tolist(),
        "top10_sv_pos": sv_pos[:10].tolist(),
    }
    return result


def main():
    print(f"Device: {DEVICE}")
    torch.manual_seed(SEED)

    # Load model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()

    # Load eval data for PPL (use longer seq_len for stable PPL measurement)
    print("Loading data...")
    EVAL_SEQ_LEN = 1024
    eval_seqs = load_owt_tokens(tokenizer, EVAL_N_SEQS, EVAL_SEQ_LEN, seed=SEED)
    ppl = eval_ppl(model, eval_seqs)
    print(f"Float PPL: {ppl:.2f} (eval seq_len={EVAL_SEQ_LEN})")
    assert 15 < ppl < 40, f"Float PPL {ppl:.2f} outside [15, 40], aborting"

    # Load analysis data (shorter seqs for gradient computation)
    analysis_seqs = load_owt_tokens(tokenizer, N_SEQS, SEQ_LEN, seed=SEED + 1)
    input_ids = analysis_seqs[:, :-1].to(DEVICE)  # [N_SEQS, SEQ_LEN]
    target_ids = analysis_seqs[:, 1:].to(DEVICE)   # [N_SEQS, SEQ_LEN]
    print(f"Analysis input shape: {input_ids.shape}")

    # Per-block analysis
    results = []
    for block_idx in range(12):
        print(f"\nBlock {block_idx}...")
        r = analyze_block(model, input_ids, target_ids, block_idx)
        if r is None:
            print(f"  SKIP block {block_idx} (gradient failed)")
            continue
        results.append(r)
        print(f"  rank_neg={r['rank_neg_95']:3d}  rank_pos={r['rank_pos_95']:3d}  "
              f"ratio={r['rank_ratio']:.1f}x")
        print(f"  |cos(s_neg,J)|={r['mean_abs_cos_neg_J']:.4f}  "
              f"|cos(s_pos,J)|={r['mean_abs_cos_pos_J']:.4f}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Block':>5} {'rank_neg':>8} {'rank_pos':>8} {'ratio':>6} "
          f"{'|cos_neg|':>10} {'|cos_pos|':>10} {'L2_neg':>8} {'L2_pos':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['block']:5d} {r['rank_neg_95']:8d} {r['rank_pos_95']:8d} "
              f"{r['rank_ratio']:6.1f} {r['mean_abs_cos_neg_J']:10.4f} "
              f"{r['mean_abs_cos_pos_J']:10.4f} {r['l2_s_neg']:8.2f} "
              f"{r['l2_s_pos']:8.2f}")

    # Verdict
    n_aligned = sum(1 for r in results if r["mean_abs_cos_neg_J"] > 0.2)
    n_low_rank = sum(1 for r in results if r["rank_ratio"] > 4.0)
    n_both = sum(1 for r in results
                 if r["mean_abs_cos_neg_J"] > 0.2 and r["rank_ratio"] > 4.0)
    mean_cos_neg = sum(r["mean_abs_cos_neg_J"] for r in results) / len(results)
    mean_cos_pos = sum(r["mean_abs_cos_pos_J"] for r in results) / len(results)
    mean_rank_neg = sum(r["rank_neg_95"] for r in results) / len(results)
    mean_rank_pos = sum(r["rank_pos_95"] for r in results) / len(results)

    if n_both >= 8:
        verdict = "CONFIRMED"
    elif mean_cos_neg < 0.05:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\nBlocks with |cos(s_neg,J)| > 0.2: {n_aligned}/12")
    print(f"Blocks with rank_pos/rank_neg > 4: {n_low_rank}/12")
    print(f"Blocks meeting BOTH criteria: {n_both}/12")
    print(f"Mean |cos(s_neg,J)| = {mean_cos_neg:.4f}, mean |cos(s_pos,J)| = {mean_cos_pos:.4f}")
    print(f"Mean rank_neg = {mean_rank_neg:.1f}, mean rank_pos = {mean_rank_pos:.1f}")
    print(f"\nVERDICT: {verdict}")

    # Save
    output = {
        "cycle": 174,
        "item": 185,
        "device": DEVICE,
        "seed": SEED,
        "n_tokens": N_TOKENS,
        "seq_len": SEQ_LEN,
        "n_seqs": N_SEQS,
        "float_ppl": ppl,
        "energy_threshold": ENERGY_THRESHOLD,
        "per_block": results,
        "summary": {
            "n_aligned_blocks": n_aligned,
            "n_low_rank_blocks": n_low_rank,
            "n_both_criteria": n_both,
            "mean_abs_cos_neg_J": mean_cos_neg,
            "mean_abs_cos_pos_J": mean_cos_pos,
            "mean_rank_neg": mean_rank_neg,
            "mean_rank_pos": mean_rank_pos,
        },
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
