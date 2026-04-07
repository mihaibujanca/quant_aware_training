"""
Cycle 238 — Item 248 (T4): Activation patching across token pairs (bump-only).

For 100 (A, B, p) token pairs sampled from held-out wikitext-2 val:
- A and B are distinct sequences, top1 at position p ∈ {32, 64, 96} differs
- Both logit margins over runner-up > 2.0 (fallback 1.5, then more positions)

For each pair × block l ∈ {1, 3, 5, 7, 9}:
  bump_only patch: h[l+1] input @ (A_seq, p) ← x_l_A - Δ̃_A + Δ̃_B
  full patch:      h[l+1] input @ (A_seq, p) ← x_l_B
  bump_attribution = (bump_logit_on_Btop - clean_logit_on_Btop) /
                     (full_logit_on_Btop - clean_logit_on_Btop)

Verdict:
  CONFIRMED    := mean bump_attribution > 0.15 in ≥3 of 5 blocks (MEDIAN if mean tails)
  REFUTED      := all 5 blocks have mean bump_attribution < 0.05
  INCONCLUSIVE := otherwise
"""
import json
import math
import random

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
N_SEQS = 200
SEQ_LEN = 128
BLOCKS = [1, 3, 5, 7, 9]
POSITIONS = [32, 64, 96]
MARGIN_MIN = 2.0
MARGIN_MIN_FALLBACK = 1.5
EXPANDED_POSITIONS = [16, 32, 48, 64, 80, 96, 112]
N_PAIRS = 100
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

RESULTS_DIR = "autoresearch/results"
FEATURES_PATH = f"{RESULTS_DIR}/cycle_235_t1_features.pt"
OUT_JSON = f"{RESULTS_DIR}/cycle_238_t4_patching.json"
OUT_EXAMPLES = f"{RESULTS_DIR}/cycle_238_t4_examples.md"
OUT_SUMMARY = f"{RESULTS_DIR}/cycle_238_t4_summary.md"

torch.manual_seed(SEED)
random.seed(SEED)
print(f"Device: {DEVICE}")

# ── Load model + tokenizer ───────────────────────────────────────────────────
print("Loading GPT-2 and tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# ── Load cycle 235 features → mean_delta_l per block ────────────────────────
print("Loading cycle 235 T1 features (mean_delta only)...")
features = torch.load(FEATURES_PATH, weights_only=False)
mean_delta = {
    l: features[f"block_{l}"]["mean_delta"].to(DEVICE).float() for l in BLOCKS
}
for l in BLOCKS:
    assert mean_delta[l].shape == (768,), (l, mean_delta[l].shape)
print(f"  mean_delta blocks: {BLOCKS}  shape: (768,)")

# Pre-stash FFN weights per block for fast Δ̃ computation.
W_fc_dev = {}
b_fc_dev = {}
W_proj_dev = {}
for l in BLOCKS:
    blk = model.transformer.h[l]
    W_fc_dev[l] = blk.mlp.c_fc.weight.detach().to(DEVICE).float()     # (768, 3072)
    b_fc_dev[l] = blk.mlp.c_fc.bias.detach().to(DEVICE).float()       # (3072,)
    W_proj_dev[l] = blk.mlp.c_proj.weight.detach().to(DEVICE).float() # (3072, 768)

# ── Tokenise held-out wikitext-2 val ─────────────────────────────────────────
print(f"\nLoading wikitext-2 val split (N_SEQS={N_SEQS}, SEQ_LEN={SEQ_LEN})...")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if r["text"].strip()]
tokens_needed = N_SEQS * SEQ_LEN
val_tokens = []
for t in texts:
    ids = tokenizer(t, return_tensors="pt", truncation=False)["input_ids"][0]
    val_tokens.append(ids)
flat = torch.cat(val_tokens, dim=0)
if flat.numel() < tokens_needed:
    raise RuntimeError(f"Need {tokens_needed} tokens but only have {flat.numel()}")
flat = flat[:tokens_needed].contiguous()
eval_ids = flat.view(N_SEQS, SEQ_LEN).contiguous()
print(f"  eval_ids shape: {tuple(eval_ids.shape)}")

# ── Clean forward pass: cache logits ─────────────────────────────────────────
print("\nClean forward pass (cache logits)...")
clean_logits = torch.zeros(N_SEQS, SEQ_LEN, tokenizer.vocab_size, dtype=torch.float32)
BATCH = 8
with torch.no_grad():
    for i in range(0, N_SEQS, BATCH):
        ids = eval_ids[i:i + BATCH].to(DEVICE)
        out = model(ids).logits.detach().float().cpu()
        clean_logits[i:i + BATCH] = out
print(f"  clean_logits shape: {tuple(clean_logits.shape)}")


# ── Pair sampling ────────────────────────────────────────────────────────────
def build_pool(positions, margin_min):
    pool = []
    for p in positions:
        logits_p = clean_logits[:, p, :]  # (N_SEQS, V)
        top2 = torch.topk(logits_p, 2, dim=-1)
        top1 = top2.indices[:, 0]  # (N_SEQS,)
        margin = (top2.values[:, 0] - top2.values[:, 1])  # (N_SEQS,)
        ok = margin > margin_min
        ok_idx = torch.nonzero(ok, as_tuple=False).flatten().tolist()
        top1_list = top1.tolist()
        for ai in range(len(ok_idx)):
            i = ok_idx[ai]
            for bi in range(ai + 1, len(ok_idx)):
                j = ok_idx[bi]
                if top1_list[i] != top1_list[j]:
                    pool.append((i, j, p, top1_list[i], top1_list[j]))
    return pool


positions_used = list(POSITIONS)
margin_used = MARGIN_MIN
pool = build_pool(positions_used, margin_used)
print(f"\nPool at margin={margin_used} positions={positions_used}: {len(pool)}")

if len(pool) < N_PAIRS:
    margin_used = MARGIN_MIN_FALLBACK
    pool = build_pool(positions_used, margin_used)
    print(f"Pool at margin={margin_used}: {len(pool)}")

if len(pool) < N_PAIRS:
    positions_used = list(EXPANDED_POSITIONS)
    pool = build_pool(positions_used, margin_used)
    print(f"Pool at margin={margin_used} expanded positions={positions_used}: {len(pool)}")

if len(pool) < N_PAIRS:
    raise RuntimeError(f"Pool too small: {len(pool)} < {N_PAIRS}")

rng = random.Random(SEED)
rng.shuffle(pool)
pairs = pool[:N_PAIRS]
print(f"Final pairs: {len(pairs)}  margin={margin_used}  positions={positions_used}")


# ── Δ̃ helper ────────────────────────────────────────────────────────────────
def compute_delta_tilde(seq_ids_batch: torch.Tensor, l: int, p: int) -> torch.Tensor:
    """Return centred Δ̃_l @ position p for a 1-row batch."""
    assert seq_ids_batch.shape[0] == 1
    ln2_out_cache = {}

    def prehook(module, args):
        ln2_out_cache["x"] = args[0][:, p, :].detach().float()  # (1, 768)

    blk = model.transformer.h[l]
    handle = blk.mlp.c_fc.register_forward_pre_hook(prehook)
    with torch.no_grad():
        _ = model(seq_ids_batch.to(DEVICE))
    handle.remove()
    x = ln2_out_cache["x"][0]  # (768,)
    preact = x @ W_fc_dev[l] + b_fc_dev[l]  # (3072,)
    delta_pre = F.gelu(preact, approximate="tanh") - F.relu(preact)
    delta_at_p = delta_pre @ W_proj_dev[l]  # (768,)
    return (delta_at_p - mean_delta[l]).detach()


# ── x_l cache: residual stream after block l at each (seq, p) ───────────────
print("\nCaching x_l per block × position...")
cached_x = {l: {} for l in BLOCKS}  # cached_x[l][(seq_idx, p)] = (768,)
for l in BLOCKS:
    stash = []  # list of (batch_start, output_tensor) — just stash output

    def make_hook(L):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            stash.append(hs.detach().float().cpu())
        return hook

    handle = model.transformer.h[l].register_forward_hook(make_hook(l))
    with torch.no_grad():
        for i in range(0, N_SEQS, BATCH):
            ids = eval_ids[i:i + BATCH].to(DEVICE)
            _ = model(ids)
    handle.remove()
    x_l_full = torch.cat(stash, dim=0)  # (N_SEQS, SEQ_LEN, 768)
    assert x_l_full.shape == (N_SEQS, SEQ_LEN, 768), (l, x_l_full.shape)
    for p in positions_used:
        cached_x[l][p] = x_l_full[:, p, :].contiguous()  # (N_SEQS, 768)
    del x_l_full
print("  x_l cache built.")


# ── Patching forward helper ──────────────────────────────────────────────────
def patched_forward(seq_ids_A_batch: torch.Tensor, l: int, p: int, v: torch.Tensor) -> torch.Tensor:
    """Return (SEQ_LEN, V) logits at row 0 after patching h[l+1] input at (0, p, :)"""
    v_local = v.detach()

    def prehook(module, args):
        hs = args[0]
        new_hs = hs.clone()
        new_hs[0, p, :] = v_local.to(dtype=new_hs.dtype, device=new_hs.device)
        rest = args[1:]
        return (new_hs,) + rest

    # Target: h[l+1] forward_pre_hook. BLOCKS max is 9 so l+1=10 exists.
    target = model.transformer.h[l + 1]
    handle = target.register_forward_pre_hook(prehook)
    try:
        with torch.no_grad():
            out = model(seq_ids_A_batch.to(DEVICE)).logits[0].detach().float().cpu()
    finally:
        handle.remove()
    return out  # (SEQ_LEN, V)


# ── Identity sanity (v_bump=x_lA at a few random pairs/blocks) ───────────────
print("\nIdentity sanity checks (v=x_lA must reproduce clean logits)...")
sanity_ok = True
sanity_max_delta = 0.0
rng_s = random.Random(SEED + 1)
for _ in range(5):
    pi = rng_s.randrange(len(pairs))
    l = BLOCKS[rng_s.randrange(len(BLOCKS))]
    A_seq, B_seq, p, A_top, B_top = pairs[pi]
    ids_A = eval_ids[A_seq:A_seq + 1]
    x_lA = cached_x[l][p][A_seq].to(DEVICE)
    patched = patched_forward(ids_A, l, p, x_lA)
    clean = clean_logits[A_seq]  # (SEQ_LEN, V)
    delta = (patched - clean).abs().max().item()
    sanity_max_delta = max(sanity_max_delta, delta)
    if delta > 1e-3:
        sanity_ok = False
print(f"  identity sanity max |Δ logit| = {sanity_max_delta:.2e}  ok={sanity_ok}")
if not sanity_ok:
    print("  WARNING: identity patch does not reproduce clean logits within 1e-3")


# ── Main patching loop ──────────────────────────────────────────────────────
print("\nMain patching loop...")
records = []
for pi, (A_seq, B_seq, p, A_top, B_top) in enumerate(pairs):
    ids_A = eval_ids[A_seq:A_seq + 1]
    clean_logits_Ap = clean_logits[A_seq, p]  # (V,)
    orig_Btop = float(clean_logits_Ap[B_top].item())
    orig_top1 = int(clean_logits_Ap.argmax().item())

    for l in BLOCKS:
        # Δ̃_A, Δ̃_B at position p
        dA = compute_delta_tilde(eval_ids[A_seq:A_seq + 1], l, p)  # (768,)
        dB = compute_delta_tilde(eval_ids[B_seq:B_seq + 1], l, p)  # (768,)
        assert dA.shape == (768,) and dB.shape == (768,)
        x_lA = cached_x[l][p][A_seq].to(DEVICE)  # (768,)
        x_lB = cached_x[l][p][B_seq].to(DEVICE)  # (768,)

        v_bump = x_lA - dA + dB
        v_full = x_lB

        # bump-only patch
        bump_logits = patched_forward(ids_A, l, p, v_bump)  # (SEQ_LEN, V)
        bump_Btop = float(bump_logits[p, B_top].item())
        bump_top1 = int(bump_logits[p].argmax().item())

        # full-residual patch
        full_logits = patched_forward(ids_A, l, p, v_full)
        full_Btop = float(full_logits[p, B_top].item())
        full_top1 = int(full_logits[p].argmax().item())

        bump_logit_delta = bump_Btop - orig_Btop
        full_logit_delta = full_Btop - orig_Btop

        if abs(full_logit_delta) < 1e-6:
            bump_attribution = None
        else:
            bump_attribution = bump_logit_delta / full_logit_delta

        records.append({
            "pair_idx": pi,
            "A_seq": A_seq,
            "B_seq": B_seq,
            "pos": p,
            "block": l,
            "A_top": A_top,
            "B_top": B_top,
            "original_top1": orig_top1,
            "bump_top1": bump_top1,
            "full_top1": full_top1,
            "orig_Btop": orig_Btop,
            "bump_Btop": bump_Btop,
            "full_Btop": full_Btop,
            "bump_logit_delta": bump_logit_delta,
            "full_logit_delta": full_logit_delta,
            "bump_attribution": bump_attribution,
        })

    if (pi + 1) % 10 == 0:
        print(f"  pair {pi + 1}/{len(pairs)}")

print(f"Total records: {len(records)} (expected {N_PAIRS * len(BLOCKS)})")


# ── Aggregate per block ──────────────────────────────────────────────────────
def summarise(vals):
    xs = [v for v in vals if v is not None and math.isfinite(v)]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "frac_gt_0.15": None, "frac_positive": None}
    t = torch.tensor(xs, dtype=torch.float64)
    mean = float(t.mean().item())
    median = float(t.median().item())
    std = float(t.std(unbiased=False).item())
    frac_gt_015 = float(((t > 0.15).sum() / n).item())
    frac_pos = float(((t > 0).sum() / n).item())
    return {"n": n, "mean": mean, "median": median, "std": std,
            "frac_gt_0.15": frac_gt_015, "frac_positive": frac_pos}


per_block_agg = {}
for l in BLOCKS:
    block_recs = [r for r in records if r["block"] == l]
    vals = [r["bump_attribution"] for r in block_recs]
    agg = summarise(vals)
    # Also record denominator validity
    full_deltas = [r["full_logit_delta"] for r in block_recs]
    n_full_pos = sum(1 for d in full_deltas if d > 0)
    n_null = sum(1 for v in vals if v is None)
    agg["n_full_positive"] = n_full_pos
    agg["n_null"] = n_null
    agg["n_total"] = len(block_recs)
    per_block_agg[l] = agg
    print(f"  block {l}: n={agg['n']}/{agg['n_total']} mean={agg['mean']} "
          f"median={agg['median']} frac>0.15={agg['frac_gt_0.15']}")

# ── Verdict (prefer median when mean/median disagree on >0.15 gate) ─────────
n_blocks_mean_above = sum(
    1 for l in BLOCKS
    if per_block_agg[l]["mean"] is not None and per_block_agg[l]["mean"] > 0.15
)
n_blocks_median_above = sum(
    1 for l in BLOCKS
    if per_block_agg[l]["median"] is not None and per_block_agg[l]["median"] > 0.15
)
# Refute: all blocks < 0.05 on the chosen statistic
mean_all_below_005 = all(
    per_block_agg[l]["mean"] is not None and per_block_agg[l]["mean"] < 0.05
    for l in BLOCKS
)
median_all_below_005 = all(
    per_block_agg[l]["median"] is not None and per_block_agg[l]["median"] < 0.05
    for l in BLOCKS
)

# Decide which statistic to trust: if mean and median disagree on any block's
# gate, prefer median (unbounded ratio tails inflate mean).
disagree = any(
    (per_block_agg[l]["mean"] is not None and per_block_agg[l]["median"] is not None) and
    ((per_block_agg[l]["mean"] > 0.15) != (per_block_agg[l]["median"] > 0.15))
    for l in BLOCKS
)
stat_used = "median" if disagree else "mean"
print(f"\nmean vs median disagreement: {disagree}  → stat_used={stat_used}")

n_above = n_blocks_median_above if stat_used == "median" else n_blocks_mean_above
all_below = median_all_below_005 if stat_used == "median" else mean_all_below_005

if n_above >= 3:
    verdict = "CONFIRMED"
elif all_below:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"
print(f"\nVerdict: {verdict}")
print(f"  n_blocks_mean_above_0.15   = {n_blocks_mean_above}/5")
print(f"  n_blocks_median_above_0.15 = {n_blocks_median_above}/5")


# ── Write JSON ───────────────────────────────────────────────────────────────
out_json = {
    "cycle": 238,
    "item": 248,
    "verdict": verdict,
    "stat_used": stat_used,
    "mean_median_disagree": disagree,
    "config": {
        "seed": SEED,
        "n_seqs": N_SEQS,
        "seq_len": SEQ_LEN,
        "blocks": BLOCKS,
        "positions_used": positions_used,
        "margin_used": margin_used,
        "n_pairs": len(pairs),
        "device": DEVICE,
    },
    "sanity": {
        "identity_max_delta": sanity_max_delta,
        "identity_ok": sanity_ok,
    },
    "pool_stats": {
        "margin_initial": MARGIN_MIN,
        "margin_final": margin_used,
        "positions_initial": POSITIONS,
        "positions_final": positions_used,
    },
    "per_block_aggregate": {str(l): per_block_agg[l] for l in BLOCKS},
    "n_blocks_mean_above_0.15": n_blocks_mean_above,
    "n_blocks_median_above_0.15": n_blocks_median_above,
    "records": records,
}
with open(OUT_JSON, "w") as f:
    json.dump(out_json, f, indent=2)
print(f"\nWrote {OUT_JSON}")


# ── Examples markdown (20 per block) ─────────────────────────────────────────
def decode_ctx(seq_idx: int, p_: int, n_chars: int = 50) -> str:
    ids = eval_ids[seq_idx, max(0, p_ - 10):p_ + 1].tolist()
    s = tokenizer.decode(ids).replace("\n", " ").replace("|", "/")
    if len(s) > n_chars:
        s = "…" + s[-n_chars:]
    return s


def decode_tok(tid: int) -> str:
    return tokenizer.decode([tid]).replace("\n", "\\n").replace("|", "/")


ex_lines = ["# Cycle 238 T4 — Activation Patching Examples (20/block)", ""]
ex_lines.append(f"Verdict: **{verdict}** | stat_used: {stat_used} | pairs: {len(pairs)}")
ex_lines.append("")
for l in BLOCKS:
    block_recs = [r for r in records if r["block"] == l][:20]
    agg = per_block_agg[l]
    ex_lines.append(f"## Block {l}")
    ex_lines.append("")
    ex_lines.append(
        f"mean={agg['mean']:.4f} median={agg['median']:.4f} std={agg['std']:.4f} "
        f"n={agg['n']}/{agg['n_total']} frac>0.15={agg['frac_gt_0.15']:.2f}"
        if agg["mean"] is not None else f"n=0 (no valid records)"
    )
    ex_lines.append("")
    for r in block_recs:
        A_ctx = decode_ctx(r["A_seq"], r["pos"])
        B_ctx = decode_ctx(r["B_seq"], r["pos"])
        A_top_s = decode_tok(r["A_top"])
        B_top_s = decode_tok(r["B_top"])
        bump_top_s = decode_tok(r["bump_top1"])
        attr = r["bump_attribution"]
        attr_s = f"{attr * 100:.1f}%" if attr is not None else "nan"
        ex_lines.append(
            f"- **pair {r['pair_idx']} p={r['pos']}**: "
            f"A ctx `{A_ctx}` → `{A_top_s}` | "
            f"B ctx `{B_ctx}` → `{B_top_s}` | "
            f"bump_top=`{bump_top_s}` | "
            f"logit on B_top: orig={r['orig_Btop']:.2f} "
            f"bump={r['bump_Btop']:.2f} full={r['full_Btop']:.2f} | "
            f"attribution={attr_s}"
        )
    ex_lines.append("")

with open(OUT_EXAMPLES, "w") as f:
    f.write("\n".join(ex_lines) + "\n")
print(f"Wrote {OUT_EXAMPLES}")


# ── Summary markdown ─────────────────────────────────────────────────────────
sum_lines = ["# Cycle 238 T4 — Bump-only Activation Patching Summary", ""]
sum_lines.append(f"**Verdict:** {verdict}")
sum_lines.append("")
sum_lines.append(
    f"**Config:** gpt2 / held-out wikitext-2 val / {N_SEQS}×{SEQ_LEN}={N_SEQS*SEQ_LEN} tokens / "
    f"blocks={BLOCKS} / n_pairs={len(pairs)} / margin_min={margin_used} / "
    f"positions={positions_used} / device {DEVICE}"
)
sum_lines.append("")
sum_lines.append(f"**stat_used:** {stat_used}  (mean/median disagree: {disagree})")
sum_lines.append("")
sum_lines.append("## Per-block bump attribution")
sum_lines.append("")
sum_lines.append("| block | n | mean | median | std | frac>0.15 | frac>0 | n_full_positive | n_null |")
sum_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for l in BLOCKS:
    a = per_block_agg[l]
    def f(v, fmt=".4f"):
        return f"{v:{fmt}}" if v is not None else "—"
    sum_lines.append(
        f"| {l} | {a['n']} | {f(a['mean'])} | {f(a['median'])} | {f(a['std'])} | "
        f"{f(a['frac_gt_0.15'], '.2f')} | {f(a['frac_positive'], '.2f')} | "
        f"{a['n_full_positive']} | {a['n_null']} |"
    )
sum_lines.append("")
sum_lines.append(
    f"**n_blocks_mean_above_0.15 = {n_blocks_mean_above}/5**  |  "
    f"**n_blocks_median_above_0.15 = {n_blocks_median_above}/5**"
)
sum_lines.append("")
sum_lines.append("## Interpretation")
sum_lines.append("")
if verdict == "CONFIRMED":
    sum_lines.append(
        f"The centred bump residual Δ̃_l alone explains > 15% of the inter-token prediction "
        f"shift at ≥3 of 5 tested blocks (by {stat_used}). Swapping only Δ̃_A → Δ̃_B in the "
        f"residual stream at position p, leaving the rest of x_l unchanged, captures a "
        f"non-trivial fraction of the full-residual patch effect on B's top-1 logit. "
        f"Combined with cycle 237's ablation-PPL catastrophe (bump subspace ≫ random 10-dim "
        f"slice) and cycle 236's large but vocabulary-misaligned logit shifts, this gives a "
        f"local per-prediction causal attribution to the bump space."
    )
elif verdict == "REFUTED":
    sum_lines.append(
        f"The centred bump residual Δ̃_l explains < 5% of the inter-token prediction shift "
        f"at ALL 5 tested blocks (by {stat_used}). Even though cycle 237 shows the bump "
        f"subspace is load-bearing at the aggregate PPL level, at the per-prediction "
        f"level the Δ̃ swap captures ~nothing of the B-token shift — the prediction "
        f"difference between tokens A and B is carried by the non-Δ̃ components of x_l."
    )
else:
    sum_lines.append(
        f"Bump attribution is non-zero but fails to pass the >0.15 gate in ≥3 blocks and "
        f"fails to fall below 0.05 in all 5 blocks (by {stat_used}). The bump carries some "
        f"per-prediction signal but not enough of the inter-token logit difference to "
        f"support a clean positive attribution at the tested blocks."
    )
sum_lines.append("")
sum_lines.append("## Sanity")
sum_lines.append("")
sum_lines.append(f"- Identity patch max |Δ logit| = {sanity_max_delta:.2e} (ok={sanity_ok})")
sum_lines.append(f"- Pool margin final = {margin_used}, positions final = {positions_used}")
sum_lines.append(f"- Pairs used = {len(pairs)} / target {N_PAIRS}")
sum_lines.append("")

with open(OUT_SUMMARY, "w") as f:
    f.write("\n".join(sum_lines) + "\n")
print(f"Wrote {OUT_SUMMARY}")

print(f"\nDone. verdict={verdict} stat={stat_used} "
      f"n_above_mean={n_blocks_mean_above}/5 n_above_median={n_blocks_median_above}/5")
