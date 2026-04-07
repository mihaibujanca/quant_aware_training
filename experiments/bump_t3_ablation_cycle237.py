"""
Cycle 237 — Item 247 (T3): Bump subspace ablation vs random subspace baseline.

For each GPT-2 block l, define the top-10 bump subspace projector
P_bump_l = V_l[:, :10] @ V_l[:, :10].T using V_l from cycle 235's T1 features
(the top right singular vectors of centred Δ̃_l on wikitext-2 train 200×512).

After each block's forward, ablate the residual stream:
    x_l ← x_l − x_l @ P_l

Measure full-model PPL on held-out wikitext-2 val (200 seqs × 512 tokens)
under five conditions:
  1) baseline (no hooks)
  2) joint bump ablation (all 12 blocks)
  3) per-block bump ablation (12 single-block runs)
  4) joint + per-block random k=10 subspace baseline (3 seeds)
  5) joint + per-block full-FFN top-10 right singular vectors baseline

Verdict gates (item 247):
  CONFIRMED    := joint_bump_ppl >= 1.5 * joint_random_mean_ppl AND
                  >=3 of 12 per_block_bump_ppl[l] >= 2.0 * per_block_random_mean[l]
  REFUTED      := joint_bump_ppl <= 1.1 * joint_random_mean_ppl
  INCONCLUSIVE := otherwise
"""
import json
import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
N_SEQS = 200
SEQ_LEN = 512
BATCH = 4
N_BLOCKS = 12
K_BUMP = 10
RANDOM_SEEDS = [42, 137, 2024]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CPU = torch.device("cpu")

RESULTS_DIR = "autoresearch/results"
FEATURES_PATH = f"{RESULTS_DIR}/cycle_235_t1_features.pt"
OUT_JSON = f"{RESULTS_DIR}/cycle_237_t3_ablation_ppl.json"
OUT_SUMMARY_MD = f"{RESULTS_DIR}/cycle_237_t3_summary.md"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load model + tokenizer ───────────────────────────────────────────────────
print("Loading GPT-2 and tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# ── Load cycle 235 bump features → P_bump_l ──────────────────────────────────
print("Loading cycle 235 T1 features...")
features = torch.load(FEATURES_PATH, weights_only=False)

V_bump = {}
P_bump = {}
for l in range(N_BLOCKS):
    V_full = features[f"block_{l}"]["V"]  # (768, 32)
    V_l = V_full[:, :K_BUMP].contiguous().to(DEVICE).float()  # (768, 10)
    V_bump[l] = V_l
    P_bump[l] = (V_l @ V_l.T).contiguous()  # (768, 768)

# One-time orthonormality check on block 0.
ortho_err = (V_bump[0].T @ V_bump[0] - torch.eye(K_BUMP, device=DEVICE)).abs().max().item()
print(f"Orthonormality check V_0.T @ V_0 - I max abs = {ortho_err:.2e}")
assert ortho_err < 1e-3, f"V_0 is not orthonormal: {ortho_err}"

# Projector idempotency on block 0: P @ P ≈ P.
idem_err = (P_bump[0] @ P_bump[0] - P_bump[0]).abs().max().item()
print(f"Idempotency check P_0 @ P_0 - P_0 max abs = {idem_err:.2e}")
assert idem_err < 1e-3, f"P_bump[0] not idempotent: {idem_err}"

# ── Tokenise held-out wikitext-2 val split ───────────────────────────────────
print(f"\nLoading wikitext-2 val split (seqs={N_SEQS}, seq_len={SEQ_LEN})...")
ds_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts_val = [r["text"] for r in ds_val if r["text"].strip()]
tokens_needed = N_SEQS * SEQ_LEN
val_tokens = []
for t in texts_val:
    ids = tokenizer(t, return_tensors="pt", truncation=False)["input_ids"][0]
    val_tokens.append(ids)
flat_val = torch.cat(val_tokens, dim=0)
if flat_val.numel() < tokens_needed:
    raise RuntimeError(
        f"Need {tokens_needed} val tokens but only have {flat_val.numel()}"
    )
flat_val = flat_val[:tokens_needed].contiguous()
eval_ids = flat_val.view(N_SEQS, SEQ_LEN).contiguous()
print(f"  eval_ids shape: {tuple(eval_ids.shape)}")


# ── eval_ppl helper ──────────────────────────────────────────────────────────
@torch.no_grad()
def eval_ppl(model_, token_ids: torch.Tensor, batch: int = BATCH) -> float:
    """CE loss over next-token prediction → exp = PPL."""
    model_.eval()
    total_loss = 0.0
    total_tokens = 0
    n = token_ids.shape[0]
    for i in range(0, n, batch):
        ids = token_ids[i:i + batch].to(DEVICE)
        logits = model_(ids).logits  # (B, T, V)
        # shift: predict token t+1 from logits at position t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_tokens += shift_labels.numel()
    mean_loss = total_loss / total_tokens
    return math.exp(mean_loss)


# ── Hook factory: ablate residual stream via projector P ─────────────────────
def make_ablation_hook(P: torch.Tensor):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            new_hs = hs - hs @ P
            return (new_hs,) + output[1:]
        else:
            return output - output @ P
    return hook


def install_hooks(P_per_block: dict) -> list:
    handles = []
    for l, P in P_per_block.items():
        h = model.transformer.h[l].register_forward_hook(make_ablation_hook(P))
        handles.append(h)
    return handles


def remove_hooks(handles: list):
    for h in handles:
        h.remove()


# ── 1. Baseline PPL (no hooks) ───────────────────────────────────────────────
print("\n[1] Baseline PPL (no ablation)...")
baseline_ppl = eval_ppl(model, eval_ids)
print(f"  baseline_ppl = {baseline_ppl:.4f}")

# ── 2. Joint bump ablation ───────────────────────────────────────────────────
print("\n[2] Joint bump ablation (all 12 blocks)...")
handles = install_hooks(P_bump)
joint_bump_ppl = eval_ppl(model, eval_ids)
remove_hooks(handles)
print(f"  joint_bump_ppl = {joint_bump_ppl:.4f}")

# ── 3. Per-block bump ablation ───────────────────────────────────────────────
print("\n[3] Per-block bump ablation (12 single-block runs)...")
per_block_bump_ppl = []
for l in range(N_BLOCKS):
    handles = install_hooks({l: P_bump[l]})
    ppl = eval_ppl(model, eval_ids)
    remove_hooks(handles)
    per_block_bump_ppl.append(ppl)
    print(f"  block {l}: ppl = {ppl:.4f}")

# ── 4. Random subspace baseline (3 seeds × joint + 12 per-block) ─────────────
print("\n[4] Random subspace baseline (3 seeds)...")
joint_random_ppl_per_seed = {}
per_block_random_ppl = {l: [] for l in range(N_BLOCKS)}
for seed in RANDOM_SEEDS:
    print(f"\n  --- seed {seed} ---")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    P_rand_l = {}
    for l in range(N_BLOCKS):
        G = torch.randn(768, K_BUMP, generator=gen)
        Q, _ = torch.linalg.qr(G)  # (768, 10) orthonormal columns
        Q = Q.to(DEVICE).float().contiguous()
        P_rand_l[l] = (Q @ Q.T).contiguous()

    # joint: all 12 installed
    handles = install_hooks(P_rand_l)
    joint_ppl = eval_ppl(model, eval_ids)
    remove_hooks(handles)
    joint_random_ppl_per_seed[seed] = joint_ppl
    print(f"  seed {seed} joint random ppl = {joint_ppl:.4f}")

    # per-block
    for l in range(N_BLOCKS):
        handles = install_hooks({l: P_rand_l[l]})
        ppl = eval_ppl(model, eval_ids)
        remove_hooks(handles)
        per_block_random_ppl[l].append(ppl)
        print(f"    seed {seed} block {l}: ppl = {ppl:.4f}")

joint_random_vals = [joint_random_ppl_per_seed[s] for s in RANDOM_SEEDS]
joint_random_mean = sum(joint_random_vals) / len(joint_random_vals)
joint_random_std = (
    sum((v - joint_random_mean) ** 2 for v in joint_random_vals) / len(joint_random_vals)
) ** 0.5

per_block_random_mean = []
per_block_random_std = []
for l in range(N_BLOCKS):
    vals = per_block_random_ppl[l]
    m = sum(vals) / len(vals)
    s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    per_block_random_mean.append(m)
    per_block_random_std.append(s)

print(f"\n  joint random mean ± std = {joint_random_mean:.4f} ± {joint_random_std:.4f}")

# ── 5. Full-FFN top-10 SVs baseline ──────────────────────────────────────────
print("\n[5] Full-FFN top-10 right SVs (calibration)...")

print("  Loading wikitext-2 train for calibration...")
ds_calib = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts_calib = [r["text"] for r in ds_calib if r["text"].strip()]
calib_tokens = []
tokens_needed_calib = N_SEQS * SEQ_LEN
for t in texts_calib:
    ids = tokenizer(t, return_tensors="pt", truncation=False)["input_ids"][0]
    calib_tokens.append(ids)
    running = sum(x.numel() for x in calib_tokens)
    if running >= tokens_needed_calib * 2:
        break
flat_calib = torch.cat(calib_tokens, dim=0)
if flat_calib.numel() < tokens_needed_calib:
    raise RuntimeError(
        f"Need {tokens_needed_calib} calib tokens but only {flat_calib.numel()}"
    )
flat_calib = flat_calib[:tokens_needed_calib].contiguous()
calib_ids = flat_calib.view(N_SEQS, SEQ_LEN).contiguous()
print(f"  calib_ids shape: {tuple(calib_ids.shape)}")

mlp_out_stash = {l: [] for l in range(N_BLOCKS)}


def make_mlp_capture_hook(l):
    def fn(module, inputs, output):
        # output of block l's mlp is (B, T, 768) tensor (post c_proj).
        mlp_out_stash[l].append(output.detach().float().cpu())
    return fn


handles = []
for l in range(N_BLOCKS):
    h = model.transformer.h[l].mlp.register_forward_hook(make_mlp_capture_hook(l))
    handles.append(h)

print("  forward pass for FFN output capture...")
with torch.no_grad():
    for i in range(0, N_SEQS, BATCH):
        batch = calib_ids[i:i + BATCH].to(DEVICE)
        _ = model(batch)

for h in handles:
    h.remove()

P_full = {}
V_full_per_block = {}
for l in range(N_BLOCKS):
    ffn_out = torch.cat(mlp_out_stash[l], dim=0).reshape(-1, 768)  # (n_tokens, 768)
    assert ffn_out.shape == (N_SEQS * SEQ_LEN, 768), (l, ffn_out.shape)
    # SVD on CPU for stability.
    _, _, Vh = torch.linalg.svd(ffn_out, full_matrices=False)
    V_full_l = Vh[:K_BUMP].T.contiguous()  # (768, 10)
    V_full_per_block[l] = V_full_l
    V_full_dev = V_full_l.to(DEVICE).float()
    P_full[l] = (V_full_dev @ V_full_dev.T).contiguous()
    print(f"  block {l}: full-FFN V shape {tuple(V_full_l.shape)}")
    mlp_out_stash[l] = None  # free
mlp_out_stash = None

# Joint full-FFN ablation
print("\n[5a] Joint full-FFN ablation...")
handles = install_hooks(P_full)
joint_full_ppl = eval_ppl(model, eval_ids)
remove_hooks(handles)
print(f"  joint_full_ppl = {joint_full_ppl:.4f}")

# Per-block full-FFN
print("\n[5b] Per-block full-FFN ablation...")
per_block_full_ppl = []
for l in range(N_BLOCKS):
    handles = install_hooks({l: P_full[l]})
    ppl = eval_ppl(model, eval_ids)
    remove_hooks(handles)
    per_block_full_ppl.append(ppl)
    print(f"  block {l}: full-FFN ppl = {ppl:.4f}")

# ── Restore sanity: baseline after all hooks removed ─────────────────────────
print("\nSanity: rerunning baseline after all hooks removed...")
baseline_final = eval_ppl(model, eval_ids)
print(f"  baseline_final = {baseline_final:.4f} (initial {baseline_ppl:.4f})")
restore_delta = abs(baseline_final - baseline_ppl)
print(f"  restore delta = {restore_delta:.2e}")
assert restore_delta < 1e-4, f"Restore sanity failed: {restore_delta}"

# ── Verdict ──────────────────────────────────────────────────────────────────
print("\n── Verdict ──────────────────────────────────────────────────────")
ratio_joint = joint_bump_ppl / joint_random_mean if joint_random_mean > 0 else float("inf")
per_block_ratio = [
    per_block_bump_ppl[l] / per_block_random_mean[l] if per_block_random_mean[l] > 0 else float("inf")
    for l in range(N_BLOCKS)
]
n_blocks_above_2x = sum(1 for r in per_block_ratio if r >= 2.0)

print(f"  joint_bump_ppl         = {joint_bump_ppl:.4f}")
print(f"  joint_random_mean      = {joint_random_mean:.4f}")
print(f"  ratio joint            = {ratio_joint:.4f}")
print(f"  per_block ratios       = {[round(r, 3) for r in per_block_ratio]}")
print(f"  n_blocks_above_2x_rand = {n_blocks_above_2x}")

if ratio_joint >= 1.5 and n_blocks_above_2x >= 3:
    verdict = "CONFIRMED"
elif ratio_joint <= 1.1:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n  VERDICT = {verdict}")

# ── Write JSON ───────────────────────────────────────────────────────────────
output = {
    "cycle": 237,
    "item": 247,
    "verdict": verdict,
    "config": {
        "seed": SEED,
        "n_seqs": N_SEQS,
        "seq_len": SEQ_LEN,
        "batch": BATCH,
        "n_blocks": N_BLOCKS,
        "k_bump": K_BUMP,
        "random_seeds": RANDOM_SEEDS,
        "device": DEVICE,
    },
    "baseline_ppl": baseline_ppl,
    "baseline_ppl_after_restore": baseline_final,
    "bump_ablation": {
        "joint": joint_bump_ppl,
        "per_block": per_block_bump_ppl,
    },
    "random_ablation": {
        "joint_per_seed": joint_random_ppl_per_seed,
        "joint_mean": joint_random_mean,
        "joint_std": joint_random_std,
        "per_block_per_seed": {
            str(l): per_block_random_ppl[l] for l in range(N_BLOCKS)
        },
        "per_block_mean": per_block_random_mean,
        "per_block_std": per_block_random_std,
    },
    "full_ffn_ablation": {
        "joint": joint_full_ppl,
        "per_block": per_block_full_ppl,
    },
    "ratios": {
        "joint_bump_vs_random": ratio_joint,
        "per_block_bump_vs_random": per_block_ratio,
    },
    "n_blocks_above_2x_random": n_blocks_above_2x,
    "sanity": {
        "ortho_err_V0": ortho_err,
        "idem_err_P0": idem_err,
        "restore_delta": restore_delta,
    },
}

# Convert any torch tensors / non-serializable types.
def _to_json(obj):
    if isinstance(obj, dict):
        return {str(k): _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj

with open(OUT_JSON, "w") as f:
    json.dump(_to_json(output), f, indent=2)
print(f"\nWrote {OUT_JSON}")

# ── Summary markdown ─────────────────────────────────────────────────────────
lines = ["# Cycle 237 T3 — Bump Subspace Ablation Summary", ""]
lines.append(f"**Verdict:** {verdict}")
lines.append("")
lines.append(
    f"**Config:** gpt2 / held-out wikitext-2 val / {N_SEQS}×{SEQ_LEN}="
    f"{N_SEQS*SEQ_LEN} tokens / k_bump={K_BUMP} / random seeds {RANDOM_SEEDS} / "
    f"device {DEVICE}"
)
lines.append("")
lines.append("## Joint ablation (all 12 blocks)")
lines.append("")
lines.append("| condition | PPL | ratio vs baseline | ratio vs random |")
lines.append("|---|---:|---:|---:|")
lines.append(f"| baseline | {baseline_ppl:.3f} | 1.000 | — |")
lines.append(
    f"| bump k={K_BUMP} | {joint_bump_ppl:.3f} | "
    f"{joint_bump_ppl/baseline_ppl:.3f} | {ratio_joint:.3f} |"
)
lines.append(
    f"| random k={K_BUMP} (mean±std, n={len(RANDOM_SEEDS)}) | "
    f"{joint_random_mean:.3f} ± {joint_random_std:.3f} | "
    f"{joint_random_mean/baseline_ppl:.3f} | 1.000 |"
)
lines.append(
    f"| full-FFN top-{K_BUMP} | {joint_full_ppl:.3f} | "
    f"{joint_full_ppl/baseline_ppl:.3f} | "
    f"{joint_full_ppl/joint_random_mean:.3f} |"
)
lines.append("")
lines.append("## Per-block single ablation")
lines.append("")
lines.append(
    "| block | bump | random mean±std | full-FFN | bump/random | above 2× |"
)
lines.append("|---:|---:|---:|---:|---:|---:|")
for l in range(N_BLOCKS):
    mark = "YES" if per_block_ratio[l] >= 2.0 else ""
    lines.append(
        f"| {l} | {per_block_bump_ppl[l]:.3f} | "
        f"{per_block_random_mean[l]:.3f} ± {per_block_random_std[l]:.3f} | "
        f"{per_block_full_ppl[l]:.3f} | {per_block_ratio[l]:.3f} | {mark} |"
    )
lines.append("")
lines.append(
    f"**n_blocks_above_2x_random = {n_blocks_above_2x} / 12**"
)
lines.append("")
lines.append("## Interpretation")
lines.append("")
if verdict == "CONFIRMED":
    lines.append(
        f"The top-{K_BUMP} centred-Δ̃ bump subspace is significantly more "
        f"load-bearing for GPT-2's residual stream than a random k={K_BUMP} subspace: "
        f"joint ablation inflates PPL by {ratio_joint:.2f}× the random baseline, and "
        f"{n_blocks_above_2x}/12 single-block ablations reach ≥2× their random "
        f"control. The bump directions are not compression slack — ablating "
        f"them degrades held-out LM performance above chance."
    )
elif verdict == "REFUTED":
    lines.append(
        f"The top-{K_BUMP} bump subspace is indistinguishable from a random "
        f"k={K_BUMP} direction at the aggregate PPL level (joint ratio "
        f"{ratio_joint:.3f} ≤ 1.1). Removing the bump dimensions costs about as "
        f"much as removing any random 10-dim slice of the residual stream — the "
        f"bump space is NOT preferentially compression-relevant. This contradicts "
        f"a 'bump carries critical information' story at the PPL-ablation level."
    )
else:
    lines.append(
        f"Joint-bump PPL ratio against random baseline is {ratio_joint:.3f} "
        f"(CONFIRMED gate ≥1.5, REFUTED gate ≤1.1). Per-block single ablations "
        f"exceed 2× random in {n_blocks_above_2x}/12 blocks (CONFIRMED gate ≥3). "
        f"The effect is neither distinguishable nor null — the bump subspace "
        f"carries some marginal compression load but not enough to clear the "
        f"preregistered gate."
    )
lines.append("")
lines.append("## Sanity")
lines.append("")
lines.append(f"- V_0.T @ V_0 - I max |·| = {ortho_err:.2e}")
lines.append(f"- P_0 @ P_0 - P_0 max |·| = {idem_err:.2e}")
lines.append(
    f"- Baseline before/after hooks: {baseline_ppl:.6f} / {baseline_final:.6f} "
    f"(|Δ|={restore_delta:.2e})"
)
lines.append("")

with open(OUT_SUMMARY_MD, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {OUT_SUMMARY_MD}")

print(f"\nDone. verdict={verdict} joint_bump={joint_bump_ppl:.3f} "
      f"joint_random={joint_random_mean:.3f} ratio={ratio_joint:.3f} "
      f"n_blocks_above_2x={n_blocks_above_2x}/12")
