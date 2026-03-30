"""
Cycle 118: Train tiny Mixtral MoE on WikiText-2, REAP expert saliency, pruning baseline.

Steps:
1. Train BPE tokenizer (vocab=4096) on WikiText-2 train
2. Train tiny Mixtral (15.5M params) for ~10M tokens on CPU
3. REAP saliency scoring of experts
4. Pruning evaluation at 25%, 50%, 75% with random baseline (3 seeds)
5. Per-layer ablation
"""

import os, sys, json, time, math, random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    MixtralConfig, MixtralForCausalLM,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("autoresearch/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXPR_RESULTS_DIR = Path("experiments/results")
EXPR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TOK_DIR = EXPR_RESULTS_DIR / "cycle_118_tokenizer"
TOK_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = EXPR_RESULTS_DIR / "cycle_118_tiny_mixtral.pt"
JSON_PATH = RESULTS_DIR / "cycle_118_moe_reap.json"
PNG_PATH  = RESULTS_DIR / "cycle_118_moe_reap.png"

SEQ_LEN    = 256
VOCAB_SIZE = 4096
BATCH_SIZE = 4
LR         = 1e-4
TARGET_TOKENS = 10_000_000
MAX_TOKENS    = 20_000_000
DEVICE        = "cpu"
# If checkpoint exists AND tokenizer exists, skip training
RESUME_FROM_CKPT = CKPT_PATH.exists() and (TOK_DIR / "tokenizer.json").exists()
torch.manual_seed(42)

# ─── 1. Load dataset ────────────────────────────────────────────────────────
print("Loading WikiText-2 ...", flush=True)
ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
ds_val   = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

train_texts = [x for x in ds_train["text"] if x.strip()]
val_texts   = [x for x in ds_val["text"]   if x.strip()]

# ─── 2. Train BPE tokenizer ─────────────────────────────────────────────────
if not RESUME_FROM_CKPT:
    print("Training BPE tokenizer ...", flush=True)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<pad>", "<eos>", "<unk>"],
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    tokenizer.save(str(TOK_DIR / "tokenizer.json"))
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
else:
    print("Loading existing tokenizer ...", flush=True)
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(TOK_DIR / "tokenizer.json"),
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
PAD_ID = hf_tok.pad_token_id
EOS_ID = hf_tok.eos_token_id

# ─── 3. Pack sequences ──────────────────────────────────────────────────────
def pack_texts(texts, seq_len):
    """Concatenate all token ids, chunk into seq_len blocks (drop last short block)."""
    all_ids = []
    for t in texts:
        ids = hf_tok.encode(t)
        all_ids.extend(ids)
        all_ids.append(EOS_ID)
    all_ids = torch.tensor(all_ids, dtype=torch.long)
    n_blocks = len(all_ids) // seq_len
    blocks = all_ids[: n_blocks * seq_len].view(n_blocks, seq_len)
    return blocks

print("Packing train tokens ...", flush=True)
train_blocks = pack_texts(train_texts, SEQ_LEN)
val_blocks   = pack_texts(val_texts,   SEQ_LEN)
print(f"  train blocks: {len(train_blocks)}, val blocks: {len(val_blocks)}", flush=True)

# ─── 4. Model ────────────────────────────────────────────────────────────────
config = MixtralConfig(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    num_local_experts=8,
    num_experts_per_tok=2,
    router_aux_loss_coef=0.001,
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=SEQ_LEN,
    sliding_window=None,
)
model = MixtralForCausalLM(config).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params/1e6:.1f}M", flush=True)

# ─── 5. Training ─────────────────────────────────────────────────────────────
if not RESUME_FROM_CKPT:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    tokens_per_step = BATCH_SIZE * SEQ_LEN
    steps_target = TARGET_TOKENS // tokens_per_step
    steps_max    = MAX_TOKENS    // tokens_per_step

    # Cosine schedule with 5% warmup
    n_warmup = max(100, steps_target // 20)
    scheduler = get_cosine_schedule_with_warmup(optimizer, n_warmup, steps_target)

    n_train = len(train_blocks)

    def get_batch(step):
        idxs = torch.randint(0, n_train, (BATCH_SIZE,))
        return train_blocks[idxs].to(DEVICE)

    print(f"Training for {steps_target} steps (~{TARGET_TOKENS/1e6:.0f}M tokens) ...", flush=True)
    t0 = time.time()
    model.train()
    train_losses = []

    for step in range(1, steps_max + 1):
        x = get_batch(step)
        out = model(input_ids=x, labels=x)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            elapsed = time.time() - t0
            tok_done = step * tokens_per_step
            print(f"  step {step:5d} | loss {loss.item():.3f} | tokens {tok_done/1e6:.1f}M | {elapsed:.0f}s", flush=True)
            train_losses.append((step, loss.item()))

        if step == steps_target:
            # Evaluate val PPL
            model.eval()
            with torch.no_grad():
                val_losses = []
                for i in range(0, min(256, len(val_blocks)), BATCH_SIZE):
                    batch = val_blocks[i:i+BATCH_SIZE].to(DEVICE)
                    out = model(input_ids=batch, labels=batch)
                    val_losses.append(out.loss.item())
                val_ppl = math.exp(sum(val_losses) / len(val_losses))
            print(f"\nVal PPL at {step} steps: {val_ppl:.2f}", flush=True)
            model.train()

            if val_ppl < 50:
                print("Val PPL < 50, stopping.", flush=True)
                break
            elif step >= steps_max:
                print(f"Warning: val PPL={val_ppl:.2f} > 50 after {MAX_TOKENS/1e6:.0f}M tokens", flush=True)
                break
            else:
                print("Val PPL >= 50, extending to 20M tokens ...", flush=True)

    # Final val eval
    model.eval()
    with torch.no_grad():
        val_losses = []
        for i in range(0, min(256*BATCH_SIZE, len(val_blocks)), BATCH_SIZE):
            batch = val_blocks[i:i+BATCH_SIZE].to(DEVICE)
            out = model(input_ids=batch, labels=batch)
            val_losses.append(out.loss.item())
        baseline_ppl = math.exp(sum(val_losses) / len(val_losses))
    print(f"Final baseline val PPL: {baseline_ppl:.4f}", flush=True)

    # Save checkpoint
    torch.save({"model_state": model.state_dict(), "config": config, "val_ppl": baseline_ppl}, CKPT_PATH)
    print(f"Checkpoint saved to {CKPT_PATH}", flush=True)
    train_losses_to_save = train_losses

else:
    print(f"Resuming from checkpoint: {CKPT_PATH}", flush=True)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    baseline_ppl = ckpt["val_ppl"]
    train_losses_to_save = []
    model.eval()
    print(f"Loaded checkpoint. Saved val PPL: {baseline_ppl:.4f}", flush=True)
    # Re-verify baseline PPL on val subset
    val_blocks_check = pack_texts(val_texts, SEQ_LEN)
    with torch.no_grad():
        val_losses = []
        for i in range(0, min(256*BATCH_SIZE, len(val_blocks_check)), BATCH_SIZE):
            batch = val_blocks_check[i:i+BATCH_SIZE].to(DEVICE)
            out = model(input_ids=batch, labels=batch)
            val_losses.append(out.loss.item())
        baseline_ppl = math.exp(sum(val_losses) / len(val_losses))
    val_blocks = val_blocks_check
    print(f"Recomputed baseline val PPL: {baseline_ppl:.4f}", flush=True)

# ─── 6. Router entropy check ─────────────────────────────────────────────────
print("Checking router entropy ...", flush=True)
# Use 256 val sequences (packs of BATCH_SIZE)
n_val_seqs = min(256, len(val_blocks))
val_subset = val_blocks[:n_val_seqs].to(DEVICE)

# Collect router logits per layer
# MixtralForCausalLM supports output_router_logits=True
all_router_logits = []  # list of [n_layers] each (n_tokens, n_experts)
with torch.no_grad():
    for i in range(0, n_val_seqs, BATCH_SIZE):
        batch = val_subset[i:i+BATCH_SIZE]
        out = model(input_ids=batch, output_router_logits=True)
        # out.router_logits: tuple of (batch*seq_len, n_experts) per layer
        if i == 0:
            all_router_logits = [[] for _ in range(len(out.router_logits))]
        for l, rl in enumerate(out.router_logits):
            all_router_logits[l].append(rl.cpu())

# Concatenate
layer_router_logits = []
for l in range(len(all_router_logits)):
    layer_router_logits.append(torch.cat(all_router_logits[l], dim=0))  # (T, n_experts)

n_layers  = len(layer_router_logits)
n_experts = layer_router_logits[0].shape[1]

router_entropies = []
for l in range(n_layers):
    # NOTE: MixtralTopKRouter already applies softmax, so layer_router_logits[l] is already a probability distribution
    mean_p = layer_router_logits[l].mean(dim=0)  # (E,) — mean gate distribution
    H = -(mean_p * torch.log2(mean_p + 1e-10)).sum().item()
    router_entropies.append(H)
    flag = " [DEGENERATE]" if H < 1.0 else ""
    print(f"  Layer {l}: router entropy H = {H:.3f} bits{flag}", flush=True)

# ─── 7. REAP saliency scoring ─────────────────────────────────────────────────
print("Computing REAP saliency scores ...", flush=True)

# In this transformers version (5.2.0), MixtralExperts stores weights as:
#   experts.gate_up_proj: (n_experts, intermediate_dim*2, hidden_dim)
#   experts.down_proj:    (n_experts, hidden_dim, intermediate_dim)
# No individual expert submodules — we compute per-expert outputs manually.
# MixtralTopKRouter.forward returns (router_logits, router_scores, router_indices)
#   where router_logits is already softmaxed.

def compute_reap_saliency(model, val_subset):
    scores = torch.zeros(n_layers, n_experts)
    total_tokens = 0

    for i in range(0, len(val_subset), BATCH_SIZE):
        batch = val_subset[i:i+BATCH_SIZE].to(DEVICE)
        B, T = batch.shape

        with torch.no_grad():
            out = model(input_ids=batch, output_router_logits=True)
        # out.router_logits: tuple of (B*T, E), already softmaxed
        rl_per_layer = out.router_logits

        # Get intermediate hidden states from each layer's input by hooking the MoE block
        layer_hidden_inputs = {}
        hooks = []
        def make_moe_input_hook(layer_idx):
            def hook(module, inp, outp):
                # inp[0]: (B, T, hidden_size) — input to MoE block
                layer_hidden_inputs[layer_idx] = inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu()
            return hook

        for l in range(n_layers):
            h = model.model.layers[l].mlp.register_forward_hook(make_moe_input_hook(l))
            hooks.append(h)

        with torch.no_grad():
            model(input_ids=batch)

        for h in hooks:
            h.remove()

        # For each layer, compute per-expert outputs manually
        for l in range(n_layers):
            # router_logits for this layer: (B*T, E), already softmaxed
            gate_softmax = rl_per_layer[l].cpu()  # (B*T, E)
            # top-2 indices
            top2_vals, top2_idxs = torch.topk(gate_softmax, 2, dim=-1)  # (B*T, 2)

            hidden = layer_hidden_inputs[l]  # (B*T, hidden_size)
            experts_mod = model.model.layers[l].mlp.experts
            # gate_up_proj: (n_experts, intermediate_dim*2, hidden_dim)
            # down_proj:    (n_experts, hidden_dim, intermediate_dim)
            gate_up = experts_mod.gate_up_proj.cpu()  # (E, 2*I, H)
            down    = experts_mod.down_proj.cpu()      # (E, H, I)

            for k in range(n_experts):
                # Tokens routed to expert k
                mask = (top2_idxs == k).any(dim=-1)  # (B*T,)
                if not mask.any():
                    continue
                h_k = hidden[mask]  # (n_routed, H)
                # Compute expert k output: SiLU-gated FFN
                # gate_up[k]: (2*I, H), gate = first I rows, up = last I rows
                gu = h_k @ gate_up[k].t()  # (n_routed, 2*I)
                I = gu.shape[1] // 2
                gate_part, up_part = gu[:, :I], gu[:, I:]
                act = F.silu(gate_part) * up_part   # (n_routed, I)
                out_k = act @ down[k].t()            # (n_routed, H)
                norms = out_k.norm(dim=-1)           # (n_routed,)
                # gate weight for expert k at routed tokens
                gate_weights = gate_softmax[mask, k]  # (n_routed,)
                scores[l, k] += (gate_weights * norms).sum().item()

        total_tokens += B * T

    scores = scores / total_tokens
    return scores

saliency_scores = compute_reap_saliency(model, val_subset)
print("Saliency scores (layer x expert):")
print(saliency_scores)

def gini(x):
    """Gini coefficient for a 1D tensor."""
    x = x.float().sort()[0]
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = torch.arange(1, n+1, dtype=torch.float)
    return ((2 * idx - n - 1) * x).sum().item() / (n * x.sum().item())

layer_ginis = []
for l in range(n_layers):
    g = gini(saliency_scores[l])
    layer_ginis.append(g)
    print(f"  Layer {l} Gini: {g:.4f}", flush=True)

# Top-4 vs bottom-4 spread
top4_bottom4_ratio = []
for l in range(n_layers):
    s = saliency_scores[l].sort()[0]
    top4 = s[-4:].sum().item()
    bot4 = s[:4].sum().item()
    ratio = top4 / (bot4 + 1e-10)
    top4_bottom4_ratio.append(ratio)
    print(f"  Layer {l} top4/bot4 spread: {ratio:.2f}", flush=True)

# Cross-layer Spearman correlation
from scipy.stats import spearmanr
spearman_mat = np.zeros((n_layers, n_layers))
s_np = saliency_scores.numpy()
for i in range(n_layers):
    for j in range(n_layers):
        r, _ = spearmanr(s_np[i], s_np[j])
        spearman_mat[i, j] = r
print("Cross-layer Spearman correlation matrix:")
print(spearman_mat.round(3))

# ─── 8. Pruning evaluation ────────────────────────────────────────────────────
print("\nPruning evaluation ...", flush=True)

# Global ranking: flatten (layer, expert) saliency scores, sort ascending
flat_saliency = saliency_scores.view(-1)  # (n_layers * n_experts,)
n_total_experts = n_layers * n_experts

def eval_ppl_with_pruning(model, val_subset, pruned_set):
    """
    pruned_set: set of (layer_idx, expert_idx) to prune.
    Zeroes out router probs for pruned experts (renormalize remaining).
    MixtralTopKRouter.forward returns (router_logits, router_scores, router_indices)
    where router_logits is already softmaxed. We hook the gate output and zero
    pruned expert probabilities, then renormalize.
    """
    hooks = []

    def make_gate_hook(layer_idx):
        pruned_experts_in_layer = [e for (ll, e) in pruned_set if ll == layer_idx]
        def hook(module, inp, outp):
            # outp: (router_logits, router_scores, router_indices)
            #   router_logits: (B*T, E) softmaxed full distribution
            #   router_scores: (B*T, top_k) renormalized top-k weights
            #   router_indices: (B*T, top_k) top-k indices
            if not pruned_experts_in_layer:
                return outp
            router_logits, router_scores, router_indices = outp
            router_logits = router_logits.clone()
            router_logits[:, pruned_experts_in_layer] = 0.0
            # Renormalize
            row_sum = router_logits.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            router_logits = router_logits / row_sum
            # Recompute top-k from modified logits
            top_k = router_scores.shape[-1]
            new_scores, new_indices = torch.topk(router_logits, top_k, dim=-1)
            new_scores = new_scores / new_scores.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            return (router_logits, new_scores, new_indices)
        return hook

    for l in range(n_layers):
        gate = model.model.layers[l].mlp.gate
        h = gate.register_forward_hook(make_gate_hook(l))
        hooks.append(h)

    losses = []
    with torch.no_grad():
        for i in range(0, len(val_subset), BATCH_SIZE):
            batch = val_subset[i:i+BATCH_SIZE].to(DEVICE)
            out = model(input_ids=batch, labels=batch)
            losses.append(out.loss.item())

    for h in hooks:
        h.remove()

    return math.exp(sum(losses) / len(losses))

# REAP pruning at each level
prune_levels = [0.25, 0.50, 0.75]
sorted_idx = flat_saliency.argsort()  # ascending (lowest first = to prune)

reap_ppls = {}
for frac in prune_levels:
    n_prune = int(frac * n_total_experts)
    pruned_flat = sorted_idx[:n_prune].tolist()
    pruned_set = {(idx // n_experts, idx % n_experts) for idx in pruned_flat}
    ppl = eval_ppl_with_pruning(model, val_subset, pruned_set)
    reap_ppls[frac] = ppl
    print(f"  REAP {int(frac*100)}% pruned: PPL = {ppl:.4f}", flush=True)

# Random baseline: 3 seeds per level
RANDOM_SEEDS = [42, 137, 2024]
random_ppls = {}  # frac -> list of PPLs
for frac in prune_levels:
    n_prune = int(frac * n_total_experts)
    seed_ppls = []
    for seed in RANDOM_SEEDS:
        rng = random.Random(seed)
        all_idx = list(range(n_total_experts))
        rng.shuffle(all_idx)
        pruned_flat = all_idx[:n_prune]
        pruned_set = {(idx // n_experts, idx % n_experts) for idx in pruned_flat}
        ppl = eval_ppl_with_pruning(model, val_subset, pruned_set)
        seed_ppls.append(ppl)
    random_ppls[frac] = seed_ppls
    mean_r = sum(seed_ppls) / len(seed_ppls)
    std_r  = (sum((x - mean_r)**2 for x in seed_ppls) / len(seed_ppls)) ** 0.5
    print(f"  Random {int(frac*100)}% pruned: PPL = {mean_r:.4f} ± {std_r:.4f} (seeds {RANDOM_SEEDS})", flush=True)

# ─── 9. Per-layer ablation ────────────────────────────────────────────────────
print("\nPer-layer ablation (50% pruning, one layer at a time) ...", flush=True)
layer_ablation_ppls = {}
for l in range(n_layers):
    # Prune lowest 50% of this layer's experts only
    n_prune_layer = n_experts // 2  # 4 of 8
    layer_scores = saliency_scores[l]
    bottom_experts = layer_scores.argsort()[:n_prune_layer].tolist()
    pruned_set = {(l, e) for e in bottom_experts}
    ppl = eval_ppl_with_pruning(model, val_subset, pruned_set)
    delta = ppl - baseline_ppl
    layer_ablation_ppls[l] = {"ppl": ppl, "delta": delta}
    print(f"  Layer {l} 50% pruned: PPL = {ppl:.4f} (Δ = {delta:+.4f})", flush=True)

# ─── 10. Validation checks ───────────────────────────────────────────────────
print("\n=== Validation ===", flush=True)
print(f"Baseline PPL: {baseline_ppl:.4f}")

# Monotonicity
ppl_25  = reap_ppls[0.25]
ppl_50  = reap_ppls[0.50]
ppl_75  = reap_ppls[0.75]
monotonic = (baseline_ppl <= ppl_25 <= ppl_50 <= ppl_75)
print(f"Pruning monotonicity: {baseline_ppl:.2f} <= {ppl_25:.2f} <= {ppl_50:.2f} <= {ppl_75:.2f}: {monotonic}")

# REAP vs random at 50%
rand_mean_50 = sum(random_ppls[0.50]) / len(random_ppls[0.50])
rand_std_50  = (sum((x - rand_mean_50)**2 for x in random_ppls[0.50]) / len(random_ppls[0.50])) ** 0.5
reap_better_than_random = ppl_50 < rand_mean_50 - rand_std_50
print(f"REAP 50% PPL={ppl_50:.4f}, random mean={rand_mean_50:.4f} ± {rand_std_50:.4f}")
print(f"REAP better than random by > 1 std: {reap_better_than_random}")

gini_majority = sum(1 for g in layer_ginis if g > 0.3) >= (n_layers * 3 // 4)
print(f"Gini > 0.3 in >= 3/4 layers: {gini_majority} (counts: {sum(1 for g in layer_ginis if g > 0.3)}/{n_layers})")

# ─── 11. Outcome verdict ─────────────────────────────────────────────────────
confirmed = (
    baseline_ppl < 50
    and monotonic
    and gini_majority
)
refuted = (
    abs(ppl_50 - rand_mean_50) < rand_std_50  # random ≈ REAP
    and not monotonic
)
if confirmed:
    outcome = "confirmed"
elif refuted:
    outcome = "refuted"
else:
    outcome = "inconclusive"
print(f"\nOutcome: {outcome.upper()}", flush=True)

# ─── 12. Save JSON results ────────────────────────────────────────────────────
results = {
    "model_params": n_params,
    "baseline_ppl": baseline_ppl,
    "train_losses": train_losses_to_save,
    "router_entropies": router_entropies,
    "saliency_scores": saliency_scores.tolist(),
    "layer_ginis": layer_ginis,
    "top4_bottom4_ratios": top4_bottom4_ratio,
    "spearman_correlation": spearman_mat.tolist(),
    "reap_ppls": {str(k): v for k, v in reap_ppls.items()},
    "random_ppls": {str(k): v for k, v in random_ppls.items()},
    "random_seeds": RANDOM_SEEDS,
    "layer_ablation_ppls": {str(k): v for k, v in layer_ablation_ppls.items()},
    "validation": {
        "monotonic": monotonic,
        "reap_better_than_random_1std": reap_better_than_random,
        "gini_majority": gini_majority,
        "baseline_ppl_ok": baseline_ppl < 50,
    },
    "outcome": outcome,
}
with open(JSON_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {JSON_PATH}", flush=True)

# ─── 13. Plots ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: Saliency heatmap (layer x expert)
ax = axes[0]
im = ax.imshow(saliency_scores.numpy(), aspect="auto", cmap="viridis")
ax.set_xlabel("Expert")
ax.set_ylabel("Layer")
ax.set_title("REAP Saliency (gate×norm)")
plt.colorbar(im, ax=ax)
for l in range(n_layers):
    ax.text(-0.5, l, f"G={layer_ginis[l]:.2f}", va="center", ha="right", fontsize=7)

# Panel 2: PPL curve with random baseline
ax = axes[1]
levels_pct = [0, 25, 50, 75]
reap_vals   = [baseline_ppl] + [reap_ppls[f/100] for f in [25, 50, 75]]
rand_means  = [baseline_ppl] + [sum(random_ppls[f/100])/3 for f in [25, 50, 75]]
rand_stds   = [0.0] + [(sum((x - sum(random_ppls[f/100])/3)**2 for x in random_ppls[f/100])/3)**0.5 for f in [25, 50, 75]]

ax.plot(levels_pct, reap_vals, "b-o", label="REAP")
ax.errorbar(levels_pct, rand_means, yerr=rand_stds, fmt="r--s", label="Random (mean±std)")
ax.set_xlabel("% Experts Pruned")
ax.set_ylabel("Perplexity")
ax.set_title("PPL vs Pruning Level")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Per-layer damage at 50%
ax = axes[2]
layers = list(layer_ablation_ppls.keys())
deltas = [layer_ablation_ppls[l]["delta"] for l in layers]
colors = ["red" if d > 0 else "green" for d in deltas]
ax.bar([f"L{l}" for l in layers], deltas, color=colors)
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xlabel("Layer")
ax.set_ylabel("PPL delta vs baseline")
ax.set_title("Per-Layer Damage (50% pruned)")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(PNG_PATH, dpi=120, bbox_inches="tight")
print(f"Plot saved to {PNG_PATH}", flush=True)

print("\n=== DONE ===", flush=True)
print(f"Outcome: {outcome}")
print(f"Baseline PPL: {baseline_ppl:.2f}")
print(f"REAP PPL @ 25/50/75%: {ppl_25:.2f} / {ppl_50:.2f} / {ppl_75:.2f}")
print(f"Random PPL @ 50%: {rand_mean_50:.2f} ± {rand_std_50:.2f}")
print(f"Layer Ginis: {[f'{g:.3f}' for g in layer_ginis]}")
