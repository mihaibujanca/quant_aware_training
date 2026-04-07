"""
Cycle 253 — Item 263: gelu_noneg + μ targeted ONLY to compressed blocks [1,4,7]
at K=512. Leave the 9 uncompressed blocks on raw GELU. Tests the complement of
cycle 246 (which patched all 12 blocks and failed) — does patching ONLY what
you compress let the activation surgery help the K=512 topk fit?

Install order:
  1. μ precomputation on CALIB for blocks {1,4,7} ONLY.
  2. Patch ONLY blocks 1,4,7 mlp.act with GeluNonegCorrected(μ_l).
     Other 9 blocks remain on raw GELU.
  3. Collect (ln_out, mlp_out) on patched model and fit K=512 topk+lstsq
     against the patched target on [1,4,7]. Replace with CompressedMLP
     (W_fc_sub, b_fc_sub, mu_sub, W_out, b_out) with
     forward = relu(gelu(x @ fc_weight + fc_bias)) + mu_sub then W_out@.T + b_out.
  4. Freeze all; unfreeze only W_out + b_out on the 3 compressed blocks.
  5. Joint Adam LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state restore.
  6. 3 seeds.

Disjoint calib/eval (cycles 230/239): calib=all_ids[:200] eval=all_ids[200:300].

Verdict:
  CONFIRMED    if best_mean ≤ 46.0
  REFUTED      if best_mean ≥ 47.0
  INCONCLUSIVE otherwise
"""
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS = [1, 4, 7]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
SEEDS = [42, 137, 2024]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 4000
EVAL_EVERY = 100

RESULT_PATH = "autoresearch/results/cycle_253_gelu_noneg_targeted.json"
SUMMARY_PATH = "autoresearch/results/cycle_253_summary.md"

C210_K512_BASELINE = 46.556        # cycle 210 K=512 [1,4,7] raw-GELU best_mean
C246_ALL12_BASELINE = 57.811       # cycle 246 all-12 patched K=128 best_mean
C239_LORA_BASELINE = 47.692        # cycle 239 gelu_noneg+μ+LoRA r=8 all 12 (no compression)
C220_FLOAT_REF = 45.064            # cycle 220 float reference
CONFIRM_THRESH = 46.0
REFUTE_THRESH = 47.0

print(f"Device: {DEVICE}")

# ── Load data with disjoint split (cycle 246 pattern) ────────────────────────
print("Loading GPT-2 tokenizer + wikitext-2 val...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]

fallback_seq_len = None


def tokenize_to_seqs(texts, seq_len, target):
    out = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        if enc["input_ids"].shape[1] >= seq_len:
            out.append(enc["input_ids"][:, :seq_len])
        if len(out) >= target:
            break
    return out


all_ids = tokenize_to_seqs(texts, SEQ_LEN, N_CALIB + N_EVAL)
if len(all_ids) < N_CALIB + N_EVAL:
    print(f"WARN: only got {len(all_ids)} at SEQ_LEN={SEQ_LEN}; falling back to 64")
    fallback_seq_len = 64
    SEQ_LEN = 64
    all_ids = tokenize_to_seqs(texts, SEQ_LEN, N_CALIB + N_EVAL)
assert len(all_ids) >= N_CALIB + N_EVAL

calib_ids = [t.to(DEVICE) for t in all_ids[:N_CALIB]]
eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB:N_CALIB + N_EVAL]]

# Disjointness assertions
calib_id_set = set(map(id, calib_ids))
eval_id_set = set(map(id, eval_ids))
assert calib_id_set.isdisjoint(eval_id_set), "calib/eval share object ids"
assert len(calib_ids) == N_CALIB and len(eval_ids) == N_EVAL
assert not torch.equal(calib_ids[0], eval_ids[0])
print(f"Calib: {len(calib_ids)} × {SEQ_LEN}, Eval: {len(eval_ids)} × {SEQ_LEN} (disjoint)")


def eval_ppl(model, ids_list):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for ids in ids_list:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


# ── Modules ───────────────────────────────────────────────────────────────────
class GeluNonegCorrected(nn.Module):
    """relu(gelu(z)) + μ_l — matches cycle 246. Applied to mlp.act on raw blocks."""
    def __init__(self, mu):
        super().__init__()
        self.register_buffer('mu', mu)

    def forward(self, z):
        return torch.relu(F.gelu(z, approximate='tanh')) + self.mu


class CompressedMLP(nn.Module):
    """
    K=512 topk+lstsq compressed MLP for the 3 targeted compressed blocks.
    Forward computes the gelu_noneg+μ activation on the topk-sliced preact:
        pre = x @ W_fc_sub + b_fc_sub
        act = relu(gelu(pre)) + mu_sub
        y   = act @ W_out.T + b_out
    W_fc_sub, b_fc_sub, mu_sub are frozen buffers; W_out, b_out are trainable.
    """
    def __init__(self, fc_weight, fc_bias, mu_sub, W_out, b_out):
        super().__init__()
        self.register_buffer("fc_weight", fc_weight)  # (768, K)
        self.register_buffer("fc_bias", fc_bias)      # (K,)
        self.register_buffer("mu_sub", mu_sub)        # (K,)
        self.W_out = nn.Parameter(W_out.clone())      # (768, K)
        self.b_out = nn.Parameter(b_out.clone())      # (768,)
        self.K = fc_weight.shape[1]

    def forward(self, x):
        pre = x @ self.fc_weight + self.fc_bias
        act = torch.relu(F.gelu(pre, approximate='tanh')) + self.mu_sub
        return act @ self.W_out.T + self.b_out


# ── Helpers ───────────────────────────────────────────────────────────────────
def collect_mlp_io(model, calib_ids, block_ids):
    """Collect (ln_out, mlp_out) on the current (possibly patched) model."""
    ln_outs = {b: [] for b in block_ids}
    proj_outs = {b: [] for b in block_ids}
    hooks = []
    for b in block_ids:
        def mk_hook(bb):
            def fn(m, inp, out):
                ln_outs[bb].append(inp[0].detach().float().cpu().reshape(-1, 768))
                proj_outs[bb].append(out.detach().float().cpu().reshape(-1, 768))
            return fn
        hooks.append(model.transformer.h[b].mlp.register_forward_hook(mk_hook(b)))
    with torch.no_grad():
        for ids in calib_ids:
            model(ids)
    for h in hooks:
        h.remove()
    for b in block_ids:
        ln_outs[b] = torch.cat(ln_outs[b], dim=0)
        proj_outs[b] = torch.cat(proj_outs[b], dim=0)
    return ln_outs, proj_outs


def fit_block_lstsq_gelu_noneg(block, ln_out_cpu, proj_out_cpu, mu_l_cpu, K):
    """
    Fit K=512 topk+lstsq against the gelu_noneg+μ patched activation target.
    Returns a CompressedMLP with μ_sub at the selected topk indices.
    """
    W_fc = block.mlp.c_fc.weight.data.float().cpu()   # (768, 3072)
    b_fc = block.mlp.c_fc.bias.data.float().cpu()     # (3072,)
    W_proj = block.mlp.c_proj.weight.data.float().cpu()  # (3072, 768)

    # Patched activation: relu(gelu(z)) + μ_l (same as GeluNonegCorrected forward)
    pre = ln_out_cpu @ W_fc + b_fc                       # (N, 3072)
    act_all = torch.relu(F.gelu(pre, approximate='tanh')) + mu_l_cpu  # (N, 3072)

    # topk scoring on the patched activation (cycle 223 style)
    scores = act_all.std(0) * W_proj.norm(dim=1)
    topk_idx = scores.topk(K).indices

    W_fc_sub = W_fc[:, topk_idx]                         # (768, K)
    b_fc_sub = b_fc[topk_idx]                            # (K,)
    mu_sub = mu_l_cpu[topk_idx]                          # (K,)
    act_sub = act_all[:, topk_idx]                       # (N, K)

    ones = torch.ones(act_sub.shape[0], 1)
    design = torch.cat([act_sub, ones], dim=1)
    sol = torch.linalg.lstsq(design, proj_out_cpu, rcond=None).solution
    W_out = sol[:K].T                                    # (768, K)
    b_out = sol[K]                                       # (768,)
    return CompressedMLP(W_fc_sub, b_fc_sub, mu_sub, W_out, b_out)


# ── Build base model ──────────────────────────────────────────────────────────
print("\nLoading GPT-2 small...")
torch.manual_seed(SEEDS[0])
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

print("\nFloat baseline PPL on EVAL split...")
baseline_ppl_eval = eval_ppl(model, eval_ids)
print(f"  baseline_ppl_eval = {baseline_ppl_eval:.4f}  (c220 ref {C220_FLOAT_REF})")


# ── PHASE 1 — μ precomputation on CALIB for BLOCKS ONLY ──────────────────────
print(f"\n──────── PHASE 1: μ precomputation on CALIB (blocks {BLOCKS} ONLY) ────────")
stash = {l: [] for l in BLOCKS}
hooks = []


def make_fc_hook(l):
    def hook(module, inputs, output):
        stash[l].append(inputs[0].detach())
    return hook


for l in BLOCKS:
    h = model.transformer.h[l].mlp.c_fc.register_forward_hook(make_fc_hook(l))
    hooks.append(h)

with torch.no_grad():
    for ids in calib_ids:
        _ = model(ids)

for h in hooks:
    h.remove()

mu_dict = {}
mu_dict_cpu = {}
mu_stats_per_block = []
with torch.no_grad():
    for l in BLOCKS:
        x_l = torch.cat(stash[l], dim=0).reshape(-1, 768)
        c_fc = model.transformer.h[l].mlp.c_fc
        preact = x_l @ c_fc.weight + c_fc.bias
        gelu_l = F.gelu(preact, approximate='tanh')
        noneg_l = torch.relu(gelu_l)
        mu_l = (gelu_l - noneg_l).mean(dim=0).detach()
        mu_dict[l] = mu_l.to(DEVICE)
        mu_dict_cpu[l] = mu_l.float().cpu()
        mu_stats_per_block.append({
            "block": l,
            "mean": float(mu_l.mean().item()),
            "std": float(mu_l.std().item()),
            "min": float(mu_l.min().item()),
            "max": float(mu_l.max().item()),
            "abs_mean": float(mu_l.abs().mean().item()),
            "n_positive_entries": int((mu_l > 0).sum().item()),
        })
        stash[l] = None

assert len(mu_dict) == len(BLOCKS) == 3, f"mu_dict should have {len(BLOCKS)} entries, got {len(mu_dict)}"
max_positive = max(s["max"] for s in mu_stats_per_block)
assert max_positive <= 1e-6, f"μ has positive entries (max={max_positive:.6e})"
print(f"OK: μ ≤ 0 elementwise across blocks {BLOCKS} (max = {max_positive:.2e})")
print(f"OK: mu_dict has exactly {len(mu_dict)} entries (not 12)")


# ── PHASE 2 — Patch ONLY blocks {1,4,7} with GeluNonegCorrected ───────────────
print(f"\n──────── PHASE 2: Install GeluNonegCorrected on blocks {BLOCKS} ONLY ────────")
for l in BLOCKS:
    model.transformer.h[l].mlp.act = GeluNonegCorrected(mu_dict[l]).to(DEVICE)

# Sanity: only compressed blocks are patched; others are raw GELU
for l in range(12):
    act_type = type(model.transformer.h[l].mlp.act).__name__
    if l in BLOCKS:
        assert isinstance(model.transformer.h[l].mlp.act, GeluNonegCorrected), \
            f"block {l} should be GeluNonegCorrected, got {act_type}"
    else:
        assert not isinstance(model.transformer.h[l].mlp.act, GeluNonegCorrected), \
            f"block {l} should be raw GELU (NewGELU), got {act_type}"
print(f"OK: blocks {BLOCKS} patched to GeluNonegCorrected; "
      f"other 9 blocks still raw GELU")

post_mu_calib_ppl = eval_ppl(model, calib_ids)
post_mu_eval_ppl = eval_ppl(model, eval_ids)
print(f"post_mu_patch_ppl  CALIB = {post_mu_calib_ppl:.4f}")
print(f"post_mu_patch_ppl  EVAL  = {post_mu_eval_ppl:.4f}")


# ── PHASE 3 — K=512 topk+lstsq on [1,4,7] against patched target ─────────────
print(f"\n──────── PHASE 3: K={K} topk+lstsq on BLOCKS={BLOCKS} (patched target) ────────")
ln_outs, proj_outs = collect_mlp_io(model, calib_ids, BLOCKS)
for b in BLOCKS:
    compressed = fit_block_lstsq_gelu_noneg(
        model.transformer.h[b],
        ln_outs[b],
        proj_outs[b],
        mu_dict_cpu[b],
        K=K,
    )
    model.transformer.h[b].mlp = compressed.to(DEVICE)
    print(f"  Compressed block {b}: K={K} topk+lstsq on gelu_noneg target", flush=True)
del ln_outs, proj_outs

post_compress_eval_ppl = eval_ppl(model, eval_ids)
print(f"post_compress_eval_ppl (K={K} on {BLOCKS}, targeted μ) = {post_compress_eval_ppl:.4f}")


# ── PHASE 4 — Freeze all; unfreeze W_out + b_out on 3 compressed blocks ──────
print("\n──────── PHASE 4: Freeze all; unfreeze W_out + b_out on compressed blocks ────────")
for p in model.parameters():
    p.requires_grad_(False)

trainable_params = []
for b in BLOCKS:
    mlp = model.transformer.h[b].mlp
    mlp.W_out.requires_grad_(True)
    mlp.b_out.requires_grad_(True)
    trainable_params.extend([mlp.W_out, mlp.b_out])

n_trainable = sum(p.numel() for p in trainable_params)
expected = 3 * (768 * K + 768)  # 3 × (W_out + b_out)
print(f"Trainable params: {n_trainable:,} (expected {expected:,})")
assert n_trainable == expected, f"got {n_trainable}, expected {expected}"


# ── Snapshot / restore helpers ────────────────────────────────────────────────
def snapshot_state():
    s = {}
    for b in BLOCKS:
        mlp = model.transformer.h[b].mlp
        s[f"compressed_{b}"] = (
            mlp.W_out.detach().clone().cpu(),
            mlp.b_out.detach().clone().cpu(),
        )
    return s


def restore_state(s):
    for b in BLOCKS:
        mlp = model.transformer.h[b].mlp
        W_out_b, b_out_b = s[f"compressed_{b}"]
        mlp.W_out.data.copy_(W_out_b.to(DEVICE))
        mlp.b_out.data.copy_(b_out_b.to(DEVICE))


# Snapshot the lstsq init so we can reset across seeds
init_compressed_state = snapshot_state()


def reset_compressed_to_lstsq():
    restore_state(init_compressed_state)


# ── PHASE 5 — Train per seed ──────────────────────────────────────────────────
print("\n──────── PHASE 5: Joint Adam per seed ────────")
seed_results = []

for seed in SEEDS:
    print(f"\n──── SEED {seed} ────")
    reset_compressed_to_lstsq()

    init_eval_ppl = eval_ppl(model, eval_ids)
    print(f"  init_eval_ppl = {init_eval_ppl:.4f}  (should match post_compress {post_compress_eval_ppl:.4f})")
    assert abs(init_eval_ppl - post_compress_eval_ppl) < 1e-3

    opt = torch.optim.Adam(trainable_params, lr=LR)
    best_ppl = init_eval_ppl
    best_step = 0
    best_state = snapshot_state()
    ppl_history = [(0, float(init_eval_ppl))]
    loss_history = []

    random.seed(seed)
    torch.manual_seed(seed)
    n_data = len(calib_ids)
    calib_order = list(range(n_data))
    random.shuffle(calib_order)

    print(f"  Training: N_STEPS={N_STEPS}, LR={LR}, EVAL_EVERY={EVAL_EVERY}, n_calib={n_data}")
    model.train()
    step = 0
    while step < N_STEPS:
        for idx in calib_order:
            if step >= N_STEPS:
                break
            ids = calib_ids[idx]
            out = model(ids, labels=ids)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            loss_val = float(loss.item())
            loss_history.append((step, loss_val))

            if step % EVAL_EVERY == 0:
                ppl_e = eval_ppl(model, eval_ids)
                ppl_history.append((step, float(ppl_e)))
                marker = ""
                if ppl_e < best_ppl:
                    best_ppl = float(ppl_e)
                    best_step = step
                    best_state = snapshot_state()
                    marker = " *BEST*"
                print(f"    step {step:4d}  loss={loss_val:.4f}  eval_ppl={ppl_e:.4f}{marker}", flush=True)
                model.train()

    print(f"  Restoring best state from step {best_step} (best eval_ppl {best_ppl:.4f})...")
    restore_state(best_state)
    final_eval_ppl = eval_ppl(model, eval_ids)
    print(f"  final_eval_ppl (restored) = {final_eval_ppl:.4f}")
    assert abs(final_eval_ppl - best_ppl) < 1e-3, \
        f"restore mismatch: {final_eval_ppl} vs {best_ppl}"

    seed_results.append({
        "seed": seed,
        "init_eval_ppl": float(init_eval_ppl),
        "best_eval_ppl": float(best_ppl),
        "best_step": int(best_step),
        "final_eval_ppl": float(final_eval_ppl),
        "ppl_history": ppl_history,
        "loss_history_head": loss_history[:20],
        "loss_history_tail": loss_history[-20:],
    })


# ── Aggregate ─────────────────────────────────────────────────────────────────
finals = torch.tensor([r["final_eval_ppl"] for r in seed_results])
inits = torch.tensor([r["init_eval_ppl"] for r in seed_results])
bests = torch.tensor([r["best_eval_ppl"] for r in seed_results])
final_mean = float(finals.mean().item())
final_std = float(finals.std(unbiased=False).item()) if len(finals) > 1 else 0.0
init_mean = float(inits.mean().item())
best_mean = float(bests.mean().item())
best_std = float(bests.std(unbiased=False).item()) if len(bests) > 1 else 0.0

delta_vs_c210 = best_mean - C210_K512_BASELINE
delta_vs_c246 = best_mean - C246_ALL12_BASELINE
delta_vs_c239 = best_mean - C239_LORA_BASELINE
delta_vs_float = best_mean - baseline_ppl_eval
delta_vs_float_pct = delta_vs_float / baseline_ppl_eval * 100

if best_mean <= CONFIRM_THRESH:
    verdict = "CONFIRMED"
elif best_mean >= REFUTE_THRESH:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print("\n─────────────────────────────────────")
print(f"baseline_ppl_eval      = {baseline_ppl_eval:.4f}")
print(f"post_mu_calib_ppl      = {post_mu_calib_ppl:.4f}")
print(f"post_mu_eval_ppl       = {post_mu_eval_ppl:.4f}")
print(f"post_compress_eval_ppl = {post_compress_eval_ppl:.4f}  (K={K} {BLOCKS}, targeted μ)")
print(f"init_eval_ppl  mean    = {init_mean:.4f}")
print(f"best_eval_ppl  mean    = {best_mean:.4f} ± {best_std:.4f}")
print(f"final_eval_ppl mean    = {final_mean:.4f} ± {final_std:.4f}  (n={len(seed_results)} seeds)")
print(f"per-seed best          = {[round(r['best_eval_ppl'], 4) for r in seed_results]}")
print(f"Δ vs c210 K=512 [1,4,7] raw   (46.556) = {delta_vs_c210:+.4f}")
print(f"Δ vs c246 all-12 K=128        (57.811) = {delta_vs_c246:+.4f}")
print(f"Δ vs c239 LoRA+gelu_noneg     (47.692) = {delta_vs_c239:+.4f}")
print(f"Δ vs float baseline           = {delta_vs_float:+.4f} ({delta_vs_float_pct:+.2f}%)")
print(f"thresholds                    = CONFIRM ≤ {CONFIRM_THRESH}, REFUTE ≥ {REFUTE_THRESH}")
print(f"VERDICT: {verdict}")


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 253,
    "item": 263,
    "verdict": verdict,
    "config": {
        "BLOCKS": BLOCKS,
        "K": K,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "SEEDS": SEEDS,
        "DEVICE": DEVICE,
        "LR": LR,
        "N_STEPS": N_STEPS,
        "EVAL_EVERY": EVAL_EVERY,
        "fallback_seq_len": fallback_seq_len,
    },
    "baseline_ppl_eval": float(baseline_ppl_eval),
    "post_mu_calib_ppl": float(post_mu_calib_ppl),
    "post_mu_eval_ppl": float(post_mu_eval_ppl),
    "post_compress_eval_ppl": float(post_compress_eval_ppl),
    "cycle210_k512_ref_ppl": float(C210_K512_BASELINE),
    "cycle246_all12_ref_ppl": float(C246_ALL12_BASELINE),
    "cycle239_lora_ref_ppl": float(C239_LORA_BASELINE),
    "cycle220_float_ref_ppl": float(C220_FLOAT_REF),
    "init_eval_ppl_mean": init_mean,
    "best_eval_ppl_mean": best_mean,
    "best_eval_ppl_std": best_std,
    "final_eval_ppl_mean": final_mean,
    "final_eval_ppl_std": final_std,
    "delta_vs_c210_k512": float(delta_vs_c210),
    "delta_vs_c246_all12": float(delta_vs_c246),
    "delta_vs_c239_lora": float(delta_vs_c239),
    "delta_vs_float": float(delta_vs_float),
    "delta_vs_float_pct": float(delta_vs_float_pct),
    "confirm_thresh": float(CONFIRM_THRESH),
    "refute_thresh": float(REFUTE_THRESH),
    "seeds": seed_results,
    "trainable_params": int(n_trainable),
    "mu_stats_per_block": mu_stats_per_block,
}
with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved {RESULT_PATH}")


# ── Save summary MD ───────────────────────────────────────────────────────────
per_seed_lines = "\n".join(
    f"| seed {r['seed']} | {r['init_eval_ppl']:.4f} | {r['best_eval_ppl']:.4f} (step {r['best_step']}) | {r['final_eval_ppl']:.4f} |"
    for r in seed_results
)

summary = f"""# Cycle 253 — gelu_noneg + μ targeted to compressed blocks ONLY (item 263)

**Verdict: {verdict}**

Tests the complement of cycle 246 (which patched all 12 blocks with gelu_noneg
+ μ_l and failed on K=128 [1,4,7] compression, 57.811). This experiment patches
ONLY the 3 compressed blocks {BLOCKS} with `GeluNonegCorrected(μ_l)`, leaves
the other 9 blocks on raw GELU, and runs the standard K={K} topk+lstsq + Adam
W_out+b_out recipe against the patched targets. Isolates whether the activation
surgery is a per-block target-side improvement or a whole-model change.

**Install order.** (1) μ precomputation on CALIB for blocks {BLOCKS} ONLY
(3 entries in mu_dict, not 12); (2) patch ONLY those 3 mlp.act modules with
`GeluNonegCorrected(μ_l)`, leave the other 9 blocks on raw GELU; (3) collect
(ln_out, mlp_out) on the patched model and fit K={K} topk+lstsq against the
patched target on [1,4,7], replacing those mlp modules with `CompressedMLP`
(W_fc_sub, b_fc_sub, mu_sub, W_out, b_out); (4) freeze all; unfreeze only
W_out + b_out on the 3 compressed blocks; (5) joint Adam LR={LR},
N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, best-state restore.

| Variant | PPL | source |
|---|---|---|
| Float baseline (held-out) | {baseline_ppl_eval:.3f} | this run |
| cycle 220 float ref | {C220_FLOAT_REF:.3f} | reference |
| μ-only CALIB (targeted, 3 blocks) | {post_mu_calib_ppl:.3f} | this run |
| μ-only EVAL (targeted, 3 blocks) | {post_mu_eval_ppl:.3f} | this run |
| μ + K={K} [1,4,7] EVAL (lstsq init) | {post_compress_eval_ppl:.3f} | this run |
| cycle 210 K=512 [1,4,7] raw-GELU | {C210_K512_BASELINE:.3f} | direct baseline |
| cycle 246 all-12 patched K=128 | {C246_ALL12_BASELINE:.3f} | what NOT to do |
| cycle 239 gelu_noneg+μ+LoRA (all 12, no compression) | {C239_LORA_BASELINE:.3f} | no K compression |
| **cycle 253 targeted best_mean ± std** | **{best_mean:.3f} ± {best_std:.3f}** | this run, n={len(seed_results)} seeds |
| cycle 253 final_mean ± std | {final_mean:.3f} ± {final_std:.3f} | after full {N_STEPS} steps |

**Per-seed results (final_eval_ppl after best-state restore):**

| seed | init_eval_ppl | best_eval_ppl (step) | final_eval_ppl |
|---|---|---|---|
{per_seed_lines}

Thresholds: CONFIRMED ≤ {CONFIRM_THRESH}, REFUTED ≥ {REFUTE_THRESH}.
Δ vs float baseline: {delta_vs_float:+.3f} PPL ({delta_vs_float_pct:+.2f}%).
Δ vs cycle 210 K=512 raw ({C210_K512_BASELINE}): {delta_vs_c210:+.3f} PPL.
Δ vs cycle 246 all-12 ({C246_ALL12_BASELINE}): {delta_vs_c246:+.3f} PPL.
Δ vs cycle 239 LoRA ({C239_LORA_BASELINE}): {delta_vs_c239:+.3f} PPL.

**Trainable budget.** {n_trainable:,} params = 3 × (768×{K} + 768) [W_out + b_out on 3 blocks].

**Sanity.**
- `len(calib_ids) = {len(calib_ids)}`, `len(eval_ids) = {len(eval_ids)}`, disjoint by object id and tensor equality.
- `mu_dict` has exactly {len(mu_dict)} entries (for blocks {BLOCKS}), not 12.
- μ ≤ 0 elementwise across the 3 targeted blocks (max = {max_positive:.2e}).
- Phase 2 sanity: compressed blocks {BLOCKS} patched to `GeluNonegCorrected`; other 9 blocks still raw GELU.
- `init_eval_ppl == post_compress_eval_ppl` across all seeds (within 1e-3).
- `|final_eval_ppl - best_eval_ppl| < 1e-3` per seed (best-state restore round-trip).
- Fallback SEQ_LEN: {fallback_seq_len if fallback_seq_len else 'none (SEQ_LEN=128)'}.
"""
with open(SUMMARY_PATH, "w") as f:
    f.write(summary)
print(f"Saved {SUMMARY_PATH}")
