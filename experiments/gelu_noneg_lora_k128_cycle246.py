"""
Cycle 246 — Item 256: gelu_noneg + μ_l (12 blocks) + LoRA r=8 (12 blocks) + K=128
W_out topk+lstsq compression on BLOCKS=[1,4,7].

Combines cycle 239's recipe (gelu_noneg + per-channel μ_l + LoRA r=8 on c_proj
across all 12 blocks, 47.692 ± 0.143 held-out) with cycle 223's K=128 W_out
channel compression on [1,4,7] (48.989 alone). Tests whether activation fix
and channel compression compose (≤ 47.0) or trade (≥ 49.5).

Install order (critical):
  1. Precompute μ_l on CALIB for all 12 blocks (cycle 228/239 recipe).
  2. Patch all 12 mlp.act with GeluNonegCorrected(μ_l).
  3. Collect (ln_out, mlp_out) on the patched model for [1,4,7] and fit
     K=128 topk+lstsq → replace mlp with CompressedMLPWithLoRA (built-in LoRA).
  4. Install LoraConv1D on the 9 uncompressed blocks' mlp.c_proj.
  5. Joint Adam on {LoRA A/B × 12 blocks} ∪ {W_out, b_out × 3 compressed blocks}.

Disjoint calib/eval split (cycles 230/239): calib=all_ids[:200] eval=all_ids[200:300].

Verdict:
  CONFIRMED   if best_mean ≤ 47.0
  REFUTED     if best_mean ≥ 49.5
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
K = 128
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
SEEDS = [42, 137, 2024]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 2000
EVAL_EVERY = 200
RANK = 8

RESULT_PATH = "autoresearch/results/cycle_246_gelu_noneg_lora_k128.json"
SUMMARY_PATH = "autoresearch/results/cycle_246_summary.md"

C223_K128_BASELINE = 48.989        # cycle 223 K=128 [1,4,7] W_out only
C239_LORA_BASELINE = 47.692        # cycle 239 gelu_noneg+μ+LoRA r=8 all 12 blocks
C220_FLOAT_REF = 45.064            # cycle 220 float reference
CONFIRM_THRESH = 47.0
REFUTE_THRESH = 49.5

print(f"Device: {DEVICE}")

# ── Load data with disjoint split (cycle 239 pattern) ────────────────────────
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
    def __init__(self, mu):
        super().__init__()
        self.register_buffer('mu', mu)

    def forward(self, z):
        return torch.relu(F.gelu(z, approximate='tanh')) + self.mu


class CompressedMLPWithLoRA(nn.Module):
    """
    K=128 topk+lstsq compressed MLP carrying a built-in LoRA r=8 residual on the
    W_out projection.

    Forward computes:
        act = gelu_noneg_corrected(x @ W_fc_sub + b_fc_sub)
            = relu(gelu(x @ fc_weight + fc_bias)) + mu_sub
        y   = act @ (W_out + A @ B).T + b_out
    where:
        x          : (..., 768)
        fc_weight  : (768, K)   buffer
        fc_bias    : (K,)       buffer
        mu_sub     : (K,)       buffer  — subset of the block's μ_l at the topk indices
        W_out      : (768, K)   parameter
        b_out      : (768,)     parameter
        A          : (K, r)     parameter
        B          : (r, 768)   parameter  (zero init)
    """
    def __init__(self, fc_weight, fc_bias, mu_sub, W_out, b_out, r=8, a_std=0.01):
        super().__init__()
        self.register_buffer("fc_weight", fc_weight)  # (768, K)
        self.register_buffer("fc_bias", fc_bias)      # (K,)
        self.register_buffer("mu_sub", mu_sub)        # (K,)
        self.W_out = nn.Parameter(W_out.clone())      # (768, K)
        self.b_out = nn.Parameter(b_out.clone())      # (768,)
        self.K = fc_weight.shape[1]
        self.r = r
        self.A = nn.Parameter(torch.randn(self.K, r) * a_std)
        self.B = nn.Parameter(torch.zeros(r, 768))

    def forward(self, x):
        pre = x @ self.fc_weight + self.fc_bias
        act = torch.relu(F.gelu(pre, approximate='tanh')) + self.mu_sub
        # Effective W_out: (768, K) + (768, K) via A@B^T style
        # A @ B is (K, 768); we want delta (768, K); use .T.
        delta = (self.A @ self.B).T
        W_eff = self.W_out + delta
        return act @ W_eff.T + self.b_out


class LoraConv1D(nn.Module):
    """LoRA wrapper around HF Conv1D (mlp.c_proj). A zero-like, B zero → delta=0 init."""
    def __init__(self, base_conv1d, r=8, a_std=0.01):
        super().__init__()
        self.base = base_conv1d
        for p in self.base.parameters():
            p.requires_grad_(False)
        nx, nf = base_conv1d.weight.shape  # (3072, 768)
        self.nx = nx
        self.nf = nf
        self.r = r
        self.A = nn.Parameter(torch.randn(nx, r) * a_std)
        self.B = nn.Parameter(torch.zeros(r, nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        W = self.base.weight + self.A @ self.B  # (nx, nf)
        out = torch.addmm(self.base.bias, x.view(-1, self.nx), W)
        return out.view(size_out)


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
    Fit K=128 topk+lstsq against the gelu_noneg+μ patched activation target.
    Returns a CompressedMLPWithLoRA with μ_sub at the selected topk indices.
    """
    W_fc = block.mlp.c_fc.weight.data.float().cpu()   # (768, 3072)
    b_fc = block.mlp.c_fc.bias.data.float().cpu()     # (3072,)
    W_proj = block.mlp.c_proj.base.weight.data.float().cpu() \
        if hasattr(block.mlp.c_proj, 'base') else block.mlp.c_proj.weight.data.float().cpu()

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
    return CompressedMLPWithLoRA(W_fc_sub, b_fc_sub, mu_sub, W_out, b_out, r=RANK)


# ── Build base model ──────────────────────────────────────────────────────────
print("\nLoading GPT-2 small...")
torch.manual_seed(SEEDS[0])
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

print("\nFloat baseline PPL on EVAL split...")
baseline_ppl_eval = eval_ppl(model, eval_ids)
print(f"  baseline_ppl_eval = {baseline_ppl_eval:.4f}  (c220 ref {C220_FLOAT_REF})")


# ── PHASE 1 — μ precomputation on CALIB ───────────────────────────────────────
print("\n──────── PHASE 1: μ precomputation on CALIB (12 blocks) ────────")
stash = {l: [] for l in range(12)}
hooks = []


def make_fc_hook(l):
    def hook(module, inputs, output):
        stash[l].append(inputs[0].detach())
    return hook


for l in range(12):
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
    for l in range(12):
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

max_positive = max(s["max"] for s in mu_stats_per_block)
assert max_positive <= 1e-6, f"μ has positive entries (max={max_positive:.6e})"
print(f"OK: μ ≤ 0 elementwise across all 12 blocks (max = {max_positive:.2e})")


# ── PHASE 2 — Patch all 12 blocks with GeluNonegCorrected ────────────────────
print("\n──────── PHASE 2: Install GeluNonegCorrected on 12 blocks ────────")
for l in range(12):
    model.transformer.h[l].mlp.act = GeluNonegCorrected(mu_dict[l]).to(DEVICE)
ids_check = set(id(model.transformer.h[i].mlp.act) for i in range(12))
assert len(ids_check) == 12
post_mu_calib_ppl = eval_ppl(model, calib_ids)
post_mu_eval_ppl = eval_ppl(model, eval_ids)
print(f"post_mu_patch_ppl  CALIB = {post_mu_calib_ppl:.4f}")
print(f"post_mu_patch_ppl  EVAL  = {post_mu_eval_ppl:.4f}")


# ── PHASE 3 — K=128 topk+lstsq on [1,4,7] against patched target ─────────────
print("\n──────── PHASE 3: K=128 topk+lstsq on BLOCKS=[1,4,7] (patched target) ────────")
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
    print(f"  Compressed block {b}: K=128 topk+lstsq + internal LoRA r={RANK}", flush=True)
del ln_outs, proj_outs

post_compress_eval_ppl = eval_ppl(model, eval_ids)
print(f"post_compress_eval_ppl (K=128 on [1,4,7], LoRA zero) = {post_compress_eval_ppl:.4f}")


# ── PHASE 4 — Install LoRA on the remaining 9 float blocks' mlp.c_proj ───────
print("\n──────── PHASE 4: LoRA r=8 on the 9 uncompressed blocks' mlp.c_proj ────────")
float_blocks = [l for l in range(12) if l not in BLOCKS]
for l in float_blocks:
    orig_cp = model.transformer.h[l].mlp.c_proj
    lora = LoraConv1D(orig_cp, r=RANK).to(DEVICE)
    model.transformer.h[l].mlp.c_proj = lora

# Freeze all, then unfreeze the exact trainable set
for p in model.parameters():
    p.requires_grad_(False)

trainable_params = []
# Compressed blocks: W_out, b_out, A, B
for b in BLOCKS:
    mlp = model.transformer.h[b].mlp
    mlp.W_out.requires_grad_(True)
    mlp.b_out.requires_grad_(True)
    mlp.A.requires_grad_(True)
    mlp.B.requires_grad_(True)
    trainable_params.extend([mlp.W_out, mlp.b_out, mlp.A, mlp.B])
# Float blocks: LoRA A, B on c_proj
for l in float_blocks:
    cp = model.transformer.h[l].mlp.c_proj
    cp.A.requires_grad_(True)
    cp.B.requires_grad_(True)
    trainable_params.extend([cp.A, cp.B])

n_trainable = sum(p.numel() for p in trainable_params)
# Expected:
#   9 float-block LoRA: 9 × (3072×8 + 8×768) = 276,480
#   3 compressed-block LoRA: 3 × (128×8 + 8×768) = 21,504
#   3 compressed W_out + b_out: 3 × (768×128 + 768) = 297,216
#   Total = 595,200
expected = 9 * (3072 * RANK + RANK * 768) + 3 * (K * RANK + RANK * 768) \
    + 3 * (768 * K + 768)
print(f"Trainable params: {n_trainable:,} (expected {expected:,})")
assert n_trainable == expected, f"got {n_trainable}, expected {expected}"

# LoRA-zero init eval (B=0 on all 12 blocks → delta=0 → identical to post_compress)
init_check_ppl = eval_ppl(model, eval_ids)
print(f"LoRA-zero init eval_ppl = {init_check_ppl:.4f}  (should match post_compress {post_compress_eval_ppl:.4f})")
assert abs(init_check_ppl - post_compress_eval_ppl) < 1e-3, \
    f"LoRA-zero init mismatch: {init_check_ppl:.6f} vs {post_compress_eval_ppl:.6f}"


# ── Snapshot / restore helpers ────────────────────────────────────────────────
def snapshot_state():
    s = {}
    for b in BLOCKS:
        mlp = model.transformer.h[b].mlp
        s[f"compressed_{b}"] = (
            mlp.W_out.detach().clone().cpu(),
            mlp.b_out.detach().clone().cpu(),
            mlp.A.detach().clone().cpu(),
            mlp.B.detach().clone().cpu(),
        )
    for l in float_blocks:
        cp = model.transformer.h[l].mlp.c_proj
        s[f"lora_{l}"] = (
            cp.A.detach().clone().cpu(),
            cp.B.detach().clone().cpu(),
        )
    return s


def restore_state(s):
    for b in BLOCKS:
        mlp = model.transformer.h[b].mlp
        W_out_b, b_out_b, A_b, B_b = s[f"compressed_{b}"]
        mlp.W_out.data.copy_(W_out_b.to(DEVICE))
        mlp.b_out.data.copy_(b_out_b.to(DEVICE))
        mlp.A.data.copy_(A_b.to(DEVICE))
        mlp.B.data.copy_(B_b.to(DEVICE))
    for l in float_blocks:
        cp = model.transformer.h[l].mlp.c_proj
        A_l, B_l = s[f"lora_{l}"]
        cp.A.data.copy_(A_l.to(DEVICE))
        cp.B.data.copy_(B_l.to(DEVICE))


def reset_lora(seed):
    """Reset all LoRA A matrices (random) and B matrices (zero). Do NOT touch
    the compressed W_out/b_out (those are the lstsq init we want to preserve)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    for l in float_blocks:
        cp = model.transformer.h[l].mlp.c_proj
        new_A = (torch.randn(cp.A.shape, generator=g) * 0.01).to(DEVICE)
        cp.A.data.copy_(new_A)
        cp.B.data.zero_()
    for b in BLOCKS:
        mlp = model.transformer.h[b].mlp
        new_A = (torch.randn(mlp.A.shape, generator=g) * 0.01).to(DEVICE)
        mlp.A.data.copy_(new_A)
        mlp.B.data.zero_()


# Snapshot the lstsq init of compressed W_out/b_out so we can reset across seeds
init_compressed_state = {}
for b in BLOCKS:
    mlp = model.transformer.h[b].mlp
    init_compressed_state[b] = (
        mlp.W_out.detach().clone().cpu(),
        mlp.b_out.detach().clone().cpu(),
    )


def reset_compressed_to_lstsq():
    for b in BLOCKS:
        mlp = model.transformer.h[b].mlp
        W_out_b, b_out_b = init_compressed_state[b]
        mlp.W_out.data.copy_(W_out_b.to(DEVICE))
        mlp.b_out.data.copy_(b_out_b.to(DEVICE))


# ── PHASE 5 — Train per seed ──────────────────────────────────────────────────
print("\n──────── PHASE 5: Joint Adam per seed ────────")
seed_results = []

for seed in SEEDS:
    print(f"\n──── SEED {seed} ────")
    reset_compressed_to_lstsq()
    reset_lora(seed)

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

delta_vs_c223 = best_mean - C223_K128_BASELINE
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
print(f"post_compress_eval_ppl = {post_compress_eval_ppl:.4f}  (K=128 [1,4,7], LoRA=0)")
print(f"init_eval_ppl  mean    = {init_mean:.4f}")
print(f"best_eval_ppl  mean    = {best_mean:.4f} ± {best_std:.4f}")
print(f"final_eval_ppl mean    = {final_mean:.4f} ± {final_std:.4f}  (n={len(seed_results)} seeds)")
print(f"per-seed best          = {[round(r['best_eval_ppl'], 4) for r in seed_results]}")
print(f"Δ vs c223 K=128 (48.989) = {delta_vs_c223:+.4f}")
print(f"Δ vs c239 LoRA  (47.692) = {delta_vs_c239:+.4f}")
print(f"Δ vs float baseline      = {delta_vs_float:+.4f} ({delta_vs_float_pct:+.2f}%)")
print(f"thresholds               = CONFIRM ≤ {CONFIRM_THRESH}, REFUTE ≥ {REFUTE_THRESH}")
print(f"VERDICT: {verdict}")


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 246,
    "item": 256,
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
        "RANK": RANK,
        "fallback_seq_len": fallback_seq_len,
    },
    "baseline_ppl_eval": float(baseline_ppl_eval),
    "post_mu_calib_ppl": float(post_mu_calib_ppl),
    "post_mu_eval_ppl": float(post_mu_eval_ppl),
    "post_compress_eval_ppl": float(post_compress_eval_ppl),
    "cycle223_k128_ref_ppl": float(C223_K128_BASELINE),
    "cycle239_lora_ref_ppl": float(C239_LORA_BASELINE),
    "cycle220_float_ref_ppl": float(C220_FLOAT_REF),
    "init_eval_ppl_mean": init_mean,
    "best_eval_ppl_mean": best_mean,
    "best_eval_ppl_std": best_std,
    "final_eval_ppl_mean": final_mean,
    "final_eval_ppl_std": final_std,
    "delta_vs_c223_k128": float(delta_vs_c223),
    "delta_vs_c239_lora": float(delta_vs_c239),
    "delta_vs_float": float(delta_vs_float),
    "delta_vs_float_pct": float(delta_vs_float_pct),
    "confirm_thresh": float(CONFIRM_THRESH),
    "refute_thresh": float(REFUTE_THRESH),
    "seeds": seed_results,
    "trainable_params": int(n_trainable),
    "trainable_breakdown": {
        "float_blocks_lora": 9 * (3072 * RANK + RANK * 768),
        "compressed_blocks_lora": 3 * (K * RANK + RANK * 768),
        "compressed_blocks_W_out_b_out": 3 * (768 * K + 768),
    },
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

summary = f"""# Cycle 246 — gelu_noneg + μ + LoRA r=8 + K=128 W_out on [1,4,7] (item 256)

**Verdict: {verdict}**

Combines cycle 239's gelu_noneg + per-channel μ_l + LoRA r=8 on every block's
`mlp.c_proj` recipe (47.692 ± 0.143 held-out, all 12 blocks) with cycle 223's
K=128 topk+lstsq W_out channel compression on BLOCKS=[1,4,7] (48.989 alone).
Tests whether activation fix and channel compression compose (≤ 47.0) or trade
(≥ 49.5).

**Install order.** (1) μ precomputation on CALIB; (2) patch 12 × mlp.act with
`GeluNonegCorrected(μ_l)`; (3) collect (ln_out, mlp_out) on the patched model
and fit K=128 topk+lstsq against the patched target on [1,4,7], replacing
those mlp modules with `CompressedMLPWithLoRA` (built-in LoRA r=8 residual);
(4) wrap the 9 float blocks' `mlp.c_proj` in `LoraConv1D` (r=8);
(5) joint Adam LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, best-state
restore. Trainable surface = LoRA A/B across all 12 blocks ∪ W_out + b_out on
the 3 compressed blocks.

| Variant | PPL | source |
|---|---|---|
| Float baseline (held-out) | {baseline_ppl_eval:.3f} | this run |
| cycle 220 float ref | {C220_FLOAT_REF:.3f} | reference |
| μ-only CALIB | {post_mu_calib_ppl:.3f} | this run |
| μ-only EVAL (held-out) | {post_mu_eval_ppl:.3f} | this run |
| μ + K=128 [1,4,7] EVAL (LoRA=0 init) | {post_compress_eval_ppl:.3f} | this run |
| cycle 223 K=128 [1,4,7] W_out-only | {C223_K128_BASELINE:.3f} | no μ, no LoRA |
| cycle 239 gelu_noneg+μ+LoRA r=8 (all 12) | {C239_LORA_BASELINE:.3f} | no K compression |
| **cycle 246 combined best_mean ± std** | **{best_mean:.3f} ± {best_std:.3f}** | this run, n={len(seed_results)} seeds |
| cycle 246 combined final_mean ± std | {final_mean:.3f} ± {final_std:.3f} | after full {N_STEPS} steps |

**Per-seed results (final_eval_ppl after best-state restore):**

| seed | init_eval_ppl | best_eval_ppl (step) | final_eval_ppl |
|---|---|---|---|
{per_seed_lines}

Thresholds: CONFIRMED ≤ {CONFIRM_THRESH}, REFUTED ≥ {REFUTE_THRESH}.
Δ vs float baseline: {delta_vs_float:+.3f} PPL ({delta_vs_float_pct:+.2f}%).
Δ vs cycle 223 K=128 (48.989): {delta_vs_c223:+.3f} PPL.
Δ vs cycle 239 LoRA-only (47.692): {delta_vs_c239:+.3f} PPL.

**Trainable budget.** {n_trainable:,} params
 = 9 × (3072×{RANK} + {RANK}×768) [float-block LoRA]
 + 3 × ({K}×{RANK} + {RANK}×768) [compressed-block LoRA]
 + 3 × (768×{K} + 768) [compressed W_out + b_out].

**Sanity.**
- `len(calib_ids) = {len(calib_ids)}`, `len(eval_ids) = {len(eval_ids)}`, disjoint by object id and tensor equality.
- μ ≤ 0 elementwise across all 12 blocks (max = {max_positive:.2e}).
- LoRA-zero init eval PPL matches post-compress eval PPL ({post_compress_eval_ppl:.4f}) within 1e-3 across all seeds.
- Fallback SEQ_LEN: {fallback_seq_len if fallback_seq_len else 'none (SEQ_LEN=128)'}.
"""
with open(SUMMARY_PATH, "w") as f:
    f.write(summary)
print(f"Saved {SUMMARY_PATH}")
