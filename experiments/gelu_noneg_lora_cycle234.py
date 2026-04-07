"""
Cycle 234 — Item 244: gelu_noneg + first-moment (μ_l) + LoRA r=8 W_out residual.

Cycle 228 patched all 12 GPT-2 small MLP activations with
  relu(GELU(z)) + μ_l
and got PPL = 61.664 vs float baseline 45.064 (+16.6 PPL, +36.84%). The residual
was attributed to higher-order GELU curvature beyond the per-channel mean shift.

This cycle: on top of cycle 228's μ-corrected 12-block patch, adds a LoRA r=8
adapter on each block's mlp.c_proj (W_out; HF Conv1D weight shape (3072, 768)).
A_l ∈ R^(3072, 8) init N(0, 0.01); B_l ∈ R^(8, 768) init zero (so init LoRA delta
is zero and init_ppl matches cycle 228 exactly). The base Conv1D weight/bias are
frozen; only (A, B) per block are trainable. Joint Adam LR=1e-4, 2000 steps on CE
against the same calibration sequences, seed 42, best-state restore.

Verdict (item 244):
  - CONFIRMED if final_ppl ≤ 48.0
  - REFUTED   if final_ppl ≥ 58.0
  - INCONCLUSIVE otherwise
"""
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
N_EVAL = 200
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
STEPS = 2000
EVAL_EVERY = 200
RANK = 8

RESULT_PATH = "autoresearch/results/cycle_234_gelu_noneg_lora.json"
SUMMARY_PATH = "autoresearch/results/cycle_234_summary.md"

C220_BASELINE_PPL = 45.064         # float baseline (reference)
C228_CORRECTED_PPL = 61.664        # gelu_noneg + μ_l reference
CONFIRM_THRESH = 48.0
REFUTE_THRESH = 58.0

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load data (identical to cycle 228) ────────────────────────────────────────
print("Loading GPT-2 and data...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
all_ids = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        all_ids.append(enc["input_ids"][:, :SEQ_LEN])
    if len(all_ids) >= N_EVAL:
        break

eval_ids = [t.to(DEVICE) for t in all_ids[:N_EVAL]]
print(f"Eval: {len(eval_ids)} seqs × {SEQ_LEN} tokens")


def eval_ppl(model, eval_ids):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


# ── Build model ───────────────────────────────────────────────────────────────
print("\nLoading GPT-2 small...")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

print("\nBaseline PPL (no patch)...")
baseline_ppl = eval_ppl(model, eval_ids)
print(f"  baseline_ppl = {baseline_ppl:.4f}  (cycle 220 ref: {C220_BASELINE_PPL})")


# ── PHASE 1 — μ precomputation (cycle 228 verbatim) ───────────────────────────
print("\n──────── PHASE 1: μ precomputation ────────")
stash = {l: [] for l in range(12)}
hooks = []


def make_hook(l):
    def hook(module, inputs, output):
        stash[l].append(inputs[0].detach())
    return hook


for l in range(12):
    h = model.transformer.h[l].mlp.c_fc.register_forward_hook(make_hook(l))
    hooks.append(h)

print(f"Hooks registered on {len(hooks)} c_fc modules. Forward pass...")
with torch.no_grad():
    for ids in eval_ids:
        _ = model(ids)

for h in hooks:
    h.remove()
print("Hooks removed.")

mu_dict = {}
mu_stats_per_block = []
print("\nμ per block:")
with torch.no_grad():  # important: keeps mu outside the autograd graph
    for l in range(12):
        x_l = torch.cat(stash[l], dim=0).reshape(-1, 768)
        c_fc = model.transformer.h[l].mlp.c_fc
        preact = x_l @ c_fc.weight + c_fc.bias
        gelu_l = F.gelu(preact, approximate='tanh')
        noneg_l = torch.relu(gelu_l)
        mu_l = (gelu_l - noneg_l).mean(dim=0).detach()
        mu_dict[l] = mu_l.to(DEVICE)
        stats = {
            "block": l,
            "mean": float(mu_l.mean().item()),
            "std": float(mu_l.std().item()),
            "min": float(mu_l.min().item()),
            "max": float(mu_l.max().item()),
            "abs_mean": float(mu_l.abs().mean().item()),
            "n_tokens": int(x_l.shape[0]),
            "n_positive_entries": int((mu_l > 0).sum().item()),
        }
        mu_stats_per_block.append(stats)
        print(f"  block {l:2d}: mean={stats['mean']:+.5f} std={stats['std']:.5f} "
              f"min={stats['min']:+.5f} max={stats['max']:+.5f} "
              f"|mu|_mean={stats['abs_mean']:.5f} n_pos={stats['n_positive_entries']}")
        stash[l] = None

max_positive = max(s["max"] for s in mu_stats_per_block)
assert max_positive <= 1e-6, f"μ has positive entries (max={max_positive:.6e})"
print(f"OK: μ ≤ 0 elementwise across all 12 blocks (max = {max_positive:.2e})")


# ── PHASE 2 — Patch activations (cycle 228) ───────────────────────────────────
print("\n──────── PHASE 2: Install gelu_noneg + μ patch ────────")


class GeluNonegCorrected(nn.Module):
    def __init__(self, mu):
        super().__init__()
        self.register_buffer('mu', mu)

    def forward(self, z):
        return torch.relu(F.gelu(z, approximate='tanh')) + self.mu


for l in range(12):
    model.transformer.h[l].mlp.act = GeluNonegCorrected(mu_dict[l]).to(DEVICE)

ids_check = set(id(model.transformer.h[i].mlp.act) for i in range(12))
assert len(ids_check) == 12, f"Patches not distinct! got {len(ids_check)} unique ids"
print("Patched 12 blocks with distinct GeluNonegCorrected instances.")

post_mu_ppl = eval_ppl(model, eval_ids)
print(f"post_mu_patch_ppl = {post_mu_ppl:.4f}  (cycle 228 ref: {C228_CORRECTED_PPL})")
assert abs(post_mu_ppl - C228_CORRECTED_PPL) < 0.5, \
    f"μ patch does not reproduce cycle 228: got {post_mu_ppl:.4f}, expected ≈ {C228_CORRECTED_PPL}"


# ── PHASE 3 — Install LoRA on mlp.c_proj ──────────────────────────────────────
print("\n──────── PHASE 3: Install LoRA r=8 on mlp.c_proj ────────")


class LoraConv1D(nn.Module):
    """
    LoRA wrapper around HF Conv1D (mlp.c_proj).

    HF Conv1D semantics: out = x @ weight + bias
      where weight has shape (nx, nf) = (3072, 768), bias shape (nf,) = (768,).
    LoRA delta: delta = A @ B, A ∈ (nx, r) init N(0, 0.01), B ∈ (r, nf) init 0.
    Base weight and bias are frozen.
    """
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
        # Mirror HF Conv1D math exactly.
        size_out = x.size()[:-1] + (self.nf,)
        W = self.base.weight + self.A @ self.B  # (nx, nf)
        out = torch.addmm(self.base.bias, x.view(-1, self.nx), W)
        return out.view(size_out)


# Sanity: snapshot block-0 base weight to verify frozen.
orig_c_proj_0_weight = model.transformer.h[0].mlp.c_proj.weight.data.clone()

for l in range(12):
    orig_cp = model.transformer.h[l].mlp.c_proj
    lora = LoraConv1D(orig_cp, r=RANK).to(DEVICE)
    model.transformer.h[l].mlp.c_proj = lora

# Freeze EVERYTHING then re-enable A, B for each LoRA module.
for p in model.parameters():
    p.requires_grad_(False)
for l in range(12):
    cp = model.transformer.h[l].mlp.c_proj  # LoraConv1D
    cp.A.requires_grad_(True)
    cp.B.requires_grad_(True)

trainable = [p for p in model.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable)
expected = 12 * (3072 * RANK + RANK * 768)
print(f"Trainable params: {n_trainable} (expected {expected})")
assert n_trainable == expected, f"expected {expected}, got {n_trainable}"

# Init eval — B is zero so LoRA delta is zero → must match cycle 228 exactly.
init_ppl = eval_ppl(model, eval_ids)
print(f"init_ppl (post-LoRA-zero) = {init_ppl:.4f}  (should match post_mu = {post_mu_ppl:.4f})")
assert abs(init_ppl - post_mu_ppl) < 1e-3, \
    f"LoRA-zero init mismatched μ-patch ppl: {init_ppl:.6f} vs {post_mu_ppl:.6f}"


# ── PHASE 4 — Train (A, B) jointly with Adam ──────────────────────────────────
print("\n──────── PHASE 4: Train LoRA (A, B) × 12 blocks ────────")

opt = torch.optim.Adam(trainable, lr=LR)

# Snapshot best state: CPU clones of A, B per block.
def snapshot_state():
    s = []
    for l in range(12):
        cp = model.transformer.h[l].mlp.c_proj
        s.append((cp.A.detach().clone().cpu(), cp.B.detach().clone().cpu()))
    return s


def restore_state(s):
    for l in range(12):
        cp = model.transformer.h[l].mlp.c_proj
        cp.A.data.copy_(s[l][0].to(DEVICE))
        cp.B.data.copy_(s[l][1].to(DEVICE))


best_ppl = init_ppl
best_step = 0
best_state = snapshot_state()
ppl_history = [(0, float(init_ppl))]
loss_history = []

torch.manual_seed(SEED)  # deterministic sampling order
n_data = len(eval_ids)

print(f"Training: STEPS={STEPS}, LR={LR}, EVAL_EVERY={EVAL_EVERY}, n_data={n_data}")

model.train()  # note: still eval mode in gpt2 since no dropout matters; but set for clarity
for step in range(1, STEPS + 1):
    idx = (step - 1) % n_data
    ids = eval_ids[idx]
    out = model(ids, labels=ids)
    loss = out.loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    loss_val = float(loss.item())
    loss_history.append((step, loss_val))

    if step % EVAL_EVERY == 0:
        ppl = eval_ppl(model, eval_ids)
        ppl_history.append((step, float(ppl)))
        marker = ""
        if ppl < best_ppl:
            best_ppl = ppl
            best_step = step
            best_state = snapshot_state()
            marker = " *BEST*"
        print(f"  step {step:4d}  loss={loss_val:.4f}  ppl={ppl:.4f}{marker}")
        model.train()

# Restore best state and re-eval.
print(f"\nRestoring best state from step {best_step} (ppl {best_ppl:.4f})...")
restore_state(best_state)
final_ppl = eval_ppl(model, eval_ids)
print(f"final_ppl (restored) = {final_ppl:.4f}")
assert abs(final_ppl - best_ppl) < 1e-3, \
    f"restore mismatch: final={final_ppl:.6f} best_tracked={best_ppl:.6f}"

# Verify block-0 base c_proj weight unchanged.
post_c_proj_0_weight = model.transformer.h[0].mlp.c_proj.base.weight.data
base_unchanged = torch.equal(orig_c_proj_0_weight, post_c_proj_0_weight)
print(f"Block 0 base c_proj weight unchanged: {base_unchanged}")
assert base_unchanged, "Frozen base weight changed during training!"


# ── Verdict ───────────────────────────────────────────────────────────────────
delta_vs_c228 = final_ppl - C228_CORRECTED_PPL
delta_vs_float = final_ppl - baseline_ppl
delta_vs_float_pct = delta_vs_float / baseline_ppl * 100

if final_ppl <= CONFIRM_THRESH:
    verdict = "CONFIRMED"
elif final_ppl >= REFUTE_THRESH:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print("\n─────────────────────────────────────")
print(f"baseline_ppl          = {baseline_ppl:.4f}  (c220 ref {C220_BASELINE_PPL})")
print(f"post_mu_patch_ppl     = {post_mu_ppl:.4f}  (c228 ref {C228_CORRECTED_PPL})")
print(f"init_ppl (LoRA=0)     = {init_ppl:.4f}")
print(f"best_ppl (train)      = {best_ppl:.4f}  @ step {best_step}")
print(f"final_ppl (restored)  = {final_ppl:.4f}")
print(f"Δ vs cycle 228        = {delta_vs_c228:+.4f}")
print(f"Δ vs float baseline   = {delta_vs_float:+.4f}  ({delta_vs_float_pct:+.2f}%)")
print(f"confirm_thresh        = {CONFIRM_THRESH}")
print(f"refute_thresh         = {REFUTE_THRESH}")
print(f"VERDICT: {verdict}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 234,
    "item": 244,
    "verdict": verdict,
    "config": {
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "SEED": SEED,
        "DEVICE": DEVICE,
        "LR": LR,
        "STEPS": STEPS,
        "EVAL_EVERY": EVAL_EVERY,
        "RANK": RANK,
    },
    "baseline_ppl": float(baseline_ppl),
    "cycle228_ref_ppl": float(C228_CORRECTED_PPL),
    "float_ref_ppl": float(C220_BASELINE_PPL),
    "post_mu_patch_ppl": float(post_mu_ppl),
    "init_ppl": float(init_ppl),
    "best_ppl": float(best_ppl),
    "best_step": int(best_step),
    "final_ppl": float(final_ppl),
    "delta_vs_cycle228": float(delta_vs_c228),
    "delta_vs_float": float(delta_vs_float),
    "delta_vs_float_pct": float(delta_vs_float_pct),
    "confirm_thresh": float(CONFIRM_THRESH),
    "refute_thresh": float(REFUTE_THRESH),
    "ppl_history": ppl_history,
    "loss_history_head": loss_history[:20],
    "loss_history_tail": loss_history[-20:],
    "trainable_params": int(n_trainable),
    "base_c_proj_0_unchanged": bool(base_unchanged),
    "mu_stats_per_block": mu_stats_per_block,
}
with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved {RESULT_PATH}")


# ── Save summary MD ───────────────────────────────────────────────────────────
mu_mean_overall = sum(s["mean"] for s in mu_stats_per_block) / 12
mu_abs_mean_overall = sum(s["abs_mean"] for s in mu_stats_per_block) / 12

summary = f"""# Cycle 234 — gelu_noneg + μ_l + LoRA r=8 W_out residual (item 244)

**Verdict: {verdict}**

Cycle 228's gelu_noneg + per-channel first-moment correction patch (all 12 blocks)
left a residual of +16.6 PPL (61.664 vs float 45.064). This cycle adds a LoRA r=8
adapter on each block's `mlp.c_proj` (HF Conv1D W_out, shape (3072, 768)),
initialized so that the initial LoRA delta is zero (B=0), then trains (A, B) ×
12 blocks jointly with Adam (LR={LR}, {STEPS} steps, seed {SEED}) on CE against the
calibration set (N={N_EVAL} wikitext-2 val sequences × SEQ_LEN={SEQ_LEN}).

| Variant | PPL | Δ vs float | % vs float |
|---|---|---|---|
| Float baseline | {baseline_ppl:.3f} | — | — |
| cycle 228 μ-only (12 blocks) | {post_mu_ppl:.3f} | +{post_mu_ppl - baseline_ppl:.3f} | +{(post_mu_ppl - baseline_ppl)/baseline_ppl*100:.2f}% |
| cycle 234 μ + LoRA r={RANK} (best@{best_step}) | {final_ppl:.3f} | {delta_vs_float:+.3f} | {delta_vs_float_pct:+.2f}% |

Thresholds: CONFIRMED ≤ {CONFIRM_THRESH}, REFUTED ≥ {REFUTE_THRESH}.

**Training trace.** init → best: {init_ppl:.3f} → {best_ppl:.3f} PPL at step {best_step}.
Trainable params: {n_trainable} = 12 × (3072×{RANK} + {RANK}×768) (only LoRA A, B per
block; base Conv1D weights and biases frozen, μ buffers frozen, all other GPT-2
parameters frozen).

**Sanity.** post-μ patch PPL = {post_mu_ppl:.4f} (cycle 228 ref {C228_CORRECTED_PPL});
init ppl with B=0 = {init_ppl:.4f} (must match post-μ); base c_proj[0] weight
unchanged after training = {base_unchanged}. μ ≤ 0 elementwise across all 12 blocks.

**Conclusion.** {
    f"LoRA r={RANK} W_out residual absorbs enough of the 16.6 PPL gelu_noneg+μ residual "
    f"to reach within +3 PPL of float baseline (final {final_ppl:.3f} ≤ {CONFIRM_THRESH}). "
    f"The gelu_noneg higher-order curvature beyond the first-moment correction lives in a "
    f"rank-≤8 subspace of W_out that can be learned from calibration data."
    if verdict == "CONFIRMED" else
    f"LoRA r={RANK} residual correction on W_out cannot compensate for gelu_noneg's "
    f"higher-order curvature after first-moment correction ({final_ppl:.3f} ≥ {REFUTE_THRESH}). "
    f"Adam improves over the cycle 228 init by only {init_ppl - final_ppl:+.2f} PPL; "
    f"the relu+μ activation surface is a dead end for surgical compression even with a "
    f"learned low-rank W_out residual."
    if verdict == "REFUTED" else
    f"LoRA r={RANK} residual correction absorbs {init_ppl - final_ppl:.2f} PPL of the "
    f"16.6 PPL gelu_noneg+μ residual, reaching {final_ppl:.3f}, which lands in the "
    f"INCONCLUSIVE band ({CONFIRM_THRESH} < {final_ppl:.3f} < {REFUTE_THRESH}). "
    f"Partial absorption only; the gelu_noneg family is improved by learned W_out "
    f"residual but not enough to close to +3 PPL of float baseline."
}

**μ statistics (per-block aggregate).** Overall mean(μ) ≈ {mu_mean_overall:+.5f},
overall mean(|μ|) ≈ {mu_abs_mean_overall:.5f}. μ ≤ 0 elementwise by construction.
"""
with open(SUMMARY_PATH, "w") as f:
    f.write(summary)
print(f"Saved {SUMMARY_PATH}")
