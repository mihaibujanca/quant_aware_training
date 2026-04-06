"""
Cycle 220 — Item 234: Surgical bump removal.
Patch all 12 GPT-2 small MLP activations with two surgically-defined variants
and measure full-model PPL on wikitext-2 validation. Isolate the negative leak
(gelu_noneg) from the positive-zone bump (gelu_nobump).
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
RESULT_PATH = "autoresearch/results/cycle_220_surgical_bump.json"
SUMMARY_PATH = "autoresearch/results/cycle_220_summary.md"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load data (reuse cycle 206 loader pattern) ────────────────────────────────
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


# ── Eval function (reuse cycle 206 pattern) ───────────────────────────────────
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


# ── Surgical activation variants ──────────────────────────────────────────────
class GeluNoneg(nn.Module):
    """Clip the negative leak. Positive-zone bump preserved."""
    def forward(self, z):
        return torch.relu(F.gelu(z, approximate='tanh'))


class GeluNobump(nn.Module):
    """Remove the positive-zone bump. Piecewise linear with continuous pieces.
    z < -2 → 0;  -2 ≤ z ≤ 2 → 0.5 * (z + 2);  z > 2 → z
    Continuity: z=-2 → 0 ✓;  z=2 → 0.5·4 = 2 = z ✓.
    """
    def forward(self, z):
        hi = (z > 2).float()
        lo = (z < -2).float()
        mid = ((z >= -2) & (z <= 2)).float()
        return hi * z + mid * (0.5 * (z + 2)) + lo * 0.0


# ── Sanity print ──────────────────────────────────────────────────────────────
print("\nActivation sanity at z ∈ {-3, -1, 0, 1, 3}:")
z_anchor = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
gelu_vals = F.gelu(z_anchor, approximate='tanh')
noneg_vals = GeluNoneg()(z_anchor)
nobump_vals = GeluNobump()(z_anchor)
for i, z in enumerate(z_anchor.tolist()):
    print(f"  z={z:+.1f}  gelu={gelu_vals[i].item():+.4f}  "
          f"noneg={noneg_vals[i].item():+.4f}  nobump={nobump_vals[i].item():+.4f}")


# ── Build model and set up patch protocol ────────────────────────────────────
print("\nLoading GPT-2 small...")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

originals = [model.transformer.h[i].mlp.act for i in range(12)]
print(f"Original activation type: {type(originals[0]).__name__}")


def patch_all(activation_cls):
    m = activation_cls().to(DEVICE)
    for i in range(12):
        model.transformer.h[i].mlp.act = m


def restore():
    for i in range(12):
        model.transformer.h[i].mlp.act = originals[i]


# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\nBaseline PPL (no patch)...")
baseline_ppl = eval_ppl(model, eval_ids)
print(f"  baseline_ppl = {baseline_ppl:.4f}")

print("\nPatch GeluNoneg → eval...")
patch_all(GeluNoneg)
noneg_ppl = eval_ppl(model, eval_ids)
print(f"  gelu_noneg_ppl = {noneg_ppl:.4f}")
restore()

print("\nPatch GeluNobump → eval...")
patch_all(GeluNobump)
nobump_ppl = eval_ppl(model, eval_ids)
print(f"  gelu_nobump_ppl = {nobump_ppl:.4f}")
restore()

print("\nRestore sanity check — re-eval baseline...")
sanity_ppl = eval_ppl(model, eval_ids)
print(f"  sanity_ppl = {sanity_ppl:.4f}  (should equal baseline)")
assert abs(sanity_ppl - baseline_ppl) < 1e-4, \
    f"Restore failed: sanity={sanity_ppl:.6f} baseline={baseline_ppl:.6f}"

# ── Verdict logic (relative thresholds from plan) ─────────────────────────────
low_thresh = baseline_ppl * 1.10
high_thresh = baseline_ppl * 1.55

noneg_delta = noneg_ppl - baseline_ppl
noneg_pct = noneg_delta / baseline_ppl * 100
nobump_delta = nobump_ppl - baseline_ppl
nobump_pct = nobump_delta / baseline_ppl * 100

if (noneg_ppl <= low_thresh and nobump_ppl >= high_thresh) or \
   (nobump_ppl <= low_thresh and noneg_ppl >= high_thresh):
    verdict = "CONFIRMED"
elif abs(noneg_ppl - nobump_ppl) <= 3:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print("\n─────────────────────────────────────")
print(f"baseline_ppl     = {baseline_ppl:.4f}")
print(f"low_thresh  (×1.10) = {low_thresh:.4f}")
print(f"high_thresh (×1.55) = {high_thresh:.4f}")
print(f"gelu_noneg_ppl   = {noneg_ppl:.4f}  (Δ={noneg_delta:+.4f}, {noneg_pct:+.2f}%)")
print(f"gelu_nobump_ppl  = {nobump_ppl:.4f}  (Δ={nobump_delta:+.4f}, {nobump_pct:+.2f}%)")
print(f"|noneg - nobump| = {abs(noneg_ppl - nobump_ppl):.4f}")
print(f"VERDICT: {verdict}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 220,
    "item": 234,
    "verdict": verdict,
    "baseline_ppl": float(baseline_ppl),
    "gelu_noneg_ppl": float(noneg_ppl),
    "gelu_nobump_ppl": float(nobump_ppl),
    "noneg_delta": float(noneg_delta),
    "noneg_pct": float(noneg_pct),
    "nobump_delta": float(nobump_delta),
    "nobump_pct": float(nobump_pct),
    "low_thresh": float(low_thresh),
    "high_thresh": float(high_thresh),
    "sanity_ppl_after_restore": float(sanity_ppl),
    "prior_baselines": {
        "cycle_184_relu_bias_pct": 87.7,
        "cycle_188_c3_mean_pct": 20.33,
        "cycle_190_hybrid_pct": 27.67,
    },
    "config": {
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "SEED": SEED,
        "DEVICE": DEVICE,
    },
}
with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved {RESULT_PATH}")

# ── Save summary MD ───────────────────────────────────────────────────────────
summary = f"""# Cycle 220 — Surgical Bump Removal (item 234)

**Verdict: {verdict}**

Patched all 12 GPT-2 small MLP activations with two surgical variants to
isolate the functional role of the negative leak vs the positive-zone bump.
Eval: wikitext-2 validation, N={N_EVAL}, SEQ_LEN={SEQ_LEN}, seed {SEED}.

| Variant | PPL | Δ vs baseline | % vs baseline |
|---|---|---|---|
| Baseline (float GELU) | {baseline_ppl:.3f} | — | — |
| Cycle 190 hybrid (GELU_neg + ReLU_pos) | — | — | +27.67% (prior) |
| GELU_noneg (clip negative leak) | {noneg_ppl:.3f} | {noneg_delta:+.3f} | {noneg_pct:+.2f}% |
| GELU_nobump (linearize positive bump) | {nobump_ppl:.3f} | {nobump_delta:+.3f} | {nobump_pct:+.2f}% |

Thresholds (relative to in-cycle baseline): low=×1.10={low_thresh:.2f}, high=×1.55={high_thresh:.2f}.

**Conclusion.** {
    "Clean separation: one sub-region carries the functional load, the other is near-linearizable."
    if verdict == "CONFIRMED" else
    "Both sub-regions degrade by similar amounts — the negative leak and positive-zone bump are entangled; neither is near-linearizable on its own."
    if verdict == "REFUTED" else
    "Both variants degrade meaningfully but neither hits the clean-separation thresholds; the load is shared between the two sub-regions."
}
"""
with open(SUMMARY_PATH, "w") as f:
    f.write(summary)
print(f"Saved {SUMMARY_PATH}")
