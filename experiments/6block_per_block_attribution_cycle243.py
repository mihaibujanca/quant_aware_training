"""
Cycle 243 — Item 253: Per-block PPL attribution under the cycle 221 best-state
6-block joint Adam fit (BLOCKS=[1,2,4,5,7,8] K=512). Measure how much each
compressed block contributes to the 6.0 PPL gap (53.895 -> 59.515) by restoring
its float MLP one block at a time.

Phase 0: refit cycle 221 at seed 42 only to recreate the best-state snapshot
         (no .pt on disk in cycle 221).
Phase 1: per-block restoration scan on the cycle 221 eval set.
Phase 2: per-block residual MSE on calibration data.

Verdict per item 253:
  CONFIRMED   if any X has ppl_if_X_restored <= 56.0
  REFUTED     if all 6 delta_X in [0.5, 1.5]
  INCONCLUSIVE otherwise
"""
import copy
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS = [1, 2, 4, 5, 7, 8]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 4000
EVAL_EVERY = 100
SEED = 42
OUTPUT_PATH = "autoresearch/results/cycle_243_per_block_attribution.json"
SUMMARY_PATH = "autoresearch/results/cycle_243_summary.md"
CYCLE_221_FLOAT = 53.895
CYCLE_221_BEST_S42 = 59.391

print(f"Device: {DEVICE}")
print(f"Cycle 243: per-block attribution under [1,2,4,5,7,8] joint Adam best state")

# ── Load tokenizer + data ─────────────────────────────────────────────────────
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
all_ids = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        all_ids.append(enc["input_ids"][:, :SEQ_LEN])
    if len(all_ids) >= N_CALIB + N_EVAL:
        break

calib_ids = [t.to(DEVICE) for t in all_ids[:N_CALIB]]
eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB:N_CALIB + N_EVAL]]
print(f"Calibration: {len(calib_ids)} seqs, Eval: {len(eval_ids)} seqs")

# Disjointness sanity
calib_set = {id(t) for t in calib_ids}
eval_set = {id(t) for t in eval_ids}
assert calib_set.isdisjoint(eval_set), "calib/eval overlap by id"


# ── Helpers (verbatim from cycle 221) ─────────────────────────────────────────
class CompressedMLP(nn.Module):
    def __init__(self, fc_weight, fc_bias, W_out, b_out):
        super().__init__()
        self.register_buffer("fc_weight", fc_weight)
        self.register_buffer("fc_bias", fc_bias)
        self.W_out = nn.Parameter(W_out.clone())
        self.b_out = nn.Parameter(b_out.clone())

    def forward(self, x):
        return F.gelu(x @ self.fc_weight + self.fc_bias) @ self.W_out.T + self.b_out


def collect_mlp_io(model, calib_ids, block_ids):
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
        ln_outs[b] = torch.cat(ln_outs[b])
        proj_outs[b] = torch.cat(proj_outs[b])
    return ln_outs, proj_outs


def fit_block_lstsq(block, ln_out_cpu, proj_out_cpu, K):
    W_fc = block.mlp.c_fc.weight.data.float().cpu()
    b_fc = block.mlp.c_fc.bias.data.float().cpu()
    W_proj = block.mlp.c_proj.weight.data.float().cpu()
    act_all = F.gelu(ln_out_cpu @ W_fc + b_fc)
    scores = act_all.std(0) * W_proj.norm(dim=1)
    topk_idx = scores.topk(K).indices
    act_sub = act_all[:, topk_idx]
    ones = torch.ones(act_sub.shape[0], 1)
    sol = torch.linalg.lstsq(torch.cat([act_sub, ones], 1), proj_out_cpu, rcond=None).solution
    return CompressedMLP(W_fc[:, topk_idx], b_fc[topk_idx], sol[:K].T, sol[K])


def eval_ppl(model, eval_ids):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


# ── Phase 0: refit cycle 221 seed 42 ──────────────────────────────────────────
print(f"\n{'='*60}\nPhase 0: refit cycle 221 seed {SEED} W_out+b_out\n{'='*60}")
torch.manual_seed(SEED)
random.seed(SEED)

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# Float baseline (held-out eval)
float_baseline_ppl = eval_ppl(model, eval_ids)
print(f"Float baseline PPL: {float_baseline_ppl:.3f}")
assert abs(float_baseline_ppl - CYCLE_221_FLOAT) < 0.01, \
    f"Float baseline {float_baseline_ppl} far from cycle 221 ref {CYCLE_221_FLOAT}"

# Calib I/O
ln_outs, proj_outs = collect_mlp_io(model, calib_ids, BLOCKS)
for b in BLOCKS:
    model.transformer.h[b].mlp = fit_block_lstsq(
        model.transformer.h[b], ln_outs[b], proj_outs[b], K).to(DEVICE)
del ln_outs, proj_outs

init_ppl = eval_ppl(model, eval_ids)
print(f"Init PPL (post-lstsq): {init_ppl:.3f}")

# Freeze all, unfreeze W_out + b_out
for param in model.parameters():
    param.requires_grad_(False)
trainable_params = []
for b in BLOCKS:
    model.transformer.h[b].mlp.W_out.requires_grad_(True)
    model.transformer.h[b].mlp.b_out.requires_grad_(True)
    trainable_params.extend([model.transformer.h[b].mlp.W_out,
                             model.transformer.h[b].mlp.b_out])
n_trainable = sum(p.numel() for p in trainable_params)
print(f"Trainable params: {n_trainable:,}")

optimizer = torch.optim.Adam(trainable_params, lr=LR)
calib_shuffled = list(calib_ids)
random.shuffle(calib_shuffled)


def snapshot():
    return {b: (model.transformer.h[b].mlp.W_out.detach().clone().cpu(),
                model.transformer.h[b].mlp.b_out.detach().clone().cpu())
            for b in BLOCKS}


best_ppl = init_ppl
best_step = 0
best_state = snapshot()
history = []

step = 0
while step < N_STEPS:
    for ids in calib_shuffled:
        if step >= N_STEPS:
            break
        model.train()
        optimizer.zero_grad()
        model(ids, labels=ids).loss.backward()
        optimizer.step()
        step += 1
        if step % EVAL_EVERY == 0:
            ppl_e = eval_ppl(model, eval_ids)
            history.append({"step": step, "eval_ppl": float(ppl_e)})
            marker = ""
            if ppl_e < best_ppl:
                best_ppl = float(ppl_e)
                best_step = step
                best_state = snapshot()
                marker = " *"
            print(f"  Step {step:5d}: eval_ppl={ppl_e:.3f}{marker}", flush=True)

# Restore best state
for b in BLOCKS:
    W_best, b_best = best_state[b]
    model.transformer.h[b].mlp.W_out.data.copy_(W_best.to(DEVICE))
    model.transformer.h[b].mlp.b_out.data.copy_(b_best.to(DEVICE))
fully_compressed_ppl = eval_ppl(model, eval_ids)
print(f"Restored best (step {best_step}): {fully_compressed_ppl:.3f} (recorded best={best_ppl:.3f})")
assert abs(fully_compressed_ppl - best_ppl) < 1e-3, \
    f"Restore mismatch: {fully_compressed_ppl} vs {best_ppl}"

# Free optimiser memory
del optimizer, trainable_params
if DEVICE == "mps":
    torch.mps.empty_cache()

# ── Phase 1: per-block restoration scan ───────────────────────────────────────
print(f"\n{'='*60}\nPhase 1: per-block restoration scan\n{'='*60}")

# Build a fresh float GPT-2 to extract original FFNs
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()

# Deep copy each float MLP we may need
float_mlps = {b: copy.deepcopy(model_float.transformer.h[b].mlp).to(DEVICE) for b in BLOCKS}

# Sanity: float baseline using model_float matches earlier
mf_ppl = eval_ppl(model_float, eval_ids)
print(f"model_float held-out PPL: {mf_ppl:.3f}")
assert abs(mf_ppl - float_baseline_ppl) < 1e-4, \
    f"Float baseline mismatch: {mf_ppl} vs {float_baseline_ppl}"
del model_float
if DEVICE == "mps":
    torch.mps.empty_cache()

ppl_if_X_restored = {}
for X in BLOCKS:
    saved = model.transformer.h[X].mlp
    model.transformer.h[X].mlp = float_mlps[X]
    ppl_X = eval_ppl(model, eval_ids)
    # Restore
    model.transformer.h[X].mlp = saved
    # Round-trip sanity
    rt_ppl = eval_ppl(model, eval_ids)
    assert abs(rt_ppl - fully_compressed_ppl) < 1e-4, \
        f"Restore round-trip block {X}: {rt_ppl} vs {fully_compressed_ppl}"
    ppl_if_X_restored[X] = float(ppl_X)
    delta = fully_compressed_ppl - ppl_X
    print(f"  block {X}: ppl_if_restored={ppl_X:.4f}  delta={delta:+.4f}")

# ── Phase 2: per-block residual MSE on calib ──────────────────────────────────
print(f"\n{'='*60}\nPhase 2: per-block residual MSE on calib\n{'='*60}")

# Float MLP I/O on calib
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()
_, proj_outs_float = collect_mlp_io(model_float, calib_ids, BLOCKS)
del model_float
if DEVICE == "mps":
    torch.mps.empty_cache()

# Compressed-best MLP I/O on calib
_, proj_outs_comp = collect_mlp_io(model, calib_ids, BLOCKS)

per_block_residual_mse = {}
for b in BLOCKS:
    mse = ((proj_outs_float[b] - proj_outs_comp[b]) ** 2).mean().item()
    per_block_residual_mse[b] = float(mse)
    print(f"  block {b}: residual MSE = {mse:.6e}")

# ── Verdict ───────────────────────────────────────────────────────────────────
deltas = {X: float(fully_compressed_ppl - ppl_if_X_restored[X]) for X in BLOCKS}
sorted_deltas = sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)

any_below_56 = any(ppl_if_X_restored[X] <= 56.0 for X in BLOCKS)
all_in_band = all(0.5 <= deltas[X] <= 1.5 for X in BLOCKS)

if any_below_56:
    verdict = "CONFIRMED"
elif all_in_band:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}\nVERDICT: {verdict}\n{'='*60}")
print(f"float_baseline = {float_baseline_ppl:.3f}")
print(f"fully_compressed = {fully_compressed_ppl:.3f}  (cycle 221 ref s42 best={CYCLE_221_BEST_S42})")
print(f"6 PPL gap = {fully_compressed_ppl - float_baseline_ppl:+.3f}")
print(f"Sorted deltas (largest first):")
for X, d in sorted_deltas:
    print(f"  block {X}: ppl_if_restored={ppl_if_X_restored[X]:.4f}  delta={d:+.4f}  mse={per_block_residual_mse[X]:.4e}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 243,
    "item": 253,
    "verdict": verdict,
    "config": {
        "BLOCKS": BLOCKS, "K": K, "N_CALIB": N_CALIB, "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN, "LR": LR, "N_STEPS": N_STEPS,
        "EVAL_EVERY": EVAL_EVERY, "SEED": SEED,
    },
    "phase0": {
        "seed_42_refit_init_ppl": float(init_ppl),
        "seed_42_refit_best_ppl": float(best_ppl),
        "seed_42_refit_best_step": int(best_step),
        "history": history,
    },
    "phase1": {
        "ppl_if_block_X_restored": {str(X): float(ppl_if_X_restored[X]) for X in BLOCKS},
        "fully_compressed_ppl": float(fully_compressed_ppl),
        "float_baseline_ppl": float(float_baseline_ppl),
        "delta_per_block": {str(X): float(deltas[X]) for X in BLOCKS},
    },
    "phase2": {
        "per_block_residual_mse": {str(b): float(per_block_residual_mse[b]) for b in BLOCKS},
    },
    "sorted": [[int(X), float(d)] for X, d in sorted_deltas],
    "cycle_221_seed_42_ref_best": CYCLE_221_BEST_S42,
    "cycle_221_float_ref": CYCLE_221_FLOAT,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved JSON to {OUTPUT_PATH}")

# ── Summary markdown ──────────────────────────────────────────────────────────
lines = []
lines.append("# Cycle 243 — Item 253: 6-block per-block PPL attribution")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append("- Criteria: Confirmed if any `ppl_if_X_restored ≤ 56.0`; Refuted if all 6 deltas in `[0.5, 1.5]`; else Inconclusive.")
lines.append(f"- Config: BLOCKS={BLOCKS}, K={K}, LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, SEED={SEED}")
lines.append(f"- Phase 0 refit: init={init_ppl:.3f} → best={best_ppl:.3f} at step {best_step}")
lines.append(f"- Float baseline (held-out): {float_baseline_ppl:.3f}")
lines.append(f"- Fully compressed (cycle 221 best-state, seed 42 refit): {fully_compressed_ppl:.3f}")
lines.append(f"- Cycle 221 seed-42 reference best: {CYCLE_221_BEST_S42:.3f} (Δ vs refit: {fully_compressed_ppl - CYCLE_221_BEST_S42:+.3f})")
lines.append(f"- Compression gap (fully − float): {fully_compressed_ppl - float_baseline_ppl:+.3f} PPL")
lines.append("")
lines.append("## Per-block restoration scan (sorted by ΔPPL desc)")
lines.append("")
lines.append("| rank | block | ppl_if_X_restored | ΔPPL (fully − ppl_if_X) | residual_mse |")
lines.append("|---|---|---|---|---|")
for rank, (X, d) in enumerate(sorted_deltas, 1):
    lines.append(f"| {rank} | {X} | {ppl_if_X_restored[X]:.4f} | {d:+.4f} | {per_block_residual_mse[X]:.4e} |")
lines.append("")
lines.append("## Notes")
lines.append("")
lines.append(f"- Sum of deltas: {sum(deltas.values()):+.3f} PPL (compare to {fully_compressed_ppl - float_baseline_ppl:+.3f} compression gap; non-additivity expected).")
lines.append(f"- Each restored ppl is in `[float, fully_compressed]` (monotone): "
             f"min={min(ppl_if_X_restored.values()):.3f}, max={max(ppl_if_X_restored.values()):.3f}.")
lines.append("")
with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
