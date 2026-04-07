"""
Cycle 262 — Item 268: K=1 + c_fc bias trainable on 3 blocks.
BLOCKS=[1,4,7], K=1. Trainable per block: {W_out (768), b_out (768), b_fc (1 scalar)}.
Total trainable across 3 blocks: 4,611 params.

Tests whether adding a per-channel pre-GELU bias on the 1-channel topk slice
gives any improvement over the cycle 230 K=1 3-block W_out-only baseline (59.245).
At K=512 (cycles 213, 241) and K=128 (cycle 248) the c_fc bias retraining was a noop.
Hypothesis: at K=1 the W_out path is starved, so the bias finally matters.

Verdict (item 268):
  CONFIRMED    if best_mean ≤ 58.5  (beats cycle 230 by ≥ 0.745 PPL)
  REFUTED      if best_mean ≥ 59.5
  INCONCLUSIVE otherwise.
"""
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

BLOCKS = [1, 4, 7]
K = 1
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 4000
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
OUTPUT_PATH = "autoresearch/results/cycle_262_k1_wfc_bias.json"
SUMMARY_PATH = "autoresearch/results/cycle_262_summary.md"

CYCLE_230_BASELINE = 59.245  # K=1 3-block W_out-only

print(f"Device: {DEVICE}")

# ── Load tokenizer + data once ────────────────────────────────────────────────
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


# ── Compressed MLP with trainable b_fc on the K-channel slice ─────────────────
class CompressedMLPBfc(nn.Module):
    """W_fc, the K-channel slice, is FROZEN (buffer). b_fc on the slice is TRAINABLE.
    W_out (768, K) and b_out (768) trainable as in cycle 230."""
    def __init__(self, fc_weight, fc_bias, W_out, b_out):
        super().__init__()
        self.register_buffer("fc_weight", fc_weight)         # (768, K) frozen
        self.b_fc = nn.Parameter(fc_bias.clone())            # (K,)     TRAINABLE
        self.W_out = nn.Parameter(W_out.clone())             # (768, K) TRAINABLE
        self.b_out = nn.Parameter(b_out.clone())             # (768,)   TRAINABLE

    def forward(self, x):
        return F.gelu(x @ self.fc_weight + self.b_fc) @ self.W_out.T + self.b_out


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
    return CompressedMLPBfc(W_fc[:, topk_idx], b_fc[topk_idx], sol[:K].T, sol[K])


def eval_ppl(model, eval_ids):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


# ── Per-seed run ──────────────────────────────────────────────────────────────
def run_seed(seed):
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
    torch.manual_seed(seed)
    random.seed(seed)

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    ln_outs, proj_outs = collect_mlp_io(model, calib_ids, BLOCKS)
    for b in BLOCKS:
        model.transformer.h[b].mlp = fit_block_lstsq(
            model.transformer.h[b], ln_outs[b], proj_outs[b], K).to(DEVICE)
    del ln_outs, proj_outs

    init_ppl = eval_ppl(model, eval_ids)
    print(f"Init PPL: {init_ppl:.3f}")

    for param in model.parameters():
        param.requires_grad_(False)
    trainable_params = []
    for b in BLOCKS:
        mlp = model.transformer.h[b].mlp
        mlp.W_out.requires_grad_(True)
        mlp.b_out.requires_grad_(True)
        mlp.b_fc.requires_grad_(True)
        trainable_params.extend([mlp.W_out, mlp.b_out, mlp.b_fc])
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_trainable:,}")

    # Sanity: we expect 3 * (768 + 768 + 1) = 4611
    assert n_trainable == 3 * (768 + 768 + 1), f"unexpected n_trainable={n_trainable}"

    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    calib_shuffled = list(calib_ids)
    random.shuffle(calib_shuffled)

    def snapshot():
        return {b: (model.transformer.h[b].mlp.W_out.detach().clone().cpu(),
                    model.transformer.h[b].mlp.b_out.detach().clone().cpu(),
                    model.transformer.h[b].mlp.b_fc.detach().clone().cpu())
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

    final_ppl_at_4000 = eval_ppl(model, eval_ids)
    print(f"Final at step {N_STEPS}: {final_ppl_at_4000:.3f}")

    for b in BLOCKS:
        W_best, bo_best, bfc_best = best_state[b]
        model.transformer.h[b].mlp.W_out.data.copy_(W_best.to(DEVICE))
        model.transformer.h[b].mlp.b_out.data.copy_(bo_best.to(DEVICE))
        model.transformer.h[b].mlp.b_fc.data.copy_(bfc_best.to(DEVICE))
    final_ppl_restored = eval_ppl(model, eval_ids)
    print(f"Restored best (step {best_step}): {final_ppl_restored:.3f} (recorded best={best_ppl:.3f})")
    assert abs(final_ppl_restored - best_ppl) < 1e-3, \
        f"Restore mismatch: {final_ppl_restored} vs {best_ppl}"

    # Capture final b_fc deltas vs init (per block) to confirm bias actually moved
    bfc_final = {b: float(best_state[b][2].squeeze().item()) for b in BLOCKS}

    result = {
        "seed": seed,
        "init_ppl": float(init_ppl),
        "best_ppl": float(best_ppl),
        "best_step": int(best_step),
        "final_ppl_at_4000": float(final_ppl_at_4000),
        "bfc_final": bfc_final,
        "history": history,
    }
    del model, optimizer, trainable_params, best_state
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return result, n_trainable


# ── Float baseline (once) ─────────────────────────────────────────────────────
print("\nFloat baseline...")
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()
float_ppl = eval_ppl(model_float, eval_ids)
print(f"Float PPL: {float_ppl:.3f}")
del model_float
if DEVICE == "mps":
    torch.mps.empty_cache()

# ── Per-seed loop ─────────────────────────────────────────────────────────────
seed_results = []
n_trainable_total = None
for s in SEEDS:
    r, n_train = run_seed(s)
    seed_results.append(r)
    n_trainable_total = n_train


def agg(key):
    vals = torch.tensor([r[key] for r in seed_results], dtype=torch.float64)
    return float(vals.mean()), float(vals.std(unbiased=False))


init_mean, init_std = agg("init_ppl")
best_mean, best_std = agg("best_ppl")
best_step_mean, best_step_std = agg("best_step")
final_mean, final_std = agg("final_ppl_at_4000")

print(f"\n--- Aggregate ---")
print(f"Init   : {init_mean:.3f} ± {init_std:.3f}")
print(f"Best   : {best_mean:.3f} ± {best_std:.3f}")
print(f"BestStp: {best_step_mean:.0f} ± {best_step_std:.0f}")
print(f"Final  : {final_mean:.3f} ± {final_std:.3f}")

# ── Verdict (item 268) ────────────────────────────────────────────────────────
delta_vs_230 = best_mean - CYCLE_230_BASELINE
if best_mean <= 58.5:
    verdict = "CONFIRMED"
elif best_mean >= 59.5:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}\nVERDICT: {verdict}\n{'='*60}")
print(f"best_mean       = {best_mean:.3f} ± {best_std:.3f}")
print(f"cycle 230 K=1   = {CYCLE_230_BASELINE} (W_out-only baseline)")
print(f"delta vs c230   = {delta_vs_230:+.3f}")
print(f"float PPL       = {float_ppl:.3f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 262,
    "item": 268,
    "verdict": verdict,
    "config": {
        "BLOCKS": BLOCKS, "K": K, "LR": LR,
        "N_STEPS": N_STEPS, "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
        "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN,
    },
    "trainable_params_total": n_trainable_total,
    "trainable_params_per_block": 768 + 768 + 1,
    "float_ppl": float(float_ppl),
    "cycle_230_baseline": CYCLE_230_BASELINE,
    "delta_vs_cycle_230": float(delta_vs_230),
    "aggregate": {
        "init_mean": init_mean, "init_std": init_std,
        "best_mean": best_mean, "best_std": best_std,
        "best_step_mean": best_step_mean, "best_step_std": best_step_std,
        "final_mean": final_mean, "final_std": final_std,
    },
    "seeds": seed_results,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON to {OUTPUT_PATH}")

# ── Summary markdown ──────────────────────────────────────────────────────────
def pct(p):
    return (p / float_ppl - 1) * 100


lines = []
lines.append(f"# Cycle 262 — Item 268: K=1 + c_fc bias trainable (3 blocks)")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Criteria: CONFIRMED if best_mean ≤ 58.5; REFUTED if ≥ 59.5; else INCONCLUSIVE.")
lines.append(f"- Trainable: 3 × (W_out 768 + b_out 768 + b_fc 1) = {n_trainable_total} params.")
lines.append(f"- Config: BLOCKS={BLOCKS}, K={K}, LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}.")
lines.append(f"- Float PPL: {float_ppl:.3f}")
lines.append("")
lines.append("## Aggregate")
lines.append("")
lines.append(f"| metric | mean | std |")
lines.append(f"|---|---|---|")
lines.append(f"| init | {init_mean:.3f} | {init_std:.3f} |")
lines.append(f"| **best** | **{best_mean:.3f}** | {best_std:.3f} |")
lines.append(f"| best_step | {best_step_mean:.0f} | {best_step_std:.0f} |")
lines.append(f"| final@{N_STEPS} | {final_mean:.3f} | {final_std:.3f} |")
lines.append("")
lines.append("## Per-seed")
lines.append("")
lines.append("| seed | init | best | best_step | final@4000 |")
lines.append("|---|---|---|---|---|")
for s in seed_results:
    lines.append(f"| {s['seed']} | {s['init_ppl']:.3f} | {s['best_ppl']:.3f} | "
                 f"{s['best_step']} | {s['final_ppl_at_4000']:.3f} |")
lines.append("")
lines.append("## Comparison to baselines")
lines.append("")
lines.append(f"- **Cycle 230 K=1 3-block W_out-only**: 59.245 (baseline).")
lines.append(f"- **This cycle (K=1 3-block, +b_fc)**: {best_mean:.3f}.")
lines.append(f"- **Delta vs cycle 230**: {delta_vs_230:+.3f} PPL.")
lines.append(f"- Cycle 213 K=512 3-block W_out+c_fc bias: noop.")
lines.append(f"- Cycle 241 K=512 6-block W_out+c_fc bias: noop (59.533 vs 59.515).")
lines.append(f"- Cycle 240 K=1 6-block: 96.886 (catastrophic).")
lines.append(f"- Float PPL: {float_ppl:.3f}.")
lines.append("")
lines.append("## b_fc trajectories")
lines.append("")
lines.append("| seed | b_fc[block 1] | b_fc[block 4] | b_fc[block 7] |")
lines.append("|---|---|---|---|")
for s in seed_results:
    bfc = s["bfc_final"]
    lines.append(f"| {s['seed']} | {bfc[1]:+.4f} | {bfc[4]:+.4f} | {bfc[7]:+.4f} |")
lines.append("")
with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
