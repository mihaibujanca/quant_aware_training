"""
Cycle 225 — Item 235: K=64/32/16 Adam compression cliff sweep.
BLOCKS=[1,4,7], K_VALUES=[64, 32, 16] (K=64 first to resolve cliff verdict early).
Adam W_out + b_out, 4000 steps, eval every 100, best-state restore, 3 seeds.

Verdict (item 235):
  CONFIRMED   if min(best_mean across K) ≤ 53.0  (any K still beats float)
  REFUTED     if all best_mean ≥ 60.0           (real channel-capacity floor)
  INCONCLUSIVE otherwise.

Reference curve (prior cycles, hard-coded):
  K=128 best_mean = 48.989  (cycle 223)
  K=256 best_mean = 47.509  (cycle 223)
  K=512 best_mean = 46.556  (cycle 210)
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
K_VALUES = [64, 32, 16]
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 4000
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
OUTPUT_PATH = "autoresearch/results/cycle_225_lower_k_cliff.json"
SUMMARY_PATH = "autoresearch/results/cycle_225_summary.md"

# Reference curve from prior cycles (K=128/256 from cycle 223, K=512 from cycle 210)
REFERENCE_CURVE = {
    128: {"best_mean": 48.989, "source": "cycle 223"},
    256: {"best_mean": 47.509, "source": "cycle 223"},
    512: {"best_mean": 46.556, "source": "cycle 210"},
}

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


# ── Helpers (verbatim from cycle 223) ─────────────────────────────────────────
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


# ── Per-seed run ──────────────────────────────────────────────────────────────
def run_seed(K, seed):
    print(f"\n{'='*60}\nK={K}  SEED {seed}\n{'='*60}")
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
        model.transformer.h[b].mlp.W_out.requires_grad_(True)
        model.transformer.h[b].mlp.b_out.requires_grad_(True)
        trainable_params.extend([model.transformer.h[b].mlp.W_out,
                                  model.transformer.h[b].mlp.b_out])
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_trainable:,} ({n_trainable/1e6:.2f}M)")

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

    final_ppl_at_4000 = eval_ppl(model, eval_ids)
    print(f"Final at step {N_STEPS}: {final_ppl_at_4000:.3f}")

    for b in BLOCKS:
        W_best, b_best = best_state[b]
        model.transformer.h[b].mlp.W_out.data.copy_(W_best.to(DEVICE))
        model.transformer.h[b].mlp.b_out.data.copy_(b_best.to(DEVICE))
    final_ppl_restored = eval_ppl(model, eval_ids)
    print(f"Restored best (step {best_step}): {final_ppl_restored:.3f} (recorded best={best_ppl:.3f})")
    assert abs(final_ppl_restored - best_ppl) < 1e-3, \
        f"Restore mismatch: {final_ppl_restored} vs {best_ppl}"

    result = {
        "seed": seed,
        "init_ppl": float(init_ppl),
        "best_ppl": float(best_ppl),
        "best_step": int(best_step),
        "final_ppl_at_4000": float(final_ppl_at_4000),
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

# ── Outer K loop ──────────────────────────────────────────────────────────────
all_K_results = []
for K in K_VALUES:
    print(f"\n\n{'#'*60}\n# K = {K}\n{'#'*60}")
    seed_results = []
    n_trainable_total = None
    for s in SEEDS:
        r, n_train = run_seed(K, s)
        seed_results.append(r)
        n_trainable_total = n_train

    def agg(key):
        vals = torch.tensor([r[key] for r in seed_results], dtype=torch.float64)
        return float(vals.mean()), float(vals.std(unbiased=False))

    init_mean, init_std = agg("init_ppl")
    best_mean, best_std = agg("best_ppl")
    best_step_mean, best_step_std = agg("best_step")
    final_mean, final_std = agg("final_ppl_at_4000")

    print(f"\n--- K={K} aggregate ---")
    print(f"Init   : {init_mean:.3f} ± {init_std:.3f}")
    print(f"Best   : {best_mean:.3f} ± {best_std:.3f}")
    print(f"BestStp: {best_step_mean:.0f} ± {best_step_std:.0f}")
    print(f"Final  : {final_mean:.3f} ± {final_std:.3f}")

    per_block = K * 768 + 768  # W_out (768,K) + b_out (768,)
    all_K_results.append({
        "K": K,
        "trainable_params_per_block": per_block,
        "total_trainable": n_trainable_total,
        "seeds": seed_results,
        "aggregate": {
            "init_mean": init_mean, "init_std": init_std,
            "best_mean": best_mean, "best_std": best_std,
            "best_step_mean": best_step_mean, "best_step_std": best_step_std,
            "final_mean": final_mean, "final_std": final_std,
        },
    })

# ── Verdict (item 235) ────────────────────────────────────────────────────────
best_means = {r["K"]: r["aggregate"]["best_mean"] for r in all_K_results}
min_best = min(best_means.values())
max_best = max(best_means.values())

if min_best <= 53.0:
    verdict = "CONFIRMED"
elif min(best_means.values()) >= 60.0:
    # all K best_mean >= 60.0
    verdict = "REFUTED" if all(bm >= 60.0 for bm in best_means.values()) else "INCONCLUSIVE"
else:
    verdict = "INCONCLUSIVE"

# Stricter: REFUTED only if ALL three K give best_mean >= 60.0
if all(bm >= 60.0 for bm in best_means.values()):
    verdict = "REFUTED"
elif min_best <= 53.0:
    verdict = "CONFIRMED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}\nVERDICT: {verdict}\n{'='*60}")
for K in K_VALUES:
    print(f"K={K:3d} best_mean = {best_means[K]:.3f}")
print(f"Float baseline   = {float_ppl:.3f}")
print(f"K=128 reference  = {REFERENCE_CURVE[128]['best_mean']} (cycle 223)")
print(f"K=256 reference  = {REFERENCE_CURVE[256]['best_mean']} (cycle 223)")
print(f"K=512 reference  = {REFERENCE_CURVE[512]['best_mean']} (cycle 210)")

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 225,
    "item": 235,
    "verdict": verdict,
    "config": {
        "BLOCKS": BLOCKS, "K_VALUES": K_VALUES, "LR": LR,
        "N_STEPS": N_STEPS, "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
        "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN,
    },
    "float_ppl": float(float_ppl),
    "reference_curve": REFERENCE_CURVE,
    "results": all_K_results,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON to {OUTPUT_PATH}")

# ── Summary markdown ──────────────────────────────────────────────────────────
def pct(p):
    return (p / float_ppl - 1) * 100

# Reference per-K trainable counts (3 blocks)
def trainable_for(K):
    return 3 * (K * 768 + 768)

curve_rows = []
for r in all_K_results:
    K = r["K"]
    bm = r["aggregate"]["best_mean"]
    bs = r["aggregate"]["best_std"]
    curve_rows.append((K, r["total_trainable"], bm, bs, pct(bm), "this cycle"))
for K, ref in REFERENCE_CURVE.items():
    curve_rows.append((K, trainable_for(K), ref["best_mean"], None, pct(ref["best_mean"]), ref["source"]))
curve_rows.sort(key=lambda x: -x[0])  # high K → low K

lines = []
lines.append(f"# Cycle 225 — Item 235: K=64/32/16 Adam compression cliff sweep")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append("- Criteria: CONFIRMED if any best_mean ≤ 53.0 (any K still beats float); "
             "REFUTED if all best_mean ≥ 60.0 (real channel-capacity floor); else INCONCLUSIVE.")
lines.append(f"- Config: BLOCKS={BLOCKS}, K_VALUES={K_VALUES}, LR={LR}, N_STEPS={N_STEPS}, "
             f"EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}")
lines.append(f"- Float PPL: {float_ppl:.3f}")
lines.append("")
lines.append("## Extended compression efficiency curve")
lines.append("")
lines.append("| K | trainable (3 blocks) | best_ppl_mean | std | %vs float | source |")
lines.append("|---|---|---|---|---|---|")
for K, n, bm, bs, p, src in curve_rows:
    std_str = f"{bs:.3f}" if bs is not None else "—"
    lines.append(f"| {K} | {n:,} | {bm:.3f} | {std_str} | {p:+.2f}% | {src} |")
lines.append("")
lines.append("## Per-K aggregates (this cycle)")
lines.append("")
for r in all_K_results:
    K = r["K"]
    a = r["aggregate"]
    lines.append(f"### K = {K}")
    lines.append("")
    lines.append("| seed | init | best | best_step | final@4000 |")
    lines.append("|---|---|---|---|---|")
    for s in r["seeds"]:
        lines.append(f"| {s['seed']} | {s['init_ppl']:.3f} | {s['best_ppl']:.3f} | "
                     f"{s['best_step']} | {s['final_ppl_at_4000']:.3f} |")
    lines.append(f"| **mean** | {a['init_mean']:.3f} | **{a['best_mean']:.3f}** | "
                 f"{a['best_step_mean']:.0f} | {a['final_mean']:.3f} |")
    lines.append(f"| **std**  | {a['init_std']:.3f} | {a['best_std']:.3f} | "
                 f"{a['best_step_std']:.0f} | {a['final_std']:.3f} |")
    lines.append("")

lines.append("## Comparison to cycle 223")
lines.append("")
k512_ref = REFERENCE_CURVE[512]["best_mean"]
k256_ref = REFERENCE_CURVE[256]["best_mean"]
k128_ref = REFERENCE_CURVE[128]["best_mean"]
lines.append(f"- K=512 (cycle 210): **{k512_ref:.3f}**")
lines.append(f"- K=256 (cycle 223): **{k256_ref:.3f}** (Δ vs K=512: {k256_ref-k512_ref:+.3f})")
lines.append(f"- K=128 (cycle 223): **{k128_ref:.3f}** (Δ vs K=512: {k128_ref-k512_ref:+.3f})")
for r in all_K_results:
    K = r["K"]
    bm = r["aggregate"]["best_mean"]
    lines.append(f"- K={K} (this cycle): **{bm:.3f}** "
                 f"(Δ vs K=512: {bm-k512_ref:+.3f}, Δ vs K=128: {bm-k128_ref:+.3f})")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
