"""
Cycle 294 — Item 304: 1-block [1] K sweep extended budget — N_STEPS=8000.

Trivial budget extension of cycle 276 (`experiments/one_block_k_sweep_cycle276.py`).
Only changes: N_STEPS=4000->8000, EVAL_EVERY=100->200, output paths, baseline
constants, verdict thresholds (item 304 verbatim), wall-time cap raised to 60 min.

SUBSETS=[[1]]; K_VALUES=[32, 64, 128, 256]. 4 K x 3 seeds {42, 137, 2024} = 12 fits.
Joint Adam W_out + b_out, 8000 steps, eval every 200, best-state restore.

Verdict per item 304 verbatim:
  CONFIRMED   iff at least 2 of 4 K-tiers shift by >= 0.5 PPL (4k - 8k >= 0.5)
  REFUTED     iff all 4 K-tiers shift by <= 0.1 PPL
  INCONCLUSIVE otherwise
"""
import json
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

BLOCKS = [1]
K_VALUES = [32, 64, 128, 256]
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 8000
EVAL_EVERY = 200
SEEDS = [42, 137, 2024]
WALL_TIME_CAP_SEC = 60 * 60  # 60 min cap (~28 min expected)
OUTPUT_PATH = "autoresearch/results/cycle_294_one_block_1_k_sweep_8k.json"
SUMMARY_PATH = "autoresearch/results/cycle_294_summary.md"

# Baselines (PPL)
FLOAT_BASELINE_REF = 53.895
CYCLE_287_K512_1BLOCK_8K = 42.78532763524365  # cycle 287 [1] K=512 8k best_mean
CYCLE_271_K512_1BLOCK_4K = 43.82045486551152  # cycle 271 [1] K=512 4k best_mean

# Cycle 276 4k baselines (best_mean per K, [1] K sweep)
CYCLE_276_BEST_MEAN_4K = {
    32: 49.6485436072763,
    64: 48.04525097092613,
    128: 46.931609124860834,
    256: 45.385373508451075,
}
# For per-seed step-4000 cross-check (must match within 1e-4):
CYCLE_276_BEST_PPL_PER_SEED_4K = {
    32: {42: 49.62303929764403, 137: 49.683719158673725, 2024: 49.63936213050516},
    64: {42: 48.10018155392079, 137: 48.011289953000926, 2024: 48.02427140585666},
    128: {42: 47.0208232233785, 137: 46.86583330272509, 2024: 46.90796684847891},
    256: {42: 45.32514579465824, 137: 45.42779559681012, 2024: 45.40317912388486},
}

# Trainable per K (block 1 only)
EXPECTED_TRAINABLE = {K: K * 768 + 768 for K in K_VALUES}

print(f"Device: {DEVICE}")

# Load tokenizer + data once
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


# ── Helpers (verbatim from cycle 276) ─────────────────────────────────────────
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


# ── Per-(K, seed) run ─────────────────────────────────────────────────────────
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
    expected = EXPECTED_TRAINABLE[K]
    assert n_trainable == expected, f"trainable mismatch K={K}: {n_trainable} vs expected {expected}"
    print(f"Trainable params: {n_trainable:,} (expect {expected:,} = {K}×768+768)")

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
    eval_at_4000 = None  # cross-check vs cycle 276

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
                if step == 4000:
                    eval_at_4000 = float(ppl_e)
                print(f"  Step {step:5d}: eval_ppl={ppl_e:.3f}{marker}", flush=True)

    final_ppl_at_step_8000 = eval_ppl(model, eval_ids)
    print(f"Final at step {N_STEPS}: {final_ppl_at_step_8000:.3f}")

    for b in BLOCKS:
        W_best, b_best = best_state[b]
        model.transformer.h[b].mlp.W_out.data.copy_(W_best.to(DEVICE))
        model.transformer.h[b].mlp.b_out.data.copy_(b_best.to(DEVICE))
    final_ppl_restored = eval_ppl(model, eval_ids)
    print(f"Restored best (step {best_step}): {final_ppl_restored:.3f} (recorded best={best_ppl:.3f})")
    assert abs(final_ppl_restored - best_ppl) < 1e-3, \
        f"Restore mismatch: {final_ppl_restored} vs {best_ppl}"

    # Cross-check vs cycle 276 step-4000 best per (K, seed)
    cyc276_ref = CYCLE_276_BEST_PPL_PER_SEED_4K[K][seed]
    delta_4000 = (eval_at_4000 - cyc276_ref) if eval_at_4000 is not None else None
    if delta_4000 is not None:
        print(f"Step-4000 cross-check: cyc294={eval_at_4000:.6f}  cyc276={cyc276_ref:.6f}  |Δ|={abs(delta_4000):.2e}")

    result = {
        "K": K,
        "seed": seed,
        "init_ppl": float(init_ppl),
        "best_ppl": float(best_ppl),
        "best_step": int(best_step),
        "final_ppl_at_step_8000": float(final_ppl_at_step_8000),
        "eval_at_step_4000": float(eval_at_4000) if eval_at_4000 is not None else None,
        "cycle_276_step_4000_ref": float(cyc276_ref),
        "step_4000_delta": float(delta_4000) if delta_4000 is not None else None,
        "trainable_params": int(n_trainable),
        "history": history,
    }
    del model, optimizer, trainable_params, best_state
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return result


# ── Float baseline (once) ─────────────────────────────────────────────────────
print("\nFloat baseline...")
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()
float_ppl = eval_ppl(model_float, eval_ids)
print(f"Float PPL: {float_ppl:.3f}")
del model_float
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Sweep: 4 K x 3 seeds = 12 fits ────────────────────────────────────────────
print(f"\n{'#'*60}\nSWEEP: {len(K_VALUES)} K x {len(SEEDS)} seeds = {len(K_VALUES)*len(SEEDS)} fits\n{'#'*60}")
wall_start = time.time()
k_results = {}
fits_completed = 0

for K in K_VALUES:
    key = str(K)
    k_results[key] = []
    for s in SEEDS:
        r = run_seed(K, s)
        k_results[key].append(r)
        fits_completed += 1
        elapsed_min = (time.time() - wall_start) / 60
        print(f"[fit {fits_completed} done, elapsed {elapsed_min:.1f} min]")
        if (time.time() - wall_start) > WALL_TIME_CAP_SEC:
            print(f"[WALL TIME CAP] elapsed {elapsed_min:.1f} min > 60 min — stopping after fit {fits_completed}")
            break
    if (time.time() - wall_start) > WALL_TIME_CAP_SEC:
        break


# ── Aggregate per-K ───────────────────────────────────────────────────────────
def agg_key(results, key):
    vals = torch.tensor([r[key] for r in results], dtype=torch.float64)
    return float(vals.mean()), float(vals.std(unbiased=False))


aggregates = {}
for K in K_VALUES:
    key = str(K)
    results = k_results.get(key, [])
    if not results:
        continue
    init_mean, init_std = agg_key(results, "init_ppl")
    best_mean, best_std = agg_key(results, "best_ppl")
    best_step_mean, best_step_std = agg_key(results, "best_step")
    final_mean, final_std = agg_key(results, "final_ppl_at_step_8000")
    eval_4000_mean, eval_4000_std = agg_key(results, "eval_at_step_4000")
    bests = [r["best_ppl"] for r in results]
    aggregates[key] = {
        "K": K,
        "n_seeds": len(results),
        "seeds": [r["seed"] for r in results],
        "trainable_params": EXPECTED_TRAINABLE[K],
        "init_mean": init_mean, "init_std": init_std,
        "best_mean": best_mean, "best_std": best_std,
        "best_min": float(min(bests)), "best_max": float(max(bests)),
        "best_step_mean": best_step_mean, "best_step_std": best_step_std,
        "final_mean": final_mean, "final_std": final_std,
        "eval_at_4000_mean": eval_4000_mean, "eval_at_4000_std": eval_4000_std,
        "cycle_276_4k_best_mean": CYCLE_276_BEST_MEAN_4K[K],
        "shift_4k_to_8k": CYCLE_276_BEST_MEAN_4K[K] - best_mean,
    }

# ── Verdict per item 304 verbatim ─────────────────────────────────────────────
shifts = {K: aggregates[str(K)]["shift_4k_to_8k"] for K in K_VALUES if str(K) in aggregates}
n_shift_ge_05 = sum(1 for v in shifts.values() if v >= 0.5)
all_shift_le_01 = (len(shifts) == len(K_VALUES)) and all(v <= 0.1 for v in shifts.values())

confirmed_clause = n_shift_ge_05 >= 2
refuted_clause = all_shift_le_01

if confirmed_clause:
    verdict = "CONFIRMED"
elif refuted_clause:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
print(f"Float PPL: {float_ppl:.3f}")
for K in K_VALUES:
    if str(K) not in aggregates:
        continue
    a = aggregates[str(K)]
    print(f"  K={K:3d} (n={a['n_seeds']}, params={a['trainable_params']:,}): "
          f"4k={a['cycle_276_4k_best_mean']:.3f} -> 8k={a['best_mean']:.3f} ± {a['best_std']:.3f}  "
          f"shift={a['shift_4k_to_8k']:+.3f}  best_step={a['best_step_mean']:.0f}")
print(f"\nVerdict logic (item 304):")
print(f"  CONFIRMED iff >= 2 of 4 K-tiers shift by >= 0.5 PPL")
print(f"  REFUTED iff all 4 K-tiers shift by <= 0.1 PPL")
print(f"  n_shift_ge_05 = {n_shift_ge_05}/4")
print(f"  all_shift_le_01 = {all_shift_le_01}")
print(f"Verdict: {verdict}")

result = {
    "cycle": 294,
    "item": 304,
    "verdict": verdict,
    "verdict_logic": {
        "rule": "CONFIRMED iff >= 2 of 4 K-tiers shift by >= 0.5 PPL (cycle 276 4k best_mean - cycle 294 8k best_mean >= 0.5); REFUTED iff all 4 K-tiers shift by <= 0.1 PPL; else INCONCLUSIVE.",
        "shifts": shifts,
        "n_shift_ge_05": n_shift_ge_05,
        "all_shift_le_01": all_shift_le_01,
        "confirmed_clause_triggered": confirmed_clause,
        "refuted_clause_triggered": refuted_clause,
    },
    "config": {
        "BLOCKS": BLOCKS, "K_VALUES": K_VALUES, "N_CALIB": N_CALIB, "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN, "LR": LR, "N_STEPS": N_STEPS,
        "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
        "WALL_TIME_CAP_SEC": WALL_TIME_CAP_SEC,
    },
    "expected_trainable_per_K": EXPECTED_TRAINABLE,
    "float_ppl": float(float_ppl),
    "k_results": k_results,
    "aggregates": aggregates,
    "baselines": {
        "float_baseline_ref": FLOAT_BASELINE_REF,
        "cycle_287_k512_1block_8k": CYCLE_287_K512_1BLOCK_8K,
        "cycle_271_k512_1block_4k": CYCLE_271_K512_1BLOCK_4K,
        "cycle_276_4k_best_mean_per_K": CYCLE_276_BEST_MEAN_4K,
    },
    "wall_time_sec": time.time() - wall_start,
    "fits_completed": fits_completed,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON to {OUTPUT_PATH}")

# ── Summary markdown ──────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 294 — Item 304: 1-block [1] K sweep — 8000 steps")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Criteria: CONFIRMED if ≥2 of 4 K-tiers shift by ≥0.5 PPL; REFUTED if all ≤0.1 PPL; else INCONCLUSIVE")
lines.append(f"- BLOCKS: {BLOCKS}; K_VALUES: {K_VALUES}")
lines.append(f"- Config: LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}")
lines.append(f"- Float PPL: {float_ppl:.3f}")
lines.append(f"- Wall time: {(time.time()-wall_start)/60:.1f} min; fits={fits_completed}")
lines.append("")
lines.append("## Per-K aggregates (4k → 8k extension)")
lines.append("")
lines.append("| K | params | n_seeds | 4k best_mean (cyc 276) | 8k best_mean ± std | shift Δ | best_min/max | best_step | final@8000 |")
lines.append("|---|---|---|---|---|---|---|---|---|")
for K in K_VALUES:
    if str(K) not in aggregates:
        continue
    a = aggregates[str(K)]
    lines.append(
        f"| {a['K']} | {a['trainable_params']:,} | {a['n_seeds']} | {a['cycle_276_4k_best_mean']:.3f} | "
        f"**{a['best_mean']:.3f} ± {a['best_std']:.3f}** | **{a['shift_4k_to_8k']:+.3f}** | "
        f"{a['best_min']:.3f} / {a['best_max']:.3f} | "
        f"{a['best_step_mean']:.0f} ± {a['best_step_std']:.0f} | {a['final_mean']:.3f} |"
    )
lines.append("")
lines.append("## Per-seed details")
lines.append("")
for K in K_VALUES:
    key = str(K)
    if not k_results.get(key):
        continue
    lines.append(f"### K={K}")
    lines.append("")
    lines.append("| seed | init | best | best_step | final@8000 | step-4000 eval | cyc 276 ref | |Δ| |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in k_results[key]:
        d = abs(r["step_4000_delta"]) if r["step_4000_delta"] is not None else float("nan")
        lines.append(f"| {r['seed']} | {r['init_ppl']:.4f} | {r['best_ppl']:.4f} | {r['best_step']} | "
                     f"{r['final_ppl_at_step_8000']:.4f} | {r['eval_at_step_4000']:.6f} | "
                     f"{r['cycle_276_step_4000_ref']:.6f} | {d:.2e} |")
    lines.append("")
lines.append("## Comparison vs baselines")
lines.append("")
lines.append(f"- Float ref:                      {FLOAT_BASELINE_REF:.3f}")
lines.append(f"- Cycle 271 [1] K=512 4k:         {CYCLE_271_K512_1BLOCK_4K:.3f} (393,984 params)")
lines.append(f"- Cycle 287 [1] K=512 8k:         {CYCLE_287_K512_1BLOCK_8K:.3f} (393,984 params)")
lines.append("")
for K in K_VALUES:
    if str(K) not in aggregates:
        continue
    a = aggregates[str(K)]
    lines.append(f"- 1-block K={K} 8k best_mean: **{a['best_mean']:.3f}**, "
                 f"shift vs cyc 276 4k: **{a['shift_4k_to_8k']:+.3f}** PPL, "
                 f"params: {a['trainable_params']:,}")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
