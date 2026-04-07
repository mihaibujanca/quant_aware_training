"""
Cycle 287 — Item 297: 1-block [1] K=512 extended budget — 8000 steps.

Direct extension of cycle 271 [1] K=512: same script, SUBSETS=[[1]],
N_STEPS=8000, EVAL_EVERY=200 (still 40 eval points). Tests whether the
2x budget extension pushes [1] K=512 below 43.0 (CONFIRMED) or whether the
4k cap was already at the floor (REFUTED).

3 seeds {42, 137, 2024}; joint Adam over W_out+b_out; best-state restore.

Verdict per item 297:
  CONFIRMED   if best_mean <= 43.0
  REFUTED     if best_mean >= 43.7
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

SUBSETS = [[1]]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 8000
EVAL_EVERY = 200
SEEDS = [42, 137, 2024]
OUTPUT_PATH = "autoresearch/results/cycle_287_one_block_1_k512_8k.json"
SUMMARY_PATH = "autoresearch/results/cycle_287_summary.md"

# Baselines (PPL)
CYCLE_271_B1_K512_4K = 43.82045486551152          # cycle 271 [1] K=512 4k aggregate (this script's 4k counterpart)
CYCLE_286_B2_K512_4K = 43.56269                   # cycle 286 [2] K=512 4k — current PPL Pareto winner at 393,984 params
FLOAT_BASELINE_REF = 53.895

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


# ── Helpers (verbatim from cycle 271) ─────────────────────────────────────────
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


# ── Per-(subset, seed) run ────────────────────────────────────────────────────
def run_seed(BLOCKS, seed):
    print(f"\n{'='*60}\nSUBSET {BLOCKS}  SEED {seed}\n{'='*60}")
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
    assert n_trainable == 393_984, f"trainable count mismatch: {n_trainable}"

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

    result = {
        "subset": list(BLOCKS),
        "seed": seed,
        "init_ppl": float(init_ppl),
        "best_ppl": float(best_ppl),
        "best_step": int(best_step),
        "final_ppl_at_step_8000": float(final_ppl_at_step_8000),
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


# ── Sweep: 1 subset x 3 seeds = 3 fits ────────────────────────────────────────
print(f"\n{'#'*60}\nSWEEP: {len(SUBSETS)} subsets x {len(SEEDS)} seeds = {len(SUBSETS)*len(SEEDS)} fits\n{'#'*60}")
wall_start = time.time()
subset_results = {}
fits_completed = 0

for BLOCKS in SUBSETS:
    key = "_".join(str(b) for b in BLOCKS)
    subset_results[key] = []
    for s in SEEDS:
        r = run_seed(BLOCKS, s)
        subset_results[key].append(r)
        fits_completed += 1
        print(f"[fit {fits_completed} done, elapsed {(time.time()-wall_start)/60:.1f} min]")


# ── Aggregate per-subset ──────────────────────────────────────────────────────
def agg_key(results, key):
    vals = torch.tensor([r[key] for r in results], dtype=torch.float64)
    return float(vals.mean()), float(vals.std(unbiased=False))


aggregates = {}
for BLOCKS in SUBSETS:
    key = "_".join(str(b) for b in BLOCKS)
    results = subset_results[key]
    if not results:
        continue
    init_mean, init_std = agg_key(results, "init_ppl")
    best_mean, best_std = agg_key(results, "best_ppl")
    best_step_mean, best_step_std = agg_key(results, "best_step")
    final_mean, final_std = agg_key(results, "final_ppl_at_step_8000")
    aggregates[key] = {
        "subset": list(BLOCKS),
        "n_seeds": len(results),
        "seeds": [r["seed"] for r in results],
        "init_mean": init_mean, "init_std": init_std,
        "best_mean": best_mean, "best_std": best_std,
        "best_step_mean": best_step_mean, "best_step_std": best_step_std,
        "final_mean": final_mean, "final_std": final_std,
    }

best_means = [aggregates[k]["best_mean"] for k in aggregates]
min_best_mean = min(best_means)
max_best_mean = max(best_means)
min_subset_key = [k for k in aggregates if aggregates[k]["best_mean"] == min_best_mean][0]

# ── Verdict per item 297 ──────────────────────────────────────────────────────
if min_best_mean <= 43.0:
    verdict = "CONFIRMED"
elif min_best_mean >= 43.7:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
print(f"Float PPL:              {float_ppl:.3f}")
for k in aggregates:
    a = aggregates[k]
    print(f"  {a['subset']} (n={a['n_seeds']}): best_mean={a['best_mean']:.3f} ± {a['best_std']:.3f}  init {a['init_mean']:.3f}  final@8000 {a['final_mean']:.3f}")
print(f"min best_mean: {min_best_mean:.3f}  ({min_subset_key})")
print(f"Cycle 271 [1] K=512 4k baseline: {CYCLE_271_B1_K512_4K}")
print(f"Cycle 286 [2] K=512 4k baseline: {CYCLE_286_B2_K512_4K}")
print(f"Float ref:                       {FLOAT_BASELINE_REF}")
print(f"Delta (best - cycle 271 [1] 4k): {min_best_mean - CYCLE_271_B1_K512_4K:+.3f}")
print(f"Delta (best - cycle 286 [2] 4k): {min_best_mean - CYCLE_286_B2_K512_4K:+.3f}")
print(f"Delta (best - float):            {min_best_mean - float(float_ppl):+.3f}")
print(f"Verdict: {verdict}")

result = {
    "cycle": 287,
    "item": 297,
    "verdict": verdict,
    "config": {
        "SUBSETS": SUBSETS, "K": K, "N_CALIB": N_CALIB, "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN, "LR": LR, "N_STEPS": N_STEPS,
        "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
    },
    "float_ppl": float(float_ppl),
    "subset_results": subset_results,
    "aggregates": aggregates,
    "min_best_mean": min_best_mean,
    "max_best_mean": max_best_mean,
    "min_subset_key": min_subset_key,
    "cycle_271_b1_k512_4k_baseline": CYCLE_271_B1_K512_4K,
    "cycle_286_b2_k512_4k_baseline": CYCLE_286_B2_K512_4K,
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "delta_vs_cycle_271_b1_k512_4k": min_best_mean - CYCLE_271_B1_K512_4K,
    "delta_vs_cycle_286_b2_k512_4k": min_best_mean - CYCLE_286_B2_K512_4K,
    "delta_vs_float": min_best_mean - float(float_ppl),
    "wall_time_sec": time.time() - wall_start,
    "fits_completed": fits_completed,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON to {OUTPUT_PATH}")

# ── Summary markdown ──────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 287 — Item 297: 1-block [1] K=512 extended budget (8000 steps)")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Criteria: CONFIRMED if best_mean ≤ 43.0; REFUTED if best_mean ≥ 43.7; else INCONCLUSIVE")
lines.append(f"- Subsets: {SUBSETS}")
lines.append(f"- Config: K={K}, LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}")
lines.append(f"- Float PPL: {float_ppl:.3f}")
lines.append(f"- Wall time: {(time.time()-wall_start)/60:.1f} min; fits={fits_completed}")
lines.append("")
lines.append("## Per-subset aggregates")
lines.append("")
lines.append("| subset | n_seeds | init_mean | best_mean ± std | best_step | final@8000 |")
lines.append("|---|---|---|---|---|---|")
for k in aggregates:
    a = aggregates[k]
    lines.append(f"| {a['subset']} | {a['n_seeds']} | {a['init_mean']:.3f} | **{a['best_mean']:.3f} ± {a['best_std']:.3f}** | {a['best_step_mean']:.0f} ± {a['best_step_std']:.0f} | {a['final_mean']:.3f} |")
lines.append("")
lines.append(f"**Min over subsets: {min_best_mean:.3f} ({min_subset_key})**")
lines.append("")
lines.append("## Per-seed details")
lines.append("")
for k in subset_results:
    if not subset_results[k]:
        continue
    a = aggregates[k]
    lines.append(f"### Subset {a['subset']}")
    lines.append("")
    lines.append("| seed | init | best | best_step | final@8000 |")
    lines.append("|---|---|---|---|---|")
    for r in subset_results[k]:
        lines.append(f"| {r['seed']} | {r['init_ppl']:.3f} | {r['best_ppl']:.3f} | {r['best_step']} | {r['final_ppl_at_step_8000']:.3f} |")
    lines.append("")
lines.append("## Comparison")
lines.append("")
lines.append(f"- Cycle 271 [1] K=512 4k:  **{CYCLE_271_B1_K512_4K:.3f}**  (393,984 params)")
lines.append(f"- Cycle 286 [2] K=512 4k:  **{CYCLE_286_B2_K512_4K:.3f}**  (393,984 params, current PPL Pareto winner)")
lines.append(f"- Float ref:               {FLOAT_BASELINE_REF:.3f}")
lines.append(f"- Min 1-block best_mean:   **{min_best_mean:.3f}**")
lines.append(f"- Delta vs cycle 271 [1] 4k: **{min_best_mean - CYCLE_271_B1_K512_4K:+.3f}** PPL")
lines.append(f"- Delta vs cycle 286 [2] 4k: **{min_best_mean - CYCLE_286_B2_K512_4K:+.3f}** PPL")
lines.append(f"- Delta vs float:            **{min_best_mean - float(float_ppl):+.3f}** PPL")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
