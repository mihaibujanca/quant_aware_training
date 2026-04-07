"""
Cycle 266 — Item 276: 2-block K=512 sweep — how far below float does the count-driven cascade go?

SUBSETS = [[1,4], [1,7], [4,7]]; K=512. Trivial SUBSETS swap on cycle 255 template.

3 seeds {42, 137, 2024} per subset = 9 fits total.
Fall back to 2 seeds {42, 137} if wall time exceeds 30 min after the first 6 fits.

Verdict per item 276 (operates on min over 3 subsets of best_mean across seeds):
  CONFIRMED   if min over subsets best_mean <= 45.0
  REFUTED     if all subsets best_mean >= 47.0
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

SUBSETS = [[1, 4], [1, 7], [4, 7]]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 4000
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
WALL_TIME_CAP_SEC = 30 * 60  # after first 6 fits, drop seed 2024 if elapsed > 30 min
OUTPUT_PATH = "autoresearch/results/cycle_266_two_block_sweep.json"
SUMMARY_PATH = "autoresearch/results/cycle_266_summary.md"

# Baselines (PPL)
CYCLE_255_K512_4BLOCK_MIN = 49.609  # min over [1,2,4,7]/[1,4,7,8]/[1,2,7,8] at K=512
CYCLE_210_K512_3BLOCK = 46.556  # 3-block [1,4,7] K=512
CYCLE_245_K512_5BLOCK = 53.826  # 5-block [1,2,4,7,8] K=512
CYCLE_221_K512_6BLOCK = 59.515  # 6-block K=512
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


# ── Helpers (verbatim from cycle 255 / 245) ───────────────────────────────────
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

    final_ppl_at_step_4000 = eval_ppl(model, eval_ids)
    print(f"Final at step {N_STEPS}: {final_ppl_at_step_4000:.3f}")

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
        "final_ppl_at_step_4000": float(final_ppl_at_step_4000),
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


# ── Sweep: 3 subsets x 3 seeds = 9 fits; fall back to 2 seeds if wall cap ─────
print(f"\n{'#'*60}\nSWEEP: {len(SUBSETS)} subsets x {len(SEEDS)} seeds = {len(SUBSETS)*len(SEEDS)} fits\n{'#'*60}")
wall_start = time.time()
subset_results = {}
fits_completed = 0
active_seeds = list(SEEDS)

for BLOCKS in SUBSETS:
    key = "_".join(str(b) for b in BLOCKS)
    subset_results[key] = []
    for s in list(active_seeds):
        # After first 6 fits, if wall time exceeds cap, drop seed 2024 from remaining.
        elapsed = time.time() - wall_start
        if fits_completed >= 6 and elapsed > WALL_TIME_CAP_SEC and 2024 in active_seeds:
            print(f"\n[WALL TIME CAP] elapsed {elapsed/60:.1f} min > 30 min after {fits_completed} fits -- dropping seed 2024 for remaining subsets")
            active_seeds = [x for x in active_seeds if x != 2024]
            if s == 2024:
                continue
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
    final_mean, final_std = agg_key(results, "final_ppl_at_step_4000")
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

# ── Verdict per item 276 ──────────────────────────────────────────────────────
if min_best_mean <= 45.0:
    verdict = "CONFIRMED"
elif all(bm >= 47.0 for bm in best_means):
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
print(f"Float PPL:              {float_ppl:.3f}")
for k in aggregates:
    a = aggregates[k]
    print(f"  {a['subset']} (n={a['n_seeds']}): best_mean={a['best_mean']:.3f} ± {a['best_std']:.3f}  init {a['init_mean']:.3f}  final@4000 {a['final_mean']:.3f}")
print(f"min best_mean: {min_best_mean:.3f}  ({min_subset_key})")
print(f"max best_mean: {max_best_mean:.3f}")
print(f"Cycle 210 3-block K=512 baseline: {CYCLE_210_K512_3BLOCK}")
print(f"Cycle 255 4-block K=512 min:      {CYCLE_255_K512_4BLOCK_MIN}")
print(f"Float ref:                        {FLOAT_BASELINE_REF}")
print(f"Delta (min - cycle 210 3b):       {min_best_mean - CYCLE_210_K512_3BLOCK:+.3f}")
print(f"Delta (min - cycle 255 4b):       {min_best_mean - CYCLE_255_K512_4BLOCK_MIN:+.3f}")
print(f"Delta (min - float):              {min_best_mean - float(float_ppl):+.3f}")
print(f"Verdict: {verdict}")

result = {
    "cycle": 266,
    "item": 276,
    "verdict": verdict,
    "config": {
        "SUBSETS": SUBSETS, "K": K, "N_CALIB": N_CALIB, "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN, "LR": LR, "N_STEPS": N_STEPS,
        "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
        "WALL_TIME_CAP_SEC": WALL_TIME_CAP_SEC,
    },
    "float_ppl": float(float_ppl),
    "subset_results": subset_results,
    "aggregates": aggregates,
    "min_best_mean": min_best_mean,
    "max_best_mean": max_best_mean,
    "min_subset_key": min_subset_key,
    "cycle_210_k512_3block_baseline": CYCLE_210_K512_3BLOCK,
    "cycle_255_k512_4block_min_baseline": CYCLE_255_K512_4BLOCK_MIN,
    "cycle_245_k512_5block_baseline": CYCLE_245_K512_5BLOCK,
    "cycle_221_k512_6block_baseline": CYCLE_221_K512_6BLOCK,
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "delta_vs_cycle_210": min_best_mean - CYCLE_210_K512_3BLOCK,
    "delta_vs_cycle_255": min_best_mean - CYCLE_255_K512_4BLOCK_MIN,
    "delta_vs_float": min_best_mean - float(float_ppl),
    "wall_time_sec": time.time() - wall_start,
    "fits_completed": fits_completed,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON to {OUTPUT_PATH}")

# ── Summary markdown ──────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 266 — Item 276: 2-block K=512 sweep")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Criteria: CONFIRMED if min best_mean ≤ 45.0; REFUTED if all best_mean ≥ 47.0; else INCONCLUSIVE")
lines.append(f"- Subsets: {SUBSETS}")
lines.append(f"- Config: K={K}, LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}")
lines.append(f"- Float PPL: {float_ppl:.3f}")
lines.append(f"- Wall time: {(time.time()-wall_start)/60:.1f} min; fits={fits_completed}")
lines.append("")
lines.append("## Per-subset aggregates")
lines.append("")
lines.append("| subset | n_seeds | init_mean | best_mean ± std | best_step | final@4000 |")
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
    lines.append("| seed | init | best | best_step | final@4000 |")
    lines.append("|---|---|---|---|---|")
    for r in subset_results[k]:
        lines.append(f"| {r['seed']} | {r['init_ppl']:.3f} | {r['best_ppl']:.3f} | {r['best_step']} | {r['final_ppl_at_step_4000']:.3f} |")
    lines.append("")
lines.append("## Comparison")
lines.append("")
lines.append(f"- Cycle 210 K=512 3-block [1,4,7]: **{CYCLE_210_K512_3BLOCK:.3f}**")
lines.append(f"- Cycle 255 K=512 4-block min:    {CYCLE_255_K512_4BLOCK_MIN:.3f}")
lines.append(f"- Cycle 245 K=512 5-block:        {CYCLE_245_K512_5BLOCK:.3f}")
lines.append(f"- Cycle 221 K=512 6-block:        {CYCLE_221_K512_6BLOCK:.3f}")
lines.append(f"- Float ref:                       {FLOAT_BASELINE_REF:.3f}")
lines.append(f"- Min 2-block best_mean:          **{min_best_mean:.3f}**")
lines.append(f"- Delta vs cycle 210 (3b):        **{min_best_mean - CYCLE_210_K512_3BLOCK:+.3f}** PPL")
lines.append(f"- Delta vs cycle 255 (4b min):    **{min_best_mean - CYCLE_255_K512_4BLOCK_MIN:+.3f}** PPL")
lines.append(f"- Delta vs float:                 **{min_best_mean - float(float_ppl):+.3f}** PPL")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
