"""
Cycle 286 — Item 296: 1-block [2] K=512 — does block 2 alone at maximum K
beat cycle 271 [1] K=512 (43.82) on the per-block axis?

Trivial SUBSETS swap on cycle 271 template (`experiments/one_block_sweep_cycle271.py`):
SUBSETS = [[2]], K=512. Joint Adam over W_out + b_out, 3 seeds, best-state restore.

Verdict per item 296 verbatim:
  CONFIRMED   iff best_mean <= 43.5
  REFUTED     iff best_mean >= 44.5
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

SUBSETS = [[2]]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 4000
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
OUTPUT_PATH = "autoresearch/results/cycle_286_block2_k512.json"
SUMMARY_PATH = "autoresearch/results/cycle_286_summary.md"

# Baselines (PPL)
CYCLE_271_K512_1BLOCK_B1 = 43.821       # [1] K=512 — current PPL Pareto winner (393,984 params)
CYCLE_271_K512_1BLOCK_B4 = 44.630       # [4] K=512
CYCLE_271_K512_1BLOCK_B7 = 45.370       # [7] K=512
CYCLE_283_B2_K64 = 47.5114804142954     # [2] K=64 (49,920 params, from cycle 283 JSON)
CYCLE_283_B2_K32 = 47.897               # [2] K=32 (25,344 params)
CYCLE_283_B2_K16 = 49.462               # [2] K=16 (13,056 params)
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
    expected_trainable = K * 768 + 768  # 393,984 for K=512
    assert n_trainable == expected_trainable, f"Trainable mismatch: {n_trainable} vs {expected_trainable}"

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
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, \
    f"Float drift: {float_ppl} vs ref {FLOAT_BASELINE_REF}"
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
    final_mean, final_std = agg_key(results, "final_ppl_at_step_4000")
    best_vals = [r["best_ppl"] for r in results]
    aggregates[key] = {
        "subset": list(BLOCKS),
        "n_seeds": len(results),
        "seeds": [r["seed"] for r in results],
        "trainable_params": results[0]["trainable_params"],
        "init_mean": init_mean, "init_std": init_std,
        "best_mean": best_mean, "best_std": best_std,
        "best_min": float(min(best_vals)),
        "best_max": float(max(best_vals)),
        "best_step_mean": best_step_mean, "best_step_std": best_step_std,
        "final_mean": final_mean, "final_std": final_std,
    }

# Single-subset min reduces to its own best_mean
key2 = "2"
best_mean = aggregates[key2]["best_mean"]

# ── Verdict per item 296 ──────────────────────────────────────────────────────
if best_mean <= 43.5:
    verdict = "CONFIRMED"
elif best_mean >= 44.5:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
print(f"Float PPL:              {float_ppl:.3f}")
for k in aggregates:
    a = aggregates[k]
    print(f"  {a['subset']} (n={a['n_seeds']}): best_mean={a['best_mean']:.3f} ± {a['best_std']:.3f}  init {a['init_mean']:.3f}  final@4000 {a['final_mean']:.3f}")
print(f"Cycle 271 [1] K=512:    {CYCLE_271_K512_1BLOCK_B1:.3f}")
print(f"Cycle 271 [4] K=512:    {CYCLE_271_K512_1BLOCK_B4:.3f}")
print(f"Cycle 271 [7] K=512:    {CYCLE_271_K512_1BLOCK_B7:.3f}")
print(f"Cycle 283 [2] K=64:     {CYCLE_283_B2_K64:.3f}")
print(f"Cycle 283 [2] K=32:     {CYCLE_283_B2_K32:.3f}")
print(f"Cycle 283 [2] K=16:     {CYCLE_283_B2_K16:.3f}")
print(f"Float ref:              {FLOAT_BASELINE_REF:.3f}")
print(f"Delta (best_mean - cycle 271 [1] K=512): {best_mean - CYCLE_271_K512_1BLOCK_B1:+.3f}")
print(f"Delta (best_mean - cycle 283 [2] K=64):  {best_mean - CYCLE_283_B2_K64:+.3f}")
print(f"Delta (best_mean - float):               {best_mean - float(float_ppl):+.3f}")
print(f"Verdict: {verdict}")

result = {
    "cycle": 286,
    "item": 296,
    "verdict": verdict,
    "verdict_logic": "CONFIRMED iff best_mean <= 43.5; REFUTED iff best_mean >= 44.5; else INCONCLUSIVE",
    "config": {
        "SUBSETS": SUBSETS, "K": K, "N_CALIB": N_CALIB, "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN, "LR": LR, "N_STEPS": N_STEPS,
        "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
    },
    "float_ppl": float(float_ppl),
    "subset_results": subset_results,
    "aggregates": aggregates,
    "best_mean": best_mean,
    "baselines": {
        "cycle_271_k512_1block_b1": CYCLE_271_K512_1BLOCK_B1,
        "cycle_271_k512_1block_b4": CYCLE_271_K512_1BLOCK_B4,
        "cycle_271_k512_1block_b7": CYCLE_271_K512_1BLOCK_B7,
        "cycle_283_b2_k64": CYCLE_283_B2_K64,
        "cycle_283_b2_k32": CYCLE_283_B2_K32,
        "cycle_283_b2_k16": CYCLE_283_B2_K16,
        "float_ref": FLOAT_BASELINE_REF,
    },
    "delta_vs_cycle_271_b1_k512": best_mean - CYCLE_271_K512_1BLOCK_B1,
    "delta_vs_cycle_283_b2_k64": best_mean - CYCLE_283_B2_K64,
    "delta_vs_float": best_mean - float(float_ppl),
    "wall_time_sec": time.time() - wall_start,
    "fits_completed": fits_completed,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON to {OUTPUT_PATH}")

# ── Summary markdown ──────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 286 — Item 296: 1-block [2] K=512")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Criteria: CONFIRMED iff best_mean ≤ 43.5; REFUTED iff best_mean ≥ 44.5; else INCONCLUSIVE")
lines.append(f"- Subset: [2], K=512")
lines.append(f"- Config: LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}")
lines.append(f"- Float PPL: {float_ppl:.3f}")
lines.append(f"- Wall time: {(time.time()-wall_start)/60:.1f} min; fits={fits_completed}")
lines.append("")
lines.append("## Aggregate")
lines.append("")
lines.append("| subset | params | n_seeds | init_mean | best_mean ± std | best_step | final@4000 |")
lines.append("|---|---|---|---|---|---|---|")
for k in aggregates:
    a = aggregates[k]
    lines.append(f"| {a['subset']} | {a['trainable_params']:,} | {a['n_seeds']} | {a['init_mean']:.3f} | **{a['best_mean']:.3f} ± {a['best_std']:.3f}** | {a['best_step_mean']:.0f} ± {a['best_step_std']:.0f} | {a['final_mean']:.3f} |")
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
lines.append(f"- Cycle 271 [1] K=512:        **{CYCLE_271_K512_1BLOCK_B1:.3f}** (393,984 params)")
lines.append(f"- Cycle 271 [4] K=512:        {CYCLE_271_K512_1BLOCK_B4:.3f}")
lines.append(f"- Cycle 271 [7] K=512:        {CYCLE_271_K512_1BLOCK_B7:.3f}")
lines.append(f"- Cycle 283 [2] K=64:         {CYCLE_283_B2_K64:.3f} (49,920 params)")
lines.append(f"- Cycle 283 [2] K=32:         {CYCLE_283_B2_K32:.3f}")
lines.append(f"- Cycle 283 [2] K=16:         {CYCLE_283_B2_K16:.3f}")
lines.append(f"- Float ref:                  {FLOAT_BASELINE_REF:.3f}")
lines.append(f"- Cycle 286 [2] K=512:        **{best_mean:.3f}** (393,984 params)")
lines.append(f"- Delta vs cycle 271 [1] K=512:  **{best_mean - CYCLE_271_K512_1BLOCK_B1:+.3f}** PPL")
lines.append(f"- Delta vs cycle 283 [2] K=64:   **{best_mean - CYCLE_283_B2_K64:+.3f}** PPL")
lines.append(f"- Delta vs float:                **{best_mean - float(float_ppl):+.3f}** PPL")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
