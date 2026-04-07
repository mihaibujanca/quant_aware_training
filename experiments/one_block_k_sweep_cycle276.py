"""
Cycle 276 — Item 286: 1-block [1] K sweep — K ∈ {32, 64, 128, 256}.

SUBSETS=[[1]]; K_VALUES=[32, 64, 128, 256]. Adapted from cycle 271 template.
4 K × 3 seeds {42, 137, 2024} = 12 fits total.
Drop seed 2024 if wall-time cap exceeded after first 8 fits.

Joint Adam W_out + b_out, 4000 steps, eval every 100, best-state restore.

Verdict per item 286:
  CONFIRMED   if K=256 best_mean ≤ 45.5 OR K=128 best_mean ≤ 47.0
  REFUTED     if all of {K=32, K=64, K=128, K=256} best_mean ≥ 50.0
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
N_STEPS = 4000
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
WALL_TIME_CAP_SEC = 30 * 60  # after first 8 fits, drop seed 2024 if elapsed > 30 min
OUTPUT_PATH = "autoresearch/results/cycle_276_one_block_k_sweep.json"
SUMMARY_PATH = "autoresearch/results/cycle_276_summary.md"

# Baselines (PPL)
CYCLE_271_K512_1BLOCK = 43.82045486551152  # cycle 271 [1] K=512 best_mean
CYCLE_272_K256_4BLOCK_MIN = 51.922407561728676  # cycle 272 [1,2,4,7] K=256
CYCLE_268_K64_4BLOCK_MIN = 55.953295278048564  # cycle 268 [1,4,7,8] K=64
CYCLE_230_K1_3BLOCK = 59.2454  # cycle 230 [1,4,7] K=1 best_mean
FLOAT_BASELINE_REF = 53.895

# Trainable per K (block 1 only)
EXPECTED_TRAINABLE = {K: K * 768 + 768 for K in K_VALUES}

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


# ── Helpers (verbatim from cycle 271 / 266 / 255) ─────────────────────────────
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
        "K": K,
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


# ── Sweep: 4 K x 3 seeds = 12 fits; fall back to 2 seeds if wall cap ──────────
print(f"\n{'#'*60}\nSWEEP: {len(K_VALUES)} K x {len(SEEDS)} seeds = {len(K_VALUES)*len(SEEDS)} fits\n{'#'*60}")
wall_start = time.time()
k_results = {}
fits_completed = 0
active_seeds = list(SEEDS)

for K in K_VALUES:
    key = str(K)
    k_results[key] = []
    for s in list(active_seeds):
        # After first 8 fits, if wall time exceeds cap, drop seed 2024 from remaining.
        elapsed = time.time() - wall_start
        if fits_completed >= 8 and elapsed > WALL_TIME_CAP_SEC and 2024 in active_seeds:
            print(f"\n[WALL TIME CAP] elapsed {elapsed/60:.1f} min > 30 min after {fits_completed} fits -- dropping seed 2024 for remaining K")
            active_seeds = [x for x in active_seeds if x != 2024]
            if s == 2024:
                continue
        r = run_seed(K, s)
        k_results[key].append(r)
        fits_completed += 1
        print(f"[fit {fits_completed} done, elapsed {(time.time()-wall_start)/60:.1f} min]")


# ── Aggregate per-K ───────────────────────────────────────────────────────────
def agg_key(results, key):
    vals = torch.tensor([r[key] for r in results], dtype=torch.float64)
    return float(vals.mean()), float(vals.std(unbiased=False))


aggregates = {}
for K in K_VALUES:
    key = str(K)
    results = k_results[key]
    if not results:
        continue
    init_mean, init_std = agg_key(results, "init_ppl")
    best_mean, best_std = agg_key(results, "best_ppl")
    best_step_mean, best_step_std = agg_key(results, "best_step")
    final_mean, final_std = agg_key(results, "final_ppl_at_step_4000")
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
    }

# ── Verdict per item 286 ──────────────────────────────────────────────────────
best_means = {K: aggregates[str(K)]["best_mean"] for K in K_VALUES if str(K) in aggregates}
k256_bm = best_means.get(256, float("inf"))
k128_bm = best_means.get(128, float("inf"))

confirmed_clause = (k256_bm <= 45.5) or (k128_bm <= 47.0)
refuted_clause = all(bm >= 50.0 for bm in best_means.values()) and len(best_means) == len(K_VALUES)

if confirmed_clause:
    verdict = "CONFIRMED"
elif refuted_clause:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
print(f"Float PPL:                       {float_ppl:.3f}")
print(f"Cycle 271 [1] K=512 baseline:    {CYCLE_271_K512_1BLOCK:.3f}")
print(f"Cycle 272 K=256 4-block min:     {CYCLE_272_K256_4BLOCK_MIN:.3f}")
print(f"Cycle 268 K=64 4-block min:      {CYCLE_268_K64_4BLOCK_MIN:.3f}")
print(f"Cycle 230 K=1 3-block:           {CYCLE_230_K1_3BLOCK:.3f}")
for K in K_VALUES:
    if str(K) not in aggregates:
        continue
    a = aggregates[str(K)]
    print(f"  K={K} (n={a['n_seeds']}, params={a['trainable_params']:,}): "
          f"best_mean={a['best_mean']:.3f} ± {a['best_std']:.3f}  "
          f"init {a['init_mean']:.3f}  final@4000 {a['final_mean']:.3f}")
print(f"\nVerdict logic: CONFIRMED iff K=256 ≤ 45.5 OR K=128 ≤ 47.0; "
      f"REFUTED iff all K ≥ 50.0; else INCONCLUSIVE")
print(f"  k256_bm={k256_bm:.3f} ≤ 45.5? {k256_bm <= 45.5}")
print(f"  k128_bm={k128_bm:.3f} ≤ 47.0? {k128_bm <= 47.0}")
print(f"  refuted_clause (all ≥ 50)? {refuted_clause}")
print(f"Verdict: {verdict}")

result = {
    "cycle": 276,
    "item": 286,
    "verdict": verdict,
    "verdict_logic": {
        "rule": "CONFIRMED iff K=256 best_mean <= 45.5 OR K=128 best_mean <= 47.0; REFUTED iff all of {K=32, K=64, K=128, K=256} best_mean >= 50.0; else INCONCLUSIVE.",
        "k256_best_mean": k256_bm,
        "k128_best_mean": k128_bm,
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
    "best_means": best_means,
    "baselines": {
        "cycle_271_k512_1block": CYCLE_271_K512_1BLOCK,
        "cycle_272_k256_4block_min": CYCLE_272_K256_4BLOCK_MIN,
        "cycle_268_k64_4block_min": CYCLE_268_K64_4BLOCK_MIN,
        "cycle_230_k1_3block": CYCLE_230_K1_3BLOCK,
        "float_baseline_ref": FLOAT_BASELINE_REF,
    },
    "deltas_vs_cycle_271_k512_1block": {
        K: aggregates[str(K)]["best_mean"] - CYCLE_271_K512_1BLOCK
        for K in K_VALUES if str(K) in aggregates
    },
    "wall_time_sec": time.time() - wall_start,
    "fits_completed": fits_completed,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON to {OUTPUT_PATH}")

# ── Summary markdown ──────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 276 — Item 286: 1-block [1] K sweep")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Criteria: CONFIRMED if K=256 best_mean ≤ 45.5 OR K=128 best_mean ≤ 47.0; REFUTED if all K ≥ 50.0; else INCONCLUSIVE")
lines.append(f"- BLOCKS: {BLOCKS}; K_VALUES: {K_VALUES}")
lines.append(f"- Config: LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}")
lines.append(f"- Float PPL: {float_ppl:.3f}")
lines.append(f"- Wall time: {(time.time()-wall_start)/60:.1f} min; fits={fits_completed}")
lines.append("")
lines.append("## Per-K aggregates")
lines.append("")
lines.append("| K | params | n_seeds | init_mean | best_mean ± std | best_min/max | best_step | final@4000 |")
lines.append("|---|---|---|---|---|---|---|---|")
for K in K_VALUES:
    if str(K) not in aggregates:
        continue
    a = aggregates[str(K)]
    lines.append(
        f"| {a['K']} | {a['trainable_params']:,} | {a['n_seeds']} | {a['init_mean']:.3f} | "
        f"**{a['best_mean']:.3f} ± {a['best_std']:.3f}** | {a['best_min']:.3f} / {a['best_max']:.3f} | "
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
    lines.append("| seed | init | best | best_step | final@4000 |")
    lines.append("|---|---|---|---|---|")
    for r in k_results[key]:
        lines.append(f"| {r['seed']} | {r['init_ppl']:.3f} | {r['best_ppl']:.3f} | {r['best_step']} | {r['final_ppl_at_step_4000']:.3f} |")
    lines.append("")
lines.append("## Comparison")
lines.append("")
lines.append(f"- Cycle 271 [1] K=512 (394K):           **{CYCLE_271_K512_1BLOCK:.3f}**")
lines.append(f"- Cycle 272 K=256 4-block [1,2,4,7] (789K): {CYCLE_272_K256_4BLOCK_MIN:.3f}")
lines.append(f"- Cycle 268 K=64 4-block [1,4,7,8] (197K):  {CYCLE_268_K64_4BLOCK_MIN:.3f}")
lines.append(f"- Cycle 230 K=1 3-block [1,4,7] (5.8K):    {CYCLE_230_K1_3BLOCK:.3f}")
lines.append(f"- Float ref:                                {FLOAT_BASELINE_REF:.3f}")
lines.append("")
for K in K_VALUES:
    if str(K) not in aggregates:
        continue
    bm = aggregates[str(K)]["best_mean"]
    delta = bm - CYCLE_271_K512_1BLOCK
    lines.append(f"- 1-block K={K} best_mean: **{bm:.3f}**, Δ vs cycle 271 K=512: **{delta:+.3f}** PPL")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
