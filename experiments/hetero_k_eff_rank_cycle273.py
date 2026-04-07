"""
Cycle 273 — Item 283: Heterogeneous K guided by cycle 267 eff_rank.
BLOCKS=[1, 4, 7, 11]. Two configs:
  hetero    : K_PER_BLOCK = {1:512, 4:512, 7:512, 11:22}  -- block 11 at its eff_rank=22 budget.
  uniform_k22: K_PER_BLOCK = {1:22, 4:22, 7:22, 11:22}    -- isolation control.

Mirrors cycle 247/255 protocol: joint Adam W_out+b_out, eval every 100 steps,
best-state snapshot/restore, across 3 seeds. 6 fits total.

Verdict per item 283:
  CONFIRMED   iff hetero best_mean ≤ 44.5 AND uniform_k22 best_mean ≥ 60.0.
  REFUTED     iff hetero best_mean ≥ 47.0.
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

BLOCKS = [1, 4, 7, 11]
CONFIGS = [
    {"name": "hetero",      "k_per_block": {1: 512, 4: 512, 7: 512, 11: 22}},
    {"name": "uniform_k22", "k_per_block": {1: 22,  4: 22,  7: 22,  11: 22}},
]
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 4000
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
OUTPUT_PATH = "autoresearch/results/cycle_273_hetero_eff_rank.json"
SUMMARY_PATH = "autoresearch/results/cycle_273_summary.md"

# Baselines (from agenda item 283 / brief)
CYCLE_266_K512_2BLOCK = 43.852      # [1,4] K=512 (cycle 266)
CYCLE_270_BLOCK11_R22 = 54.469      # block 11 single-block rank-22 retrain
CYCLE_247_HETERO_6BLOCK = 64.45     # cycle 247 hetero K {4,5}=512 6-block
CYCLE_255_K512_4BLOCK = 49.609      # cycle 255 K=512 4-block min [1,2,4,7]
CYCLE_268_K64_4BLOCK = 55.953       # cycle 268 K=64 4-block min
FLOAT_BASELINE_REF = 53.895

# Verdict thresholds (item 283)
THRESH_CONFIRMED_HETERO_LE = 44.5
THRESH_CONFIRMED_CONTROL_GE = 60.0
THRESH_REFUTED_HETERO_GE = 47.0

print(f"Device: {DEVICE}")
print(f"BLOCKS: {BLOCKS}")
for cfg in CONFIGS:
    trainable = sum(cfg["k_per_block"][b] * 768 + 768 for b in BLOCKS)
    print(f"  cfg={cfg['name']:12s}  K_PER_BLOCK={cfg['k_per_block']}  trainable={trainable:,}")

# Sanity-check expected counts.
EXPECTED_HETERO = 3 * (512 * 768 + 768) + 1 * (22 * 768 + 768)  # = 1,198,080
EXPECTED_UNIFORM = 4 * (22 * 768 + 768)                          # = 70,656
assert sum(CONFIGS[0]["k_per_block"][b] * 768 + 768 for b in BLOCKS) == EXPECTED_HETERO
assert sum(CONFIGS[1]["k_per_block"][b] * 768 + 768 for b in BLOCKS) == EXPECTED_UNIFORM
print(f"Expected trainable: hetero={EXPECTED_HETERO:,}, uniform_k22={EXPECTED_UNIFORM:,}")

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


# ── Helpers ───────────────────────────────────────────────────────────────────
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
    mlp = CompressedMLP(W_fc[:, topk_idx], b_fc[topk_idx], sol[:K].T, sol[K])
    with torch.no_grad():
        recon = act_sub @ sol[:K] + sol[K]
        residual_mse = float(((recon - proj_out_cpu) ** 2).mean())
    return mlp, residual_mse


def eval_ppl(model, eval_ids):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


# ── Per-(config, seed) run ────────────────────────────────────────────────────
def run_seed(seed, k_per_block, cfg_name):
    print(f"\n{'='*60}\nCFG {cfg_name}  SEED {seed}\n{'='*60}")
    torch.manual_seed(seed)
    random.seed(seed)

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    ln_outs, proj_outs = collect_mlp_io(model, calib_ids, BLOCKS)
    per_block_init_mse = {}
    for b in BLOCKS:
        K = k_per_block[b]
        mlp, mse = fit_block_lstsq(
            model.transformer.h[b], ln_outs[b], proj_outs[b], K)
        model.transformer.h[b].mlp = mlp.to(DEVICE)
        per_block_init_mse[b] = mse
        print(f"  block {b} K={K}: residual MSE={mse:.4e}")
    del ln_outs, proj_outs

    init_ppl = eval_ppl(model, eval_ids)
    print(f"Init PPL: {init_ppl:.3f}")

    # Freeze all, unfreeze W_out/b_out for BLOCKS.
    for param in model.parameters():
        param.requires_grad_(False)
    trainable_params = []
    for b in BLOCKS:
        model.transformer.h[b].mlp.W_out.requires_grad_(True)
        model.transformer.h[b].mlp.b_out.requires_grad_(True)
        trainable_params.extend([model.transformer.h[b].mlp.W_out,
                                  model.transformer.h[b].mlp.b_out])
    n_trainable = sum(p.numel() for p in trainable_params)
    expected = sum(k_per_block[b] * 768 + 768 for b in BLOCKS)
    print(f"Trainable params: {n_trainable:,} ({n_trainable/1e6:.3f}M)")
    assert n_trainable == expected, \
        f"Trainable mismatch: got {n_trainable}, expected {expected}"

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
        "cfg_name": cfg_name,
        "seed": seed,
        "k_per_block": {str(b): int(k_per_block[b]) for b in BLOCKS},
        "init_ppl": float(init_ppl),
        "best_ppl": float(best_ppl),
        "best_step": int(best_step),
        "final_ppl_at_step_4000": float(final_ppl_at_step_4000),
        "trainable_params": int(n_trainable),
        "per_block_init_mse": {str(b): per_block_init_mse[b] for b in BLOCKS},
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

# ── Sweep: 2 configs × 3 seeds = 6 fits ───────────────────────────────────────
print(f"\n{'#'*60}\nSWEEP: {len(CONFIGS)} configs × {len(SEEDS)} seeds = {len(CONFIGS)*len(SEEDS)} fits\n{'#'*60}")
all_results = {cfg["name"]: [] for cfg in CONFIGS}
for cfg in CONFIGS:
    for s in SEEDS:
        r = run_seed(s, cfg["k_per_block"], cfg["name"])
        all_results[cfg["name"]].append(r)

# ── Aggregate per cfg ─────────────────────────────────────────────────────────
def agg_key(results, key):
    vals = torch.tensor([r[key] for r in results], dtype=torch.float64)
    return float(vals.mean()), float(vals.std(unbiased=False))


aggregates = {}
for cfg in CONFIGS:
    name = cfg["name"]
    results = all_results[name]
    init_mean, init_std = agg_key(results, "init_ppl")
    best_mean, best_std = agg_key(results, "best_ppl")
    best_step_mean, best_step_std = agg_key(results, "best_step")
    final_mean, final_std = agg_key(results, "final_ppl_at_step_4000")
    aggregates[name] = {
        "k_per_block": {str(b): int(cfg["k_per_block"][b]) for b in BLOCKS},
        "n_seeds": len(results),
        "seeds": [r["seed"] for r in results],
        "init_mean": init_mean, "init_std": init_std,
        "best_mean": best_mean, "best_std": best_std,
        "best_step_mean": best_step_mean, "best_step_std": best_step_std,
        "final_mean": final_mean, "final_std": final_std,
        "trainable_params": results[0]["trainable_params"],
    }

hetero_best = aggregates["hetero"]["best_mean"]
control_best = aggregates["uniform_k22"]["best_mean"]

# ── Verdict per item 283 ──────────────────────────────────────────────────────
if hetero_best <= THRESH_CONFIRMED_HETERO_LE and control_best >= THRESH_CONFIRMED_CONTROL_GE:
    verdict = "CONFIRMED"
elif hetero_best >= THRESH_REFUTED_HETERO_GE:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
print(f"Float PPL:                      {float_ppl:.3f}")
for name in aggregates:
    a = aggregates[name]
    print(f"  {name:12s}: best_mean={a['best_mean']:.3f} ± {a['best_std']:.3f}  init {a['init_mean']:.3f}  final@4000 {a['final_mean']:.3f}  trainable={a['trainable_params']:,}")
print(f"hetero best_mean:               {hetero_best:.3f}")
print(f"uniform_k22 best_mean:          {control_best:.3f}")
print(f"Cycle 266 K=512 [1,4] baseline: {CYCLE_266_K512_2BLOCK}")
print(f"Cycle 270 block11 r=22:         {CYCLE_270_BLOCK11_R22}")
print(f"Cycle 255 K=512 4-block min:    {CYCLE_255_K512_4BLOCK}")
print(f"Float baseline (run):           {float_ppl:.3f}")
print(f"Δ hetero vs cycle 266 [1,4]:    {hetero_best - CYCLE_266_K512_2BLOCK:+.3f}")
print(f"Δ hetero vs cycle 255 4-block:  {hetero_best - CYCLE_255_K512_4BLOCK:+.3f}")
print(f"Δ hetero vs float:              {hetero_best - float(float_ppl):+.3f}")
print(f"Verdict thresholds: CONFIRMED hetero ≤ {THRESH_CONFIRMED_HETERO_LE} AND control ≥ {THRESH_CONFIRMED_CONTROL_GE}; "
      f"REFUTED hetero ≥ {THRESH_REFUTED_HETERO_GE}")
print(f"Verdict: {verdict}")

result = {
    "cycle": 273,
    "item": 283,
    "verdict": verdict,
    "config": {
        "BLOCKS": BLOCKS,
        "CONFIGS": [{"name": c["name"], "k_per_block": {str(b): int(c["k_per_block"][b]) for b in BLOCKS}} for c in CONFIGS],
        "N_CALIB": N_CALIB, "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN, "LR": LR, "N_STEPS": N_STEPS,
        "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
        "trainable_params_hetero": EXPECTED_HETERO,
        "trainable_params_uniform_k22": EXPECTED_UNIFORM,
    },
    "float_ppl": float(float_ppl),
    "all_results": all_results,
    "aggregates": aggregates,
    "hetero_best_mean": hetero_best,
    "control_best_mean": control_best,
    "thresholds": {
        "confirmed_hetero_le": THRESH_CONFIRMED_HETERO_LE,
        "confirmed_control_ge": THRESH_CONFIRMED_CONTROL_GE,
        "refuted_hetero_ge": THRESH_REFUTED_HETERO_GE,
    },
    "baselines": {
        "cycle_266_k512_2block_1_4": CYCLE_266_K512_2BLOCK,
        "cycle_270_block11_r22": CYCLE_270_BLOCK11_R22,
        "cycle_247_hetero_6block": CYCLE_247_HETERO_6BLOCK,
        "cycle_255_k512_4block": CYCLE_255_K512_4BLOCK,
        "cycle_268_k64_4block": CYCLE_268_K64_4BLOCK,
        "float_baseline_ref": FLOAT_BASELINE_REF,
    },
    "delta_hetero_vs_cycle_266": hetero_best - CYCLE_266_K512_2BLOCK,
    "delta_hetero_vs_cycle_255": hetero_best - CYCLE_255_K512_4BLOCK,
    "delta_hetero_vs_float": hetero_best - float(float_ppl),
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON to {OUTPUT_PATH}")

# ── Summary markdown ──────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 273 — Item 283: eff_rank-guided heterogeneous K")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Criteria: CONFIRMED iff hetero best_mean ≤ {THRESH_CONFIRMED_HETERO_LE} AND uniform_k22 best_mean ≥ {THRESH_CONFIRMED_CONTROL_GE}; "
             f"REFUTED iff hetero ≥ {THRESH_REFUTED_HETERO_GE}; else INCONCLUSIVE")
lines.append(f"- BLOCKS = {BLOCKS}")
lines.append(f"- hetero K_PER_BLOCK = {CONFIGS[0]['k_per_block']}; trainable = {EXPECTED_HETERO:,}")
lines.append(f"- uniform_k22 K_PER_BLOCK = {CONFIGS[1]['k_per_block']}; trainable = {EXPECTED_UNIFORM:,}")
lines.append(f"- LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}")
lines.append(f"- Float PPL: {float_ppl:.3f}")
lines.append("")
lines.append("## Per-config aggregates")
lines.append("")
lines.append("| cfg | trainable | init_mean | best_mean ± std | best_step | final@4000 |")
lines.append("|---|---|---|---|---|---|")
for name in aggregates:
    a = aggregates[name]
    lines.append(f"| {name} | {a['trainable_params']:,} | {a['init_mean']:.3f} | **{a['best_mean']:.3f} ± {a['best_std']:.3f}** | {a['best_step_mean']:.0f} ± {a['best_step_std']:.0f} | {a['final_mean']:.3f} |")
lines.append("")
lines.append("## Per-seed details")
lines.append("")
for name in all_results:
    a = aggregates[name]
    lines.append(f"### {name} (K_PER_BLOCK = {a['k_per_block']})")
    lines.append("")
    lines.append("| seed | init | best | best_step | final@4000 |")
    lines.append("|---|---|---|---|---|")
    for r in all_results[name]:
        lines.append(f"| {r['seed']} | {r['init_ppl']:.3f} | {r['best_ppl']:.3f} | {r['best_step']} | {r['final_ppl_at_step_4000']:.3f} |")
    lines.append("")
lines.append("## Per-block residual MSE (lstsq init, seed 42)")
lines.append("")
for name in all_results:
    a = aggregates[name]
    seed42 = next(r for r in all_results[name] if r["seed"] == 42)
    lines.append(f"### {name}")
    lines.append("")
    lines.append("| block | K | residual MSE |")
    lines.append("|---|---|---|")
    for b in BLOCKS:
        K = CONFIGS[0]["k_per_block"][b] if name == "hetero" else CONFIGS[1]["k_per_block"][b]
        lines.append(f"| {b} | {K} | {seed42['per_block_init_mse'][str(b)]:.4e} |")
    lines.append("")
lines.append("## Trajectory snippets")
lines.append("")
for name in all_results:
    lines.append(f"### {name}")
    for r in all_results[name]:
        hist = r["history"]
        init_line = f"init={r['init_ppl']:.2f}"
        q = hist[len(hist)//4] if hist else None
        h = hist[len(hist)//2] if hist else None
        tq = hist[3*len(hist)//4] if hist else None
        f_ = hist[-1] if hist else None
        parts = [init_line]
        if q: parts.append(f"step{q['step']}={q['eval_ppl']:.2f}")
        if h: parts.append(f"step{h['step']}={h['eval_ppl']:.2f}")
        parts.append(f"best=step{r['best_step']}@{r['best_ppl']:.2f}")
        if tq: parts.append(f"step{tq['step']}={tq['eval_ppl']:.2f}")
        if f_: parts.append(f"step{f_['step']}={f_['eval_ppl']:.2f}")
        lines.append(f"- seed {r['seed']}: " + ", ".join(parts))
    lines.append("")
lines.append("## Comparison")
lines.append("")
lines.append(f"- Cycle 266 K=512 2-block [1,4]:    **{CYCLE_266_K512_2BLOCK:.3f}**")
lines.append(f"- Cycle 270 block 11 rank-22 retrain: {CYCLE_270_BLOCK11_R22:.3f}")
lines.append(f"- Cycle 247 hetero K (6 blocks):    {CYCLE_247_HETERO_6BLOCK:.3f}")
lines.append(f"- Cycle 255 K=512 4-block min:      {CYCLE_255_K512_4BLOCK:.3f}")
lines.append(f"- Cycle 268 K=64 4-block min:       {CYCLE_268_K64_4BLOCK:.3f}")
lines.append(f"- Float ref:                        {FLOAT_BASELINE_REF:.3f}")
lines.append(f"- **hetero best_mean: {hetero_best:.3f}**")
lines.append(f"- **uniform_k22 best_mean: {control_best:.3f}**")
lines.append(f"- Δ hetero vs cycle 266 [1,4]: **{hetero_best - CYCLE_266_K512_2BLOCK:+.3f}** PPL")
lines.append(f"- Δ hetero vs cycle 255 K=512 4-block: **{hetero_best - CYCLE_255_K512_4BLOCK:+.3f}** PPL")
lines.append(f"- Δ hetero vs float (run): **{hetero_best - float(float_ppl):+.3f}** PPL")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
