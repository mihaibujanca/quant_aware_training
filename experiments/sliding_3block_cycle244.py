"""
Cycle 244 — Item 254: Sliding 3-block sweep.

Cycle 242 tested [1,2,3] (Δ=+0.58) and [8,9,10] (Δ=+8.64). Fill the gap:
map ΔPPL vs window start index across three mid/late-range windows
  CONFIG A: [3,4,5]
  CONFIG B: [5,6,7]
  CONFIG C: [6,7,8]
Recipe mirrors cycle 242: topk+lstsq init -> joint Adam on W_out+b_out only,
K=512, 3 seeds {42,137,2024}, disjoint calib=all_ids[:200]/eval=all_ids[200:300],
SEQ_LEN=128, best-state snapshot+restore.

Verdict logic (item 254):
- CONFIRMED if any window mean_best <= 48.5
- REFUTED if all window mean_best >= 52.0
- INCONCLUSIVE otherwise
"""
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

CONFIGS = [[3, 4, 5], [5, 6, 7], [6, 7, 8]]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 4000
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
OUTPUT_PATH = "autoresearch/results/cycle_244_sliding_3block.json"
SUMMARY_PATH = "autoresearch/results/cycle_244_summary.md"
CYCLE_210_BASELINE = 46.556       # BLOCKS=[1,4,7] W_out+b_out joint Adam 3-block
CYCLE_210_STD = 0.028
CYCLE_242_EARLY = 47.135          # BLOCKS=[1,2,3] cycle 242 anchor
CYCLE_242_EARLY_STD = 0.093
CYCLE_242_LATE = 55.192           # BLOCKS=[8,9,10] cycle 242 anchor
CYCLE_242_LATE_STD = 0.040

print(f"Device: {DEVICE}")
print(f"CONFIGS: {CONFIGS}")
print(f"SEEDS: {SEEDS}")

# ── Load data (once, seed 42) ────────────────────────────────────────────────
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
torch.manual_seed(42)
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

# Sanity: disjoint by object identity
assert len(set(id(t) for t in calib_ids) & set(id(t) for t in eval_ids)) == 0


class CompressedMLP(nn.Module):
    """W_out + b_out trainable. W_fc submatrix + b_fc subset frozen buffers."""
    def __init__(self, fc_weight_sub, fc_bias_sub, W_out, b_out):
        super().__init__()
        self.register_buffer("fc_weight", fc_weight_sub)  # (768, K)
        self.register_buffer("fc_bias", fc_bias_sub)       # (K,)
        self.W_out = nn.Parameter(W_out.clone())           # (768, K)
        self.b_out = nn.Parameter(b_out.clone())           # (768,)

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
    W_fc = block.mlp.c_fc.weight.data.float().cpu()        # (768, 3072)
    b_fc = block.mlp.c_fc.bias.data.float().cpu()          # (3072,)
    W_proj = block.mlp.c_proj.weight.data.float().cpu()    # (3072, 768)
    act_all = F.gelu(ln_out_cpu @ W_fc + b_fc)              # (N, 3072)
    scores = act_all.std(0) * W_proj.norm(dim=1)
    topk_idx = scores.topk(K).indices                       # (K,)
    W_fc_sub = W_fc[:, topk_idx]                            # (768, K)
    b_fc_sub = b_fc[topk_idx]                                # (K,)
    act_sub = act_all[:, topk_idx]                           # (N, K)
    ones = torch.ones(act_sub.shape[0], 1)
    sol = torch.linalg.lstsq(torch.cat([act_sub, ones], 1), proj_out_cpu, rcond=None).solution
    return CompressedMLP(
        fc_weight_sub=W_fc_sub,
        fc_bias_sub=b_fc_sub,
        W_out=sol[:K].T,    # (768, K)
        b_out=sol[K],       # (768,)
    )


def eval_ppl(model, eval_ids):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


def run_cfg_seed(cfg_blocks, seed):
    cfg_label = "_".join(str(b) for b in cfg_blocks)
    print(f"\n{'='*60}\nCONFIG {cfg_blocks} | SEED {seed}\n{'='*60}")
    torch.manual_seed(seed)
    random.seed(seed)

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    ln_outs, proj_outs = collect_mlp_io(model, calib_ids, cfg_blocks)
    for b in cfg_blocks:
        model.transformer.h[b].mlp = fit_block_lstsq(
            model.transformer.h[b], ln_outs[b], proj_outs[b], K).to(DEVICE)
    del ln_outs, proj_outs

    init_ppl = eval_ppl(model, eval_ids)
    print(f"Init PPL: {init_ppl:.3f}")

    for param in model.parameters():
        param.requires_grad_(False)
    trainable_params = []
    for b in cfg_blocks:
        mlp = model.transformer.h[b].mlp
        assert not mlp.fc_weight.requires_grad, f"block {b} fc_weight leaked grad"
        mlp.W_out.requires_grad_(True)
        mlp.b_out.requires_grad_(True)
        trainable_params.extend([mlp.W_out, mlp.b_out])
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_trainable:,} ({n_trainable/1e6:.3f}M)")

    # Untouched blocks must remain frozen
    for b in range(12):
        if b in cfg_blocks:
            assert not isinstance(model.transformer.h[b].mlp.fc_weight, nn.Parameter)
        else:
            assert not model.transformer.h[b].mlp.c_fc.weight.requires_grad

    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    calib_shuffled = list(calib_ids)
    random.shuffle(calib_shuffled)

    def snapshot():
        return {b: (model.transformer.h[b].mlp.W_out.detach().clone().cpu(),
                    model.transformer.h[b].mlp.b_out.detach().clone().cpu())
                for b in cfg_blocks}

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

    # Restore best
    for b in cfg_blocks:
        W_best, b_best = best_state[b]
        model.transformer.h[b].mlp.W_out.data.copy_(W_best.to(DEVICE))
        model.transformer.h[b].mlp.b_out.data.copy_(b_best.to(DEVICE))
    final_ppl_restored = eval_ppl(model, eval_ids)
    print(f"Restored best (step {best_step}): {final_ppl_restored:.3f} (recorded={best_ppl:.3f})")
    assert abs(final_ppl_restored - best_ppl) < 1e-3, \
        f"Restore mismatch: {final_ppl_restored} vs {best_ppl}"

    result = {
        "cfg": cfg_label,
        "blocks": cfg_blocks,
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


# ── Float baseline (once) ────────────────────────────────────────────────────
print("\nFloat baseline...")
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()
float_ppl = eval_ppl(model_float, eval_ids)
print(f"Float PPL: {float_ppl:.3f}")
del model_float
if DEVICE == "mps":
    torch.mps.empty_cache()

# ── Outer loop: configs × seeds ──────────────────────────────────────────────
all_results = {}  # keyed by "_".join(blocks)
for cfg_blocks in CONFIGS:
    cfg_label = "_".join(str(b) for b in cfg_blocks)
    all_results[cfg_label] = []
    for seed in SEEDS:
        all_results[cfg_label].append(run_cfg_seed(cfg_blocks, seed))


def agg_list(vals):
    t = torch.tensor(vals, dtype=torch.float64)
    return float(t.mean()), float(t.std(unbiased=False))


configs_agg = {}
for cfg_blocks in CONFIGS:
    cfg_label = "_".join(str(b) for b in cfg_blocks)
    seeds_out = all_results[cfg_label]
    best_vals = [r["best_ppl"] for r in seeds_out]
    init_vals = [r["init_ppl"] for r in seeds_out]
    final_vals = [r["final_ppl_at_step_4000"] for r in seeds_out]
    best_steps = [r["best_step"] for r in seeds_out]
    mean_best, std_best = agg_list(best_vals)
    mean_init, std_init = agg_list(init_vals)
    mean_final, std_final = agg_list(final_vals)
    configs_agg[cfg_label] = {
        "blocks": cfg_blocks,
        "mean_ppl": mean_best,
        "std_ppl": std_best,
        "per_seed_ppl": best_vals,
        "per_seed_init": init_vals,
        "per_seed_final_at_4000": final_vals,
        "per_seed_best_step": best_steps,
        "init_mean": mean_init,
        "init_std": std_init,
        "final_mean": mean_final,
        "final_std": std_final,
        "delta_vs_c210_baseline": float(mean_best - CYCLE_210_BASELINE),
        "delta_vs_c242_early": float(mean_best - CYCLE_242_EARLY),
        "delta_vs_c242_late": float(mean_best - CYCLE_242_LATE),
        "per_seed_history": [r["history"] for r in seeds_out],
    }

# ── Verdict ──────────────────────────────────────────────────────────────────
means = {lab: configs_agg[lab]["mean_ppl"] for lab in configs_agg}
any_le_485 = any(m <= 48.5 for m in means.values())
all_ge_520 = all(m >= 52.0 for m in means.values())
if any_le_485:
    verdict = "CONFIRMED"
elif all_ge_520:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
print(f"Float PPL (held-out):         {float_ppl:.3f}")
print(f"Cycle 210 [1,4,7] baseline:   {CYCLE_210_BASELINE} ± {CYCLE_210_STD}")
print(f"Cycle 242 [1,2,3] anchor:     {CYCLE_242_EARLY} ± {CYCLE_242_EARLY_STD}")
print(f"Cycle 242 [8,9,10] anchor:    {CYCLE_242_LATE} ± {CYCLE_242_LATE_STD}")
for cfg_blocks in CONFIGS:
    cl = "_".join(str(b) for b in cfg_blocks)
    ca = configs_agg[cl]
    print(f"CONFIG {cfg_blocks} mean±std:     "
          f"{ca['mean_ppl']:.3f} ± {ca['std_ppl']:.3f}  "
          f"Δvs[1,4,7]={ca['delta_vs_c210_baseline']:+.3f}  "
          f"Δvs[1,2,3]={ca['delta_vs_c242_early']:+.3f}  "
          f"Δvs[8,9,10]={ca['delta_vs_c242_late']:+.3f}")
print(f"Verdict: {verdict}")

result = {
    "cycle": 244,
    "item": 254,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "baseline_1_4_7": CYCLE_210_BASELINE,
    "baseline_1_4_7_std": CYCLE_210_STD,
    "anchor_c242_1_2_3": CYCLE_242_EARLY,
    "anchor_c242_1_2_3_std": CYCLE_242_EARLY_STD,
    "anchor_c242_8_9_10": CYCLE_242_LATE,
    "anchor_c242_8_9_10_std": CYCLE_242_LATE_STD,
    "configs": configs_agg,
    "per_window_means": means,
    "verdict_any_le_48_5": any_le_485,
    "verdict_all_ge_52_0": all_ge_520,
    "config": {
        "K": K,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "LR": LR,
        "N_STEPS": N_STEPS,
        "EVAL_EVERY": EVAL_EVERY,
        "SEEDS": SEEDS,
        "CONFIGS": CONFIGS,
        "trainable_surface": "W_out + b_out (W_fc matrix + b_fc frozen)",
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON to {OUTPUT_PATH}")

# ── Summary markdown ─────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 244 — Item 254: Sliding 3-block sweep")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Criteria: CONFIRMED if any window mean_best ≤ 48.5; REFUTED if all ≥ 52.0; else INCONCLUSIVE")
lines.append(f"- Float PPL (held-out, same split as cycles 210/241/242): **{float_ppl:.3f}**")
lines.append(f"- Cycle 210 [1,4,7] 3-block joint Adam baseline: **{CYCLE_210_BASELINE} ± {CYCLE_210_STD}**")
lines.append(f"- Cycle 242 [1,2,3] early anchor: **{CYCLE_242_EARLY} ± {CYCLE_242_EARLY_STD}**")
lines.append(f"- Cycle 242 [8,9,10] late anchor: **{CYCLE_242_LATE} ± {CYCLE_242_LATE_STD}**")
lines.append(f"- Config: K={K}, LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}")
lines.append(f"- Trainable surface per block: W_out + b_out (no W_fc / b_fc training)")
lines.append("")
lines.append("## Per-config × per-seed best PPL")
lines.append("")
lines.append("| config | seed | init | best | best_step | final@4000 |")
lines.append("|---|---|---|---|---|---|")
for cfg_blocks in CONFIGS:
    cfg_label = "_".join(str(b) for b in cfg_blocks)
    for r in all_results[cfg_label]:
        lines.append(f"| {cfg_blocks} | {r['seed']} | {r['init_ppl']:.3f} | {r['best_ppl']:.3f} | {r['best_step']} | {r['final_ppl_at_step_4000']:.3f} |")
    ca = configs_agg[cfg_label]
    lines.append(f"| **{cfg_blocks} mean** |  | {ca['init_mean']:.3f} | **{ca['mean_ppl']:.3f}** |  | {ca['final_mean']:.3f} |")
    lines.append(f"| **{cfg_blocks} std**  |  | {ca['init_std']:.3f} | {ca['std_ppl']:.3f} |  | {ca['final_std']:.3f} |")
lines.append("")
lines.append("## Comparison to baselines / anchors")
lines.append("")
lines.append(f"| config | mean_best ± std | Δ vs [1,4,7]={CYCLE_210_BASELINE:.3f} | Δ vs c242 [1,2,3]={CYCLE_242_EARLY:.3f} | Δ vs c242 [8,9,10]={CYCLE_242_LATE:.3f} |")
lines.append("|---|---|---|---|---|")
for cfg_blocks in CONFIGS:
    cfg_label = "_".join(str(b) for b in cfg_blocks)
    ca = configs_agg[cfg_label]
    lines.append(f"| {cfg_blocks} | {ca['mean_ppl']:.3f} ± {ca['std_ppl']:.3f} | {ca['delta_vs_c210_baseline']:+.3f} | {ca['delta_vs_c242_early']:+.3f} | {ca['delta_vs_c242_late']:+.3f} |")
lines.append("")
lines.append("## Verdict rationale")
lines.append("")
for cfg_blocks in CONFIGS:
    cfg_label = "_".join(str(b) for b in cfg_blocks)
    ca = configs_agg[cfg_label]
    lines.append(f"- {cfg_blocks} mean_best = {ca['mean_ppl']:.3f}")
lines.append(f"- any ≤ 48.5? {any_le_485}; all ≥ 52.0? {all_ge_520}")
if verdict == "CONFIRMED":
    lines.append(f"- **CONFIRMED**: at least one window matches the early-cascade regime (≤48.5).")
elif verdict == "REFUTED":
    lines.append(f"- **REFUTED**: every mid/late window is uniformly hard (all ≥52.0).")
else:
    lines.append(f"- **INCONCLUSIVE**: windows straddle the 48.5–52.0 band.")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
