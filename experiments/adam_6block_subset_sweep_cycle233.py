"""
Cycle 233 — Item 243: Alternative 6-block selections.

Sweep three alternative 6-block subsets to test whether cycle 221's 59.5 PPL
ceiling with BLOCKS=[1,2,4,5,7,8] is structural (any 6 blocks) or a
block-selection artifact.

Subsets:
  (a) [1,2,3,4,5,6] contiguous early
  (b) [3,4,5,6,7,8] contiguous middle
  (c) [1,3,5,7,8,10] spread

Phase 1: SEED=42 only, build topk+lstsq init, eval init, train Adam
         W_out+b_out 4000 steps with eval+best-state restore.
Phase 2: if any subset's best_ppl < 59.0, rerun that subset (and any tied
         within 0.5 PPL) with SEEDS=[42, 137, 2024].

Verdict:
  CONFIRMED if best_mean (phase 2 if run, else phase 1) ≤ 58.0 for any subset
  REFUTED   if all three subsets' best ∈ [59.0, 62.0]
  INCONCLUSIVE otherwise
"""
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

SUBSETS = [
    [1, 2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7, 8],
    [1, 3, 5, 7, 8, 10],
]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 4000
EVAL_EVERY = 100
SEEDS_P1 = [42]
SEEDS_P2 = [42, 137, 2024]
GATE_P2 = 59.0  # rerun subsets whose phase1 best < GATE_P2 with more seeds
TIE_WIDTH = 0.5  # and any tied within 0.5 PPL of gate-passing subsets
CYCLE_221_BASELINE = 59.515
OUTPUT_PATH = "autoresearch/results/cycle_233_6block_subset_sweep.json"
SUMMARY_PATH = "autoresearch/results/cycle_233_summary.md"

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


# ── Per-seed run (parameterised on BLOCKS) ────────────────────────────────────
def run_seed(blocks, seed):
    print(f"\n{'-'*60}\nSUBSET {blocks}  SEED {seed}\n{'-'*60}", flush=True)
    torch.manual_seed(seed)
    random.seed(seed)

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    ln_outs, proj_outs = collect_mlp_io(model, calib_ids, blocks)
    for b in blocks:
        model.transformer.h[b].mlp = fit_block_lstsq(
            model.transformer.h[b], ln_outs[b], proj_outs[b], K).to(DEVICE)
    del ln_outs, proj_outs

    init_ppl = eval_ppl(model, eval_ids)
    print(f"Init PPL: {init_ppl:.3f}", flush=True)

    # Freeze all, unfreeze W_out/b_out for blocks
    for param in model.parameters():
        param.requires_grad_(False)
    trainable_params = []
    for b in blocks:
        model.transformer.h[b].mlp.W_out.requires_grad_(True)
        model.transformer.h[b].mlp.b_out.requires_grad_(True)
        trainable_params.extend([model.transformer.h[b].mlp.W_out,
                                  model.transformer.h[b].mlp.b_out])
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_trainable:,} ({n_trainable/1e6:.2f}M)", flush=True)

    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    calib_shuffled = list(calib_ids)
    random.shuffle(calib_shuffled)

    def snapshot():
        return {b: (model.transformer.h[b].mlp.W_out.detach().clone().cpu(),
                    model.transformer.h[b].mlp.b_out.detach().clone().cpu())
                for b in blocks}

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
    print(f"Final at step {N_STEPS}: {final_ppl_at_step_4000:.3f}", flush=True)

    # Restore best
    for b in blocks:
        W_best, b_best = best_state[b]
        model.transformer.h[b].mlp.W_out.data.copy_(W_best.to(DEVICE))
        model.transformer.h[b].mlp.b_out.data.copy_(b_best.to(DEVICE))
    final_ppl_restored = eval_ppl(model, eval_ids)
    print(f"Restored best (step {best_step}): {final_ppl_restored:.3f} (recorded best={best_ppl:.3f})", flush=True)
    assert abs(final_ppl_restored - best_ppl) < 1e-3, \
        f"Restore mismatch: {final_ppl_restored} vs {best_ppl}"

    result = {
        "subset": list(blocks),
        "seed": seed,
        "init_ppl": float(init_ppl),
        "best_ppl": float(best_ppl),
        "best_step": int(best_step),
        "final_ppl_at_step_4000": float(final_ppl_at_step_4000),
        "history": history,
    }
    del model, optimizer, trainable_params, best_state
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return result


# ── Float baseline (once) ─────────────────────────────────────────────────────
print("\nFloat baseline...", flush=True)
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()
float_ppl = eval_ppl(model_float, eval_ids)
print(f"Float PPL: {float_ppl:.3f}", flush=True)
del model_float
if DEVICE == "mps":
    torch.mps.empty_cache()

# ── Phase 1: 1 seed × 3 subsets ───────────────────────────────────────────────
print(f"\n{'='*60}\nPHASE 1 (SEED=42 per subset)\n{'='*60}", flush=True)
phase1 = {}  # subset_key -> result dict
for subset in SUBSETS:
    key = ",".join(str(b) for b in subset)
    r = run_seed(subset, SEEDS_P1[0])
    phase1[key] = r

print(f"\n{'='*60}\nPHASE 1 SUMMARY\n{'='*60}", flush=True)
for subset in SUBSETS:
    key = ",".join(str(b) for b in subset)
    r = phase1[key]
    print(f"  {subset}: init={r['init_ppl']:.3f} best={r['best_ppl']:.3f} "
          f"(step {r['best_step']}) final@4000={r['final_ppl_at_step_4000']:.3f}", flush=True)
print(f"Cycle 221 baseline [1,2,4,5,7,8]: {CYCLE_221_BASELINE}", flush=True)

# ── Phase 2 gating ────────────────────────────────────────────────────────────
phase1_bests = {",".join(str(b) for b in s): phase1[",".join(str(b) for b in s)]["best_ppl"]
                for s in SUBSETS}
passing = [s for s in SUBSETS if phase1_bests[",".join(str(b) for b in s)] < GATE_P2]
phase2 = {}  # subset_key -> list of per-seed results (all seeds in SEEDS_P2)
phase2_subsets = []

if passing:
    # include any subset tied within 0.5 PPL of any passing subset's best
    min_passing = min(phase1_bests[",".join(str(b) for b in s)] for s in passing)
    for s in SUBSETS:
        key = ",".join(str(b) for b in s)
        if phase1_bests[key] <= min_passing + TIE_WIDTH:
            phase2_subsets.append(s)
    print(f"\n{'='*60}\nPHASE 2 GATE PASSED: {len(phase2_subsets)} subset(s) will rerun with seeds {SEEDS_P2}\n{'='*60}", flush=True)
    for subset in phase2_subsets:
        key = ",".join(str(b) for b in subset)
        per_seed = []
        # reuse seed 42 from phase 1
        per_seed.append(phase1[key])
        for seed in SEEDS_P2:
            if seed == 42:
                continue
            per_seed.append(run_seed(subset, seed))
        phase2[key] = per_seed
else:
    print(f"\n{'='*60}\nPHASE 2 GATE NOT PASSED (no subset under {GATE_P2})\n{'='*60}", flush=True)


# ── Aggregate ─────────────────────────────────────────────────────────────────
def agg_best(results_list):
    vals = torch.tensor([r["best_ppl"] for r in results_list], dtype=torch.float64)
    return float(vals.mean()), float(vals.std(unbiased=False))


subset_summaries = []
for subset in SUBSETS:
    key = ",".join(str(b) for b in subset)
    p1 = phase1[key]
    entry = {
        "subset": list(subset),
        "phase1_seed42": p1,
    }
    if key in phase2:
        best_mean, best_std = agg_best(phase2[key])
        entry["phase2_seeds"] = phase2[key]
        entry["best_mean"] = best_mean
        entry["best_std"] = best_std
    else:
        entry["phase2_seeds"] = None
        entry["best_mean"] = p1["best_ppl"]
        entry["best_std"] = 0.0
    subset_summaries.append(entry)


# Verdict
any_confirmed = any(s["best_mean"] <= 58.0 for s in subset_summaries)
all_refuted = all(59.0 <= s["best_mean"] <= 62.0 for s in subset_summaries)
if any_confirmed:
    verdict = "CONFIRMED"
elif all_refuted:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}\nAGGREGATE\n{'='*60}", flush=True)
print(f"Float PPL: {float_ppl:.3f}", flush=True)
print(f"Cycle 221 [1,2,4,5,7,8] baseline: {CYCLE_221_BASELINE}", flush=True)
for s in subset_summaries:
    line = f"  {s['subset']}: best_mean={s['best_mean']:.3f}"
    if s["phase2_seeds"] is not None:
        line += f" ± {s['best_std']:.3f} (3 seeds)"
    else:
        line += " (1 seed)"
    line += f"  Δ vs c221 = {s['best_mean'] - CYCLE_221_BASELINE:+.3f}"
    print(line, flush=True)
print(f"\nVerdict: {verdict}", flush=True)

result = {
    "cycle": 233,
    "item": 243,
    "verdict": verdict,
    "config": {
        "SUBSETS": SUBSETS,
        "K": K, "N_CALIB": N_CALIB, "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN, "LR": LR, "N_STEPS": N_STEPS,
        "EVAL_EVERY": EVAL_EVERY,
        "SEEDS_P1": SEEDS_P1, "SEEDS_P2": SEEDS_P2,
        "GATE_P2": GATE_P2, "TIE_WIDTH": TIE_WIDTH,
    },
    "float_ppl": float(float_ppl),
    "cycle_221_baseline": CYCLE_221_BASELINE,
    "cycle_221_subset": [1, 2, 4, 5, 7, 8],
    "phase1": {k: v for k, v in phase1.items()},
    "phase2": {k: v for k, v in phase2.items()} if phase2 else None,
    "subset_summaries": subset_summaries,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON to {OUTPUT_PATH}", flush=True)

# ── Summary markdown ──────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 233 — Item 243: Alternative 6-block subsets")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Criteria: Confirmed = any subset best_mean ≤ 58.0; Refuted = all three ∈ [59.0, 62.0]; else Inconclusive")
lines.append(f"- Config: K={K}, LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, SEEDS_P1={SEEDS_P1}, SEEDS_P2={SEEDS_P2}")
lines.append(f"- Float PPL: {float_ppl:.3f}")
lines.append(f"- Cycle 221 baseline [1,2,4,5,7,8] best_mean: {CYCLE_221_BASELINE}")
lines.append("")
lines.append("## Per-subset results")
lines.append("")
lines.append("| subset | init (s42) | phase1 best (s42) | phase2 best_mean | Δ vs c221 |")
lines.append("|---|---|---|---|---|")
for s in subset_summaries:
    p1 = s["phase1_seed42"]
    subset_str = str(s["subset"])
    init_s = f"{p1['init_ppl']:.3f}"
    p1_best = f"{p1['best_ppl']:.3f}"
    if s["phase2_seeds"] is not None:
        p2_str = f"{s['best_mean']:.3f} ± {s['best_std']:.3f}"
    else:
        p2_str = "—"
    dlt = f"{s['best_mean'] - CYCLE_221_BASELINE:+.3f}"
    lines.append(f"| {subset_str} | {init_s} | {p1_best} | {p2_str} | {dlt} |")
lines.append(f"| [1,2,4,5,7,8] (c221) | 86.686 | — | {CYCLE_221_BASELINE:.3f} ± 0.053 | 0.000 |")
lines.append("")
lines.append("## Phase 1 trajectories (seed 42)")
lines.append("")
for s in subset_summaries:
    p1 = s["phase1_seed42"]
    hist = p1["history"]
    q = hist[len(hist)//4] if hist else None
    h = hist[len(hist)//2] if hist else None
    tq = hist[3*len(hist)//4] if hist else None
    f_ = hist[-1] if hist else None
    parts = [f"init={p1['init_ppl']:.2f}"]
    if q: parts.append(f"step{q['step']}={q['eval_ppl']:.2f}")
    if h: parts.append(f"step{h['step']}={h['eval_ppl']:.2f}")
    parts.append(f"best=step{p1['best_step']}@{p1['best_ppl']:.2f}")
    if tq: parts.append(f"step{tq['step']}={tq['eval_ppl']:.2f}")
    if f_: parts.append(f"step{f_['step']}={f_['eval_ppl']:.2f}")
    lines.append(f"- **{s['subset']}**: " + ", ".join(parts))
lines.append("")
if phase2:
    lines.append("## Phase 2 per-seed detail")
    lines.append("")
    for subset in SUBSETS:
        key = ",".join(str(b) for b in subset)
        if key not in phase2:
            continue
        lines.append(f"### {subset}")
        lines.append("")
        lines.append("| seed | init | best | best_step | final@4000 |")
        lines.append("|---|---|---|---|---|")
        for r in phase2[key]:
            lines.append(f"| {r['seed']} | {r['init_ppl']:.3f} | {r['best_ppl']:.3f} | {r['best_step']} | {r['final_ppl_at_step_4000']:.3f} |")
        lines.append("")
lines.append("## Interpretation")
lines.append("")
if verdict == "CONFIRMED":
    lines.append(f"At least one alternative subset beats 58.0 PPL — the 59.5 ceiling is block-selection dependent.")
elif verdict == "REFUTED":
    lines.append(f"All three alternative subsets land in [59.0, 62.0] — the 59.5 PPL floor is structural to the 6-block count, not a specific block selection.")
else:
    lines.append(f"Mixed/inconclusive — results span wider than the decision bands.")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}", flush=True)
