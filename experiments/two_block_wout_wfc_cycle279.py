"""
Cycle 279 — Item 289: [1,4] K=512 W_out + W_fc trainable
Does unfreezing W_fc help at the 2-block Pareto winner?

SLICED semantics (NOT full matrix): trainable = W_out (512x768) + b_out (768) +
W_fc_slice (768x512) + b_fc_slice (512) per block. Per block 787,712. Two blocks
= 1,575,424. This matches cycle 213/241/248/262 W_fc retraining family
(FullCompressedMLP trains the topk-sliced (768, K) columns of W_fc, NOT the
full (768, 3072) c_fc matrix). The agenda item 289 setup line claimed
W_fc_l (768x3072) → 5,512,704 total — that REPEATS the cycle 248 agenda error.
Faithful continuation requires the sliced form for apples-to-apples with
cycles 213/241/248/262.

Verdict per item 289 verbatim:
  CONFIRMED (W_fc noop) if best_mean in (43.6, 44.1)
  REFUTED (W_fc helps)  if best_mean <= 43.5
  INCONCLUSIVE          if best_mean in (43.5, 43.6) — marginal — OR best_mean >= 44.1
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

BLOCKS = [1, 4]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 4000
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
OUTPUT_PATH = "autoresearch/results/cycle_279_two_block_wout_wfc.json"
SUMMARY_PATH = "autoresearch/results/cycle_279_summary.md"

# Baselines (PPL)
CYCLE_266_K512_2BLOCK_1_4 = 43.852           # [1,4] W_out-only K=512 (best_mean)
CYCLE_213_K512_3BLOCK_WOUT_WFC = 46.424      # [1,4,7] W_out+W_fc K=512
CYCLE_248_K128_6BLOCK_WOUT_WFC = 67.609      # 6-block K=128 W_out+W_fc
CYCLE_271_K512_1BLOCK_1 = 43.821             # [1] W_out-only K=512 (current Pareto)
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


# ── FullCompressedMLP from cycle 213 (sliced W_fc trainable) ─────────────────
class FullCompressedMLP(nn.Module):
    """Both W_fc (sliced 768×K) and W_out (768×K) trainable; biases trainable too."""
    def __init__(self, fc_weight, fc_bias, W_out, b_out):
        super().__init__()
        self.W_fc = nn.Parameter(fc_weight.clone())   # (768, K) topk slice
        self.b_fc = nn.Parameter(fc_bias.clone())     # (K,) topk slice
        self.W_out = nn.Parameter(W_out.clone())      # (768, K)
        self.b_out = nn.Parameter(b_out.clone())      # (768,)

    def forward(self, x):
        return F.gelu(x @ self.W_fc + self.b_fc) @ self.W_out.T + self.b_out


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
    return FullCompressedMLP(W_fc[:, topk_idx], b_fc[topk_idx], sol[:K].T, sol[K])


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
        mlp.W_fc.requires_grad_(True)
        mlp.b_fc.requires_grad_(True)
        mlp.W_out.requires_grad_(True)
        mlp.b_out.requires_grad_(True)
        trainable_params.extend([mlp.W_fc, mlp.b_fc, mlp.W_out, mlp.b_out])
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_trainable:,} ({n_trainable/1e6:.2f}M)")

    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    calib_shuffled = list(calib_ids)
    random.shuffle(calib_shuffled)

    def snapshot():
        return {b: (model.transformer.h[b].mlp.W_fc.detach().clone().cpu(),
                    model.transformer.h[b].mlp.b_fc.detach().clone().cpu(),
                    model.transformer.h[b].mlp.W_out.detach().clone().cpu(),
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
        W_fc_b, b_fc_b, W_out_b, b_out_b = best_state[b]
        model.transformer.h[b].mlp.W_fc.data.copy_(W_fc_b.to(DEVICE))
        model.transformer.h[b].mlp.b_fc.data.copy_(b_fc_b.to(DEVICE))
        model.transformer.h[b].mlp.W_out.data.copy_(W_out_b.to(DEVICE))
        model.transformer.h[b].mlp.b_out.data.copy_(b_out_b.to(DEVICE))
    final_ppl_restored = eval_ppl(model, eval_ids)
    print(f"Restored best (step {best_step}): {final_ppl_restored:.3f} (recorded best={best_ppl:.3f})")
    assert abs(final_ppl_restored - best_ppl) < 1e-3, \
        f"Restore mismatch: {final_ppl_restored} vs {best_ppl}"

    result = {
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


# ── Sweep: 3 seeds ────────────────────────────────────────────────────────────
print(f"\n{'#'*60}\nSWEEP: BLOCKS={BLOCKS} K={K} | W_out+W_fc SLICED | {len(SEEDS)} seeds\n{'#'*60}")
wall_start = time.time()
seed_results = []
for s in SEEDS:
    r = run_seed(s)
    seed_results.append(r)
    print(f"[fit done, elapsed {(time.time()-wall_start)/60:.1f} min]")


# ── Aggregate ─────────────────────────────────────────────────────────────────
def agg_key(results, key):
    vals = torch.tensor([r[key] for r in results], dtype=torch.float64)
    return float(vals.mean()), float(vals.std(unbiased=False))


init_mean, init_std = agg_key(seed_results, "init_ppl")
best_mean, best_std = agg_key(seed_results, "best_ppl")
best_step_mean, best_step_std = agg_key(seed_results, "best_step")
final_mean, final_std = agg_key(seed_results, "final_ppl_at_step_4000")

# ── Verdict per item 289 ──────────────────────────────────────────────────────
if best_mean <= 43.5:
    verdict = "REFUTED"  # W_fc helps (item 289 verbatim: REFUTED label = W_fc helps)
elif 43.6 <= best_mean <= 44.1:
    verdict = "CONFIRMED"  # W_fc noop
elif 43.5 < best_mean < 43.6:
    verdict = "INCONCLUSIVE"  # marginal
else:
    verdict = "INCONCLUSIVE"  # >= 44.1, regression

print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
print(f"Float PPL:            {float_ppl:.3f}")
print(f"BLOCKS={BLOCKS} K={K} W_out+W_fc (SLICED, n={len(seed_results)}):")
print(f"  init_mean  = {init_mean:.3f} ± {init_std:.3f}")
print(f"  best_mean  = {best_mean:.3f} ± {best_std:.3f}")
print(f"  best_step  = {best_step_mean:.0f} ± {best_step_std:.0f}")
print(f"  final@4000 = {final_mean:.3f} ± {final_std:.3f}")
print(f"\nBaselines:")
print(f"  Cycle 266 [1,4] K=512 W_out-only:        {CYCLE_266_K512_2BLOCK_1_4:.3f}")
print(f"  Cycle 213 [1,4,7] K=512 W_out+W_fc:      {CYCLE_213_K512_3BLOCK_WOUT_WFC:.3f}")
print(f"  Cycle 248 6-block K=128 W_out+W_fc:      {CYCLE_248_K128_6BLOCK_WOUT_WFC:.3f}")
print(f"  Cycle 271 [1] K=512 W_out-only:          {CYCLE_271_K512_1BLOCK_1:.3f}")
print(f"  Float ref:                                {FLOAT_BASELINE_REF:.3f}")
print(f"\nDelta vs cycle 266 [1,4]: {best_mean - CYCLE_266_K512_2BLOCK_1_4:+.3f}")
print(f"Delta vs cycle 213 3b W_out+W_fc: {best_mean - CYCLE_213_K512_3BLOCK_WOUT_WFC:+.3f}")
print(f"Delta vs float: {best_mean - float(float_ppl):+.3f}")
print(f"Verdict: {verdict}")

result = {
    "cycle": 279,
    "item": 289,
    "verdict": verdict,
    "config": {
        "BLOCKS": BLOCKS, "K": K, "N_CALIB": N_CALIB, "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN, "LR": LR, "N_STEPS": N_STEPS,
        "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
    },
    "sliced_semantics_note": (
        "Trainable = W_fc_slice (768, K=512) + b_fc_slice (512) + W_out (512, 768) + "
        "b_out (768) per block. Per-block 787,712. Two-block total 1,575,424. "
        "Cycle 213 FullCompressedMLP semantics — trains the topk-sliced columns of c_fc, "
        "NOT the full (768, 3072) c_fc matrix. Agenda item 289 setup line erroneously claims "
        "W_fc_l (768x3072) → 5,512,704 total params; that repeats the cycle 248 agenda error. "
        "Faithful continuation of cycles 213/241/248/262 W_fc retraining family REQUIRES "
        "the sliced form for apples-to-apples comparison."
    ),
    "float_ppl": float(float_ppl),
    "seed_results": seed_results,
    "aggregate": {
        "n_seeds": len(seed_results),
        "init_mean": init_mean, "init_std": init_std,
        "best_mean": best_mean, "best_std": best_std,
        "best_step_mean": best_step_mean, "best_step_std": best_step_std,
        "final_mean": final_mean, "final_std": final_std,
    },
    "cycle_266_k512_2block_1_4_baseline": CYCLE_266_K512_2BLOCK_1_4,
    "cycle_213_k512_3block_wout_wfc_baseline": CYCLE_213_K512_3BLOCK_WOUT_WFC,
    "cycle_248_k128_6block_wout_wfc_baseline": CYCLE_248_K128_6BLOCK_WOUT_WFC,
    "cycle_271_k512_1block_1_baseline": CYCLE_271_K512_1BLOCK_1,
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "delta_vs_cycle_266": best_mean - CYCLE_266_K512_2BLOCK_1_4,
    "delta_vs_cycle_213": best_mean - CYCLE_213_K512_3BLOCK_WOUT_WFC,
    "delta_vs_float": best_mean - float(float_ppl),
    "wall_time_sec": time.time() - wall_start,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON to {OUTPUT_PATH}")

# ── Summary markdown ──────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 279 — Item 289: [1,4] K=512 W_out + W_fc trainable (SLICED)")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Criteria (item 289 verbatim): CONFIRMED (W_fc noop) if best_mean ∈ (43.6, 44.1); REFUTED (W_fc helps) if best_mean ≤ 43.5; INCONCLUSIVE otherwise.")
lines.append(f"- BLOCKS={BLOCKS}, K={K}, LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}")
lines.append(f"- Float PPL: {float_ppl:.3f}")
lines.append(f"- Wall time: {(time.time()-wall_start)/60:.1f} min; fits={len(seed_results)}")
lines.append("")
lines.append("## SLICED-vs-full semantics note")
lines.append("")
lines.append("**Trainable = W_fc slice (768, K=512) + b_fc slice (512) + W_out (512, 768) + b_out (768) per block.**")
lines.append("Per-block: 787,712. Two-block total: **1,575,424**. ")
lines.append("This is the cycle 213 FullCompressedMLP convention — Adam trains the topk-sliced columns of c_fc, NOT the full (768, 3072) c_fc matrix.")
lines.append("")
lines.append("Item 289's setup line erroneously claimed W_fc_l (768×3072) trainable → 5,512,704 total params — repeats the cycle 248 agenda mistake. Faithful continuation of the cycle 213/241/248/262 W_fc retraining family REQUIRES the sliced form for apples-to-apples comparison.")
lines.append("")
lines.append("## Aggregate")
lines.append("")
lines.append("| metric | value |")
lines.append("|---|---|")
lines.append(f"| init_mean | {init_mean:.3f} ± {init_std:.3f} |")
lines.append(f"| **best_mean** | **{best_mean:.3f} ± {best_std:.3f}** |")
lines.append(f"| best_step_mean | {best_step_mean:.0f} ± {best_step_std:.0f} |")
lines.append(f"| final@4000_mean | {final_mean:.3f} ± {final_std:.3f} |")
lines.append("")
lines.append("## Per-seed details")
lines.append("")
lines.append("| seed | init | best | best_step | final@4000 |")
lines.append("|---|---|---|---|---|")
for r in seed_results:
    lines.append(f"| {r['seed']} | {r['init_ppl']:.3f} | {r['best_ppl']:.3f} | {r['best_step']} | {r['final_ppl_at_step_4000']:.3f} |")
lines.append("")
lines.append("## Comparison")
lines.append("")
lines.append(f"- Cycle 266 [1,4] K=512 W_out-only:        **{CYCLE_266_K512_2BLOCK_1_4:.3f}** (788K)")
lines.append(f"- Cycle 213 [1,4,7] K=512 W_out+W_fc:      {CYCLE_213_K512_3BLOCK_WOUT_WFC:.3f}")
lines.append(f"- Cycle 248 6-block K=128 W_out+W_fc:      {CYCLE_248_K128_6BLOCK_WOUT_WFC:.3f}")
lines.append(f"- Cycle 271 [1] K=512 W_out-only:          {CYCLE_271_K512_1BLOCK_1:.3f} (394K, current Pareto winner)")
lines.append(f"- Float ref:                                {FLOAT_BASELINE_REF:.3f}")
lines.append(f"- Cycle 279 [1,4] K=512 W_out+W_fc SLICED: **{best_mean:.3f}** (1,575,424 trainable)")
lines.append(f"- Delta vs cycle 266 [1,4]:                **{best_mean - CYCLE_266_K512_2BLOCK_1_4:+.3f}** PPL")
lines.append(f"- Delta vs cycle 213 (3b W_out+W_fc):      **{best_mean - CYCLE_213_K512_3BLOCK_WOUT_WFC:+.3f}** PPL")
lines.append(f"- Delta vs float:                           **{best_mean - float(float_ppl):+.3f}** PPL")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
