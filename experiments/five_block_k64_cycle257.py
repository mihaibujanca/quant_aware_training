"""
Cycle 257 — Item 267: 5-block [1,2,4,7,8] at K=64.
Extend the K-vs-PPL curve at 5 blocks below K=128. Cycle 245 K=512 was
near-float (53.826); cycle 251 K=128 landed at 59.119 (cycle 254 confirmed
~59 floor at K=128 with 8k steps). Where does K=64 sit?

Confirmed if: best_ppl_mean <= 64.0 (5-block K=64 robust, K-curve gradual).
Refuted    if: best_ppl_mean >= 75.0 (channel-binding below K=128 at 5 blocks).
Inconclusive: (64.0, 75.0).
"""
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

BLOCKS = [1, 2, 4, 7, 8]
K = 64
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 4000
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
OUTPUT_PATH = "autoresearch/results/cycle_257_five_block_k64.json"
SUMMARY_PATH = "autoresearch/results/cycle_257_summary.md"
CYCLE_245_K512_BASELINE = 53.826  # 5-block [1,2,4,7,8] K=512 mean
CYCLE_251_K128_BASELINE = 59.119  # 5-block [1,2,4,7,8] K=128 mean
CYCLE_230_K64_3B_BASELINE = 50.781  # 3-block K=64 (item 267 context)
FLOAT_PPL_BASELINE = 53.895

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

    # Freeze all, unfreeze W_out/b_out for BLOCKS
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

    # Best state snapshot (CPU)
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

    # Restore best
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
        "final_ppl_at_step_4000": float(final_ppl_at_step_4000),
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

# ── Run all seeds ─────────────────────────────────────────────────────────────
seed_results = [run_seed(s) for s in SEEDS]

# ── Aggregate ─────────────────────────────────────────────────────────────────
def agg(key):
    vals = torch.tensor([r[key] for r in seed_results], dtype=torch.float64)
    return float(vals.mean()), float(vals.std(unbiased=False))

init_mean, init_std = agg("init_ppl")
best_mean, best_std = agg("best_ppl")
best_step_mean, best_step_std = agg("best_step")
final_mean, final_std = agg("final_ppl_at_step_4000")

# Verdict per item 267 criteria
if best_mean <= 64.0:
    verdict = "CONFIRMED"
elif best_mean >= 75.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
print(f"Float PPL:              {float_ppl:.3f}")
print(f"Init PPL (mean±std):    {init_mean:.3f} ± {init_std:.3f}")
print(f"Best PPL (mean±std):    {best_mean:.3f} ± {best_std:.3f}")
print(f"Best step (mean±std):   {best_step_mean:.0f} ± {best_step_std:.0f}")
print(f"Final@4000 (mean±std):  {final_mean:.3f} ± {final_std:.3f}")
print(f"Cycle 245 K=512 5-block baseline: {CYCLE_245_K512_BASELINE}")
print(f"Cycle 251 K=128 5-block baseline: {CYCLE_251_K128_BASELINE}")
print(f"Cycle 230 K=64 3-block baseline:  {CYCLE_230_K64_3B_BASELINE}")
print(f"Float baseline (cycle 245):       {FLOAT_PPL_BASELINE}")
print(f"Delta (best_mean - cycle 245 K=512): {best_mean - CYCLE_245_K512_BASELINE:+.3f}")
print(f"Delta (best_mean - cycle 251 K=128): {best_mean - CYCLE_251_K128_BASELINE:+.3f}")
print(f"Delta (best_mean - cycle 230 K=64 3b): {best_mean - CYCLE_230_K64_3B_BASELINE:+.3f}")
print(f"Delta (best_mean - float):             {best_mean - FLOAT_PPL_BASELINE:+.3f}")
print(f"Verdict: {verdict}")

result = {
    "cycle": 257,
    "item": 267,
    "verdict": verdict,
    "config": {
        "BLOCKS": BLOCKS, "K": K, "N_CALIB": N_CALIB, "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN, "LR": LR, "N_STEPS": N_STEPS,
        "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
    },
    "float_ppl": float(float_ppl),
    "seeds": seed_results,
    "aggregate": {
        "init_mean": init_mean, "init_std": init_std,
        "best_mean": best_mean, "best_std": best_std,
        "best_step_mean": best_step_mean, "best_step_std": best_step_std,
        "final_mean": final_mean, "final_std": final_std,
    },
    "cycle_245_k512_5block_baseline": CYCLE_245_K512_BASELINE,
    "cycle_251_k128_5block_baseline": CYCLE_251_K128_BASELINE,
    "cycle_230_k64_3block_baseline": CYCLE_230_K64_3B_BASELINE,
    "float_ppl_baseline": FLOAT_PPL_BASELINE,
    "delta_vs_cycle_245_k512": best_mean - CYCLE_245_K512_BASELINE,
    "delta_vs_cycle_251_k128": best_mean - CYCLE_251_K128_BASELINE,
    "delta_vs_cycle_230_k64_3block": best_mean - CYCLE_230_K64_3B_BASELINE,
    "delta_vs_float": best_mean - FLOAT_PPL_BASELINE,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON to {OUTPUT_PATH}")

# ── Summary markdown ──────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 257 — Item 267: 5-block [1,2,4,7,8] at K=64")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Criteria: Confirmed ≤64.0, Refuted ≥75.0, else Inconclusive")
lines.append(f"- Config: BLOCKS={BLOCKS}, K={K}, LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}")
lines.append(f"- Float PPL: {float_ppl:.3f}")
lines.append("")
lines.append("## Per-seed results")
lines.append("")
lines.append("| seed | init | best | best_step | final@4000 |")
lines.append("|---|---|---|---|---|")
for r in seed_results:
    lines.append(f"| {r['seed']} | {r['init_ppl']:.3f} | {r['best_ppl']:.3f} | {r['best_step']} | {r['final_ppl_at_step_4000']:.3f} |")
lines.append(f"| **mean** | {init_mean:.3f} | **{best_mean:.3f}** | {best_step_mean:.0f} | {final_mean:.3f} |")
lines.append(f"| **std**  | {init_std:.3f} | {best_std:.3f} | {best_step_std:.0f} | {final_std:.3f} |")
lines.append("")
lines.append("## Trajectory description")
lines.append("")
for r in seed_results:
    lines.append(f"### Seed {r['seed']}")
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
    lines.append("- " + ", ".join(parts))
lines.append("")
lines.append("## Comparison")
lines.append("")
lines.append(f"- Cycle 257 5-block [1,2,4,7,8] K=64 best_mean: **{best_mean:.3f} ± {best_std:.3f}**")
lines.append(f"- Cycle 245 5-block [1,2,4,7,8] K=512: **{CYCLE_245_K512_BASELINE:.3f}**")
lines.append(f"- Delta vs cycle 245 K=512: **{best_mean - CYCLE_245_K512_BASELINE:+.3f}** PPL")
lines.append(f"- Cycle 251 5-block [1,2,4,7,8] K=128: {CYCLE_251_K128_BASELINE:.3f}")
lines.append(f"- Delta vs cycle 251 K=128: **{best_mean - CYCLE_251_K128_BASELINE:+.3f}** PPL")
lines.append(f"- Cycle 230 3-block K=64: {CYCLE_230_K64_3B_BASELINE:.3f}")
lines.append(f"- Delta vs cycle 230 K=64 3-block: **{best_mean - CYCLE_230_K64_3B_BASELINE:+.3f}** PPL")
lines.append(f"- Float PPL baseline: {FLOAT_PPL_BASELINE:.3f}")
lines.append(f"- Delta vs float: **{best_mean - FLOAT_PPL_BASELINE:+.3f}** PPL")
lines.append(f"- Overfitting check: final@4000 mean = {final_mean:.3f}")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
