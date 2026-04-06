"""
Cycle 229 — Item 239: Sequential per-block Adam at 6 blocks.

Apply the cycle 215 sequential per-block recipe to BLOCKS=[1,2,4,5,7,8] K=512.
For each seed, build topk+lstsq init, then sequentially correct each block in
ascending order: freeze everything, unfreeze only that block's W_out and b_out,
fresh Adam, 2000 steps with eval every 100 and best-state restore. Compare to
cycle 221 (6-block joint Adam, best_mean=59.515).

Confirmed if mean final_ppl across 3 seeds <= 58.0.
Refuted   if mean final_ppl >= 60.0.
Inconclusive otherwise.
"""
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS = [1, 2, 4, 5, 7, 8]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
LR = 1e-4
EVAL_EVERY = 100
N_STEPS_PER_BLOCK = 2000
SEEDS = [42, 137, 2024]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_229_sequential_6block.json"
SUMMARY_PATH = "autoresearch/results/cycle_229_summary.md"

# Reference baselines
JOINT_6BLOCK_BASELINE = 59.515       # cycle 221 (best_mean, 6-block joint Adam)
SEQUENTIAL_3BLOCK_BASELINE = 47.253  # cycle 215 (final_ppl_mean, sequential)
JOINT_3BLOCK_BASELINE = 46.556       # cycle 210 (joint 3-block Adam best)

print(f"Device: {DEVICE}")


# ── Data (shared across seeds) ────────────────────────────────────────────────
print("Loading GPT-2 tokenizer + wikitext ...")
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
calib_ids_base = [t.to(DEVICE) for t in all_ids[:N_CALIB]]
eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB:N_CALIB + N_EVAL]]
print(f"Calibration: {len(calib_ids_base)} seqs, Eval: {len(eval_ids)} seqs")


# ── Helpers (verbatim from cycle 215/221) ─────────────────────────────────────
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
        ln_outs[b] = torch.cat(ln_outs[b], dim=0)
        proj_outs[b] = torch.cat(proj_outs[b], dim=0)
    return ln_outs, proj_outs


def fit_block_lstsq(block, ln_out_cpu, proj_out_cpu, K):
    W_fc = block.mlp.c_fc.weight.data.float().cpu()
    b_fc = block.mlp.c_fc.bias.data.float().cpu()
    W_proj = block.mlp.c_proj.weight.data.float().cpu()
    act_all = F.gelu(ln_out_cpu @ W_fc + b_fc)
    scores = act_all.std(0) * W_proj.norm(dim=1)
    topk_idx = scores.topk(K).indices
    W_fc_sub = W_fc[:, topk_idx]
    b_fc_sub = b_fc[topk_idx]
    act_sub = act_all[:, topk_idx]
    ones = torch.ones(act_sub.shape[0], 1)
    sol = torch.linalg.lstsq(torch.cat([act_sub, ones], 1), proj_out_cpu, rcond=None).solution
    return CompressedMLP(W_fc_sub, b_fc_sub, sol[:K].T, sol[K])


def eval_ppl(model, eval_ids):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


# ── Float baseline (once) ─────────────────────────────────────────────────────
print("Float baseline PPL...")
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()
float_ppl = eval_ppl(model_float, eval_ids)
print(f"Float PPL: {float_ppl:.3f}")
del model_float
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Per-seed sequential correction ────────────────────────────────────────────
def run_seed(seed):
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
    torch.manual_seed(seed)
    random.seed(seed)

    # Build topk+lstsq init for all 6 blocks
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    ln_outs, proj_outs = collect_mlp_io(model, calib_ids_base, BLOCKS)
    for b in BLOCKS:
        compressed = fit_block_lstsq(model.transformer.h[b], ln_outs[b], proj_outs[b], K)
        model.transformer.h[b].mlp = compressed.to(DEVICE)
    del ln_outs, proj_outs
    init_ppl = eval_ppl(model, eval_ids)
    print(f"Init PPL: {init_ppl:.3f}", flush=True)

    # Sequential per-block correction
    block_records = []
    prev_ppl = init_ppl
    for block_idx in BLOCKS:
        print(f"\n--- Correcting block {block_idx} ---", flush=True)

        # Freeze everything
        for param in model.parameters():
            param.requires_grad_(False)

        # Unfreeze only this block's W_out and b_out
        mlp = model.transformer.h[block_idx].mlp
        mlp.W_out.requires_grad_(True)
        mlp.b_out.requires_grad_(True)
        trainable = [mlp.W_out, mlp.b_out]
        n_trainable = sum(p.numel() for p in trainable)
        assert n_trainable == 768 * K + 768, f"expected {768*K+768}, got {n_trainable}"
        print(f"Trainable params: {n_trainable:,}", flush=True)

        # Snapshot pre-block state for collapse rollback
        pre_state = {
            "W_out": mlp.W_out.detach().clone(),
            "b_out": mlp.b_out.detach().clone(),
        }

        optimizer = torch.optim.Adam(trainable, lr=LR)
        calib_shuffled = list(calib_ids_base)
        random.shuffle(calib_shuffled)

        best_ppl = float("inf")
        best_step = 0
        best_state = {
            "W_out": mlp.W_out.detach().clone(),
            "b_out": mlp.b_out.detach().clone(),
        }
        history = []

        step = 0
        last_loss = None
        while step < N_STEPS_PER_BLOCK:
            for ids in calib_shuffled:
                if step >= N_STEPS_PER_BLOCK:
                    break
                model.train()
                optimizer.zero_grad()
                out = model(ids, labels=ids)
                out.loss.backward()
                optimizer.step()
                step += 1
                last_loss = out.loss.item()

                if step % EVAL_EVERY == 0:
                    ppl_e = eval_ppl(model, eval_ids)
                    history.append({
                        "step": step,
                        "eval_ppl": float(ppl_e),
                        "calib_loss": float(last_loss),
                    })
                    if ppl_e < best_ppl:
                        best_ppl = float(ppl_e)
                        best_step = step
                        best_state = {
                            "W_out": mlp.W_out.detach().clone(),
                            "b_out": mlp.b_out.detach().clone(),
                        }
                    print(f"  block {block_idx} step {step:4d}: eval_ppl={ppl_e:.3f} "
                          f"loss={last_loss:.4f} best={best_ppl:.3f}@{best_step}", flush=True)

        # Final PPL before restore
        final_ppl_block = eval_ppl(model, eval_ids)

        # Collapse check: if best_ppl > prev_ppl, training made things worse
        collapsed = best_ppl > prev_ppl
        if collapsed:
            print(f"  COLLAPSE: best_ppl {best_ppl:.3f} > pre_block_ppl {prev_ppl:.3f}. "
                  f"Restoring pre-block state.", flush=True)
            with torch.no_grad():
                mlp.W_out.copy_(pre_state["W_out"])
                mlp.b_out.copy_(pre_state["b_out"])
            restored_ppl = eval_ppl(model, eval_ids)
            print(f"  restored PPL: {restored_ppl:.3f}", flush=True)
            effective_ppl = restored_ppl
        else:
            with torch.no_grad():
                mlp.W_out.copy_(best_state["W_out"])
                mlp.b_out.copy_(best_state["b_out"])
            restored_ppl = eval_ppl(model, eval_ids)
            print(f"  restored best state PPL: {restored_ppl:.3f} "
                  f"(recorded best={best_ppl:.3f})", flush=True)
            assert abs(restored_ppl - best_ppl) < 1e-3, \
                f"Restore mismatch: {restored_ppl} vs {best_ppl}"
            effective_ppl = restored_ppl

        block_records.append({
            "block": block_idx,
            "pre_block_ppl": float(prev_ppl),
            "best_ppl": float(best_ppl),
            "best_step": int(best_step),
            "final_ppl_pre_restore": float(final_ppl_block),
            "restored_ppl": float(restored_ppl),
            "collapsed": bool(collapsed),
            "history": history,
        })
        prev_ppl = effective_ppl

    final_ppl = eval_ppl(model, eval_ids)
    print(f"\nSeed {seed} FINAL PPL: {final_ppl:.3f}", flush=True)

    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return {
        "seed": seed,
        "init_ppl": float(init_ppl),
        "blocks": block_records,
        "final_ppl": float(final_ppl),
    }


# ── Run all seeds ─────────────────────────────────────────────────────────────
seed_results = []
for s in SEEDS:
    seed_results.append(run_seed(s))


# ── Aggregate ─────────────────────────────────────────────────────────────────
def mean_std(vals):
    t = torch.tensor(vals, dtype=torch.float64)
    return float(t.mean()), float(t.std(unbiased=False))


init_mean, init_std = mean_std([r["init_ppl"] for r in seed_results])
final_mean, final_std = mean_std([r["final_ppl"] for r in seed_results])

# Per-block restored_ppl after each sequential step (indexed by BLOCKS order)
per_block_restored = {b: [] for b in BLOCKS}
for r in seed_results:
    for rec in r["blocks"]:
        per_block_restored[rec["block"]].append(rec["restored_ppl"])

per_block_agg = {}
for b in BLOCKS:
    m, s = mean_std(per_block_restored[b])
    per_block_agg[f"after_block_{b}_mean"] = m
    per_block_agg[f"after_block_{b}_std"] = s

aggregate = {
    "init_ppl_mean": init_mean, "init_ppl_std": init_std,
    "final_ppl_mean": final_mean, "final_ppl_std": final_std,
    **per_block_agg,
}

# Verdict per item 239
if final_mean <= 58.0:
    verdict = "CONFIRMED"
elif final_mean >= 60.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*94}")
header = (
    f"{'seed':<7}{'init':>9}"
    + "".join(f"{'after_b'+str(b):>11}" for b in BLOCKS)
    + f"{'final':>10}"
)
print(header)
print("-" * 94)
for r in seed_results:
    row = f"{r['seed']:<7}{r['init_ppl']:>9.3f}"
    for rec in r["blocks"]:
        row += f"{rec['restored_ppl']:>11.3f}"
    row += f"{r['final_ppl']:>10.3f}"
    print(row)
print("-" * 94)
mean_row = f"{'mean':<7}{init_mean:>9.3f}"
std_row  = f"{'std':<7}{init_std:>9.3f}"
for b in BLOCKS:
    mean_row += f"{per_block_agg[f'after_block_{b}_mean']:>11.3f}"
    std_row  += f"{per_block_agg[f'after_block_{b}_std']:>11.3f}"
mean_row += f"{final_mean:>10.3f}"
std_row  += f"{final_std:>10.3f}"
print(mean_row)
print(std_row)
print("=" * 94)
print(f"Float PPL baseline:           {float_ppl:.3f}")
print(f"Joint 6-block (cycle 221):    {JOINT_6BLOCK_BASELINE:.3f}")
print(f"Sequential 3-block (c215):    {SEQUENTIAL_3BLOCK_BASELINE:.3f}")
print(f"Joint 3-block (c210):         {JOINT_3BLOCK_BASELINE:.3f}")
print(f"Sequential 6-block (mean):    {final_mean:.3f}")
print(f"Delta vs joint 6-block:       {final_mean - JOINT_6BLOCK_BASELINE:+.3f} PPL")
print(f"Verdict: {verdict}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 229,
    "item": 239,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "joint_6block_baseline_cycle221": JOINT_6BLOCK_BASELINE,
    "sequential_3block_baseline_cycle215": SEQUENTIAL_3BLOCK_BASELINE,
    "joint_3block_baseline_cycle210": JOINT_3BLOCK_BASELINE,
    "seeds": seed_results,
    "aggregate": aggregate,
    "config": {
        "BLOCKS": BLOCKS, "K": K, "N_CALIB": N_CALIB, "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN, "LR": LR, "EVAL_EVERY": EVAL_EVERY,
        "N_STEPS_PER_BLOCK": N_STEPS_PER_BLOCK, "SEEDS": SEEDS,
        "DEVICE": DEVICE,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON to {OUTPUT_PATH}")

# ── Summary markdown ──────────────────────────────────────────────────────────
lines = []
lines.append("# Cycle 229 — Item 239: Sequential 6-block Adam W_out correction")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Criteria: Confirmed `final_mean <= 58.0`; Refuted `final_mean >= 60.0`; Inconclusive otherwise.")
lines.append(f"- Config: BLOCKS={BLOCKS}, K={K}, LR={LR}, N_STEPS_PER_BLOCK={N_STEPS_PER_BLOCK}, EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}")
lines.append(f"- Float PPL: {float_ppl:.3f}")
lines.append("")
lines.append("## Per-seed table")
lines.append("")
header_md = "| seed | init |" + "".join(f" after_b{b} |" for b in BLOCKS) + " final |"
sep_md = "|---|---|" + "".join("---|" for _ in BLOCKS) + "---|"
lines.append(header_md)
lines.append(sep_md)
for r in seed_results:
    row = f"| {r['seed']} | {r['init_ppl']:.3f} |"
    for rec in r["blocks"]:
        row += f" {rec['restored_ppl']:.3f} |"
    row += f" {r['final_ppl']:.3f} |"
    lines.append(row)
mean_md = f"| **mean** | {init_mean:.3f} |"
std_md  = f"| **std**  | {init_std:.3f} |"
for b in BLOCKS:
    mean_md += f" {per_block_agg[f'after_block_{b}_mean']:.3f} |"
    std_md  += f" {per_block_agg[f'after_block_{b}_std']:.3f} |"
mean_md += f" **{final_mean:.3f}** |"
std_md  += f" {final_std:.3f} |"
lines.append(mean_md)
lines.append(std_md)
lines.append("")
lines.append("## Trajectory description")
lines.append("")
for r in seed_results:
    lines.append(f"### Seed {r['seed']}")
    parts = [f"init={r['init_ppl']:.2f}"]
    for rec in r["blocks"]:
        tag = " [collapsed]" if rec["collapsed"] else ""
        parts.append(f"b{rec['block']}={rec['restored_ppl']:.2f}@{rec['best_step']}{tag}")
    parts.append(f"final={r['final_ppl']:.2f}")
    lines.append("- " + ", ".join(parts))
lines.append("")
lines.append("## Comparison")
lines.append("")
lines.append(f"- Sequential 6-block (this cycle): **{final_mean:.3f} ± {final_std:.3f}**")
lines.append(f"- Joint 6-block (cycle 221, best_mean): **{JOINT_6BLOCK_BASELINE:.3f}**  → Δ = **{final_mean - JOINT_6BLOCK_BASELINE:+.3f}** PPL")
lines.append(f"- Sequential 3-block (cycle 215, final_mean): **{SEQUENTIAL_3BLOCK_BASELINE:.3f}**  → Δ = **{final_mean - SEQUENTIAL_3BLOCK_BASELINE:+.3f}** PPL")
lines.append(f"- Joint 3-block (cycle 210, best): **{JOINT_3BLOCK_BASELINE:.3f}**  → Δ = **{final_mean - JOINT_3BLOCK_BASELINE:+.3f}** PPL")
lines.append(f"- Float PPL: **{float_ppl:.3f}**")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
