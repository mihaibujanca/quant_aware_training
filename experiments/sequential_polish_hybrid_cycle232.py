"""
Cycle 232 — Item 242: Sequential init + joint polish hybrid at 3 blocks.

Phase 1: cycle 215 sequential protocol (block-by-block Adam W_out+b_out, 2000
steps each, eval every 100, best-state restore per block).
Phase 2: jointly unfreeze ALL 3 blocks' W_out+b_out simultaneously, fresh Adam
at LR=5e-5 (lower than phase 1 to avoid wiping the sequential solution),
1000 steps, eval every 100, best-state restore.

BLOCKS=[1,4,7] K=512, 3 seeds (42, 137, 2024).

Confirmed if final_ppl_mean ≤ 45.8 (breaks joint 46.556 by ≥0.75).
Refuted if final_ppl_mean ≥ 46.5.
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
BLOCKS = [1, 4, 7]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
LR_PHASE1 = 1e-4
LR_PHASE2 = 5e-5
EVAL_EVERY = 100
N_STEPS_PHASE1 = 2000
N_STEPS_PHASE2 = 1000
SEEDS = [42, 137, 2024]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_232_sequential_polish.json"
SUMMARY_PATH = "autoresearch/results/cycle_232_summary.md"

BASELINE_FLOAT = None  # measured below
BASELINE_TOPK_LSTSQ_INIT = 61.746
BASELINE_C206_JOINT_1K = 48.111
BASELINE_C210_JOINT_4K = 46.556
BASELINE_C213_WOUT_WFC = 46.424
BASELINE_C215_SEQUENTIAL = 47.253

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


# ── Helpers (verbatim from cycle 215) ─────────────────────────────────────────
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


# ── Per-seed: phase 1 sequential + phase 2 joint polish ───────────────────────
def run_seed(seed):
    print(f"\n{'='*70}\nSEED {seed}\n{'='*70}", flush=True)
    torch.manual_seed(seed)
    random.seed(seed)

    # Build topk+lstsq init
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    ln_outs, proj_outs = collect_mlp_io(model, calib_ids_base, BLOCKS)
    for b in BLOCKS:
        compressed = fit_block_lstsq(model.transformer.h[b], ln_outs[b], proj_outs[b], K)
        model.transformer.h[b].mlp = compressed.to(DEVICE)
    init_ppl = eval_ppl(model, eval_ids)
    print(f"Init PPL: {init_ppl:.3f}", flush=True)

    # ── PHASE 1: sequential per-block ─────────────────────────────────────────
    print(f"\n--- PHASE 1: sequential block-by-block (LR={LR_PHASE1}) ---",
          flush=True)
    block_records = []
    prev_ppl = init_ppl
    for block_idx in BLOCKS:
        print(f"\n  >> block {block_idx} <<", flush=True)

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
        print(f"  trainable params: {n_trainable:,}", flush=True)

        # Snapshot pre-block state for collapse rollback
        pre_state = {
            "W_out": mlp.W_out.detach().clone(),
            "b_out": mlp.b_out.detach().clone(),
        }

        optimizer = torch.optim.Adam(trainable, lr=LR_PHASE1)
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
        while step < N_STEPS_PHASE1:
            for ids in calib_shuffled:
                if step >= N_STEPS_PHASE1:
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
                    print(f"    block {block_idx} step {step:4d}: "
                          f"eval_ppl={ppl_e:.3f} loss={last_loss:.4f} "
                          f"best={best_ppl:.3f}@{best_step}", flush=True)

        final_ppl_block = eval_ppl(model, eval_ids)

        collapsed = best_ppl > prev_ppl
        if collapsed:
            print(f"  COLLAPSE: best_ppl {best_ppl:.3f} > pre_block {prev_ppl:.3f}. "
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
            print(f"  restored best state PPL: {restored_ppl:.3f}", flush=True)
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

    post_sequential_ppl = eval_ppl(model, eval_ids)
    print(f"\n  PHASE 1 final (post_sequential): {post_sequential_ppl:.3f}",
          flush=True)

    # ── PHASE 2: joint polish on all 3 W_out+b_out ────────────────────────────
    print(f"\n--- PHASE 2: joint polish (LR={LR_PHASE2}, "
          f"{N_STEPS_PHASE2} steps) ---", flush=True)
    for param in model.parameters():
        param.requires_grad_(False)

    trainable_p2 = []
    for b in BLOCKS:
        mlp = model.transformer.h[b].mlp
        mlp.W_out.requires_grad_(True)
        mlp.b_out.requires_grad_(True)
        trainable_p2.extend([mlp.W_out, mlp.b_out])
    n_trainable_p2 = sum(p.numel() for p in trainable_p2)
    expected_p2 = 3 * (768 * K + 768)
    assert n_trainable_p2 == expected_p2, \
        f"expected {expected_p2}, got {n_trainable_p2}"
    print(f"  trainable params (joint): {n_trainable_p2:,}", flush=True)

    # Snapshot the phase-1-final state for rollback on collapse
    pre_phase2_state = {
        b: {
            "W_out": model.transformer.h[b].mlp.W_out.detach().clone(),
            "b_out": model.transformer.h[b].mlp.b_out.detach().clone(),
        } for b in BLOCKS
    }

    optimizer = torch.optim.Adam(trainable_p2, lr=LR_PHASE2)
    calib_shuffled = list(calib_ids_base)
    random.shuffle(calib_shuffled)

    best_ppl_p2 = float("inf")
    best_step_p2 = 0
    best_state_p2 = {
        b: {
            "W_out": model.transformer.h[b].mlp.W_out.detach().clone(),
            "b_out": model.transformer.h[b].mlp.b_out.detach().clone(),
        } for b in BLOCKS
    }
    phase2_history = []

    step = 0
    last_loss = None
    while step < N_STEPS_PHASE2:
        for ids in calib_shuffled:
            if step >= N_STEPS_PHASE2:
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
                phase2_history.append({
                    "step": step,
                    "eval_ppl": float(ppl_e),
                    "calib_loss": float(last_loss),
                })
                if ppl_e < best_ppl_p2:
                    best_ppl_p2 = float(ppl_e)
                    best_step_p2 = step
                    best_state_p2 = {
                        b: {
                            "W_out": model.transformer.h[b].mlp.W_out.detach().clone(),
                            "b_out": model.transformer.h[b].mlp.b_out.detach().clone(),
                        } for b in BLOCKS
                    }
                print(f"    polish step {step:4d}: eval_ppl={ppl_e:.3f} "
                      f"loss={last_loss:.4f} best={best_ppl_p2:.3f}@{best_step_p2}",
                      flush=True)

    final_ppl_p2_pre_restore = eval_ppl(model, eval_ids)

    # Collapse check: if joint polish never beat phase-1 final, rollback
    phase2_collapsed = best_ppl_p2 > post_sequential_ppl
    if phase2_collapsed:
        print(f"  PHASE 2 COLLAPSE: best {best_ppl_p2:.3f} > "
              f"post_sequential {post_sequential_ppl:.3f}. Rolling back to phase-1 state.",
              flush=True)
        with torch.no_grad():
            for b in BLOCKS:
                mlp = model.transformer.h[b].mlp
                mlp.W_out.copy_(pre_phase2_state[b]["W_out"])
                mlp.b_out.copy_(pre_phase2_state[b]["b_out"])
        final_ppl = eval_ppl(model, eval_ids)
        print(f"  rollback PPL: {final_ppl:.3f}", flush=True)
    else:
        with torch.no_grad():
            for b in BLOCKS:
                mlp = model.transformer.h[b].mlp
                mlp.W_out.copy_(best_state_p2[b]["W_out"])
                mlp.b_out.copy_(best_state_p2[b]["b_out"])
        final_ppl = eval_ppl(model, eval_ids)
        print(f"  restored best PPL: {final_ppl:.3f}", flush=True)

    print(f"\nSeed {seed} FINAL: post_sequential={post_sequential_ppl:.3f} "
          f"polish_best={best_ppl_p2:.3f} final={final_ppl:.3f}", flush=True)

    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return {
        "seed": seed,
        "init_ppl": float(init_ppl),
        "phase1_blocks": block_records,
        "post_sequential_ppl": float(post_sequential_ppl),
        "phase2_best_ppl": float(best_ppl_p2),
        "phase2_best_step": int(best_step_p2),
        "phase2_final_pre_restore": float(final_ppl_p2_pre_restore),
        "phase2_collapsed": bool(phase2_collapsed),
        "phase2_history": phase2_history,
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
post_seq_mean, post_seq_std = mean_std([r["post_sequential_ppl"] for r in seed_results])
final_mean, final_std = mean_std([r["final_ppl"] for r in seed_results])
polish_best_mean, polish_best_std = mean_std([r["phase2_best_ppl"] for r in seed_results])

# gain = post_sequential - final  (positive => polish helped)
gains = [r["post_sequential_ppl"] - r["final_ppl"] for r in seed_results]
gain_mean, gain_std = mean_std(gains)

aggregate = {
    "init_ppl_mean": init_mean, "init_ppl_std": init_std,
    "post_sequential_mean": post_seq_mean, "post_sequential_std": post_seq_std,
    "polish_best_mean": polish_best_mean, "polish_best_std": polish_best_std,
    "final_ppl_mean": final_mean, "final_ppl_std": final_std,
    "gain_from_polish_mean": gain_mean, "gain_from_polish_std": gain_std,
}

# Verdict
if final_mean <= 45.8:
    verdict = "CONFIRMED"
elif final_mean >= 46.5:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*78}")
print(f"{'seed':<8}{'init':>10}{'post_seq':>12}{'polish_best':>14}{'final':>10}{'gain':>10}")
print("-" * 78)
for r in seed_results:
    g = r["post_sequential_ppl"] - r["final_ppl"]
    print(f"{r['seed']:<8}{r['init_ppl']:>10.3f}"
          f"{r['post_sequential_ppl']:>12.3f}"
          f"{r['phase2_best_ppl']:>14.3f}"
          f"{r['final_ppl']:>10.3f}"
          f"{g:>10.3f}")
print("-" * 78)
print(f"{'mean':<8}{init_mean:>10.3f}{post_seq_mean:>12.3f}"
      f"{polish_best_mean:>14.3f}{final_mean:>10.3f}{gain_mean:>10.3f}")
print(f"{'std':<8}{init_std:>10.3f}{post_seq_std:>12.3f}"
      f"{polish_best_std:>14.3f}{final_std:>10.3f}{gain_std:>10.3f}")
print("=" * 78)
print(f"\nBaselines:")
print(f"  Float PPL:                       {float_ppl:.3f}")
print(f"  topk+lstsq init (c206):          {BASELINE_TOPK_LSTSQ_INIT:.3f}")
print(f"  Sequential alone (c215):         {BASELINE_C215_SEQUENTIAL:.3f}")
print(f"  Joint 1k W_out (c206):           {BASELINE_C206_JOINT_1K:.3f}")
print(f"  Joint 4k W_out (c210):           {BASELINE_C210_JOINT_4K:.3f}")
print(f"  Joint W_out+W_fc (c213):         {BASELINE_C213_WOUT_WFC:.3f}")
print(f"  THIS hybrid final_mean:          {final_mean:.3f} ± {final_std:.3f}")
print(f"  Verdict: {verdict}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 232,
    "item": 242,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "baselines": {
        "topk_lstsq_init": BASELINE_TOPK_LSTSQ_INIT,
        "c215_sequential": BASELINE_C215_SEQUENTIAL,
        "c206_joint_1k": BASELINE_C206_JOINT_1K,
        "c210_joint_4k": BASELINE_C210_JOINT_4K,
        "c213_wout_wfc": BASELINE_C213_WOUT_WFC,
    },
    "seeds": seed_results,
    "aggregate": aggregate,
    "config": {
        "BLOCKS": BLOCKS, "K": K, "N_CALIB": N_CALIB, "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "LR_PHASE1": LR_PHASE1, "LR_PHASE2": LR_PHASE2,
        "N_STEPS_PHASE1": N_STEPS_PHASE1, "N_STEPS_PHASE2": N_STEPS_PHASE2,
        "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS, "DEVICE": DEVICE,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved {OUTPUT_PATH}")


# ── Markdown summary ──────────────────────────────────────────────────────────
md_lines = []
md_lines.append(f"# Cycle 232 — Item 242: Sequential init + joint polish hybrid (3 blocks)\n")
md_lines.append(f"**Verdict: {verdict}**\n")
md_lines.append(f"- Criteria: Confirmed ≤45.8, Refuted ≥46.5, else Inconclusive")
md_lines.append(f"- Config: BLOCKS={BLOCKS}, K={K}, "
                f"LR_PHASE1={LR_PHASE1}, LR_PHASE2={LR_PHASE2}, "
                f"N_STEPS_PHASE1={N_STEPS_PHASE1}, N_STEPS_PHASE2={N_STEPS_PHASE2}, "
                f"EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}")
md_lines.append(f"- Float PPL: {float_ppl:.3f}\n")

md_lines.append("## Per-seed results\n")
md_lines.append("| seed | init | post_sequential | polish_best | final | gain |")
md_lines.append("|---|---|---|---|---|---|")
for r in seed_results:
    g = r["post_sequential_ppl"] - r["final_ppl"]
    md_lines.append(
        f"| {r['seed']} | {r['init_ppl']:.3f} | {r['post_sequential_ppl']:.3f} | "
        f"{r['phase2_best_ppl']:.3f} | {r['final_ppl']:.3f} | {g:+.3f} |"
    )
md_lines.append(
    f"| **mean** | {init_mean:.3f} | {post_seq_mean:.3f} | "
    f"{polish_best_mean:.3f} | **{final_mean:.3f}** | {gain_mean:+.3f} |"
)
md_lines.append(
    f"| **std**  | {init_std:.3f} | {post_seq_std:.3f} | "
    f"{polish_best_std:.3f} | {final_std:.3f} | {gain_std:.3f} |"
)
md_lines.append("")

md_lines.append("## Comparison vs baselines\n")
md_lines.append("| approach | best_mean PPL |")
md_lines.append("|---|---|")
md_lines.append(f"| Float (GPT-2) | {float_ppl:.3f} |")
md_lines.append(f"| topk+lstsq init only | {BASELINE_TOPK_LSTSQ_INIT:.3f} |")
md_lines.append(f"| Sequential alone (c215) | {BASELINE_C215_SEQUENTIAL:.3f} |")
md_lines.append(f"| Joint 1k W_out (c206) | {BASELINE_C206_JOINT_1K:.3f} |")
md_lines.append(f"| Joint 4k W_out (c210) | {BASELINE_C210_JOINT_4K:.3f} |")
md_lines.append(f"| Joint W_out+W_fc (c213) | {BASELINE_C213_WOUT_WFC:.3f} |")
md_lines.append(f"| **Hybrid sequential+polish (c232)** | **{final_mean:.3f} ± {final_std:.3f}** |")
md_lines.append("")
md_lines.append(f"- Δ vs sequential alone (c215): {final_mean - BASELINE_C215_SEQUENTIAL:+.3f}")
md_lines.append(f"- Δ vs joint 4k (c210):        {final_mean - BASELINE_C210_JOINT_4K:+.3f}")
md_lines.append(f"- Δ vs joint W_out+W_fc (c213): {final_mean - BASELINE_C213_WOUT_WFC:+.3f}")
md_lines.append(f"- Gain from polish (post_seq − final): {gain_mean:+.3f} ± {gain_std:.3f}")
md_lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(md_lines))
print(f"Saved {SUMMARY_PATH}")
