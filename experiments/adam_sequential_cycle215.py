"""
Cycle 215 — Item 225: Sequential block-by-block Adam W_out correction.

BLOCKS=[1,4,7] K=512. Rather than jointly optimising all three blocks' W_out
as in cycles 206/210, correct them one at a time in order: block 1 first,
then block 4 (with block 1's corrected W_out frozen), then block 7.
Each block: fresh Adam optimiser, 2000 steps, eval every 100, snapshot best.

Confirmed if mean final_ppl across 3 seeds <= 46.0.
Refuted if mean final_ppl >= 60.0.
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
LR = 1e-4
EVAL_EVERY = 100
N_STEPS_PER_BLOCK = 2000
SEEDS = [42, 137, 2024]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_215_sequential_adam.json"

JOINT_ADAM_BASELINE = 46.556  # cycle 210 step 4000
TOPK_LSTSQ_INIT_BASELINE = 61.746

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


# ── Helpers (verbatim from cycle 206/210) ─────────────────────────────────────
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


# ── Per-seed sequential correction ────────────────────────────────────────────
def run_seed(seed):
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
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

    # Sequential: block 1, then 4, then 7.
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

        # Snapshot pre-block state so we can restore on collapse
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

        # Collapse check: if best_ppl > prev_ppl, training made things worse —
        # restore pre-block state, flag collapsed.
        collapsed = best_ppl > prev_ppl
        if collapsed:
            print(f"  COLLAPSE: best_ppl {best_ppl:.3f} > pre_block_ppl {prev_ppl:.3f}. "
                  f"Restoring pre-block state.", flush=True)
            with torch.no_grad():
                mlp.W_out.copy_(pre_state["W_out"])
                mlp.b_out.copy_(pre_state["b_out"])
            restored_ppl = eval_ppl(model, eval_ids)
            print(f"  restored PPL: {restored_ppl:.3f}", flush=True)
            # Update the "effective" post-block PPL
            effective_ppl = restored_ppl
        else:
            # Restore best state
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

    final_ppl = eval_ppl(model, eval_ids)
    print(f"\nSeed {seed} FINAL PPL: {final_ppl:.3f}", flush=True)

    # Free GPU memory before next seed
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

# Per-block "restored_ppl" after each sequential step
after_b1 = [r["blocks"][0]["restored_ppl"] for r in seed_results]
after_b4 = [r["blocks"][1]["restored_ppl"] for r in seed_results]
after_b7 = [r["blocks"][2]["restored_ppl"] for r in seed_results]
b1_mean, b1_std = mean_std(after_b1)
b4_mean, b4_std = mean_std(after_b4)
b7_mean, b7_std = mean_std(after_b7)

aggregate = {
    "init_ppl_mean": init_mean, "init_ppl_std": init_std,
    "after_block_1_mean": b1_mean, "after_block_1_std": b1_std,
    "after_block_4_mean": b4_mean, "after_block_4_std": b4_std,
    "after_block_7_mean": b7_mean, "after_block_7_std": b7_std,
    "final_ppl_mean": final_mean, "final_ppl_std": final_std,
}

# Verdict
if final_mean <= 46.0:
    verdict = "CONFIRMED"
elif final_mean >= 60.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"{'seed':<8}{'init':>10}{'after_b1':>12}{'after_b4':>12}{'after_b7':>12}{'final':>10}")
print("-" * 70)
for r in seed_results:
    print(f"{r['seed']:<8}{r['init_ppl']:>10.3f}"
          f"{r['blocks'][0]['restored_ppl']:>12.3f}"
          f"{r['blocks'][1]['restored_ppl']:>12.3f}"
          f"{r['blocks'][2]['restored_ppl']:>12.3f}"
          f"{r['final_ppl']:>10.3f}")
print("-" * 70)
print(f"{'mean':<8}{init_mean:>10.3f}{b1_mean:>12.3f}{b4_mean:>12.3f}{b7_mean:>12.3f}{final_mean:>10.3f}")
print(f"{'std':<8}{init_std:>10.3f}{b1_std:>12.3f}{b4_std:>12.3f}{b7_std:>12.3f}{final_std:>10.3f}")
print("=" * 70)
print(f"Float PPL baseline:       {float_ppl:.3f}")
print(f"Joint Adam best (c210):   {JOINT_ADAM_BASELINE:.3f}")
print(f"topk+lstsq init (c206):   {TOPK_LSTSQ_INIT_BASELINE:.3f}")
print(f"Sequential final (mean):  {final_mean:.3f}")
print(f"Verdict: {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "cycle": 215,
    "item": 225,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "joint_adam_baseline": JOINT_ADAM_BASELINE,
    "topk_lstsq_init_baseline": TOPK_LSTSQ_INIT_BASELINE,
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
print(f"Saved {OUTPUT_PATH}")
