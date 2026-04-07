"""
Cycle 241 — Item 251: Adam W_out + W_fc bias 6-block joint correction.

BLOCKS=[1,2,4,5,7,8] K=512. Trainable surface: W_out + b_out + FULL (3072,) c_fc
bias per compressed block. W_fc matrix stays FROZEN (register_buffer). The c_fc
bias is kept at the full (3072,) size (not the (K,)=512 topk subset) to match
the agenda's 2.38M trainable-budget arithmetic exactly — the forward indexes
b_fc[topk_idx] so the 2560 non-selected bias entries receive zero gradient but
are counted.

Per-block trainable = 512*768 (W_out) + 768 (b_out) + 3072 (b_fc full) = 397,056
6 blocks × 397,056 = 2,382,336 ≈ 2.38M (matches agenda "≈ 2.38M").

Confirmed if: best_ppl_mean <= 58.0
Refuted if: best_ppl_mean in [59.0, 60.5]
Inconclusive otherwise.

Baselines:
- Cycle 221 W_out-only 6-block: 59.515 ± 0.094
- Cycle 213 W_out+W_fc 3-block: 46.424 (the agenda-reported best_mean with W_fc bias)
"""
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

BLOCKS = [1, 2, 4, 5, 7, 8]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 4000
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
OUTPUT_PATH = "autoresearch/results/cycle_241_wout_wfc_6block.json"
SUMMARY_PATH = "autoresearch/results/cycle_241_summary.md"
CYCLE_221_BASELINE = 59.515
CYCLE_221_STD = 0.094
CYCLE_213_3B_BASELINE = 46.424
EXPECTED_TRAINABLE = 6 * (K * 768 + 768 + 3072)  # 2,382,336

print(f"Device: {DEVICE}")
print(f"Expected trainable params: {EXPECTED_TRAINABLE:,}")

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


class CompressedMLPWfcBias(nn.Module):
    """W_fc frozen (buffer); FULL c_fc bias (3072,) trainable; W_out, b_out trainable.

    Forward uses b_fc[topk_idx] so only K entries affect the output.
    Non-selected entries get zero gradient but still count toward trainable budget.
    """
    def __init__(self, fc_weight_sub, fc_bias_full, topk_idx, W_out, b_out):
        super().__init__()
        # Frozen buffers — W_fc submatrix (post-topk) and the topk index vector.
        self.register_buffer("fc_weight", fc_weight_sub)     # (768, K)
        self.register_buffer("topk_idx", topk_idx)            # (K,) long
        # FULL pre-topk c_fc bias as trainable Parameter
        self.b_fc_full = nn.Parameter(fc_bias_full.clone())  # (3072,)
        self.W_out = nn.Parameter(W_out.clone())              # (768, K)
        self.b_out = nn.Parameter(b_out.clone())              # (768,)

    def forward(self, x):
        # Select the K bias entries that match the topk columns of W_fc
        b_fc_sub = self.b_fc_full[self.topk_idx]              # (K,)
        return F.gelu(x @ self.fc_weight + b_fc_sub) @ self.W_out.T + self.b_out


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
    W_fc = block.mlp.c_fc.weight.data.float().cpu()            # (768, 3072)
    b_fc = block.mlp.c_fc.bias.data.float().cpu()              # (3072,)
    W_proj = block.mlp.c_proj.weight.data.float().cpu()        # (3072, 768)
    act_all = F.gelu(ln_out_cpu @ W_fc + b_fc)                  # (N, 3072)
    scores = act_all.std(0) * W_proj.norm(dim=1)
    topk_idx = scores.topk(K).indices                            # (K,)
    act_sub = act_all[:, topk_idx]                               # (N, K)
    ones = torch.ones(act_sub.shape[0], 1)
    sol = torch.linalg.lstsq(torch.cat([act_sub, ones], 1), proj_out_cpu, rcond=None).solution
    # Return CompressedMLPWfcBias with FULL b_fc (3072,) as trainable, W_fc submatrix as buffer
    return CompressedMLPWfcBias(
        fc_weight_sub=W_fc[:, topk_idx],      # (768, K) frozen
        fc_bias_full=b_fc,                     # (3072,) trainable
        topk_idx=topk_idx.long(),              # (K,) buffer
        W_out=sol[:K].T,                       # (768, K) trainable
        b_out=sol[K],                          # (768,)
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

    # Freeze everything, then unfreeze only {b_fc_full, W_out, b_out} on BLOCKS
    for param in model.parameters():
        param.requires_grad_(False)
    trainable_params = []
    for b in BLOCKS:
        mlp = model.transformer.h[b].mlp
        # W_fc matrix must stay frozen — it is a buffer so requires_grad is
        # N/A, but assert it anyway via the buffer check.
        assert not mlp.fc_weight.requires_grad, f"block {b} fc_weight leaked grad"
        mlp.b_fc_full.requires_grad_(True)
        mlp.W_out.requires_grad_(True)
        mlp.b_out.requires_grad_(True)
        trainable_params.extend([mlp.b_fc_full, mlp.W_out, mlp.b_out])
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_trainable:,} ({n_trainable/1e6:.3f}M)")
    assert n_trainable == EXPECTED_TRAINABLE, \
        f"trainable count mismatch: got {n_trainable} expected {EXPECTED_TRAINABLE}"

    # Assert W_fc buffers (compressed blocks) are not in the optimizer
    # and confirm untouched blocks' c_fc weights haven't been unfrozen
    for b in range(12):
        if b in BLOCKS:
            # compressed: fc_weight is a buffer, not a Parameter
            assert not isinstance(model.transformer.h[b].mlp.fc_weight, nn.Parameter)
        else:
            # untouched: original c_fc.weight must be frozen
            assert not model.transformer.h[b].mlp.c_fc.weight.requires_grad

    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    calib_shuffled = list(calib_ids)
    random.shuffle(calib_shuffled)

    def snapshot():
        return {b: (model.transformer.h[b].mlp.b_fc_full.detach().clone().cpu(),
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

    # Restore best
    for b in BLOCKS:
        bfc_best, W_best, b_best = best_state[b]
        model.transformer.h[b].mlp.b_fc_full.data.copy_(bfc_best.to(DEVICE))
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


# Float baseline (once)
print("\nFloat baseline...")
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()
float_ppl = eval_ppl(model_float, eval_ids)
print(f"Float PPL: {float_ppl:.3f}")
del model_float
if DEVICE == "mps":
    torch.mps.empty_cache()

seed_results = [run_seed(s) for s in SEEDS]


def agg(key):
    vals = torch.tensor([r[key] for r in seed_results], dtype=torch.float64)
    return float(vals.mean()), float(vals.std(unbiased=False))


init_mean, init_std = agg("init_ppl")
best_mean, best_std = agg("best_ppl")
best_step_mean, best_step_std = agg("best_step")
final_mean, final_std = agg("final_ppl_at_step_4000")

if best_mean <= 58.0:
    verdict = "CONFIRMED"
elif 59.0 <= best_mean <= 60.5:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
print(f"Float PPL:              {float_ppl:.3f}")
print(f"Init PPL (mean±std):    {init_mean:.3f} ± {init_std:.3f}")
print(f"Best PPL (mean±std):    {best_mean:.3f} ± {best_std:.3f}")
print(f"Best step (mean±std):   {best_step_mean:.0f} ± {best_step_std:.0f}")
print(f"Final@4000 (mean±std):  {final_mean:.3f} ± {final_std:.3f}")
print(f"Cycle 221 W_out-only 6b baseline: {CYCLE_221_BASELINE} ± {CYCLE_221_STD}")
print(f"Cycle 213 W_out+W_fc-bias 3b baseline: {CYCLE_213_3B_BASELINE}")
print(f"Delta vs cycle 221 (best_mean - 59.515): {best_mean - CYCLE_221_BASELINE:+.3f}")
print(f"Verdict: {verdict}")

result = {
    "cycle": 241,
    "item": 251,
    "verdict": verdict,
    "config": {
        "BLOCKS": BLOCKS, "K": K, "N_CALIB": N_CALIB, "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN, "LR": LR, "N_STEPS": N_STEPS,
        "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
        "trainable_params_expected": EXPECTED_TRAINABLE,
        "trainable_surface": "W_out + b_out + c_fc_full_bias (W_fc matrix frozen)",
    },
    "float_ppl": float(float_ppl),
    "seeds": seed_results,
    "aggregate": {
        "init_mean": init_mean, "init_std": init_std,
        "best_mean": best_mean, "best_std": best_std,
        "best_step_mean": best_step_mean, "best_step_std": best_step_std,
        "final_mean": final_mean, "final_std": final_std,
    },
    "cycle_221_wout_only_6block": CYCLE_221_BASELINE,
    "cycle_221_wout_only_6block_std": CYCLE_221_STD,
    "cycle_213_wout_wfc_bias_3block": CYCLE_213_3B_BASELINE,
    "delta_vs_cycle_221": float(best_mean - CYCLE_221_BASELINE),
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON to {OUTPUT_PATH}")

# Summary markdown
lines = []
lines.append(f"# Cycle 241 — Item 251: Adam W_out+W_fc-bias 6-block")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Criteria: Confirmed ≤58.0, Refuted ∈[59.0,60.5], else Inconclusive")
lines.append(f"- Config: BLOCKS={BLOCKS}, K={K}, LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}")
lines.append(f"- Trainable surface: W_out + b_out + full c_fc bias (3072,) — W_fc matrix FROZEN")
lines.append(f"- Trainable params: {EXPECTED_TRAINABLE:,} ({EXPECTED_TRAINABLE/1e6:.3f}M)")
lines.append(f"- Float PPL (held-out): {float_ppl:.3f}")
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
lines.append(f"- Cycle 221 W_out-only 6-block baseline: **{CYCLE_221_BASELINE:.3f} ± {CYCLE_221_STD:.3f}**")
lines.append(f"- Cycle 241 W_out+W_fc-bias 6-block best_mean: **{best_mean:.3f} ± {best_std:.3f}**")
lines.append(f"- Delta vs cycle 221: **{best_mean - CYCLE_221_BASELINE:+.3f}** PPL")
lines.append(f"- Cycle 213 W_out+W_fc-bias 3-block baseline: {CYCLE_213_3B_BASELINE:.3f}")
lines.append(f"- Overfit check: final@4000 mean = {final_mean:.3f}")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
