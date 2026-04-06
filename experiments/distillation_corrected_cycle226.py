"""
Cycle 226 — Item 236: KD distillation baseline rerun with CORRECTED loss normalization.

Cycle 224 was downgraded to INCONCLUSIVE because F.kl_div(..., reduction='batchmean')
on (B=1, T=128, V=50257) divides only by B=1 → kd_loss summed over T·V≈6.4M entries
(~30-50) while CE averages over tokens (~4). The 50/50 mix was effectively ~10× KD-weighted.

Fix: reshape student/teacher logits to (B*T, V) before kl_div so batchmean divides by
B*T, giving per-token KD comparable to CE.

Phase 1: α-sweep (SEED=42, 2000 steps, EVAL_EVERY=200, α ∈ {0.0, 0.5, 1.0}).
Phase 2: best α full run (SEEDS=[42, 137, 2024], 4000 steps, EVAL_EVERY=100).

Verdict (item 236):
  Confirmed if best_mean ≤ 44.0
  Refuted    if best_mean ≥ 47.0
  Else INCONCLUSIVE
"""
import copy
import json
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS = [1, 4, 7]
K = 512
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
TAU = 2.0

# Sweep phase
SWEEP_SEED = 42
SWEEP_STEPS = 2000
SWEEP_EVAL_EVERY = 200
SWEEP_ALPHAS = [0.0, 0.5, 1.0]

# Full run phase
FULL_SEEDS = [42, 137, 2024]
FULL_STEPS = 4000
FULL_EVAL_EVERY = 100

OUTPUT_PATH = "autoresearch/results/cycle_226_distillation_corrected.json"
SUMMARY_PATH = "autoresearch/results/cycle_226_summary.md"

# Baselines for comparison
CYCLE_213_WOUT_WFC_CE = 46.424
CYCLE_224_BROKEN_KD = 60.938

print(f"Device: {DEVICE}")

# ── Load tokenizer + data ─────────────────────────────────────────────────────
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
class FullCompressedMLP(nn.Module):
    """Both W_fc and W_out are trainable (matches cycle 213)."""

    def __init__(self, fc_weight, fc_bias, W_out, b_out):
        super().__init__()
        self.W_fc = nn.Parameter(fc_weight.clone())   # (768, K)
        self.b_fc = nn.Parameter(fc_bias.clone())     # (K,)
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


# ── Float baseline + persistent teacher ───────────────────────────────────────
print("\nLoading float teacher...")
teacher = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
float_ppl = eval_ppl(teacher, eval_ids)
print(f"Float teacher PPL: {float_ppl:.3f}")


# ── Snapshot/restore ──────────────────────────────────────────────────────────
def snapshot_compressed(model):
    out = {}
    for b in BLOCKS:
        mlp = model.transformer.h[b].mlp
        out[b] = (
            mlp.W_out.detach().clone().cpu(),
            mlp.b_out.detach().clone().cpu(),
            mlp.W_fc.detach().clone().cpu(),
            mlp.b_fc.detach().clone().cpu(),
        )
    return out


def restore_compressed(model, state):
    for b in BLOCKS:
        W_out, b_out, W_fc, b_fc = state[b]
        mlp = model.transformer.h[b].mlp
        mlp.W_out.data.copy_(W_out.to(DEVICE))
        mlp.b_out.data.copy_(b_out.to(DEVICE))
        mlp.W_fc.data.copy_(W_fc.to(DEVICE))
        mlp.b_fc.data.copy_(b_fc.to(DEVICE))


# ── Per-(seed, alpha) run ────────────────────────────────────────────────────
def run_seed(seed, alpha, n_steps, eval_every, tag=""):
    print(f"\n{'='*60}\nSEED {seed}  α={alpha}  steps={n_steps}  {tag}\n{'='*60}")
    torch.manual_seed(seed)
    random.seed(seed)

    student = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    student.eval()
    ln_outs, proj_outs = collect_mlp_io(student, calib_ids, BLOCKS)
    for b in BLOCKS:
        student.transformer.h[b].mlp = fit_block_lstsq(
            student.transformer.h[b], ln_outs[b], proj_outs[b], K).to(DEVICE)
    del ln_outs, proj_outs

    init_ppl = eval_ppl(student, eval_ids)
    print(f"Init PPL (topk+lstsq): {init_ppl:.3f}")

    # Freeze all, unfreeze W_out/b_out/W_fc/b_fc of compressed blocks
    for param in student.parameters():
        param.requires_grad_(False)
    trainable_params = []
    for b in BLOCKS:
        mlp = student.transformer.h[b].mlp
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

    best_ppl = init_ppl
    best_step = 0
    best_state = snapshot_compressed(student)
    history = []

    step = 0
    idx = 0
    print(f"\nTraining {n_steps} steps with KD+CE α={alpha} (tau={TAU}) — CORRECTED norm...")
    while step < n_steps:
        if idx >= len(calib_shuffled):
            random.shuffle(calib_shuffled)
            idx = 0
        ids = calib_shuffled[idx]
        idx += 1

        student.train()
        optimizer.zero_grad()

        if alpha > 0.0:
            with torch.no_grad():
                teacher_logits = teacher(ids).logits  # (B, T, V)
            student_logits = student(ids).logits

            # CORRECTED KD: reshape (B,T,V) -> (B*T, V) so batchmean divides by B*T
            B, T, V = student_logits.shape
            student_log_softmax = F.log_softmax(
                student_logits.reshape(B * T, V) / TAU, dim=-1)
            teacher_softmax = F.softmax(
                teacher_logits.reshape(B * T, V) / TAU, dim=-1)
            kd_loss = F.kl_div(student_log_softmax, teacher_softmax,
                               reduction='batchmean') * (TAU ** 2)
        else:
            student_logits = student(ids).logits
            B, T, V = student_logits.shape
            kd_loss = torch.tensor(0.0, device=DEVICE)

        ce_loss = F.cross_entropy(
            student_logits[:, :-1].reshape(-1, V),
            ids[:, 1:].reshape(-1),
        )
        total_loss = (1.0 - alpha) * ce_loss + alpha * kd_loss

        total_loss.backward()
        optimizer.step()
        step += 1

        if step <= 5:
            print(f"  step {step}: kd={kd_loss.item():.4f} ce={ce_loss.item():.4f} "
                  f"total={total_loss.item():.4f}", flush=True)

        if step % eval_every == 0:
            ppl_e = eval_ppl(student, eval_ids)
            history.append({"step": step, "eval_ppl": float(ppl_e)})
            marker = ""
            if ppl_e < best_ppl:
                best_ppl = float(ppl_e)
                best_step = step
                best_state = snapshot_compressed(student)
                marker = " *"
            print(f"  Step {step:5d}: eval_ppl={ppl_e:.3f}{marker}", flush=True)

    # Restore best
    restore_compressed(student, best_state)
    best_ppl_restored = eval_ppl(student, eval_ids)
    print(f"Restored best (step {best_step}): {best_ppl_restored:.3f} "
          f"(recorded={best_ppl:.3f})")
    assert abs(best_ppl_restored - best_ppl) < 1e-3, \
        f"Restore mismatch: {best_ppl_restored} vs {best_ppl}"

    result = {
        "seed": seed,
        "alpha": alpha,
        "init_ppl": float(init_ppl),
        "best_ppl": float(best_ppl),
        "best_ppl_restored": float(best_ppl_restored),
        "best_step": int(best_step),
        "history": history,
    }
    del student, optimizer, trainable_params, best_state
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return result


# ── Phase 1: α-sweep ──────────────────────────────────────────────────────────
print(f"\n{'#'*60}\n# PHASE 1: α-sweep (SEED={SWEEP_SEED}, {SWEEP_STEPS} steps)\n{'#'*60}")
sweep_results = []
for a in SWEEP_ALPHAS:
    r = run_seed(SWEEP_SEED, a, SWEEP_STEPS, SWEEP_EVAL_EVERY, tag="[sweep]")
    sweep_results.append(r)

print(f"\n{'='*60}\nSWEEP SUMMARY\n{'='*60}")
print(f"{'alpha':>8}  {'init':>10}  {'best':>10}  {'best_step':>10}")
for r in sweep_results:
    print(f"  {r['alpha']:6.2f}  {r['init_ppl']:10.3f}  {r['best_ppl']:10.3f}  {r['best_step']:10d}")

best_sweep = min(sweep_results, key=lambda r: r['best_ppl'])
best_alpha = best_sweep['alpha']
print(f"\nBest α: {best_alpha} (best_ppl={best_sweep['best_ppl']:.3f}, "
      f"best_step={best_sweep['best_step']})")


# ── Phase 2: full run at best α ───────────────────────────────────────────────
print(f"\n{'#'*60}\n# PHASE 2: full run (SEEDS={FULL_SEEDS}, "
      f"{FULL_STEPS} steps, α={best_alpha})\n{'#'*60}")
seed_results = []
for s in FULL_SEEDS:
    r = run_seed(s, best_alpha, FULL_STEPS, FULL_EVAL_EVERY, tag="[full]")
    seed_results.append(r)


# ── Aggregate ─────────────────────────────────────────────────────────────────
def agg(results, key):
    vals = torch.tensor([r[key] for r in results], dtype=torch.float64)
    return float(vals.mean()), float(vals.std(unbiased=False))


init_mean, init_std = agg(seed_results, "init_ppl")
best_mean, best_std = agg(seed_results, "best_ppl")
best_step_mean, best_step_std = agg(seed_results, "best_step")

# Verdict
if best_mean <= 44.0:
    verdict = "CONFIRMED"
elif best_mean >= 47.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

# Special note: if best α = 0 (pure CE wins), KD provides no benefit
# → effectively cycle 213 reproduction → mark INCONCLUSIVE in summary
kd_no_benefit = (best_alpha == 0.0)
if kd_no_benefit:
    verdict_note = (f"NOTE: best α=0 (pure CE wins) — KD provides no benefit; "
                    f"this is essentially cycle 213 reproduction. "
                    f"Marking INCONCLUSIVE because hypothesis 'KD beats CE' is not supported.")
    verdict = "INCONCLUSIVE"
else:
    verdict_note = ""

print(f"\n{'='*60}\nFULL RUN AGGREGATE\n{'='*60}")
print(f"Float PPL:                 {float_ppl:.3f}")
print(f"Init PPL  (mean ± std):    {init_mean:.3f} ± {init_std:.3f}")
print(f"Best PPL  (mean ± std):    {best_mean:.3f} ± {best_std:.3f}")
print(f"Best step (mean ± std):    {best_step_mean:.0f} ± {best_step_std:.0f}")
print(f"Cycle 213 W_out+W_fc CE:   {CYCLE_213_WOUT_WFC_CE}")
print(f"Cycle 224 broken KD:       {CYCLE_224_BROKEN_KD}")
print(f"Delta (best - c213 CE):    {best_mean - CYCLE_213_WOUT_WFC_CE:+.3f}")
print(f"Best α: {best_alpha}")
print(f"Verdict: {verdict}")
if verdict_note:
    print(verdict_note)

result = {
    "cycle": 226,
    "item": 236,
    "verdict": verdict,
    "config": {
        "BLOCKS": BLOCKS, "K": K, "LR": LR, "TAU": TAU,
        "SWEEP_STEPS": SWEEP_STEPS, "SWEEP_EVAL_EVERY": SWEEP_EVAL_EVERY,
        "SWEEP_ALPHAS": SWEEP_ALPHAS,
        "FULL_STEPS": FULL_STEPS, "FULL_EVAL_EVERY": FULL_EVAL_EVERY,
        "SEEDS": FULL_SEEDS,
        "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN,
    },
    "alpha_sweep": [
        {"alpha": r["alpha"], "init_ppl": r["init_ppl"],
         "best_ppl": r["best_ppl"], "best_step": r["best_step"]}
        for r in sweep_results
    ],
    "best_alpha": best_alpha,
    "full_run": {
        "seeds": seed_results,
        "aggregate": {
            "init_mean": init_mean, "init_std": init_std,
            "best_mean": best_mean, "best_std": best_std,
            "best_step_mean": best_step_mean, "best_step_std": best_step_std,
        },
    },
    "baselines": {
        "cycle_213_pure_ce": CYCLE_213_WOUT_WFC_CE,
        "cycle_224_broken_kd": CYCLE_224_BROKEN_KD,
        "float": float(float_ppl),
    },
    "verdict_note": verdict_note,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON to {OUTPUT_PATH}")

# ── Summary markdown ──────────────────────────────────────────────────────────
lines = []
lines.append("# Cycle 226 — Item 236: KD distillation (CORRECTED loss normalization)")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
if verdict_note:
    lines.append("")
    lines.append(f"_{verdict_note}_")
lines.append("")
lines.append("- Criteria (item 236): Confirmed ≤44.0, Refuted ≥47.0, else Inconclusive")
lines.append(f"- Fix: reshape `(B,T,V)` → `(B*T, V)` before `kl_div(reduction='batchmean')` "
             f"so KD divides by per-token count (B·T) and matches CE scale.")
lines.append(f"- Config: BLOCKS={BLOCKS}, K={K}, LR={LR}, τ={TAU}")
lines.append(f"- Float teacher PPL: {float_ppl:.3f}")
lines.append("")
lines.append("## Phase 1 — α-sweep (SEED=42, 2000 steps, eval_every=200)")
lines.append("")
lines.append("| α | init | best | best_step |")
lines.append("|---|---|---|---|")
for r in sweep_results:
    lines.append(f"| {r['alpha']:.2f} | {r['init_ppl']:.3f} | "
                 f"{r['best_ppl']:.3f} | {r['best_step']} |")
lines.append("")
lines.append(f"**Best α = {best_alpha}** (best_ppl={best_sweep['best_ppl']:.3f})")
lines.append("")
lines.append(f"## Phase 2 — full run (SEEDS={FULL_SEEDS}, {FULL_STEPS} steps, α={best_alpha})")
lines.append("")
lines.append("| seed | init | best | best_step |")
lines.append("|---|---|---|---|")
for r in seed_results:
    lines.append(f"| {r['seed']} | {r['init_ppl']:.3f} | "
                 f"{r['best_ppl']:.3f} | {r['best_step']} |")
lines.append(f"| **mean** | {init_mean:.3f} | **{best_mean:.3f}** | {best_step_mean:.0f} |")
lines.append(f"| **std**  | {init_std:.3f} | {best_std:.3f} | {best_step_std:.0f} |")
lines.append("")
lines.append("## Comparison vs baselines")
lines.append("")
lines.append("| training | loss | best PPL |")
lines.append("|---|---|---|")
lines.append(f"| W_out+W_fc | CE only (cycle 213) | {CYCLE_213_WOUT_WFC_CE} |")
lines.append(f"| W_out+W_fc | KD+CE BROKEN norm (cycle 224) | {CYCLE_224_BROKEN_KD} |")
lines.append(f"| W_out+W_fc | KD+CE CORRECTED α={best_alpha} (this) | "
             f"**{best_mean:.3f} ± {best_std:.3f}** |")
lines.append("")
lines.append(f"- Δ vs cycle 213 (CE same surface): **{best_mean - CYCLE_213_WOUT_WFC_CE:+.3f} PPL**")
lines.append(f"- Δ vs cycle 224 (broken KD):       **{best_mean - CYCLE_224_BROKEN_KD:+.3f} PPL**")
lines.append(f"- Float ceiling: {float_ppl:.3f}")
lines.append("")
lines.append("## Per-seed trajectories")
lines.append("")
for r in seed_results:
    lines.append(f"### Seed {r['seed']}")
    hist = r["history"]
    parts = [f"init={r['init_ppl']:.2f}"]
    if hist:
        q = hist[len(hist)//4]
        h = hist[len(hist)//2]
        tq = hist[3*len(hist)//4]
        f_ = hist[-1]
        parts.append(f"step{q['step']}={q['eval_ppl']:.2f}")
        parts.append(f"step{h['step']}={h['eval_ppl']:.2f}")
        parts.append(f"best=step{r['best_step']}@{r['best_ppl']:.2f}")
        parts.append(f"step{tq['step']}={tq['eval_ppl']:.2f}")
        parts.append(f"step{f_['step']}={f_['eval_ppl']:.2f}")
    lines.append("- " + ", ".join(parts))
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
