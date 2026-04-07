"""
Cycle 275 — Item 285: Hybrid Pareto recipe — [1] K=512 + block 11 rank-22 c_proj.

Combines cycle 271's [1] K=512 (CompressedMLP topk+lstsq + Adam W_out+b_out) with
cycle 270's block 11 rank-22 weight-SVD c_proj retrain (LowRankCProj, U_22 Σ_22
frozen, B=Vh[:22,:] + bias trainable).

Per seed: load fresh GPT-2 small, install BOTH compressors in the same model,
joint Adam over [block_1.W_out, block_1.b_out, block_11.B, block_11.bias], 4000
steps, EVAL_EVERY=100, best-state restore. 3 seeds {42, 137, 2024}.

Verdict (item 285 verbatim):
  CONFIRMED if best_mean ≤ 43.5
  REFUTED   if best_mean ≥ 45.0
  INCONCLUSIVE otherwise.
"""
import json
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
BLOCK_1 = 1
BLOCK_11 = 11
K = 512
RANK_11 = 22
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR = 1e-4
N_STEPS = 4000
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
LOAD_SEED = 42
OUTPUT_PATH = "autoresearch/results/cycle_275_hybrid_block1_block11.json"
SUMMARY_PATH = "autoresearch/results/cycle_275_summary.md"

# Verdict thresholds (item 285 verbatim)
CONFIRMED_LE = 43.5
REFUTED_GE = 45.0

# Baselines
CYCLE_270_BLOCK11_R22_RETRAIN = 54.469
CYCLE_271_K512_1BLOCK_1 = 43.821
CYCLE_266_K512_2BLOCK_1_4 = 43.852
CYCLE_273_HETERO_K11_22 = 52.637
FLOAT_BASELINE_REF = 53.895

torch.manual_seed(LOAD_SEED)
print(f"Device: {DEVICE}", flush=True)

# ── Load tokenizer + data ─────────────────────────────────────────────────────
print("Loading GPT-2 and data...", flush=True)
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
print(f"Calibration: {len(calib_ids)} seqs, Eval: {len(eval_ids)} seqs", flush=True)


# ── Helpers (verbatim from cycle 271) ─────────────────────────────────────────
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


# ── LowRankCProj (verbatim from cycle 270) ────────────────────────────────────
class LowRankCProj(nn.Module):
    """Replacement for HF Conv1D c_proj at rank r.

    Stores frozen A = U[:, :r] * Σ[:r] (3072, r) as buffer, trainable B (r, 768)
    as nn.Parameter (initialised from Vh[:r, :]), trainable bias (768,) initialised
    from c_proj.bias. Forward: x @ (A @ B) + b. Matches HF Conv1D semantics
    (input @ weight + bias).
    """

    def __init__(self, A_frozen, B_init, b_init, nf=768):
        super().__init__()
        self.register_buffer("A", A_frozen.contiguous())
        self.B = nn.Parameter(B_init.clone().contiguous())
        self.bias = nn.Parameter(b_init.clone().contiguous())
        self.nf = nf

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        W = self.A @ self.B
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), W)
        x = x.view(size_out)
        return x


# ── Float baseline + SVD of c_proj_11 (once) ──────────────────────────────────
print("\nFloat baseline (single load)...", flush=True)
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()
t0 = time.time()
float_ppl = eval_ppl(model_float, eval_ids)
print(f"Float PPL: {float_ppl:.3f}  ({time.time()-t0:.1f}s)", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from reference {FLOAT_BASELINE_REF} by > 0.5"
)

# Pull block 11 c_proj weight and bias for SVD
c_proj_11_weight = model_float.transformer.h[BLOCK_11].mlp.c_proj.weight.data.detach().clone()
c_proj_11_bias = model_float.transformer.h[BLOCK_11].mlp.c_proj.bias.data.detach().clone()
assert c_proj_11_weight.shape == (3072, 768), c_proj_11_weight.shape
print(f"\nSVD of block {BLOCK_11} c_proj.weight (rank-{RANK_11})...", flush=True)
W_cpu = c_proj_11_weight.float().cpu()
t0 = time.time()
U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
print(f"  SVD done in {time.time()-t0:.1f}s; U {tuple(U.shape)}, S {tuple(S.shape)}, Vh {tuple(Vh.shape)}", flush=True)
A22 = (U[:, :RANK_11] * S[:RANK_11]).contiguous()  # (3072, 22)
B22_init = Vh[:RANK_11, :].contiguous()  # (22, 768)
print(f"  S[0..4]: {S[:5].tolist()}", flush=True)
print(f"  S[21]: {float(S[21]):.4f}, S[22]: {float(S[22]):.4f}", flush=True)
energy_22 = float((S[:RANK_11] ** 2).sum() / (S ** 2).sum())
print(f"  Top-{RANK_11} energy fraction: {energy_22:.4f}", flush=True)

del model_float
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Per-seed run ──────────────────────────────────────────────────────────────
def run_seed(seed):
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
    torch.manual_seed(seed)
    random.seed(seed)

    # Fresh model load
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()

    # ── Step A: collect MLP I/O for block 1, install K=512 CompressedMLP ──
    ln_outs, proj_outs = collect_mlp_io(model, calib_ids, [BLOCK_1])
    model.transformer.h[BLOCK_1].mlp = fit_block_lstsq(
        model.transformer.h[BLOCK_1], ln_outs[BLOCK_1], proj_outs[BLOCK_1], K
    ).to(DEVICE)
    del ln_outs, proj_outs

    # ── Step B: install LowRankCProj at block 11 ──
    lowrank_11 = LowRankCProj(
        A_frozen=A22.to(device=DEVICE, dtype=torch.float32),
        B_init=B22_init.to(device=DEVICE, dtype=torch.float32),
        b_init=c_proj_11_bias.to(device=DEVICE, dtype=torch.float32),
        nf=768,
    ).to(DEVICE)
    model.transformer.h[BLOCK_11].mlp.c_proj = lowrank_11

    init_ppl = eval_ppl(model, eval_ids)
    print(f"Init PPL (block 1 K={K} lstsq + block 11 rank-{RANK_11} no Adam): {init_ppl:.4f}", flush=True)

    # ── Step C: freeze all, unfreeze the 4 trainable tensors ──
    for p in model.parameters():
        p.requires_grad_(False)

    block_1_mlp = model.transformer.h[BLOCK_1].mlp
    block_1_mlp.W_out.requires_grad_(True)
    block_1_mlp.b_out.requires_grad_(True)
    lowrank_11.B.requires_grad_(True)
    lowrank_11.bias.requires_grad_(True)
    trainable_params = [block_1_mlp.W_out, block_1_mlp.b_out, lowrank_11.B, lowrank_11.bias]
    n_trainable = sum(p.numel() for p in trainable_params)
    expected = K * 768 + 768 + RANK_11 * 768 + 768
    print(f"Trainable params: {n_trainable:,} (expect {expected:,} = {K}×768+768+{RANK_11}×768+768)", flush=True)
    assert n_trainable == expected, f"trainable count {n_trainable} != {expected}"

    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    calib_shuffled = list(calib_ids)
    random.shuffle(calib_shuffled)

    def snapshot():
        return {
            "block_1_W_out": block_1_mlp.W_out.detach().clone().cpu(),
            "block_1_b_out": block_1_mlp.b_out.detach().clone().cpu(),
            "block_11_B": lowrank_11.B.detach().clone().cpu(),
            "block_11_bias": lowrank_11.bias.detach().clone().cpu(),
        }

    best_ppl = init_ppl
    best_step = 0
    best_state = snapshot()
    history = [{"step": 0, "eval_ppl": float(init_ppl)}]

    step = 0
    t_train = time.time()
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
                if step % 500 == 0:
                    print(f"  Step {step:5d}: eval_ppl={ppl_e:.4f}  best={best_ppl:.4f}@{best_step}{marker}", flush=True)

    final_ppl = eval_ppl(model, eval_ids)
    print(f"Final at step {N_STEPS}: {final_ppl:.4f}", flush=True)

    # Restore best
    block_1_mlp.W_out.data.copy_(best_state["block_1_W_out"].to(DEVICE))
    block_1_mlp.b_out.data.copy_(best_state["block_1_b_out"].to(DEVICE))
    lowrank_11.B.data.copy_(best_state["block_11_B"].to(DEVICE))
    lowrank_11.bias.data.copy_(best_state["block_11_bias"].to(DEVICE))
    restored_ppl = eval_ppl(model, eval_ids)
    print(f"Restored best (step {best_step}): {restored_ppl:.4f} (recorded best={best_ppl:.4f})", flush=True)
    assert abs(restored_ppl - best_ppl) < 1e-3, (
        f"Restore mismatch seed {seed}: {restored_ppl} vs {best_ppl}"
    )
    dt = time.time() - t_train
    print(f"  seed {seed}: init={init_ppl:.4f} best={best_ppl:.4f}@{best_step} final@{N_STEPS}={final_ppl:.4f} restored={restored_ppl:.4f}  [{dt/60:.1f} min]", flush=True)

    result = {
        "seed": seed,
        "init_ppl": float(init_ppl),
        "best_ppl": float(best_ppl),
        "best_step": int(best_step),
        "final_ppl_at_n_steps": float(final_ppl),
        "restored_best_ppl": float(restored_ppl),
        "trainable_params": int(n_trainable),
        "history": history,
    }
    del model, optimizer, trainable_params, best_state, lowrank_11
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return result


# ── Run all seeds ─────────────────────────────────────────────────────────────
wall_start = time.time()
seed_results = []
for seed in SEEDS:
    r = run_seed(seed)
    seed_results.append(r)
    elapsed = (time.time() - wall_start) / 60
    print(f"[seed {seed} done, elapsed {elapsed:.1f} min]", flush=True)


# ── Aggregate ─────────────────────────────────────────────────────────────────
def agg(results, key):
    vals = torch.tensor([r[key] for r in results], dtype=torch.float64)
    return float(vals.mean()), float(vals.std(unbiased=False))


init_mean, init_std = agg(seed_results, "init_ppl")
best_mean, best_std = agg(seed_results, "best_ppl")
best_step_mean, best_step_std = agg(seed_results, "best_step")
final_mean, final_std = agg(seed_results, "final_ppl_at_n_steps")
best_min = float(min(r["best_ppl"] for r in seed_results))
best_max = float(max(r["best_ppl"] for r in seed_results))

print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
print(f"Float PPL:                   {float_ppl:.4f}")
print(f"init_mean (no Adam):         {init_mean:.4f} ± {init_std:.4f}")
print(f"best_mean:                   {best_mean:.4f} ± {best_std:.4f}  (n={len(SEEDS)})")
print(f"best_step_mean:              {best_step_mean:.0f} ± {best_step_std:.0f}")
print(f"final@{N_STEPS}_mean:        {final_mean:.4f}")
print(f"best_min/best_max:           {best_min:.4f} / {best_max:.4f}")


# ── Verdict ───────────────────────────────────────────────────────────────────
if best_mean <= CONFIRMED_LE:
    verdict = "CONFIRMED"
elif best_mean >= REFUTED_GE:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

delta_vs_271 = best_mean - CYCLE_271_K512_1BLOCK_1
delta_vs_270 = best_mean - CYCLE_270_BLOCK11_R22_RETRAIN
delta_vs_266 = best_mean - CYCLE_266_K512_2BLOCK_1_4
delta_vs_float = best_mean - float_ppl

print(f"\n=== Verdict (item 285 verbatim) ===")
print(f"  best_mean = {best_mean:.4f}; CONFIRMED iff ≤ {CONFIRMED_LE}; REFUTED iff ≥ {REFUTED_GE}")
print(f"  Δ vs cycle 271 [1] K=512  ({CYCLE_271_K512_1BLOCK_1}): {delta_vs_271:+.4f}")
print(f"  Δ vs cycle 270 b11 r=22   ({CYCLE_270_BLOCK11_R22_RETRAIN}): {delta_vs_270:+.4f}")
print(f"  Δ vs cycle 266 [1,4] K=512({CYCLE_266_K512_2BLOCK_1_4}): {delta_vs_266:+.4f}")
print(f"  Δ vs float                ({float_ppl:.3f}): {delta_vs_float:+.4f}")
print(f"  VERDICT: {verdict}")


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 275,
    "item": 285,
    "verdict": verdict,
    "config": {
        "BLOCK_1": BLOCK_1, "BLOCK_11": BLOCK_11, "K": K, "RANK_11": RANK_11,
        "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN, "DEVICE": DEVICE,
        "LR": LR, "N_STEPS": N_STEPS, "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
    },
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "seed_results": seed_results,
    "aggregate": {
        "init_mean": init_mean, "init_std": init_std,
        "best_mean": best_mean, "best_std": best_std,
        "best_min": best_min, "best_max": best_max,
        "best_step_mean": best_step_mean, "best_step_std": best_step_std,
        "final_mean": final_mean, "final_std": final_std,
        "n_seeds": len(SEEDS),
    },
    "verdict_logic": {
        "best_mean": best_mean,
        "threshold_confirmed_le": CONFIRMED_LE,
        "threshold_refuted_ge": REFUTED_GE,
        "verdict": verdict,
    },
    "baselines": {
        "cycle_270_block11_r22_retrain": CYCLE_270_BLOCK11_R22_RETRAIN,
        "cycle_271_k512_1block_1": CYCLE_271_K512_1BLOCK_1,
        "cycle_266_k512_2block_1_4": CYCLE_266_K512_2BLOCK_1_4,
        "cycle_273_hetero_k11_22": CYCLE_273_HETERO_K11_22,
        "float": FLOAT_BASELINE_REF,
    },
    "deltas": {
        "delta_vs_cycle_271": delta_vs_271,
        "delta_vs_cycle_270": delta_vs_270,
        "delta_vs_cycle_266": delta_vs_266,
        "delta_vs_float": delta_vs_float,
    },
    "svd_summary": {
        "rank": RANK_11,
        "top_5_sv": S[:5].tolist(),
        "S_at_22": float(S[21]),
        "S_at_23": float(S[22]),
        "energy_fraction": energy_22,
    },
    "trainable_params_total": 411648,
    "wall_time_sec": time.time() - wall_start,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved JSON to {OUTPUT_PATH}")


# ── Summary markdown ──────────────────────────────────────────────────────────
def fmt(x, n=4):
    return f"{x:.{n}f}"


lines = []
lines.append(f"# Cycle 275 — Item 285: Hybrid [1] K=512 + block 11 rank-22 c_proj")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Criteria (item 285 verbatim): CONFIRMED if best_mean ≤ {CONFIRMED_LE}; REFUTED if best_mean ≥ {REFUTED_GE}; else INCONCLUSIVE")
lines.append(f"- Float PPL: {fmt(float_ppl)} (ref {FLOAT_BASELINE_REF})")
lines.append(f"- Trainable: 411,648 = block 1 W_out (393,216) + b_out (768) + block 11 B (16,896) + bias (768)")
lines.append(f"- Config: K={K}, RANK_11={RANK_11}, LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}")
lines.append(f"- Wall time: {(time.time()-wall_start)/60:.1f} min")
lines.append("")
lines.append("## Per-seed details")
lines.append("")
lines.append("| seed | init | best | best_step | final@4000 | restored |")
lines.append("|---|---|---|---|---|---|")
for r in seed_results:
    lines.append(f"| {r['seed']} | {fmt(r['init_ppl'])} | {fmt(r['best_ppl'])} | {r['best_step']} | {fmt(r['final_ppl_at_n_steps'])} | {fmt(r['restored_best_ppl'])} |")
lines.append("")
lines.append("## Aggregate")
lines.append("")
lines.append(f"- init_mean: **{fmt(init_mean)} ± {fmt(init_std)}**")
lines.append(f"- best_mean: **{fmt(best_mean)} ± {fmt(best_std)}**  (min {fmt(best_min)}, max {fmt(best_max)})")
lines.append(f"- best_step_mean: {best_step_mean:.0f} ± {best_step_std:.0f}")
lines.append(f"- final@{N_STEPS}_mean: {fmt(final_mean)}")
lines.append("")
lines.append("## Verdict logic (item 285 verbatim)")
lines.append("")
lines.append(f"- best_mean = {fmt(best_mean)}; CONFIRMED iff ≤ {CONFIRMED_LE}; REFUTED iff ≥ {REFUTED_GE}")
lines.append(f"- VERDICT: **{verdict}**")
lines.append("")
lines.append("## Baselines and deltas")
lines.append("")
lines.append(f"- cycle 271 [1] K=512: **{CYCLE_271_K512_1BLOCK_1}** (394K). Δ = {delta_vs_271:+.4f}")
lines.append(f"- cycle 270 block 11 r=22 retrain: {CYCLE_270_BLOCK11_R22_RETRAIN} (17.7K). Δ = {delta_vs_270:+.4f}")
lines.append(f"- cycle 266 [1,4] K=512: {CYCLE_266_K512_2BLOCK_1_4} (788K). Δ = {delta_vs_266:+.4f}")
lines.append(f"- cycle 273 hetero K_11=22 (channel-K, NOT weight-SVD): {CYCLE_273_HETERO_K11_22}")
lines.append(f"- Float: {FLOAT_BASELINE_REF}. Δ = {delta_vs_float:+.4f}")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {SUMMARY_PATH}")
