"""
Cycle 309 — Item 319: Δy-aligned rank-r LASER at block 8.

Replace c_proj_8.weight with a rank-r factorization whose row space is
constrained to span(Vh_y[:r]), where Vh_y is the right-singular basis of
Δy = (GELU - ReLU)(z_8) @ W_proj_8 measured on calib data.

Geometric construction:
  W_new = A_y @ B
    A_y    = W_proj @ Vh_y[:r].T   ∈ R^(3072, r)   (frozen)
    B_init = Vh_y[:r]              ∈ R^(r, 768)    (trainable)
    bias   = c_proj.bias clone     ∈ R^(768,)      (trainable)
  At init: A_y @ B = W_proj @ (Vh_y[:r].T @ Vh_y[:r])
         = orthogonal projection of W_proj's columns onto span(Vh_y[:r]).

Paired baseline (cycle 280-style W_proj basis):
    A_w = U_w[:, :r] * S_w[:r]     (frozen)
    B_w = Vh_w[:r]                  (trainable)

For r ∈ {1, 2}, two bases (dy / wproj):
  - no-retrain PPL on the 100-seq eval slice
  - 3-seed Adam retrain (LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore)

Verdict (item 319 verbatim):
  CONFIRMED iff r=1 Δy-aligned retrain best_mean ≤ 60.0
  REFUTED   iff r=1 Δy-aligned retrain best_mean ≥ 110.0
  INCONCLUSIVE otherwise
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

from experiments.curvature_residual_geometric import UnifiedCurvatureFramework

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_BLOCK = 8
RANKS = [1, 2]
BASES = ["dy", "wproj"]  # "dy" = Δy basis, "wproj" = cycle 280 W_proj basis
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 4000
LR = 1e-4
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
LOAD_SEED = 42
OUTPUT_PATH = "autoresearch/results/cycle_309_block8_dy_aligned_laser.json"
SUMMARY_PATH = "autoresearch/results/cycle_309_summary.md"

# Verdict thresholds (item 319 verbatim, applied to dy-basis r=1 retrain best_mean)
CONFIRMED_RETRAIN_LE = 60.0
REFUTED_RETRAIN_GE = 110.0

FLOAT_BASELINE_REF = 53.895  # cycle 270/280/305 ref
CYCLE_306_B8_TOP1_E = 0.9141        # Δy top-1 energy at b8
CYCLE_306_B8_DY_EFF_RANK = 1.197    # Δy stable rank at b8
CYCLE_306_B8_E_KEPT_TOP2_WPROJ = 0.4722  # 47% of Δy in Vh_w[:2]

torch.manual_seed(LOAD_SEED)
print(f"Device: {DEVICE}", flush=True)


# ── Load model + data ─────────────────────────────────────────────────────────
print("Loading GPT-2 and wikitext-2 val...", flush=True)
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

calib_ids_cpu = [t for t in all_ids[:N_CALIB]]
eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB:N_CALIB + N_EVAL]]
print(f"Calibration: {len(calib_ids_cpu)} seqs, Eval: {len(eval_ids)} seqs", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def eval_ppl(model, eval_ids):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


class LowRankCProj(nn.Module):
    """Replacement for HF Conv1D c_proj at rank r.

    Forward: y = (x @ A) @ B + bias  ==  x @ (A @ B) + bias.
    Matches HF Conv1D semantics (input @ weight + bias).

    A is the frozen factor (3072, r); B is the trainable factor (r, 768).
    bias (768,) is also trainable.
    """

    def __init__(self, A_frozen, B_init, b_init, nf=768):
        super().__init__()
        self.register_buffer("A", A_frozen.contiguous())
        self.B = nn.Parameter(B_init.clone().contiguous())
        self.bias = nn.Parameter(b_init.clone().contiguous())
        self.nf = nf

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        W = self.A @ self.B  # (3072, 768)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), W)
        x = x.view(size_out)
        return x


# ── Float baseline ────────────────────────────────────────────────────────────
print("Float baseline PPL on eval slice...", flush=True)
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
t0 = time.time()
float_ppl = eval_ppl(model, eval_ids)
print(f"Float PPL: {float_ppl:.4f}  ({time.time()-t0:.1f}s)", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from ref {FLOAT_BASELINE_REF} by > 0.5"
)


# ── Originals ─────────────────────────────────────────────────────────────────
c_proj_orig_weight = model.transformer.h[TARGET_BLOCK].mlp.c_proj.weight.data.detach().clone()
c_proj_orig_bias = model.transformer.h[TARGET_BLOCK].mlp.c_proj.bias.data.detach().clone()
assert c_proj_orig_weight.shape == (3072, 768), (
    f"c_proj.weight shape {c_proj_orig_weight.shape} != (3072, 768)"
)
W_proj_cpu = c_proj_orig_weight.float().cpu()  # (3072, 768)


# ── W_proj SVD (paired baseline basis) ────────────────────────────────────────
print(f"\nSVD of c_proj_{TARGET_BLOCK}.weight (W_proj basis)...", flush=True)
t0 = time.time()
Uw, Sw, Vhw = torch.linalg.svd(W_proj_cpu, full_matrices=False)
print(
    f"  Uw {tuple(Uw.shape)}, Sw {tuple(Sw.shape)}, Vhw {tuple(Vhw.shape)}  "
    f"({time.time()-t0:.1f}s)",
    flush=True,
)
print(f"  Top-5 W_proj singular values: {[float(s) for s in Sw[:5].tolist()]}", flush=True)


# ── Δy SVD (Δy-aligned basis) ─────────────────────────────────────────────────
print(f"\nComputing Δy at block {TARGET_BLOCK} (single forward pass on {N_CALIB} calib seqs)...", flush=True)
calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]
fw = UnifiedCurvatureFramework(
    model=model,
    sigma=lambda z: F.gelu(z, approximate="tanh"),
    sigma_lin=F.relu,
    blocks=[TARGET_BLOCK],
    calib_ids=calib_ids_dev,
    device=DEVICE,
    tokenizer=tokenizer,
    A_factors=None,
    sigma_name="GELU",
    sigma_lin_name="ReLU",
)
t0 = time.time()
fw.capture_inputs()
print(f"  captured c_fc inputs in {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
c = fw.compute_Dy(TARGET_BLOCK)
Dy = c["Dy"]                  # (N_TOK, 768) CPU float32
W_proj_fw = c["W_proj"]       # (3072, 768) CPU float32
N_TOK = Dy.shape[0]
assert Dy.shape == (N_TOK, 768), f"Dy {tuple(Dy.shape)}"
assert torch.allclose(W_proj_fw, W_proj_cpu, atol=1e-6), (
    "framework W_proj differs from model c_proj.weight"
)
print(f"  Dy shape ({N_TOK}, 768)  ({time.time()-t0:.1f}s)", flush=True)

t0 = time.time()
Uy, Sy, Vhy = torch.linalg.svd(Dy, full_matrices=False)
print(
    f"  Δy SVD: Uy {tuple(Uy.shape)}, Sy {tuple(Sy.shape)}, Vhy {tuple(Vhy.shape)}  "
    f"({time.time()-t0:.1f}s)",
    flush=True,
)
print(f"  Top-5 Δy singular values: {[float(s) for s in Sy[:5].tolist()]}", flush=True)
Sy2 = Sy * Sy
Dy_total_energy = float(Sy2.sum().item())
Dy_top1_E_measured = float(Sy2[0].item() / Dy_total_energy)
Dy_top2_E_measured = float(Sy2[:2].sum().item() / Dy_total_energy)
print(
    f"  Δy top1_E={Dy_top1_E_measured:.4f} (cycle 306 ref {CYCLE_306_B8_TOP1_E:.4f}), "
    f"top2_E={Dy_top2_E_measured:.4f}",
    flush=True,
)
assert abs(Dy_top1_E_measured - CYCLE_306_B8_TOP1_E) < 1e-3, (
    f"Δy top1_E {Dy_top1_E_measured} vs cycle 306 ref {CYCLE_306_B8_TOP1_E}"
)

# Free framework caches before training loops
fw.free_block(TARGET_BLOCK)
del fw, calib_ids_dev, Dy, Uy
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Cross-basis sanity: how much of W_proj's columns live in span(Vh_y[:r]) ──
print("\nCross-basis projection sanity:", flush=True)
for r in RANKS:
    Vy_r = Vhy[:r, :].contiguous()  # (r, 768)
    II = Vy_r @ Vy_r.T
    off_diag = float((II - torch.eye(r)).abs().max().item())
    assert off_diag < 1e-4, f"Vhy[:{r}] not orthonormal: {off_diag}"
    Wproj_in_VyR = W_proj_cpu @ Vy_r.T @ Vy_r  # (3072, 768)
    proj_energy = float((Wproj_in_VyR * Wproj_in_VyR).sum().item())
    full_energy = float((W_proj_cpu * W_proj_cpu).sum().item())
    energy_frac = proj_energy / full_energy
    print(
        f"  r={r}: Vhy[:{r}] orthonormal off_diag={off_diag:.2e}; "
        f"||W_proj @ Vhy[:r]^T Vhy[:r]||_F^2 / ||W_proj||_F^2 = {energy_frac:.4f}",
        flush=True,
    )


# ── Build factor pairs for each (basis, r) cell ───────────────────────────────
def build_factors(basis: str, r: int):
    """Return (A_frozen (3072, r), B_init (r, 768)) on CPU float32."""
    if basis == "dy":
        Vy_r = Vhy[:r, :].contiguous()              # (r, 768)
        A = (W_proj_cpu @ Vy_r.T).contiguous()      # (3072, r)
        B = Vy_r.clone()                            # (r, 768)
        return A, B
    elif basis == "wproj":
        A = (Uw[:, :r] * Sw[:r]).contiguous()       # (3072, r)
        B = Vhw[:r, :].contiguous()                 # (r, 768)
        return A, B
    else:
        raise ValueError(f"unknown basis {basis}")


# ── No-retrain sweep: 4 cells (r ∈ {1, 2}) × (basis ∈ {dy, wproj}) ───────────
print("\n=== No-retrain sweep ===", flush=True)
device_dtype = c_proj_orig_weight.dtype
device_target = c_proj_orig_weight.device

noretrain_results = {}
for basis in BASES:
    for r in RANKS:
        A, B = build_factors(basis, r)
        W_r = (A @ B).to(device=device_target, dtype=device_dtype)
        assert W_r.shape == c_proj_orig_weight.shape
        with torch.no_grad():
            model.transformer.h[TARGET_BLOCK].mlp.c_proj.weight.data.copy_(W_r)
        t0 = time.time()
        ppl_r = eval_ppl(model, eval_ids)
        dt = time.time() - t0
        delta = ppl_r - float_ppl
        pct = (ppl_r / float_ppl - 1) * 100
        key = f"{basis}_r{r}"
        print(
            f"  {key:10s}: PPL={ppl_r:.4f}  Δfloat={delta:+.3f}  ({pct:+.2f}%)  [{dt:.1f}s]",
            flush=True,
        )
        noretrain_results[key] = {
            "basis": basis,
            "rank": r,
            "ppl": float(ppl_r),
            "delta_float": float(delta),
            "pct_above_float": float(pct),
        }

# Restore original weight
with torch.no_grad():
    model.transformer.h[TARGET_BLOCK].mlp.c_proj.weight.data.copy_(c_proj_orig_weight)
ppl_restored = eval_ppl(model, eval_ids)
assert abs(ppl_restored - float_ppl) < 1e-3, (
    f"restore failed: {ppl_restored} vs {float_ppl}"
)
print(f"  Restored, PPL={ppl_restored:.4f} (matches float {float_ppl:.4f})", flush=True)


# ── Retrain sweep: same 4 cells × 3 seeds ─────────────────────────────────────
print(f"\n=== Retrain sweep (Adam LR={LR}, N_STEPS={N_STEPS}, {len(SEEDS)} seeds) ===", flush=True)
retrain_cells = {}
for basis in BASES:
    for r in RANKS:
        cell_key = f"{basis}_r{r}"
        print(f"\n--- cell {cell_key} ---", flush=True)
        A_init, B_init = build_factors(basis, r)
        n_trainable_expected = r * 768 + 768

        seeds_data = []
        for seed in SEEDS:
            print(f"  seed {seed}:", flush=True)
            torch.manual_seed(seed)
            random.seed(seed)

            model_s = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
            model_s.eval()

            lowrank = LowRankCProj(
                A_frozen=A_init.to(device=DEVICE, dtype=torch.float32),
                B_init=B_init.to(device=DEVICE, dtype=torch.float32),
                b_init=c_proj_orig_bias.to(device=DEVICE, dtype=torch.float32),
                nf=768,
            ).to(DEVICE)
            model_s.transformer.h[TARGET_BLOCK].mlp.c_proj = lowrank

            init_ppl = eval_ppl(model_s, eval_ids)
            assert abs(init_ppl - noretrain_results[cell_key]["ppl"]) < 1e-2, (
                f"{cell_key} seed {seed}: init {init_ppl} != noretrain "
                f"{noretrain_results[cell_key]['ppl']}"
            )

            for p in model_s.parameters():
                p.requires_grad_(False)
            lowrank.B.requires_grad_(True)
            lowrank.bias.requires_grad_(True)
            trainable = [lowrank.B, lowrank.bias]
            n_trainable = sum(p.numel() for p in trainable)
            assert n_trainable == n_trainable_expected, (
                f"trainable {n_trainable} != {n_trainable_expected}"
            )

            optimizer = torch.optim.Adam(trainable, lr=LR)

            calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]
            random.shuffle(calib_ids_dev)

            best_ppl = init_ppl
            best_step = 0
            best_state = {
                "B": lowrank.B.detach().clone(),
                "bias": lowrank.bias.detach().clone(),
            }
            history = [{"step": 0, "eval_ppl": float(init_ppl)}]

            step = 0
            t_train = time.time()
            while step < N_STEPS:
                for ids in calib_ids_dev:
                    if step >= N_STEPS:
                        break
                    model_s.train()
                    optimizer.zero_grad()
                    out = model_s(ids, labels=ids)
                    loss = out.loss
                    loss.backward()
                    optimizer.step()
                    step += 1

                    if step % EVAL_EVERY == 0:
                        ppl_e = eval_ppl(model_s, eval_ids)
                        history.append({
                            "step": step,
                            "eval_ppl": float(ppl_e),
                            "calib_loss": float(loss.item()),
                        })
                        if ppl_e < best_ppl:
                            best_ppl = ppl_e
                            best_step = step
                            best_state = {
                                "B": lowrank.B.detach().clone(),
                                "bias": lowrank.bias.detach().clone(),
                            }
                        if step % 500 == 0:
                            print(
                                f"      step {step:4d}: eval_ppl={ppl_e:.4f}  best={best_ppl:.4f}@{best_step}",
                                flush=True,
                            )

            final_ppl = eval_ppl(model_s, eval_ids)
            with torch.no_grad():
                lowrank.B.data.copy_(best_state["B"])
                lowrank.bias.data.copy_(best_state["bias"])
            restored_ppl = eval_ppl(model_s, eval_ids)
            assert abs(restored_ppl - best_ppl) < 1e-3, (
                f"{cell_key} seed {seed}: restored {restored_ppl} != best {best_ppl}"
            )
            dt = time.time() - t_train
            print(
                f"    init={init_ppl:.4f} best={best_ppl:.4f}@{best_step} "
                f"final={final_ppl:.4f} restored={restored_ppl:.4f}  [{dt/60:.1f} min]",
                flush=True,
            )

            seeds_data.append({
                "seed": seed,
                "init_ppl": float(init_ppl),
                "best_ppl": float(best_ppl),
                "best_step": int(best_step),
                "final_ppl_at_n_steps": float(final_ppl),
                "restored_best_ppl": float(restored_ppl),
                "history": history,
            })

            del model_s, lowrank, optimizer
            if DEVICE == "mps":
                torch.mps.empty_cache()

        best_ppls = [s["best_ppl"] for s in seeds_data]
        best_mean = float(sum(best_ppls) / len(best_ppls))
        best_var = float(sum((x - best_mean) ** 2 for x in best_ppls) / len(best_ppls))
        best_std = math.sqrt(best_var)
        print(f"  cell {cell_key}: best_mean={best_mean:.4f} ± {best_std:.4f}", flush=True)
        retrain_cells[cell_key] = {
            "basis": basis,
            "rank": r,
            "n_trainable": n_trainable_expected,
            "seeds": seeds_data,
            "aggregate": {
                "best_mean": best_mean,
                "best_std": best_std,
                "best_min": float(min(best_ppls)),
                "best_max": float(max(best_ppls)),
            },
        }


# ── Verdict (item 319 verbatim, on dy r=1 retrain best_mean) ──────────────────
dy_r1_best_mean = retrain_cells["dy_r1"]["aggregate"]["best_mean"]
if dy_r1_best_mean <= CONFIRMED_RETRAIN_LE:
    verdict = "CONFIRMED"
elif dy_r1_best_mean >= REFUTED_RETRAIN_GE:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print("\n=== Verdict ===", flush=True)
print(f"  dy_r1 retrain best_mean = {dy_r1_best_mean:.4f}", flush=True)
print(f"  CONFIRMED ≤ {CONFIRMED_RETRAIN_LE} ; REFUTED ≥ {REFUTED_RETRAIN_GE}", flush=True)
print(f"  VERDICT: {verdict}", flush=True)

# Comparison summary across cells
print("\n=== Cell summary (no-retrain → retrain best_mean) ===", flush=True)
for cell in ["dy_r1", "dy_r2", "wproj_r1", "wproj_r2"]:
    nr = noretrain_results[cell]["ppl"]
    rt = retrain_cells[cell]["aggregate"]["best_mean"]
    rt_std = retrain_cells[cell]["aggregate"]["best_std"]
    print(
        f"  {cell:10s}: no-retrain={nr:9.4f}  retrain={rt:9.4f} ± {rt_std:.4f}  "
        f"(Δfloat={rt - float_ppl:+.3f})",
        flush=True,
    )


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 309,
    "item": 319,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "target_block": TARGET_BLOCK,
    "noretrain": noretrain_results,
    "retrain_cells": retrain_cells,
    "verdict_logic": {
        "metric": "dy_r1_retrain_best_mean",
        "value": dy_r1_best_mean,
        "threshold_confirmed_le": CONFIRMED_RETRAIN_LE,
        "threshold_refuted_ge": REFUTED_RETRAIN_GE,
        "verdict": verdict,
    },
    "config": {
        "TARGET_BLOCK": TARGET_BLOCK,
        "RANKS": RANKS,
        "BASES": BASES,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE,
        "N_STEPS": N_STEPS,
        "LR": LR,
        "EVAL_EVERY": EVAL_EVERY,
        "SEEDS": SEEDS,
        "n_calib_tokens": N_TOK,
    },
    "geom": {
        "Dy_top1_E_measured": Dy_top1_E_measured,
        "Dy_top2_E_measured": Dy_top2_E_measured,
        "cycle_306_top1_E_ref": CYCLE_306_B8_TOP1_E,
        "cycle_306_dy_eff_rank_ref": CYCLE_306_B8_DY_EFF_RANK,
        "cycle_306_E_kept_top2_via_Wproj_ref": CYCLE_306_B8_E_KEPT_TOP2_WPROJ,
        "Wproj_top5_S": [float(s) for s in Sw[:5].tolist()],
        "Dy_top5_S": [float(s) for s in Sy[:5].tolist()],
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_PATH}", flush=True)


# ── Summary markdown ──────────────────────────────────────────────────────────
def fmt(x, n=4):
    return f"{x:.{n}f}"


lines = []
lines.append(f"# Cycle 309 — Item 319: Δy-aligned rank-r LASER at block {TARGET_BLOCK}")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Float PPL: {fmt(float_ppl)} (ref {FLOAT_BASELINE_REF})")
lines.append(f"- Block: {TARGET_BLOCK}, c_proj.weight (3072, 768)")
lines.append(
    f"- Cycle 306 ref: Δy top1_E={CYCLE_306_B8_TOP1_E}, "
    f"stable rank={CYCLE_306_B8_DY_EFF_RANK}, "
    f"E_kept_top2 via W_proj={CYCLE_306_B8_E_KEPT_TOP2_WPROJ}"
)
lines.append("")
lines.append("## No-retrain PPL (eval slice, 100 seqs × 128 tok)")
lines.append("")
lines.append("| basis | r | PPL | Δfloat | %above |")
lines.append("|---|---:|---:|---:|---:|")
for basis in BASES:
    for r in RANKS:
        nr = noretrain_results[f"{basis}_r{r}"]
        lines.append(
            f"| {basis} | {r} | {fmt(nr['ppl'])} | "
            f"{nr['delta_float']:+.3f} | {nr['pct_above_float']:+.2f}% |"
        )
lines.append("")
lines.append("## Retrain best_mean (3 seeds, Adam LR=1e-4, 4000 steps, best-state restore)")
lines.append("")
lines.append("| basis | r | n_train | best_mean | best_std | best_min | best_max | Δfloat |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for basis in BASES:
    for r in RANKS:
        cell = retrain_cells[f"{basis}_r{r}"]
        agg = cell["aggregate"]
        lines.append(
            f"| {basis} | {r} | {cell['n_trainable']} | "
            f"{fmt(agg['best_mean'])} | {fmt(agg['best_std'])} | "
            f"{fmt(agg['best_min'])} | {fmt(agg['best_max'])} | "
            f"{agg['best_mean'] - float_ppl:+.3f} |"
        )
lines.append("")
lines.append("## Per-seed retrain detail")
lines.append("")
lines.append("| cell | seed | init | best | best_step | final | restored |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for basis in BASES:
    for r in RANKS:
        cell_key = f"{basis}_r{r}"
        for s in retrain_cells[cell_key]["seeds"]:
            lines.append(
                f"| {cell_key} | {s['seed']} | {fmt(s['init_ppl'])} | "
                f"{fmt(s['best_ppl'])} | {s['best_step']} | "
                f"{fmt(s['final_ppl_at_n_steps'])} | {fmt(s['restored_best_ppl'])} |"
            )
lines.append("")
lines.append("## Verdict logic (item 319 verbatim)")
lines.append("")
lines.append(f"- dy_r1 retrain best_mean = {fmt(dy_r1_best_mean)}")
lines.append(f"- CONFIRMED iff ≤ {CONFIRMED_RETRAIN_LE} (Δy basis is the right basis at b8)")
lines.append(f"- REFUTED iff ≥ {REFUTED_RETRAIN_GE} (b8 needs rank > 1 regardless of basis)")
lines.append(f"- **VERDICT: {verdict}**")
lines.append("")
lines.append("## Geometric construction recap")
lines.append("")
lines.append("- dy basis: A_y = W_proj @ Vh_y[:r].T frozen (3072, r); B_init = Vh_y[:r] (r, 768) trainable; bias trainable")
lines.append("- wproj basis (paired baseline, cycle 280-style): A_w = U_w[:, :r] * S_w[:r] frozen; B_w = Vh_w[:r] trainable; bias trainable")
lines.append("- W_new = A @ B replaces c_proj_8.weight in-place; rest of model frozen")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)
print("\nDone.", flush=True)
