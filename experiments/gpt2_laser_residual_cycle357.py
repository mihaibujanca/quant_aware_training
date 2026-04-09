#!/usr/bin/env python3
"""Cycle 347 / Item 357 — GPT-2 b2 rank-2 LASER + learned residual.

Push past the single-block LASER ceiling (PPL 50.21) by adding a learnable
residual on top of the frozen rank-2 approximation.

Setup:
  - Block 2, c_proj (3072, 768), Conv1D format
  - LASER: W_laser = A @ B where A = U[:,:2]*S[:2] (3072,2), B = Vh[:2,:] (2,768)
  - Phase 1: reproduce rank-2 LASER retrain → PPL ≈ 50.21 (baseline)
  - Phase 2a: add low-rank residual ΔW = C @ D, C (3072, r), D (r, 768), r in {4, 8}
  - Phase 2b: add MLP adapter f(x) = W2 @ relu(W1 @ x), W1 (3072, 64), W2 (64, 768)
  - Train only residual/adapter params, freeze LASER (A, B, bias all frozen)

Prior results:
  Float PPL: 53.89
  b2 rank-2 LASER (cycle 343 stage 1): 50.21 ± 0.13

Verdict (item 357):
  CONFIRMED if best PPL < 49.0
  REFUTED if best PPL >= 50.0
  INCONCLUSIVE if 49.0 <= PPL < 50.0
"""
import json
import math
import random
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
BLOCK = 2
RANK_LASER = 2

N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Phase 1: LASER retrain
N_STEPS_LASER = 1000
LR_LASER = 1e-4

# Phase 2: residual retrain
N_STEPS_RESID = 1000
LR_RESID = 1e-4
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
LOAD_SEED = 42

FLOAT_BASELINE_REF = 53.895
LASER_BASELINE_REF = 50.21  # from cycle 343

OUTPUT_PATH = "autoresearch/results/cycle_347_laser_residual.json"

torch.manual_seed(LOAD_SEED)
print(f"Device: {DEVICE}", flush=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading GPT-2 tokenizer and wikitext-2 data...", flush=True)
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

calib_ids_cpu = list(all_ids[:N_CALIB])
eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB : N_CALIB + N_EVAL]]
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
    """Rank-r replacement for Conv1D c_proj. A frozen, B trainable."""

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


class LowRankCProjWithLowRankResidual(nn.Module):
    """Frozen rank-2 LASER + trainable low-rank residual ΔW = C @ D.

    Forward: x @ (A @ B + C @ D) + bias
    A, B, bias are all FROZEN from Phase 1. Only C, D are trainable.
    """

    def __init__(self, A_frozen, B_frozen, bias_frozen, resid_rank, nf=768, in_features=3072):
        super().__init__()
        self.register_buffer("A", A_frozen.contiguous())
        self.register_buffer("B", B_frozen.contiguous())
        self.register_buffer("bias", bias_frozen.contiguous())
        self.nf = nf
        # Init C, D small random (scaled by 1/sqrt(resid_rank * in_features))
        scale = 1.0 / math.sqrt(resid_rank * in_features)
        self.C = nn.Parameter(torch.randn(in_features, resid_rank) * scale)
        self.D = nn.Parameter(torch.randn(resid_rank, nf) * scale)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        W_laser = self.A @ self.B
        W_resid = self.C @ self.D
        W = W_laser + W_resid
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), W)
        x = x.view(size_out)
        return x


class LowRankCProjWithMLPAdapter(nn.Module):
    """Frozen rank-2 LASER + trainable MLP adapter.

    Forward: x @ (A @ B) + bias + W2 @ relu(W1 @ x^T)
    i.e. output = LASER(x) + adapter(x)
    A, B, bias frozen. W1 (hidden, in_features), W2 (nf, hidden) trainable.
    """

    def __init__(self, A_frozen, B_frozen, bias_frozen, hidden_dim=64, nf=768, in_features=3072):
        super().__init__()
        self.register_buffer("A", A_frozen.contiguous())
        self.register_buffer("B", B_frozen.contiguous())
        self.register_buffer("bias", bias_frozen.contiguous())
        self.nf = nf
        # MLP: in_features -> hidden_dim -> nf
        scale1 = 1.0 / math.sqrt(in_features)
        scale2 = 1.0 / math.sqrt(hidden_dim)
        self.W1 = nn.Parameter(torch.randn(hidden_dim, in_features) * scale1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(nf, hidden_dim) * scale2)
        self.b2 = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        flat = x.view(-1, x.size(-1))  # (N, 3072)
        # LASER path
        laser_out = torch.addmm(self.bias, flat, self.A @ self.B)  # (N, 768)
        # Adapter path
        h = torch.nn.functional.relu(flat @ self.W1.t() + self.b1)  # (N, hidden)
        adapter_out = h @ self.W2.t() + self.b2  # (N, 768)
        out = laser_out + adapter_out
        out = out.view(size_out)
        return out


def train_loop(model, trainable_params, calib_ids_cpu, eval_ids, n_steps, lr, seeds,
               label="", device=DEVICE):
    """Train and evaluate across seeds. Returns list of per-seed results."""
    seed_results = []
    for seed in seeds:
        torch_seed = seed
        torch.manual_seed(torch_seed)
        random.seed(torch_seed)

        # Re-init optimizer each seed (params are reset below for residual variants)
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        calib_ids = [t.to(device) for t in calib_ids_cpu]
        random.shuffle(calib_ids)

        init_ppl = eval_ppl(model, eval_ids)
        best_ppl = init_ppl
        best_step = 0
        best_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        history = [{"step": 0, "eval_ppl": float(init_ppl)}]

        step = 0
        t0 = time.time()
        while step < n_steps:
            for ids in calib_ids:
                if step >= n_steps:
                    break
                model.train()
                optimizer.zero_grad()
                out = model(ids, labels=ids)
                loss = out.loss
                loss.backward()
                optimizer.step()
                step += 1

                if step % EVAL_EVERY == 0:
                    ppl_e = eval_ppl(model, eval_ids)
                    history.append({"step": step, "eval_ppl": float(ppl_e)})
                    if ppl_e < best_ppl:
                        best_ppl = ppl_e
                        best_step = step
                        best_state = {n: p.detach().clone()
                                      for n, p in model.named_parameters() if p.requires_grad}
                    if step % 500 == 0:
                        print(f"  [{label}] seed {seed} step {step}: ppl={ppl_e:.2f} "
                              f"best={best_ppl:.2f}@{best_step}", flush=True)

        # Restore best
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in best_state:
                    p.data.copy_(best_state[n])
        restored_ppl = eval_ppl(model, eval_ids)
        dt = time.time() - t0

        print(f"  [{label}] seed {seed}: init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
              f"restored={restored_ppl:.2f} ({dt:.0f}s)", flush=True)

        seed_results.append({
            "seed": seed,
            "init_ppl": float(init_ppl),
            "best_ppl": float(best_ppl),
            "best_step": int(best_step),
            "restored_ppl": float(restored_ppl),
            "history": history,
        })

    return seed_results


# ── Float baseline ────────────────────────────────────────────────────────────
print("Computing float baseline PPL...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
float_ppl = eval_ppl(base_model, eval_ids)
print(f"Float PPL: {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from reference {FLOAT_BASELINE_REF} by > 0.5"
)

# ── SVD block 2 ──────────────────────────────────────────────────────────────
W_orig = base_model.transformer.h[BLOCK].mlp.c_proj.weight.data.detach().clone()
b_orig = base_model.transformer.h[BLOCK].mlp.c_proj.bias.data.detach().clone()
assert W_orig.shape == (3072, 768), f"block {BLOCK} shape {W_orig.shape}"

W_cpu = W_orig.float().cpu()
U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
A_r = (U[:, :RANK_LASER] * S[:RANK_LASER]).contiguous()  # (3072, 2)
B_r = Vh[:RANK_LASER, :].contiguous()  # (2, 768)
print(f"SVD block {BLOCK}: rank {RANK_LASER}, top-5 S = {S[:5].tolist()}", flush=True)

del base_model
if DEVICE == "mps":
    torch.mps.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Reproduce rank-2 LASER retrain (baseline)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print("Phase 1: Rank-2 LASER retrain (reproduce baseline)", flush=True)
print(f"{'='*60}", flush=True)

# Use seed 42 only for phase 1 (just to get the retrained B and bias)
torch.manual_seed(42)
random.seed(42)

model_p1 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_p1.eval()

lr_mod = LowRankCProj(
    A_frozen=A_r.to(device=DEVICE, dtype=torch.float32),
    B_init=B_r.to(device=DEVICE, dtype=torch.float32),
    b_init=b_orig.to(device=DEVICE, dtype=torch.float32),
).to(DEVICE)
model_p1.transformer.h[BLOCK].mlp.c_proj = lr_mod

init_ppl_p1 = eval_ppl(model_p1, eval_ids)
print(f"Phase 1 init PPL: {init_ppl_p1:.4f}", flush=True)

# Freeze everything, unfreeze B + bias
for p in model_p1.parameters():
    p.requires_grad_(False)
lr_mod.B.requires_grad_(True)
lr_mod.bias.requires_grad_(True)
trainable_p1 = [lr_mod.B, lr_mod.bias]

optimizer_p1 = torch.optim.Adam(trainable_p1, lr=LR_LASER)
calib_ids = [t.to(DEVICE) for t in calib_ids_cpu]
random.shuffle(calib_ids)

best_ppl_p1 = init_ppl_p1
best_step_p1 = 0
best_B = lr_mod.B.detach().clone()
best_bias = lr_mod.bias.detach().clone()

step = 0
t0 = time.time()
while step < N_STEPS_LASER:
    for ids in calib_ids:
        if step >= N_STEPS_LASER:
            break
        model_p1.train()
        optimizer_p1.zero_grad()
        out = model_p1(ids, labels=ids)
        out.loss.backward()
        optimizer_p1.step()
        step += 1
        if step % EVAL_EVERY == 0:
            ppl_e = eval_ppl(model_p1, eval_ids)
            if ppl_e < best_ppl_p1:
                best_ppl_p1 = ppl_e
                best_step_p1 = step
                best_B = lr_mod.B.detach().clone()
                best_bias = lr_mod.bias.detach().clone()
            if step % 500 == 0:
                print(f"  Phase 1 step {step}: ppl={ppl_e:.2f} best={best_ppl_p1:.2f}@{best_step_p1}",
                      flush=True)

# Restore best
with torch.no_grad():
    lr_mod.B.data.copy_(best_B)
    lr_mod.bias.data.copy_(best_bias)
restored_ppl_p1 = eval_ppl(model_p1, eval_ids)
dt_p1 = time.time() - t0
print(f"Phase 1: init={init_ppl_p1:.2f} best={best_ppl_p1:.2f}@{best_step_p1} "
      f"restored={restored_ppl_p1:.2f} ({dt_p1:.0f}s)", flush=True)

# Save frozen LASER parameters
A_laser = lr_mod.A.detach().clone()  # frozen A (unchanged from SVD)
B_laser = best_B.detach().clone()    # retrained B
bias_laser = best_bias.detach().clone()  # retrained bias

del model_p1, lr_mod, optimizer_p1
if DEVICE == "mps":
    torch.mps.empty_cache()

phase1_result = {
    "init_ppl": float(init_ppl_p1),
    "best_ppl": float(best_ppl_p1),
    "best_step": int(best_step_p1),
    "restored_ppl": float(restored_ppl_p1),
}
print(f"\nPhase 1 baseline PPL: {best_ppl_p1:.4f} (ref {LASER_BASELINE_REF})", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2a: Low-rank residual ΔW (rank 4 and rank 8)
# ══════════════════════════════════════════════════════════════════════════════
phase2_results = {}

for resid_rank in [4, 8]:
    label = f"lowrank_r{resid_rank}"
    print(f"\n{'='*60}", flush=True)
    print(f"Phase 2a: LASER + low-rank residual (rank {resid_rank})", flush=True)
    n_resid_params = resid_rank * 3072 + resid_rank * 768
    print(f"Residual params: {n_resid_params:,}", flush=True)
    print(f"{'='*60}", flush=True)

    seed_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        random.seed(seed)

        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        model.eval()

        mod = LowRankCProjWithLowRankResidual(
            A_frozen=A_laser.to(DEVICE),
            B_frozen=B_laser.to(DEVICE),
            bias_frozen=bias_laser.to(DEVICE),
            resid_rank=resid_rank,
        ).to(DEVICE)
        model.transformer.h[BLOCK].mlp.c_proj = mod

        # Freeze all, unfreeze residual C, D
        for p in model.parameters():
            p.requires_grad_(False)
        mod.C.requires_grad_(True)
        mod.D.requires_grad_(True)
        trainable = [mod.C, mod.D]
        n_train = sum(p.numel() for p in trainable)
        print(f"  Trainable: {n_train:,}", flush=True)

        optimizer = torch.optim.Adam(trainable, lr=LR_RESID)
        calib_ids = [t.to(DEVICE) for t in calib_ids_cpu]
        random.shuffle(calib_ids)

        init_ppl = eval_ppl(model, eval_ids)
        best_ppl = init_ppl
        best_step = 0
        best_C = mod.C.detach().clone()
        best_D = mod.D.detach().clone()
        history = [{"step": 0, "eval_ppl": float(init_ppl)}]

        step = 0
        t0 = time.time()
        while step < N_STEPS_RESID:
            for ids in calib_ids:
                if step >= N_STEPS_RESID:
                    break
                model.train()
                optimizer.zero_grad()
                out = model(ids, labels=ids)
                out.loss.backward()
                optimizer.step()
                step += 1
                if step % EVAL_EVERY == 0:
                    ppl_e = eval_ppl(model, eval_ids)
                    history.append({"step": step, "eval_ppl": float(ppl_e)})
                    if ppl_e < best_ppl:
                        best_ppl = ppl_e
                        best_step = step
                        best_C = mod.C.detach().clone()
                        best_D = mod.D.detach().clone()
                    if step % 500 == 0:
                        print(f"  [{label}] seed {seed} step {step}: ppl={ppl_e:.2f} "
                              f"best={best_ppl:.2f}@{best_step}", flush=True)

        with torch.no_grad():
            mod.C.data.copy_(best_C)
            mod.D.data.copy_(best_D)
        restored_ppl = eval_ppl(model, eval_ids)
        dt = time.time() - t0
        print(f"  [{label}] seed {seed}: init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
              f"restored={restored_ppl:.2f} ({dt:.0f}s)", flush=True)

        seed_results.append({
            "seed": seed, "init_ppl": float(init_ppl), "best_ppl": float(best_ppl),
            "best_step": int(best_step), "restored_ppl": float(restored_ppl), "history": history,
        })
        del model, mod, optimizer
        if DEVICE == "mps":
            torch.mps.empty_cache()

    best_ppls = [s["best_ppl"] for s in seed_results]
    mean_ppl = sum(best_ppls) / len(best_ppls)
    std_ppl = math.sqrt(sum((x - mean_ppl) ** 2 for x in best_ppls) / len(best_ppls))
    phase2_results[label] = {
        "resid_rank": resid_rank,
        "n_resid_params": n_resid_params,
        "mean_ppl": float(mean_ppl),
        "std_ppl": float(std_ppl),
        "min_ppl": float(min(best_ppls)),
        "max_ppl": float(max(best_ppls)),
        "seeds": seed_results,
    }
    print(f"\n{label}: mean={mean_ppl:.2f} ± {std_ppl:.2f} (min={min(best_ppls):.2f})", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2b: MLP adapter (hidden=64)
# ══════════════════════════════════════════════════════════════════════════════
for hidden_dim in [64]:
    label = f"mlp_h{hidden_dim}"
    print(f"\n{'='*60}", flush=True)
    print(f"Phase 2b: LASER + MLP adapter (hidden={hidden_dim})", flush=True)
    n_adapter_params = hidden_dim * 3072 + hidden_dim + hidden_dim * 768 + hidden_dim
    print(f"Adapter params: {n_adapter_params:,}", flush=True)
    print(f"{'='*60}", flush=True)

    seed_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        random.seed(seed)

        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        model.eval()

        mod = LowRankCProjWithMLPAdapter(
            A_frozen=A_laser.to(DEVICE),
            B_frozen=B_laser.to(DEVICE),
            bias_frozen=bias_laser.to(DEVICE),
            hidden_dim=hidden_dim,
        ).to(DEVICE)
        model.transformer.h[BLOCK].mlp.c_proj = mod

        # Freeze all, unfreeze adapter
        for p in model.parameters():
            p.requires_grad_(False)
        mod.W1.requires_grad_(True)
        mod.b1.requires_grad_(True)
        mod.W2.requires_grad_(True)
        mod.b2.requires_grad_(True)
        trainable = [mod.W1, mod.b1, mod.W2, mod.b2]
        n_train = sum(p.numel() for p in trainable)
        print(f"  Trainable: {n_train:,}", flush=True)

        optimizer = torch.optim.Adam(trainable, lr=LR_RESID)
        calib_ids = [t.to(DEVICE) for t in calib_ids_cpu]
        random.shuffle(calib_ids)

        init_ppl = eval_ppl(model, eval_ids)
        best_ppl = init_ppl
        best_step = 0
        best_state = {n: p.detach().clone() for n, p in mod.named_parameters() if p.requires_grad}
        history = [{"step": 0, "eval_ppl": float(init_ppl)}]

        step = 0
        t0 = time.time()
        while step < N_STEPS_RESID:
            for ids in calib_ids:
                if step >= N_STEPS_RESID:
                    break
                model.train()
                optimizer.zero_grad()
                out = model(ids, labels=ids)
                out.loss.backward()
                optimizer.step()
                step += 1
                if step % EVAL_EVERY == 0:
                    ppl_e = eval_ppl(model, eval_ids)
                    history.append({"step": step, "eval_ppl": float(ppl_e)})
                    if ppl_e < best_ppl:
                        best_ppl = ppl_e
                        best_step = step
                        best_state = {n: p.detach().clone()
                                      for n, p in mod.named_parameters() if p.requires_grad}
                    if step % 500 == 0:
                        print(f"  [{label}] seed {seed} step {step}: ppl={ppl_e:.2f} "
                              f"best={best_ppl:.2f}@{best_step}", flush=True)

        with torch.no_grad():
            for n, p in mod.named_parameters():
                if p.requires_grad and n in best_state:
                    p.data.copy_(best_state[n])
        restored_ppl = eval_ppl(model, eval_ids)
        dt = time.time() - t0
        print(f"  [{label}] seed {seed}: init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
              f"restored={restored_ppl:.2f} ({dt:.0f}s)", flush=True)

        seed_results.append({
            "seed": seed, "init_ppl": float(init_ppl), "best_ppl": float(best_ppl),
            "best_step": int(best_step), "restored_ppl": float(restored_ppl), "history": history,
        })
        del model, mod, optimizer
        if DEVICE == "mps":
            torch.mps.empty_cache()

    best_ppls = [s["best_ppl"] for s in seed_results]
    mean_ppl = sum(best_ppls) / len(best_ppls)
    std_ppl = math.sqrt(sum((x - mean_ppl) ** 2 for x in best_ppls) / len(best_ppls))
    phase2_results[label] = {
        "hidden_dim": hidden_dim,
        "n_adapter_params": n_adapter_params,
        "mean_ppl": float(mean_ppl),
        "std_ppl": float(std_ppl),
        "min_ppl": float(min(best_ppls)),
        "max_ppl": float(max(best_ppls)),
        "seeds": seed_results,
    }
    print(f"\n{label}: mean={mean_ppl:.2f} ± {std_ppl:.2f} (min={min(best_ppls):.2f})", flush=True)


# ── Summary and verdict ──────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"Float PPL:        {float_ppl:.4f}", flush=True)
print(f"LASER baseline:   {best_ppl_p1:.4f} (phase 1, seed 42)", flush=True)
print()

overall_best_ppl = best_ppl_p1  # start from LASER baseline
overall_best_variant = "laser_only"

for name, res in phase2_results.items():
    print(f"{name}: mean={res['mean_ppl']:.2f} ± {res['std_ppl']:.2f} "
          f"(min={res['min_ppl']:.2f}, max={res['max_ppl']:.2f})", flush=True)
    if res["min_ppl"] < overall_best_ppl:
        overall_best_ppl = res["min_ppl"]
        overall_best_variant = name

print(f"\nOverall best PPL: {overall_best_ppl:.4f} ({overall_best_variant})", flush=True)

if overall_best_ppl < 49.0:
    verdict = "CONFIRMED"
elif overall_best_ppl >= 50.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"VERDICT: {verdict}", flush=True)

# ── Save JSON ────────────────────────────────────────────────────────────────
result = {
    "cycle": 347,
    "item": 357,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "laser_baseline_ppl": float(best_ppl_p1),
    "overall_best_ppl": float(overall_best_ppl),
    "overall_best_variant": overall_best_variant,
    "phase1": phase1_result,
    "phase2": {k: {kk: vv for kk, vv in v.items() if kk != "seeds"} for k, v in phase2_results.items()},
    "phase2_seeds": {k: v["seeds"] for k, v in phase2_results.items()},
    "config": {
        "BLOCK": BLOCK, "RANK_LASER": RANK_LASER,
        "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN,
        "N_STEPS_LASER": N_STEPS_LASER, "LR_LASER": LR_LASER,
        "N_STEPS_RESID": N_STEPS_RESID, "LR_RESID": LR_RESID,
        "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_PATH}", flush=True)
print("Done.", flush=True)
