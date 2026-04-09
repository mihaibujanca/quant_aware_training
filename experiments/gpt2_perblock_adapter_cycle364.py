#!/usr/bin/env python3
"""Cycle 355 / Item 364 — Per-block adapter sweep: is b2 special?

Tests whether topk+lstsq + h=64 adapter works on blocks beyond b2.
Blocks tested: [b1, b4, b7, b10] — spanning early/mid/late positions.
2 seeds, 1000 steps per block.

Prior result: cycle 350 single-block b2, h=64 adapter → delta ~1.94 PPL.
Question: is this b2-specific or universal?

Structure per block:
  Phase 1: topk+lstsq k=512 (6x compression) → measure compressed-only PPL
  Phase 2: freeze compressed proj, add h=64 adapter, train 1000 steps
           → measure PPL delta (compressed+adapter vs compressed-only)

Adapter: W1 (64, 512), b1 (64), W2 (768, 64), b2 (768)
         = 64*512 + 64 + 768*64 + 768 = 32768+64+49152+768 = 82752 params

Verdict (item 364):
  CONFIRMED if adapter delta > 0.5 PPL on >= 3/4 blocks
  REFUTED if only b2 benefits (0/4 blocks pass, delta <= 0.5)
  INCONCLUSIVE otherwise (1-2 blocks pass)
"""
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS = [1, 4, 7, 10]       # early / early-mid / mid-late / late
ADAPTER_DELTA_THRESHOLD = 0.5 # PPL improvement from adapter to count as "working"
N_BLOCKS_CONFIRMED = 3        # need >= 3/4 for CONFIRMED

HIDDEN_FRACTION = 1 / 6       # k=512 of 3072 (same as all prior adapter cycles)
ADAPTER_HIDDEN = 64

N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

N_STEPS_P1 = 500    # compressed-only retrain (shorter — just to set baseline)
N_STEPS_P2 = 1000   # adapter training
LR_P1 = 1e-4
LR_P2 = 1e-4
EVAL_EVERY = 100
SEEDS = [42, 137]
LOAD_SEED = 42

FLOAT_BASELINE_REF = 53.895
B2_ADAPTER_REF = 46.22   # cycle 350 h=64 b2 best

OUTPUT_PATH = "autoresearch/results/cycle_355_perblock_adapter.json"

torch.manual_seed(LOAD_SEED)
print(f"Device: {DEVICE}", flush=True)
print(f"Blocks to test: {BLOCKS}", flush=True)
print(f"Adapter hidden: {ADAPTER_HIDDEN}, k~{int(3072/6)}", flush=True)

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


def collect_block_activations(model, block_id, calib_ids, device):
    """Collect MLP hidden/output/preact for a single block."""
    data = {"input": [], "output": [], "preact": [], "hidden": []}

    def hook_fn(module, inputs, output):
        x = inputs[0].detach()
        z = module.c_fc(x).detach()
        h = F.gelu(z).detach()
        y = output.detach()
        data["input"].append(x.reshape(-1, x.shape[-1]).cpu())
        data["output"].append(y.reshape(-1, y.shape[-1]).cpu())
        data["preact"].append(z.reshape(-1, z.shape[-1]).cpu())
        data["hidden"].append(h.reshape(-1, h.shape[-1]).cpu())

    hook = model.transformer.h[block_id].mlp.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        for ids in calib_ids:
            ids_dev = ids.to(device) if ids.device.type != device else ids
            model(ids_dev)
    hook.remove()

    return {
        "input": torch.cat(data["input"], dim=0).float(),
        "output": torch.cat(data["output"], dim=0).float(),
        "preact": torch.cat(data["preact"], dim=0).float(),
        "hidden": torch.cat(data["hidden"], dim=0).float(),
    }


def topk_lstsq_compress(model, blk, acts, hidden_fraction):
    """Run topk channel selection + lstsq for c_proj. Returns compress_cache dict."""
    hidden = acts["hidden"]
    output = acts["output"]
    preact = acts["preact"]

    hidden_dim = hidden.shape[1]
    k = max(1, int(round(hidden_dim * hidden_fraction)))

    # Channel scoring (identical to c362/c354)
    std_h = hidden.std(dim=0)
    mean_h = hidden.mean(dim=0)
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    deriv = (0.5 * (1.0 + torch.erf(preact * inv_sqrt_2))
             + preact * torch.exp(-0.5 * preact.square()) * inv_sqrt_2pi)
    mean_deriv = deriv.mean(dim=0)

    c_proj = model.transformer.h[blk].mlp.c_proj
    c_proj_weight = c_proj.weight.detach().cpu().float()  # (3072, 768) Conv1D
    output_norms = c_proj_weight.norm(dim=1)              # (3072,)

    channel_scores = std_h * output_norms * (mean_h.abs() + mean_deriv.abs() + 1e-6)

    dead_mask = hidden.abs().amax(dim=0) <= 1e-6
    available = torch.arange(hidden_dim)
    live = available[~dead_mask]
    if live.numel() < k:
        live = available
    scores = channel_scores[live]
    topk_idx = torch.topk(scores, k=k, largest=True).indices
    selected = live[topk_idx].sort().values

    c_fc = model.transformer.h[blk].mlp.c_fc
    c_fc_weight_full = c_fc.weight.detach().cpu().float()  # (768, 3072) Conv1D
    c_fc_bias_full = c_fc.bias.detach().cpu().float()

    fc_w = c_fc_weight_full.T[selected]  # (k, 768)
    fc_b = c_fc_bias_full[selected]      # (k,)

    # lstsq for output projection
    hidden_sel = hidden[:, selected].float()
    design = torch.cat([hidden_sel, torch.ones(hidden_sel.shape[0], 1)], dim=1)
    solution = torch.linalg.lstsq(design, output.float()).solution
    proj_w = solution[:-1].T.contiguous()  # (768, k)
    proj_b = solution[-1].contiguous()     # (768,)

    with torch.no_grad():
        recon = hidden_sel @ proj_w.T + proj_b
        mse = F.mse_loss(recon, output).item()
        cos = F.cosine_similarity(recon.reshape(1, -1), output.reshape(1, -1)).item()

    return {
        "fc_w": fc_w, "fc_b": fc_b,
        "proj_w": proj_w, "proj_b": proj_b,
        "selected": selected, "k": k,
        "mse": mse, "cos": cos,
        "dead": int(dead_mask.sum().item()),
    }


class CompressedGPT2MLPWithAdapter(nn.Module):
    """Compressed MLP (topk+lstsq) with optional h-dim MLP adapter.

    Compressed path: frozen c_fc (k channels) → GELU → trainable c_proj (k→768)
    Adapter path:    GELU hidden (k) → W1 (h, k) → ReLU → W2 (768, h)
    output = compressed(x) + adapter(gelu_hidden)  [if adapter_active]
    """

    def __init__(self, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias,
                 adapter_hidden=0):
        super().__init__()
        k = c_fc_weight.shape[0]
        self.register_buffer("fc_weight", c_fc_weight.contiguous())  # (k, 768)
        self.register_buffer("fc_bias", c_fc_bias.contiguous())      # (k,)
        self.proj_weight = nn.Parameter(c_proj_weight.contiguous())  # (768, k)
        self.proj_bias = nn.Parameter(c_proj_bias.contiguous())      # (768,)

        self.adapter_active = adapter_hidden > 0
        if self.adapter_active:
            scale1 = 1.0 / math.sqrt(k)
            scale2 = 1.0 / math.sqrt(adapter_hidden)
            self.adapter_W1 = nn.Parameter(torch.randn(adapter_hidden, k) * scale1)
            self.adapter_b1 = nn.Parameter(torch.zeros(adapter_hidden))
            self.adapter_W2 = nn.Parameter(torch.randn(768, adapter_hidden) * scale2)
            self.adapter_b2 = nn.Parameter(torch.zeros(768))

    def forward(self, x):
        h = F.linear(x, self.fc_weight, self.fc_bias)
        h = F.gelu(h)
        out = F.linear(h, self.proj_weight, self.proj_bias)
        if self.adapter_active:
            a = F.relu(F.linear(h, self.adapter_W1, self.adapter_b1))
            out = out + F.linear(a, self.adapter_W2, self.adapter_b2)
        return out


# ── Float baseline ────────────────────────────────────────────────────────────
print("\nComputing float baseline PPL...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
float_ppl = eval_ppl(base_model, eval_ids)
print(f"Float PPL: {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from reference {FLOAT_BASELINE_REF} by > 0.5"
)

# ── Pre-compute topk+lstsq for all blocks ────────────────────────────────────
print("\nCollecting calibration activations for all target blocks...", flush=True)
calib_dev = [t.to(DEVICE) for t in calib_ids_cpu]
compress_cache = {}
for blk in BLOCKS:
    acts = collect_block_activations(base_model, blk, calib_dev, DEVICE)
    cc = topk_lstsq_compress(base_model, blk, acts, HIDDEN_FRACTION)
    compress_cache[blk] = cc
    print(f"  Block {blk}: k={cc['k']}, dead={cc['dead']}, "
          f"lstsq MSE={cc['mse']:.4f}, cos={cc['cos']:.6f}", flush=True)

del base_model
if DEVICE == "mps":
    torch.mps.empty_cache()

# Adapter param count
k_val = compress_cache[BLOCKS[0]]["k"]
n_adapter_per_block = (ADAPTER_HIDDEN * k_val + ADAPTER_HIDDEN
                       + 768 * ADAPTER_HIDDEN + 768)
print(f"\nAdapter params per block: {n_adapter_per_block:,} "
      f"(W1:{ADAPTER_HIDDEN}×{k_val} + W2:768×{ADAPTER_HIDDEN})", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Per-block experiment loop
# ══════════════════════════════════════════════════════════════════════════════
block_results = {}

for blk in BLOCKS:
    print(f"\n{'='*60}", flush=True)
    print(f"Block {blk}", flush=True)
    print(f"{'='*60}", flush=True)
    cc = compress_cache[blk]

    # ── Phase 1: compressed-only baseline ────────────────────────────────────
    print(f"  Phase 1: topk+lstsq compressed-only baseline ({N_STEPS_P1} steps)", flush=True)
    torch.manual_seed(LOAD_SEED)
    random.seed(LOAD_SEED)

    model_p1 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    comp_mlp_p1 = CompressedGPT2MLPWithAdapter(
        c_fc_weight=cc["fc_w"].to(device=DEVICE, dtype=torch.float32),
        c_fc_bias=cc["fc_b"].to(device=DEVICE, dtype=torch.float32),
        c_proj_weight=cc["proj_w"].to(device=DEVICE, dtype=torch.float32),
        c_proj_bias=cc["proj_b"].to(device=DEVICE, dtype=torch.float32),
        adapter_hidden=0,
    ).to(DEVICE)
    model_p1.transformer.h[blk].mlp = comp_mlp_p1

    init_ppl_p1 = eval_ppl(model_p1, eval_ids)
    print(f"  Phase 1 init PPL (post-lstsq, no retrain): {init_ppl_p1:.4f}", flush=True)

    # Freeze all except proj
    for p in model_p1.parameters():
        p.requires_grad_(False)
    comp_mlp_p1.proj_weight.requires_grad_(True)
    comp_mlp_p1.proj_bias.requires_grad_(True)
    trainable_p1 = [comp_mlp_p1.proj_weight, comp_mlp_p1.proj_bias]
    n_train_p1 = sum(p.numel() for p in trainable_p1)
    print(f"  Phase 1 trainable params: {n_train_p1:,}", flush=True)

    opt_p1 = torch.optim.Adam(trainable_p1, lr=LR_P1)
    calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]
    random.shuffle(calib_ids_dev)

    best_ppl_p1 = init_ppl_p1
    best_state_p1 = {
        "proj_w": comp_mlp_p1.proj_weight.detach().clone(),
        "proj_b": comp_mlp_p1.proj_bias.detach().clone(),
    }
    best_step_p1 = 0

    step = 0
    t0 = time.time()
    while step < N_STEPS_P1:
        for ids in calib_ids_dev:
            if step >= N_STEPS_P1:
                break
            model_p1.train()
            opt_p1.zero_grad()
            out = model_p1(ids, labels=ids)
            out.loss.backward()
            opt_p1.step()
            step += 1
            if step % EVAL_EVERY == 0:
                ppl_e = eval_ppl(model_p1, eval_ids)
                if ppl_e < best_ppl_p1:
                    best_ppl_p1 = ppl_e
                    best_step_p1 = step
                    best_state_p1 = {
                        "proj_w": comp_mlp_p1.proj_weight.detach().clone(),
                        "proj_b": comp_mlp_p1.proj_bias.detach().clone(),
                    }
                if step % 200 == 0:
                    print(f"    P1 step {step}: {ppl_e:.2f} best={best_ppl_p1:.2f}@{best_step_p1}",
                          flush=True)

    # Restore best
    with torch.no_grad():
        comp_mlp_p1.proj_weight.data.copy_(best_state_p1["proj_w"])
        comp_mlp_p1.proj_bias.data.copy_(best_state_p1["proj_b"])
    compressed_only_ppl = eval_ppl(model_p1, eval_ids)
    dt_p1 = time.time() - t0
    print(f"  Phase 1: init={init_ppl_p1:.2f} best={best_ppl_p1:.2f} "
          f"restored={compressed_only_ppl:.2f} ({dt_p1:.0f}s)", flush=True)

    # Save retrained proj for Phase 2
    retrained_proj_w = best_state_p1["proj_w"].cpu().clone()
    retrained_proj_b = best_state_p1["proj_b"].cpu().clone()

    del model_p1, comp_mlp_p1, opt_p1
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # ── Phase 2: adapter training (2 seeds) ──────────────────────────────────
    print(f"\n  Phase 2: adapter h={ADAPTER_HIDDEN} ({N_STEPS_P2} steps, {len(SEEDS)} seeds)",
          flush=True)

    seed_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        random.seed(seed)

        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        comp_mlp = CompressedGPT2MLPWithAdapter(
            c_fc_weight=cc["fc_w"].to(device=DEVICE, dtype=torch.float32),
            c_fc_bias=cc["fc_b"].to(device=DEVICE, dtype=torch.float32),
            c_proj_weight=retrained_proj_w.to(device=DEVICE, dtype=torch.float32),
            c_proj_bias=retrained_proj_b.to(device=DEVICE, dtype=torch.float32),
            adapter_hidden=ADAPTER_HIDDEN,
        ).to(DEVICE)
        model.transformer.h[blk].mlp = comp_mlp

        init_ppl = eval_ppl(model, eval_ids)

        # Freeze all, unfreeze adapter only
        for p in model.parameters():
            p.requires_grad_(False)
        comp_mlp.adapter_W1.requires_grad_(True)
        comp_mlp.adapter_b1.requires_grad_(True)
        comp_mlp.adapter_W2.requires_grad_(True)
        comp_mlp.adapter_b2.requires_grad_(True)
        trainable = [comp_mlp.adapter_W1, comp_mlp.adapter_b1,
                     comp_mlp.adapter_W2, comp_mlp.adapter_b2]
        n_train = sum(p.numel() for p in trainable)
        assert n_train == n_adapter_per_block, f"{n_train} != {n_adapter_per_block}"

        opt = torch.optim.Adam(trainable, lr=LR_P2)
        calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]
        random.shuffle(calib_ids_dev)

        best_ppl = init_ppl
        best_step = 0
        best_state = {
            "W1": comp_mlp.adapter_W1.detach().clone(),
            "b1": comp_mlp.adapter_b1.detach().clone(),
            "W2": comp_mlp.adapter_W2.detach().clone(),
            "b2": comp_mlp.adapter_b2.detach().clone(),
        }
        history = [{"step": 0, "eval_ppl": float(init_ppl)}]

        step = 0
        t0 = time.time()
        while step < N_STEPS_P2:
            for ids in calib_ids_dev:
                if step >= N_STEPS_P2:
                    break
                model.train()
                opt.zero_grad()
                out = model(ids, labels=ids)
                out.loss.backward()
                opt.step()
                step += 1
                if step % EVAL_EVERY == 0:
                    ppl_e = eval_ppl(model, eval_ids)
                    history.append({"step": step, "eval_ppl": float(ppl_e)})
                    if ppl_e < best_ppl:
                        best_ppl = ppl_e
                        best_step = step
                        best_state = {
                            "W1": comp_mlp.adapter_W1.detach().clone(),
                            "b1": comp_mlp.adapter_b1.detach().clone(),
                            "W2": comp_mlp.adapter_W2.detach().clone(),
                            "b2": comp_mlp.adapter_b2.detach().clone(),
                        }
                    if step % 500 == 0:
                        print(f"    seed {seed} step {step}: "
                              f"{ppl_e:.2f} best={best_ppl:.2f}@{best_step}", flush=True)

        # Restore best
        with torch.no_grad():
            comp_mlp.adapter_W1.data.copy_(best_state["W1"])
            comp_mlp.adapter_b1.data.copy_(best_state["b1"])
            comp_mlp.adapter_W2.data.copy_(best_state["W2"])
            comp_mlp.adapter_b2.data.copy_(best_state["b2"])
        restored_ppl = eval_ppl(model, eval_ids)
        dt = time.time() - t0

        adapter_delta = compressed_only_ppl - restored_ppl
        print(f"    seed {seed}: init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
              f"restored={restored_ppl:.2f} delta={adapter_delta:+.2f} ({dt:.0f}s)", flush=True)

        seed_results.append({
            "seed": seed,
            "init_ppl": float(init_ppl),
            "best_ppl": float(best_ppl),
            "best_step": int(best_step),
            "restored_ppl": float(restored_ppl),
            "adapter_delta": float(adapter_delta),
            "history": history,
        })

        del model, comp_mlp, opt
        if DEVICE == "mps":
            torch.mps.empty_cache()

    # Summarize block
    mean_adapter_delta = sum(s["adapter_delta"] for s in seed_results) / len(seed_results)
    best_adapter_delta = max(s["adapter_delta"] for s in seed_results)
    block_passes = mean_adapter_delta > ADAPTER_DELTA_THRESHOLD

    print(f"\n  Block {blk} summary:", flush=True)
    print(f"    compressed-only PPL: {compressed_only_ppl:.4f}", flush=True)
    print(f"    adapter delta (mean): {mean_adapter_delta:+.4f}", flush=True)
    print(f"    adapter delta (best): {best_adapter_delta:+.4f}", flush=True)
    print(f"    passes threshold (>{ADAPTER_DELTA_THRESHOLD}): {block_passes}", flush=True)

    block_results[blk] = {
        "compressed_only_ppl": float(compressed_only_ppl),
        "phase1_init_ppl": float(init_ppl_p1),
        "phase1_best_ppl": float(best_ppl_p1),
        "mean_adapter_delta": float(mean_adapter_delta),
        "best_adapter_delta": float(best_adapter_delta),
        "passes": bool(block_passes),
        "seeds": seed_results,
        "lstsq_mse": cc["mse"],
        "lstsq_cos": cc["cos"],
        "k": cc["k"],
    }


# ── Final verdict ─────────────────────────────────────────────────────────────
n_passing = sum(1 for v in block_results.values() if v["passes"])

if n_passing >= N_BLOCKS_CONFIRMED:
    verdict = "CONFIRMED"
elif n_passing == 0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"Float PPL:          {float_ppl:.4f}", flush=True)
print(f"Adapter threshold:  delta > {ADAPTER_DELTA_THRESHOLD} PPL", flush=True)
print(f"Blocks passing:     {n_passing}/{len(BLOCKS)}", flush=True)
print(f"", flush=True)
print(f"{'Block':>6} | {'Comp-only':>10} | {'Delta mean':>11} | {'Delta best':>11} | {'Pass':>5}", flush=True)
print(f"{'-'*6}-+-{'-'*10}-+-{'-'*11}-+-{'-'*11}-+-{'-'*5}", flush=True)
for blk in BLOCKS:
    r = block_results[blk]
    print(f"  b{blk:>2}  | {r['compressed_only_ppl']:>10.4f} | "
          f"{r['mean_adapter_delta']:>+11.4f} | "
          f"{r['best_adapter_delta']:>+11.4f} | {'YES' if r['passes'] else 'no':>5}", flush=True)
print(f"", flush=True)
print(f"VERDICT: {verdict}", flush=True)
print(f"  CONFIRMED: delta > {ADAPTER_DELTA_THRESHOLD} on >= {N_BLOCKS_CONFIRMED}/4 blocks", flush=True)
print(f"  REFUTED:   0/4 blocks pass (b2 was special)", flush=True)
print(f"  INCONCLUSIVE: 1-2 blocks pass", flush=True)
print(f"  b2 reference (cycle 350): compressed+adapter best PPL {B2_ADAPTER_REF}", flush=True)

# ── Save JSON ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

result = {
    "cycle": 355,
    "item": 364,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "n_passing": n_passing,
    "n_blocks": len(BLOCKS),
    "blocks_tested": BLOCKS,
    "adapter_delta_threshold": ADAPTER_DELTA_THRESHOLD,
    "per_block": {
        str(blk): block_results[blk]
        for blk in BLOCKS
    },
    "config": {
        "BLOCKS": BLOCKS,
        "HIDDEN_FRACTION": HIDDEN_FRACTION,
        "ADAPTER_HIDDEN": ADAPTER_HIDDEN,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE,
        "N_STEPS_P1": N_STEPS_P1,
        "LR_P1": float(LR_P1),
        "N_STEPS_P2": N_STEPS_P2,
        "LR_P2": float(LR_P2),
        "EVAL_EVERY": EVAL_EVERY,
        "SEEDS": SEEDS,
        "ADAPTER_DELTA_THRESHOLD": ADAPTER_DELTA_THRESHOLD,
    },
    "references": {
        "float_ppl_ref": FLOAT_BASELINE_REF,
        "b2_adapter_best_c350": B2_ADAPTER_REF,
        "depends_on_item": 359,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved results to {OUTPUT_PATH}", flush=True)
print("Done.", flush=True)
