#!/usr/bin/env python3
"""Cycle 379 / Item 387 — Compression-accuracy Pareto curve (k × adapter h sweep).

Sweep grid (8 configs):
  - Float (k=3072):     h ∈ {0, 32, 64, 128} — 4 configs
  - Compressed (k=512): h ∈ {0, 32, 64, 128} — 4 configs

Blocks: [b2, b1, b10], single seed 42.
Train on wikitext-2-val, eval on wikitext-2-test.
For h=0: no training, just eval.

Pareto comparison is on FFN params only (not total model params).

Confirmed if: any compressed config within 2 PPL of float+adapter at ≤80% FFN params.
Refuted if: all compressed configs Pareto-dominated by float.
"""
import json
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS = [2, 1, 10]
HIDDEN_DIM = 768
FFN_DIM = 3072
K_FLOAT = 3072
K_COMP = 512
H_VALUES = [0, 32, 64, 128]
LR = 1e-4
N_STEPS = 2000
SEED = 42
N_CALIB = 200
N_EVAL = 200
SEQ_LEN = 128
EVAL_EVERY = 100

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_JSON = "autoresearch/results/cycle_379_pareto_sweep.json"
OUTPUT_PLOT = "autoresearch/results/cycle_379_pareto.png"

torch.manual_seed(SEED)
random.seed(SEED)
print(f"Device: {DEVICE}", flush=True)


# ── FFN param counting ────────────────────────────────────────────────────────
def ffn_params_per_block(k, h):
    """Total FFN params for one block at hidden size k and adapter h."""
    # fc: (k, 768) + (k,); proj: (768, k) + (768,)
    base = 2 * HIDDEN_DIM * k + k + HIDDEN_DIM
    if h > 0:
        # Compressed adapter input is from GELU hidden dim=k; Float adapter input is GELU hidden dim=k (float: k=FFN_DIM=3072)
        adapter = k * h + h + HIDDEN_DIM * h + HIDDEN_DIM
        return base + adapter
    return base


def total_ffn_params(k, h, n_blocks=3):
    return ffn_params_per_block(k, h) * n_blocks


# Spot-check: k=512, h=64
_check = 2 * 768 * 512 + 512 + 768 + 512 * 64 + 64 + 768 * 64 + 768
assert ffn_params_per_block(512, 64) == _check, f"{ffn_params_per_block(512, 64)} != {_check}"
print(f"Param check k=512,h=64 per block: {ffn_params_per_block(512, 64)} (expected {_check})", flush=True)
print(f"Total for 3 blocks: {total_ffn_params(512, 64)}", flush=True)


# ── Data loading ──────────────────────────────────────────────────────────────
print("Loading tokenizer and wikitext-2...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def tokenize_split(split_name, n_sequences):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split_name)
    texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
    ids_list = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
        if enc["input_ids"].shape[1] >= SEQ_LEN:
            ids_list.append(enc["input_ids"][:, :SEQ_LEN])
        if len(ids_list) >= n_sequences:
            break
    return ids_list


val_ids_cpu = tokenize_split("validation", N_CALIB + 50)
calib_ids_cpu = val_ids_cpu[:N_CALIB]
val_eval_ids = [t.to(DEVICE) for t in val_ids_cpu[N_CALIB: N_CALIB + 50]]
test_ids_cpu = tokenize_split("test", N_EVAL)
test_eval_ids = [t.to(DEVICE) for t in test_ids_cpu[:N_EVAL]]
print(f"Val calib: {len(calib_ids_cpu)}, Val eval: {len(val_eval_ids)}, Test: {len(test_eval_ids)}", flush=True)


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


def run_steps_with_val_selection(model, trainable_params, n_steps, calib_dev, val_ids, label=""):
    opt = torch.optim.Adam(trainable_params, lr=LR)
    shuffled = list(calib_dev)
    random.shuffle(shuffled)

    init_ppl = eval_ppl(model, val_ids)
    best_val_ppl = init_ppl
    best_state = {id(p): p.detach().clone() for p in trainable_params}
    print(f"  {label} init val PPL: {init_ppl:.4f}", flush=True)

    step = 0
    while step < n_steps:
        for ids in shuffled:
            if step >= n_steps:
                break
            model.train()
            opt.zero_grad()
            model(ids, labels=ids).loss.backward()
            opt.step()
            step += 1
            if step % EVAL_EVERY == 0:
                ppl_e = eval_ppl(model, val_ids)
                if ppl_e < best_val_ppl:
                    best_val_ppl = ppl_e
                    best_state = {id(p): p.detach().clone() for p in trainable_params}
                if step % 500 == 0:
                    print(f"  {label} step {step}/{n_steps}: val_ppl={ppl_e:.2f} "
                          f"best_val={best_val_ppl:.2f}", flush=True)

    with torch.no_grad():
        for p in trainable_params:
            p.data.copy_(best_state[id(p)])
    return best_val_ppl


def collect_block_activations(model, block_ids, calib_ids, device):
    block_data = {b: {"input": [], "output": [], "preact": [], "hidden": []} for b in block_ids}
    hooks = []

    def make_hook(blk_id):
        def hook_fn(module, inputs, output):
            x = inputs[0].detach()
            z = module.c_fc(x).detach()
            h = F.gelu(z).detach()
            y = output.detach()
            block_data[blk_id]["input"].append(x.reshape(-1, x.shape[-1]).cpu())
            block_data[blk_id]["output"].append(y.reshape(-1, y.shape[-1]).cpu())
            block_data[blk_id]["preact"].append(z.reshape(-1, z.shape[-1]).cpu())
            block_data[blk_id]["hidden"].append(h.reshape(-1, h.shape[-1]).cpu())
        return hook_fn

    for blk_id in block_ids:
        hook = model.transformer.h[blk_id].mlp.register_forward_hook(make_hook(blk_id))
        hooks.append(hook)

    model.eval()
    with torch.no_grad():
        for ids in calib_ids:
            ids_dev = ids.to(device) if ids.device.type != device else ids
            model(ids_dev)

    for hook in hooks:
        hook.remove()

    result = {}
    for blk_id in block_ids:
        result[blk_id] = {
            "input": torch.cat(block_data[blk_id]["input"], dim=0).float(),
            "output": torch.cat(block_data[blk_id]["output"], dim=0).float(),
            "preact": torch.cat(block_data[blk_id]["preact"], dim=0).float(),
            "hidden": torch.cat(block_data[blk_id]["hidden"], dim=0).float(),
        }
    return result


# ── Float baseline + compression calibration ──────────────────────────────────
print("\nLoading GPT-2 for calibration...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()

float_test_ppl = eval_ppl(base_model, test_eval_ids)
print(f"Float baseline test PPL: {float_test_ppl:.4f} (ref ~56.52)", flush=True)

calib_dev = [t.to(DEVICE) for t in calib_ids_cpu]
block_acts = collect_block_activations(base_model, BLOCKS, calib_dev, DEVICE)

compress_cache = {}
for blk in BLOCKS:
    acts = block_acts[blk]
    hidden = acts["hidden"]
    output = acts["output"]
    preact = acts["preact"]

    hidden_dim = hidden.shape[1]
    k = K_COMP

    std_h = hidden.std(dim=0)
    mean_h = hidden.mean(dim=0)
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    deriv = (0.5 * (1.0 + torch.erf(preact * inv_sqrt_2))
             + preact * torch.exp(-0.5 * preact.square()) * inv_sqrt_2pi)
    mean_deriv = deriv.mean(dim=0)

    c_proj = base_model.transformer.h[blk].mlp.c_proj
    c_proj_weight = c_proj.weight.detach().cpu().float()   # Conv1D: (3072, 768)
    output_norms = c_proj_weight.norm(dim=1)

    channel_scores = std_h * output_norms * (mean_h.abs() + mean_deriv.abs() + 1e-6)

    dead_mask = hidden.abs().amax(dim=0) <= 1e-6
    available = torch.arange(hidden_dim)
    live = available[~dead_mask]
    if live.numel() < k:
        live = available
    scores = channel_scores[live]
    topk_idx = torch.topk(scores, k=k, largest=True).indices
    selected = live[topk_idx].sort().values

    c_fc = base_model.transformer.h[blk].mlp.c_fc
    c_fc_weight_full = c_fc.weight.detach().cpu().float()
    c_fc_bias_full = c_fc.bias.detach().cpu().float()

    fc_w = c_fc_weight_full.T[selected]   # (k, 768)
    fc_b = c_fc_bias_full[selected]       # (k,)

    hidden_sel = hidden[:, selected].float()
    design = torch.cat([hidden_sel, torch.ones(hidden_sel.shape[0], 1)], dim=1)
    solution = torch.linalg.lstsq(design, output.float()).solution
    proj_w = solution[:-1].T.contiguous()   # (768, k)
    proj_b = solution[-1].contiguous()      # (768,)

    compress_cache[blk] = {
        "fc_w": fc_w, "fc_b": fc_b,
        "proj_w": proj_w, "proj_b": proj_b,
        "selected": selected, "k": k,
    }
    print(f"Block {blk}: k={k}", flush=True)

# Save full float weights for float configs
float_weights_cache = {}
for blk in BLOCKS:
    mlp = base_model.transformer.h[blk].mlp
    float_weights_cache[blk] = {
        "fc_w": mlp.c_fc.weight.detach().float().T.contiguous(),    # (3072, 768)
        "fc_b": mlp.c_fc.bias.detach().float(),
        "proj_w": mlp.c_proj.weight.detach().float().T.contiguous(),  # (768, 3072)
        "proj_b": mlp.c_proj.bias.detach().float(),
    }

del block_acts, base_model
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Model classes ─────────────────────────────────────────────────────────────
class GPT2MLPWithAdapter(nn.Module):
    """Generic MLP with optional residual adapter on GELU hidden dim.

    fc_weight: (k, d_model), proj_weight: (d_model, k)
    Adapter: (k → h → d_model), applied after GELU on hidden, added to proj output.
    h=0: no adapter, pure compressed/float forward pass.
    """

    def __init__(self, fc_weight, fc_bias, proj_weight, proj_bias, h=0,
                 train_proj=False):
        super().__init__()
        k = fc_weight.shape[0]
        d_model = proj_weight.shape[0]
        self.k = k
        self.d_model = d_model

        self.register_buffer("fc_weight", fc_weight.contiguous())
        self.register_buffer("fc_bias", fc_bias.contiguous())

        if train_proj:
            self.proj_weight = nn.Parameter(proj_weight.contiguous())
            self.proj_bias = nn.Parameter(proj_bias.contiguous())
        else:
            self.register_buffer("proj_weight", proj_weight.contiguous())
            self.register_buffer("proj_bias", proj_bias.contiguous())
        self._train_proj = train_proj

        self.h = h
        if h > 0:
            scale1 = 1.0 / math.sqrt(k)
            scale2 = 1.0 / math.sqrt(h)
            self.adapter_W1 = nn.Parameter(torch.randn(h, k) * scale1)
            self.adapter_b1 = nn.Parameter(torch.zeros(h))
            self.adapter_W2 = nn.Parameter(torch.randn(d_model, h) * scale2)
            self.adapter_b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        hidden = F.gelu(F.linear(x, self.fc_weight, self.fc_bias))
        out = F.linear(hidden, self.proj_weight, self.proj_bias)
        if self.h > 0:
            a = F.relu(F.linear(hidden, self.adapter_W1, self.adapter_b1))
            out = out + F.linear(a, self.adapter_W2, self.adapter_b2)
        return out

    def trainable_params(self):
        params = []
        if self._train_proj:
            params.extend([self.proj_weight, self.proj_bias])
        if self.h > 0:
            params.extend([self.adapter_W1, self.adapter_b1,
                            self.adapter_W2, self.adapter_b2])
        return params


# ── Sweep ─────────────────────────────────────────────────────────────────────
configs_results = []

for is_float in [True, False]:
    k = K_FLOAT if is_float else K_COMP
    label_kind = "float" if is_float else "comp"

    for h in H_VALUES:
        params_total = total_ffn_params(k, h)
        config_label = f"{label_kind}_k{k}_h{h}"
        print(f"\n{'='*60}", flush=True)
        print(f"Config: {config_label}  |  FFN params: {params_total}", flush=True)
        print(f"{'='*60}", flush=True)

        torch.manual_seed(SEED)
        random.seed(SEED)

        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        model.eval()

        for blk in BLOCKS:
            if is_float:
                fw = float_weights_cache[blk]
                train_proj = False  # float proj frozen
                mlp = GPT2MLPWithAdapter(
                    fc_weight=fw["fc_w"].to(DEVICE),
                    fc_bias=fw["fc_b"].to(DEVICE),
                    proj_weight=fw["proj_w"].to(DEVICE),
                    proj_bias=fw["proj_b"].to(DEVICE),
                    h=h,
                    train_proj=False,
                ).to(DEVICE)
            else:
                c = compress_cache[blk]
                mlp = GPT2MLPWithAdapter(
                    fc_weight=c["fc_w"].to(DEVICE, torch.float32),
                    fc_bias=c["fc_b"].to(DEVICE, torch.float32),
                    proj_weight=c["proj_w"].to(DEVICE, torch.float32),
                    proj_bias=c["proj_b"].to(DEVICE, torch.float32),
                    h=h,
                    train_proj=True,  # compressed: also fine-tune proj
                ).to(DEVICE)
            model.transformer.h[blk].mlp = mlp

        for p in model.parameters():
            p.requires_grad_(False)

        trainable = []
        for blk in BLOCKS:
            for p in model.transformer.h[blk].mlp.trainable_params():
                p.requires_grad_(True)
                trainable.append(p)

        n_trainable = sum(p.numel() for p in trainable)
        print(f"  Trainable params: {n_trainable}", flush=True)

        if h == 0 or n_trainable == 0:
            # No training — just eval
            best_val_ppl = eval_ppl(model, val_eval_ids)
            test_ppl = eval_ppl(model, test_eval_ids)
            print(f"  No training. Val PPL: {best_val_ppl:.4f}, Test PPL: {test_ppl:.4f}", flush=True)
        else:
            t0 = time.time()
            best_val_ppl = run_steps_with_val_selection(
                model, trainable, N_STEPS, calib_dev, val_eval_ids,
                label=config_label
            )
            elapsed = time.time() - t0
            test_ppl = eval_ppl(model, test_eval_ids)
            print(f"  Best val PPL: {best_val_ppl:.4f}, Test PPL: {test_ppl:.4f}, "
                  f"time={elapsed:.1f}s", flush=True)

        configs_results.append({
            "label": config_label,
            "k": k,
            "h": h,
            "is_float": is_float,
            "total_ffn_params": params_total,
            "best_val_ppl": float(best_val_ppl),
            "test_ppl": float(test_ppl),
        })

        del model, trainable
        if DEVICE == "mps":
            torch.mps.empty_cache()


# ── Pareto frontier ───────────────────────────────────────────────────────────
def is_pareto_optimal(idx, results):
    """A config is Pareto-optimal if no other config has fewer params AND lower PPL."""
    xi = results[idx]["total_ffn_params"]
    yi = results[idx]["test_ppl"]
    for j, r in enumerate(results):
        if j == idx:
            continue
        if r["total_ffn_params"] <= xi and r["test_ppl"] <= yi:
            if r["total_ffn_params"] < xi or r["test_ppl"] < yi:
                return False
    return True


for i, r in enumerate(configs_results):
    r["pareto_optimal"] = is_pareto_optimal(i, configs_results)

pareto_frontier = [r for r in configs_results if r["pareto_optimal"]]


# ── Verdict ───────────────────────────────────────────────────────────────────
float_configs = [r for r in configs_results if r["is_float"]]
comp_configs = [r for r in configs_results if not r["is_float"]]

best_float = min(float_configs, key=lambda r: r["test_ppl"])
best_comp = min(comp_configs, key=lambda r: r["test_ppl"])

# Check: any compressed within 2 PPL of best float+adapter at <=80% float FFN params
verdict = "REFUTED"
for cr in comp_configs:
    for fr in float_configs:
        if cr["total_ffn_params"] <= 0.80 * fr["total_ffn_params"]:
            if cr["test_ppl"] <= fr["test_ppl"] + 2.0:
                verdict = "CONFIRMED"
                break
    if verdict == "CONFIRMED":
        break

print(f"\n{'='*60}", flush=True)
print(f"VERDICT: {verdict}", flush=True)
print(f"Float baseline test PPL:  {float_test_ppl:.4f}", flush=True)
print(f"Best float+adapter:       {best_float['test_ppl']:.4f} (h={best_float['h']}, "
      f"params={best_float['total_ffn_params']})", flush=True)
print(f"Best compressed+adapter:  {best_comp['test_ppl']:.4f} (h={best_comp['h']}, "
      f"params={best_comp['total_ffn_params']})", flush=True)
print(f"\nAll configs:", flush=True)
for r in sorted(configs_results, key=lambda x: x["total_ffn_params"]):
    po = "*" if r["pareto_optimal"] else " "
    print(f"  {po} {r['label']:30s}  params={r['total_ffn_params']:>9d}  "
          f"test_ppl={r['test_ppl']:.4f}", flush=True)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

colors = {"float": "#1f77b4", "comp": "#d62728"}
markers = {0: "o", 32: "s", 64: "^", 128: "D"}

for r in configs_results:
    kind = "float" if r["is_float"] else "comp"
    ax.scatter(r["total_ffn_params"], r["test_ppl"],
               color=colors[kind], marker=markers[r["h"]],
               s=100, zorder=5,
               label=f"{r['label']} ppl={r['test_ppl']:.1f}")

# Draw Pareto frontier line (sort by params)
pf_sorted = sorted(pareto_frontier, key=lambda x: x["total_ffn_params"])
if len(pf_sorted) > 1:
    pf_x = [r["total_ffn_params"] for r in pf_sorted]
    pf_y = [r["test_ppl"] for r in pf_sorted]
    ax.step(pf_x, pf_y, where="post", color="black", linewidth=1.5,
            linestyle="--", label="Pareto frontier", zorder=3)

ax.set_xlabel("Total FFN params (3 blocks)", fontsize=12)
ax.set_ylabel("Test PPL (wikitext-2-test)", fontsize=12)
ax.set_title(f"Compression-accuracy Pareto curve — k×h sweep\nVerdict: {verdict}", fontsize=13)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(float_test_ppl, color="gray", linestyle=":", alpha=0.7, label="Float baseline")
plt.tight_layout()

os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
plt.savefig(OUTPUT_PLOT, dpi=120, bbox_inches="tight")
print(f"\nPlot saved to {OUTPUT_PLOT}", flush=True)
plt.close()


# ── Save JSON ─────────────────────────────────────────────────────────────────
output = {
    "configs": configs_results,
    "pareto_frontier": pareto_frontier,
    "verdict": verdict,
    "float_baseline_test_ppl": float(float_test_ppl),
    "best_compressed": {
        "k": best_comp["k"], "h": best_comp["h"],
        "ppl": best_comp["test_ppl"], "params": best_comp["total_ffn_params"],
    },
    "best_float": {
        "k": best_float["k"], "h": best_float["h"],
        "ppl": best_float["test_ppl"], "params": best_float["total_ffn_params"],
    },
    "config": {
        "blocks": BLOCKS,
        "k_float": K_FLOAT,
        "k_comp": K_COMP,
        "h_values": H_VALUES,
        "n_steps": N_STEPS,
        "seed": SEED,
        "val_used_for": "training and checkpoint selection",
        "test_used_for": "final evaluation only",
    },
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"Results saved to {OUTPUT_JSON}", flush=True)
