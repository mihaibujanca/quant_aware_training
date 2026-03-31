"""Cycle 153: Joint training + LoRA corrections (item 163).

Compare: (A) joint-trained rank-64 + corrections, (B) topk+polish + corrections.
Seed 42.

Verdict:
  CONFIRMED    if joint+corr PPL < 57 (beats topk+corr ~59 by >3%)
  REFUTED      if joint+corr PPL >= 59 (corrections don't help joint model)
  INCONCLUSIVE if 57-59
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnLayer,
    CompressedFfnPack,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _layer_target_hidden,
    _select_hidden_units,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

SEED = 42
BLOCKS = [7, 5, 4, 1]
D_MODEL = 768
D_FFN = 3072
D_BN = 64
RANK = 8
SEQ_LEN = 128
BATCH_SIZE = 4
JOINT_STEPS = 2000
CORRECTION_STEPS = 2000
POLISH_STEPS = 500
LR = 1e-4
HIDDEN_FRACTION = 512 / D_FFN
MAX_TOKENS = 4096

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class LinearFactorFFN(nn.Module):
    def __init__(self, original_mlp, d_bn):
        super().__init__()
        self.c_fc = original_mlp.c_fc
        self.act = original_mlp.act
        self.encoder = nn.Linear(D_FFN, d_bn, bias=False)
        self.decoder = nn.Linear(d_bn, D_MODEL, bias=True)
        if hasattr(original_mlp.c_proj, 'bias') and original_mlp.c_proj.bias is not None:
            self.decoder.bias.data.copy_(original_mlp.c_proj.bias.data)
        for p in self.c_fc.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.decoder(self.encoder(self.act(self.c_fc(x))))


class LowRankCorrection(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(d, r))
        self.V = nn.Parameter(torch.randn(r, d) * 0.01)

    def forward(self, x):
        return x + (x @ self.V.t()) @ self.U.t()


def train_params(model, tokens, n_steps, lr, device, seed, params):
    model.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=0)
    rng = torch.Generator(); rng.manual_seed(seed)
    for step in range(n_steps):
        idx = torch.randperm(tokens.shape[0], generator=rng)[:BATCH_SIZE]
        batch = tokens[idx].to(device)
        opt.zero_grad()
        raw = model(batch[:, :-1])
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        loss.backward(); opt.step(); sched.step()
        if (step + 1) % 500 == 0:
            print(f"      step {step+1}/{n_steps}: loss={loss.item():.4f}", flush=True)
    model.requires_grad_(False)


def add_corrections(model, blocks, rank):
    corrections = nn.ModuleDict({str(b): LowRankCorrection(D_MODEL, rank) for b in blocks})
    hooks = []
    for bid in blocks:
        corr = corrections[str(bid)]
        block = model.transformer.h[bid]
        def make_hook(c):
            def hook(mod, inp, out):
                return (c(out[0]),) + out[1:] if isinstance(out, tuple) else c(out)
            return hook
        hooks.append(block.register_forward_hook(make_hook(corr)))
    return corrections, hooks


if __name__ == "__main__":
    _set_seed(SEED)
    print("Loading GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    device = teacher.device

    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher, text_path=str(OWT_PATH), sequence_length=SEQ_LEN,
        probe_train_sequences=200, probe_holdout_sequences=200,
        eval_val_sequences=200, dataset_name="openwebtext", shuffle_seed=SEED,
    ))

    bl = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    print(f"  Float PPL: {bl['perplexity']:.3f}", flush=True)

    results = {}

    # === Condition A: Joint-trained + corrections ===
    print(f"\n{'='*50}\nCONDITION A: Joint rank-64 + corrections\n{'='*50}", flush=True)
    _set_seed(SEED)
    joint_model = copy.deepcopy(teacher.model); joint_model.eval()
    for bid in BLOCKS:
        joint_model.transformer.h[bid].mlp = LinearFactorFFN(joint_model.transformer.h[bid].mlp, D_BN)

    # Joint training (2000 steps)
    joint_params = []
    for bid in BLOCKS:
        joint_params.extend(joint_model.transformer.h[bid].mlp.encoder.parameters())
        joint_params.extend(joint_model.transformer.h[bid].mlp.decoder.parameters())
    print(f"  Joint training {JOINT_STEPS} steps...", flush=True)
    train_params(joint_model, corpus.probe_holdout, JOINT_STEPS, LR, device, SEED, joint_params)
    joint_ppl = _evaluate_model(teacher, joint_model, corpus, batch_size=BATCH_SIZE)
    print(f"  Joint PPL: {joint_ppl['perplexity']:.3f}", flush=True)

    # Add corrections
    corrections_a, hooks_a = add_corrections(joint_model, BLOCKS, RANK)
    corr_params = list(corrections_a.parameters())
    print(f"  Training corrections {CORRECTION_STEPS} steps...", flush=True)
    train_params(joint_model, corpus.probe_holdout, CORRECTION_STEPS, LR, device, SEED + 1, corr_params)
    joint_corr_ppl = _evaluate_model(teacher, joint_model, corpus, batch_size=BATCH_SIZE)
    print(f"  Joint + corrections PPL: {joint_corr_ppl['perplexity']:.3f}", flush=True)
    for h in hooks_a:
        h.remove()
    results["joint_corr"] = {"joint_ppl": joint_ppl["perplexity"], "corrected_ppl": joint_corr_ppl["perplexity"]}
    del joint_model

    # === Condition B: topk+polish + corrections (baseline) ===
    print(f"\n{'='*50}\nCONDITION B: topk+polish + corrections\n{'='*50}", flush=True)
    budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION)
    atlas = probe_ffn_geometry(teacher, corpus, layer_ids=tuple(sorted(BLOCKS)),
                                max_tokens_per_layer=MAX_TOKENS, batch_size=BATCH_SIZE)
    layers = []
    for layer in atlas.layer_geometries:
        if layer.layer_id not in BLOCKS:
            continue
        k = _layer_target_hidden(layer, budget)
        sel = _select_hidden_units(layer, budget, k)
        sw = _standard_weight(layer.handle.c_fc).float()
        sb = _standard_bias(layer.handle.c_fc)
        use_ob = _standard_bias(layer.handle.c_proj) is not None
        stw = sw[sel].contiguous()
        stb = None if sb is None else sb[sel].contiguous()
        h = layer.hidden_sample[:, sel].float()
        o = layer.output_sample.float()
        swo, sbo = _fit_output_projection(h, o, use_bias=use_ob)
        layers.append(CompressedFfnLayer(
            layer_id=layer.layer_id, activation_kind=layer.activation_kind,
            source_hidden_dim=layer.hidden_dim, compressed_hidden_dim=int(k),
            c_fc_weight=stw, c_fc_bias=stb, c_proj_weight=swo, c_proj_bias=sbo,
            selected_indices=sel, source_param_count=0, compressed_param_count=0, notes={}))
    pack = CompressedFfnPack(method="geometric", model_family=atlas.model_family,
        dataset_name=atlas.dataset_name, layers=tuple(layers), metadata={})
    topk_model = swap_compressed_ffns(teacher, pack, inplace=False); topk_model.eval()

    # Corrections on UNPOLISHED topk (per cycle 151 lesson)
    unc_ppl = _evaluate_model(teacher, topk_model, corpus, batch_size=BATCH_SIZE)
    print(f"  Unpolished topk PPL: {unc_ppl['perplexity']:.3f}", flush=True)

    corrections_b, hooks_b = add_corrections(topk_model, BLOCKS, RANK)
    corr_params_b = list(corrections_b.parameters())
    print(f"  Training corrections {CORRECTION_STEPS} steps...", flush=True)
    train_params(topk_model, corpus.probe_holdout, CORRECTION_STEPS, LR, device, SEED + 1, corr_params_b)
    topk_corr_ppl = _evaluate_model(teacher, topk_model, corpus, batch_size=BATCH_SIZE)
    print(f"  topk + corrections PPL: {topk_corr_ppl['perplexity']:.3f}", flush=True)
    for h in hooks_b:
        h.remove()
    results["topk_corr"] = {"unpolished_ppl": unc_ppl["perplexity"], "corrected_ppl": topk_corr_ppl["perplexity"]}
    del topk_model

    # Verdict
    print(f"\n{'='*50}\nVERDICT\n{'='*50}", flush=True)
    jc = results["joint_corr"]["corrected_ppl"]
    tc = results["topk_corr"]["corrected_ppl"]
    print(f"  Joint + corrections:  {jc:.3f}", flush=True)
    print(f"  topk + corrections:   {tc:.3f}", flush=True)

    if jc < 57.0:
        verdict = "CONFIRMED"
        detail = f"Joint+corr ({jc:.1f}) < 57, beats topk+corr ({tc:.1f})"
    elif jc >= tc * 0.97:
        verdict = "REFUTED"
        detail = f"Joint+corr ({jc:.1f}) >= topk+corr ({tc:.1f})"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"Joint+corr ({jc:.1f}) vs topk+corr ({tc:.1f})"

    print(f"  VERDICT: {verdict}\n  Detail: {detail}", flush=True)

    output = {
        "cycle": 153, "item": 163, "seed": SEED,
        "float_ppl": bl["perplexity"],
        "results": results, "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_153_joint_corrections.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved JSON\nDone.", flush=True)
