"""Cycle 138: AE k=4 stacking — GPT-2 FFN (item 152).

3 conditions: topk baseline, AE stacking, AE+LoRA corrections.
Seed 42 initial sweep. Escalate to 3 seeds if CONFIRMED threshold crossed.

Verdict:
  CONFIRMED    if AE stacking PPL < 67 OR AE+corrections < 57
  REFUTED      if AE stacking PPL >= 69
  INCONCLUSIVE if 67-69
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    FfnGeometryAtlas,
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
BLOCKS = [7, 5, 4, 1]  # k=4, order from prior cycles
D_MODEL = 768
D_FFN = 3072
D_BN = 64
K = 512
SEQ_LEN = 128
BATCH_SIZE = 4
AE_STEPS = 500
POLISH_STEPS = 500
CORRECTION_STEPS = 500
LR = 1e-4
CORRECTION_RANK = 8
HIDDEN_FRACTION = K / D_FFN
MAX_TOKENS = 4096

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _param_count(w, b=None):
    return w.numel() + (0 if b is None else b.numel())


class BottleneckFFN(nn.Module):
    def __init__(self, original_mlp, d_bn):
        super().__init__()
        self.c_fc = original_mlp.c_fc
        self.act = original_mlp.act
        self.encoder = nn.Linear(D_FFN, d_bn)
        self.decoder = nn.Linear(d_bn, D_MODEL)
        if hasattr(original_mlp.c_proj, 'bias') and original_mlp.c_proj.bias is not None:
            self.decoder.bias.data.copy_(original_mlp.c_proj.bias.data)
        for p in self.c_fc.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        return self.decoder(F.relu(self.encoder(h)))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.encoder.parameters()) + \
               sum(p.numel() for p in self.decoder.parameters())


class LowRankCorrection(nn.Module):
    def __init__(self, d_model, rank):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(d_model, rank))
        self.V = nn.Parameter(torch.randn(rank, d_model) * 0.01)

    def forward(self, x):
        return x + (x @ self.V.t()) @ self.U.t()


def train_params(model, tokens, n_steps, lr, device, seed, params_to_train):
    """Train specified params under CE loss."""
    model.requires_grad_(False)
    for p in params_to_train:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params_to_train, lr=lr)
    rng = torch.Generator(); rng.manual_seed(seed)
    losses = []
    for step in range(n_steps):
        idx = torch.randperm(tokens.shape[0], generator=rng)[:BATCH_SIZE]
        batch = tokens[idx].to(device)
        opt.zero_grad()
        raw = model(batch[:, :-1])
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        loss.backward(); opt.step(); losses.append(float(loss.item()))
        if (step + 1) % 100 == 0:
            print(f"      step {step+1}/{n_steps}: loss={loss.item():.4f}", flush=True)
    model.requires_grad_(False)
    return losses


def get_ae_params(model, block_ids):
    params = []
    for bid in block_ids:
        mlp = model.transformer.h[bid].mlp
        if isinstance(mlp, BottleneckFFN):
            params.extend(mlp.encoder.parameters())
            params.extend(mlp.decoder.parameters())
    return params


def get_topk_params(model, block_ids):
    params = []
    for bid in block_ids:
        mlp = model.transformer.h[bid].mlp
        if hasattr(mlp, 'c_proj'):
            params.append(mlp.c_proj.weight)
            if mlp.c_proj.bias is not None:
                params.append(mlp.c_proj.bias)
    return params


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
    float_ppl = bl["perplexity"]
    print(f"  Float PPL: {float_ppl:.3f}", flush=True)

    budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION)
    atlas = probe_ffn_geometry(teacher, corpus, layer_ids=tuple(sorted(BLOCKS)),
                                max_tokens_per_layer=MAX_TOKENS, batch_size=BATCH_SIZE)

    results = {}

    # === CONDITION 1: topk baseline ===
    print(f"\n{'='*50}\nCONDITION 1: topk+lstsq stacking\n{'='*50}", flush=True)
    _set_seed(SEED)

    # Build per-block compressed layers
    compressed_layers = []
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
        sp = sw.numel() + (0 if sb is None else sb.numel()) + _standard_weight(layer.handle.c_proj).numel()
        cp = stw.numel() + (0 if stb is None else stb.numel()) + swo.numel() + (0 if sbo is None else sbo.numel())
        compressed_layers.append(CompressedFfnLayer(
            layer_id=layer.layer_id, activation_kind=layer.activation_kind,
            source_hidden_dim=layer.hidden_dim, compressed_hidden_dim=int(k),
            c_fc_weight=stw, c_fc_bias=stb, c_proj_weight=swo, c_proj_bias=sbo,
            selected_indices=sel, source_param_count=sp, compressed_param_count=cp,
            notes={"selection_mode": budget.selection_mode}))

    pack = CompressedFfnPack(method="geometric", model_family=atlas.model_family,
        dataset_name=atlas.dataset_name, layers=tuple(compressed_layers),
        metadata={"sequence_length": atlas.sequence_length})
    topk_model = swap_compressed_ffns(teacher, pack, inplace=False); topk_model.eval()

    pre_topk = _evaluate_model(teacher, topk_model, corpus, batch_size=BATCH_SIZE)
    print(f"  Pre-polish: {pre_topk['perplexity']:.3f}", flush=True)

    topk_params = get_topk_params(topk_model, BLOCKS)
    print(f"  Training {POLISH_STEPS} polish steps...", flush=True)
    train_params(topk_model, corpus.probe_holdout, POLISH_STEPS, LR, device, SEED, topk_params)
    post_topk = _evaluate_model(teacher, topk_model, corpus, batch_size=BATCH_SIZE)
    print(f"  Post-polish: {post_topk['perplexity']:.3f}", flush=True)
    results["topk"] = {"pre": pre_topk["perplexity"], "post": post_topk["perplexity"]}
    del topk_model

    # === CONDITION 2: AE stacking ===
    print(f"\n{'='*50}\nCONDITION 2: AE stacking (d_bn={D_BN})\n{'='*50}", flush=True)

    # Train each AE per-block independently
    trained_aes = {}
    for bid in BLOCKS:
        print(f"  Training AE for block {bid}...", flush=True)
        _set_seed(SEED)
        ae_model = copy.deepcopy(teacher.model); ae_model.eval()
        ae_model.transformer.h[bid].mlp = BottleneckFFN(ae_model.transformer.h[bid].mlp, D_BN)
        ae_params = list(ae_model.transformer.h[bid].mlp.encoder.parameters()) + \
                    list(ae_model.transformer.h[bid].mlp.decoder.parameters())
        train_params(ae_model, corpus.probe_holdout, AE_STEPS, LR, device, SEED, ae_params)
        # Save trained AE module
        trained_aes[bid] = ae_model.transformer.h[bid].mlp
        sb_ppl = _evaluate_model(teacher, ae_model, corpus, batch_size=BATCH_SIZE)
        print(f"    Single-block PPL: {sb_ppl['perplexity']:.3f}, params: {trained_aes[bid].n_params}", flush=True)
        del ae_model

    # Stack all 4 AE blocks
    ae_stack = copy.deepcopy(teacher.model); ae_stack.eval()
    for bid in BLOCKS:
        ae_stack.transformer.h[bid].mlp = trained_aes[bid]

    pre_ae = _evaluate_model(teacher, ae_stack, corpus, batch_size=BATCH_SIZE)
    print(f"  AE stacked pre-polish: {pre_ae['perplexity']:.3f}", flush=True)

    # Joint polish
    ae_joint_params = get_ae_params(ae_stack, BLOCKS)
    print(f"  Joint polish {POLISH_STEPS} steps ({len(ae_joint_params)} param tensors)...", flush=True)
    train_params(ae_stack, corpus.probe_holdout, POLISH_STEPS, LR, device, SEED, ae_joint_params)
    post_ae = _evaluate_model(teacher, ae_stack, corpus, batch_size=BATCH_SIZE)
    print(f"  AE stacked post-polish: {post_ae['perplexity']:.3f}", flush=True)
    results["ae"] = {"pre": pre_ae["perplexity"], "post": post_ae["perplexity"]}

    # === CONDITION 3: AE + LoRA corrections ===
    print(f"\n{'='*50}\nCONDITION 3: AE + LoRA corrections (r={CORRECTION_RANK})\n{'='*50}", flush=True)

    corrections = nn.ModuleDict()
    hooks = []
    for bid in BLOCKS:
        corr = LowRankCorrection(D_MODEL, CORRECTION_RANK)
        corrections[str(bid)] = corr
        block = ae_stack.transformer.h[bid]

        def make_hook(c):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    return (c(output[0]),) + output[1:]
                return c(output)
            return hook

        hooks.append(block.register_forward_hook(make_hook(corr)))

    corr_params = list(corrections.parameters())
    total_corr = sum(p.numel() for p in corr_params)
    print(f"  LoRA correction params: {total_corr}", flush=True)

    print(f"  Training corrections {CORRECTION_STEPS} steps...", flush=True)
    train_params(ae_stack, corpus.probe_holdout, CORRECTION_STEPS, LR, device, SEED, corr_params)
    post_ae_corr = _evaluate_model(teacher, ae_stack, corpus, batch_size=BATCH_SIZE)
    print(f"  AE + corrections PPL: {post_ae_corr['perplexity']:.3f}", flush=True)
    results["ae_corr"] = {"ppl": post_ae_corr["perplexity"], "corr_params": total_corr}

    for h in hooks:
        h.remove()

    # === VERDICT ===
    print(f"\n{'='*50}\nVERDICT\n{'='*50}", flush=True)
    ae_post = results["ae"]["post"]
    topk_post = results["topk"]["post"]
    ae_corr_ppl = results["ae_corr"]["ppl"]

    print(f"  topk post-polish:  {topk_post:.3f}", flush=True)
    print(f"  AE post-polish:    {ae_post:.3f}", flush=True)
    print(f"  AE + corrections:  {ae_corr_ppl:.3f}", flush=True)
    print(f"  Float:             {float_ppl:.3f}", flush=True)

    if ae_post < 67.0 or ae_corr_ppl < 57.0:
        verdict = "CONFIRMED"
        detail = f"AE stacking={ae_post:.1f} (<67) or AE+corr={ae_corr_ppl:.1f} (<57)"
    elif ae_post >= 69.0:
        verdict = "REFUTED"
        detail = f"AE stacking={ae_post:.1f} (>=69, within 2% of topk {topk_post:.1f})"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"AE stacking={ae_post:.1f} (67-69 range)"

    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  Detail: {detail}", flush=True)

    # Save
    output = {
        "cycle": 138, "item": 152, "seed": SEED,
        "float_ppl": float_ppl,
        "topk": results["topk"],
        "ae": results["ae"],
        "ae_corr": results["ae_corr"],
        "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_138_ae_stacking.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON\nDone.", flush=True)
