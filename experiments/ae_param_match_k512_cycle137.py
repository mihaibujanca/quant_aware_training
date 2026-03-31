"""Cycle 137: Bottleneck AE param-matched at k=512 (6x) — GPT-2 FFN (item 151).

Three methods: topk+lstsq, full AE (d_bn=512), param-matched AE (d_bn=64).
4 blocks × 3 methods × 3 seeds × pre/post 500-step polish.

Verdict:
  CONFIRMED    if param-matched AE beats topk by >=3% post-polish on >=2/4 blocks
  REFUTED      if param-matched matches or loses to topk post-polish
  INCONCLUSIVE if full AE beats topk but param-matched doesn't
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
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

SEEDS = [42, 137, 7]
BLOCKS = [4, 7]  # Scoped down from [1,4,7,11] for time
D_MODEL = 768
D_FFN = 3072
K = 512  # 6x compression
D_BN_FULL = 512
D_BN_PARAM = 64  # param-matched
SEQ_LEN = 128
BATCH_SIZE = 4
AE_TRAIN_STEPS = 500
POLISH_STEPS = 500
LR = 1e-4
HIDDEN_FRACTION = K / D_FFN
EVAL_SEQS = 200
MAX_TOKENS = 4096

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


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
        z = F.relu(self.encoder(h))
        return self.decoder(z)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.encoder.parameters()) + \
               sum(p.numel() for p in self.decoder.parameters())


def train_model(model, tokens, n_steps, lr, device, seed, block_ids=None):
    """Train learnable params under CE loss."""
    model.requires_grad_(False)
    params = []
    for bid in (block_ids or BLOCKS):
        mlp = model.transformer.h[bid].mlp
        if isinstance(mlp, BottleneckFFN):
            for p in mlp.encoder.parameters():
                p.requires_grad_(True); params.append(p)
            for p in mlp.decoder.parameters():
                p.requires_grad_(True); params.append(p)
        elif hasattr(mlp, 'c_proj'):
            mlp.c_proj.weight.requires_grad_(True)
            params.append(mlp.c_proj.weight)
            if mlp.c_proj.bias is not None:
                mlp.c_proj.bias.requires_grad_(True)
                params.append(mlp.c_proj.bias)
    if not params:
        return []
    opt = torch.optim.Adam(params, lr=lr)
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
    model.requires_grad_(False)
    return losses


if __name__ == "__main__":
    print("Loading GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    device = teacher.device

    _set_seed(42)
    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher, text_path=str(OWT_PATH), sequence_length=SEQ_LEN,
        probe_train_sequences=200, probe_holdout_sequences=200,
        eval_val_sequences=EVAL_SEQS, dataset_name="openwebtext", shuffle_seed=42,
    ))

    bl = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    print(f"  Float PPL: {bl['perplexity']:.3f}", flush=True)

    budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION)
    atlas = probe_ffn_geometry(teacher, corpus, layer_ids=tuple(sorted(BLOCKS)),
                                max_tokens_per_layer=MAX_TOKENS, batch_size=BATCH_SIZE)

    results = {}
    for seed in SEEDS:
        print(f"\n{'='*50}\nSEED {seed}\n{'='*50}", flush=True)
        for bid in BLOCKS:
            print(f"\n  Block {bid}:", flush=True)

            # 1. topk+lstsq
            _set_seed(seed)
            from aleph.lm_geometric_compression import _layer_target_hidden, _select_hidden_units, _standard_weight, _standard_bias, _fit_output_projection, CompressedFfnLayer, CompressedFfnPack
            layer = [l for l in atlas.layer_geometries if l.layer_id == bid][0]
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
            sp = sw.numel() + (0 if sb is None else sb.numel()) + _standard_weight(layer.handle.c_proj).numel() + (0 if _standard_bias(layer.handle.c_proj) is None else _standard_bias(layer.handle.c_proj).numel())
            cp = stw.numel() + (0 if stb is None else stb.numel()) + swo.numel() + (0 if sbo is None else sbo.numel())
            cl = CompressedFfnLayer(layer_id=bid, activation_kind=layer.activation_kind,
                source_hidden_dim=layer.hidden_dim, compressed_hidden_dim=int(k),
                c_fc_weight=stw, c_fc_bias=stb, c_proj_weight=swo, c_proj_bias=sbo,
                selected_indices=sel, source_param_count=sp, compressed_param_count=cp,
                notes={"selection_mode": budget.selection_mode})
            pack = CompressedFfnPack(method="geometric", model_family=atlas.model_family,
                dataset_name=atlas.dataset_name, layers=(cl,), metadata={"sequence_length": atlas.sequence_length})
            topk_model = swap_compressed_ffns(teacher, pack, inplace=False); topk_model.eval()
            pre_topk = _evaluate_model(teacher, topk_model, corpus, batch_size=BATCH_SIZE)
            train_model(topk_model, corpus.probe_holdout, POLISH_STEPS, LR, device, seed, [bid])
            post_topk = _evaluate_model(teacher, topk_model, corpus, batch_size=BATCH_SIZE)
            topk_params = cp
            print(f"    topk: pre={pre_topk['perplexity']:.2f}, post={post_topk['perplexity']:.2f}, params={topk_params}", flush=True)
            del topk_model

            # 2. Full AE (d_bn=512)
            _set_seed(seed)
            ae_full = copy.deepcopy(teacher.model); ae_full.eval()
            ae_full.transformer.h[bid].mlp = BottleneckFFN(ae_full.transformer.h[bid].mlp, D_BN_FULL)
            ae_full_params = ae_full.transformer.h[bid].mlp.n_params
            train_model(ae_full, corpus.probe_holdout, AE_TRAIN_STEPS, LR, device, seed, [bid])
            pre_ae_full = _evaluate_model(teacher, ae_full, corpus, batch_size=BATCH_SIZE)
            train_model(ae_full, corpus.probe_holdout, POLISH_STEPS, LR, device, seed, [bid])
            post_ae_full = _evaluate_model(teacher, ae_full, corpus, batch_size=BATCH_SIZE)
            print(f"    full AE: pre={pre_ae_full['perplexity']:.2f}, post={post_ae_full['perplexity']:.2f}, params={ae_full_params}", flush=True)
            del ae_full

            # 3. Param-matched AE (d_bn=64)
            _set_seed(seed)
            ae_pm = copy.deepcopy(teacher.model); ae_pm.eval()
            ae_pm.transformer.h[bid].mlp = BottleneckFFN(ae_pm.transformer.h[bid].mlp, D_BN_PARAM)
            ae_pm_params = ae_pm.transformer.h[bid].mlp.n_params
            train_model(ae_pm, corpus.probe_holdout, AE_TRAIN_STEPS, LR, device, seed, [bid])
            pre_ae_pm = _evaluate_model(teacher, ae_pm, corpus, batch_size=BATCH_SIZE)
            train_model(ae_pm, corpus.probe_holdout, POLISH_STEPS, LR, device, seed, [bid])
            post_ae_pm = _evaluate_model(teacher, ae_pm, corpus, batch_size=BATCH_SIZE)
            print(f"    param AE: pre={pre_ae_pm['perplexity']:.2f}, post={post_ae_pm['perplexity']:.2f}, params={ae_pm_params}", flush=True)
            del ae_pm

            key = f"{seed}_{bid}"
            results[key] = {
                "seed": seed, "block": bid,
                "topk": {"pre": pre_topk["perplexity"], "post": post_topk["perplexity"], "params": int(topk_params)},
                "full_ae": {"pre": pre_ae_full["perplexity"], "post": post_ae_full["perplexity"], "params": int(ae_full_params)},
                "param_ae": {"pre": pre_ae_pm["perplexity"], "post": post_ae_pm["perplexity"], "params": int(ae_pm_params)},
            }

    # Summary
    print(f"\n{'='*50}\nSUMMARY (post-polish)\n{'='*50}", flush=True)
    pm_wins = 0
    for bid in BLOCKS:
        topk_vals = [results[f"{s}_{bid}"]["topk"]["post"] for s in SEEDS]
        full_vals = [results[f"{s}_{bid}"]["full_ae"]["post"] for s in SEEDS]
        pm_vals = [results[f"{s}_{bid}"]["param_ae"]["post"] for s in SEEDS]
        tm, fm, pm_m = np.mean(topk_vals), np.mean(full_vals), np.mean(pm_vals)
        rel_full = (fm - tm) / tm * 100
        rel_pm = (pm_m - tm) / tm * 100
        if pm_m < tm * 0.97:
            pm_wins += 1
        print(f"  B{bid}: topk={tm:.2f}±{np.std(topk_vals):.2f}, "
              f"full={fm:.2f}({rel_full:+.1f}%), pm={pm_m:.2f}({rel_pm:+.1f}%)", flush=True)

    if pm_wins >= 2:
        verdict = "CONFIRMED"
        detail = f"Param-matched AE beats topk by >=3% on {pm_wins}/4 blocks"
    elif pm_wins == 0:
        verdict = "REFUTED"
        detail = "Param-matched AE matches or loses to topk on all blocks"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"Param-matched AE beats topk on {pm_wins}/4 blocks"

    print(f"\n  VERDICT: {verdict}\n  Detail: {detail}", flush=True)

    output = {
        "cycle": 137, "item": 151, "seeds": SEEDS, "blocks": BLOCKS,
        "float_ppl": bl["perplexity"],
        "results": results, "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_137_ae_param_match.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved JSON\nDone.", flush=True)
