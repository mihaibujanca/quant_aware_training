"""Cycle 148: Linear AE ablation (item 158).

ReLU AE vs linear AE vs topk at k=512, blocks [4, 7], 3 seeds.
Pre-polish only. Tests whether AE advantage is from ReLU or training.

Verdict:
  CONFIRMED    if ReLU beats linear by >=2% on >=2 blocks
  REFUTED      if linear matches ReLU within 1%
  INCONCLUSIVE if 1-2% or block-dependent
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

SEEDS = [42, 137, 7]
BLOCKS = [4, 7]
D_MODEL = 768
D_FFN = 3072
D_BN = 64
K = 512
SEQ_LEN = 128
BATCH_SIZE = 4
AE_STEPS = 500
LR = 1e-4
HIDDEN_FRACTION = K / D_FFN
MAX_TOKENS = 4096

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class BottleneckFFN(nn.Module):
    def __init__(self, original_mlp, d_bn, use_relu=True):
        super().__init__()
        self.c_fc = original_mlp.c_fc
        self.act = original_mlp.act
        self.encoder = nn.Linear(D_FFN, d_bn)
        self.decoder = nn.Linear(d_bn, D_MODEL)
        self.use_relu = use_relu
        if hasattr(original_mlp.c_proj, 'bias') and original_mlp.c_proj.bias is not None:
            self.decoder.bias.data.copy_(original_mlp.c_proj.bias.data)
        for p in self.c_fc.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        z = self.encoder(h)
        if self.use_relu:
            z = F.relu(z)
        return self.decoder(z)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.encoder.parameters()) + \
               sum(p.numel() for p in self.decoder.parameters())


def train_ae(model, bid, tokens, n_steps, lr, device, seed):
    model.requires_grad_(False)
    mlp = model.transformer.h[bid].mlp
    params = list(mlp.encoder.parameters()) + list(mlp.decoder.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    rng = torch.Generator(); rng.manual_seed(seed)
    for step in range(n_steps):
        idx = torch.randperm(tokens.shape[0], generator=rng)[:BATCH_SIZE]
        batch = tokens[idx].to(device)
        opt.zero_grad()
        raw = model(batch[:, :-1])
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        loss.backward(); opt.step()
    model.requires_grad_(False)


if __name__ == "__main__":
    print("Loading GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    device = teacher.device

    _set_seed(42)
    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher, text_path=str(OWT_PATH), sequence_length=SEQ_LEN,
        probe_train_sequences=200, probe_holdout_sequences=200,
        eval_val_sequences=200, dataset_name="openwebtext", shuffle_seed=42,
    ))

    bl = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    print(f"  Float PPL: {bl['perplexity']:.3f}", flush=True)

    atlas = probe_ffn_geometry(teacher, corpus, layer_ids=tuple(sorted(BLOCKS)),
                                max_tokens_per_layer=MAX_TOKENS, batch_size=BATCH_SIZE)

    results = {}
    for seed in SEEDS:
        print(f"\n{'='*50}\nSEED {seed}\n{'='*50}", flush=True)
        for bid in BLOCKS:
            # ReLU AE
            _set_seed(seed)
            relu_model = copy.deepcopy(teacher.model); relu_model.eval()
            relu_model.transformer.h[bid].mlp = BottleneckFFN(relu_model.transformer.h[bid].mlp, D_BN, use_relu=True)
            train_ae(relu_model, bid, corpus.probe_holdout, AE_STEPS, LR, device, seed)
            relu_m = _evaluate_model(teacher, relu_model, corpus, batch_size=BATCH_SIZE)
            del relu_model

            # Linear AE
            _set_seed(seed)
            lin_model = copy.deepcopy(teacher.model); lin_model.eval()
            lin_model.transformer.h[bid].mlp = BottleneckFFN(lin_model.transformer.h[bid].mlp, D_BN, use_relu=False)
            train_ae(lin_model, bid, corpus.probe_holdout, AE_STEPS, LR, device, seed)
            lin_m = _evaluate_model(teacher, lin_model, corpus, batch_size=BATCH_SIZE)
            del lin_model

            # Topk
            layer = [l for l in atlas.layer_geometries if l.layer_id == bid][0]
            budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION)
            k = _layer_target_hidden(layer, budget)
            sel = _select_hidden_units(layer, budget, k)
            sw = _standard_weight(layer.handle.c_fc).float()
            sb = _standard_bias(layer.handle.c_fc)
            stw = sw[sel].contiguous()
            stb = None if sb is None else sb[sel].contiguous()
            h = layer.hidden_sample[:, sel].float()
            o = layer.output_sample.float()
            use_ob = _standard_bias(layer.handle.c_proj) is not None
            swo, sbo = _fit_output_projection(h, o, use_bias=use_ob)
            cl = CompressedFfnLayer(layer_id=bid, activation_kind=layer.activation_kind,
                source_hidden_dim=layer.hidden_dim, compressed_hidden_dim=int(k),
                c_fc_weight=stw, c_fc_bias=stb, c_proj_weight=swo, c_proj_bias=sbo,
                selected_indices=sel, source_param_count=0, compressed_param_count=0, notes={})
            pack = CompressedFfnPack(method="geometric", model_family=atlas.model_family,
                dataset_name=atlas.dataset_name, layers=(cl,), metadata={"sequence_length": atlas.sequence_length})
            topk_model = swap_compressed_ffns(teacher, pack, inplace=False); topk_model.eval()
            topk_m = _evaluate_model(teacher, topk_model, corpus, batch_size=BATCH_SIZE)
            del topk_model

            print(f"  B{bid}: relu={relu_m['perplexity']:.2f}, linear={lin_m['perplexity']:.2f}, topk={topk_m['perplexity']:.2f}", flush=True)
            results[f"{seed}_{bid}"] = {
                "relu": relu_m["perplexity"], "linear": lin_m["perplexity"], "topk": topk_m["perplexity"],
            }

    # Summary
    print(f"\n{'='*50}\nSUMMARY\n{'='*50}", flush=True)
    relu_wins = 0
    for bid in BLOCKS:
        rv = [results[f"{s}_{bid}"]["relu"] for s in SEEDS]
        lv = [results[f"{s}_{bid}"]["linear"] for s in SEEDS]
        tv = [results[f"{s}_{bid}"]["topk"] for s in SEEDS]
        rm, lm, tm = np.mean(rv), np.mean(lv), np.mean(tv)
        rel_rl = (rm - lm) / lm * 100
        print(f"  B{bid}: relu={rm:.2f}±{np.std(rv):.2f}, linear={lm:.2f}±{np.std(lv):.2f}, topk={tm:.2f}±{np.std(tv):.2f}, relu_vs_lin={rel_rl:+.1f}%", flush=True)
        if rm < lm * 0.98:
            relu_wins += 1

    if relu_wins >= 2:
        verdict = "CONFIRMED"
    elif relu_wins == 0 and all(abs(np.mean([results[f"{s}_{b}"]["relu"] for s in SEEDS]) -
                                     np.mean([results[f"{s}_{b}"]["linear"] for s in SEEDS])) /
                                np.mean([results[f"{s}_{b}"]["linear"] for s in SEEDS]) < 0.01 for b in BLOCKS):
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    detail = f"ReLU beats linear by >=2% on {relu_wins}/{len(BLOCKS)} blocks"
    print(f"\n  VERDICT: {verdict}\n  Detail: {detail}", flush=True)

    output = {
        "cycle": 148, "item": 158, "seeds": SEEDS, "blocks": BLOCKS,
        "d_bn": D_BN, "float_ppl": bl["perplexity"],
        "results": results, "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_148_linear_ae_ablation.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved JSON\nDone.", flush=True)
