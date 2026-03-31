"""Cycle 149: Joint stacking-aware compression training — GPT-2 FFN k=4 (item 159).

Train all 4 compressed blocks jointly under stacked CE loss.
Linear factorization (rank 64), no ReLU. SVD init vs random init vs topk baseline.

Verdict:
  CONFIRMED    if joint-SVD PPL <= 68.6 (beats topk ~70 by >=2%)
  REFUTED      if joint-SVD PPL >= 68.6
  INCONCLUSIVE if better by 2-5% but single seed
"""
from __future__ import annotations

import copy
import json
import math
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
SEQ_LEN = 128
BATCH_SIZE = 4
JOINT_STEPS = 2000
POLISH_STEPS = 500
LR = 1e-4
HIDDEN_FRACTION = 512 / D_FFN  # 6x for topk baseline
MAX_TOKENS = 4096
EVAL_POINTS = [500, 1000, 1500, 2000]

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class LinearFactorFFN(nn.Module):
    """Replace c_proj with rank-r linear factorization. c_fc frozen."""
    def __init__(self, original_mlp, d_bn, init_mode="svd"):
        super().__init__()
        self.c_fc = original_mlp.c_fc
        self.act = original_mlp.act
        self.encoder = nn.Linear(D_FFN, d_bn, bias=False)
        self.decoder = nn.Linear(d_bn, D_MODEL, bias=True)

        # Copy original bias
        if hasattr(original_mlp.c_proj, 'bias') and original_mlp.c_proj.bias is not None:
            self.decoder.bias.data.copy_(original_mlp.c_proj.bias.data)

        if init_mode == "svd":
            # SVD init from original c_proj
            # Conv1D weight: (in_features, out_features) = (D_FFN, D_MODEL)
            # So W = c_proj.weight has shape (3072, 768) and forward is input @ W
            W = original_mlp.c_proj.weight.data.float()  # (3072, 768) for Conv1D
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            # W ≈ U[:, :r] @ diag(S[:r]) @ Vt[:r, :]
            # encoder maps (batch, D_FFN) -> (batch, d_bn): input @ encoder.weight.T
            # So encoder.weight should be (d_bn, D_FFN) for nn.Linear
            # W_enc.weight = Vt[:d_bn, :] — but wait, Vt here is from the SVD of (3072, 768)
            # U: (3072, 768), S: (768,), Vt: (768, 768)
            # For rank-64: W ≈ U[:, :64] @ diag(S[:64]) @ Vt[:64, :]
            # encoder: h -> h @ U[:, :64] → shape (batch, 64)
            # decoder: z -> z @ diag(S[:64]) @ Vt[:64, :] → shape (batch, 768)
            self.encoder.weight.data = U[:, :d_bn].T.contiguous()  # (64, 3072)
            self.decoder.weight.data = (torch.diag(S[:d_bn]) @ Vt[:d_bn, :]).T.contiguous()  # (768, 64)

        # Freeze c_fc
        for p in self.c_fc.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        h = self.act(self.c_fc(x))  # (batch, seq, 3072)
        z = self.encoder(h)          # (batch, seq, 64) — linear, no ReLU
        return self.decoder(z)        # (batch, seq, 768)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.encoder.parameters()) + \
               sum(p.numel() for p in self.decoder.parameters())


def train_joint(model, tokens, n_steps, lr, device, seed, eval_fn=None, eval_points=None):
    """Train all learnable params jointly under CE loss with cosine schedule."""
    model.requires_grad_(False)
    params = []
    for bid in BLOCKS:
        mlp = model.transformer.h[bid].mlp
        if isinstance(mlp, LinearFactorFFN):
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

    opt = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=0)
    rng = torch.Generator(); rng.manual_seed(seed)
    losses = []
    checkpoints = {}

    for step in range(n_steps):
        idx = torch.randperm(tokens.shape[0], generator=rng)[:BATCH_SIZE]
        batch = tokens[idx].to(device)
        opt.zero_grad()
        raw = model(batch[:, :-1])
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        loss.backward(); opt.step(); scheduler.step()
        losses.append(float(loss.item()))

        if (step + 1) % 200 == 0:
            print(f"      step {step+1}/{n_steps}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.6f}", flush=True)

        if eval_fn and eval_points and (step + 1) in eval_points:
            model.eval()
            ppl = eval_fn()
            model.train()
            checkpoints[step + 1] = ppl
            print(f"      eval@{step+1}: PPL={ppl:.3f}", flush=True)

    model.eval(); model.requires_grad_(False)
    return losses, checkpoints


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

    # === Condition A: Joint SVD init ===
    print(f"\n{'='*50}\nCONDITION A: Joint SVD init ({JOINT_STEPS} steps)\n{'='*50}", flush=True)
    _set_seed(SEED)
    svd_model = copy.deepcopy(teacher.model); svd_model.eval()
    for bid in BLOCKS:
        svd_model.transformer.h[bid].mlp = LinearFactorFFN(svd_model.transformer.h[bid].mlp, D_BN, init_mode="svd")
    total_params = sum(svd_model.transformer.h[bid].mlp.n_params for bid in BLOCKS)
    print(f"  Total params: {total_params:,}", flush=True)

    # Eval at step 0 (SVD init quality)
    pre_svd = _evaluate_model(teacher, svd_model, corpus, batch_size=BATCH_SIZE)
    print(f"  SVD init PPL (step 0): {pre_svd['perplexity']:.3f}", flush=True)

    def eval_svd():
        return _evaluate_model(teacher, svd_model, corpus, batch_size=BATCH_SIZE)["perplexity"]

    svd_losses, svd_checkpoints = train_joint(svd_model, corpus.probe_holdout, JOINT_STEPS, LR, device, SEED,
                                               eval_fn=eval_svd, eval_points=EVAL_POINTS)
    post_svd = _evaluate_model(teacher, svd_model, corpus, batch_size=BATCH_SIZE)
    print(f"  Final PPL: {post_svd['perplexity']:.3f}", flush=True)
    results["joint_svd"] = {"pre": pre_svd["perplexity"], "post": post_svd["perplexity"],
                            "checkpoints": svd_checkpoints, "params": total_params}
    del svd_model

    # === Condition B: Joint random init ===
    print(f"\n{'='*50}\nCONDITION B: Joint random init ({JOINT_STEPS} steps)\n{'='*50}", flush=True)
    _set_seed(SEED)
    rand_model = copy.deepcopy(teacher.model); rand_model.eval()
    for bid in BLOCKS:
        rand_model.transformer.h[bid].mlp = LinearFactorFFN(rand_model.transformer.h[bid].mlp, D_BN, init_mode="random")

    pre_rand = _evaluate_model(teacher, rand_model, corpus, batch_size=BATCH_SIZE)
    print(f"  Random init PPL (step 0): {pre_rand['perplexity']:.3f}", flush=True)

    def eval_rand():
        return _evaluate_model(teacher, rand_model, corpus, batch_size=BATCH_SIZE)["perplexity"]

    rand_losses, rand_checkpoints = train_joint(rand_model, corpus.probe_holdout, JOINT_STEPS, LR, device, SEED,
                                                 eval_fn=eval_rand, eval_points=EVAL_POINTS)
    post_rand = _evaluate_model(teacher, rand_model, corpus, batch_size=BATCH_SIZE)
    print(f"  Final PPL: {post_rand['perplexity']:.3f}", flush=True)
    results["joint_random"] = {"pre": pre_rand["perplexity"], "post": post_rand["perplexity"],
                               "checkpoints": rand_checkpoints, "params": total_params}
    del rand_model

    # === Condition C: topk+lstsq+polish baseline ===
    print(f"\n{'='*50}\nCONDITION C: topk+lstsq+polish ({POLISH_STEPS} steps)\n{'='*50}", flush=True)
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
        dataset_name=atlas.dataset_name, layers=tuple(layers), metadata={"sequence_length": atlas.sequence_length})
    topk_model = swap_compressed_ffns(teacher, pack, inplace=False); topk_model.eval()

    pre_topk = _evaluate_model(teacher, topk_model, corpus, batch_size=BATCH_SIZE)
    print(f"  topk pre-polish PPL: {pre_topk['perplexity']:.3f}", flush=True)

    topk_losses, _ = train_joint(topk_model, corpus.probe_holdout, POLISH_STEPS, LR, device, SEED)
    post_topk = _evaluate_model(teacher, topk_model, corpus, batch_size=BATCH_SIZE)
    topk_params = sum(p.numel() for bid in BLOCKS
                      for p in topk_model.transformer.h[bid].mlp.c_proj.parameters())
    print(f"  topk post-polish PPL: {post_topk['perplexity']:.3f} (params: {topk_params:,})", flush=True)
    results["topk_baseline"] = {"pre": pre_topk["perplexity"], "post": post_topk["perplexity"],
                                "params": topk_params}
    del topk_model

    # === Verdict ===
    print(f"\n{'='*50}\nVERDICT\n{'='*50}", flush=True)
    svd_ppl = results["joint_svd"]["post"]
    rand_ppl = results["joint_random"]["post"]
    topk_ppl = results["topk_baseline"]["post"]

    print(f"  Joint SVD:     {svd_ppl:.3f} ({total_params:,} params)", flush=True)
    print(f"  Joint random:  {rand_ppl:.3f} ({total_params:,} params)", flush=True)
    print(f"  topk+polish:   {topk_ppl:.3f} ({topk_params:,} params)", flush=True)
    print(f"  Float:         {bl['perplexity']:.3f}", flush=True)

    rel = (svd_ppl - topk_ppl) / topk_ppl * 100
    if svd_ppl <= topk_ppl * 0.98:
        verdict = "CONFIRMED"
        detail = f"Joint SVD ({svd_ppl:.1f}) beats topk ({topk_ppl:.1f}) by {-rel:.1f}%"
    elif svd_ppl >= topk_ppl * 0.98:
        verdict = "REFUTED"
        detail = f"Joint SVD ({svd_ppl:.1f}) within 2% of topk ({topk_ppl:.1f}), rel={rel:+.1f}%"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"rel={rel:+.1f}%"

    print(f"  VERDICT: {verdict}\n  Detail: {detail}", flush=True)

    output = {
        "cycle": 149, "item": 159, "seed": SEED,
        "float_ppl": bl["perplexity"], "d_bn": D_BN,
        "results": results, "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_149_joint_stacking.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved JSON\nDone.", flush=True)
