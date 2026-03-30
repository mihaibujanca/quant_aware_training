"""Cycle 133: Direct regression hybrid — GPT-2 FFN k=4 (item 127).

Hybrid: direct rank-256 regression for middle blocks (4,7) + topk k=512
for edge blocks (1,11). Compare stacked PPL vs all-topk and all-direct.
3 seeds, 500 Adam polish.

Verdict:
  CONFIRMED    if hybrid outperforms both baselines by >=2% relative PPL
  REFUTED      if all-topk matches or beats hybrid
  INCONCLUSIVE if hybrid beats one but not the other
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

SEEDS = [42, 123, 7]
BLOCKS = [7, 5, 4, 1]
D_MODEL = 768
DIRECT_RANK = 256
TOPK_K = 512
HIDDEN_FRACTION = TOPK_K / 3072  # 512/3072
POLISH_STEPS = 500
POLISH_LR = 1e-4
MAX_TOKENS_PER_LAYER = 4096
BATCH_SIZE = 4
SEQ_LEN = 128
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200
EVAL_SEQS = 200

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class DirectRegressionFFN(nn.Module):
    """Rank-r linear approximation of FFN: output = (input @ V_r) @ A + bias."""
    def __init__(self, V_r, A, bias):
        super().__init__()
        self.V_r = nn.Parameter(V_r, requires_grad=False)  # (768, r)
        self.A = nn.Parameter(A)                             # (r, 768)
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        out = (x @ self.V_r) @ self.A
        if self.bias is not None:
            out = out + self.bias
        return out

    # Mimic c_proj for polish compatibility
    @property
    def c_proj(self):
        return self

    @property
    def weight(self):
        return self.A  # polishable

    # Make it look like an MLP with c_fc and c_proj for compatibility
    @property
    def c_fc(self):
        return None


@torch.no_grad()
def fit_direct_regression(model, block_id, tokens, rank, device):
    """Fit rank-r direct regression replacement for FFN block."""
    inputs_list, outputs_list = [], []

    # Hook to collect FFN input and output
    block = model.transformer.h[block_id]
    ffn_inputs, ffn_outputs = [], []

    def in_hook(mod, inp):
        ffn_inputs.append(inp[0].detach().cpu())

    def out_hook(mod, inp, out):
        ffn_outputs.append(out.detach().cpu())

    h1 = block.mlp.register_forward_pre_hook(in_hook)
    h2 = block.mlp.register_forward_hook(out_hook)

    for i in range(0, tokens.shape[0], BATCH_SIZE):
        batch = tokens[i:i+BATCH_SIZE].to(device)
        model(batch[:, :-1])

    h1.remove(); h2.remove()

    X = torch.cat(ffn_inputs, dim=0).reshape(-1, D_MODEL).float()   # (n, 768)
    Y = torch.cat(ffn_outputs, dim=0).reshape(-1, D_MODEL).float()  # (n, 768)

    # SVD of X, take top-r right singular vectors
    _, _, Vt = torch.linalg.svd(X, full_matrices=False)
    V_r = Vt[:rank, :].T  # (768, r)

    # Project and solve lstsq
    X_r = X @ V_r  # (n, r)
    A, _, _, _ = torch.linalg.lstsq(X_r, Y)  # (r, 768)

    # Bias = mean(Y - X_r @ A)
    residual = Y - X_r @ A
    bias = residual.mean(dim=0)

    return V_r, A, bias


def _param_count(w, b=None):
    return w.numel() + (0 if b is None else b.numel())


def _derive_block_pack(atlas, budget, block_ids):
    compressed_layers = []
    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
            continue
        k = _layer_target_hidden(layer, budget)
        selected = _select_hidden_units(layer, budget, k)
        sw = _standard_weight(layer.handle.c_fc).float()
        sb = _standard_bias(layer.handle.c_fc)
        use_ob = _standard_bias(layer.handle.c_proj) is not None
        stw = sw[selected].contiguous()
        stb = None if sb is None else sb[selected].contiguous()
        h = layer.hidden_sample[:, selected].float()
        o = layer.output_sample.float()
        swo, sbo = _fit_output_projection(h, o, use_bias=use_ob)
        sp = _param_count(sw, sb) + _param_count(_standard_weight(layer.handle.c_proj), _standard_bias(layer.handle.c_proj))
        cp = _param_count(stw, stb) + _param_count(swo, sbo)
        compressed_layers.append(CompressedFfnLayer(
            layer_id=layer.layer_id, activation_kind=layer.activation_kind,
            source_hidden_dim=layer.hidden_dim, compressed_hidden_dim=int(k),
            c_fc_weight=stw, c_fc_bias=stb, c_proj_weight=swo, c_proj_bias=sbo,
            selected_indices=selected, source_param_count=sp, compressed_param_count=cp,
            notes={"selection_mode": budget.selection_mode},
        ))
    return CompressedFfnPack(method="geometric", model_family=atlas.model_family,
        dataset_name=atlas.dataset_name, layers=tuple(compressed_layers),
        metadata={"sequence_length": atlas.sequence_length})


def polish_model(model, tokens, n_steps, lr, device, seed, block_ids):
    """Polish all compressed blocks jointly under CE loss."""
    model.requires_grad_(False)
    params = []
    for bid in block_ids:
        mlp = model.transformer.h[bid].mlp
        if isinstance(mlp, DirectRegressionFFN):
            mlp.A.requires_grad_(True)
            params.append(mlp.A)
            if mlp.bias is not None:
                mlp.bias.requires_grad_(True)
                params.append(mlp.bias)
        elif hasattr(mlp, 'c_proj'):
            mlp.c_proj.weight.requires_grad_(True)
            params.append(mlp.c_proj.weight)
            if mlp.c_proj.bias is not None:
                mlp.c_proj.bias.requires_grad_(True)
                params.append(mlp.c_proj.bias)

    if not params:
        return []

    opt = torch.optim.Adam(params, lr=lr)
    model.train()
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
    model.eval(); model.requires_grad_(False)
    return losses


if __name__ == "__main__":
    print("Loading GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    device = teacher.device

    _set_seed(42)
    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher, text_path=str(OWT_PATH), sequence_length=SEQ_LEN,
        probe_train_sequences=PROBE_TRAIN_SEQS, probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
        eval_val_sequences=EVAL_SEQS, dataset_name="openwebtext", shuffle_seed=42,
    ))

    bl = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    print(f"  Float PPL: {bl['perplexity']:.3f}", flush=True)

    # Probe geometry for topk
    budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION)
    atlas = probe_ffn_geometry(teacher, corpus, layer_ids=tuple(sorted(BLOCKS)),
                                max_tokens_per_layer=MAX_TOKENS_PER_LAYER, batch_size=BATCH_SIZE)

    CONDITIONS = {
        "all_topk": {1: "topk", 4: "topk", 5: "topk", 7: "topk"},
        "all_direct": {1: "direct", 4: "direct", 5: "direct", 7: "direct"},
        "hybrid": {1: "topk", 4: "direct", 5: "topk", 7: "direct"},
    }

    results = {}
    for seed in SEEDS:
        print(f"\n{'='*50}\nSEED {seed}\n{'='*50}", flush=True)

        for cond_name, block_methods in CONDITIONS.items():
            print(f"\n  {cond_name}:", flush=True)
            _set_seed(seed)

            # Build model with per-block method
            model = copy.deepcopy(teacher.model)
            model.eval()

            topk_blocks = [bid for bid, m in block_methods.items() if m == "topk"]
            direct_blocks = [bid for bid, m in block_methods.items() if m == "direct"]

            # Apply topk to topk blocks
            if topk_blocks:
                pack = _derive_block_pack(atlas, budget, topk_blocks)
                topk_model = swap_compressed_ffns(teacher, pack, inplace=False)
                for bid in topk_blocks:
                    model.transformer.h[bid].mlp = topk_model.transformer.h[bid].mlp
                del topk_model

            # Apply direct regression to direct blocks
            for bid in direct_blocks:
                V_r, A, bias = fit_direct_regression(teacher.model, bid, corpus.probe_holdout, DIRECT_RANK, device)
                dr = DirectRegressionFFN(V_r, A, bias)
                model.transformer.h[bid].mlp = dr

            # Pre-polish eval
            pre_m = _evaluate_model(teacher, model, corpus, batch_size=BATCH_SIZE)
            pre_ppl = pre_m["perplexity"]
            print(f"    pre-polish PPL: {pre_ppl:.3f}", flush=True)

            # Polish
            losses = polish_model(model, corpus.probe_holdout, POLISH_STEPS, POLISH_LR, device, seed, BLOCKS)
            if losses:
                print(f"    polish: {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)

            # Post-polish eval
            post_m = _evaluate_model(teacher, model, corpus, batch_size=BATCH_SIZE)
            post_ppl = post_m["perplexity"]
            print(f"    post-polish PPL: {post_ppl:.3f}", flush=True)

            results[f"{seed}_{cond_name}"] = {
                "seed": seed, "condition": cond_name,
                "pre_ppl": pre_ppl, "post_ppl": post_ppl,
            }
            del model

    # Summary
    print(f"\n{'='*50}\nSUMMARY\n{'='*50}", flush=True)
    for cond in CONDITIONS:
        pre_vals = [results[f"{s}_{cond}"]["pre_ppl"] for s in SEEDS]
        post_vals = [results[f"{s}_{cond}"]["post_ppl"] for s in SEEDS]
        print(f"  {cond:12s}: pre={np.mean(pre_vals):.2f}±{np.std(pre_vals):.2f}, "
              f"post={np.mean(post_vals):.2f}±{np.std(post_vals):.2f}", flush=True)

    # Verdict
    topk_post = np.mean([results[f"{s}_all_topk"]["post_ppl"] for s in SEEDS])
    direct_post = np.mean([results[f"{s}_all_direct"]["post_ppl"] for s in SEEDS])
    hybrid_post = np.mean([results[f"{s}_hybrid"]["post_ppl"] for s in SEEDS])

    beats_topk = (topk_post - hybrid_post) / topk_post * 100
    beats_direct = (direct_post - hybrid_post) / direct_post * 100

    if beats_topk >= 2 and beats_direct >= 2:
        verdict = "CONFIRMED"
        detail = f"Hybrid beats all-topk by {beats_topk:.1f}% and all-direct by {beats_direct:.1f}%"
    elif beats_topk < 0:
        verdict = "REFUTED"
        detail = f"All-topk beats hybrid by {-beats_topk:.1f}%"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"Hybrid vs topk: {beats_topk:+.1f}%, vs direct: {beats_direct:+.1f}%"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  Detail: {detail}", flush=True)

    # Save
    output = {
        "cycle": 133, "item": 127, "seeds": SEEDS,
        "direct_rank": DIRECT_RANK, "topk_k": TOPK_K,
        "float_ppl": bl["perplexity"],
        "results": results, "verdict": verdict, "verdict_detail": detail,
        "summary": {
            "all_topk_post": topk_post, "all_direct_post": direct_post,
            "hybrid_post": hybrid_post,
            "hybrid_vs_topk_pct": beats_topk, "hybrid_vs_direct_pct": beats_direct,
        },
    }
    with open(RESULTS_DIR / "cycle_133_hybrid_regression.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON", flush=True)
    print("Done.", flush=True)
