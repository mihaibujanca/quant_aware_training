"""Cycle 143: Eigenstructure-aware channel selection — GPT-2 FFN (item 155).

Three methods at k=512 (6x): topk, importance, whitened importance.
4 blocks [1, 4, 7, 11], 3 seeds. Pre-polish only (polish erases differences).

Verdict:
  CONFIRMED    if whitened beats topk by >=2% on >=2/4 blocks
  REFUTED      if whitened matches topk
  INCONCLUSIVE if <2% or block-dependent
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

SEEDS = [42, 137, 271]
BLOCKS = [1, 4, 7, 11]
D_MODEL = 768
D_FFN = 3072
K = 512
SEQ_LEN = 128
BATCH_SIZE = 4
HIDDEN_FRACTION = K / D_FFN
MAX_TOKENS = 4096

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def compute_activations(model, tokens, block_id, device, batch_size=4):
    """Collect post-GELU activations for a block."""
    acts = []
    def hook_fn(mod, inp, out):
        acts.append(out.detach().cpu())
    h = model.transformer.h[block_id].mlp.c_fc.register_forward_hook(hook_fn)
    for i in range(0, tokens.shape[0], batch_size):
        batch = tokens[i:i+batch_size].to(device)
        model(batch[:, :-1])
    h.remove()
    z = torch.cat(acts, dim=0).reshape(-1, D_FFN).float()
    return F.gelu(z)


def compute_importance(gelu_z, w_down):
    """Raw importance = ||W_down[:,j]|| * rms(act_j)."""
    w_norms = w_down.norm(dim=0)  # (3072,)
    rms = (gelu_z ** 2).mean(dim=0).sqrt()  # (3072,)
    return w_norms * rms


def compute_whitened_importance(gelu_z, w_down, eps=1e-6):
    """Whitened importance using covariance C^{-1/2}."""
    n = gelu_z.shape[0]
    h_c = gelu_z - gelu_z.mean(dim=0, keepdim=True)
    C = (h_c.T @ h_c) / (n - 1)  # (3072, 3072)

    # Eigendecompose
    eigvals, eigvecs = torch.linalg.eigh(C)
    eigvals = eigvals.clamp(min=eps * eigvals.max())

    # C^{-1/2} columns
    C_inv_sqrt = eigvecs @ torch.diag(1.0 / eigvals.sqrt()) @ eigvecs.T  # (3072, 3072)

    # Whitened importance_j = ||C^{-1/2}[:, j]|| * rms(act_j)
    col_norms = C_inv_sqrt.norm(dim=0)  # (3072,)
    rms = (gelu_z ** 2).mean(dim=0).sqrt()
    w_norms = w_down.norm(dim=0)

    return col_norms * rms * w_norms


def build_compressed_with_selection(teacher, atlas, block_id, selected_indices):
    """Build single-block compressed model with specified channel selection."""
    layer = [l for l in atlas.layer_geometries if l.layer_id == block_id][0]
    sw = _standard_weight(layer.handle.c_fc).float()
    sb = _standard_bias(layer.handle.c_fc)
    use_ob = _standard_bias(layer.handle.c_proj) is not None
    stw = sw[selected_indices].contiguous()
    stb = None if sb is None else sb[selected_indices].contiguous()
    h = layer.hidden_sample[:, selected_indices].float()
    o = layer.output_sample.float()
    swo, sbo = _fit_output_projection(h, o, use_bias=use_ob)
    sp = sw.numel(); cp = stw.numel() + swo.numel()
    cl = CompressedFfnLayer(
        layer_id=block_id, activation_kind=layer.activation_kind,
        source_hidden_dim=layer.hidden_dim, compressed_hidden_dim=len(selected_indices),
        c_fc_weight=stw, c_fc_bias=stb, c_proj_weight=swo, c_proj_bias=sbo,
        selected_indices=selected_indices, source_param_count=sp, compressed_param_count=cp,
        notes={})
    pack = CompressedFfnPack(method="eigenstructure", model_family=atlas.model_family,
        dataset_name=atlas.dataset_name, layers=(cl,),
        metadata={"sequence_length": atlas.sequence_length})
    return swap_compressed_ffns(teacher, pack, inplace=False)


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

    budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION)
    atlas = probe_ffn_geometry(teacher, corpus, layer_ids=tuple(sorted(BLOCKS)),
                                max_tokens_per_layer=MAX_TOKENS, batch_size=BATCH_SIZE)

    # Compute selection methods for each block
    print(f"\nComputing channel selections...", flush=True)
    selections = {}
    for bid in BLOCKS:
        gelu_z = compute_activations(teacher.model, corpus.probe_holdout, bid, device)
        w_down = _standard_weight(teacher.model.transformer.h[bid].mlp.c_proj).float()

        imp = compute_importance(gelu_z, w_down)
        imp_sel = imp.argsort(descending=True)[:K]

        print(f"  Block {bid}: computing whitened importance (3072x3072 eigendecomp)...", flush=True)
        wh_imp = compute_whitened_importance(gelu_z, w_down)
        wh_sel = wh_imp.argsort(descending=True)[:K]

        # Topk selection
        layer = [l for l in atlas.layer_geometries if l.layer_id == bid][0]
        topk_sel = _select_hidden_units(layer, budget, _layer_target_hidden(layer, budget))

        # Jaccard overlap
        imp_set = set(imp_sel.tolist())
        wh_set = set(wh_sel.tolist())
        topk_set = set(topk_sel.tolist())
        j_imp_topk = len(imp_set & topk_set) / len(imp_set | topk_set)
        j_wh_topk = len(wh_set & topk_set) / len(wh_set | topk_set)
        j_wh_imp = len(wh_set & imp_set) / len(wh_set | imp_set)

        selections[bid] = {
            "topk": topk_sel, "importance": imp_sel, "whitened": wh_sel,
            "jaccard_imp_topk": j_imp_topk, "jaccard_wh_topk": j_wh_topk, "jaccard_wh_imp": j_wh_imp,
        }
        print(f"    Jaccard: imp-topk={j_imp_topk:.3f}, wh-topk={j_wh_topk:.3f}, wh-imp={j_wh_imp:.3f}", flush=True)

    # Evaluate
    results = {}
    for seed in SEEDS:
        print(f"\n{'='*50}\nSEED {seed}\n{'='*50}", flush=True)
        for bid in BLOCKS:
            for method, sel in [("topk", selections[bid]["topk"]),
                                 ("importance", selections[bid]["importance"]),
                                 ("whitened", selections[bid]["whitened"])]:
                model = build_compressed_with_selection(teacher, atlas, bid, sel)
                model.eval()
                m = _evaluate_model(teacher, model, corpus, batch_size=BATCH_SIZE)
                ppl = m["perplexity"]
                results[f"{seed}_{bid}_{method}"] = ppl
                del model
            topk_ppl = results[f"{seed}_{bid}_topk"]
            imp_ppl = results[f"{seed}_{bid}_importance"]
            wh_ppl = results[f"{seed}_{bid}_whitened"]
            print(f"  B{bid}: topk={topk_ppl:.2f}, imp={imp_ppl:.2f}, wh={wh_ppl:.2f}", flush=True)

    # Summary
    print(f"\n{'='*50}\nSUMMARY\n{'='*50}", flush=True)
    wh_wins = 0
    for bid in BLOCKS:
        topk_vals = [results[f"{s}_{bid}_topk"] for s in SEEDS]
        imp_vals = [results[f"{s}_{bid}_importance"] for s in SEEDS]
        wh_vals = [results[f"{s}_{bid}_whitened"] for s in SEEDS]
        tm, im, wm = np.mean(topk_vals), np.mean(imp_vals), np.mean(wh_vals)
        rel_wh_topk = (wm - tm) / tm * 100
        rel_imp_topk = (im - tm) / tm * 100
        if wm < tm * 0.98:
            wh_wins += 1
        print(f"  B{bid}: topk={tm:.2f}, imp={im:.2f}({rel_imp_topk:+.1f}%), wh={wm:.2f}({rel_wh_topk:+.1f}%)", flush=True)

    if wh_wins >= 2:
        verdict = "CONFIRMED"
    elif wh_wins == 0:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    detail = f"Whitened beats topk by >=2% on {wh_wins}/4 blocks"
    print(f"\n  VERDICT: {verdict}\n  Detail: {detail}", flush=True)

    output = {
        "cycle": 143, "item": 155, "seeds": SEEDS, "blocks": BLOCKS, "k": K,
        "float_ppl": bl["perplexity"],
        "results": results,
        "selections": {str(b): {k: v for k, v in selections[b].items()
                                 if k.startswith("jaccard")} for b in BLOCKS},
        "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_143_eigenstructure.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved JSON\nDone.", flush=True)
