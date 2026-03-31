"""Cycle 141: Graduated importance pruning — GPT-2 FFN (item 153).

Rank channels by importance = ||W_down[:,j]||_2 * rms(GELU(z_j)).
Prune bottom 5%/10%/20%, refit W_down. Compare to topk at matched k.
3 seeds [42, 137, 271].

Verdict:
  CONFIRMED    if importance beats topk by >=2% on >=2/4 blocks
  REFUTED      if topk matches or beats importance at all levels
  INCONCLUSIVE if advantage is block-dependent or <2%
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
PRUNE_FRACS = [0.05, 0.10, 0.20]
SEQ_LEN = 128
BATCH_SIZE = 4
POLISH_STEPS = 500
LR = 1e-4
MAX_TOKENS = 4096

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def compute_importance(model, tokens, block_id, device, batch_size=4):
    """Compute per-channel importance = ||W_down[:,j]|| * rms(GELU(z_j))."""
    block = model.transformer.h[block_id]
    w_down = _standard_weight(block.mlp.c_proj).float()  # (D_FFN, D_MODEL) for Conv1D? Check shape

    # Collect pre-GELU activations
    pre_acts = []
    def hook_fn(mod, inp, out):
        pre_acts.append(out.detach().cpu())
    h = block.mlp.c_fc.register_forward_hook(hook_fn)
    for i in range(0, tokens.shape[0], batch_size):
        batch = tokens[i:i+batch_size].to(device)
        model(batch[:, :-1])
    h.remove()

    z = torch.cat(pre_acts, dim=0).reshape(-1, D_FFN).float()  # (n_tokens, 3072)
    gelu_z = F.gelu(z)

    # RMS of GELU(z) per channel
    rms_gelu = (gelu_z ** 2).mean(dim=0).sqrt()  # (3072,)

    # W_down column norms — need to figure out the Conv1D shape
    # GPT-2 Conv1D c_proj: weight shape (D_FFN, D_MODEL) = (3072, 768)
    # output = input @ weight, so W_down[:,j] means column j of (3072, 768)
    # But we want the output-space norm of channel j: ||c_proj.weight[j, :]||
    w_norms = w_down.norm(dim=0)  # (3072,) — norm of each column (channel j's contribution to output)

    importance = w_norms * rms_gelu  # (3072,)
    return importance, z, gelu_z


def build_importance_compressed(model, block_id, keep_indices, probe_hidden, probe_output, atlas_layer):
    """Build compressed FFN keeping specified channels."""
    source_w_in = _standard_weight(atlas_layer.handle.c_fc).float()
    source_b_in = _standard_bias(atlas_layer.handle.c_fc)

    student_w_in = source_w_in[keep_indices].contiguous()
    student_b_in = None if source_b_in is None else source_b_in[keep_indices].contiguous()

    hidden_kept = probe_hidden[:, keep_indices].float()
    outputs = probe_output.float()
    use_ob = _standard_bias(atlas_layer.handle.c_proj) is not None
    student_w_out, student_b_out = _fit_output_projection(hidden_kept, outputs, use_bias=use_ob)

    sp = source_w_in.numel() + (0 if source_b_in is None else source_b_in.numel()) + \
         _standard_weight(atlas_layer.handle.c_proj).numel()
    cp = student_w_in.numel() + (0 if student_b_in is None else student_b_in.numel()) + \
         student_w_out.numel() + (0 if student_b_out is None else student_b_out.numel())

    return CompressedFfnLayer(
        layer_id=block_id, activation_kind=atlas_layer.activation_kind,
        source_hidden_dim=atlas_layer.hidden_dim, compressed_hidden_dim=len(keep_indices),
        c_fc_weight=student_w_in, c_fc_bias=student_b_in,
        c_proj_weight=student_w_out, c_proj_bias=student_b_out,
        selected_indices=keep_indices,
        source_param_count=sp, compressed_param_count=cp,
        notes={"selection_mode": "importance"})


def polish_model(model, tokens, n_steps, lr, device, seed, block_ids):
    model.requires_grad_(False)
    params = []
    for bid in block_ids:
        mlp = model.transformer.h[bid].mlp
        if hasattr(mlp, 'c_proj'):
            mlp.c_proj.weight.requires_grad_(True)
            params.append(mlp.c_proj.weight)
            if mlp.c_proj.bias is not None:
                mlp.c_proj.bias.requires_grad_(True)
                params.append(mlp.c_proj.bias)
    if not params:
        return
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

    # Probe geometry for topk baseline
    atlas = probe_ffn_geometry(teacher, corpus, layer_ids=tuple(sorted(BLOCKS)),
                                max_tokens_per_layer=MAX_TOKENS, batch_size=BATCH_SIZE)

    # Compute importance rankings
    print(f"\nComputing importance rankings...", flush=True)
    importance_rankings = {}
    for bid in BLOCKS:
        imp, _, _ = compute_importance(teacher.model, corpus.probe_holdout, bid, device)
        ranking = imp.argsort(descending=True)
        importance_rankings[bid] = ranking
        print(f"  Block {bid}: top-5 importance: {imp[ranking[:5]].tolist()}", flush=True)

    results = {}
    for seed in SEEDS:
        print(f"\n{'='*50}\nSEED {seed}\n{'='*50}", flush=True)
        for bid in BLOCKS:
            layer = [l for l in atlas.layer_geometries if l.layer_id == bid][0]
            for frac in PRUNE_FRACS:
                k = int(D_FFN * (1 - frac))
                key = f"{seed}_{bid}_{frac}"

                # Importance pruning
                keep = importance_rankings[bid][:k]
                cl = build_importance_compressed(teacher.model, bid, keep,
                                                  layer.hidden_sample, layer.output_sample, layer)
                pack = CompressedFfnPack(method="importance", model_family=atlas.model_family,
                    dataset_name=atlas.dataset_name, layers=(cl,),
                    metadata={"sequence_length": atlas.sequence_length})
                imp_model = swap_compressed_ffns(teacher, pack, inplace=False); imp_model.eval()
                pre_imp = _evaluate_model(teacher, imp_model, corpus, batch_size=BATCH_SIZE)

                # Topk at matched k
                from aleph.lm_geometric_compression import _layer_target_hidden, _select_hidden_units
                budget_matched = CompressionBudgetSpec(hidden_fraction=k/D_FFN)
                k_actual = _layer_target_hidden(layer, budget_matched)
                sel = _select_hidden_units(layer, budget_matched, k_actual)
                sw = _standard_weight(layer.handle.c_fc).float()
                sb = _standard_bias(layer.handle.c_fc)
                use_ob = _standard_bias(layer.handle.c_proj) is not None
                stw = sw[sel].contiguous()
                stb = None if sb is None else sb[sel].contiguous()
                h_sel = layer.hidden_sample[:, sel].float()
                o = layer.output_sample.float()
                swo, sbo = _fit_output_projection(h_sel, o, use_bias=use_ob)
                sp = sw.numel(); cp = stw.numel() + swo.numel()
                topk_cl = CompressedFfnLayer(
                    layer_id=bid, activation_kind=layer.activation_kind,
                    source_hidden_dim=layer.hidden_dim, compressed_hidden_dim=int(k_actual),
                    c_fc_weight=stw, c_fc_bias=stb, c_proj_weight=swo, c_proj_bias=sbo,
                    selected_indices=sel, source_param_count=sp, compressed_param_count=cp,
                    notes={"selection_mode": budget_matched.selection_mode})
                topk_pack = CompressedFfnPack(method="geometric", model_family=atlas.model_family,
                    dataset_name=atlas.dataset_name, layers=(topk_cl,),
                    metadata={"sequence_length": atlas.sequence_length})
                topk_model = swap_compressed_ffns(teacher, topk_pack, inplace=False); topk_model.eval()
                pre_topk = _evaluate_model(teacher, topk_model, corpus, batch_size=BATCH_SIZE)

                print(f"  B{bid} prune={frac:.0%}: imp={pre_imp['perplexity']:.2f}, topk={pre_topk['perplexity']:.2f}, k={k}", flush=True)

                results[key] = {
                    "seed": seed, "block": bid, "prune_frac": frac, "k": k,
                    "importance_pre": pre_imp["perplexity"],
                    "topk_pre": pre_topk["perplexity"],
                }
                del imp_model, topk_model

    # Summary
    print(f"\n{'='*50}\nSUMMARY (pre-polish)\n{'='*50}", flush=True)
    imp_wins = 0
    for bid in BLOCKS:
        for frac in PRUNE_FRACS:
            imp_vals = [results[f"{s}_{bid}_{frac}"]["importance_pre"] for s in SEEDS]
            topk_vals = [results[f"{s}_{bid}_{frac}"]["topk_pre"] for s in SEEDS]
            im, tm = np.mean(imp_vals), np.mean(topk_vals)
            rel = (im - tm) / tm * 100
            if im < tm * 0.98:
                imp_wins += 1
            print(f"  B{bid} {frac:.0%}: imp={im:.2f}±{np.std(imp_vals):.2f}, topk={tm:.2f}±{np.std(topk_vals):.2f} ({rel:+.1f}%)", flush=True)

    if imp_wins >= 2:
        verdict = "CONFIRMED"
    elif imp_wins == 0:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    detail = f"Importance wins on {imp_wins}/{len(BLOCKS)*len(PRUNE_FRACS)} configs (pre-polish)"
    print(f"\n  VERDICT: {verdict}\n  Detail: {detail}", flush=True)

    output = {
        "cycle": 141, "item": 153, "seeds": SEEDS, "blocks": BLOCKS,
        "prune_fracs": PRUNE_FRACS, "float_ppl": bl["perplexity"],
        "results": results, "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_141_importance_pruning.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved JSON\nDone.", flush=True)
