"""Cycle 132: Attention projection pruning — GPT-2 (item 126).

SVD of composed W_V @ W_O per head in high-amplification layers.
Apply rank-r approximation (r in {8, 11, 16} out of 64 head dim).
Measure stacked PPL with FFN k=4 compression + attention pruning.
3 seeds [42, 123, 7].

Verdict:
  CONFIRMED    if amp reduction >=20% AND stacked PPL improves by >=2%
  REFUTED      if stacked PPL degrades (pruned > unpruned + 1)
  INCONCLUSIVE if amp reduced but PPL within ±1
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
FFN_BLOCKS = [7, 5, 4, 1]
TARGET_LAYERS = [0, 1, 3, 4, 5, 10]  # attn_frac > 1.0 from cycle 104
RANKS = [8, 11, 16]
N_HEADS = 12
HEAD_DIM = 64
D_MODEL = 768
SEQ_LEN = 128
BATCH_SIZE = 4
HIDDEN_FRACTION = 1 / 6
POLISH_STEPS = 500
POLISH_LR = 1e-4
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200
EVAL_SEQS = 200

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _param_count(w, b=None):
    return w.numel() + (0 if b is None else b.numel())


def apply_composed_svd_pruning(model, layer_ids, rank):
    """Apply rank-r SVD to composed W_V @ W_O per head, modifying c_proj weights in-place.

    For head h: M_h = V_h @ O_h (768, 768). SVD truncate to rank r.
    O_h' = pinv(V_h) @ M_h_r keeps V_h unchanged, only modifies c_proj.
    """
    for lid in layer_ids:
        attn = model.transformer.h[lid].attn
        c_attn_w = attn.c_attn.weight.data  # (768, 2304) Conv1D
        c_proj_w = attn.c_proj.weight.data   # (768, 768) Conv1D

        for h in range(N_HEADS):
            # V_h: columns 1536+h*64:1536+(h+1)*64 of c_attn weight
            v_start = 1536 + h * HEAD_DIM
            V_h = c_attn_w[:, v_start:v_start + HEAD_DIM].float()  # (768, 64)

            # O_h: rows h*64:(h+1)*64 of c_proj weight
            o_start = h * HEAD_DIM
            O_h = c_proj_w[o_start:o_start + HEAD_DIM, :].float()  # (64, 768)

            # Compose
            M_h = V_h @ O_h  # (768, 768), rank 64

            # SVD, truncate to rank r
            U, S, Vt = torch.linalg.svd(M_h, full_matrices=False)
            r = min(rank, len(S))
            M_h_r = U[:, :r] @ torch.diag(S[:r]) @ Vt[:r, :]

            # O_h' = pinv(V_h) @ M_h_r
            V_pinv = torch.linalg.pinv(V_h)  # (64, 768)
            O_h_new = V_pinv @ M_h_r  # (64, 768)

            # Replace in c_proj
            c_proj_w[o_start:o_start + HEAD_DIM, :] = O_h_new.to(c_proj_w.dtype)


# Compression helpers (reused from cycle 125/130)
def _derive_block_pack(atlas, budget, block_ids):
    compressed_layers = []
    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
            continue
        k = _layer_target_hidden(layer, budget)
        selected = _select_hidden_units(layer, budget, k)
        sw_in = _standard_weight(layer.handle.c_fc).float()
        sb_in = _standard_bias(layer.handle.c_fc)
        use_ob = _standard_bias(layer.handle.c_proj) is not None
        st_w_in = sw_in[selected].contiguous()
        st_b_in = None if sb_in is None else sb_in[selected].contiguous()
        h = layer.hidden_sample[:, selected].float()
        o = layer.output_sample.float()
        st_w_out, st_b_out = _fit_output_projection(h, o, use_bias=use_ob)
        sp = _param_count(sw_in, sb_in) + _param_count(_standard_weight(layer.handle.c_proj), _standard_bias(layer.handle.c_proj))
        cp = _param_count(st_w_in, st_b_in) + _param_count(st_w_out, st_b_out)
        compressed_layers.append(CompressedFfnLayer(
            layer_id=layer.layer_id, activation_kind=layer.activation_kind,
            source_hidden_dim=layer.hidden_dim, compressed_hidden_dim=int(k),
            c_fc_weight=st_w_in, c_fc_bias=st_b_in,
            c_proj_weight=st_w_out, c_proj_bias=st_b_out,
            selected_indices=selected, source_param_count=sp, compressed_param_count=cp,
            notes={"selection_mode": budget.selection_mode},
        ))
    return CompressedFfnPack(method="geometric", model_family=atlas.model_family,
        dataset_name=atlas.dataset_name, layers=tuple(compressed_layers),
        metadata={"sequence_length": atlas.sequence_length})


def _polish_block(model, bid, tokens, n_steps, lr, device, seed):
    mlp = model.transformer.h[bid].mlp
    if not hasattr(mlp, "c_proj"):
        return []
    model.requires_grad_(False)
    mlp.c_proj.weight.requires_grad_(True)
    if mlp.c_proj.bias is not None:
        mlp.c_proj.bias.requires_grad_(True)
    params = [mlp.c_proj.weight] + ([mlp.c_proj.bias] if mlp.c_proj.bias is not None else [])
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


def _extract_polished(model, bid):
    mlp = model.transformer.h[bid].mlp
    return {"w": mlp.c_proj.weight.detach().clone(),
            "b": mlp.c_proj.bias.detach().clone() if mlp.c_proj.bias is not None else None}


def _inject_polished(model, bid, p):
    with torch.no_grad():
        model.transformer.h[bid].mlp.c_proj.weight.copy_(p["w"])
        if p["b"] is not None and model.transformer.h[bid].mlp.c_proj.bias is not None:
            model.transformer.h[bid].mlp.c_proj.bias.copy_(p["b"])


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

    # Probe and build compressed FFN
    budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION)
    atlas = probe_ffn_geometry(teacher, corpus, layer_ids=tuple(sorted(FFN_BLOCKS)),
                                max_tokens_per_layer=MAX_TOKENS_PER_LAYER, batch_size=BATCH_SIZE)

    results = {}
    for seed in SEEDS:
        print(f"\n{'='*50}\nSEED {seed}\n{'='*50}", flush=True)

        # Build FFN-only compressed model
        polished = {}
        for bid in FFN_BLOCKS:
            _set_seed(seed)
            p = _derive_block_pack(atlas, budget, [bid])
            s = swap_compressed_ffns(teacher, p, inplace=False); s.eval()
            _polish_block(s, bid, corpus.probe_holdout, POLISH_STEPS, POLISH_LR, device, seed)
            polished[bid] = _extract_polished(s, bid)
            del s

        _set_seed(seed)
        fp = _derive_block_pack(atlas, budget, FFN_BLOCKS)
        ffn_model = swap_compressed_ffns(teacher, fp, inplace=False); ffn_model.eval()
        for bid in FFN_BLOCKS:
            _inject_polished(ffn_model, bid, polished[bid])

        ffn_m = _evaluate_model(teacher, ffn_model, corpus, batch_size=BATCH_SIZE)
        ffn_ppl = ffn_m["perplexity"]
        print(f"  FFN-only PPL: {ffn_ppl:.3f}", flush=True)

        for rank in RANKS:
            # FFN + attention SVD pruning
            combined_model = copy.deepcopy(ffn_model)
            apply_composed_svd_pruning(combined_model, TARGET_LAYERS, rank)
            combined_m = _evaluate_model(teacher, combined_model, corpus, batch_size=BATCH_SIZE)
            combined_ppl = combined_m["perplexity"]
            delta = combined_ppl - ffn_ppl
            print(f"  FFN + attn SVD r={rank}: PPL={combined_ppl:.3f} (delta={delta:+.3f})", flush=True)

            results[f"{seed}_{rank}"] = {
                "seed": seed, "rank": rank,
                "ffn_ppl": ffn_ppl, "combined_ppl": combined_ppl,
                "delta": delta,
            }
            del combined_model
        del ffn_model

    # Summary
    print(f"\n{'='*50}\nSUMMARY\n{'='*50}", flush=True)
    for rank in RANKS:
        deltas = [results[f"{s}_{rank}"]["delta"] for s in SEEDS]
        mean_d, std_d = np.mean(deltas), np.std(deltas)
        ffn_ppls = [results[f"{s}_{rank}"]["ffn_ppl"] for s in SEEDS]
        comb_ppls = [results[f"{s}_{rank}"]["combined_ppl"] for s in SEEDS]
        print(f"  r={rank}: FFN={np.mean(ffn_ppls):.2f}, FFN+SVD={np.mean(comb_ppls):.2f}, "
              f"delta={mean_d:+.2f}±{std_d:.2f}", flush=True)

    # Verdict: check if stacked PPL improves or degrades
    best_rank = min(RANKS, key=lambda r: np.mean([results[f"{s}_{r}"]["delta"] for s in SEEDS]))
    best_deltas = [results[f"{s}_{best_rank}"]["delta"] for s in SEEDS]
    mean_best = np.mean(best_deltas)
    ffn_mean = np.mean([results[f"{s}_{best_rank}"]["ffn_ppl"] for s in SEEDS])
    threshold_2pct = ffn_mean * 0.02

    if mean_best < -threshold_2pct:
        verdict = "CONFIRMED"
        detail = f"r={best_rank} improves PPL by {-mean_best:.2f} (>{threshold_2pct:.2f} = 2% of {ffn_mean:.1f})"
    elif mean_best > 1.0:
        verdict = "REFUTED"
        detail = f"Best r={best_rank} degrades PPL by {mean_best:.2f} > 1.0"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"Best r={best_rank} delta={mean_best:+.2f} within ±1"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  Detail: {detail}", flush=True)

    # Save
    output = {
        "cycle": 132, "item": 126, "seeds": SEEDS, "ranks": RANKS,
        "target_layers": TARGET_LAYERS, "float_ppl": bl["perplexity"],
        "results": results, "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_132_attn_pruning.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON", flush=True)
    print("Done.", flush=True)
