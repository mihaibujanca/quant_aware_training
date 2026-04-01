"""Cycle 185 / Item 194: Compression ratio sweep — GPT-2 FFN k={2,4,6,8}.

Sweep topk+lstsq+polish at k=2 (hidden=256, 12x), k=4 (512, 6x), k=6 (768, 4x),
k=8 (1024, 3x) on blocks [1,4,7,11]. 3 seeds. Report PPL vs compression tradeoff.

Verdict:
  CONFIRMED    if PPL monotone with k AND k=6 or k=8 achieves PPL <45
  REFUTED      if k=8 still >55 PPL (compression floor unrelated to ratio)
  INCONCLUSIVE if non-monotonic or only marginal improvement at higher k
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Optional

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
    _select_hidden_units,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [42, 137, 7]
K_VALUES = [2, 4, 6, 8]
HIDDEN_DIMS = {2: 256, 4: 512, 6: 768, 8: 1024}
BLOCKS = [1, 4, 7, 11]
GPT2_FFN_HIDDEN = 3072
GPT2_HIDDEN = 768
SEQ_LEN = 128
EVAL_SEQS = 200
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200
MAX_TOKENS_PER_LAYER = 4096
POLISH_STEPS = 500
LR = 1e-4
BATCH_SIZE = 4

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import random

def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _param_count(w: torch.Tensor, b: Optional[torch.Tensor]) -> int:
    return w.numel() + (0 if b is None else b.numel())


def _find_mlp(model: nn.Module, block_id: int):
    return model.transformer.h[block_id].mlp


def _derive_topk_pack(atlas, block_ids: list[int], hidden_dim: int) -> CompressedFfnPack:
    budget = CompressionBudgetSpec(
        hidden_fraction=hidden_dim / GPT2_FFN_HIDDEN,
        min_hidden=32,
        selection_mode="topk",
        per_layer_hidden={bid: hidden_dim for bid in block_ids},
    )
    compressed_layers = []
    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
            continue
        k = hidden_dim
        selected = _select_hidden_units(layer, budget, k)
        source_w_in = _standard_weight(layer.handle.c_fc).to(dtype=torch.float32)
        source_b_in = _standard_bias(layer.handle.c_fc)
        use_output_bias = _standard_bias(layer.handle.c_proj) is not None
        student_w_in = source_w_in[selected].contiguous()
        student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()
        hidden = layer.hidden_sample[:, selected].to(dtype=torch.float32)
        outputs = layer.output_sample.to(dtype=torch.float32)
        student_w_out, student_b_out = _fit_output_projection(hidden, outputs, use_bias=use_output_bias)
        source_param_count = _param_count(source_w_in, source_b_in) + _param_count(
            _standard_weight(layer.handle.c_proj), _standard_bias(layer.handle.c_proj))
        compressed_param_count = _param_count(student_w_in, student_b_in) + _param_count(student_w_out, student_b_out)
        compressed_layers.append(CompressedFfnLayer(
            layer_id=layer.layer_id,
            activation_kind=layer.activation_kind,
            source_hidden_dim=layer.hidden_dim,
            compressed_hidden_dim=k,
            c_fc_weight=student_w_in,
            c_fc_bias=student_b_in,
            c_proj_weight=student_w_out,
            c_proj_bias=student_b_out,
            selected_indices=selected,
            source_param_count=source_param_count,
            compressed_param_count=compressed_param_count,
            notes={"selection_mode": "topk"},
        ))
    return CompressedFfnPack(
        method="geometric", model_family="gpt2", dataset_name="openwebtext",
        layers=tuple(compressed_layers), metadata={"sequence_length": SEQ_LEN},
    )


def _joint_polish(student_model, block_ids, polish_tokens, n_steps, lr, device, seed):
    if n_steps == 0:
        return []
    student_model.requires_grad_(False)
    params = []
    for bid in block_ids:
        mlp = _find_mlp(student_model, bid)
        if mlp is None or not hasattr(mlp, "c_proj"):
            continue
        w = mlp.c_proj.weight
        w.requires_grad_(True)
        params.append(w)
        b = getattr(mlp.c_proj, "bias", None)
        if b is not None:
            b.requires_grad_(True)
            params.append(b)
    if not params:
        return []
    optimizer = torch.optim.Adam(params, lr=lr)
    student_model.train()
    losses = []
    n_seqs = polish_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(seed)
    for step in range(n_steps):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = polish_tokens[idx].to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        optimizer.zero_grad()
        raw = student_model(inputs)
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    student_model.eval()
    student_model.requires_grad_(False)
    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2 teacher...", flush=True)
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()
device = teacher.device
print(f"  Teacher on device={device}", flush=True)

# Original FFN params per block: c_fc (3072*768+3072) + c_proj (768*3072+768) = 4,722,944 per block
original_ffn_params_per_block = GPT2_FFN_HIDDEN * GPT2_HIDDEN + GPT2_FFN_HIDDEN + GPT2_HIDDEN * GPT2_FFN_HIDDEN + GPT2_HIDDEN
print(f"\nOriginal FFN params/block: {original_ffn_params_per_block:,}", flush=True)

# ---------------------------------------------------------------------------
all_seed_results: dict[int, dict] = {}

for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
    _set_seed(seed)

    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher,
        text_path=str(OWT_PATH),
        sequence_length=SEQ_LEN,
        probe_train_sequences=PROBE_TRAIN_SEQS,
        probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
        eval_val_sequences=EVAL_SEQS,
        dataset_name="openwebtext",
        shuffle_seed=seed,
    ))

    bl = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = float(bl["perplexity"])
    print(f"  Teacher baseline PPL: {baseline_ppl:.3f}", flush=True)

    if not (15.0 <= baseline_ppl <= 65.0):
        print(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [15, 65]", flush=True)
        sys.exit(1)

    seed_results = {"baseline_ppl": baseline_ppl}

    # Probe geometry once per seed (reused for all k values)
    _set_seed(seed)
    atlas = probe_ffn_geometry(
        teacher, corpus,
        layer_ids=tuple(BLOCKS),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )

    for k in K_VALUES:
        hidden_dim = HIDDEN_DIMS[k]
        compressed_params_per_block = hidden_dim * GPT2_HIDDEN + hidden_dim + GPT2_HIDDEN * hidden_dim + GPT2_HIDDEN
        total_compressed_params = len(BLOCKS) * compressed_params_per_block
        compression_ratio = (len(BLOCKS) * original_ffn_params_per_block) / total_compressed_params

        print(f"\n  --- k={k} (hidden={hidden_dim}, {compression_ratio:.1f}x compression) ---", flush=True)

        _set_seed(seed + k)
        pack = _derive_topk_pack(atlas, BLOCKS, hidden_dim)

        # Stacked unpolished
        student = swap_compressed_ffns(teacher, pack, inplace=False)
        student.eval()
        m = _evaluate_model(teacher, student, corpus, batch_size=BATCH_SIZE)
        stacked_raw = float(m["perplexity"])
        print(f"    stacked unpolished: PPL={stacked_raw:.3f}", flush=True)

        # Polish
        _set_seed(seed + k)
        losses = _joint_polish(student, BLOCKS, corpus.probe_holdout, POLISH_STEPS, LR, device, seed + k)
        if losses:
            print(f"    polish loss: {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
        student.eval()
        m = _evaluate_model(teacher, student, corpus, batch_size=BATCH_SIZE)
        stacked_polished = float(m["perplexity"])
        print(f"    stacked polished: PPL={stacked_polished:.3f} | params={total_compressed_params:,} | {compression_ratio:.1f}x", flush=True)
        del student

        seed_results[f"k{k}"] = {
            "hidden_dim": hidden_dim,
            "compressed_params": total_compressed_params,
            "compression_ratio": compression_ratio,
            "stacked_raw": stacked_raw,
            "stacked_polished": stacked_polished,
        }

    all_seed_results[seed] = seed_results


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATE RESULTS\n{'='*60}", flush=True)
print(f"\n  {'k':<4} {'hidden':<8} {'params':<12} {'ratio':<8} {'mean_PPL':<12} {'std_PPL'}", flush=True)
print(f"  {'-'*60}", flush=True)

summary = {}
for k in K_VALUES:
    ppls = [all_seed_results[s][f"k{k}"]["stacked_polished"] for s in SEEDS]
    mean_ppl = sum(ppls) / len(ppls)
    std_ppl = (sum((p - mean_ppl)**2 for p in ppls) / len(ppls)) ** 0.5
    hidden_dim = HIDDEN_DIMS[k]
    compressed_params = all_seed_results[SEEDS[0]][f"k{k}"]["compressed_params"]
    ratio = all_seed_results[SEEDS[0]][f"k{k}"]["compression_ratio"]
    summary[k] = {"mean_ppl": mean_ppl, "std_ppl": std_ppl, "params": compressed_params, "ratio": ratio}
    print(f"  {k:<4} {hidden_dim:<8} {compressed_params:<12,} {ratio:<8.1f} {mean_ppl:<12.2f} ±{std_ppl:.2f}", flush=True)

# Verdict
ppl_vals = [summary[k]["mean_ppl"] for k in K_VALUES]
is_monotone = all(ppl_vals[i] > ppl_vals[i+1] for i in range(len(ppl_vals)-1))
best_ppl = min(ppl_vals)
k8_ppl = summary[8]["mean_ppl"]
k6_ppl = summary[6]["mean_ppl"]

print(f"\n  Monotone: {is_monotone}  best PPL: {best_ppl:.2f}  k=6: {k6_ppl:.2f}  k=8: {k8_ppl:.2f}", flush=True)

if is_monotone and (k6_ppl < 45.0 or k8_ppl < 45.0):
    verdict = "CONFIRMED"
    verdict_note = f"Monotone PPL curve AND k=6 ({k6_ppl:.1f}) or k=8 ({k8_ppl:.1f}) achieves PPL <45"
elif k8_ppl > 55.0:
    verdict = "REFUTED"
    verdict_note = f"k=8 PPL={k8_ppl:.1f} >55 — compression floor independent of ratio"
else:
    verdict = "INCONCLUSIVE"
    verdict_note = f"{'Non-monotone' if not is_monotone else 'Marginal improvement'}: k=6={k6_ppl:.1f}, k=8={k8_ppl:.1f}"

print(f"\n  VERDICT: {verdict}", flush=True)
print(f"  {verdict_note}", flush=True)

# Save
results = {
    "cycle": 185, "item": 194,
    "verdict": verdict, "verdict_note": verdict_note,
    "summary": {str(k): {**v, "per_seed_ppls": [all_seed_results[s][f"k{k}"]["stacked_polished"] for s in SEEDS]} for k, v in summary.items()},
    "per_seed": {str(s): all_seed_results[s] for s in SEEDS},
    "config": {
        "blocks": BLOCKS, "k_values": K_VALUES, "hidden_dims": HIDDEN_DIMS,
        "polish_steps": POLISH_STEPS, "lr": LR, "seeds": SEEDS,
        "eval_seqs": EVAL_SEQS, "seq_len": SEQ_LEN,
    },
}

out_path = RESULTS_DIR / "cycle_185_compression_ratio_sweep.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}", flush=True)
