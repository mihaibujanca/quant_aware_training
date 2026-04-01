"""Cycle 186 / Item 195: Fewer-block compression — GPT-2 FFN k=4 on 2 blocks vs 4 blocks.

Run topk+lstsq+polish on all 6 pairs from {1,4,7,11}. Compare best 2-block PPL vs
4-block baseline (~59 PPL from cycle 185). Atlas probed once on all 4 blocks, reused
for all pair runs.

Verdict:
  CONFIRMED    if best 2-block pair achieves mean PPL <42
  REFUTED      if all 2-block pairs give mean PPL >48
  INCONCLUSIVE if best pair mean PPL in [42, 48] range
"""
from __future__ import annotations

import copy
import itertools
import json
import random
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
K = 4
HIDDEN_DIM = 512
ALL_BLOCKS = [1, 4, 7, 11]
BLOCK_PAIRS = list(itertools.combinations([1, 4, 7, 11], 2))  # 6 pairs
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
print(f"\nBlock pairs to test: {BLOCK_PAIRS}", flush=True)

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

    # Probe atlas on ALL 4 blocks at once — reuse for all pair runs
    _set_seed(seed)
    print(f"\n  Probing atlas on blocks {ALL_BLOCKS}...", flush=True)
    atlas = probe_ffn_geometry(
        teacher, corpus,
        layer_ids=tuple(ALL_BLOCKS),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )
    print(f"  Atlas ready ({len(atlas.layer_geometries)} layers)", flush=True)

    # 4-block baseline
    print(f"\n  --- 4-block baseline {ALL_BLOCKS} ---", flush=True)
    _set_seed(seed + K)
    pack_4block = _derive_topk_pack(atlas, ALL_BLOCKS, HIDDEN_DIM)
    student_4block = swap_compressed_ffns(teacher, pack_4block, inplace=False)
    student_4block.eval()
    m_raw = _evaluate_model(teacher, student_4block, corpus, batch_size=BATCH_SIZE)
    print(f"    unpolished PPL={float(m_raw['perplexity']):.3f}", flush=True)
    _set_seed(seed + K)
    losses = _joint_polish(student_4block, ALL_BLOCKS, corpus.probe_holdout, POLISH_STEPS, LR, device, seed + K)
    if losses:
        print(f"    polish loss: {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
    student_4block.eval()
    m4 = _evaluate_model(teacher, student_4block, corpus, batch_size=BATCH_SIZE)
    fourblock_ppl = float(m4["perplexity"])
    print(f"    4-block polished PPL={fourblock_ppl:.3f}", flush=True)
    del student_4block

    # All 6 pairs
    pair_ppls: dict[str, float] = {}
    for pair in BLOCK_PAIRS:
        pair_key = str(pair)
        pair_list = list(pair)
        pair_seed = seed + pair[0] * 100 + pair[1]

        print(f"\n  --- pair {pair} ---", flush=True)
        _set_seed(pair_seed)
        pack_pair = _derive_topk_pack(atlas, pair_list, HIDDEN_DIM)
        student_pair = swap_compressed_ffns(teacher, pack_pair, inplace=False)
        student_pair.eval()
        m_raw = _evaluate_model(teacher, student_pair, corpus, batch_size=BATCH_SIZE)
        print(f"    unpolished PPL={float(m_raw['perplexity']):.3f}", flush=True)
        _set_seed(pair_seed)
        losses = _joint_polish(student_pair, pair_list, corpus.probe_holdout, POLISH_STEPS, LR, device, pair_seed)
        if losses:
            print(f"    polish loss: {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
        student_pair.eval()
        mp = _evaluate_model(teacher, student_pair, corpus, batch_size=BATCH_SIZE)
        pair_ppl = float(mp["perplexity"])
        print(f"    polished PPL={pair_ppl:.3f}", flush=True)
        # Sanity flags
        if pair_ppl < baseline_ppl:
            print(f"    WARNING: pair PPL {pair_ppl:.2f} < teacher baseline {baseline_ppl:.2f}", flush=True)
        if pair_ppl > 200:
            print(f"    WARNING: pair PPL {pair_ppl:.2f} >200 — possible polish divergence", flush=True)
        pair_ppls[pair_key] = pair_ppl
        del student_pair

    all_seed_results[seed] = {
        "baseline_ppl": baseline_ppl,
        "4block_ppl": fourblock_ppl,
        "pairs": pair_ppls,
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATE RESULTS\n{'='*60}", flush=True)

summary_4block_ppls = [all_seed_results[s]["4block_ppl"] for s in SEEDS]
mean_4block = sum(summary_4block_ppls) / len(summary_4block_ppls)
std_4block = (sum((p - mean_4block)**2 for p in summary_4block_ppls) / len(summary_4block_ppls)) ** 0.5

pair_summary: dict[str, dict] = {}
for pair in BLOCK_PAIRS:
    pair_key = str(pair)
    ppls = [all_seed_results[s]["pairs"][pair_key] for s in SEEDS]
    mean_ppl = sum(ppls) / len(ppls)
    std_ppl = (sum((p - mean_ppl)**2 for p in ppls) / len(ppls)) ** 0.5
    pair_summary[pair_key] = {"mean_ppl": mean_ppl, "std_ppl": std_ppl, "per_seed_ppls": ppls}

best_pair_key = min(pair_summary, key=lambda k: pair_summary[k]["mean_ppl"])
best_ppl = pair_summary[best_pair_key]["mean_ppl"]

print(f"\n  {'pair':<12} {'mean_PPL':<12} {'std_PPL'}", flush=True)
print(f"  {'-'*40}", flush=True)
for pair in BLOCK_PAIRS:
    pk = str(pair)
    mark = " <-- best" if pk == best_pair_key else ""
    print(f"  {pk:<12} {pair_summary[pk]['mean_ppl']:<12.2f} ±{pair_summary[pk]['std_ppl']:.2f}{mark}", flush=True)
print(f"  {'4-block':<12} {mean_4block:<12.2f} ±{std_4block:.2f}", flush=True)

# Verdict
if best_ppl < 42.0:
    verdict = "CONFIRMED"
    verdict_note = f"Best 2-block pair {best_pair_key} achieves mean PPL {best_ppl:.2f} <42"
elif best_ppl > 48.0:
    verdict = "REFUTED"
    verdict_note = f"All 2-block pairs give mean PPL >{48}: best={best_ppl:.2f} for pair {best_pair_key}"
else:
    verdict = "INCONCLUSIVE"
    verdict_note = f"Best 2-block pair {best_pair_key} gives mean PPL {best_ppl:.2f} in [42,48] range"

print(f"\n  VERDICT: {verdict}", flush=True)
print(f"  {verdict_note}", flush=True)

# Save
results = {
    "cycle": 186, "item": 195,
    "verdict": verdict, "verdict_note": verdict_note,
    "best_pair": best_pair_key, "best_ppl": best_ppl,
    "per_seed": {str(s): all_seed_results[s] for s in SEEDS},
    "summary": {
        "4block": {"mean_ppl": mean_4block, "std_ppl": std_4block, "per_seed_ppls": summary_4block_ppls},
        "pairs": pair_summary,
    },
    "config": {
        "all_blocks": ALL_BLOCKS,
        "block_pairs": [str(p) for p in BLOCK_PAIRS],
        "k": K, "hidden_dim": HIDDEN_DIM,
        "polish_steps": POLISH_STEPS, "lr": LR, "seeds": SEEDS,
        "eval_seqs": EVAL_SEQS, "seq_len": SEQ_LEN,
    },
}

out_path = RESULTS_DIR / "cycle_186_fewer_blocks.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}", flush=True)
