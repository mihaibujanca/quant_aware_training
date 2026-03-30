"""Cycle 131: Learned bottleneck AE for FFN compression — POC (item 88).

Single-block swap POC at k=384 (8x) on blocks 4 and 7.
E(h) = ReLU(W_enc @ h + b_enc), D(z) = W_dec @ z + b_dec.
Trained under model CE loss, 1000 steps. 3 seeds. Compare vs topk+lstsq.

Verdict:
  CONFIRMED    if AE beats topk by >=5% relative PPL on >=1 block (outside 1-std)
  REFUTED      if AE within 2% AND encoder nonlinearity < 5%
  INCONCLUSIVE otherwise
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
    TeacherBundle,
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

# Config
SEEDS = [42, 137, 7]
BLOCKS = [4, 7]
K = 384  # bottleneck dim (8x compression from 3072)
D_MODEL = 768
D_FFN = 3072
SEQ_LEN = 128
BATCH_SIZE = 4
AE_TRAIN_STEPS = 1000
AE_LR = 1e-4
POLISH_STEPS = 500
POLISH_LR = 1e-4
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200
EVAL_SEQS = 200
HIDDEN_FRACTION_K384 = K / D_FFN  # 384/3072 = 0.125

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _param_count(w, b=None):
    return w.numel() + (0 if b is None else b.numel())


class BottleneckFFN(nn.Module):
    """Replace an FFN block's hidden layer with a learned bottleneck.
    Forward: x -> frozen_c_fc(x) -> GELU -> E -> D + bias
    """
    def __init__(self, original_mlp, k: int):
        super().__init__()
        self.c_fc = original_mlp.c_fc  # frozen
        self.act = original_mlp.act    # frozen GELU
        self.encoder = nn.Linear(D_FFN, k)
        self.decoder = nn.Linear(k, D_MODEL)
        # Copy original c_proj bias if present
        if hasattr(original_mlp.c_proj, 'bias') and original_mlp.c_proj.bias is not None:
            self.decoder.bias.data.copy_(original_mlp.c_proj.bias.data)
        # Freeze c_fc
        for p in self.c_fc.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        h = self.act(self.c_fc(x))  # (batch, seq, 3072)
        z = F.relu(self.encoder(h))  # (batch, seq, k)
        return self.decoder(z)       # (batch, seq, 768)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.encoder.parameters()) + \
               sum(p.numel() for p in self.decoder.parameters())


def train_ae(model, block_id, tokens, n_steps, lr, device, seed):
    """Train AE encoder+decoder under model CE loss."""
    model.requires_grad_(False)
    mlp = model.transformer.h[block_id].mlp
    for p in mlp.encoder.parameters():
        p.requires_grad_(True)
    for p in mlp.decoder.parameters():
        p.requires_grad_(True)

    params = list(mlp.encoder.parameters()) + list(mlp.decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    model.eval()  # keep BN/dropout off but train AE
    losses = []
    rng = torch.Generator()
    rng.manual_seed(seed)
    n_seqs = tokens.shape[0]

    for step in range(n_steps):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = tokens[idx].to(device)
        inp = batch[:, :-1].contiguous()
        tgt = batch[:, 1:].contiguous()
        opt.zero_grad()
        raw = model(inp)
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % 200 == 0:
            print(f"      step {step+1}/{n_steps}: loss={loss.item():.4f}", flush=True)

    model.requires_grad_(False)
    return losses


@torch.no_grad()
def measure_nonlinearity(model, block_id, tokens, device, n_samples=1000):
    """Measure encoder nonlinearity: ||E(h) - E_linear(h)|| / ||E(h)||."""
    mlp = model.transformer.h[block_id].mlp
    # Collect hidden activations
    hiddens = []
    hook_fn = lambda mod, inp, out: hiddens.append(out.detach())
    h = mlp.c_fc.register_forward_hook(hook_fn)

    batch = tokens[:min(16, tokens.shape[0])].to(device)
    model(batch[:, :-1].contiguous())
    h.remove()

    h_all = torch.cat(hiddens, dim=0).reshape(-1, D_FFN)[:n_samples]
    h_gelu = F.gelu(h_all)

    # E(h) = ReLU(W_enc @ h_gelu + b_enc)
    E_h = F.relu(mlp.encoder(h_gelu))
    # E_linear(h) = W_enc @ h_gelu + b_enc (no ReLU)
    E_linear_h = mlp.encoder(h_gelu)

    diff_norm = (E_h - E_linear_h).norm(dim=1)
    e_norm = E_h.norm(dim=1) + 1e-12
    nonlinearity = float((diff_norm / e_norm).mean())
    return nonlinearity


# topk+lstsq helpers (from cycle 125 pattern)
def _derive_block_pack(atlas, budget, block_ids):
    compressed_layers = []
    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
            continue
        k = _layer_target_hidden(layer, budget)
        selected = _select_hidden_units(layer, budget, k)
        source_w_in = _standard_weight(layer.handle.c_fc).float()
        source_b_in = _standard_bias(layer.handle.c_fc)
        use_ob = _standard_bias(layer.handle.c_proj) is not None
        student_w_in = source_w_in[selected].contiguous()
        student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()
        hidden = layer.hidden_sample[:, selected].float()
        outputs = layer.output_sample.float()
        student_w_out, student_b_out = _fit_output_projection(hidden, outputs, use_bias=use_ob)
        sp = _param_count(source_w_in, source_b_in) + _param_count(
            _standard_weight(layer.handle.c_proj), _standard_bias(layer.handle.c_proj))
        cp = _param_count(student_w_in, student_b_in) + _param_count(student_w_out, student_b_out)
        compressed_layers.append(CompressedFfnLayer(
            layer_id=layer.layer_id, activation_kind=layer.activation_kind,
            source_hidden_dim=layer.hidden_dim, compressed_hidden_dim=int(k),
            c_fc_weight=student_w_in, c_fc_bias=student_b_in,
            c_proj_weight=student_w_out, c_proj_bias=student_b_out,
            selected_indices=selected, source_param_count=sp, compressed_param_count=cp,
            notes={"selection_mode": budget.selection_mode},
        ))
    return CompressedFfnPack(
        method="geometric", model_family=atlas.model_family,
        dataset_name=atlas.dataset_name, layers=tuple(compressed_layers),
        metadata={"sequence_length": atlas.sequence_length},
    )


def _polish_block(model, block_id, tokens, n_steps, lr, device, seed):
    mlp = model.transformer.h[block_id].mlp
    if not hasattr(mlp, "c_proj"):
        return []
    model.requires_grad_(False)
    mlp.c_proj.weight.requires_grad_(True)
    if mlp.c_proj.bias is not None:
        mlp.c_proj.bias.requires_grad_(True)
    params = [mlp.c_proj.weight] + ([mlp.c_proj.bias] if mlp.c_proj.bias is not None else [])
    opt = torch.optim.Adam(params, lr=lr)
    model.train()
    rng = torch.Generator()
    rng.manual_seed(seed)
    losses = []
    for step in range(n_steps):
        idx = torch.randperm(tokens.shape[0], generator=rng)[:BATCH_SIZE]
        batch = tokens[idx].to(device)
        opt.zero_grad()
        raw = model(batch[:, :-1].contiguous())
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].contiguous().reshape(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    model.eval()
    model.requires_grad_(False)
    return losses


if __name__ == "__main__":
    print("Loading GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    float_model = teacher.model
    float_model.eval()
    device = teacher.device

    _set_seed(42)
    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher, text_path=str(OWT_PATH), sequence_length=SEQ_LEN,
        probe_train_sequences=PROBE_TRAIN_SEQS, probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
        eval_val_sequences=EVAL_SEQS, dataset_name="openwebtext", shuffle_seed=42,
    ))

    bl = _evaluate_model(teacher, float_model, corpus, batch_size=BATCH_SIZE)
    print(f"  Float PPL: {bl['perplexity']:.3f}", flush=True)

    budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION_K384)
    atlas = probe_ffn_geometry(teacher, corpus, layer_ids=tuple(sorted(BLOCKS)),
                                max_tokens_per_layer=MAX_TOKENS_PER_LAYER, batch_size=BATCH_SIZE)

    results = {}
    for seed in SEEDS:
        print(f"\n{'='*50}", flush=True)
        print(f"SEED {seed}", flush=True)
        print(f"{'='*50}", flush=True)

        for bid in BLOCKS:
            print(f"\n  Block {bid}:", flush=True)

            # topk+lstsq baseline
            _set_seed(seed)
            pack = _derive_block_pack(atlas, budget, [bid])
            topk_model = swap_compressed_ffns(teacher, pack, inplace=False)
            topk_model.eval()
            _polish_block(topk_model, bid, corpus.probe_holdout, POLISH_STEPS, POLISH_LR, device, seed)
            topk_m = _evaluate_model(teacher, topk_model, corpus, batch_size=BATCH_SIZE)
            topk_ppl = topk_m["perplexity"]
            print(f"    topk+lstsq k={K}: PPL={topk_ppl:.3f}", flush=True)
            del topk_model

            # Bottleneck AE
            _set_seed(seed)
            ae_model = copy.deepcopy(float_model)
            ae_model.eval()
            orig_mlp = ae_model.transformer.h[bid].mlp
            bottleneck = BottleneckFFN(orig_mlp, K)
            ae_model.transformer.h[bid].mlp = bottleneck

            print(f"    AE training {AE_TRAIN_STEPS} steps...", flush=True)
            ae_losses = train_ae(ae_model, bid, corpus.probe_holdout, AE_TRAIN_STEPS, AE_LR, device, seed)
            ae_m = _evaluate_model(teacher, ae_model, corpus, batch_size=BATCH_SIZE)
            ae_ppl = ae_m["perplexity"]
            nonlin = measure_nonlinearity(ae_model, bid, corpus.eval_val, device)
            print(f"    AE k={K}: PPL={ae_ppl:.3f}, nonlinearity={nonlin:.4f}", flush=True)

            key = f"{bid}_{seed}"
            results[key] = {
                "block": bid, "seed": seed,
                "topk_ppl": topk_ppl, "ae_ppl": ae_ppl,
                "nonlinearity": nonlin,
                "ae_params": bottleneck.n_params,
                "ae_final_loss": ae_losses[-1] if ae_losses else None,
            }
            del ae_model

    # Summary
    print(f"\n{'='*50}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*50}", flush=True)

    for bid in BLOCKS:
        topk_vals = [results[f"{bid}_{s}"]["topk_ppl"] for s in SEEDS]
        ae_vals = [results[f"{bid}_{s}"]["ae_ppl"] for s in SEEDS]
        nonlin_vals = [results[f"{bid}_{s}"]["nonlinearity"] for s in SEEDS]
        topk_mean, topk_std = np.mean(topk_vals), np.std(topk_vals)
        ae_mean, ae_std = np.mean(ae_vals), np.std(ae_vals)
        nonlin_mean = np.mean(nonlin_vals)
        rel_diff = (ae_mean - topk_mean) / topk_mean * 100
        print(f"  Block {bid}:", flush=True)
        print(f"    topk+lstsq: {topk_mean:.2f} ± {topk_std:.2f}", flush=True)
        print(f"    AE:         {ae_mean:.2f} ± {ae_std:.2f}", flush=True)
        print(f"    Relative:   {rel_diff:+.1f}%", flush=True)
        print(f"    Nonlinearity: {nonlin_mean:.4f}", flush=True)

    # Verdict
    any_ae_wins_5pct = False
    all_within_2pct = True
    all_nonlin_below_5pct = True

    for bid in BLOCKS:
        topk_vals = [results[f"{bid}_{s}"]["topk_ppl"] for s in SEEDS]
        ae_vals = [results[f"{bid}_{s}"]["ae_ppl"] for s in SEEDS]
        nonlin_vals = [results[f"{bid}_{s}"]["nonlinearity"] for s in SEEDS]
        topk_mean, topk_std = np.mean(topk_vals), np.std(topk_vals)
        ae_mean, ae_std = np.mean(ae_vals), np.std(ae_vals)

        # AE wins if ae_mean < topk_mean * 0.95 AND ranges don't overlap within 1 std
        if ae_mean < topk_mean * 0.95 and (ae_mean + ae_std) < (topk_mean - topk_std):
            any_ae_wins_5pct = True
        if abs(ae_mean - topk_mean) / topk_mean > 0.02:
            all_within_2pct = False
        if np.mean(nonlin_vals) >= 0.05:
            all_nonlin_below_5pct = False

    if any_ae_wins_5pct:
        verdict = "CONFIRMED"
        detail = "AE beats topk+lstsq by >=5% relative PPL on >=1 block"
    elif all_within_2pct and all_nonlin_below_5pct:
        verdict = "REFUTED"
        detail = "AE within 2% of topk AND encoder near-linear (<5% nonlinearity)"
    else:
        verdict = "INCONCLUSIVE"
        detail = "Mixed results"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  Detail: {detail}", flush=True)

    # Save
    output = {
        "cycle": 131, "item": 88, "k": K, "blocks": BLOCKS, "seeds": SEEDS,
        "float_ppl": bl["perplexity"],
        "per_config": results,
        "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_131_bottleneck_ae.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON", flush=True)
    print("Done.", flush=True)
