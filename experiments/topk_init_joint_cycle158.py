"""Cycle 158: Topk+lstsq warm-start for joint rank-64 training (item 168).

Initialize joint training from topk+lstsq solution (projected to rank-64 via SVD)
instead of random init, to avoid bad optimization basins (cycle 155: 91.1±1.0).

Criteria:
  CONFIRMED    if topk-init joint PPL < 68 AND std < 2.0 (reliable AND better than polish)
  REFUTED      if PPL within 2% of topk+lstsq+polish (~70.7), i.e. >= 69.3
  INCONCLUSIVE if better than random-init (91.1) but worse than topk+corrections (~59)
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
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
)

SEEDS = [42, 7, 137]
D_MODEL = 768
D_FFN = 3072
D_BN = 64          # bottleneck rank
K_TOPK = 4         # topk units to select
BLOCKS = [7, 5, 4, 1]
SEQ_LEN = 128
BATCH_SIZE = 4
JOINT_STEPS = 2000
LR = 1e-4
NOISE_SCALE = 1e-3  # relative noise injected to active dims

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LinearFactorFFN(nn.Module):
    """Rank-D_BN factored FFN: x -> c_fc -> act -> encoder -> decoder -> output."""

    def __init__(self, original_mlp, d_bn):
        super().__init__()
        self.c_fc = original_mlp.c_fc
        self.act = original_mlp.act
        self.encoder = nn.Linear(D_FFN, d_bn, bias=False)
        self.decoder = nn.Linear(d_bn, D_MODEL, bias=True)
        if hasattr(original_mlp.c_proj, 'bias') and original_mlp.c_proj.bias is not None:
            self.decoder.bias.data.copy_(original_mlp.c_proj.bias.data)
        for p in self.c_fc.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.decoder(self.encoder(self.act(self.c_fc(x))))


def init_from_topk_lstsq(ffn_module: LinearFactorFFN, compressed_layer, seed: int) -> None:
    """Initialize encoder/decoder from topk+lstsq solution via SVD projection to rank-D_BN.

    Steps:
    1. Build W_approx = W_out_lstsq (shape D_MODEL x D_FFN via outer product structure)
       Actually: W_approx is the lstsq W_out (D_MODEL x k) applied to selected k columns of hidden.
       We embed this as a sparse D_FFN x D_MODEL matrix and do SVD to get rank-D_BN init.
    2. Build sparse matrix: rows = selected_indices, W_sparse[selected, :] = W_out_lstsq.T
    3. SVD of W_sparse (D_FFN x D_MODEL): W_sparse ≈ U[:, :r] @ S[:r] @ Vh[:r, :]
       encoder = (U[:, :r] @ diag(sqrt(S[:r]))).T  -> shape (r, D_FFN)
       decoder = diag(sqrt(S[:r])) @ Vh[:r, :]     -> shape (r, D_MODEL)
    4. Add noise scaled to 1e-3 * std of active encoder weights.
    """
    selected = compressed_layer.selected_indices  # shape (k,)
    w_out = compressed_layer.c_proj_weight        # shape (D_MODEL, k)
    bias_out = compressed_layer.c_proj_bias       # shape (D_MODEL,) or None

    k = selected.shape[0]
    device = ffn_module.encoder.weight.device
    dtype = ffn_module.encoder.weight.dtype

    selected = selected.to(device)
    w_out = w_out.to(device=device, dtype=torch.float32)

    # Build sparse (D_FFN x D_MODEL) matrix
    W_sparse = torch.zeros(D_FFN, D_MODEL, device=device, dtype=torch.float32)
    W_sparse[selected] = w_out.t()  # (k, D_MODEL)

    # SVD decomposition -> rank-D_BN approximation
    U, S, Vh = torch.linalg.svd(W_sparse, full_matrices=False)
    # U: (D_FFN, min(D_FFN,D_MODEL)), S: (min,), Vh: (min, D_MODEL)
    r = min(D_BN, S.shape[0])
    U_r = U[:, :r]   # (D_FFN, r)
    S_r = S[:r]       # (r,)
    Vh_r = Vh[:r, :]  # (r, D_MODEL)

    sqrt_S = S_r.sqrt()
    enc_weight = (U_r * sqrt_S.unsqueeze(0)).t().contiguous()  # (r, D_FFN)
    dec_weight = (sqrt_S.unsqueeze(1) * Vh_r).contiguous()     # (r, D_MODEL)

    # Pad to D_BN if r < D_BN (shouldn't happen for D_BN=64, D_FFN=3072, D_MODEL=768)
    if r < D_BN:
        pad_enc = torch.zeros(D_BN - r, D_FFN, device=device, dtype=torch.float32)
        pad_dec = torch.zeros(D_BN - r, D_MODEL, device=device, dtype=torch.float32)
        enc_weight = torch.cat([enc_weight, pad_enc], dim=0)
        dec_weight = torch.cat([dec_weight, pad_dec], dim=0)

    # Add noise: scale = noise_scale * std of active encoder rows
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    active_enc = enc_weight[:r]  # only compute std on SVD-derived rows
    noise_std = NOISE_SCALE * active_enc.std().item()
    noise_enc = torch.randn(enc_weight.shape, generator=g, device=device, dtype=torch.float32) * noise_std
    noise_dec = torch.randn(dec_weight.shape, generator=g, device=device, dtype=torch.float32) * noise_std

    with torch.no_grad():
        ffn_module.encoder.weight.copy_((enc_weight + noise_enc).to(dtype))
        ffn_module.decoder.weight.copy_((dec_weight + noise_dec).to(dtype).t())  # Linear stores (out, in)
        if bias_out is not None:
            ffn_module.decoder.bias.copy_(bias_out.to(device=device, dtype=dtype))


def compute_effective_rank(weight: torch.Tensor, threshold: float = 0.01) -> int:
    """Count singular values > threshold * max singular value."""
    sv = torch.linalg.svdvals(weight.float())
    return int((sv > threshold * sv[0]).sum().item())


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

    # --- Build topk+lstsq pack once (geometry is from the original teacher model) ---
    print("\nBuilding topk+lstsq compression pack...", flush=True)
    budget = CompressionBudgetSpec(
        per_layer_hidden={bid: K_TOPK for bid in BLOCKS},
        selection_mode="topk",
    )
    atlas = probe_ffn_geometry(teacher, corpus, layer_ids=tuple(BLOCKS), batch_size=BATCH_SIZE)
    pack = derive_ffn_compression(atlas, budget)
    compressed_map = {layer.layer_id: layer for layer in pack.layers}
    print(f"  Compressed layers: {[l.layer_id for l in pack.layers]}", flush=True)

    results_per_seed = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---", flush=True)
        _set_seed(seed)

        model = copy.deepcopy(teacher.model)
        model.eval()

        # Replace FFN blocks with LinearFactorFFN
        for bid in BLOCKS:
            model.transformer.h[bid].mlp = LinearFactorFFN(
                model.transformer.h[bid].mlp, D_BN
            )

        # Initialize from topk+lstsq solution
        for bid in BLOCKS:
            init_from_topk_lstsq(
                model.transformer.h[bid].mlp,
                compressed_map[bid],
                seed=seed,
            )

        # Verify pre-training PPL
        model.eval()
        pre_m = _evaluate_model(teacher, model, corpus, batch_size=BATCH_SIZE)
        pre_ppl = pre_m["perplexity"]
        print(f"  Pre-training PPL: {pre_ppl:.3f} (expected ~95)", flush=True)

        # Effective rank before training
        rank_before = {}
        for bid in BLOCKS:
            enc_w = model.transformer.h[bid].mlp.encoder.weight  # (D_BN, D_FFN)
            rank_before[bid] = compute_effective_rank(enc_w)
        print(f"  Effective rank before training: {rank_before}", flush=True)

        # Joint training
        model.requires_grad_(False)
        params = []
        for bid in BLOCKS:
            for p in model.transformer.h[bid].mlp.encoder.parameters():
                p.requires_grad_(True); params.append(p)
            for p in model.transformer.h[bid].mlp.decoder.parameters():
                p.requires_grad_(True); params.append(p)

        opt = torch.optim.AdamW(params, lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=JOINT_STEPS, eta_min=0)
        rng = torch.Generator(); rng.manual_seed(seed)

        loss_log = []
        for step in range(JOINT_STEPS):
            idx = torch.randperm(corpus.probe_holdout.shape[0], generator=rng)[:BATCH_SIZE]
            batch = corpus.probe_holdout[idx].to(device)
            opt.zero_grad()
            raw = model(batch[:, :-1])
            logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
            loss.backward(); opt.step(); sched.step()
            if step % 500 == 0:
                print(f"    Step {step:4d}: loss={loss.item():.4f}", flush=True)
                loss_log.append({"step": step, "loss": float(loss.item())})

        model.eval(); model.requires_grad_(False)
        post_m = _evaluate_model(teacher, model, corpus, batch_size=BATCH_SIZE)
        post_ppl = post_m["perplexity"]
        print(f"  Post-training PPL: {post_ppl:.3f}", flush=True)

        # Effective rank after training
        rank_after = {}
        for bid in BLOCKS:
            enc_w = model.transformer.h[bid].mlp.encoder.weight
            rank_after[bid] = compute_effective_rank(enc_w)
        print(f"  Effective rank after training: {rank_after}", flush=True)

        results_per_seed[seed] = {
            "pre_ppl": pre_ppl,
            "post_ppl": post_ppl,
            "rank_before": rank_before,
            "rank_after": rank_after,
            "loss_log": loss_log,
        }
        del model

    # Summary
    post_ppls = [results_per_seed[s]["post_ppl"] for s in SEEDS]
    pre_ppls = [results_per_seed[s]["pre_ppl"] for s in SEEDS]
    median_ppl = float(np.median(post_ppls))
    mean_ppl = float(np.mean(post_ppls))
    std_ppl = float(np.std(post_ppls))
    min_ppl = float(np.min(post_ppls))
    max_ppl = float(np.max(post_ppls))

    # Check effective rank expansion
    all_rank_after = []
    for s in SEEDS:
        all_rank_after.extend(results_per_seed[s]["rank_after"].values())
    median_rank_after = float(np.median(all_rank_after))
    rank_stuck = median_rank_after <= 5

    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY ({len(SEEDS)} seeds)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Pre-training PPL:  {np.mean(pre_ppls):.3f} ± {np.std(pre_ppls):.3f}", flush=True)
    print(f"  Post-training PPL: {mean_ppl:.3f} ± {std_ppl:.3f}", flush=True)
    print(f"  Median: {median_ppl:.3f}, Min: {min_ppl:.3f}, Max: {max_ppl:.3f}", flush=True)
    print(f"  Median effective rank after: {median_rank_after:.1f}", flush=True)
    print(f"  Rank stuck (<=5): {rank_stuck}", flush=True)
    print(f"\nBaselines:", flush=True)
    print(f"  Random-init joint (cycle 155): 91.1 ± 1.0", flush=True)
    print(f"  Topk+lstsq+polish (cycle ~):   70.7", flush=True)
    print(f"  Topk+corrections:              ~59", flush=True)

    # Verdict
    TOPK_POLISH_PPL = 70.7
    RANDOM_INIT_PPL = 91.1
    if median_ppl < 68.0 and std_ppl < 2.0:
        verdict = "CONFIRMED"
        detail = f"Median {median_ppl:.1f} < 68 AND std {std_ppl:.1f} < 2.0 — warm start reliably better"
    elif median_ppl >= TOPK_POLISH_PPL * 0.98:  # within 2% of 70.7
        verdict = "REFUTED"
        detail = f"PPL {median_ppl:.1f} >= {TOPK_POLISH_PPL*0.98:.1f} (within 2% of topk+polish): warm start doesn't help"
    elif rank_stuck:
        verdict = "REFUTED"
        detail = f"Rank stuck at {median_rank_after:.1f} <= 5: just topk+lstsq with extra Adam steps, not true joint"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"PPL {median_ppl:.1f} better than random-init ({RANDOM_INIT_PPL}) but not below 68; std={std_ppl:.2f}"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  Detail:  {detail}", flush=True)

    output = {
        "cycle": 158,
        "item": 168,
        "seeds": SEEDS,
        "float_ppl": bl["perplexity"],
        "topk_polish_baseline": TOPK_POLISH_PPL,
        "random_init_baseline": RANDOM_INIT_PPL,
        "topk_corrections_baseline": 59.0,
        "d_bn": D_BN,
        "k_topk": K_TOPK,
        "noise_scale": NOISE_SCALE,
        "joint_steps": JOINT_STEPS,
        "lr": LR,
        "per_seed": {str(s): results_per_seed[s] for s in SEEDS},
        "post_ppl_median": median_ppl,
        "post_ppl_mean": mean_ppl,
        "post_ppl_std": std_ppl,
        "post_ppl_min": min_ppl,
        "post_ppl_max": max_ppl,
        "pre_ppl_mean": float(np.mean(pre_ppls)),
        "median_effective_rank_after": median_rank_after,
        "rank_stuck": rank_stuck,
        "verdict": verdict,
        "verdict_detail": detail,
    }
    out_path = RESULTS_DIR / "cycle_158_topk_init_joint.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)
    print("Done.", flush=True)
