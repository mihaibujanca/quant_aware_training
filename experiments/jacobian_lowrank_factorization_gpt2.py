"""
Cycle 100: Low-rank Jacobian factorization for FFN compression — GPT-2 FFN.

Test: global U @ diag(S(x)) @ V^T factorization (shared U,V, token-varying S)
with S(x) predicted via ridge regression.

Phases:
  1. Per-token Jacobian rank audit (2048 tokens, blocks [1,4,7,11])
  2. Global factorization via mean-Jacobian SVD; test r in {32,64,128,256}
  3. S(x) predictor via ridge regression (train 1536, eval 512 tokens)
  4. Single-block swap PPL (3 seeds), compare vs baseline/oracle/random null/topk+lstsq
"""
import copy
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    ProbeCorpusSpec,
    TeacherLoadSpec,
    build_probe_corpus,
    load_nanochat_teacher,
    _evaluate_model,
)

OUTPUT_DIR = ROOT / "autoresearch" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = ROOT / "data" / "shakespeare.txt"

DEVICE = "cpu"
AUDIT_BLOCKS = [1, 4, 7, 11]
N_TOKENS_AUDIT = 2048      # Phase 1 audit tokens
SEQ_LEN = 128
RANKS = [32, 64, 128, 256]
N_TRAIN = 1536             # Phase 3 train tokens
N_EVAL = 512               # Phase 3 eval tokens
SEEDS = [42, 123, 7]
PROBE_HOLDOUT_SEQ = 32     # sequences for PPL eval per seed

# ---------------------------------------------------------------------------
# GELU derivative (GPT-2 tanh approximation)
# ---------------------------------------------------------------------------
def gelu_grad(z: torch.Tensor) -> torch.Tensor:
    k = math.sqrt(2.0 / math.pi)
    t = torch.tanh(k * (z + 0.044715 * z ** 3))
    dt = 1.0 - t ** 2
    inner_deriv = k * (1.0 + 3.0 * 0.044715 * z ** 2)
    return 0.5 * (1.0 + t) + 0.5 * z * dt * inner_deriv


def effective_rank(s: torch.Tensor, threshold: float) -> int:
    energy = (s ** 2).cumsum(0) / (s ** 2).sum()
    idx = (energy >= threshold).nonzero(as_tuple=False)
    return len(s) if len(idx) == 0 else int(idx[0, 0].item()) + 1


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading GPT-2...")
t0 = time.time()
teacher = load_nanochat_teacher(TeacherLoadSpec(
    source="hf_pretrained",
    hf_model_name="gpt2",
    device=DEVICE,
))
model = teacher.model
tokenizer = teacher.tokenizer
model.eval()
print(f"  loaded in {time.time()-t0:.1f}s")

N_BLOCKS = 12
W_ins, W_outs, b_ins, b_outs = [], [], [], []
for b in range(N_BLOCKS):
    c_fc = model.transformer.h[b].mlp.c_fc
    c_proj = model.transformer.h[b].mlp.c_proj
    W_in = c_fc.weight.T.detach().float()    # (3072, 768)
    W_out = c_proj.weight.T.detach().float() # (768, 3072)
    b_in = c_fc.bias.detach().float() if c_fc.bias is not None else None
    b_out = c_proj.bias.detach().float() if c_proj.bias is not None else None
    W_ins.append(W_in)
    W_outs.append(W_out)
    b_ins.append(b_in)
    b_outs.append(b_out)
print(f"  W_in={W_ins[0].shape}, W_out={W_outs[0].shape}")


# ---------------------------------------------------------------------------
# Helper: collect activations (z = pre-GELU, x = FFN input) for N tokens
# ---------------------------------------------------------------------------
def collect_activations(n_tokens: int, shuffle_seed: int):
    """Returns ffn_z[b] shape (T,3072), ffn_x[b] shape (T,768) for AUDIT_BLOCKS."""
    n_seq = max(1, n_tokens // SEQ_LEN)
    corpus = build_probe_corpus(ProbeCorpusSpec(
        text_path=str(DATA_PATH),
        sequence_length=SEQ_LEN,
        probe_train_sequences=n_seq,
        probe_holdout_sequences=PROBE_HOLDOUT_SEQ,
        tokenizer=tokenizer,
        shuffle_seed=shuffle_seed,
        dataset_name="shakespeare",
    ))
    tokens = corpus.probe_train  # (n_seq, SEQ_LEN)

    ffn_z: dict[int, list] = {b: [] for b in AUDIT_BLOCKS}
    ffn_x: dict[int, list] = {b: [] for b in AUDIT_BLOCKS}
    hooks = []

    def make_z_hook(blk):
        def hook(module, inp, out):
            ffn_z[blk].append(out.detach().cpu().float())
        return hook

    def make_x_hook(blk):
        def hook(module, inp, out):
            # inp[0] is x before FFN
            ffn_x[blk].append(inp[0].detach().cpu().float())
        return hook

    for b in AUDIT_BLOCKS:
        hooks.append(model.transformer.h[b].mlp.c_fc.register_forward_hook(make_z_hook(b)))
        hooks.append(model.transformer.h[b].mlp.register_forward_hook(make_x_hook(b)))

    with torch.no_grad():
        for seq_idx in range(tokens.shape[0]):
            model(tokens[seq_idx:seq_idx+1].to(DEVICE), use_cache=False)

    for h in hooks:
        h.remove()

    for b in AUDIT_BLOCKS:
        z = torch.cat(ffn_z[b], dim=0)
        x = torch.cat(ffn_x[b], dim=0)
        if z.dim() == 3:
            z = z.reshape(-1, z.shape[-1])
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        ffn_z[b] = z
        ffn_x[b] = x

    return ffn_z, ffn_x, corpus


# ===========================================================================
# Phase 1 — Per-token Jacobian rank audit (2048 tokens)
# ===========================================================================
print("\n=== Phase 1: Per-token rank audit (2048 tokens) ===")
ffn_z_audit, ffn_x_audit, corpus_seed42 = collect_activations(N_TOKENS_AUDIT, shuffle_seed=42)

phase1_results = {}
for b in AUDIT_BLOCKS:
    z_all = ffn_z_audit[b]  # (T, 3072)
    T = z_all.shape[0]
    W_in, W_out = W_ins[b], W_outs[b]
    ranks_95, ranks_99 = [], []
    for t in range(T):
        sigma_p = gelu_grad(z_all[t])
        J = (W_out * sigma_p.unsqueeze(0)) @ W_in  # (768, 768)
        s = torch.linalg.svdvals(J)
        ranks_95.append(effective_rank(s, 0.95))
        ranks_99.append(effective_rank(s, 0.99))
    median_95 = float(torch.tensor(ranks_95).float().median().item())
    median_99 = float(torch.tensor(ranks_99).float().median().item())
    phase1_results[b] = {"median_rank_95": median_95, "median_rank_99": median_99, "n_tokens": T}
    print(f"  Block {b:2d}: rank_95 median={median_95:.0f}, rank_99 median={median_99:.0f}  (T={T})")


# ===========================================================================
# Phase 2 — Global factorization via mean-Jacobian SVD
# ===========================================================================
print("\n=== Phase 2: Global factorization (mean-Jacobian SVD) ===")
phase2_results = {}

for b in AUDIT_BLOCKS:
    print(f"\n  Block {b}:")
    z_all = ffn_z_audit[b]  # (T, 3072)
    T = z_all.shape[0]
    W_in, W_out = W_ins[b], W_outs[b]

    # Compute mean Jacobian
    J_sum = torch.zeros(768, 768)
    for t in range(T):
        sigma_p = gelu_grad(z_all[t])
        J_sum += (W_out * sigma_p.unsqueeze(0)) @ W_in
    J_bar = J_sum / T  # (768, 768)

    # Full SVD of J_bar
    U_full, Sigma_full, Vh_full = torch.linalg.svd(J_bar, full_matrices=True)
    V_full = Vh_full.T  # (768, 768)

    # Sanity check: full-rank reconstruction
    J_recon_full = U_full @ torch.diag(Sigma_full) @ V_full.T
    sanity_err = (J_bar - J_recon_full).norm() / J_bar.norm()
    print(f"    Full-rank reconstruction error: {sanity_err:.2e}  (should be ~0)")
    assert sanity_err < 1e-4, f"Full-rank sanity FAILED: {sanity_err}"

    block_results = {"sanity_full_rank_err": float(sanity_err)}

    for r in RANKS:
        U_r = U_full[:, :r]   # (768, r)
        V_r = V_full[:, :r]   # (768, r)
        # Oracle diagonal: S(x) = diag(U^T @ J(x) @ V)
        rel_errors = []
        for t in range(T):
            sigma_p = gelu_grad(z_all[t])
            J_t = (W_out * sigma_p.unsqueeze(0)) @ W_in  # (768, 768)
            S_t = (U_r.T @ J_t @ V_r).diag()             # (r,)
            J_recon = U_r @ torch.diag(S_t) @ V_r.T      # (768, 768)
            rel_err = (J_t - J_recon).norm() / J_t.norm()
            rel_errors.append(float(rel_err.item()))
        median_err = float(torch.tensor(rel_errors).median().item())
        mean_err = float(torch.tensor(rel_errors).mean().item())
        block_results[f"r{r}_median_rel_err"] = median_err
        block_results[f"r{r}_mean_rel_err"] = mean_err
        print(f"    r={r:3d}: median_rel_err={median_err:.4f}, mean_rel_err={mean_err:.4f}")

    phase2_results[b] = block_results

# Refutation gate
r128_errors = [phase2_results[b]["r128_median_rel_err"] for b in AUDIT_BLOCKS]
median_r128 = float(torch.tensor(r128_errors).median().item())
print(f"\n  Refutation gate: median relative error at r=128 = {median_r128:.4f} (threshold: 0.5)")
refuted = median_r128 > 0.5
if refuted:
    print("  REFUTED: factorization fails at r=128")
else:
    print("  Gate PASSED: proceeding to Phase 3")


# ===========================================================================
# Phase 3 — S(x) predictor via ridge regression
# ===========================================================================
print("\n=== Phase 3: S(x) predictor (ridge regression) ===")
# Use first AUDIT_BLOCK for detailed analysis, store results for all
# We need 2048 tokens: first 1536 train, last 512 eval
# Already have ffn_x_audit (2048 tokens) and ffn_z_audit

phase3_results = {}

for b in AUDIT_BLOCKS:
    z_all = ffn_z_audit[b]   # (T, 3072)
    x_all = ffn_x_audit[b]   # (T, 768)
    T = z_all.shape[0]

    W_in, W_out = W_ins[b], W_outs[b]
    U_full, Sigma_full, Vh_full = torch.linalg.svd(
        sum((W_out * gelu_grad(z_all[t]).unsqueeze(0)) @ W_in for t in range(T)) / T,
        full_matrices=True
    )
    V_full = Vh_full.T

    # For r=64 (primary rank)
    for r in [64]:
        U_r = U_full[:, :r]
        V_r = V_full[:, :r]

        # Compute oracle S(x) for all tokens
        S_all = torch.zeros(T, r)
        for t in range(T):
            sigma_p = gelu_grad(z_all[t])
            J_t = (W_out * sigma_p.unsqueeze(0)) @ W_in
            S_all[t] = (U_r.T @ J_t @ V_r).diag()

        # Split train/eval
        n_train = min(N_TRAIN, T - N_EVAL)
        n_eval = min(N_EVAL, T - n_train)
        X_train = x_all[:n_train].numpy()
        S_train = S_all[:n_train].numpy()
        X_eval = x_all[n_train:n_train+n_eval].numpy()
        S_eval = S_all[n_train:n_train+n_eval].numpy()

        # Ridge regression with cross-validation
        alphas = [1e-4, 1e-2, 1.0, 10.0, 100.0]
        ridge = RidgeCV(alphas=alphas, fit_intercept=True)
        ridge.fit(X_train, S_train)
        S_pred_eval = ridge.predict(X_eval)

        mse = float(np.mean((S_pred_eval - S_eval) ** 2))
        mse_oracle_baseline = float(np.mean(S_eval ** 2))  # predicting 0
        rel_mse = mse / (mse_oracle_baseline + 1e-12)
        print(f"  Block {b}, r={r}: ridge MSE={mse:.4e}, rel_MSE={rel_mse:.4f}, best_alpha={ridge.alpha_:.4f}")
        print(f"    predictor params: {r}*768+{r} = {r*768+r}")

        phase3_results[b] = {
            "r": r,
            "train_tokens": int(n_train),
            "eval_tokens": int(n_eval),
            "ridge_mse": mse,
            "oracle_mse_baseline": mse_oracle_baseline,
            "rel_mse": rel_mse,
            "best_alpha": float(ridge.alpha_),
            "predictor_params": r * 768 + r,
        }

    # Store (U_r, V_r, ridge) for Phase 4 for this block
    phase3_results[b]["_U_r"] = U_r
    phase3_results[b]["_V_r"] = V_r
    phase3_results[b]["_ridge"] = ridge


# ===========================================================================
# Phase 4 — Single-block swap PPL
# ===========================================================================
print("\n=== Phase 4: Single-block swap PPL ===")

# Custom FFN module
class JacobianFactorizedMLP(nn.Module):
    """FFN replaced by f(x) = U @ diag(S_pred(x)) @ V^T @ x + b_out."""
    def __init__(self, U: torch.Tensor, V: torch.Tensor,
                 A: torch.Tensor, b_ridge: torch.Tensor,
                 b_out: torch.Tensor | None):
        super().__init__()
        # U: (768, r), V: (768, r), A: (r, 768), b_ridge: (r,)
        self.register_buffer("U", U.float())
        self.register_buffer("V", V.float())
        self.register_buffer("A", A.float())
        self.register_buffer("b_ridge", b_ridge.float())
        if b_out is not None:
            self.register_buffer("b_out", b_out.float())
        else:
            self.b_out = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 768)
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])  # (T, 768)
        S = x_flat @ self.A.T + self.b_ridge  # (T, r)
        # f(x_t) = U @ diag(S_t) @ V^T @ x_t = U @ (S_t * (V^T @ x_t))
        VTx = x_flat @ self.V     # (T, r)
        SVTx = S * VTx            # (T, r)  element-wise
        out = SVTx @ self.U.T     # (T, 768)
        if self.b_out is not None:
            out = out + self.b_out
        return out.reshape(shape)


class JacobianFactorizedMLPOracle(nn.Module):
    """Oracle: uses true S(x) computed from Jacobian (vectorized)."""
    def __init__(self, U: torch.Tensor, V: torch.Tensor,
                 W_in: torch.Tensor, W_out_mat: torch.Tensor,
                 b_out: torch.Tensor | None,
                 b_in: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("U", U.float())           # (768, r)
        self.register_buffer("V", V.float())           # (768, r)
        # Precompute M = (U^T @ W_out) * (W_in @ V)^T for vectorized S computation
        # S[t,i] = sum_h sigma_p[t,h] * (U^T@W_out)[i,h] * (W_in@V)[h,i]
        UtWout = (U.T @ W_out_mat.float()).contiguous()   # (r, 3072)
        WinV = (W_in.float() @ V.float()).contiguous()    # (3072, r)
        M = UtWout * WinV.T   # (r, 3072): precomputed factor
        self.register_buffer("M", M.float())    # (r, 3072)
        if b_out is not None:
            self.register_buffer("b_out", b_out.float())
        else:
            self.b_out = None
        self.register_buffer("W_in", W_in.float())
        if b_in is not None:
            self.register_buffer("b_in", b_in.float())
        else:
            self.b_in = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 768)
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])  # (T, 768)
        # z = x @ W_in^T + b_in (Jacobian is independent of bias, but gelu_grad uses z with bias)
        z = x_flat @ self.W_in.T
        if self.b_in is not None:
            z = z + self.b_in
        sigma_p = gelu_grad(z)             # (T, 3072)
        # S[t, i] = sigma_p[t] . M[i]  (dot product per row of M)
        S = sigma_p @ self.M.T             # (T, r)
        # out = U @ (S * (V^T @ x))
        VTx = x_flat @ self.V              # (T, r)
        out = (S * VTx) @ self.U.T         # (T, 768)
        if self.b_out is not None:
            out = out + self.b_out
        return out.reshape(shape)


def eval_ppl_model(swapped_model, teacher, corpus) -> float:
    metrics = _evaluate_model(teacher, swapped_model, corpus, batch_size=4)
    return float(metrics["perplexity"])


from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    derive_ffn_compression,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

phase4_results = {}

for b in AUDIT_BLOCKS:
    print(f"\n  Block {b}:")
    U_r = phase3_results[b]["_U_r"]
    V_r = phase3_results[b]["_V_r"]
    ridge = phase3_results[b]["_ridge"]
    r = phase3_results[b]["r"]

    A = torch.tensor(ridge.coef_, dtype=torch.float32)      # (r, 768)
    b_ridge = torch.tensor(ridge.intercept_, dtype=torch.float32)  # (r,)
    b_out_buf = b_outs[b]
    b_in_buf = b_ins[b]

    W_in_b = W_ins[b]
    W_out_b = W_outs[b]

    ppl_baseline_list = []
    ppl_oracle_list = []
    ppl_pred_list = []
    ppl_random_list = []

    for seed in SEEDS:
        corpus = build_probe_corpus(ProbeCorpusSpec(
            text_path=str(DATA_PATH),
            sequence_length=SEQ_LEN,
            probe_train_sequences=8,
            probe_holdout_sequences=PROBE_HOLDOUT_SEQ,
            tokenizer=tokenizer,
            shuffle_seed=seed,
            dataset_name="shakespeare",
        ))

        # Baseline
        ppl_base = eval_ppl_model(model, teacher, corpus)
        ppl_baseline_list.append(ppl_base)

        # Predicted S(x)
        mlp_pred = JacobianFactorizedMLP(U_r, V_r, A, b_ridge, b_out_buf)
        model_pred = copy.deepcopy(model)
        model_pred.transformer.h[b].mlp = mlp_pred
        model_pred.eval()
        ppl_p = eval_ppl_model(model_pred, teacher, corpus)
        ppl_pred_list.append(ppl_p)
        del model_pred

        # Oracle S(x)
        mlp_oracle = JacobianFactorizedMLPOracle(U_r, V_r, W_in_b, W_out_b, b_out_buf, b_in=b_in_buf)
        model_oracle = copy.deepcopy(model)
        model_oracle.transformer.h[b].mlp = mlp_oracle
        model_oracle.eval()
        ppl_o = eval_ppl_model(model_oracle, teacher, corpus)
        ppl_oracle_list.append(ppl_o)
        del model_oracle

        # Random U, V null (oracle S on random basis)
        torch.manual_seed(seed + 1000)
        Q1, _ = torch.linalg.qr(torch.randn(768, r))
        Q2, _ = torch.linalg.qr(torch.randn(768, r))
        mlp_rand = JacobianFactorizedMLPOracle(Q1, Q2, W_in_b, W_out_b, b_out_buf, b_in=b_in_buf)
        model_rand = copy.deepcopy(model)
        model_rand.transformer.h[b].mlp = mlp_rand
        model_rand.eval()
        ppl_r = eval_ppl_model(model_rand, teacher, corpus)
        ppl_random_list.append(ppl_r)
        del model_rand

        print(f"    seed={seed}: base={ppl_base:.2f}, oracle={ppl_o:.2f}, pred={ppl_p:.2f}, rand={ppl_r:.2f}")

    # topk+lstsq at equivalent compression
    # Factorized params: U(768xr) + V(768xr) + A(rx768) + b(r) = 3*768*r + r ≈ 148K for r=64
    # topk+lstsq: k*768 (W_in_k) + k*768 (W_out_k lstsq) = 2k*768 → k=1.5*r for param match
    k_topk = max(32, int(1.5 * r))
    print(f"\n    topk+lstsq baseline (k={k_topk}):")
    ppl_topk_list = []
    for seed in SEEDS:
        corpus_topk = build_probe_corpus(ProbeCorpusSpec(
            text_path=str(DATA_PATH),
            sequence_length=SEQ_LEN,
            probe_train_sequences=8,
            probe_holdout_sequences=PROBE_HOLDOUT_SEQ,
            tokenizer=tokenizer,
            shuffle_seed=seed,
            dataset_name="shakespeare",
        ))
        try:
            atlas = probe_ffn_geometry(
                teacher=teacher,
                corpus=corpus_topk,
                layer_ids=(b,),
                batch_size=4,
            )
            pack = derive_ffn_compression(
                atlas=atlas,
                budget=CompressionBudgetSpec(
                    per_layer_hidden={b: k_topk},
                    selection_mode="topk",
                ),
            )
            model_topk = swap_compressed_ffns(teacher, pack)
            model_topk.eval()
            ppl_tk = eval_ppl_model(model_topk, teacher, corpus_topk)
            ppl_topk_list.append(ppl_tk)
            del model_topk
            print(f"      seed={seed}: topk PPL={ppl_tk:.2f}")
        except Exception as e:
            print(f"      seed={seed}: topk FAILED: {e}")
            ppl_topk_list.append(float("nan"))

    def mean(lst):
        vals = [v for v in lst if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    ppl_base_mean = mean(ppl_baseline_list)
    ppl_oracle_mean = mean(ppl_oracle_list)
    ppl_pred_mean = mean(ppl_pred_list)
    ppl_rand_mean = mean(ppl_random_list)
    ppl_topk_mean = mean(ppl_topk_list)

    rel_oracle = (ppl_oracle_mean - ppl_base_mean) / ppl_base_mean * 100
    rel_pred = (ppl_pred_mean - ppl_base_mean) / ppl_base_mean * 100
    rel_rand = (ppl_rand_mean - ppl_base_mean) / ppl_base_mean * 100
    rel_topk = (ppl_topk_mean - ppl_base_mean) / ppl_base_mean * 100

    print(f"\n  Block {b} summary (r=64):")
    print(f"    Baseline PPL: {ppl_base_mean:.2f}")
    print(f"    Oracle PPL:   {ppl_oracle_mean:.2f}  ({rel_oracle:+.1f}%)")
    print(f"    Predicted PPL:{ppl_pred_mean:.2f}  ({rel_pred:+.1f}%)")
    print(f"    Random PPL:   {ppl_rand_mean:.2f}  ({rel_rand:+.1f}%)")
    print(f"    TopK PPL:     {ppl_topk_mean:.2f}  ({rel_topk:+.1f}%)")

    phase4_results[b] = {
        "r": r,
        "k_topk": k_topk,
        "ppl_baseline": ppl_base_mean,
        "ppl_oracle": ppl_oracle_mean,
        "ppl_pred": ppl_pred_mean,
        "ppl_random_null": ppl_rand_mean,
        "ppl_topk": ppl_topk_mean,
        "rel_ppl_oracle": rel_oracle,
        "rel_ppl_pred": rel_pred,
        "rel_ppl_random": rel_rand,
        "rel_ppl_topk": rel_topk,
        "seeds_baseline": ppl_baseline_list,
        "seeds_oracle": ppl_oracle_list,
        "seeds_pred": ppl_pred_list,
        "seeds_random": ppl_random_list,
        "seeds_topk": ppl_topk_list,
    }


# ===========================================================================
# Evaluate criteria
# ===========================================================================
print("\n=== Outcome Evaluation ===")
# Confirmed: rank-64 factorization achieves <= +15% single-block PPL for >= 2/4 blocks
# Refuted: (a) median rel error > 0.5 at r=128, OR (b) rank-64 > +30% everywhere

blocks_within_15pct = sum(
    1 for b in AUDIT_BLOCKS
    if phase4_results[b]["rel_ppl_pred"] <= 15.0
)
blocks_within_30pct = sum(
    1 for b in AUDIT_BLOCKS
    if phase4_results[b]["rel_ppl_pred"] <= 30.0
)

print(f"  Blocks within +15% PPL (pred): {blocks_within_15pct}/4")
print(f"  Blocks within +30% PPL (pred): {blocks_within_30pct}/4")
print(f"  Refutation gate (r=128 error): {median_r128:.4f} (>0.5 → refuted)")

if refuted:
    outcome = "REFUTED"
    outcome_reason = f"mean-Jacobian SVD median relative error at r=128 = {median_r128:.3f} > 0.5"
elif blocks_within_15pct >= 2 and not refuted:
    outcome = "CONFIRMED"
    outcome_reason = f"{blocks_within_15pct}/4 blocks achieve ≤+15% PPL at r=64"
elif blocks_within_30pct == 0 and not refuted:
    outcome = "REFUTED"
    outcome_reason = f"rank-64 > +30% PPL in all blocks"
else:
    outcome = "INCONCLUSIVE"
    outcome_reason = f"{blocks_within_15pct}/4 blocks ≤+15%, {blocks_within_30pct}/4 ≤+30%"

print(f"\n  OUTCOME: {outcome} — {outcome_reason}")


# ===========================================================================
# Save results
# ===========================================================================
results = {
    "cycle": 100,
    "item": 103,
    "outcome": outcome,
    "outcome_reason": outcome_reason,
    "phase1_rank_audit": {str(b): v for b, v in phase1_results.items()},
    "phase2_factorization": {str(b): {k: v for k, v in phase2_results[b].items() if not k.startswith("_")} for b in AUDIT_BLOCKS},
    "phase2_refutation_gate": {
        "median_r128_error": median_r128,
        "refuted": refuted,
    },
    "phase3_predictor": {str(b): {k: v for k, v in phase3_results[b].items() if not k.startswith("_")} for b in AUDIT_BLOCKS},
    "phase4_ppl": {str(b): phase4_results[b] for b in AUDIT_BLOCKS},
}

json_path = OUTPUT_DIR / "cycle_100_jacobian_factorization.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved JSON: {json_path}")


# ===========================================================================
# Plot
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Phase 1 rank audit
ax = axes[0]
blocks_plot = AUDIT_BLOCKS
med95 = [phase1_results[b]["median_rank_95"] for b in blocks_plot]
med99 = [phase1_results[b]["median_rank_99"] for b in blocks_plot]
ax.bar([x - 0.2 for x in range(len(blocks_plot))], med95, 0.4, label="rank_95", alpha=0.8)
ax.bar([x + 0.2 for x in range(len(blocks_plot))], med99, 0.4, label="rank_99", alpha=0.8)
ax.set_xticks(range(len(blocks_plot)))
ax.set_xticklabels([f"B{b}" for b in blocks_plot])
ax.axhline(768, color="red", linestyle="--", alpha=0.5, label="max=768")
ax.set_ylabel("Median effective rank")
ax.set_title("Phase 1: Per-token Jacobian rank")
ax.legend()

# Panel 2: Phase 2 reconstruction error vs rank
ax = axes[1]
for b in AUDIT_BLOCKS:
    errs = [phase2_results[b][f"r{r}_median_rel_err"] for r in RANKS]
    ax.plot(RANKS, errs, "o-", label=f"B{b}")
ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="refutation threshold")
ax.set_xlabel("Rank r")
ax.set_ylabel("Median relative Frobenius error")
ax.set_title("Phase 2: Mean-Jacobian factorization error")
ax.legend()
ax.set_xscale("log")

# Panel 3: Phase 4 PPL comparison
ax = axes[2]
x_pos = np.arange(len(AUDIT_BLOCKS))
w = 0.2
labels = ["Oracle", "Pred", "Random", "TopK"]
colors = ["green", "blue", "orange", "purple"]
for i, (label, color, key) in enumerate(zip(
    labels, colors, ["rel_ppl_oracle", "rel_ppl_pred", "rel_ppl_random", "rel_ppl_topk"]
)):
    vals = [phase4_results[b][key] for b in AUDIT_BLOCKS]
    ax.bar(x_pos + (i-1.5)*w, vals, w, label=label, alpha=0.8, color=color)
ax.axhline(15.0, color="green", linestyle="--", alpha=0.7, label="+15% threshold")
ax.axhline(30.0, color="red", linestyle="--", alpha=0.7, label="+30% refutation")
ax.axhline(0.0, color="black", linestyle="-", alpha=0.3)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"B{b}" for b in AUDIT_BLOCKS])
ax.set_ylabel("Relative PPL change (%)")
ax.set_title(f"Phase 4: Single-block swap PPL (r=64)\n{outcome}")
ax.legend(fontsize=7)

plt.suptitle(f"Cycle 100: Jacobian Factorization — {outcome}", fontsize=12)
plt.tight_layout()
plot_path = OUTPUT_DIR / "cycle_100_jacobian_factorization.png"
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved plot: {plot_path}")

print("\nDone.")
