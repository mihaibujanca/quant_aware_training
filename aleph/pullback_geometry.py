"""Pullback-metric geometry for transformer residual streams.

Implements the curved inference framework (Manson 2025, arXiv 2507.21107)
for measuring semantic geometry of residual stream trajectories, plus
G-weighted least-squares for compression.

The pullback metric G = U^T U is induced by the unembedding matrix U.
Under this metric, distances in residual space correspond to shifts in
output token probabilities — a semantically meaningful geometry.

Key quantities:
  - Salience S_l = ||x^(l+1) - x^(l)||_G  (semantic velocity)
  - Curvature kappa_l via 3-point central differences under G
  - G-weighted lstsq for semantically-aware compression fitting
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def collect_residual_trajectory(
    model: nn.Module,
    input_ids: Tensor,
) -> list[Tensor]:
    """Collect residual stream states at all layer boundaries.

    Hooks post-embedding and each block output (post-residual-add,
    pre-LayerNorm — per MP7: geometry must be measured before LayerNorm).

    Args:
        model: GPT2LMHeadModel (or compatible).
        input_ids: [batch, seq_len] token IDs.

    Returns:
        List of 13 tensors [batch, seq_len, d_model] for x^(0)..x^(12).
        x^(0) = post-embedding + positional, x^(l) = output of block l-1.
    """
    trajectory: list[Tensor] = []
    hooks: list[torch.utils.hooks.RemovableHook] = []

    # x^(0): post-embedding (output of transformer.drop)
    def _embed_hook(_module: nn.Module, _input: tuple, output: Tensor) -> None:
        trajectory.append(output.detach())

    hooks.append(model.transformer.drop.register_forward_hook(_embed_hook))

    # x^(1)..x^(12): output of each transformer block
    for block in model.transformer.h:

        def _block_hook(_module: nn.Module, _input: tuple, output: tuple) -> None:
            # Block output is (hidden_states, ...) — first element is residual
            trajectory.append(output[0].detach())

        hooks.append(block.register_forward_hook(_block_hook))

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return trajectory


def get_unembedding(model: nn.Module) -> Tensor:
    """Extract the unembedding matrix U from a GPT-2 model.

    Returns:
        U: [vocab_size, d_model] unembedding weight (detached).
    """
    return model.lm_head.weight.detach()


def pullback_norm(v: Tensor, U: Tensor) -> Tensor:
    """Compute ||v||_G = ||Uv||_2 under pullback metric G = U^T U.

    Args:
        v: [..., d_model] vectors.
        U: [vocab_size, d_model] unembedding matrix.

    Returns:
        [...] tensor of G-norms.
    """
    # v @ U^T gives [..., vocab_size], then L2 norm over last dim
    return torch.linalg.norm(v @ U.T, dim=-1)


def pullback_inner(u: Tensor, v: Tensor, U: Tensor) -> Tensor:
    """Compute <u, v>_G = (Uu) . (Uv) under pullback metric.

    Args:
        u, v: [..., d_model] vectors.
        U: [vocab_size, d_model] unembedding matrix.

    Returns:
        [...] tensor of G-inner-products.
    """
    return ((u @ U.T) * (v @ U.T)).sum(dim=-1)


def compute_salience(
    trajectory: list[Tensor],
    U: Tensor,
    *,
    reduce: str = "mean",
) -> Tensor:
    """Per-layer salience: how far the residual stream moves under G.

    S_l = ||x^(l+1) - x^(l)||_G for l = 0..L-1.

    Args:
        trajectory: list of L+1 tensors [batch, seq_len, d_model].
        U: [vocab_size, d_model] unembedding matrix.
        reduce: "mean" averages over batch and seq_len, "none" keeps all.

    Returns:
        [L] if reduce="mean", [L, batch, seq_len] if reduce="none".
    """
    n_layers = len(trajectory) - 1
    salience = []
    for l in range(n_layers):
        delta = trajectory[l + 1] - trajectory[l]
        s = pullback_norm(delta, U)  # [batch, seq_len]
        salience.append(s)
    salience = torch.stack(salience, dim=0)  # [L, batch, seq_len]
    if reduce == "mean":
        return salience.mean(dim=(1, 2))  # [L]
    return salience


def compute_curvature(
    trajectory: list[Tensor],
    U: Tensor,
    *,
    reduce: str = "mean",
    eps: float = 1e-8,
) -> Tensor:
    """Per-layer curvature via 3-point central differences under G.

    For layers l = 1..L-1 (need l-1 and l+1):
      v_l = (x^(l+1) - x^(l-1)) / 2       (velocity)
      a_l = x^(l+1) - 2*x^(l) + x^(l-1)   (acceleration)
      kappa_l = sqrt(||a||²_G ||v||²_G - <a,v>²_G) / max(||v||³_G, eps)

    Args:
        trajectory: list of L+1 tensors [batch, seq_len, d_model].
        U: [vocab_size, d_model] unembedding matrix.
        reduce: "mean" averages over batch and seq_len, "none" keeps all.
        eps: numerical floor for ||v||³_G to avoid division by zero.

    Returns:
        [L-1] if reduce="mean", [L-1, batch, seq_len] if reduce="none".
        For GPT-2 (13 trajectory points, L=12): returns 11 values for layers 1-11.
    """
    n_layers = len(trajectory) - 1  # L
    curvature = []
    for l in range(1, n_layers):
        v = (trajectory[l + 1] - trajectory[l - 1]) / 2.0
        a = trajectory[l + 1] - 2.0 * trajectory[l] + trajectory[l - 1]

        norm_a_sq = pullback_norm(a, U).square()
        norm_v_sq = pullback_norm(v, U).square()
        inner_av = pullback_inner(a, v, U)

        # kappa = sqrt(||a||²||v||² - <a,v>²) / ||v||³
        numerator_sq = (norm_a_sq * norm_v_sq - inner_av.square()).clamp(min=0.0)
        numerator = numerator_sq.sqrt()
        denominator = norm_v_sq.sqrt().pow(3).clamp(min=eps)
        kappa = numerator / denominator
        curvature.append(kappa)

    curvature = torch.stack(curvature, dim=0)  # [L-1, batch, seq_len]
    if reduce == "mean":
        return curvature.mean(dim=(1, 2))  # [L-1]
    return curvature


def gweighted_lstsq(
    hidden: Tensor,
    outputs: Tensor,
    U: Tensor,
    *,
    use_bias: bool,
) -> tuple[Tensor, Tensor | None]:
    """G-weighted least-squares: minimize ||U(y - ŷ)||² instead of ||y - ŷ||².

    Transforms outputs into the semantic space defined by G = U^T U,
    solves standard lstsq there, then transforms back.

    Args:
        hidden: [N, k] design matrix (hidden activations).
        outputs: [N, d_model] target outputs.
        U: [vocab_size, d_model] unembedding matrix.
        use_bias: if True, appends a ones column to the design matrix.

    Returns:
        (W, b) where W is [d_model, k] and b is [d_model] or None.
    """
    if hidden.ndim != 2 or outputs.ndim != 2:
        raise ValueError("hidden and outputs must both be [N, D] matrices.")

    # G = U^T U, shape [d_model, d_model]
    G = U.T @ U
    # Regularize for numerical stability
    G = G + 1e-6 * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    L = torch.linalg.cholesky(G)  # [d_model, d_model], lower triangular

    # Transform outputs: Y' = Y @ L^T
    outputs_transformed = outputs @ L.T

    if use_bias:
        design = torch.cat([hidden, torch.ones(hidden.shape[0], 1, device=hidden.device, dtype=hidden.dtype)], dim=1)
    else:
        design = hidden

    # Solve standard lstsq in transformed space
    solution = torch.linalg.lstsq(design, outputs_transformed).solution

    # Recover: W_transformed maps to L^T-space, so W_original = W_transformed @ L^{-T}
    # solution is [k(+1), d_model] in transformed space
    L_inv_T = torch.linalg.inv(L).T  # [d_model, d_model]
    solution_original = solution @ L_inv_T

    if use_bias:
        return solution_original[:-1].t().contiguous(), solution_original[-1].contiguous()
    return solution_original.t().contiguous(), None
