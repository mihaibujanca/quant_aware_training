"""
Unified curvature residual geometric framework (item 312).

Geometric object: for an activation pair (sigma, sigma_lin) and a transformer
block l whose MLP is `y = W_proj @ sigma(W_fc @ ln_2(x) + b_fc)` (HF GPT-2
Conv1D convention), the curvature residual is

    delta(z) = sigma(z) - sigma_lin(z)  in  R^d_ff
    Dy(x)    = delta(z(x)) @ W_proj     in  R^d_model    (per token)

This module computes six measurements of Dy for any (sigma, sigma_lin) pair,
subsuming cycles 217 (raw rank_95), 218 (centred Δ̃ downstream alignment),
235 (vocab projection), 258 (pullback κ shift under a first-moment-matched
swap), 301 (projection onto a frozen A factor's kept subspace).

The module is pure analysis. No training. Linalg is float32 on CPU.

Usage:
    from experiments.curvature_residual_geometric import UnifiedCurvatureFramework
    fw = UnifiedCurvatureFramework(
        model, sigma=lambda z: F.gelu(z, approximate='tanh'),
        sigma_lin=F.relu, blocks=range(12), calib_ids=calib_ids,
        device=DEVICE,
    )
    result = fw.run_all_panels()   # dict keyed by block with six panels each
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

N_BLOCKS_DEFAULT = 12
D_MODEL = 768
D_FF = 3072
CHUNK = 4096


def _top_left_svs(W: torch.Tensor, k: int = 20) -> torch.Tensor:
    """Top-k left singular vectors of W as columns. Returns (d, k)."""
    U, _, _ = torch.linalg.svd(W, full_matrices=False)
    return U[:, :k].contiguous()


def capture_cfc_inputs(
    model: nn.Module, blocks: Sequence[int], calib_ids: Sequence[torch.Tensor],
    device: torch.device,
) -> dict:
    """Forward the calib corpus once; capture ln_2(x) at each hooked c_fc input.
    Returns dict[l] -> (N_tokens, 768) CPU float32."""
    stash = {l: [] for l in blocks}

    def make_prehook(l):
        def fn(_mod, args):
            stash[l].append(args[0].detach().float().cpu())
        return fn

    hooks = []
    for l in blocks:
        h = model.transformer.h[l].mlp.c_fc.register_forward_pre_hook(make_prehook(l))
        hooks.append(h)
    with torch.no_grad():
        for ids in calib_ids:
            _ = model(ids.to(device))
    for h in hooks:
        h.remove()
    out = {}
    for l in blocks:
        t = torch.cat(stash[l], dim=0).reshape(-1, D_MODEL)
        out[l] = t
        stash[l] = None
    return out


class UnifiedCurvatureFramework:
    """Unified framework over any (sigma, sigma_lin) activation pair.

    Constructor args:
        model: HF GPT2LMHeadModel (or compatible structure with transformer.h).
        sigma: callable z -> sigma(z); must accept (*, 3072) float32 CPU tensor.
        sigma_lin: callable z -> sigma_lin(z); same signature as sigma.
        blocks: iterable of block indices to analyse.
        calib_ids: list of (1, T) LongTensor input ids (GPU).
        device: torch.device for the forward pass.
        tokenizer: optional GPT2TokenizerFast for vocab decoding (panel 3).
        A_factors: optional dict[l] -> Vh_2xd (rank-r kept output basis of W_proj),
                   for panel 4.
        sigma_name / sigma_lin_name: strings for provenance.
    """

    def __init__(
        self,
        model: nn.Module,
        sigma: Callable[[torch.Tensor], torch.Tensor],
        sigma_lin: Callable[[torch.Tensor], torch.Tensor],
        blocks: Iterable[int],
        calib_ids: Sequence[torch.Tensor],
        device: torch.device,
        tokenizer: Optional[GPT2TokenizerFast] = None,
        A_factors: Optional[dict] = None,
        sigma_name: str = "sigma",
        sigma_lin_name: str = "sigma_lin",
    ):
        self.model = model
        self.sigma = sigma
        self.sigma_lin = sigma_lin
        self.blocks: List[int] = list(blocks)
        self.calib_ids = calib_ids
        self.device = device
        self.tokenizer = tokenizer
        self.A_factors = A_factors or {}
        self.sigma_name = sigma_name
        self.sigma_lin_name = sigma_lin_name

        # Cached: mlp_inputs[l] (N, 768) CPU float32
        self._mlp_inputs: Optional[dict] = None
        # Cached: per block (Dy, Dy_centred, top-20 Vh, mean_delta_3072)
        self._per_block_cache: dict = {}

    # ── Shared plumbing ──────────────────────────────────────────────────────
    def capture_inputs(self):
        if self._mlp_inputs is None:
            self._mlp_inputs = capture_cfc_inputs(
                self.model, self.blocks, self.calib_ids, self.device
            )
        return self._mlp_inputs

    def compute_Dy(self, l: int) -> dict:
        """Compute Dy and Dy_centred for block l (chunked on CPU)."""
        if l in self._per_block_cache:
            return self._per_block_cache[l]
        mlp_inputs = self.capture_inputs()
        x = mlp_inputs[l]  # (N, 768) CPU float32
        block = self.model.transformer.h[l]
        W_fc = block.mlp.c_fc.weight.detach().float().cpu()      # (768, 3072)
        b_fc = block.mlp.c_fc.bias.detach().float().cpu()        # (3072,)
        W_proj = block.mlp.c_proj.weight.detach().float().cpu()  # (3072, 768)
        assert W_fc.shape == (D_MODEL, D_FF), f"W_fc {W_fc.shape} block {l}"
        assert W_proj.shape == (D_FF, D_MODEL), f"W_proj {W_proj.shape} block {l}"

        parts = []
        delta_mean_accum = torch.zeros(D_FF, dtype=torch.float32)
        count = 0
        for i in range(0, x.shape[0], CHUNK):
            xc = x[i:i + CHUNK]
            preact = xc @ W_fc + b_fc            # (c, 3072)
            delta_post = self.sigma(preact) - self.sigma_lin(preact)  # (c, 3072)
            delta_mean_accum += delta_post.sum(0)
            count += delta_post.shape[0]
            parts.append(delta_post @ W_proj)    # (c, 768)
        Dy = torch.cat(parts, dim=0)              # (N, 768)
        delta_mean_3072 = delta_mean_accum / count  # (3072,)
        Dy_mean = Dy.mean(0)                       # (768,)
        Dy_centred = Dy - Dy_mean

        entry = {
            "Dy": Dy,
            "Dy_centred": Dy_centred,
            "Dy_mean": Dy_mean,
            "delta_mean_3072": delta_mean_3072,
            "W_fc": W_fc,
            "b_fc": b_fc,
            "W_proj": W_proj,
        }
        self._per_block_cache[l] = entry
        return entry

    def free_block(self, l: int):
        """Drop Dy caches for block l after all panels computed."""
        self._per_block_cache.pop(l, None)

    # ── Panel 1: cycle 217 raw Δy rank_95 ────────────────────────────────────
    def panel_raw_rank(self, l: int) -> dict:
        c = self.compute_Dy(l)
        Dy = c["Dy"]
        U, S, Vh = torch.linalg.svd(Dy, full_matrices=False)
        energy = S ** 2
        cum = energy.cumsum(0) / energy.sum()
        rank_95 = int((cum < 0.95).sum().item()) + 1
        rank_99 = int((cum < 0.99).sum().item()) + 1
        return {
            "rank_95_raw": rank_95,
            "rank_99_raw": rank_99,
            "top5_singular_values_raw": [float(s) for s in S[:5].tolist()],
            "delta_norm_mean": float(torch.linalg.norm(Dy, dim=1).mean()),
            "Vh_top3_raw": Vh[:3].clone(),
        }

    # ── Panel 2: cycle 218 centred Δ̃ downstream alignment ──────────────────
    def panel_centred_alignment(self, l: int, l_next: Optional[int] = None) -> dict:
        c = self.compute_Dy(l)
        Dy_c = c["Dy_centred"]
        U, S, Vh = torch.linalg.svd(Dy_c, full_matrices=False)
        energy = S ** 2
        cum = energy.cumsum(0) / energy.sum()
        rank_95_c = int((cum < 0.95).sum().item()) + 1
        rank_99_c = int((cum < 0.99).sum().item()) + 1
        V20 = Vh[:20].clone()  # (20, 768)

        out = {
            "rank_95_centred": rank_95_c,
            "rank_99_centred": rank_99_c,
            "sigma0": float(S[0].item()),
            "sigma1": float(S[1].item()),
            "sigma2": float(S[2].item()),
            "Vh_top20_centred": V20,
            "Vh_top5_centred": V20[:5].clone(),
        }

        if l_next is None:
            l_next = l + 1
        if l_next < len(self.model.transformer.h):
            blk_next = self.model.transformer.h[l_next]
            W_fc_next = blk_next.mlp.c_fc.weight.detach().float().cpu()   # (768, 3072)
            W_attn_next = blk_next.attn.c_attn.weight.detach().float().cpu()  # (768, 2304)
            assert W_attn_next.shape == (D_MODEL, 3 * D_MODEL)
            Q = W_attn_next[:, 0:D_MODEL]
            K = W_attn_next[:, D_MODEL:2 * D_MODEL]
            V = W_attn_next[:, 2 * D_MODEL:3 * D_MODEL]

            U_cfc = _top_left_svs(W_fc_next, 20)
            U_q = _top_left_svs(Q, 20)
            U_k = _top_left_svs(K, 20)
            U_v = _top_left_svs(V, 20)

            readers = {"c_fc": U_cfc, "Q": U_q, "K": U_k, "V": U_v}
            for name, B in readers.items():
                M = V20 @ B  # (20, 20)
                cos = torch.linalg.svdvals(M)  # (20,)
                out[f"{name}_top5_cos_sum"] = float(cos[:5].sum().item())
                out[f"{name}_top20_cos_sum"] = float(cos.sum().item())
        return out

    # ── Panel 3: cycle 235 vocab projection ──────────────────────────────────
    def panel_vocab_projection(
        self, l: int, wte: torch.Tensor, top_k_dirs: int = 5, top_n_tokens: int = 20,
    ) -> dict:
        c = self.compute_Dy(l)
        Dy_c = c["Dy_centred"]
        U, S, Vh = torch.linalg.svd(Dy_c, full_matrices=False)
        top_dirs = []
        for i in range(top_k_dirs):
            v = Vh[i]
            logit_impact = wte @ v  # (V,)
            top_boost = torch.topk(logit_impact, top_n_tokens)
            top_supp = torch.topk(-logit_impact, top_n_tokens)
            entry = {
                "dir_idx": i,
                "singular_value": float(S[i].item()),
                "boost_ids": [int(i) for i in top_boost.indices.tolist()],
                "boost_scores": [float(s) for s in top_boost.values.tolist()],
                "supp_ids": [int(i) for i in top_supp.indices.tolist()],
                "supp_scores": [float(-s) for s in top_supp.values.tolist()],
            }
            if self.tokenizer is not None:
                entry["boost_tokens"] = [self.tokenizer.decode([i]) for i in entry["boost_ids"]]
                entry["supp_tokens"] = [self.tokenizer.decode([i]) for i in entry["supp_ids"]]
            top_dirs.append(entry)
        return {"top_directions": top_dirs}

    # ── Panel 4: cycle 301 A-projection ──────────────────────────────────────
    def panel_A_projection(self, l: int, rank: int = 2) -> Optional[dict]:
        c = self.compute_Dy(l)
        W_proj = c["W_proj"]
        Dy = c["Dy"]
        U_, S_, Vh = torch.linalg.svd(W_proj, full_matrices=False)
        Vh_r = Vh[:rank, :].contiguous()
        # Orthonormality sanity
        II = Vh_r @ Vh_r.T
        off_diag = float((II - torch.eye(rank)).abs().max().item())
        Dy_norm_sq = float((Dy * Dy).sum().item())
        Dy_kept = Dy @ Vh_r.T  # (N, rank)
        e_top = float((Dy_kept * Dy_kept).sum().item()) / Dy_norm_sq
        return {
            "rank": rank,
            "E_kept": e_top,
            "Dy_frob_sq": Dy_norm_sq,
            "W_proj_top5_S": [float(s) for s in S_[:5].tolist()],
            "Vh_r_max_off_diag": off_diag,
        }

    # ── Panel 5: cycle 258 κ shift via first-moment linearised swap ─────────
    def compute_b_leak(self) -> dict:
        """Per-block per-channel offset b_leak = E[sigma(z) - sigma_lin(z)]
        in R^3072, computed over the captured calib slice. This is the
        first-moment correction that matches the mean of sigma under sigma_lin.
        """
        b_leak = {}
        for l in self.blocks:
            c = self.compute_Dy(l)
            b_leak[l] = c["delta_mean_3072"].to(self.device)
        return b_leak


class LinearisedAct(nn.Module):
    """Patch module: sigma_lin(z) + b_leak. Matches the first moment of sigma(z)
    at the 3072-dim intermediate level (cycle 258 variant B)."""

    def __init__(self, sigma_lin: Callable[[torch.Tensor], torch.Tensor], b_leak: torch.Tensor):
        super().__init__()
        self.sigma_lin = sigma_lin
        self.register_buffer("b_leak", b_leak)

    def forward(self, z):
        return self.sigma_lin(z) + self.b_leak
