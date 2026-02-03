"""Transformer-specific geometry report extraction utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from aleph.qgeom.core import RunGeometryReport
from aleph.qgeom.metrics import build_layer_geometry_report
from aleph.quantization import get_quant_range


def _quantize_decompose(
    x: torch.Tensor,
    scale: float,
    zero_point: float,
    num_bits: int,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Quantize tensor and return (dequant, round_err, saturation_err, sat_mask)."""

    quant_min, quant_max = get_quant_range(num_bits)

    x_np = x.detach().cpu().numpy()
    scale = float(scale)
    zero_point = float(zero_point)

    x_scaled = x_np / scale + zero_point
    rounded = np.round(x_scaled)
    clipped = np.clip(rounded, quant_min, quant_max)

    dequant = (clipped - zero_point) * scale
    round_err = (rounded - x_scaled) * scale
    saturation_err = (clipped - rounded) * scale
    sat_mask = rounded != clipped

    dequant_t = torch.from_numpy(dequant).to(device=x.device, dtype=x.dtype)
    return dequant_t, round_err, saturation_err, sat_mask


def quantize_decompose_tensor(
    x: torch.Tensor,
    scale: float,
    zero_point: float,
    num_bits: int,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Public wrapper around quantization decomposition."""

    return _quantize_decompose(x, scale, zero_point, num_bits)


def collect_transformer_traces(
    model,
    x: torch.Tensor,
    scale_factors: List[float],
    zero_points: List[float],
    *,
    num_bits: int = 4,
    max_points_per_layer: int = 20000,
    rng_seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """Collect per-quant-point float/quant traces for fitting correction models."""

    rng = np.random.default_rng(rng_seed)
    model.eval()

    traces: List[Dict[str, np.ndarray]] = []

    with torch.no_grad():
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        float_x = model.dropout(model.embedding(x) + model.pos_embedding(pos))
        quant_x = float_x.clone()
        sf_idx = 0

        for layer_idx, layer in enumerate(model.layers):
            # ---- Attention quant point ----
            float_ln = layer.ln1(float_x)
            float_attn = layer.attn(float_ln)
            float_post = float_x + float_attn

            quant_ln = layer.ln1(quant_x)
            quant_attn = layer.attn(quant_ln)
            quant_pre_q = quant_x + quant_attn

            quant_post, round_err, sat_err, sat_mask = _quantize_decompose(
                quant_pre_q,
                scale_factors[sf_idx],
                zero_points[sf_idx],
                num_bits,
            )
            scale_value = float(scale_factors[sf_idx])
            sf_idx += 1

            traces.append(
                _pack_trace(
                    layer_index=2 * layer_idx,
                    sub_layer="attn",
                    z_float=float_post,
                    z_quant=quant_post,
                    quant_pre=quant_pre_q,
                    round_err=round_err,
                    sat_err=sat_err,
                    sat_mask=sat_mask,
                    scale_value=scale_value,
                    max_points=max_points_per_layer,
                    rng=rng,
                )
            )

            float_x = float_post
            quant_x = quant_post

            # ---- FFN quant point ----
            float_ln = layer.ln2(float_x)
            float_ffn = layer.ffn(float_ln)
            float_post = float_x + float_ffn

            quant_ln = layer.ln2(quant_x)
            quant_ffn = layer.ffn(quant_ln)
            quant_pre_q = quant_x + quant_ffn

            quant_post, round_err, sat_err, sat_mask = _quantize_decompose(
                quant_pre_q,
                scale_factors[sf_idx],
                zero_points[sf_idx],
                num_bits,
            )
            scale_value = float(scale_factors[sf_idx])
            sf_idx += 1

            traces.append(
                _pack_trace(
                    layer_index=2 * layer_idx + 1,
                    sub_layer="ffn",
                    z_float=float_post,
                    z_quant=quant_post,
                    quant_pre=quant_pre_q,
                    round_err=round_err,
                    sat_err=sat_err,
                    sat_mask=sat_mask,
                    scale_value=scale_value,
                    max_points=max_points_per_layer,
                    rng=rng,
                )
            )

            float_x = float_post
            quant_x = quant_post

    return traces


def _pack_trace(
    *,
    layer_index: int,
    sub_layer: str,
    z_float: torch.Tensor,
    z_quant: torch.Tensor,
    quant_pre: torch.Tensor,
    round_err: np.ndarray,
    sat_err: np.ndarray,
    sat_mask: np.ndarray,
    scale_value: float,
    max_points: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Pack and optionally subsample a trace tensor triplet."""

    zf = z_float.detach().cpu().numpy().reshape(-1, z_float.shape[-1])
    zq = z_quant.detach().cpu().numpy().reshape(-1, z_quant.shape[-1])
    zpre = quant_pre.detach().cpu().numpy().reshape(-1, quant_pre.shape[-1])
    rerr = round_err.reshape(-1, round_err.shape[-1])
    serr = sat_err.reshape(-1, sat_err.shape[-1])
    smask = sat_mask.reshape(-1, sat_mask.shape[-1])

    n = zf.shape[0]
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        zf = zf[idx]
        zq = zq[idx]
        zpre = zpre[idx]
        rerr = rerr[idx]
        serr = serr[idx]
        smask = smask[idx]

    return {
        "layer_index": np.array(layer_index, dtype=np.int64),
        "sub_layer": np.array(sub_layer),
        "z_float": zf,
        "z_quant": zq,
        "quant_residual": (zq - zpre),
        "round_err": rerr,
        "sat_err": serr,
        "sat_mask": smask.astype(np.float32),
        "scale": np.array(scale_value, dtype=np.float32),
        "true_error": (zf - zq),
    }


def collect_transformer_layer_reports(
    model,
    x: torch.Tensor,
    scale_factors: List[float],
    zero_points: List[float],
    *,
    num_bits: int = 4,
    task_name: str = "shakespeare",
) -> RunGeometryReport:
    """Collect per-quant-point geometry reports for TransformerWithCorrection."""

    model.eval()

    with torch.no_grad():
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        float_x = model.dropout(model.embedding(x) + model.pos_embedding(pos))
        quant_x = float_x.clone()

        layer_reports = []
        prev_total_error = None
        sf_idx = 0

        for layer_idx, layer in enumerate(model.layers):
            # ---- Attention quant point ----
            float_ln = layer.ln1(float_x)
            float_attn = layer.attn(float_ln)
            float_post = float_x + float_attn

            quant_ln = layer.ln1(quant_x)
            quant_attn = layer.attn(quant_ln)
            quant_pre_q = quant_x + quant_attn

            quant_post, round_err, sat_err, sat_mask = _quantize_decompose(
                quant_pre_q,
                scale_factors[sf_idx],
                zero_points[sf_idx],
                num_bits,
            )
            sf_idx += 1

            report = build_layer_geometry_report(
                layer_index=2 * layer_idx,
                linear_error=round_err,
                nonlinear_error=sat_err,
                float_pre_activation=float_post.detach().cpu().numpy(),
                quant_pre_activation=quant_pre_q.detach().cpu().numpy(),
                saturation_mask=sat_mask,
                prev_total_error=prev_total_error,
                metadata={"sub_layer": "attn", "transformer_layer": layer_idx},
            )
            layer_reports.append(report)

            float_x = float_post
            quant_x = quant_post
            prev_total_error = quant_x.detach().cpu().numpy() - float_x.detach().cpu().numpy()

            # ---- FFN quant point ----
            float_ln = layer.ln2(float_x)
            float_ffn = layer.ffn(float_ln)
            float_post = float_x + float_ffn

            quant_ln = layer.ln2(quant_x)
            quant_ffn = layer.ffn(quant_ln)
            quant_pre_q = quant_x + quant_ffn

            quant_post, round_err, sat_err, sat_mask = _quantize_decompose(
                quant_pre_q,
                scale_factors[sf_idx],
                zero_points[sf_idx],
                num_bits,
            )
            sf_idx += 1

            report = build_layer_geometry_report(
                layer_index=2 * layer_idx + 1,
                linear_error=round_err,
                nonlinear_error=sat_err,
                float_pre_activation=float_post.detach().cpu().numpy(),
                quant_pre_activation=quant_pre_q.detach().cpu().numpy(),
                saturation_mask=sat_mask,
                prev_total_error=prev_total_error,
                metadata={"sub_layer": "ffn", "transformer_layer": layer_idx},
            )
            layer_reports.append(report)

            float_x = float_post
            quant_x = quant_post
            prev_total_error = quant_x.detach().cpu().numpy() - float_x.detach().cpu().numpy()

    return RunGeometryReport(
        model_name=model.__class__.__name__,
        task_name=task_name,
        bit_width=num_bits,
        layer_reports=layer_reports,
        metadata={"n_tokens": int(T), "batch_size": int(B)},
    )


def run_report_to_layer_gain_map(run_report: RunGeometryReport) -> Dict[int, float]:
    """Convert report scores into per-layer predicted gain values."""

    gains = {}
    for report in run_report.layer_reports:
        gains[int(report.layer_index)] = float(
            report.correctability_score * report.linear_error_norm
        )
    return gains
