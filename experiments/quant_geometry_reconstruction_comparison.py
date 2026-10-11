"""Compare a weight-map repair with post-ReLU least-squares reconstruction."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from aleph.qgeom.geometry_story import build_geometry_stories
from experiments.quant_geometry_ledger_figures import axes_style, probes, shape


def main():
    story = build_geometry_stories()[0]
    ledger = story.ledger
    # Named A/B probes are excluded from this explicitly described calibration set.
    H, Y = ledger.quant_post[:-2], ledger.fp_post[:-2]
    C = torch.linalg.lstsq(H, Y, driver="gelsd").solution
    fitted = ledger.quant_post @ C
    states = [ledger.fp_post, ledger.transported_post, fitted]
    points = np.concatenate([state.numpy() for state in states])
    low, high = points.min(axis=0), points.max(axis=0)
    center, radius = (low+high)/2, max(high-low)*.625
    bounds = tuple((c-radius, c+radius) for c in center)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    fig.suptitle("A map repair and an activation fit optimize different objects", fontsize=17)
    titles = ["FP activated target: φ(xW)", "Map repair after ReLU: φ(xWq)R",
              "Calibration fit: φ(xWq)C"]
    for i, (ax, state, title) in enumerate(zip(axes, states, titles)):
        if i:
            shape(ax, ledger.fp_post, story.path_lengths, color="#64748b", dashed=True)
        shape(ax, state, story.path_lengths)
        probes(ax, state)
        axes_style(ax, bounds, title)
    fig.supxlabel("Same Figure 1 map and input shapes. Dashed = FP target. C fits grid/circle samples; A/B are excluded.", fontsize=11)
    output = Path("docs/figures/quant_geometry")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "geometry_repair_vs_reconstruction.png", dpi=150, facecolor="white")
    plt.close(fig)
    map_loss = (H @ story.decomposition.R - Y).square().mean().item()
    fit_loss = (H @ C - Y).square().mean().item()
    W, Wq, R = (story.decomposition.weight, story.decomposition.quantized_weight,
                story.decomposition.R)
    R_map_loss = (W - Wq @ R).square().sum().item()
    C_map_loss = (W - Wq @ C).square().sum().item()
    assert fit_loss < map_loss
    assert not torch.allclose(C, R)
    assert R_map_loss < 1e-20 and C_map_loss > 1e-6
    normal_residual = (H.T @ (H @ C - Y)).abs().max().item()
    assert normal_residual < 1e-10
    payload = {"W": W.tolist(), "Wq": Wq.tolist(), "R": R.tolist(), "C": C.tolist(),
               "calibration": "Equal weight per sampled grid-line/circle point; named probes A/B excluded",
               "sample_generator": "aleph.qgeom.geometry_story.build_geometry_stories()[0].points[:-2]",
               "calibration_samples": len(H), "activation": "ReLU", "bias": [0., 0.],
               "precision": str(W.dtype), "quantization_bits": 2,
               "randomness": "None: fixed weights and deterministically sampled input paths",
               "R_map_frobenius_squared": R_map_loss, "C_map_frobenius_squared": C_map_loss,
               "R_post_activation_mse": map_loss, "C_post_activation_mse": fit_loss,
               "least_squares_normal_residual": normal_residual}
    (output / "repair_vs_reconstruction_fixture.json").write_text(json.dumps(payload, indent=2)+"\n")
    print(f"Activation fit differs from map repair: R MSE={map_loss:.6g}, C MSE={fit_loss:.6g}; LS residual={normal_residual:.3g}")


if __name__ == "__main__":
    main()
