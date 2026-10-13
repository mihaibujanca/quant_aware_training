"""Compare map, data-conditioned preactivation, and activation repairs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from aleph.qgeom.geometry_story import build_geometry_stories, build_rectangular_measurement_story
from experiments import quant_geometry_style as style


def main():
    paths = build_geometry_stories()[0]
    rectangular = build_rectangular_measurement_story()
    decomposition = rectangular.decomposition
    W, Wq, R_map = decomposition.weight, decomposition.quantized_weight, decomposition.R
    d = rectangular.fiber_direction
    p = paths.points
    alpha = torch.tensor([.7, -.7], dtype=W.dtype)
    offsets = [-.15, 0., .15]
    # A tilted central input sheet; the three parallel sheets give rank-3 data.
    inputs = p @ torch.linalg.pinv(Wq) + (p @ alpha)[:, None] * d
    calibration = torch.cat([inputs[:-2] + offset*d for offset in offsets])
    Q, F = calibration @ Wq, calibration @ W
    H, Y = torch.relu(Q), torch.relu(F)
    R_pre = torch.linalg.lstsq(Q, F, driver="gelsd").solution
    R_post = torch.linalg.lstsq(H, Y, driver="gelsd").solution
    matrices = {"R_map": R_map, "R_pre_data": R_pre, "R_post_data": R_post}
    losses = {name: {"map_frobenius_squared": (W-Wq@R).square().sum().item(),
                     "calibration_pre_mse": (Q@R-F).square().mean().item(),
                     "calibration_post_mse": (H@R-Y).square().mean().item()}
              for name, R in matrices.items()}
    residuals = {"map": (Wq.T @ (Wq@R_map-W)).abs().max().item(),
                 "pre_data": (Q.T @ (Q@R_pre-F)).abs().max().item(),
                 "post_data": (H.T @ (H@R_post-Y)).abs().max().item()}
    second_moment = calibration.T @ calibration / len(calibration)
    assert int(torch.linalg.matrix_rank(calibration)) == 3
    assert int(torch.linalg.matrix_rank(W)) == int(torch.linalg.matrix_rank(Wq)) == 2
    assert int(torch.linalg.matrix_rank(H)) == 2
    assert all(value < 1e-9 for value in residuals.values())
    # The correlated sheet adds a predictable response along the Q blind direction.
    torch.testing.assert_close(R_pre, R_map + alpha[:, None] * (d@W)[None, :])
    torch.testing.assert_close(inputs@Wq@R_pre, inputs@W)
    for name, R in matrices.items():
        delta = W-Wq@R
        weighted_map_mse = torch.trace(delta.T@second_moment@delta) / W.shape[1]
        torch.testing.assert_close(weighted_map_mse,
                                   torch.tensor(losses[name]["calibration_pre_mse"], dtype=W.dtype))
    for key, best in [("map_frobenius_squared", "R_map"),
                      ("calibration_pre_mse", "R_pre_data"),
                      ("calibration_post_mse", "R_post_data")]:
        assert all(losses[best][key] < values[key] for name, values in losses.items() if name != best)
    # Equal weighting of orthogonal input directions recovers the map optimizer.
    isotropic_inputs = torch.cat((torch.eye(3, dtype=W.dtype), -torch.eye(3, dtype=W.dtype)))
    R_isotropic = torch.linalg.lstsq(isotropic_inputs@Wq, isotropic_inputs@W, driver="gelsd").solution
    torch.testing.assert_close(R_isotropic, R_map)

    output = Path("docs/figures/quant_geometry")
    output.mkdir(parents=True, exist_ok=True)
    raw = inputs @ Wq
    targets = [inputs @ W, torch.relu(inputs @ W)]
    rows = [[raw] + [raw@R for R in matrices.values()],
            [torch.relu(raw)] + [torch.relu(raw)@R for R in matrices.values()]]
    points = torch.cat([state for row in rows for state in row] + targets).numpy()
    low, high = points.min(axis=0), points.max(axis=0)
    center, radius = (low+high)/2, max(high-low)*.58
    bounds = tuple((c-radius, c+radius) for c in center)
    fig, axes = plt.subplots(2, 4, figsize=(17, 9.2), constrained_layout=True)
    fig.suptitle("Three objectives give three different matrices", fontsize=19)
    labels = ["Q: no correction", r"$R$: map repair",
              r"$R_{\rm pre,data}$: preactivation fit", r"$R_{\rm post,data}$: output fit"]
    for row, states in enumerate(rows):
        for column, state in enumerate(states):
            ax = axes[row, column]
            style.shape(ax, targets[row], paths.path_lengths, "fp")
            style.shape(ax, state, paths.path_lengths, "q" if column == 0 else "repaired")
            map_loss = (W-Wq).square().sum().item() if column == 0 else losses[list(matrices)[column-1]]["map_frobenius_squared"]
            label = f"{labels[column]}\nMap loss: {map_loss:.4f}" if row == 0 else ""
            style.axes_style(ax, bounds, label)
            if column == 0:
                ax.set_ylabel(("Before ReLU" if row == 0 else "ReLU first, then correction") + "\nOutput coordinate 2")
            key = "calibration_pre_mse" if row == 0 else "calibration_post_mse"
            mse = ((Q-F).square().mean() if row == 0 else (H-Y).square().mean()).item() if column == 0 else losses[list(matrices)[column-1]][key]
            ax.text(.03, .04, f"Calibration MSE: {mse:.4f}", transform=ax.transAxes, fontsize=10,
                    bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    style.legend(fig, [style.state_handle("fp"), style.state_handle("q"), style.state_handle("repaired")])
    fig.savefig(output / "geometry_repair_vs_reconstruction.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    payload = {"W": W.tolist(), "Wq": Wq.tolist(),
               **{name: R.tolist() for name, R in matrices.items()},
               "calibration": {"construction": "x = p @ pinv(Wq) + (p @ alpha + t) * d",
                               "parameter_source": "build_geometry_stories()[0].points[:-2]",
                               "alpha": alpha.tolist(), "d": d.tolist(), "t": offsets,
                               "weighting": "Equal weight per sample, including duplicate grid/circle intersections; A/B excluded on all sheets",
                               "samples": len(calibration), "input_rank": 3,
                               "second_moment": second_moment.tolist(),
                               "second_moment_eigenvalues": torch.linalg.eigvalsh(second_moment).tolist()},
               "display": {"inputs": inputs.tolist(), "path_lengths": paths.path_lengths, "t": 0.,
                           "q_probes": raw[-2:].tolist(), "fp_probes": targets[0][-2:].tolist()},
               "activation": "ReLU", "bias": [0., 0.], "precision": str(W.dtype), "quantization_bits": 2,
               "fp_rank": 2, "q_rank": 2, "q_singular_values": decomposition.singular_values.tolist(),
               "randomness": "None: fixed weights, paths, correlation and offsets",
               "uncorrected": {"map_frobenius_squared": (W-Wq).square().sum().item(),
                               "calibration_pre_mse": (Q-F).square().mean().item(),
                               "calibration_post_mse": (H-Y).square().mean().item()},
               "objectives": losses, "normal_equation_residuals": residuals,
               "isotropic_pre_fit": R_isotropic.tolist()}
    (output / "repair_vs_reconstruction_fixture.json").write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"objectives": losses, "normal_equation_residuals": residuals}, indent=2))


if __name__ == "__main__":
    main()
