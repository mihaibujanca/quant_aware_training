"""Compare map, data-conditioned preactivation, and activation repairs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from aleph.qgeom.geometry_story import build_geometry_stories, build_rectangular_measurement_story
from experiments.quant_geometry_ledger_figures import axes_style, probes, shape


def input_figure(output, inputs, calibration, lengths):
    fig = plt.figure(figsize=(11, 5.5), constrained_layout=True)
    fig.suptitle("The input geometry that each linear objective weights", fontsize=17)
    angles = np.linspace(0, 2*np.pi, 25)
    polar = np.linspace(0, np.pi, 13)
    sphere = (np.outer(np.cos(angles), np.sin(polar)),
              np.outer(np.sin(angles), np.sin(polar)),
              np.outer(np.ones_like(angles), np.cos(polar)))
    limit = max(1.1, float(calibration.abs().max())*1.08)
    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1, projection="3d")
        ax.plot_wireframe(*sphere, color="#94a3b8", alpha=.45, lw=.6)
        if i == 0:
            for direction in np.vstack((np.eye(3), -np.eye(3))):
                ax.quiver(0, 0, 0, *direction, color="#2563eb", lw=1.7, arrow_length_ratio=.1)
        else:
            points = calibration[::7].numpy()
            ax.scatter(*points.T, color="#059669", s=5, alpha=.35, depthshade=False)
            grid = inputs[:sum(lengths[:-1])].numpy().reshape(len(lengths)-1, lengths[0], 3)[::2]
            ax.plot_surface(grid[:, :, 0], grid[:, :, 1], grid[:, :, 2],
                            color="#059669", alpha=.12)
            offset = 0
            for length in lengths:
                line = inputs[offset:offset+length].numpy()
                ax.plot(*line.T, color="#059669", alpha=.6, lw=.7)
                offset += length
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
               xlabel="Input 1", ylabel="Input 2", zlabel="Input 3",
               title=["Map fit: equal weight to input directions",
                      "Data fit: correlated samples near a tilted plane"][i])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=28, azim=-120)
    fig.supxlabel("Gray unit sphere = equal directional weighting. Green = calibration geometry, with finite thickness.", fontsize=11)
    fig.savefig(output / "geometry_reconstruction_inputs.png", dpi=150, facecolor="white")
    plt.close(fig)


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
    input_figure(output, inputs, calibration, paths.path_lengths)
    raw = inputs @ Wq
    targets = [inputs @ W, torch.relu(inputs @ W)]
    rows = [[raw] + [raw@R for R in matrices.values()],
            [torch.relu(raw)] + [torch.relu(raw)@R for R in matrices.values()]]
    points = torch.cat([state for row in rows for state in row] + targets).numpy()
    low, high = points.min(axis=0), points.max(axis=0)
    center, radius = (low+high)/2, max(high-low)*.58
    bounds = tuple((c-radius, c+radius) for c in center)
    fig, axes = plt.subplots(2, 4, figsize=(17, 9.2), constrained_layout=True)
    fig.suptitle("Three objectives choose three different repairs", fontsize=19)
    labels = ["Q: no correction", r"$R_{\rm map}$: fit the map",
              r"$R_{\rm pre,data}$: fit preactivations", r"$R_{\rm post,data}$: fit activations"]
    for row, states in enumerate(rows):
        for column, state in enumerate(states):
            ax = axes[row, column]
            shape(ax, targets[row], paths.path_lengths, color="#64748b", dashed=True)
            shape(ax, state, paths.path_lengths)
            probes(ax, state)
            map_loss = (W-Wq).square().sum().item() if column == 0 else losses[list(matrices)[column-1]]["map_frobenius_squared"]
            label = f"{labels[column]}\nMap loss: {map_loss:.4f}" if row == 0 else ""
            axes_style(ax, bounds, label)
            if column == 0:
                ax.set_ylabel(("Before ReLU" if row == 0 else "ReLU first, then correction") + "\nOutput coordinate 2")
            key = "calibration_pre_mse" if row == 0 else "calibration_post_mse"
            mse = ((Q-F).square().mean() if row == 0 else (H-Y).square().mean()).item() if column == 0 else losses[list(matrices)[column-1]][key]
            ax.text(.03, .04, f"Calibration MSE: {mse:.4f}", transform=ax.transAxes, fontsize=10,
                    bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    fig.supxlabel("Dashed = FP target for the same inputs. Grid/circle = central input sheet; fits also see two parallel sheets. A/B excluded.", fontsize=11)
    fig.savefig(output / "geometry_repair_vs_reconstruction.png", dpi=150, facecolor="white")
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
