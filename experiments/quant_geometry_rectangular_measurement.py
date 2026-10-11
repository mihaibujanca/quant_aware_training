"""Visualize architectural dimension reduction with no quantization rank drop."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aleph.qgeom.geometry_story import build_rectangular_measurement_story


def main():
    story = build_rectangular_measurement_story()
    decomposition = story.decomposition
    W, Wq, R, U = (t.numpy() for t in (decomposition.weight, decomposition.quantized_weight,
                                      decomposition.R, decomposition.U))
    inputs = story.points.numpy()
    center = inputs.mean(axis=0)
    direction = story.fiber_direction.numpy()
    q, fp, repaired = inputs @ Wq, inputs @ W, inputs @ Wq @ R
    fig = plt.figure(figsize=(12, 6.8))
    fig.subplots_adjust(left=.05, right=.98, bottom=.15, top=.83, wspace=.22)
    fig.suptitle("Three input directions, two measurements: rank stays 2", fontsize=18, y=.97)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    grid = np.linspace(-.5, .5, 5)
    a, b = np.meshgrid(grid, grid)
    plane = a[..., None]*Wq[:, 0] + b[..., None]*Wq[:, 1]
    ax.plot_surface(plane[:, :, 0], plane[:, :, 1], plane[:, :, 2], alpha=.2, color="#2563eb")
    ax.plot_wireframe(plane[:, :, 0], plane[:, :, 1], plane[:, :, 2], color="#2563eb", alpha=.35, lw=.7)
    fp_plane = a[..., None]*W[:, 0] + b[..., None]*W[:, 1]
    ax.plot_wireframe(fp_plane[:, :, 0], fp_plane[:, :, 1], fp_plane[:, :, 2], color="#d97706", alpha=.6, lw=.8)
    fiber = center + np.linspace(-.6, .6, 30)[:, None]*direction
    ax.plot(*fiber.T, color="#0f172a", lw=2)
    for point, label, color in zip(inputs, ["A", "B"], ["#d97706", "#9333ea"]):
        ax.scatter(*point, color=color, s=65, depthshade=False)
        ax.text(*(point + np.array([.04, .04, .05])), label, color=color, fontsize=13)
    ax.scatter(*center, color="#059669", s=35, depthshade=False)
    ax.set(title="1. Measured input planes turn; rank stays 2",
           xlabel="Input 1", ylabel="Input 2", zlabel="Input 3",
           xlim=(-.8, 1.), ylim=(-.8, 1.), zlim=(-.8, 1.))
    ax.view_init(elev=20, azim=35)
    ax.set_box_aspect((1, 1, 1))
    fig.supxlabel("Left: blue = Q measured directions; orange mesh = FP directions; black fiber = constant Q output", fontsize=11, y=.03)

    ax = fig.add_subplot(1, 2, 2)
    ax.set(title="2. Shared Q output; distinct FP targets", xlabel="Output coordinate 1", ylabel="Output coordinate 2",
           xlim=(.3, .95), ylim=(.38, .75), aspect="equal")
    ax.grid(alpha=.15)
    ax.scatter(*q[0], color="#2563eb", marker="s", s=80, zorder=5)
    ax.annotate("Q: A = B = (0.7, 0.5)", q[0], xytext=(-15, -30), textcoords="offset points", fontsize=11)
    ax.scatter(*repaired[0], color="#059669", s=65, zorder=5)
    ax.annotate("R-repaired: A = B", repaired[0], xytext=(-65, 35), textcoords="offset points", color="#059669", fontsize=11)
    ax.annotate("", xy=repaired[0], xytext=q[0], arrowprops={"arrowstyle": "->", "lw": 2, "color": "#059669"})
    for point, label, color, offset in zip(fp, ["A", "B"], ["#d97706", "#9333ea"], [(-20, 15), (-45, -25)]):
        ax.scatter(*point, color=color, s=70, zorder=5)
        ax.annotate(f"FP {label}", point, xytext=offset, textcoords="offset points", color=color, fontsize=12)
        ax.annotate("", xy=point, xytext=repaired[0], arrowprops={"arrowstyle": "->", "lw": 2, "color": color})
    ax.text(.02, .95, "Green arrow: output-side repair R\nOrange/purple arrows: input-dependent xU", transform=ax.transAxes,
            va="top", fontsize=11)
    output = Path("docs/figures/quant_geometry")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "geometry_rectangular_fibers.png", dpi=150, facecolor="white")
    plt.close(fig)
    payload = {"bits": 2, "W": W.tolist(), "Wq": Wq.tolist(), "R": R.tolist(), "U": U.tolist(),
               "inputs": inputs.tolist(), "fiber_direction": direction.tolist(),
               "fp_rank": int(np.linalg.matrix_rank(W)), "q_rank": int(np.linalg.matrix_rank(Wq)),
               "q_singular_values": np.linalg.svd(Wq, compute_uv=False).tolist(),
               "q_outputs": q.tolist(), "fp_outputs": fp.tolist(), "repaired_outputs": repaired.tolist()}
    (output / "rectangular_measurement_fixture.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("Generated full-rank rectangular measurement figure and exact fixture data")


if __name__ == "__main__":
    main()
