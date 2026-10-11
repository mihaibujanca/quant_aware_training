"""Draw weight directions and measurement fibers for the scalar R/U example."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aleph.qgeom.geometry_story import build_scalar_measurement_story


def main():
    story = build_scalar_measurement_story()
    w, wq, u = (v.numpy() for v in (story.w, story.wq, story.u))
    recovered = float(story.r) * wq
    points = story.points.numpy()
    output = Path("docs/figures/quant_geometry")
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    fig.suptitle("Scalar repair changes gain, not measurement direction", fontsize=17)
    for ax in axes:
        ax.axhline(0, color="#cbd5e1", lw=.8)
        ax.axvline(0, color="#cbd5e1", lw=.8)
        ax.set(xlim=(-1.15, 1.55), ylim=(-1.05, 1.55), aspect="equal",
               xlabel="Input coordinate 1", ylabel="Input coordinate 2")
    ax = axes[0]
    ax.set_title("1. Recoverable gain and missing direction")
    for vector, color, label, offset in [
        (wq, "#2563eb", "wq = (1, 1)", (8, 12)),
        (w, "#d97706", "w = (1, 0.55)", (-60, -50)),
        (recovered, "#059669", "r wq = (0.775, 0.775)", (-150, 12)),
    ]:
        ax.annotate("", xy=vector, xytext=(0, 0),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 3})
        ax.annotate(label, vector, xytext=offset, textcoords="offset points", color=color, fontsize=11)
    ax.annotate("", xy=w, xytext=recovered,
                arrowprops={"arrowstyle": "->", "color": "#9333ea", "lw": 2.5})
    ax.annotate("u = (0.225, −0.225)", (recovered+w)/2, xytext=(-155, 0),
                textcoords="offset points", color="#9333ea", fontsize=11)
    # A small right angle depicts u perpendicular to the quantized direction.
    along, normal = wq / np.linalg.norm(wq), u / np.linalg.norm(u)
    corner = np.array([recovered + .08*along, recovered + .08*(along+normal), recovered + .08*normal])
    ax.plot(corner[:, 0], corner[:, 1], color="#9333ea", lw=1.2)
    ax.text(.02, .05, "wq and w are both nonzero: rank stays 1.\nTheir lengths and directions differ.",
            transform=ax.transAxes, fontsize=11, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})

    ax = axes[1]
    ax.set_title("2. Equal Q reading; different FP readings")
    direction = np.array([-wq[1], wq[0]]) / np.linalg.norm(wq)
    fp_direction = np.array([-w[1], w[0]]) / np.linalg.norm(w)
    t = np.linspace(-4, 4, 100)
    for level in [-.6, 0., .6, 1.2]:
        line = level*wq/(wq@wq) + t[:, None]*direction
        ax.plot(line[:, 0], line[:, 1], color="#2563eb", lw=2 if level == .6 else .8,
                alpha=1 if level == .6 else .3, label="Q fibers: xᵀwq constant" if level == .6 else None)
    for level in points @ w:
        line = level*w/(w@w) + t[:, None]*fp_direction
        ax.plot(line[:, 0], line[:, 1], "--", color="#d97706", lw=1.8,
                label="FP fibers: xᵀw constant" if level == (points @ w)[0] else None)
    for point, label, marker, offset in zip(points, ["A", "B"], ["o", "s"], [(-150, -30), (10, 12)]):
        ax.scatter(*point, s=70, marker=marker, color="#0f172a", zorder=5)
        ax.annotate(f"{label}: Q = 0.6, FP = {point @ w:.2f}", point, xytext=offset,
                    textcoords="offset points", fontsize=11,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
    ax.legend(loc="lower left", fontsize=10, framealpha=.95)
    fig.savefig(output / "geometry_scalar_fibers.png", dpi=150, facecolor="white")
    plt.close(fig)
    payload = {"bits": 2, "w": w.tolist(), "wq": wq.tolist(), "r": float(story.r),
               "u": u.tolist(), "inputs": points.tolist(), "q_responses": (points @ wq).tolist(),
               "fp_responses": (points @ w).tolist(), "repaired_responses": (points @ recovered).tolist()}
    (output / "scalar_measurement_fixture.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("Generated scalar weight-direction/fiber figure and exact fixture data")


if __name__ == "__main__":
    main()
