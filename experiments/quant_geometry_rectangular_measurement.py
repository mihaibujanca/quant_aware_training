"""Visualize architectural dimension reduction with no quantization rank drop."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aleph.qgeom.geometry_story import build_rectangular_measurement_story
from experiments import quant_geometry_style as style


def main():
    story = build_rectangular_measurement_story()
    decomposition = story.decomposition
    W, Wq, R, U = (t.numpy() for t in (decomposition.weight, decomposition.quantized_weight,
                                      decomposition.R, decomposition.U))
    inputs = story.points.numpy()
    center = inputs.mean(axis=0)
    direction = story.fiber_direction.numpy()
    q, fp, repaired = inputs @ Wq, inputs @ W, inputs @ Wq @ R
    # A 3->2 map flattens exactly one input direction. W_q flattens d; W flattens a tilted direction n.
    n = np.linalg.svd(W.T)[2][-1]
    n = n / n[0]
    np.testing.assert_allclose(n @ W, 0, atol=1e-12)
    np.testing.assert_allclose(direction @ Wq, 0, atol=1e-12)
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(17, 5.4), constrained_layout=True)
    fig.suptitle("Inputs that differ only along the direction $W_q$ flattens", fontsize=17)

    ax = fig.add_subplot(1, 4, 1, projection="3d")
    t = np.linspace(-.55, .55, 2)[:, None]
    ax.plot(*(center + t*direction).T, color=style.Q, lw=3)
    ax.plot(*(center + .5*t*n).T, "--", color=style.FP, lw=3)
    for point, name in zip(inputs, "CD"):
        ax.scatter(*point, color=style.INPUT, marker=style.MARKER[name], s=70, depthshade=False)
        ax.text(*(point + .06), name, fontsize=13)
    ax.set(title="1. Input space: C and D lie on one fiber of $W_q$", xlabel="Input 1", ylabel="Input 2",
           zlabel="Input 3", xticks=[], yticks=[], zticks=[])
    ax.view_init(elev=18, azim=35)
    ax.set_box_aspect((1, 1, 1))

    bounds = ((.25, .7), (-.35, .05))
    panels = [("2. Quantized: the same point", [("q", q, None)], []),
              ("3. FP: different points", [("fp", fp, True)], []),
              ("4. $R$ moves the shared point;\n$xU$ supplies the difference", [("fp", fp, True), ("repaired", repaired, None)],
               [(q[0], repaired[0], "R", None)] + [(repaired[0], fp[j], "U", style.IDENTITY[c]) for j, c in enumerate("CD")])]
    for i, (title, states, arrows) in enumerate(panels):
        ax = fig.add_subplot(1, 4, i + 2)
        if i == 2:
            style.named(ax, q, "CD", "q", offsets=[(8, 8), (8, -18)])
        for kind, state, by_identity in states:
            style.named(ax, state, "CD", kind, by_identity=bool(by_identity),
                        offsets=[(-4, 10), (-4, 10)] if kind == "fp" else [(8, 8), (-30, -22)])
        for start, end, term, color in arrows:
            style.arrow(ax, start, end, term, color=color)
        style.axes_style(ax, bounds, title)
        ax.grid(alpha=.15)
    style.legend(fig, [Line2D([], [], color=style.Q, lw=3, label="fiber of $W_q$ through C and D (direction $d$)"),
                       Line2D([], [], color=style.FP, lw=3, linestyle="--", label="a fiber of $W$ (the direction $W$ flattens)"),
                       style.state_handle("q", "quantized"), style.state_handle("fp", "FP target for C"),
                       Line2D([], [], color=style.IDENTITY["D"], lw=0, marker="s", markerfacecolor="white",
                              markeredgewidth=2, label="FP target for D"),
                       style.state_handle("repaired", "after $R$"),
                       Line2D([], [], color=style.IDENTITY["C"], lw=2.4, label="$xU$ for C"),
                       Line2D([], [], color=style.IDENTITY["D"], lw=2.4, label="$xU$ for D")], ncol=4)
    output = Path("docs/figures/quant_geometry")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "geometry_rectangular_fibers.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    payload = {"bits": 2, "W": W.tolist(), "Wq": Wq.tolist(), "R": R.tolist(), "U": U.tolist(),
               "inputs": inputs.tolist(), "fiber_direction": direction.tolist(), "fp_flattened_direction": n.tolist(),
               "fp_rank": int(np.linalg.matrix_rank(W)), "q_rank": int(np.linalg.matrix_rank(Wq)),
               "q_singular_values": np.linalg.svd(Wq, compute_uv=False).tolist(),
               "q_outputs": q.tolist(), "fp_outputs": fp.tolist(), "repaired_outputs": repaired.tolist()}
    (output / "rectangular_measurement_fixture.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("Generated full-rank rectangular measurement figure and exact fixture data")


if __name__ == "__main__":
    main()
