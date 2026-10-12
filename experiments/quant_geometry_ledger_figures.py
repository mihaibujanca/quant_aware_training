"""Draw the exact counterfactual states and signed vectors in the local ledger.

Run from the repository root with python -m experiments.quant_geometry_ledger_figures.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from aleph.qgeom import ErrorLedger, MapDecomposition, WeightQuantizer
from aleph.qgeom.geometry_story import build_geometry_stories


def shape(ax, state, lengths, *, color="#2563eb", dashed=False):
    points = state.numpy()
    offset = 0
    for j, length in enumerate(lengths):
        path = points[offset:offset + length]
        ax.plot(path[:, 0], path[:, 1], "--" if dashed else "-", color=color,
                lw=2.5 if j == len(lengths) - 1 else .7, alpha=.65)
        offset += length


def probes(ax, state):
    points = state[-2:].numpy()
    collision = np.linalg.norm(points[0] - points[1]) < 1e-10
    for i, (p, color, marker) in enumerate(zip(points, ["#d97706", "#9333ea"], ["o", "s"])):
        ax.scatter(*p, color=color, marker=marker, s=65, zorder=5)
        if not collision:
            ax.annotate("AB"[i], p, xytext=(8, 10 if i == 0 else -18),
                        textcoords="offset points", color=color, fontsize=12)
    if collision:
        ax.annotate("A = B", points[0], xytext=(8, -18), textcoords="offset points", fontsize=12)


def axes_style(ax, bounds, title):
    ax.axhline(0, color="#cbd5e1", lw=.7)
    ax.axvline(0, color="#cbd5e1", lw=.7)
    ax.set(xlim=bounds[0], ylim=bounds[1], aspect="equal",
           title=title, xlabel="Output coordinate 1", ylabel="Output coordinate 2")


def main():
    output = Path("docs/figures/quant_geometry")
    output.mkdir(parents=True, exist_ok=True)
    stories = build_geometry_stories()
    specifications = [
        ("uncomp", stories[0], "Uncompensated: the displacement left by an unapplied correction",
         "s₁ = A(h): corrected after activation", "s₀ = h: quantized activation", "e_uncomp = s₀ − s₁"),
        ("nonlinear", stories[1], "Nonlinear: changing the order changes the shape",
         "s₂ = φ(A(zq)): correct, then activate", "s₁ = A(h): activate, then correct", "e_NL = s₁ − s₂"),
        ("range", stories[2], "Residual deformation: its effect after activation",
         "s₃ = φ(A(zq) + xU): full precision", "s₂ = φ(A(zq)): repaired route", "e_U = s₂ − s₃"),
    ]
    for key, story, title, source_label, destination_label, vector_label in specifications:
        ledger = story.ledger
        source, destination = {
            "uncomp": (ledger.transported_post, ledger.quant_post),
            "nonlinear": (ledger.recoverable_post, ledger.transported_post),
            "range": (ledger.fp_post, ledger.recoverable_post),
        }[key]
        points = np.concatenate((source.numpy(), destination.numpy()))
        low, high = points.min(axis=0), points.max(axis=0)
        center = (low + high) / 2
        radius = max(high - low) * .625
        bounds = tuple((c - radius, c + radius) for c in center)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
        fig.suptitle(title, fontsize=16)
        for ax, state, label in zip(axes[:2], (source, destination), (source_label, destination_label)):
            shape(ax, state, story.path_lengths)
            probes(ax, state)
            axes_style(ax, bounds, label)
        ax = axes[2]
        shape(ax, source, story.path_lengths, color="#64748b", dashed=True)
        shape(ax, destination, story.path_lengths)
        # Each arrow is exactly destination(x) - source(x), with correspondence
        # retained. These are error arrows, opposite to the repair direction.
        for i in range(len(source) - 2, len(source)):
            color = ["#d97706", "#9333ea"][i - len(source) + 2]
            if torch.linalg.vector_norm(destination[i] - source[i]) > 1e-10:
                ax.annotate("", xy=destination[i].numpy(), xytext=source[i].numpy(),
                            arrowprops={"arrowstyle": "->", "color": color, "lw": 2.5,
                                        "connectionstyle": f"arc3,rad={.2 if i == len(source)-2 else -.2}" if key == "nonlinear" else "arc3,rad=0"})
        probes(ax, destination)
        axes_style(ax, bounds, vector_label + "\ndashed → blue, for A and B")
        fig.savefig(output / f"ledger_geometry_{key}.png", dpi=150, facecolor="white")
        plt.close(fig)

    W = torch.tensor([[1., -.35], [.28, .22]], dtype=torch.float64)
    decomposition = MapDecomposition.from_weights(W, WeightQuantizer(2)(W).quantized)
    ledger = ErrorLedger.compute(torch.tensor([[1., 2.]], dtype=W.dtype), decomposition)
    states = [t[0].numpy() for t in (ledger.quant_post, ledger.transported_post,
                                    ledger.recoverable_post, ledger.fp_post)]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.set_title("The ledger is a path of signed displacements: s₃ → s₂ → s₁ → s₀", fontsize=14)
    labels = ["s₀ = s₂ = (1, 0)", "s₁ = (1, −0.35)", "", "s₃ = (1.56, 0.09)"]
    offsets = [(-100, 14), (-110, -22), (0, 0), (8, 10)]
    for p, label, offset in zip(states, labels, offsets):
        ax.scatter(*p, color="#0f172a", s=55, zorder=5)
        if label:
            ax.annotate(label, p, xytext=offset, textcoords="offset points", fontsize=12)
    for i, (name, color, curvature) in enumerate([
        ("uncomp: (0, +0.35)", "#2563eb", .2),
        ("nonlinear: (0, −0.35)", "#d97706", .2),
        ("e_U (residual): (−0.56, −0.09)", "#9333ea", 0),
    ]):
        ax.annotate("", xy=states[i], xytext=states[i+1],
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 2.5,
                                "connectionstyle": f"arc3,rad={curvature}"})
        ax.plot([], [], color=color, lw=2.5, label=name)
    ax.legend(loc="lower right", fontsize=11)
    ax.set(xlim=(.65, 1.85), ylim=(-.55, .23), aspect="equal",
           xlabel="Output coordinate 1", ylabel="Output coordinate 2")
    ax.grid(alpha=.15)
    fig.savefig(output / "ledger_geometry_cancellation.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Generated three exact ledger state comparisons and the signed cancellation path")


if __name__ == "__main__":
    main()
