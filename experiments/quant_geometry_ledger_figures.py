"""Draw a forward Q-to-FP repair account and its outstanding balances.

Run from the repository root with python -m experiments.quant_geometry_ledger_figures.
"""
import json
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


def pair(ax, source, destination, lengths, bounds, title, *, curved=False):
    """Draw actual forward displacements, retaining each input's identity."""
    shape(ax, source, lengths, color="#64748b", dashed=True)
    shape(ax, destination, lengths)
    for j in range(len(source)-2, len(source)):
        color = ["#d97706", "#9333ea"][j-len(source)+2]
        ax.scatter(*source[j].numpy(), facecolors="none", edgecolors=color,
                   marker=["o", "s"][j-len(source)+2], s=65, zorder=5)
        if torch.linalg.vector_norm(destination[j]-source[j]) > 1e-10:
            ax.annotate("", xy=destination[j].numpy(), xytext=source[j].numpy(),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 2.5,
                                    "connectionstyle": f"arc3,rad={.2 if j == len(source)-2 else -.2}" if curved else "arc3,rad=0"})
    probes(ax, destination)
    axes_style(ax, bounds, title)


def main():
    output = Path("docs/figures/quant_geometry")
    output.mkdir(parents=True, exist_ok=True)
    stories = build_geometry_stories()
    fixture = {}
    for story in stories:
        L = story.ledger
        h, transported, repaired, y = L.quant_post, L.transported_post, L.recoverable_post, L.fp_post
        c_R, c_NL, c_U = transported-h, repaired-transported, y-repaired
        torch.testing.assert_close(c_R, -L.uncomp)
        torch.testing.assert_close(c_NL, -L.nonlinear)
        torch.testing.assert_close(c_U, -L.range)
        torch.testing.assert_close(c_R+c_NL+c_U, y-h)
        torch.testing.assert_close(c_NL+c_U, y-transported)
        fixture[story.key] = {"W": story.decomposition.weight.tolist(),
                              "Wq": story.decomposition.quantized_weight.tolist(),
                              "R": story.decomposition.R.tolist(), "U": story.decomposition.U.tolist(),
                              "inputs_AB": story.points[-2:].tolist(),
                              "path_AB": {"h": h[-2:].tolist(), "hR": transported[-2:].tolist(),
                                          "phi_zqR": repaired[-2:].tolist(), "y": y[-2:].tolist()},
                              "corrections_AB": {"total": (y-h)[-2:].tolist(), "R": c_R[-2:].tolist(),
                                                 "after_R": (y-transported)[-2:].tolist(),
                                                 "NL": c_NL[-2:].tolist(), "U": c_U[-2:].tolist()}}
    # Independent coordinates verify the signs promised in the captions.
    torch.testing.assert_close(-stories[1].ledger.nonlinear[-2:],
                               torch.tensor([[0., .235], [0., .07]], dtype=torch.float64))
    torch.testing.assert_close(-stories[2].ledger.range[-2:],
                               torch.tensor([[-.2, -.12], [.2, .2]], dtype=torch.float64))

    L = stories[0].ledger
    values = [L.quant_post, L.transported_post, L.fp_post]
    points = torch.cat(values).numpy()
    low, high = points.min(axis=0), points.max(axis=0)
    center, radius = (low+high)/2, max(high-low)*.625
    bounds = tuple((c-radius, c+radius) for c in center)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    fig.suptitle("Correction needed → apply R → inspect the outstanding balance", fontsize=16)
    for ax, source, destination, title in zip(axes,
        [L.quant_post, L.quant_post, L.transported_post],
        [L.fp_post, L.transported_post, L.fp_post],
        [r"Need $c_{\rm total}=y-h$", r"Apply $c_R=hR-h$", r"Still owed: $y-hR$"]):
        pair(ax, source, destination, stories[0].path_lengths, bounds, title)
    fig.supxlabel("Arrows go from the dashed starting shape to the blue destination. Filled A/B mark where each motion arrives.", fontsize=10)
    fig.savefig(output / "ledger_geometry_uncomp.png", dpi=150, facecolor="white")
    plt.close(fig)

    for key, story, title, source_label, destination_label, vector_label in [
        ("nonlinear", stories[1], "Exact linear repair: activation leaves a balance after R",
         r"$hR$: R applied after ReLU", r"$\phi(z_qR)=y$: repair before ReLU", r"$c_{\rm NL}=\phi(z_qR)-hR$"),
        ("range", stories[2], "Incomplete linear repair: finish the account at FP",
         r"$\phi(z_qR)$: activated repaired route", r"$y=\phi(z_qR+xU)$: FP target", r"$c_U=y-\phi(z_qR)$")]:
        L = story.ledger
        source, destination = (L.transported_post, L.recoverable_post) if key == "nonlinear" else (L.recoverable_post, L.fp_post)
        points = torch.cat((source, destination)).numpy()
        low, high = points.min(axis=0), points.max(axis=0)
        center, radius = (low+high)/2, max(high-low)*.625
        bounds = tuple((c-radius, c+radius) for c in center)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
        fig.suptitle(title, fontsize=16)
        for ax, state, label in zip(axes[:2], (source, destination), (source_label, destination_label)):
            shape(ax, state, story.path_lengths)
            probes(ax, state)
            axes_style(ax, bounds, label)
        pair(axes[2], source, destination, story.path_lengths, bounds,
             vector_label + "\nstarting state → next state", curved=key == "nonlinear")
        fig.savefig(output / f"ledger_geometry_{key}.png", dpi=150, facecolor="white")
        plt.close(fig)

    W = torch.tensor([[1., -.35], [.28, .22]], dtype=torch.float64)
    decomposition = MapDecomposition.from_weights(W, WeightQuantizer(2)(W).quantized)
    L = ErrorLedger.compute(torch.tensor([[1., 2.]], dtype=W.dtype), decomposition)
    states = [t[0].numpy() for t in (L.quant_post, L.transported_post, L.recoverable_post, L.fp_post)]
    expected = torch.tensor([[0., -.35], [0., .35], [.56, .09]], dtype=W.dtype)
    torch.testing.assert_close(torch.stack([-L.uncomp[0], -L.nonlinear[0], -L.range[0]]), expected)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.set_title(r"Q → FP: $h\rightarrow hR\rightarrow\phi(z_qR)\rightarrow y$", fontsize=14)
    labels = [r"$h=\phi(z_qR)=(1,0)$", r"$hR=(1,-0.35)$", "", r"$y=(1.56,0.09)$"]
    for point, label, offset in zip(states, labels, [(-105, 14), (-115, -22), (0, 0), (8, 10)]):
        ax.scatter(*point, color="#0f172a", s=55, zorder=5)
        if label:
            ax.annotate(label, point, xytext=offset, textcoords="offset points", fontsize=12)
    for i, (name, color, curvature) in enumerate([
        (r"$c_R=(0,-0.35)$", "#2563eb", -.2),
        (r"$c_{\rm NL}=(0,+0.35)$", "#d97706", -.2),
        (r"$c_U=(+0.56,+0.09)$", "#9333ea", 0)]):
        ax.annotate("", xy=states[i+1], xytext=states[i],
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 2.5,
                                "connectionstyle": f"arc3,rad={curvature}"})
        ax.plot([], [], color=color, lw=2.5, label=name)
    ax.legend(loc="lower right", fontsize=11)
    ax.set(xlim=(.65, 1.85), ylim=(-.55, .23), aspect="equal",
           xlabel="Output coordinate 1", ylabel="Output coordinate 2")
    ax.grid(alpha=.15)
    fig.savefig(output / "ledger_geometry_cancellation.png", dpi=150, facecolor="white")
    plt.close(fig)
    fixture["cancellation"] = {"input": [1., 2.], "path": [s.tolist() for s in states],
                                "corrections_R_NL_U": expected.tolist(), "total": (.56, .09)}

    # One layer receiving an upstream-displaced input: actual vectors, not a symbolic flowchart.
    W, Wq = stories[0].decomposition.weight, stories[0].decomposition.quantized_weight
    a = torch.tensor([1., .5], dtype=W.dtype)
    a_fp = torch.tensor([.7, .8], dtype=W.dtype)
    hq, g, target = torch.relu(a@Wq), torch.relu(a@W), torch.relu(a_fp@W)
    local, incoming, total = g-hq, target-g, target-hq
    torch.testing.assert_close(local+incoming, total)
    torch.testing.assert_close(local, torch.tensor([-.175, -.45], dtype=W.dtype))
    torch.testing.assert_close(incoming, torch.tensor([-.405, .045], dtype=W.dtype))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    fig.suptitle("One layer: what arrived, what changed here, what the oracle resets", fontsize=16)
    ax = axes[0]
    for value, label, color, offset in [(a, r"Arriving $a$", "#2563eb", (8, -22)),
                                       (a_fp, r"Reference $a_{\rm fp}$", "#9333ea", (8, 12))]:
        ax.scatter(*value.numpy(), color=color, s=65)
        ax.annotate(label, value.numpy(), xytext=offset, textcoords="offset points", fontsize=12)
    ax.annotate("", xy=a_fp.numpy(), xytext=a.numpy(), arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#9333ea"})
    ax.set(title="The arriving input already differs", xlim=(.5, 1.2), ylim=(.3, 1.), aspect="equal",
           xlabel="Input coordinate 1", ylabel="Input coordinate 2")
    ax.grid(alpha=.15)
    ax = axes[1]
    for value, label, offset in [(hq, r"Q output $h_q$", (-115, 12)),
                                 (g, r"Same-input FP $g$", (10, -20)),
                                 (target, r"Reference FP $y_{\rm ref}$", (-5, 15))]:
        ax.scatter(*value.numpy(), color="#0f172a", s=65, zorder=5)
        ax.annotate(label, value.numpy(), xytext=offset, textcoords="offset points", fontsize=11)
    for source, destination, label, color, style, curvature in [
        (hq, g, r"Local: $g-h_q$", "#2563eb", "-", 0),
        (g, target, r"Incoming: $y_{\rm ref}-g$", "#9333ea", "-", 0),
        (hq, target, r"Oracle reset: $y_{\rm ref}-h_q$", "#0f172a", "--", .3)]:
        ax.annotate("", xy=destination.numpy(), xytext=source.numpy(),
                    arrowprops={"arrowstyle": "->", "lw": 2.3, "color": color, "linestyle": style,
                                "connectionstyle": f"arc3,rad={curvature}"})
        ax.plot([], [], color=color, linestyle=style, lw=2.3, label=label)
    ax.set(title="Reconcile the output balance", xlim=(.25, 1.15), ylim=(.82, 1.72), aspect="equal",
           xlabel="Output coordinate 1", ylabel="Output coordinate 2")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=.15)
    fig.savefig(output / "repair_network_balance.png", dpi=150, facecolor="white")
    plt.close(fig)
    fixture["network"] = {"W": W.tolist(), "Wq": Wq.tolist(), "a": a.tolist(), "a_fp": a_fp.tolist(),
                          "h_q": hq.tolist(), "g": g.tolist(), "y_ref": target.tolist(),
                          "local": local.tolist(), "incoming": incoming.tolist(), "oracle_reset": total.tolist()}
    (output / "repair_ledger_figure_fixture.json").write_text(json.dumps(fixture, indent=2)+"\n")
    print("Generated four forward repair figures, a network/oracle balance, and exact fixture data")


if __name__ == "__main__":
    main()
