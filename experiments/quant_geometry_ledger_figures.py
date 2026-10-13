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

from aleph.qgeom.geometry_story import build_geometry_stories, build_typical_ledger_story
from experiments import quant_geometry_style as style


def main():
    output = Path("docs/figures/quant_geometry")
    output.mkdir(parents=True, exist_ok=True)
    stories = build_geometry_stories()
    fixture = {}
    typical = build_typical_ledger_story()
    for story in stories[:2]:
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
    L = typical.ledger
    path = torch.stack((L.quant_post, L.transported_post, L.recoverable_post, L.fp_post))
    c_R, c_NL, c_U = path[1]-path[0], path[2]-path[1], path[3]-path[2]
    torch.testing.assert_close(c_R, torch.tensor([[-.1, .14]]*2, dtype=torch.float64))
    torch.testing.assert_close(c_NL, torch.tensor([[-.025, -.14]]*2, dtype=torch.float64))
    torch.testing.assert_close(c_U, torch.tensor([[.15, 0.], [-.15, 0.]], dtype=torch.float64))
    torch.testing.assert_close(c_R+c_NL+c_U, -L.total)
    fixture["typical"] = {"W": typical.decomposition.weight.tolist(),
                          "Wq": typical.decomposition.quantized_weight.tolist(),
                          "R": typical.decomposition.R.tolist(), "U": typical.decomposition.U.tolist(),
                          "inputs_CD": typical.pair.tolist(),
                          "path_CD": {"h": path[0].tolist(), "hR": path[1].tolist(),
                                      "phi_zqR": path[2].tolist(), "y": path[3].tolist()},
                          "corrections_CD": {"total": (path[3]-path[0]).tolist(), "R": c_R.tolist(),
                                             "NL": c_NL.tolist(), "U": c_U.tolist()}}

    def save(fig, name):
        fig.savefig(output / name, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)

    def n(t):
        return t.numpy()

    # Figure 3: correction needed, the step R makes, correction still needed (square layer W1).
    L, lengths = stories[0].ledger, stories[0].path_lengths
    h, hR, y = n(L.quant_post), n(L.transported_post), n(L.fp_post)
    bounds = style.bounds_for(h, hR, y, pad=.625)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), constrained_layout=True)
    fig.suptitle("Correction needed → apply $R$ → correction still needed", fontsize=16)
    panels = [(h, "q", None, r"1. Needed: $c_{\rm total}=y-h$"),
              (hR, "repaired", h, r"2. Apply $R$: $c_R=hR-h$"),
              (hR, "repaired", None, r"3. Still needed: $y-hR$")]
    for ax, (state, kind, before, title) in zip(axes, panels):
        style.shape(ax, y, lengths, "fp")
        if before is not None:
            style.shape(ax, before, lengths, "q", alpha=.35)
        style.shape(ax, state, lengths, kind)
        style.named(ax, state[-2:], "AB", kind)
        style.named(ax, y[-2:], "AB", "fp", offsets=[(-18, 6), (-18, -14)])
        for j in (-2, -1):
            if np.linalg.norm(y[j]-state[j]) > 1e-9:
                style.arrow(ax, state[j], y[j], "total", curvature=.25)
        style.axes_style(ax, bounds, title)
    style.legend(fig, [style.state_handle("fp"), style.state_handle("q"), style.state_handle("repaired"),
                       style.term_handle("total", "correction needed (named points)")])
    save(fig, "ledger_geometry_uncomp.png")

    # Figure 4: for W1 the correction still needed is c_NL: repair before ReLU reaches y.
    L, lengths = stories[1].ledger, stories[1].path_lengths
    hR, before_relu, y = n(L.transported_post), n(L.recoverable_post), n(L.fp_post)
    bounds = style.bounds_for(hR, y, pad=.625)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), constrained_layout=True)
    fig.suptitle("The linear repair is exact, but ReLU acts first", fontsize=16)
    for ax, state, title in [(axes[0], hR, r"1. Correct after ReLU: $hR$"),
                             (axes[1], before_relu, r"2. Correct before ReLU: $\phi(z_qR)=y$")]:
        style.shape(ax, y, lengths, "fp")
        style.shape(ax, state, lengths, "repaired")
        style.named(ax, state[-2:], "AB", "repaired")
        style.axes_style(ax, bounds, title)
    style.shape(axes[2], y, lengths, "fp")
    style.shape(axes[2], hR, lengths, "repaired")
    style.named(axes[2], hR[-2:], "AB", "repaired")
    style.named(axes[2], y[-2:], "AB", "fp", offsets=[(-18, 6), (-18, -14)])
    for j in (-2, -1):
        style.arrow(axes[2], hR[j], y[j], "NL", curvature=.25)
    style.axes_style(axes[2], bounds, r"3. $c_{\rm NL}=\phi(z_qR)-hR$")
    style.legend(fig, [style.state_handle("fp"), style.state_handle("repaired"),
                       style.term_handle("NL", r"$c_{\rm NL}$")])
    save(fig, "ledger_geometry_nonlinear.png")

    # Figure 6: W2, two input planes through C and D share phi(z_qR); their FP targets differ.
    sheet_C, sheet_D = typical.sheets
    lengths = typical.path_lengths
    shared, yC, yD = n(sheet_C.recoverable_post), n(sheet_C.fp_post), n(sheet_D.fp_post)
    torch.testing.assert_close(sheet_C.recoverable_post, sheet_D.recoverable_post)
    bounds = style.bounds_for(shared, yC, yD)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4), constrained_layout=True)
    fig.suptitle("Same corrected output, different FP targets", fontsize=16)
    style.shape(axes[0], shared, lengths, "repaired")
    style.named(axes[0], shared[-2:], "CD", "repaired")
    style.axes_style(axes[0], bounds, r"1. Corrected before ReLU: $\phi(z_qR)$" + "\nthe same for both planes")
    style.shape(axes[1], yC, lengths, "fp", color=style.IDENTITY["C"])
    style.shape(axes[1], yD, lengths, "fp", color=style.IDENTITY["D"])
    style.named(axes[1], np.stack((yC[-2], yD[-1])), "CD", "fp", by_identity=True)
    style.axes_style(axes[1], bounds, "2. FP targets $y$\nplane through C (orange), through D (purple)")
    zoom = ((.25, .7), (-.15, .3))
    style.shape(axes[2], yC, lengths, "fp", color=style.IDENTITY["C"])
    style.shape(axes[2], yD, lengths, "fp", color=style.IDENTITY["D"])
    style.shape(axes[2], shared, lengths, "repaired")
    style.named(axes[2], shared[-2:], "CD", "repaired", offsets=[(8, 8), (-8, -20)])
    style.named(axes[2], np.stack((yC[-2], yD[-1])), "CD", "fp", offsets=[(4, -20), (-14, -20)], by_identity=True)
    style.arrow(axes[2], shared[-2], yC[-2], "U", curvature=-.5, color=style.IDENTITY["C"])
    style.arrow(axes[2], shared[-1], yD[-1], "U", curvature=.5, color=style.IDENTITY["D"])
    style.axes_style(axes[2], zoom, r"3. $c_U=y-\phi(z_qR)$ (zoomed)" + "\nopposite for C and D")
    style.legend(fig, [style.state_handle("repaired"), style.state_handle("fp", "FP target for C"),
                       style.Line2D([], [], color=style.IDENTITY["D"], linestyle="--", lw=2.5, marker="s",
                                    markerfacecolor="white", markeredgewidth=2, label="FP target for D"),
                       style.Line2D([], [], color=style.IDENTITY["C"], lw=2.4, label="$c_U$ for C"),
                       style.Line2D([], [], color=style.IDENTITY["D"], lw=2.4, label="$c_U$ for D")])
    save(fig, "ledger_geometry_range.png")

    # Figure 7: the full ledger for C and D.
    h, hR, rep_, ys = n(path[0, 0]), n(path[1, 0]), n(path[2, 0]), n(path[3])
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.set_title(r"$C$ and $D$: $h\rightarrow hR\rightarrow\phi(z_qR)\rightarrow y$", fontsize=14)
    for point, kind, label, offset in [(h, "q", r"$h=(0.6,0)$", (-25, -42)), (hR, "repaired", r"$hR=(0.5,0.14)$", (8, 6)),
                                       (rep_, "repaired", r"$\phi(z_qR)=(0.475,0)$", (-55, -24)),
                                       (ys[0], "fp", r"$y_C=(0.625,0)$", (8, -24)), (ys[1], "fpD", r"$y_D=(0.325,0)$", (-30, -24))]:
        color = style.IDENTITY["D"] if kind == "fpD" else style.STATE[kind][0]
        kind = "fp" if kind == "fpD" else kind
        ax.scatter(*point, s=60, zorder=6, linewidths=2, edgecolors=color,
                   facecolors="white" if kind == "fp" else color)
        ax.annotate(label, point, xytext=offset, textcoords="offset points", fontsize=11)
    style.arrow(ax, h, hR, "R", curvature=.2)
    style.arrow(ax, hR, rep_, "NL", curvature=.2)
    style.arrow(ax, rep_, ys[0], "U", curvature=-.5, color=style.IDENTITY["C"])
    style.arrow(ax, rep_, ys[1], "U", curvature=.5, color=style.IDENTITY["D"])
    style.arrow(ax, h, ys[0], "total", curvature=.6, lw=1.6, color=style.IDENTITY["C"])
    style.arrow(ax, h, ys[1], "total", curvature=.6, lw=1.6, color=style.IDENTITY["D"])
    ax.set(xlim=(.25, .7), ylim=(-.12, .3), aspect="equal", xlabel="Output coordinate 1", ylabel="Output coordinate 2")
    ax.grid(alpha=.15)
    from matplotlib.lines import Line2D
    style.legend(fig, [style.state_handle("q"), style.state_handle("repaired"),
                       style.state_handle("fp", "FP target for C"),
                       Line2D([], [], color=style.IDENTITY["D"], lw=0, marker="o", markerfacecolor="white",
                              markeredgewidth=2, label="FP target for D"),
                       style.term_handle("R", r"$c_R$ (shared)"), style.term_handle("NL", r"$c_{\rm NL}$ (shared)"),
                       Line2D([], [], color=style.IDENTITY["C"], lw=2.4, label=r"$c_U$ for C"),
                       Line2D([], [], color=style.IDENTITY["D"], lw=2.4, label=r"$c_U$ for D"),
                       Line2D([], [], color=style.IDENTITY["C"], lw=1.6, linestyle="--", label=r"$c_{\rm total}$ for C"),
                       Line2D([], [], color=style.IDENTITY["D"], lw=1.6, linestyle="--", label=r"$c_{\rm total}$ for D")],
                 ncol=4)
    save(fig, "ledger_geometry_cancellation.png")

    # Figure 8: one W2 layer receiving an input that already differs from FP; the FP input is D.
    W, Wq = typical.decomposition.weight, typical.decomposition.quantized_weight
    a_fp = typical.pair[1]
    a = torch.tensor([.4, .4, -.4], dtype=W.dtype)
    hq, g, target = torch.relu(a@Wq), torch.relu(a@W), torch.relu(a_fp@W)
    local, incoming, total = g-hq, target-g, target-hq
    torch.testing.assert_close(local+incoming, total)
    torch.testing.assert_close(local, torch.tensor([-.2, .18], dtype=W.dtype))
    torch.testing.assert_close(incoming, torch.tensor([-.275, -.18], dtype=W.dtype))
    fig = plt.figure(figsize=(11, 5.8), constrained_layout=True)
    fig.suptitle("One layer in a network: error received, error added, and the oracle", fontsize=16)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for value, kind, label in [(a, "q", r"received $a$"), (a_fp, "fp", r"FP input $a_{\rm fp}=D$")]:
        color = style.STATE[kind][0]
        ax.scatter(*n(value), s=65, depthshade=False, edgecolors=color, linewidths=2,
                   facecolors="white" if kind == "fp" else color)
        ax.text(*(n(value) + .05), label, fontsize=11)
    ax.plot(*n(torch.stack((a, a_fp)).T), color="#0f172a", lw=1.5, linestyle=":")
    ax.set(title="Input space", xlabel="Input 1", ylabel="Input 2", zlabel="Input 3",
           xlim=(0, .6), ylim=(.2, .6), zlim=(-.8, -.2))
    ax.view_init(elev=20, azim=35)
    ax = fig.add_subplot(1, 2, 2)
    for value, kind, label, offset in [(hq, "q", r"$h_q=\phi(aW_q)$", (-30, -22)),
                                       (g, "repaired", r"$g=\phi(aW)$", (10, -4)),
                                       (target, "fp", r"$y_{\rm ref}=\phi(a_{\rm fp}W)$", (-40, -22))]:
        color = style.STATE[kind][0] if kind != "repaired" else "#0f172a"
        ax.scatter(*n(value), s=65, zorder=6, linewidths=2, edgecolors=color,
                   facecolors="white" if kind == "fp" else color)
        ax.annotate(label, n(value), xytext=offset, textcoords="offset points", fontsize=11)
    for source, destination, color, ls, curvature in [(hq, g, "#0f172a", "-", 0), (g, target, "#0f172a", ":", 0),
                                                       (hq, target, "#0f172a", "--", .3)]:
        ax.annotate("", xy=n(destination), xytext=n(source),
                    arrowprops={"arrowstyle": "->", "lw": 2.3, "color": color, "linestyle": ls,
                                "connectionstyle": f"arc3,rad={curvature}"})
    ax.set(title="Output space", xlim=(.2, .9), ylim=(-.2, .3), aspect="equal",
           xlabel="Output coordinate 1", ylabel="Output coordinate 2")
    ax.grid(alpha=.15)
    style.legend(fig, [style.state_handle("q"), style.state_handle("fp"),
                       style.Line2D([], [], marker="o", color="#0f172a", lw=0, label="FP layer on the received input, $g$"),
                       style.Line2D([], [], color="#0f172a", lw=2.3, label=r"local $c_{\rm local}=g-h_q$"),
                       style.Line2D([], [], color="#0f172a", lw=2.3, linestyle=":", label=r"incoming $c_{\rm incoming}=y_{\rm ref}-g$"),
                       style.term_handle("total", r"oracle: $y_{\rm ref}-h_q$")], ncol=3)
    save(fig, "repair_network_balance.png")
    fixture["network"] = {"W": W.tolist(), "Wq": Wq.tolist(), "a": a.tolist(), "a_fp": a_fp.tolist(),
                          "h_q": hq.tolist(), "g": g.tolist(), "y_ref": target.tolist(),
                          "local": local.tolist(), "incoming": incoming.tolist(), "oracle_reset": total.tolist()}
    (output / "repair_ledger_figure_fixture.json").write_text(json.dumps(fixture, indent=2)+"\n")
    print("Generated forward repair figures, a network/oracle balance, and exact fixture data")


if __name__ == "__main__":
    main()
