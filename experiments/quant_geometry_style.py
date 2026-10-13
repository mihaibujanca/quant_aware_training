"""One visual convention for every quantization-geometry figure.

Color says which state a shape or point is in; marker shape says which input it is.
  input (before any map)      near-black, solid
  FP target                   orange, dashed, hollow markers
  quantized, uncorrected      gray, solid
  after a correction          blue, solid
Ledger terms are arrows: c_R blue, c_NL green, c_U purple, total black dashed.
Where C and D appear together, they need different FP targets: C's is orange, D's purple,
and an arrow belonging to only one of them takes that point's color.
"""
import numpy as np
from matplotlib.lines import Line2D  # noqa: F401  (re-exported for figure keys)

INPUT, FP, Q, REPAIRED = "#111827", "#d97706", "#8a8f98", "#2563eb"
STATE = {"input": (INPUT, "-"), "fp": (FP, "--"), "q": (Q, "-"), "repaired": (REPAIRED, "-")}
STATE_LABEL = {"input": "input", "fp": "FP target", "q": "quantized, uncorrected", "repaired": "after correction"}
TERM = {"R": REPAIRED, "NL": "#059669", "U": "#9333ea", "total": "#0f172a"}
MARKER = {"A": "o", "B": "s", "C": "o", "D": "s"}
IDENTITY = {"C": FP, "D": "#9333ea"}


def shape(ax, points, lengths, kind, *, lw=2.5, thin=.7, alpha=.75, linestyle=None, color=None):
    """Draw grid lines thin and the circle (last path) thick, in the state's style."""
    state_color, style = STATE[kind]
    style, color = linestyle or style, color or state_color
    points = np.asarray(points)
    offset = 0
    for j, length in enumerate(lengths):
        path = points[offset:offset + length]
        ax.plot(path[:, 0], path[:, 1], style, color=color, alpha=alpha,
                lw=lw if j == len(lengths) - 1 else thin)
        offset += length


def named(ax, points, names, kind, *, offsets=None, s=70, by_identity=False):
    """Mark named inputs: marker shape = which input, color = state. Coincident points share a label.

    by_identity colors FP targets of C and D differently, for figures that compare them."""
    color = STATE[kind][0]
    points = np.asarray(points)
    offsets = offsets or [(8, 8), (8, -18)]
    coincident = len(points) == 2 and np.linalg.norm(points[0] - points[1]) < 1e-10
    for point, name, offset in zip(points, names, offsets):
        if by_identity:
            color = IDENTITY[name]
        filled = kind != "fp"
        ax.scatter(*point, marker=MARKER[name], s=s, zorder=6, linewidths=2,
                   facecolors=color if filled else "white", edgecolors=color)
        if not coincident:
            ax.annotate(name, point, xytext=offset, textcoords="offset points", fontsize=12, color="#0f172a")
    if coincident:
        ax.annotate(f"{names[0]} = {names[1]}", points[0], xytext=offsets[1], textcoords="offset points",
                    fontsize=12, color="#0f172a")


def arrow(ax, start, end, term, *, curvature=0., lw=2.4, color=None):
    style = "--" if term == "total" else "-"
    ax.annotate("", xy=np.asarray(end), xytext=np.asarray(start),
                arrowprops={"arrowstyle": "->", "color": color or TERM[term], "lw": lw, "linestyle": style,
                            "connectionstyle": f"arc3,rad={curvature}"})


def state_handle(kind, label=None):
    color, style = STATE[kind]
    return Line2D([], [], color=color, linestyle=style, lw=2.5, marker="o", markersize=7, markeredgewidth=2,
                  markerfacecolor="white" if kind == "fp" else color, markeredgecolor=color,
                  label=label or STATE_LABEL[kind])


def term_handle(term, label):
    return Line2D([], [], color=TERM[term], linestyle="--" if term == "total" else "-", lw=2.4, label=label)


def legend(fig, handles, ncol=None):
    """One figure-level key below the panels, so every figure states its encoding."""
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol or len(handles), fontsize=11,
               frameon=False)


def axes_style(ax, bounds, title, *, xlabel="Output coordinate 1", ylabel="Output coordinate 2"):
    ax.axhline(0, color="#cbd5e1", lw=.7)
    ax.axvline(0, color="#cbd5e1", lw=.7)
    ax.set(xlim=bounds[0], ylim=bounds[1], aspect="equal", title=title, xlabel=xlabel, ylabel=ylabel)


def bounds_for(*states, pad=.6):
    points = np.concatenate([np.asarray(s) for s in states])
    low, high = points.min(axis=0), points.max(axis=0)
    center, radius = (low + high) / 2, max(high - low) * pad
    return tuple((c - radius, c + radius) for c in center)
