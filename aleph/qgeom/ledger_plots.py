"""Explanatory plots for exact ledgers and inspectable MLP interventions.

Functions return Matplotlib figures; callers own saving/closing. Component
figures deliberately show signed vectors rather than misleading stacked
percentages. The 2D plot requires 2D inputs and outputs; summary plots support
arbitrary widths. This module does not select a Matplotlib backend.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from aleph.qgeom.instrumented_mlp import ActivationTrace
from aleph.qgeom.ledger import ErrorLedger
from aleph.qgeom.map_decomposition import MapDecomposition

COMPONENT_LABELS = {
    "uncomp": "Uncompensated recoverable deformation",
    "nonlinear": "Nonlinearity-induced error",
    "range": "Residual deformation error",
}
COMPONENT_SHORT_LABELS = {"uncomp": "uncomp", "nonlinear": "nonlinear", "range": "residual (U)"}
COLORS = {"uncomp": "#2563eb", "nonlinear": "#d97706", "range": "#9333ea"}


def _array(tensor):
    return tensor.detach().cpu().numpy()


def plot_component_2d(x: torch.Tensor, ledger: ErrorLedger, component: str):
    """Input footprint, representation-space vectors and one signed example."""
    if x.shape[1] != 2 or ledger.total.shape[1] != 2:
        raise ValueError("component illustration requires 2D input and output")
    starts = {"uncomp": ledger.quant_post,
              "nonlinear": ledger.transported_post,
              "range": ledger.recoverable_post}
    ends = {"uncomp": ledger.transported_post,
            "nonlinear": ledger.recoverable_post,
            "range": ledger.fp_post}
    labels = {"uncomp": ("h", "A(h)"), "nonlinear": ("A(h)", "ReLU(A(zq))"),
              "range": ("ReLU(A(zq))", "FP output")}
    vector = ledger.components()[component]
    values = _array(vector)
    inputs = _array(x)
    norms = np.linalg.norm(values, axis=1)
    sample = int(norms.argmax())
    start, end = _array(starts[component]), _array(ends[component])
    first, second = labels[component]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.suptitle(COMPONENT_LABELS[component], fontsize=17, fontweight="bold")
    points = axes[0].scatter(inputs[:, 0], inputs[:, 1], c=norms, cmap="viridis", s=14, rasterized=True)
    axes[0].scatter(*inputs[sample], facecolors="none", edgecolors="#ef4444", s=130, linewidths=2)
    axes[0].set(title="Where this mechanism appears", xlabel="Input coordinate 1", ylabel="Input coordinate 2")
    fig.colorbar(points, ax=axes[0], label="Component vector norm", shrink=.8)
    indices = np.linspace(0, len(inputs) - 1, min(100, len(inputs)), dtype=int)
    axes[1].scatter(end[indices, 0], end[indices, 1], s=15, color="#64748b", alpha=.55,
                    label=f"Arrow origin: {second}")
    axes[1].quiver(end[indices, 0], end[indices, 1], values[indices, 0], values[indices, 1],
                   color=COLORS[component], angles="xy", scale_units="xy", scale=1, alpha=.65, width=.005)
    endpoints = np.concatenate((start, end))
    lo, hi = endpoints.min(axis=0), endpoints.max(axis=0)
    padding = np.maximum((hi-lo)*.1, .15)
    axes[1].set(xlim=(lo[0]-padding[0], hi[0]+padding[0]), ylim=(lo[1]-padding[1], hi[1]+padding[1]),
                title=f"Signed vector = {first} − {second}", xlabel="Output coordinate 1", ylabel="Output coordinate 2")
    axes[1].legend(loc="best", fontsize=8)
    positions = np.arange(2)
    for offset, data, label, color in [(-.24, start[sample], first, "#334155"),
                                      (0, end[sample], second, "#94a3b8"),
                                      (.24, values[sample], "Difference (this component)", COLORS[component])]:
        axes[2].bar(positions + offset, data, width=.23, label=label, color=color)
    axes[2].axhline(0, color="#64748b", linewidth=.8)
    axes[2].set(xticks=positions, xticklabels=["Coordinate 1", "Coordinate 2"], ylabel="Signed value",
                title=f"Largest example: x = ({inputs[sample, 0]:.2f}, {inputs[sample, 1]:.2f})")
    axes[2].legend(loc="best", fontsize=8)
    for ax in axes[:2]:
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, color="#cbd5e1", linewidth=.6)
        ax.axvline(0, color="#cbd5e1", linewidth=.6)
    return fig


def plot_map_decomposition(decomposition: MapDecomposition):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    fig.suptitle("Map reconstruction: W = Wq R + U (row-vector convention)", fontsize=16)
    matrices = [(decomposition.weight, "Full-precision W"),
                (decomposition.quantized_weight, "Quantized Wq"),
                (decomposition.R, "Recoverable correction R"),
                (decomposition.U, "Residual deformation U"),
                (decomposition.weight - decomposition.quantized_weight @ decomposition.R - decomposition.U,
                 "Reconstruction residual")]
    for ax, (matrix, title) in zip(axes.flat, matrices):
        values = _array(matrix)
        limit = max(np.abs(values).max(), 1e-15)
        im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit)
        for (row, col), value in np.ndenumerate(values):
            ax.text(col, row, f"{value:.3g}", ha="center", va="center", fontsize=12,
                    color="white" if abs(value) > limit*.55 else "#0f172a")
        ax.set(title=title, xlabel="Column", ylabel="Row")
        ax.set_xticks(np.arange(values.shape[1]))
        ax.set_yticks(np.arange(values.shape[0]))
        fig.colorbar(im, ax=ax, shrink=.7)
    ax = axes[1, 2]
    for name, matrix in [("Wq", decomposition.quantized_weight), ("R", decomposition.R), ("U", decomposition.U)]:
        ax.plot(np.arange(min(matrix.shape)) + 1, _array(torch.linalg.svdvals(matrix)), "o-", label=name)
    ax.set(title=f"Singular spectra (rank Wq = {decomposition.rank})", xlabel="Singular index", ylabel="Singular value")
    ax.legend()
    ax.grid(alpha=.2)
    return fig


def plot_ledger_accounting(ledger: ErrorLedger):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    fig.suptitle("Exact per-sample accounting, including interaction", fontsize=16)
    for name, vector in ledger.components().items():
        axes[0].plot(_array(torch.linalg.vector_norm(vector, dim=1)), label=COMPONENT_SHORT_LABELS[name], color=COLORS[name], alpha=.8)
    axes[0].plot(_array(torch.linalg.vector_norm(ledger.total, dim=1)), color="#0f172a", label="total", linewidth=2)
    axes[0].set(title="Component norms need not sum", xlabel="Sample (grid order)", ylabel="Vector norm")
    axes[0].legend()
    gram = _array(ledger.gram())
    lim = max(np.abs(gram).max(), 1e-15)
    im = axes[1].imshow(gram, cmap="RdBu_r", vmin=-lim, vmax=lim)
    labels = [COMPONENT_SHORT_LABELS[name] for name in ledger.components()]
    axes[1].set(xticks=range(3), yticks=range(3), xticklabels=labels, yticklabels=labels,
                title="Gram: negative = cancellation")
    for (row, col), value in np.ndenumerate(gram):
        axes[1].text(col, row, f"{value:.4f}", ha="center", va="center",
                     color="white" if abs(value) > lim*.55 else "#0f172a")
    fig.colorbar(im, ax=axes[1], shrink=.8)
    residual = _array(ledger.reconstruction_error.abs().amax(dim=1))
    axes[2].plot(residual, color="#059669")
    axes[2].set(title=f"Telescoping residual: max {residual.max():.2e}", xlabel="Sample", ylabel="Max absolute coordinate residual")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[2].grid(alpha=.2)
    return fig


def plot_activation_trace(ordinary: ActivationTrace, corrected: ActivationTrace, *, title: str):
    """Show FP/raw/consumed states, creation vs propagation and oracle sizes."""
    if len(ordinary.layers) != len(corrected.layers):
        raise ValueError("traces must share architecture")
    count = len(ordinary.layers)
    depth = np.arange(count)
    norm = lambda tensor: float(torch.linalg.vector_norm(tensor, dim=1).mean())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(title, fontsize=17, fontweight="bold")
    axes[0, 0].plot(depth, [norm(l.total_error) for l in ordinary.layers], "o-", label="Ordinary Q")
    axes[0, 0].plot(depth, [norm(l.total_error) for l in corrected.layers], "o--", label="Intervention: raw before correction")
    axes[0, 0].plot(depth, [norm(l.corrected_post-l.fp_post) for l in corrected.layers], "o-", label="Intervention: consumed after correction")
    axes[0, 0].set(title="Damage stays visible before injection", xlabel="Linear layer index", ylabel="Mean error norm")
    axes[0, 0].legend(fontsize=9)
    sample = int(ordinary.layers[-1].total_error.square().sum(dim=1).argmax())
    width = max(l.fp_post.shape[1] for l in ordinary.layers)
    values, row_labels = [], []
    for i in range(count):
        for name, tensor in [("FP", ordinary.layers[i].fp_post),
                             ("ordinary Q", ordinary.layers[i].quant_post),
                             ("oracle consumed", corrected.layers[i].corrected_post)]:
            row = np.full(width, np.nan)
            row[:tensor.shape[1]] = _array(tensor[sample])
            values.append(row)
            row_labels.append(f"L{i} {name}")
    values = np.array(values)
    limit = max(np.nanmax(np.abs(values)), 1e-15)
    im = axes[0, 1].imshow(np.ma.masked_invalid(values), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axes[0, 1].set(title=f"Post-activation states, sample {sample}", yticks=range(len(row_labels)),
                   yticklabels=row_labels, xticks=range(width), xlabel="Neuron (blank = absent)")
    fig.colorbar(im, ax=axes[0, 1], label="Activation", shrink=.8)
    for offset, label, values, color in [(-.24, "Local creation", [norm(l.ledger.total) for l in ordinary.layers], "#2563eb"),
                                       (0, "Propagated", [norm(l.propagated_error) for l in ordinary.layers], "#d97706"),
                                       (.24, "Total", [norm(l.total_error) for l in ordinary.layers], "#334155")]:
        axes[1, 0].bar(depth+offset, values, width=.23, label=label, color=color)
    axes[1, 0].set(title="Creation vs fate: vectors add, norms need not", xlabel="Linear layer index", ylabel="Mean norm (ordinary Q)")
    axes[1, 0].legend(fontsize=9)
    for name, vectors in {**{k: [l.ledger.components()[k] for l in corrected.layers] for k in COLORS},
                          "oracle residual": [l.oracle_residual for l in corrected.layers]}.items():
        axes[1, 1].plot(depth, [norm(v) for v in vectors], "o-", label=COMPONENT_SHORT_LABELS.get(name, name),
                       color=COLORS.get(name, "#0f172a"))
    sites = [l.index for l in corrected.layers if l.correction_applied]
    for site in sites:
        axes[1, 1].axvline(site, color="#059669", alpha=.25, linewidth=4)
    axes[1, 1].set(title=f"Local ledger and oracle; green = sites {sites}", xlabel="Linear layer index", ylabel="Mean vector norm (intervention run)")
    axes[1, 1].legend(fontsize=9)
    for ax in [axes[0, 0], axes[1, 0], axes[1, 1]]:
        ax.set_xticks(depth)
        ax.grid(axis="y", alpha=.2)
    return fig


def plot_rank_policy(truncated: MapDecomposition, retained: MapDecomposition):
    """2D diagnostic distinguishing a discarded singular value from a null one."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.suptitle("A truncation residual need not mean lost information", fontsize=17, fontweight="bold")
    spectra = _array(truncated.singular_values)
    axes[0].semilogy([1, 2], spectra, "o-", color="#334155", label="Nonzero singular values of Wq")
    for decomp, color, label in [(truncated, "#d97706", "rtol = 1e-6 (rank 1)"),
                                 (retained, "#2563eb", "rtol = 1e-10 (rank 2)")]:
        axes[0].axhline(decomp.rtol * spectra.max(), color=color, linestyle="--", label=label)
    axes[0].set(title="Wq = diag(1, 1e-8) is invertible", xlabel="Singular index", ylabel="Singular value / cutoff", xticks=[1, 2])
    axes[0].legend(fontsize=8)
    norms = [float(torch.linalg.norm(d.U)) for d in (truncated, retained)]
    axes[1].bar([0, 1], norms, color=["#d97706", "#2563eb"])
    axes[1].set(title="W = I: remainder depends on cutoff", xticks=[0, 1], xticklabels=["Rank 1 policy", "Rank 2 policy"], ylabel="Frobenius norm of U")
    for i, value in enumerate(norms):
        axes[1].text(i, value + .03, f"{value:.3g}", ha="center")
    axes[1].set_ylim(0, 1.15)
    # The second input direction is weak, not absent, in the quantized map.
    x = torch.tensor([[0., 1.]], dtype=retained.weight.dtype, device=retained.weight.device)
    zq = x @ retained.quantized_weight
    values = [float((zq @ truncated.R)[0, 1]),
              float((zq @ truncated.R + x @ truncated.U)[0, 1]),
              float((zq @ retained.R)[0, 1])]
    axes[2].bar(range(3), values, color=["#d97706", "#059669", "#2563eb"])
    axes[2].set(title="Second output for x = (0, 1)", xticks=range(3),
                xticklabels=["Truncated R", "Truncated R\n+ oracle xU", "Full R\n(gain 1e8)"], ylabel="Recovered coordinate")
    for i, value in enumerate(values):
        axes[2].text(i, value + .03, f"{value:.3g}", ha="center")
    axes[2].set_ylim(0, 1.15)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    return fig
