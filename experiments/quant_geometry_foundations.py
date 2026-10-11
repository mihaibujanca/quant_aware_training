"""Reproduce exact-ledger figures and MLP trace examples without training.

Run from repository root:
    .venv/bin/python -m experiments.quant_geometry_foundations
Outputs: docs/figures/quant_geometry/ (PNG figures + run manifest).
"""
import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from aleph.qgeom import CorrectionBridge, ErrorLedger, InstrumentedMLP, MapDecomposition, WeightQuantizer
from aleph.qgeom.ledger_plots import (
    plot_activation_trace, plot_component_2d, plot_ledger_accounting, plot_map_decomposition,
    plot_rank_policy,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/quant_geometry"))
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    seed = 24
    torch.manual_seed(seed)
    quantizer = WeightQuantizer(bits=2)
    # A bounded 2-bit quantizer removes the weak input row and a negative shear.
    # This single fixture exposes all three mechanisms and their cancellation.
    W = torch.tensor([[1., -.35], [.28, .22]], dtype=torch.float64)
    q = quantizer(W)
    decomp = MapDecomposition.from_weights(W, q.quantized)
    axis = torch.linspace(-2, 2, 41, dtype=W.dtype)
    x = torch.cartesian_prod(axis, axis)
    ledger = ErrorLedger.compute(x, decomp)
    figures = []

    def save(figure, filename):
        figure.savefig(output / filename, dpi=150, facecolor="white")
        plt.close(figure)
        figures.append(filename)

    save(plot_map_decomposition(decomp), "map_decomposition.png")
    for component in ledger.components():
        save(plot_component_2d(x, ledger, component), f"component_{component}.png")
    save(plot_ledger_accounting(ledger), "ledger_accounting.png")

    # Deliberately prescribed matrices, NOT an output of the 2-bit quantizer:
    # show that numerical truncation is different from a true null direction.
    audit_weight = torch.eye(2, dtype=torch.float64)
    audit_quantized = torch.diag(torch.tensor([1., 1e-8], dtype=torch.float64))
    truncated = MapDecomposition.from_weights(audit_weight, audit_quantized, rtol=1e-6)
    retained = MapDecomposition.from_weights(audit_weight, audit_quantized, rtol=1e-10)
    save(plot_rank_policy(truncated, retained), "rank_policy_audit.png")

    # Untrained model: tests the apparatus, not quantization performance.
    model = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 2))
    runner = InstrumentedMLP(model, quantizer=quantizer)
    mlp_x = x.float()
    ordinary = runner.run(mlp_x, quantized_layers=[0])
    immediate = runner.run(mlp_x, quantized_layers=[0], bridges=[CorrectionBridge(0, 0)])
    all_quant = runner.run(mlp_x)
    all_immediate = runner.run(mlp_x, bridges=[CorrectionBridge(i, i) for i in range(3)])
    save(plot_activation_trace(ordinary, immediate,
                              title="One quantized layer: raw damage, then exact restoration"), "activation_trace.png")
    save(plot_activation_trace(all_quant, all_immediate,
                              title="All layers quantized: immediate oracle isolates each source"), "oracle_trace.png")
    tolerance = dict(atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(immediate.output, immediate.fp.post_acts[-1], **tolerance)
    for layer in all_immediate.layers:
        torch.testing.assert_close(layer.input, layer.fp_input, **tolerance)
        torch.testing.assert_close(layer.corrected_post, layer.fp_post, **tolerance)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    manifest = {
        "revision": revision, "working_tree_dirty": dirty, "seed": seed,
        "torch_version": torch.__version__, "matplotlib_version": matplotlib.__version__,
        "dataset": "41x41 Cartesian grid, [-2,2]^2; no fit/evaluation split (algebra demo)",
        "weight_convention": "row vectors, [input, output]",
        "quantization": {"bits": 2, "bounds": [-1, 1], "scheme": q.metadata.scheme,
                         "rounding": q.metadata.rounding, "single_layer_scale": float(q.metadata.scale)},
        "single_layer": {"dtype": "float64", "weight": W.tolist(), "bias": [0, 0],
                         "activation": "ReLU", "diagnostics": decomp.diagnostics(),
                         "max_telescoping_residual": float(ledger.reconstruction_error.abs().max()),
                         "mean_component_energy": {k: float(v.square().sum(dim=1).mean()) for k, v in ledger.components().items()},
                         "gram": ledger.gram().tolist(),
                         "mean_total_energy": float(ledger.total.square().sum(dim=1).mean())},
        "mlp": {"dimensions": [2, 5, 3, 2], "dtype": "float32", "trained": False,
                "biases": [b.tolist() for b in runner.biases],
                "weights_row_convention": [w.tolist() for w in runner.weights],
                "scales": [float(q.metadata.scale) for q in runner.quantized_weights],
                "single_source": {"quantized_layers": [0], "bridges": [[0, 0]]},
                "all_immediate": {"quantized_layers": [0, 1, 2], "bridges": [[i, i] for i in range(3)]},
                "single_source_output_max_residual": float((immediate.output-immediate.fp.post_acts[-1]).abs().max()),
                "all_immediate_output_max_residual": float((all_immediate.output-all_immediate.fp.post_acts[-1]).abs().max()),
                "oracle_check_tolerances": tolerance},
        "rank_policy_audit": {
            "matrix_source": "prescribed diagnostic fixture, not the 2-bit quantizer",
            "W": audit_weight.tolist(), "Wq": audit_quantized.tolist(),
            "dtype": "float64", "truncated_rtol": truncated.rtol, "retained_rtol": retained.rtol,
            "truncated_R": truncated.R.tolist(), "truncated_U": truncated.U.tolist(),
            "retained_R": retained.R.tolist(), "retained_U": retained.U.tolist(),
        },
        "figures": figures,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output_dir": str(output), "figures": len(figures),
                      "ledger_residual": manifest["single_layer"]["max_telescoping_residual"],
                      "oracle_residual": manifest["mlp"]["all_immediate_output_max_residual"]}, indent=2))


if __name__ == "__main__":
    main()
