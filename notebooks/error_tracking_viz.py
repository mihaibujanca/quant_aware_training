# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: quant-aware-training
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import torch
import torch.nn as nn

@dataclass
class ErrorDecomposition:
    """Track different sources of error at each layer"""
    layer_idx: int

    # Linear (compensatable) errors
    weight_quant_error: np.ndarray      # (W_q - W) @ x
    activation_round_error: np.ndarray   # Rounding without saturation

    # Nonlinear (non-compensatable) errors
    saturation_error: np.ndarray         # From clipping to [qmin, qmax]
    relu_interaction_error: np.ndarray   # ReLU(x + e) - ReLU(x) - linear_approx

    # Metadata
    n_saturated: int                     # Count of saturated activations
    n_relu_flipped: int                  # Count where ReLU decision flipped

    @property
    def total_linear_error(self):
        return self.weight_quant_error + self.activation_round_error

    @property
    def total_nonlinear_error(self):
        return self.saturation_error + self.relu_interaction_error

    @property
    def total_error(self):
        return self.total_linear_error + self.total_nonlinear_error

    @property
    def linear_fraction(self):
        """What fraction of error is linearly compensatable?"""
        lin_norm = np.linalg.norm(self.total_linear_error)
        total_norm = np.linalg.norm(self.total_error)
        return lin_norm / total_norm if total_norm > 0 else 1.0

    def summary(self):
        return {
            'layer': self.layer_idx,
            'weight_quant': np.linalg.norm(self.weight_quant_error),
            'round': np.linalg.norm(self.activation_round_error),
            'saturation': np.linalg.norm(self.saturation_error),
            'relu_flip': np.linalg.norm(self.relu_interaction_error),
            'n_saturated': self.n_saturated,
            'n_relu_flipped': self.n_relu_flipped,
            'linear_fraction': self.linear_fraction,
        }


class ErrorTracker:
    """
    Run parallel FP32 and quantized forward passes,
    decomposing error at each layer into linear vs nonlinear components.
    """

    def __init__(self, bits=8, symmetric=True):
        self.bits = bits
        self.symmetric = symmetric
        if symmetric:
            self.qmin = -(2**(bits-1))
            self.qmax = 2**(bits-1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**bits - 1

    def quantize(self, x, scale=None):
        """Quantize and return (quantized, scale, was_saturated_mask)"""
        x_np = x.numpy() if isinstance(x, torch.Tensor) else x

        if scale is None:
            if self.symmetric:
                scale = np.abs(x_np).max() / self.qmax if np.abs(x_np).max() > 0 else 1.0
            else:
                scale = (x_np.max() - x_np.min()) / (self.qmax - self.qmin)
                scale = scale if scale > 0 else 1.0

        # Quantize
        x_scaled = x_np / scale
        x_rounded = np.round(x_scaled)
        x_clipped = np.clip(x_rounded, self.qmin, self.qmax)

        # Track what was saturated
        was_saturated = (x_rounded != x_clipped)

        # Dequantize
        x_deq = x_clipped * scale

        # Decompose error
        round_error = (x_rounded * scale - x_np)  # Error from rounding alone
        clip_error = (x_clipped - x_rounded) * scale  # Additional error from clipping

        return x_deq, scale, was_saturated, round_error, clip_error

    def analyze_relu_error(self, x_fp, x_quant, error_before_relu):
        """
        Analyze how ReLU interacts with quantization error.

        Three cases:
        1. Both positive: ReLU is identity, error passes through linearly
        2. Both negative: ReLU zeros both, error is zeroed (good!)
        3. Different signs: Nonlinear interaction (bad!)
        """
        x_fp_np = x_fp.numpy() if isinstance(x_fp, torch.Tensor) else x_fp
        x_q_np = x_quant.numpy() if isinstance(x_quant, torch.Tensor) else x_quant

        # After ReLU
        y_fp = np.maximum(x_fp_np, 0)
        y_q = np.maximum(x_q_np, 0)

        # Actual error after ReLU
        actual_error = y_q - y_fp

        # What the error WOULD be if ReLU were linear (identity)
        linear_error = error_before_relu

        # The nonlinear part is the difference
        nonlinear_error = actual_error - linear_error

        # Count flips: where ReLU decision differs
        fp_positive = x_fp_np > 0
        q_positive = x_q_np > 0
        n_flipped = np.sum(fp_positive != q_positive)

        return actual_error, linear_error, nonlinear_error, n_flipped

    def forward_with_decomposition(self, model, x) -> List[ErrorDecomposition]:
        """
        Run forward pass tracking all error components.
        Returns list of ErrorDecomposition for each layer.
        """
        decompositions = []

        x_fp = x.clone().float()
        x_q = x.clone().float()

        for layer_idx, layer in enumerate(model.layers[:-1]):
            W = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy() if layer.bias is not None else np.zeros(W.shape[0])

            # === FP32 path ===
            x_fp_pre_relu = (x_fp.numpy() @ W.T + b)

            # === Quantized path with decomposition ===

            # Step 1: Quantize weights
            W_q, w_scale, _, w_round_err, w_clip_err = self.quantize(W)
            b_q, b_scale, _, _, _ = self.quantize(b)

            # Weight quantization error contribution
            # (W_q - W) @ x
            weight_quant_error = x_q.numpy() @ (W_q - W).T

            # Step 2: Compute pre-activation with quantized weights
            x_q_pre_act = x_q.numpy() @ W_q.T + b_q

            # Step 3: Quantize activations
            x_q_post_quant, act_scale, was_saturated, round_err, clip_err = self.quantize(x_q_pre_act)

            # Current accumulated error (before ReLU)
            error_pre_relu = x_q_post_quant - x_fp_pre_relu

            # Step 4: Apply ReLU and analyze interaction
            x_fp_post_relu = np.maximum(x_fp_pre_relu, 0)
            x_q_post_relu = np.maximum(x_q_post_quant, 0)

            actual_error, linear_error_component, nonlinear_relu_error, n_flipped = \
                self.analyze_relu_error(x_fp_pre_relu, x_q_post_quant, error_pre_relu)

            # Build decomposition
            decomp = ErrorDecomposition(
                layer_idx=layer_idx,
                weight_quant_error=weight_quant_error.flatten(),
                activation_round_error=round_err.flatten(),
                saturation_error=clip_err.flatten(),
                relu_interaction_error=nonlinear_relu_error.flatten(),
                n_saturated=int(was_saturated.sum()),
                n_relu_flipped=n_flipped,
            )
            decompositions.append(decomp)

            # Update for next layer
            x_fp = torch.tensor(x_fp_post_relu, dtype=torch.float32)
            x_q = torch.tensor(x_q_post_relu, dtype=torch.float32)

        return decompositions


def visualize_error_decomposition(decompositions: List[ErrorDecomposition]):
    """
    Create visualization showing linear vs nonlinear error at each layer.
    """
    n_layers = len(decompositions)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Data for plotting
    layers = [d.layer_idx for d in decompositions]
    weight_err = [np.linalg.norm(d.weight_quant_error) for d in decompositions]
    round_err = [np.linalg.norm(d.activation_round_error) for d in decompositions]
    sat_err = [np.linalg.norm(d.saturation_error) for d in decompositions]
    relu_err = [np.linalg.norm(d.relu_interaction_error) for d in decompositions]

    # Plot 1: Stacked bar of error components
    ax = axes[0, 0]
    width = 0.6
    ax.bar(layers, weight_err, width, label='Weight quant (linear)', color='steelblue')
    ax.bar(layers, round_err, width, bottom=weight_err, label='Rounding (linear)', color='lightblue')
    bottom_nonlin = np.array(weight_err) + np.array(round_err)
    ax.bar(layers, sat_err, width, bottom=bottom_nonlin, label='Saturation (nonlinear)', color='coral')
    ax.bar(layers, relu_err, width, bottom=bottom_nonlin + np.array(sat_err),
           label='ReLU flip (nonlinear)', color='red')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Error norm')
    ax.set_title('Error Decomposition by Layer')
    ax.legend()
    ax.set_xticks(layers)

    # Plot 2: Linear fraction over layers
    ax = axes[0, 1]
    linear_frac = [d.linear_fraction for d in decompositions]
    colors = ['green' if f > 0.8 else 'orange' if f > 0.5 else 'red' for f in linear_frac]
    ax.bar(layers, linear_frac, color=colors, edgecolor='black')
    ax.axhline(0.5, color='gray', linestyle='--', label='50% threshold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Fraction of error that is linear')
    ax.set_title('Compensatable Error Fraction\n(green=mostly fixable, red=mostly unfixable)')
    ax.set_ylim(0, 1)
    ax.set_xticks(layers)

    # Plot 3: Saturation events
    ax = axes[1, 0]
    n_sat = [d.n_saturated for d in decompositions]
    n_flip = [d.n_relu_flipped for d in decompositions]
    x_pos = np.arange(n_layers)
    width = 0.35
    ax.bar(x_pos - width/2, n_sat, width, label='Saturated activations', color='coral')
    ax.bar(x_pos + width/2, n_flip, width, label='ReLU flips', color='purple')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Count')
    ax.set_title('Nonlinear Events per Layer')
    ax.legend()
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layers)

    # Plot 4: Cumulative error growth
    ax = axes[1, 1]
    total_err = [np.linalg.norm(d.total_error) for d in decompositions]
    linear_err = [np.linalg.norm(d.total_linear_error) for d in decompositions]
    nonlin_err = [np.linalg.norm(d.total_nonlinear_error) for d in decompositions]

    ax.plot(layers, total_err, 'k-o', linewidth=2, markersize=8, label='Total error')
    ax.plot(layers, linear_err, 'b--s', linewidth=2, markersize=6, label='Linear component')
    ax.plot(layers, nonlin_err, 'r--^', linewidth=2, markersize=6, label='Nonlinear component')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Error norm')
    ax.set_title('Error Growth Through Network')
    ax.legend()
    ax.set_xticks(layers)

    plt.tight_layout()
    return fig


def visualize_relu_flip_regions(W, b, bits=8, n_grid=200):
    """
    Visualize where in input space ReLU decisions flip due to quantization.
    These are the "danger zones" where nonlinear error is introduced.
    """
    W_q, _ = quantize_matrix_np(W, bits)
    b_q, _ = quantize_matrix_np(b.reshape(1, -1), bits)
    b_q = b_q.flatten()

    # Create grid
    x = np.linspace(-2, 2, n_grid)
    xx, yy = np.meshgrid(x, x)
    points = np.stack([xx.flatten(), yy.flatten()], axis=1)

    # Pre-ReLU activations
    pre_relu_fp = points @ W.T + b
    pre_relu_q = points @ W_q.T + b_q

    # For each output dimension, find flip regions
    n_outputs = W.shape[0]

    fig, axes = plt.subplots(1, n_outputs + 1, figsize=(5 * (n_outputs + 1), 5))

    total_flips = np.zeros((n_grid, n_grid))

    for i in range(n_outputs):
        fp_positive = pre_relu_fp[:, i] > 0
        q_positive = pre_relu_q[:, i] > 0
        flipped = (fp_positive != q_positive).reshape(n_grid, n_grid)
        total_flips += flipped

        ax = axes[i]
        ax.contourf(xx, yy, flipped.astype(float), levels=[0.5, 1.5], colors=['red'], alpha=0.5)
        ax.contour(xx, yy, pre_relu_fp[:, i].reshape(n_grid, n_grid),
                  levels=[0], colors='blue', linewidths=2, linestyles='--')
        ax.contour(xx, yy, pre_relu_q[:, i].reshape(n_grid, n_grid),
                  levels=[0], colors='red', linewidths=2)
        ax.set_title(f'Output dim {i}\nBlue=FP32 boundary, Red=INT{bits}')
        ax.set_aspect('equal')

    # Total flip regions
    ax = axes[-1]
    im = ax.contourf(xx, yy, total_flips, levels=np.arange(n_outputs + 2) - 0.5, cmap='Reds')
    plt.colorbar(im, ax=ax, label='# dimensions flipped')
    ax.set_title('Total ReLU flip regions\n(darker = more danger)')
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def quantize_matrix_np(W, bits=8):
    """Numpy version of matrix quantization"""
    qmax = 2**(bits-1) - 1
    scale = np.abs(W).max() / qmax if np.abs(W).max() > 0 else 1.0
    W_q = np.round(W / scale) * scale
    return W_q, scale


def analyze_correctability(model, x_samples, bits=8):
    """
    For a batch of samples, compute:
    1. Total error at output
    2. Best possible linear correction (least squares)
    3. Residual error after correction (the uncorrectable part)

    This tells you: if you had an oracle linear correction layer,
    how much error would remain?
    """
    tracker = ErrorTracker(bits=bits)

    # Collect outputs
    y_fp_list = []
    y_q_list = []

    for x in x_samples:
        x = x.unsqueeze(0) if x.dim() == 1 else x

        # FP32 forward
        y_fp = x.clone()
        for layer in model.layers:
            y_fp = layer(y_fp)
            if layer != model.layers[-1]:
                y_fp = torch.relu(y_fp)

        # Quantized forward (simplified)
        y_q = x.clone()
        for layer in model.layers:
            W_q, _ = quantize_matrix_np(layer.weight.detach().numpy(), bits)
            b_q, _ = quantize_matrix_np(layer.bias.detach().numpy(), bits)
            y_q_np = y_q.numpy() @ W_q.T + b_q
            y_q_np, _, _, _, _ = tracker.quantize(y_q_np)
            if layer != model.layers[-1]:
                y_q_np = np.maximum(y_q_np, 0)
            y_q = torch.tensor(y_q_np, dtype=torch.float32)

        y_fp_list.append(y_fp.detach().numpy().flatten())
        y_q_list.append(y_q.numpy().flatten())

    Y_fp = np.array(y_fp_list)
    Y_q = np.array(y_q_list)

    # Error before correction
    error_before = Y_q - Y_fp
    error_norm_before = np.linalg.norm(error_before, axis=1).mean()

    # Best linear correction: find A, b such that Y_q @ A + b ≈ Y_fp
    # Least squares: A = (Y_q^T Y_q)^{-1} Y_q^T Y_fp
    Y_q_aug = np.hstack([Y_q, np.ones((Y_q.shape[0], 1))])  # Add bias term

    try:
        correction_params, residuals, rank, s = np.linalg.lstsq(Y_q_aug, Y_fp, rcond=None)
        Y_corrected = Y_q_aug @ correction_params
        error_after = Y_corrected - Y_fp
        error_norm_after = np.linalg.norm(error_after, axis=1).mean()
    except:
        error_norm_after = error_norm_before
        correction_params = None

    correctability = 1 - (error_norm_after / error_norm_before) if error_norm_before > 0 else 1.0

    return {
        'error_before_correction': error_norm_before,
        'error_after_correction': error_norm_after,
        'correctability': correctability,  # 1.0 = fully correctable, 0.0 = not at all
        'correction_params': correction_params
    }


# ============ Demo ============
class SimpleNet(nn.Module):
    def __init__(self, dims=[2, 8, 8, 4, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model
    model = SimpleNet(dims=[2, 8, 8, 4, 1])
    for layer in model.layers:
        nn.init.xavier_uniform_(layer.weight, gain=2.0)

    print("=" * 60)
    print("ERROR DECOMPOSITION ANALYSIS")
    print("=" * 60)

    # Single sample decomposition
    tracker = ErrorTracker(bits=8)
    x = torch.randn(1, 2)
    decompositions = tracker.forward_with_decomposition(model, x)

    print("\nPer-layer error breakdown:")
    print("-" * 80)
    print(f"{'Layer':<8} {'Weight Q':<12} {'Round':<12} {'Saturate':<12} {'ReLU flip':<12} {'Linear %':<10}")
    print("-" * 80)
    for d in decompositions:
        s = d.summary()
        print(f"{s['layer']:<8} {s['weight_quant']:<12.4f} {s['round']:<12.4f} "
              f"{s['saturation']:<12.4f} {s['relu_flip']:<12.4f} {s['linear_fraction']*100:<10.1f}%")

    # Visualize
    fig = visualize_error_decomposition(decompositions)
    plt.savefig('plots/error_decomposition.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("RELU FLIP REGIONS")
    print("=" * 60)

    # Visualize ReLU flip regions for first layer
    W = model.layers[0].weight.detach().numpy()
    b = model.layers[0].bias.detach().numpy()
    fig = visualize_relu_flip_regions(W, b, bits=4)  # Use 4-bit to exaggerate
    plt.savefig('plots/relu_flip_regions.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("CORRECTABILITY ANALYSIS")
    print("=" * 60)

    # Test correctability at different bit widths
    x_samples = torch.randn(500, 2)

    print("\nHow much error can be fixed by a linear correction layer?")
    print("-" * 50)
    for bits in [8, 6, 4, 2]:
        result = analyze_correctability(model, x_samples, bits=bits)
        print(f"{bits}-bit: {result['correctability']*100:.1f}% correctable "
              f"(error {result['error_before_correction']:.4f} → {result['error_after_correction']:.4f})")

# %%
