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

# %% [markdown]
# ```
#
# ## The Key Relationships
#
# ### 1. **Volume ↔ Entropy (Linear Case)**
#
# For a uniform distribution over a region:
# ```
# H = log₂(Volume)
# ```
#
# For a Gaussian (what errors often converge to):
# ```
# H = (n/2) log₂(2πe) + (1/2) log₂(det(Σ))
#     └─────────────┘   └─────────────────┘
#      dimension term      volume term
# ```
#
# The determinant of the covariance IS the volume of the uncertainty ellipsoid.
#
# ### 2. **Linear Transforms Scale Both Equally**
# ```
# H_after = H_before + log|det(W)|
# Vol_after = Vol_before × |det(W)|
#
# So: ΔH = log(Vol_after / Vol_before)
# ```
#
# This is why spectral analysis (singular values of W) tells you about error growth — it's also telling you about entropy growth.
#
# ### 3. **Nonlinear Ops Break the Relationship**
#
# ReLU and clipping both:
# - Reduce volume (geometrically)
# - Reduce entropy (information-theoretically)
#
# But this isn't "free precision" — it's **information destruction**. Multiple inputs map to the same output.
#
# ### 4. **The Saturation Catastrophe in Entropy Terms**
#
# When you clip values:
# ```
# Before: H_before = some finite value
# After:  H_after = H_before - log₂(n_collapsed)
#
# where n_collapsed = number of distinct values that got mapped to the clamp limit

# %%
import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

class GeometricEntropyTracker:
    """
    Track both geometric (volume-based) and information-theoretic
    (entropy-based) measures of quantization error.
    """

    def __init__(self, bits=8):
        self.bits = bits
        self.delta = 1.0 / (2 ** bits)  # Quantization step size

    def hypercube_volume(self, n_dims):
        """Volume of n-dimensional quantization error hypercube"""
        return self.delta ** n_dims

    def hypercube_entropy(self, n_dims):
        """
        Entropy of uniform distribution over hypercube.
        H = log(volume) for continuous uniform.
        """
        return n_dims * np.log2(self.delta)  # Will be negative

    def parallelepiped_volume(self, W, base_delta=None):
        """
        After linear transform W, hypercube becomes parallelepiped.
        Volume scales by |det(W)|.
        """
        delta = base_delta if base_delta else self.delta
        n_dims = W.shape[0]

        # For non-square W, use product of singular values
        if W.shape[0] != W.shape[1]:
            singular_values = np.linalg.svd(W, compute_uv=False)
            scale = np.prod(singular_values[:min(W.shape)])
        else:
            scale = np.abs(np.linalg.det(W))

        return scale * (delta ** n_dims)

    def entropy_after_linear(self, W, base_entropy=None):
        """
        Entropy change under linear transform.
        H_after = H_before + log|det(W)|
        """
        if base_entropy is None:
            base_entropy = self.hypercube_entropy(W.shape[1])

        if W.shape[0] != W.shape[1]:
            singular_values = np.linalg.svd(W, compute_uv=False)
            log_det = np.sum(np.log2(singular_values[:min(W.shape)] + 1e-10))
        else:
            sign, log_det = np.linalg.slogdet(W)
            log_det = log_det / np.log(2)  # Convert to bits

        return base_entropy + log_det

    def estimate_entropy_from_samples(self, samples, n_bins=50):
        """
        Empirically estimate entropy from samples using histogram.
        """
        if samples.ndim == 1:
            hist, _ = np.histogram(samples, bins=n_bins, density=True)
            hist = hist[hist > 0]  # Remove zeros
            bin_width = (samples.max() - samples.min()) / n_bins
            # Differential entropy approximation
            return -np.sum(hist * np.log2(hist + 1e-10)) * bin_width
        else:
            # For multivariate, use covariance-based estimate (assumes Gaussian-ish)
            cov = np.cov(samples.T)
            sign, log_det = np.linalg.slogdet(cov)
            n = samples.shape[1]
            # Entropy of multivariate Gaussian: 0.5 * log((2πe)^n |Σ|)
            return 0.5 * (n * np.log2(2 * np.pi * np.e) + log_det / np.log(2))

    def information_loss_from_clipping(self, samples, clip_min, clip_max):
        """
        Estimate information lost when values are clipped.

        Information loss occurs when multiple input values map to the same output.
        """
        n_total = len(samples)
        n_clipped_high = np.sum(samples > clip_max)
        n_clipped_low = np.sum(samples < clip_min)
        n_clipped = n_clipped_high + n_clipped_low

        if n_clipped == 0:
            return 0.0

        # Before clipping: each value is distinguishable
        # After clipping: all values above clip_max are indistinguishable

        # Rough estimate: bits lost ≈ log2(n_clipped) for each clipping region
        # because n_clipped values collapse to 1

        bits_lost = 0
        if n_clipped_high > 1:
            bits_lost += np.log2(n_clipped_high)
        if n_clipped_low > 1:
            bits_lost += np.log2(n_clipped_low)

        return bits_lost

    def relu_entropy_reduction(self, samples):
        """
        Estimate entropy change from ReLU.

        ReLU maps all negative values to 0, collapsing a region of input space.
        """
        n_total = len(samples)
        n_negative = np.sum(samples < 0)
        n_positive = n_total - n_negative

        if n_negative == 0:
            return 0.0  # No change

        # Entropy before: H(samples)
        H_before = self.estimate_entropy_from_samples(samples)

        # Entropy after: H(relu(samples))
        samples_after = np.maximum(samples, 0)

        # The zeros are all indistinguishable, so we have:
        # - A point mass at 0 with probability n_negative/n_total
        # - A continuous distribution for positive values

        p_zero = n_negative / n_total
        p_positive = n_positive / n_total

        if n_positive > 1:
            positive_samples = samples[samples > 0]
            H_positive_part = self.estimate_entropy_from_samples(positive_samples)
        else:
            H_positive_part = 0

        # Mixture entropy (approximation)
        # H ≈ p_pos * H_positive + H(p_zero, p_pos)
        if p_zero > 0 and p_positive > 0:
            binary_entropy = -p_zero * np.log2(p_zero) - p_positive * np.log2(p_positive)
        else:
            binary_entropy = 0

        H_after = p_positive * H_positive_part + binary_entropy

        return H_before - H_after  # Entropy reduction


def visualize_entropy_geometry_connection():
    """
    Visualize how volume and entropy relate through transformations.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Initial error distribution (uniform in square)
    n_samples = 5000
    errors = np.random.uniform(-0.5, 0.5, size=(n_samples, 2))

    tracker = GeometricEntropyTracker(bits=8)

    # Stage 1: Initial hypercube
    ax = axes[0, 0]
    ax.scatter(errors[:, 0], errors[:, 1], alpha=0.3, s=1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    vol = 1.0  # Unit square scaled by delta
    H = tracker.estimate_entropy_from_samples(errors)
    ax.set_title(f'Initial Error Hypercube\nVolume ∝ 1.0, Entropy ≈ {H:.2f} bits')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    # Stage 2: After linear transform (rotation + scale)
    W = np.array([[1.5, 0.5], [-0.3, 1.2]])
    errors_transformed = errors @ W.T

    ax = axes[0, 1]
    ax.scatter(errors_transformed[:, 0], errors_transformed[:, 1], alpha=0.3, s=1)
    ax.set_aspect('equal')
    vol_after_W = np.abs(np.linalg.det(W))
    H_after_W = tracker.estimate_entropy_from_samples(errors_transformed)
    ax.set_title(f'After Linear W\nVolume × {vol_after_W:.2f}, Entropy ≈ {H_after_W:.2f} bits')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    # Stage 3: After ReLU (shift then clip)
    shift = np.array([0.3, 0.2])
    errors_shifted = errors_transformed + shift
    errors_relu = np.maximum(errors_shifted, 0)

    ax = axes[0, 2]
    ax.scatter(errors_relu[:, 0], errors_relu[:, 1], alpha=0.3, s=1)
    ax.set_aspect('equal')
    n_clipped = np.sum((errors_shifted < 0).any(axis=1))
    H_after_relu = tracker.estimate_entropy_from_samples(errors_relu)
    ax.set_title(f'After ReLU\n{n_clipped} points clipped, Entropy ≈ {H_after_relu:.2f} bits')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    # Bottom row: 1D marginals showing the distributions
    ax = axes[1, 0]
    ax.hist(errors[:, 0], bins=50, density=True, alpha=0.7)
    ax.set_title('Initial (dim 0)')
    ax.set_xlabel('Error')

    ax = axes[1, 1]
    ax.hist(errors_transformed[:, 0], bins=50, density=True, alpha=0.7)
    ax.set_title('After W (dim 0)')
    ax.set_xlabel('Error')

    ax = axes[1, 2]
    ax.hist(errors_relu[:, 0], bins=50, density=True, alpha=0.7)
    # Note the spike at 0
    ax.set_title('After ReLU (dim 0)\nNote spike at 0 = lost information')
    ax.set_xlabel('Error')

    plt.tight_layout()
    return fig


def analyze_layer_by_layer_entropy(model, x_samples, bits=8):
    """
    Track entropy of the error distribution through each layer.
    Compare geometric prediction vs empirical measurement.
    """
    tracker = GeometricEntropyTracker(bits=bits)

    results = []

    # Get all layer outputs for FP32
    fp_activations = []
    q_activations = []

    for x in x_samples:
        x_fp = x.unsqueeze(0).float()
        x_q = x.unsqueeze(0).float()

        fp_acts = [x_fp.numpy().flatten()]
        q_acts = [x_q.numpy().flatten()]

        for layer in model.layers[:-1]:
            # FP path
            x_fp = layer(x_fp)
            fp_acts.append(x_fp.detach().numpy().flatten())
            x_fp = torch.relu(x_fp)
            fp_acts.append(x_fp.detach().numpy().flatten())

            # Quantized path
            W_q = quantize_simple(layer.weight.detach().numpy(), bits)
            b_q = quantize_simple(layer.bias.detach().numpy(), bits)
            x_q_np = x_q.numpy() @ W_q.T + b_q
            x_q_np = quantize_simple(x_q_np, bits)
            q_acts.append(x_q_np.flatten())
            x_q_np = np.maximum(x_q_np, 0)
            q_acts.append(x_q_np.flatten())
            x_q = torch.tensor(x_q_np, dtype=torch.float32)

        fp_activations.append(fp_acts)
        q_activations.append(q_acts)

    # Compute errors at each stage
    n_stages = len(fp_activations[0])

    for stage in range(n_stages):
        fp_vals = np.array([fa[stage] for fa in fp_activations])
        q_vals = np.array([qa[stage] for qa in q_activations])
        errors = q_vals - fp_vals

        # Empirical entropy of error
        H_empirical = tracker.estimate_entropy_from_samples(errors)

        # Volume estimate (using covariance determinant as proxy)
        if errors.shape[1] > 1:
            cov = np.cov(errors.T)
            sign, logdet = np.linalg.slogdet(cov)
            vol_proxy = np.exp(logdet / 2)  # sqrt(det(cov)) ~ "radius"
        else:
            vol_proxy = np.std(errors)

        results.append({
            'stage': stage,
            'entropy': H_empirical,
            'volume_proxy': vol_proxy,
            'error_norm_mean': np.linalg.norm(errors, axis=1).mean(),
            'n_dims': errors.shape[1]
        })

    return results


def quantize_simple(x, bits):
    """Simple symmetric quantization"""
    qmax = 2**(bits-1) - 1
    scale = np.abs(x).max() / qmax if np.abs(x).max() > 0 else 1.0
    return np.round(x / scale) * scale


def visualize_entropy_vs_geometry(layer_results):
    """Plot entropy and geometric measures together"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    stages = [r['stage'] for r in layer_results]
    entropies = [r['entropy'] for r in layer_results]
    volumes = [r['volume_proxy'] for r in layer_results]
    error_norms = [r['error_norm_mean'] for r in layer_results]

    ax = axes[0]
    ax.plot(stages, entropies, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Stage (layer × 2 for pre/post ReLU)')
    ax.set_ylabel('Entropy (bits)')
    ax.set_title('Error Entropy Through Network')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(stages, volumes, 's-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Stage')
    ax.set_ylabel('Volume proxy (sqrt det cov)')
    ax.set_title('Error "Volume" Through Network')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    # Entropy vs log(volume) should be roughly linear
    ax.scatter(np.log(np.array(volumes) + 1e-10), entropies, s=50)
    ax.set_xlabel('log(Volume proxy)')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy vs log(Volume)\nShould be roughly linear')

    # Fit line
    valid = np.array(volumes) > 1e-10
    if valid.sum() > 2:
        z = np.polyfit(np.log(np.array(volumes)[valid]), np.array(entropies)[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(np.log(min(np.array(volumes)[valid])),
                            np.log(max(np.array(volumes)[valid])), 100)
        ax.plot(x_line, p(x_line), 'r--', label=f'slope={z[0]:.2f}')
        ax.legend()

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# Demo
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, dims=[2, 4, 4, 2]):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("ENTROPY-GEOMETRY CONNECTION")
    print("=" * 60)

    # Visualize basic connection
    fig = visualize_entropy_geometry_connection()
    plt.savefig('plots/entropy_geometry_basic.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Track through network
    model = SimpleNet(dims=[2, 4, 4, 2])
    for layer in model.layers:
        nn.init.xavier_uniform_(layer.weight, gain=1.5)

    x_samples = torch.randn(500, 2)
    results = analyze_layer_by_layer_entropy(model, x_samples, bits=8)

    print("\nEntropy and volume through layers:")
    print("-" * 60)
    for r in results:
        print(f"Stage {r['stage']}: H={r['entropy']:.2f} bits, "
              f"vol={r['volume_proxy']:.4f}, err_norm={r['error_norm_mean']:.4f}")

    fig = visualize_entropy_vs_geometry(results)
    plt.savefig('plots/entropy_vs_geometry.png', dpi=150, bbox_inches='tight')
    plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class ErrorShapeTracker:
    """
    Track the geometry of the error region through linear layers.

    Start with a hypercube (from initial quantization),
    transform through layers, observe how shape evolves.
    """

    def __init__(self, dims, bits=8):
        self.dims = dims
        self.bits = bits
        self.delta = 1.0 / (2 ** bits)  # Quantization step

    def initial_error_hypercube(self, n_samples=1000):
        """
        Initial error is uniform in [-delta/2, delta/2]^d
        Represent by sampling vertices or interior points.
        """
        # For low dims, can use actual vertices
        if self.dims <= 10:
            # Hypercube has 2^d vertices
            from itertools import product
            vertices = np.array(list(product([-1, 1], repeat=self.dims))) * self.delta / 2
            return vertices
        else:
            # Sample uniformly from hypercube
            return np.random.uniform(-self.delta/2, self.delta/2, size=(n_samples, self.dims))

    def transform_linear(self, error_points, W):
        """
        Linear transformation: error' = W @ error
        """
        return error_points @ W.T

    def add_quantization_error(self, error_points, new_delta=None):
        """
        After quantizing activations, we add another hypercube of error.
        This is a Minkowski sum.

        For point clouds: sample from the sum.
        """
        if new_delta is None:
            new_delta = self.delta

        n_points = len(error_points)
        new_dims = error_points.shape[1]

        # Approximate Minkowski sum by adding random offset from new hypercube
        offsets = np.random.uniform(-new_delta/2, new_delta/2, size=(n_points, new_dims))
        return error_points + offsets

    def apply_relu_to_error(self, error_points, activation_center):
        """
        ReLU effect on error region.

        If activation + error < 0, gets clipped.
        This depends on where the "true" activation is.

        activation_center: the FP32 activation value (or typical value)
        error_points: the error region around it
        """
        # Points in activation space
        activation_points = activation_center + error_points

        # After ReLU
        activation_post_relu = np.maximum(activation_points, 0)

        # New center
        center_post_relu = np.maximum(activation_center, 0)

        # Error after ReLU
        error_post_relu = activation_post_relu - center_post_relu

        return error_post_relu, center_post_relu

    def apply_saturation(self, error_points, activation_center, qmin, qmax):
        """
        Clipping effect on error region.
        """
        activation_points = activation_center + error_points
        activation_clipped = np.clip(activation_points, qmin, qmax)
        center_clipped = np.clip(activation_center, qmin, qmax)
        error_clipped = activation_clipped - center_clipped

        return error_clipped, center_clipped

    def measure_shape(self, error_points):
        """
        Compute shape statistics:
        - Volume (via covariance determinant)
        - Principal axes (via SVD)
        - Bounding box
        - Convexity measure
        """
        stats = {}

        # Covariance
        cov = np.cov(error_points.T)

        # Volume proxy
        sign, logdet = np.linalg.slogdet(cov)
        stats['log_volume'] = logdet / 2  # sqrt(det) for "radius"

        # Principal axes (singular values of centered points)
        centered = error_points - error_points.mean(axis=0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        stats['singular_values'] = S
        stats['condition_number'] = S.max() / (S.min() + 1e-10)
        stats['principal_directions'] = Vt

        # Bounding box
        stats['bbox_min'] = error_points.min(axis=0)
        stats['bbox_max'] = error_points.max(axis=0)
        stats['bbox_volume'] = np.prod(stats['bbox_max'] - stats['bbox_min'])

        # Convex hull volume (only for low dims)
        if self.dims <= 8 and len(error_points) > self.dims + 1:
            try:
                hull = ConvexHull(error_points)
                stats['convex_hull_volume'] = hull.volume
            except:
                stats['convex_hull_volume'] = None

        return stats


def run_linear_only_experiment(dims=2, n_layers=4, bits=8):
    """
    Track error shape through linear layers only (no nonlinearities).
    """
    tracker = ErrorShapeTracker(dims=dims, bits=bits)

    # Initialize error shape
    error_points = tracker.initial_error_hypercube(n_samples=2000)

    history = [{
        'layer': 'input',
        'points': error_points.copy(),
        'stats': tracker.measure_shape(error_points)
    }]

    print(f"Layer input: {len(error_points)} points, dims={error_points.shape[1]}")

    for i in range(n_layers):
        # Random weight matrix (with some structure)
        W = np.random.randn(dims, dims) * 0.5
        W = W + np.eye(dims) * 0.5  # Add some identity to prevent collapse

        # Quantize W
        W_quant = np.round(W * (2**bits)) / (2**bits)

        # The error in W also contributes
        W_error = W - W_quant

        # Transform error region
        error_points = tracker.transform_linear(error_points, W_quant)

        # Add new quantization error (Minkowski sum)
        error_points = tracker.add_quantization_error(error_points)

        stats = tracker.measure_shape(error_points)
        history.append({
            'layer': f'linear_{i}',
            'points': error_points.copy(),
            'stats': stats,
            'W': W_quant,
            'W_spectral_norm': np.linalg.norm(W_quant, ord=2)
        })

        print(f"Layer {i}: spectral_norm={np.linalg.norm(W_quant, ord=2):.3f}, "
              f"condition={stats['condition_number']:.2f}, "
              f"log_vol={stats['log_volume']:.3f}")

    return history


def run_with_nonlinearities_experiment(dims=2, n_layers=4, bits=8):
    """
    Track error shape with ReLU and saturation.
    """
    tracker = ErrorShapeTracker(dims=dims, bits=bits)
    qmin, qmax = -1.0, 1.0  # Activation range

    # Initialize
    error_points = tracker.initial_error_hypercube(n_samples=2000)
    activation_center = np.random.uniform(0.2, 0.8, size=dims)  # Start positive

    history = [{
        'layer': 'input',
        'points': error_points.copy(),
        'center': activation_center.copy(),
        'stats': tracker.measure_shape(error_points)
    }]

    for i in range(n_layers):
        # Weight matrix
        W = np.random.randn(dims, dims) * 0.5 + np.eye(dims) * 0.3
        W_quant = np.round(W * (2**bits)) / (2**bits)

        # Linear transform
        error_points = tracker.transform_linear(error_points, W_quant)
        activation_center = W_quant @ activation_center

        # Add quantization error
        error_points = tracker.add_quantization_error(error_points)

        stats_pre_relu = tracker.measure_shape(error_points)
        n_points_pre = len(error_points)

        # Apply ReLU
        error_points, activation_center = tracker.apply_relu_to_error(
            error_points, activation_center
        )

        # Apply saturation
        error_points, activation_center = tracker.apply_saturation(
            error_points, activation_center, qmin, qmax
        )

        # Remove points that collapsed to the same value (degenerate)
        # This represents information loss
        unique_points = np.unique(np.round(error_points, decimals=6), axis=0)
        collapse_ratio = 1 - len(unique_points) / n_points_pre

        stats_post = tracker.measure_shape(error_points)

        history.append({
            'layer': f'layer_{i}',
            'points': error_points.copy(),
            'center': activation_center.copy(),
            'stats_pre_nonlin': stats_pre_relu,
            'stats_post_nonlin': stats_post,
            'collapse_ratio': collapse_ratio,
            'W_spectral_norm': np.linalg.norm(W_quant, ord=2)
        })

        print(f"Layer {i}: "
              f"spectral_norm={np.linalg.norm(W_quant, ord=2):.3f}, "
              f"log_vol_pre={stats_pre_relu['log_volume']:.3f}, "
              f"log_vol_post={stats_post['log_volume']:.3f}, "
              f"collapsed={collapse_ratio*100:.1f}%")

    return history


def visualize_2d_evolution(history):
    """
    For 2D case, visualize the error shape at each layer.
    """
    n_layers = len(history)
    fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(4 * ((n_layers + 1) // 2), 8))
    axes = axes.flatten()

    for i, h in enumerate(history):
        ax = axes[i]
        points = h['points']

        ax.scatter(points[:, 0], points[:, 1], alpha=0.3, s=1)

        # Draw convex hull
        if len(points) > 3:
            try:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], 'r-', linewidth=1)
            except:
                pass

        # Draw principal axes
        stats = h.get('stats') or h.get('stats_post_nonlin')
        if stats and 'principal_directions' in stats:
            center = points.mean(axis=0)
            for j, (s, v) in enumerate(zip(stats['singular_values'][:2],
                                           stats['principal_directions'][:2])):
                ax.arrow(center[0], center[1], v[0]*s*0.5, v[1]*s*0.5,
                        head_width=0.02, color=['blue', 'green'][j], linewidth=2)

        title = h['layer']
        if 'collapse_ratio' in h:
            title += f"\ncollapsed: {h['collapse_ratio']*100:.1f}%"
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)

    plt.tight_layout()
    return fig


def visualize_shape_evolution_stats(history):
    """
    Plot scalar statistics across layers (works for any dimension).
    """
    layers = [h['layer'] for h in history]

    # Extract stats
    log_vols = []
    conditions = []
    spectral_norms = []
    collapse_ratios = []

    for h in history:
        stats = h.get('stats') or h.get('stats_post_nonlin')
        log_vols.append(stats['log_volume'] if stats else 0)
        conditions.append(stats['condition_number'] if stats else 1)
        spectral_norms.append(h.get('W_spectral_norm', 1))
        collapse_ratios.append(h.get('collapse_ratio', 0))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(log_vols, 'o-', linewidth=2, markersize=8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45)
    ax.set_ylabel('log(Volume)')
    ax.set_title('Error Region Volume (log scale)')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(conditions, 's-', linewidth=2, markersize=8, color='orange')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45)
    ax.set_ylabel('Condition Number')
    ax.set_title('Shape Elongation\n(high = stretched in one direction)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.bar(range(1, len(spectral_norms)), spectral_norms[1:], color='steelblue')
    ax.set_xticks(range(1, len(layers)))
    ax.set_xticklabels(layers[1:], rotation=45)
    ax.set_ylabel('Spectral Norm ||W||')
    ax.set_title('Weight Matrix Spectral Norms\n(error amplification factor)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.bar(range(len(collapse_ratios)), collapse_ratios, color='red', alpha=0.7)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45)
    ax.set_ylabel('Collapse Ratio')
    ax.set_title('Information Loss from Nonlinearities\n(fraction of points collapsed)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# Run it
if __name__ == "__main__":
    print("=" * 60)
    print("LINEAR ONLY (2D)")
    print("=" * 60)
    history_linear = run_linear_only_experiment(dims=2, n_layers=5, bits=8)

    fig = visualize_2d_evolution(history_linear)
    plt.savefig('plots/error_shape_linear_2d.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("WITH NONLINEARITIES (2D)")
    print("=" * 60)
    history_nonlin = run_with_nonlinearities_experiment(dims=2, n_layers=5, bits=8)

    fig = visualize_2d_evolution(history_nonlin)
    plt.savefig('plots/error_shape_nonlin_2d.png', dpi=150, bbox_inches='tight')
    plt.show()

    fig = visualize_shape_evolution_stats(history_nonlin)
    plt.savefig('plots/error_shape_stats.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("HIGHER DIMENSIONAL (10D)")
    print("=" * 60)
    history_10d = run_with_nonlinearities_experiment(dims=10, n_layers=5, bits=8)

    fig = visualize_shape_evolution_stats(history_10d)
    plt.savefig('plots/error_shape_stats_10d.png', dpi=150, bbox_inches='tight')
    plt.show()

# %%

# %%
