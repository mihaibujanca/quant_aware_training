import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
import torch
import torch.nn as nn
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import warnings

class QuantizationGeometry:
    """
    Track the actual geometry of quantization error regions,
    accounting for:
    - Hypercube (not ball) initial error
    - ReLU creating polytopes
    - Overflow/underflow saturation
    - Accumulation of errors
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

    def get_scale(self, x_range):
        """Compute quantization scale for a given range"""
        if self.symmetric:
            return x_range / self.qmax
        else:
            return x_range / (self.qmax - self.qmin)

    def quantization_error_vertices_1d(self, x_true, scale):
        """
        For a single value, return the possible error interval.
        Accounts for rounding AND saturation.
        """
        x_q = np.clip(np.round(x_true / scale), self.qmin, self.qmax)
        x_deq = x_q * scale

        # The error depends on where x_true falls
        # If not saturated: error in [-scale/2, scale/2]
        # If saturated: error is one-sided

        if x_true / scale > self.qmax:
            # Overflow: true value > max representable
            # error = x_deq - x_true = qmax*scale - x_true < 0
            return (x_deq - x_true, x_deq - x_true)  # Point, not interval
        elif x_true / scale < self.qmin:
            # Underflow
            return (x_deq - x_true, x_deq - x_true)
        else:
            # Normal case: could round up or down
            return (-scale/2, scale/2)

    def error_hypercube_vertices_2d(self, scale):
        """
        In 2D, the quantization error region is a square (hypercube).
        Returns vertices of error region.
        """
        d = scale / 2
        return np.array([
            [-d, -d],
            [-d,  d],
            [ d,  d],
            [ d, -d]
        ])

    def transform_polytope_affine(self, vertices, W, b):
        """
        Apply affine transformation to polytope vertices.
        v' = W @ v + b
        """
        return vertices @ W.T + b

    def transform_polytope_relu(self, vertices):
        """
        Apply ReLU to polytope. This is tricky because ReLU
        is piecewise linear - we need to handle edge intersections.

        For each dimension, ReLU clips at 0. The output polytope
        is the intersection of the input with the positive orthant,
        with clipped vertices added where edges cross axes.
        """
        n_dims = vertices.shape[1]
        result_vertices = []

        # For 2D, we can enumerate cases more explicitly
        if n_dims == 2:
            return self._relu_polytope_2d(vertices)
        else:
            # General case: clip and hope for the best
            # (proper implementation needs half-space intersection)
            clipped = np.maximum(vertices, 0)
            return clipped

    def _relu_polytope_2d(self, vertices):
        """
        Properly compute ReLU of a 2D polytope by finding
        intersections with axes.
        """
        n = len(vertices)
        new_vertices = []

        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]

            # Add v1 if it's in positive orthant
            if v1[0] >= 0 and v1[1] >= 0:
                new_vertices.append(v1.copy())

            # Check for crossing into/out of positive orthant
            # X-axis crossing
            if (v1[0] < 0) != (v2[0] < 0) and not (v1[0] == 0 and v2[0] == 0):
                t = -v1[0] / (v2[0] - v1[0]) if v2[0] != v1[0] else 0
                crossing = v1 + t * (v2 - v1)
                if crossing[1] >= 0:  # Only if in positive y
                    new_vertices.append(np.array([0, crossing[1]]))

            # Y-axis crossing
            if (v1[1] < 0) != (v2[1] < 0) and not (v1[1] == 0 and v2[1] == 0):
                t = -v1[1] / (v2[1] - v1[1]) if v2[1] != v1[1] else 0
                crossing = v1 + t * (v2 - v1)
                if crossing[0] >= 0:  # Only if in positive x
                    new_vertices.append(np.array([crossing[0], 0]))

        # Add origin if the polytope spans it
        if len(new_vertices) > 0:
            has_neg_x = any(v[0] < 0 for v in vertices)
            has_neg_y = any(v[1] < 0 for v in vertices)
            has_pos_x = any(v[0] > 0 for v in vertices)
            has_pos_y = any(v[1] > 0 for v in vertices)
            if has_neg_x and has_neg_y and has_pos_x and has_pos_y:
                new_vertices.append(np.array([0.0, 0.0]))

        if len(new_vertices) < 3:
            # Degenerate case
            return np.array(new_vertices) if new_vertices else np.array([[0, 0]])

        # Sort vertices by angle to get proper polygon
        new_vertices = np.array(new_vertices)
        center = new_vertices.mean(axis=0)
        angles = np.arctan2(new_vertices[:, 1] - center[1],
                          new_vertices[:, 0] - center[0])
        sorted_idx = np.argsort(angles)
        return new_vertices[sorted_idx]

    def propagate_error_region(self, model, x_center, initial_scale, n_layers=None):
        """
        Propagate the error polytope through the network,
        tracking the actual geometry at each layer.
        """
        # Start with hypercube error region centered at origin
        # (error is relative to true value)
        error_vertices = self.error_hypercube_vertices_2d(initial_scale)

        history = [{
            'layer': 'input',
            'vertices': error_vertices.copy(),
            'type': 'hypercube'
        }]

        layers = list(model.layers[:-1]) if n_layers is None else list(model.layers[:n_layers])

        x_current = x_center.clone()

        for i, layer in enumerate(layers):
            W = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy() if layer.bias is not None else np.zeros(W.shape[0])

            # 1. Affine transformation of error region
            # If x_true has error e, then Wx + b has error W @ e
            error_vertices = self.transform_polytope_affine(error_vertices, W, np.zeros_like(b))

            history.append({
                'layer': f'linear_{i}',
                'vertices': error_vertices.copy(),
                'type': 'affine_transformed'
            })

            # 2. Add new quantization error (expands the region)
            # The output gets quantized, adding another hypercube of error
            x_current = layer(x_current)
            output_scale = self.get_scale(x_current.abs().max().item() * 2)

            if output_scale > 0:
                quant_error = self.error_hypercube_vertices_2d(output_scale)
                # Minkowski sum: expand polytope by quantization error
                error_vertices = self._minkowski_sum_2d(error_vertices, quant_error)

            history.append({
                'layer': f'quantize_{i}',
                'vertices': error_vertices.copy(),
                'type': 'post_quantization'
            })

            # 3. ReLU (clips the region)
            # But wait - ReLU on error is tricky because it depends on
            # where the TRUE activation is, not just the error
            x_true_approx = x_current.detach().numpy().flatten()

            # The error region in activation space is x_true + error_vertices
            activation_region = error_vertices + x_true_approx

            # Apply ReLU
            activation_region_post_relu = self.transform_polytope_relu(activation_region)

            # Transform back to error space
            x_post_relu = np.maximum(x_true_approx, 0)
            error_vertices = activation_region_post_relu - x_post_relu

            history.append({
                'layer': f'relu_{i}',
                'vertices': error_vertices.copy(),
                'activation_center': x_post_relu.copy(),
                'type': 'post_relu'
            })

            x_current = torch.relu(x_current)

        return history

    def _minkowski_sum_2d(self, P, Q):
        """
        Compute Minkowski sum of two 2D polygons.
        Result is the "inflation" of P by Q.
        """
        if len(P) < 3 or len(Q) < 3:
            # Degenerate case
            return P

        # Sort both polygons by angle
        def sort_by_angle(vertices):
            center = vertices.mean(axis=0)
            angles = np.arctan2(vertices[:, 1] - center[1],
                              vertices[:, 0] - center[0])
            return vertices[np.argsort(angles)]

        P = sort_by_angle(P)
        Q = sort_by_angle(Q)

        # Simple approach: compute all pairwise sums and take convex hull
        sums = []
        for p in P:
            for q in Q:
                sums.append(p + q)
        sums = np.array(sums)

        if len(sums) < 3:
            return sums

        try:
            hull = ConvexHull(sums)
            return sums[hull.vertices]
        except:
            return sums


def visualize_error_geometry(history, figsize=(16, 4)):
    """
    Visualize the actual error polytope at each stage.
    """
    n_stages = len(history)
    fig, axes = plt.subplots(1, min(n_stages, 8), figsize=figsize)
    if n_stages == 1:
        axes = [axes]

    colors = {
        'hypercube': 'blue',
        'affine_transformed': 'green',
        'post_quantization': 'orange',
        'post_relu': 'red'
    }

    for i, (ax, stage) in enumerate(zip(axes, history[:8])):
        vertices = stage['vertices']

        if len(vertices) >= 3:
            # Close the polygon
            poly_verts = np.vstack([vertices, vertices[0]])
            ax.fill(poly_verts[:, 0], poly_verts[:, 1],
                   alpha=0.3, color=colors.get(stage['type'], 'gray'))
            ax.plot(poly_verts[:, 0], poly_verts[:, 1],
                   'o-', color=colors.get(stage['type'], 'gray'), markersize=3)
        elif len(vertices) > 0:
            ax.scatter(vertices[:, 0], vertices[:, 1],
                      color=colors.get(stage['type'], 'gray'))

        ax.set_title(f"{stage['layer']}\n({stage['type']})", fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

    plt.tight_layout()
    return fig


def compare_ellipsoid_vs_polytope(model, x_center, scale, n_samples=1000):
    """
    Empirically compare the ellipsoid approximation vs actual error distribution.
    """
    # Sample actual errors
    errors = []
    x_np = x_center.numpy().flatten()

    for _ in range(n_samples):
        # Add random quantization-like error (uniform in hypercube)
        noise = np.random.uniform(-scale/2, scale/2, size=x_np.shape)
        x_noisy = torch.tensor(x_np + noise, dtype=torch.float32).unsqueeze(0)
        x_clean = x_center.clone()

        # Forward both through network
        with torch.no_grad():
            # We'll track error at each layer
            x_n, x_c = x_noisy, x_clean
            for layer in model.layers[:-1]:
                x_n = torch.relu(layer(x_n))
                x_c = torch.relu(layer(x_c))

            final_error = (x_n - x_c).numpy().flatten()
            errors.append(final_error)

    errors = np.array(errors)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter of actual errors
    if errors.shape[1] >= 2:
        axes[0].scatter(errors[:, 0], errors[:, 1], alpha=0.3, s=10)
        axes[0].set_xlabel('Error dim 0')
        axes[0].set_ylabel('Error dim 1')
        axes[0].set_title('Actual Error Distribution')
        axes[0].set_aspect('equal')

        # Overlay ellipsoid approximation
        cov = np.cov(errors.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Draw ellipse
        theta = np.linspace(0, 2*np.pi, 100)
        ellipse = np.column_stack([np.cos(theta), np.sin(theta)])
        ellipse = ellipse @ np.diag(np.sqrt(eigenvalues) * 2) @ eigenvectors.T
        axes[0].plot(ellipse[:, 0], ellipse[:, 1], 'r-', linewidth=2, label='Gaussian ellipse')
        axes[0].legend()

    # Histogram of error magnitudes
    magnitudes = np.linalg.norm(errors, axis=1)
    axes[1].hist(magnitudes, bins=50, density=True, alpha=0.7)
    axes[1].set_xlabel('Error magnitude')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Error Magnitude Distribution')

    # If errors were Gaussian, magnitude would be chi-distributed
    from scipy import stats
    if errors.shape[1] >= 2:
        x_range = np.linspace(0, magnitudes.max(), 100)
        # Chi distribution with k=n_dims degrees of freedom, scaled
        scale_est = np.sqrt(np.mean(magnitudes**2) / errors.shape[1])
        chi_pdf = stats.chi.pdf(x_range / scale_est, df=errors.shape[1]) / scale_est
        axes[1].plot(x_range, chi_pdf, 'r-', linewidth=2, label='Chi (Gaussian assumption)')
        axes[1].legend()

    plt.tight_layout()
    return fig, errors


def analyze_saturation_events(model, n_samples=1000, bits=8):
    """
    Track how often saturation (overflow/underflow) occurs at each layer.
    This is when the ellipsoid model really breaks down.
    """
    qmax = 2**(bits-1) - 1
    qmin = -(2**(bits-1))

    saturation_counts = {}

    x_samples = torch.randn(n_samples, 2)

    for layer_idx, layer in enumerate(model.layers[:-1]):
        saturation_counts[layer_idx] = {'overflow': 0, 'underflow': 0, 'total_activations': 0}

    for x in x_samples:
        x = x.unsqueeze(0)

        for layer_idx, layer in enumerate(model.layers[:-1]):
            x = layer(x)

            # Compute scale for this activation
            scale = x.abs().max().item() / qmax if x.abs().max().item() > 0 else 1.0

            # Check for saturation (values that would clip)
            x_scaled = x / scale
            overflow = (x_scaled > qmax).sum().item()
            underflow = (x_scaled < qmin).sum().item()

            saturation_counts[layer_idx]['overflow'] += overflow
            saturation_counts[layer_idx]['underflow'] += underflow
            saturation_counts[layer_idx]['total_activations'] += x.numel()

            x = torch.relu(x)

    return saturation_counts


# ============ Run experiments ============
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Create network with 2D intermediate representations for visualization
    model = SimpleNet(dims=[2, 2, 2, 1])  # Keep 2D throughout for visualization

    # Initialize with larger weights to see effects
    for layer in model.layers:
        nn.init.xavier_uniform_(layer.weight, gain=3.0)

    print("=== Experiment 1: Actual Error Polytope Geometry ===")
    geom = QuantizationGeometry(bits=8)
    x_center = torch.tensor([[1.0, 0.5]])
    initial_scale = 0.1

    history = geom.propagate_error_region(model, x_center, initial_scale)

    for stage in history:
        verts = stage['vertices']
        if len(verts) > 0:
            # Compute "size" metrics
            ranges = verts.max(axis=0) - verts.min(axis=0)
            print(f"{stage['layer']:20s} | vertices: {len(verts):3d} | "
                  f"range: [{ranges[0]:.4f}, {ranges[1]:.4f}]")

    fig = visualize_error_geometry(history)
    plt.savefig('error_polytopes.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n=== Experiment 2: Ellipsoid vs Reality ===")
    fig, errors = compare_ellipsoid_vs_polytope(model, x_center, initial_scale)
    plt.savefig('ellipsoid_vs_reality.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n=== Experiment 3: Saturation Events ===")
    sat_counts = analyze_saturation_events(model, n_samples=1000, bits=8)
    for layer_idx, counts in sat_counts.items():
        total = counts['total_activations']
        overflow_pct = 100 * counts['overflow'] / total if total > 0 else 0
        underflow_pct = 100 * counts['underflow'] / total if total > 0 else 0
        print(f"Layer {layer_idx}: overflow={overflow_pct:.2f}%, underflow={underflow_pct:.2f}%")

    print("\n=== Experiment 4: Different bit widths ===")
    for bits in [8, 4, 2]:
        geom = QuantizationGeometry(bits=bits)
        history = geom.propagate_error_region(model, x_center, initial_scale * (256 / 2**bits))
        final_verts = history[-1]['vertices']
        if len(final_verts) > 0:
            final_range = final_verts.max(axis=0) - final_verts.min(axis=0)
            print(f"{bits}-bit: final error range = {final_range}")

# ## Key Insights

# ### The actual error region is:

# 1. **Initially a hypercube** (not a ball)
# 2. **Sheared into a parallelepiped** by linear layers
# 3. **Expanded by Minkowski sum** when new quantization error is added
# 4. **Truncated into a polytope** by ReLU
# 5. **Made asymmetric** by saturation events

# ### Why this matters for your binary classifier:

# The decision boundary is a hyperplane. What you really care about is: **does the error polytope cross the decision boundary?**
# ```
#   True boundary: ----+----
#                      |
#   Error polytope:   [===]  ← if this spans the boundary,
#                      |       you get misclassification
