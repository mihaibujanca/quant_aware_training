import numpy as np
import matplotlib.pyplot as plt
import torch


# =============================================================================
# Model wrappers for plotting
# =============================================================================


class QuantizedWrapper:
    """Wraps a model to call forward_quantized() for plotting."""

    def __init__(self, model, scale_factors, zero_points, num_bits=8):
        self.model = model
        self.sf = scale_factors
        self.zp = zero_points
        self.bits = num_bits

    def eval(self):
        self.model.eval()

    def __call__(self, x):
        return self.model.forward_quantized(x, self.sf, self.zp, num_bits=self.bits)


class OracleWrapper:
    """Wraps a model to call forward_with_oracle_correction() for plotting."""

    def __init__(self, model, scale_factors, zero_points, correct_every_n=3, num_bits=8):
        self.model = model
        self.sf = scale_factors
        self.zp = zero_points
        self.n = correct_every_n
        self.bits = num_bits

    def eval(self):
        self.model.eval()

    def __call__(self, x):
        out, _ = self.model.forward_with_oracle_correction(
            x, self.sf, self.zp, self.n, num_bits=self.bits
        )
        return out


class LearnedCorrectionWrapper:
    """Wraps a model to call forward_quantized_with_correction() for plotting."""

    def __init__(self, model, scale_factors, zero_points, num_bits=8):
        self.model = model
        self.sf = scale_factors
        self.zp = zero_points
        self.bits = num_bits

    def eval(self):
        self.model.eval()

    def __call__(self, x):
        return self.model.forward_quantized_with_correction(
            x, self.sf, self.zp, num_bits=self.bits
        )


# =============================================================================
# Plotting
# =============================================================================


def plot_decision_boundary(model, X, y, title, ax, embedding=None):
    """
    Plot decision boundary for a 2D classifier.

    Args:
        model: PyTorch model for classification
        X: 2D input data of shape (n_samples, 2) - original low-dimensional data
        y: Labels of shape (n_samples,)
        title: Plot title
        ax: Matplotlib axis to plot on
        embedding: Optional AffineEmbedding object. If provided, grid points are
                   transformed through the embedding before being passed to the model.
                   This allows visualizing decision boundaries of models trained on
                   high-dimensional embedded data in the original 2D space.
    """
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Create 2D grid points
    grid_2d = np.c_[xx.ravel(), yy.ravel()]

    # If embedding provided, transform 2D grid to high-D before passing to model
    if embedding is not None:
        grid_high_d = embedding.transform(grid_2d)
        grid = torch.tensor(grid_high_d, dtype=torch.float32)
    else:
        grid = torch.tensor(grid_2d, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        Z = model(grid).argmax(1).numpy()
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="black", s=20)
    ax.set_title(title)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())