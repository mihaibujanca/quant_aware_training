import numpy as np


class AffineEmbedding:
    """
    Invertible affine embedding from low-dimensional to high-dimensional space.

    Transforms X_low -> X_high = X_low @ W + b

    The transform is saved so we can:
    1. Apply the same transform to new data (e.g., for visualization grid)
    2. Compute the pseudoinverse to project back to low-D
    """

    def __init__(self, input_dim, target_dim, random_state=None):
        """
        Initialize the affine embedding.

        Args:
            input_dim: Input dimensionality (e.g., 2 for 2D data)
            target_dim: Target dimensionality for embedding
            random_state: Random seed for reproducibility
        """
        if random_state is not None:
            np.random.seed(random_state)

        self.input_dim = input_dim
        self.target_dim = target_dim

        # Random projection matrix (input_dim x target_dim)
        # Scale to preserve approximate distances
        self.W = np.random.randn(input_dim, target_dim).astype(np.float32) / np.sqrt(input_dim)

        # Random bias
        self.b = np.random.randn(target_dim).astype(np.float32) * 0.1

        # Precompute pseudoinverse for inverse_transform
        self._W_pinv = np.linalg.pinv(self.W)

    def transform(self, X):
        """
        Embed low-dimensional data into high-dimensional space.

        Args:
            X: Array of shape (n_samples, input_dim)

        Returns:
            X_embedded: Array of shape (n_samples, target_dim)
        """
        X = np.asarray(X, dtype=np.float32)
        return (X @ self.W + self.b).astype(np.float32)

    def inverse_transform(self, X_high):
        """
        Project high-dimensional data back to low-dimensional space.

        Uses the pseudoinverse, so this is a least-squares approximation.
        For data that was embedded with transform(), this recovers the original.

        Args:
            X_high: Array of shape (n_samples, target_dim)

        Returns:
            X_low: Array of shape (n_samples, input_dim)
        """
        X_high = np.asarray(X_high, dtype=np.float32)
        return ((X_high - self.b) @ self._W_pinv).astype(np.float32)

    def __call__(self, X):
        """Shorthand for transform()."""
        return self.transform(X)


def embed_dataset_in_high_dimensional_space(X, target_dim=100, random_state=None):
    """
    Embed a low-dimensional dataset into a higher-dimensional space.

    Returns both the embedded data and the embedding object for later use.

    Args:
        X: Input array of shape (n_samples, input_dim)
        target_dim: Target dimensionality for the embedding
        random_state: Random seed for reproducibility

    Returns:
        X_embedded: Array of shape (n_samples, target_dim)
        embedding: AffineEmbedding object that can transform new points
    """
    input_dim = X.shape[1]
    embedding = AffineEmbedding(input_dim, target_dim, random_state=random_state)
    X_embedded = embedding.transform(X)
    return X_embedded, embedding


def make_spirals(n_samples=1000, noise=0.5, n_turns=2, random_state=None):
    """
    Generate a 2D spiral dataset with two interleaved spirals.

    This is a challenging dataset that requires complex non-linear decision boundaries.

    Args:
        n_samples: Total number of samples (split evenly between classes)
        noise: Standard deviation of Gaussian noise added to the data
        n_turns: Number of turns in each spiral (more turns = harder)
        random_state: Random seed for reproducibility

    Returns:
        X: Array of shape (n_samples, 2) with coordinates
        y: Array of shape (n_samples,) with class labels (0 or 1)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = n_samples // 2

    # Generate spiral for class 0
    theta0 = np.linspace(0, n_turns * 2 * np.pi, n)
    r0 = theta0 / (n_turns * 2 * np.pi)
    x0 = r0 * np.cos(theta0) + np.random.randn(n) * noise * 0.1
    y0 = r0 * np.sin(theta0) + np.random.randn(n) * noise * 0.1

    # Generate spiral for class 1 (rotated by pi)
    theta1 = np.linspace(0, n_turns * 2 * np.pi, n)
    r1 = theta1 / (n_turns * 2 * np.pi)
    x1 = r1 * np.cos(theta1 + np.pi) + np.random.randn(n) * noise * 0.1
    y1 = r1 * np.sin(theta1 + np.pi) + np.random.randn(n) * noise * 0.1

    X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(int)

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]
