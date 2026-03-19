import os
import urllib.request

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from skimage.measure import find_contours

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, 'data')


# =============================================================================
# MNIST
# =============================================================================


def load_mnist_flat(batch_size=256, data_dir=_DEFAULT_DATA_DIR):
    """
    Load MNIST dataset with images flattened to 784-dim vectors.

    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# =============================================================================
# Shakespeare (character-level)
# =============================================================================


def load_shakespeare(seq_len=128, data_dir=_DEFAULT_DATA_DIR):
    """
    Load tiny Shakespeare dataset for character-level language modeling.

    Returns:
        train_X: Training sequences (token ids), shape (n_train, seq_len)
        train_Y: Training targets, shape (n_train, seq_len)
        test_X: Test sequences, shape (n_test, seq_len)
        test_Y: Test targets, shape (n_test, seq_len)
        vocab_size: Number of unique characters
        char_to_idx: Dict mapping char -> int
        idx_to_char: Dict mapping int -> char
    """
    data_path = os.path.join(data_dir, 'shakespeare.txt')
    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(data_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading Shakespeare from {url}...")
        urllib.request.urlretrieve(url, data_path)

    with open(data_path, 'r') as f:
        text = f.read()

    # Build vocabulary
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    # Tokenize
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

    # Split into sequences
    n_seqs = len(data) // (seq_len + 1)
    data = data[:n_seqs * (seq_len + 1)]
    data = data.view(n_seqs, seq_len + 1)

    X = data[:, :-1]
    Y = data[:, 1:]

    # Train/test split (90/10)
    n_train = int(len(X) * 0.9)
    train_X, train_Y = X[:n_train], Y[:n_train]
    test_X, test_Y = X[n_train:], Y[n_train:]

    return train_X, train_Y, test_X, test_Y, len(chars), char_to_idx, idx_to_char


# =============================================================================
# Synthetic datasets
# =============================================================================


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


class IdentityEmbedding:
    """Identity embedding used for native low-dimensional reference runs."""

    def __init__(self, dim: int):
        self.input_dim = dim
        self.target_dim = dim
        self.W = np.eye(dim, dtype=np.float32)
        self.b = np.zeros(dim, dtype=np.float32)
        self._W_pinv = self.W

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return X.astype(np.float32)

    def inverse_transform(self, X_high):
        X_high = np.asarray(X_high, dtype=np.float32)
        return X_high.astype(np.float32)

    def __call__(self, X):
        return self.transform(X)


@dataclass(frozen=True)
class EmbeddedClassificationSplit:
    """One split of an embedded classification benchmark."""

    X_low: np.ndarray
    X_high: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class EmbeddedMultiLabelSplit:
    """One split of an embedded multi-label classification benchmark."""

    X_low: np.ndarray
    X_high: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class EmbeddedClassificationBenchmark:
    """Benchmark bundle exposing both intrinsic and observed geometry."""

    name: str
    intrinsic_dim: int
    observed_dim: int
    embedding: AffineEmbedding | IdentityEmbedding
    train: EmbeddedClassificationSplit
    val: EmbeddedClassificationSplit
    test: EmbeddedClassificationSplit
    boundary_points_low: np.ndarray | None = None
    true_curvature_low: np.ndarray | None = None


@dataclass(frozen=True)
class EmbeddedMultiLabelBenchmark:
    """Benchmark bundle for embedded multi-label classification."""

    name: str
    intrinsic_dim: int
    observed_dim: int
    embedding: AffineEmbedding | IdentityEmbedding
    train: EmbeddedMultiLabelSplit
    val: EmbeddedMultiLabelSplit
    test: EmbeddedMultiLabelSplit
    label_names: tuple[str, ...]


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


def make_rough_spirals(
    n_samples=1000,
    noise=0.35,
    n_turns=4,
    random_state=None,
):
    """Generate a spiral dataset with localized high-frequency roughness.

    The roughness is concentrated in a few windows along arclength, so the
    underlying function has long smooth stretches plus locally hard regions.
    """
    rng = np.random.default_rng(random_state)
    n = n_samples // 2
    theta = np.linspace(0.0, n_turns * 2.0 * np.pi, n, dtype=np.float32)
    t = theta / max(float(theta[-1]), 1e-6)

    rough = (
        0.075 * np.exp(-0.5 * ((t - 0.28) / 0.08) ** 2) * np.sin(17.0 * theta + 0.35) +
        0.060 * np.exp(-0.5 * ((t - 0.57) / 0.07) ** 2) * np.sin(23.0 * theta - 0.55) +
        0.045 * np.exp(-0.5 * ((t - 0.82) / 0.05) ** 2) * np.sin(31.0 * theta + 0.80)
    ).astype(np.float32)
    base_radius = t.astype(np.float32) + rough

    radial_noise = rng.normal(scale=noise * 0.05, size=n).astype(np.float32)
    tangent_noise = rng.normal(scale=noise * 0.03, size=n).astype(np.float32)

    radius0 = base_radius + radial_noise
    x0 = radius0 * np.cos(theta) - tangent_noise * np.sin(theta)
    y0 = radius0 * np.sin(theta) + tangent_noise * np.cos(theta)

    radius1 = base_radius + rng.normal(scale=noise * 0.05, size=n).astype(np.float32)
    tangent1 = rng.normal(scale=noise * 0.03, size=n).astype(np.float32)
    x1 = radius1 * np.cos(theta + np.pi) - tangent1 * np.sin(theta + np.pi)
    y1 = radius1 * np.sin(theta + np.pi) + tangent1 * np.cos(theta + np.pi)

    X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])]).astype(np.float32)
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(np.int64)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def _sample_unit_directions(n_samples: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    directions = rng.normal(size=(n_samples, dim)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.clip(norms, 1e-8, None)
    return directions.astype(np.float32)


def _rough_hypershell_radius(directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float32)
    dim = directions.shape[1]
    u0 = directions[:, 0]
    u1 = directions[:, 1]
    u2 = directions[:, 2] if dim >= 3 else np.zeros(len(directions), dtype=np.float32)
    extra = directions[:, 3:] if dim > 3 else np.zeros((len(directions), 0), dtype=np.float32)

    phase_xy = np.arctan2(u1, u0 + 1e-8).astype(np.float32)
    phase_z = np.arctan2(u2, np.sqrt(u0 * u0 + u1 * u1) + 1e-8).astype(np.float32)
    extra_phase = extra.sum(axis=1).astype(np.float32) if extra.shape[1] > 0 else np.zeros(len(directions), dtype=np.float32)

    window_a = np.exp(
        -0.5 * (
            ((u0 - 0.78) / 0.22) ** 2
            + ((u1 - 0.10) / 0.22) ** 2
            + ((u2 - 0.42) / 0.26) ** 2
        )
    ).astype(np.float32)
    window_b = np.exp(
        -0.5 * (
            ((u0 + 0.30) / 0.20) ** 2
            + ((u1 + 0.72) / 0.20) ** 2
            + ((u2 - 0.18) / 0.24) ** 2
        )
    ).astype(np.float32)

    if extra.shape[1] > 0:
        extra_window_a = np.exp(-0.5 * ((extra[:, 0] + 0.55) / 0.28) ** 2).astype(np.float32)
        extra_window_b = np.exp(-0.5 * ((extra[:, -1] - 0.35) / 0.24) ** 2).astype(np.float32)
        window_a = window_a * extra_window_a
        window_b = window_b * extra_window_b

    smooth_bulge = (
        0.08 * (u0 * u2)
        + 0.05 * np.sin(2.0 * phase_xy - 1.2 * phase_z + 0.7 * extra_phase)
    ).astype(np.float32)
    rough = (
        0.18 * window_a * np.sin(12.0 * phase_xy + 3.5 * phase_z + 1.5 * extra_phase)
        + 0.14 * window_b * np.sin(18.0 * phase_xy - 4.0 * phase_z - 1.2 * extra_phase + 0.6)
    ).astype(np.float32)
    radius = 1.0 + smooth_bulge + rough
    return np.clip(radius, 0.55, 1.45).astype(np.float32)


def make_rough_hypershell(
    n_samples=2000,
    intrinsic_dim=3,
    noise=0.10,
    random_state=None,
):
    """Generate a boundary-focused classification dataset in 3D or 5D.

    The decision boundary is a hypersphere-like shell with a few localized
    corrugated patches. Most of the boundary is smooth; a small number of
    angular windows are deliberately rough.
    """
    if intrinsic_dim < 3:
        raise ValueError("rough hypershell requires intrinsic_dim >= 3")

    rng = np.random.default_rng(random_state)
    n_focus = int(n_samples * 0.70)
    n_uniform = n_samples - n_focus

    directions = _sample_unit_directions(n_focus, intrinsic_dim, rng)
    boundary_radius = _rough_hypershell_radius(directions)
    radial_noise = rng.normal(scale=noise, size=n_focus).astype(np.float32)
    focused_r = np.clip(boundary_radius + radial_noise, 0.10, None)
    X_focus = directions * focused_r[:, None]

    uniform_scale = 1.45
    X_uniform = rng.uniform(
        low=-uniform_scale,
        high=uniform_scale,
        size=(n_uniform, intrinsic_dim),
    ).astype(np.float32)

    X = np.vstack([X_focus, X_uniform]).astype(np.float32)
    radii = np.linalg.norm(X, axis=1).astype(np.float32)
    directions_all = X / np.clip(radii[:, None], 1e-8, None)
    signed = (_rough_hypershell_radius(directions_all) - radii).astype(np.float32)
    y = (signed >= 0.0).astype(np.int64)

    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def make_embedded_classification_benchmark(
    dataset: str = "spirals",
    *,
    observed_dim: int = 100,
    n_samples: int = 2000,
    noise: float | None = None,
    n_turns: int = 3,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    random_state: int | None = None,
) -> EmbeddedClassificationBenchmark:
    """Create a high-D-first synthetic benchmark with intrinsic coordinates retained.

    The returned benchmark always includes:
    - X_low: intrinsic low-dimensional coordinates for diagnostics/geometry
    - X_high: observed inputs used by the models
    - embedding: the low->high affine map (identity for native reference runs)
    - y: labels
    """
    if dataset == "spirals":
        X_low, y = make_spirals(
            n_samples=n_samples,
            noise=0.5 if noise is None else noise,
            n_turns=n_turns,
            random_state=random_state,
        )
    elif dataset == "rough_spirals":
        X_low, y = make_rough_spirals(
            n_samples=n_samples,
            noise=0.35 if noise is None else noise,
            n_turns=max(n_turns, 4),
            random_state=random_state,
        )
    elif dataset == "moons":
        from sklearn.datasets import make_moons

        X_low, y = make_moons(
            n_samples=n_samples,
            noise=0.15 if noise is None else noise,
            random_state=random_state,
        )
        X_low = X_low.astype(np.float32)
        y = y.astype(np.int64)
    elif dataset == "rough_hypershell_3d":
        X_low, y = make_rough_hypershell(
            n_samples=n_samples,
            intrinsic_dim=3,
            noise=0.10 if noise is None else noise,
            random_state=random_state,
        )
    elif dataset == "rough_hypershell_5d":
        X_low, y = make_rough_hypershell(
            n_samples=n_samples,
            intrinsic_dim=5,
            noise=0.11 if noise is None else noise,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unsupported dataset {dataset!r}. Choose 'spirals', 'rough_spirals', 'moons', "
            "'rough_hypershell_3d', or 'rough_hypershell_5d'."
        )

    intrinsic_dim = X_low.shape[1]
    if observed_dim == intrinsic_dim:
        embedding = IdentityEmbedding(intrinsic_dim)
        X_high = X_low.astype(np.float32)
    else:
        X_high, embedding = embed_dataset_in_high_dimensional_space(
            X_low, target_dim=observed_dim, random_state=random_state
        )

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_fraction,
        random_state=random_state,
        stratify=y,
    )
    val_size = val_fraction / max(1.0 - test_fraction, 1e-8)
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size,
        random_state=random_state,
        stratify=y[train_idx],
    )

    def _split(ix: np.ndarray) -> EmbeddedClassificationSplit:
        return EmbeddedClassificationSplit(
            X_low=X_low[ix].astype(np.float32),
            X_high=X_high[ix].astype(np.float32),
            y=y[ix].astype(np.int64),
        )

    return EmbeddedClassificationBenchmark(
        name=dataset,
        intrinsic_dim=intrinsic_dim,
        observed_dim=observed_dim,
        embedding=embedding,
        train=_split(train_idx),
        val=_split(val_idx),
        test=_split(test_idx),
    )


def _wrapped_angle_delta(theta: np.ndarray, center: float) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float32)
    return ((theta - center + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _raised_cosine_window(theta: np.ndarray, center: float, half_width: float) -> np.ndarray:
    delta = np.abs(_wrapped_angle_delta(theta, center))
    window = np.zeros_like(delta, dtype=np.float32)
    mask = delta <= half_width
    if np.any(mask):
        window[mask] = 0.5 * (1.0 + np.cos(np.pi * delta[mask] / max(half_width, 1e-6)))
    return window.astype(np.float32)


def _radial_shape_radius(shape: str, theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float32)
    if shape == "flower":
        radius = 1.0 + 0.18 * np.cos(6.0 * theta)
    elif shape == "rounded_star":
        radius = 0.90 + 0.22 * np.cos(5.0 * theta) + 0.10 * np.cos(10.0 * theta)
    elif shape == "dumbbell":
        radius = 0.82 + 0.33 * np.cos(2.0 * theta)
    elif shape == "windowed_ripple_ring":
        base = 1.02 + 0.05 * np.cos(2.0 * theta - 0.35)
        ripple_a = 0.12 * _raised_cosine_window(theta, center=-1.05, half_width=0.48) * np.cos(18.0 * (theta + 0.12))
        ripple_b = 0.10 * _raised_cosine_window(theta, center=1.42, half_width=0.42) * np.cos(16.0 * (theta - 0.28))
        radius = base + ripple_a + ripple_b
    elif shape == "corrugated_dumbbell":
        base = 0.84 + 0.31 * np.cos(2.0 * theta + 0.08)
        ripple = 0.12 * _raised_cosine_window(theta, center=0.62, half_width=0.46) * np.cos(16.0 * (theta - 0.62))
        shoulder = 0.04 * np.cos(6.0 * theta - 0.15)
        radius = base + shoulder + ripple
    else:
        raise ValueError(
            f"Unsupported boundary dataset {shape!r}. "
            "Choose 'flower', 'rounded_star', 'dumbbell', 'windowed_ripple_ring', "
            "'corrugated_dumbbell', 'serpentine_tube', or 'multi_island'."
        )
    return np.maximum(radius.astype(np.float32), 0.15)


def _closed_curve_curvature(points: np.ndarray) -> np.ndarray:
    prev_pts = np.roll(points, 1, axis=0)
    next_pts = np.roll(points, -1, axis=0)
    vel = 0.5 * (next_pts - prev_pts)
    acc = next_pts - 2.0 * points + prev_pts
    numer = np.abs(vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0])
    denom = np.power(np.sum(vel ** 2, axis=1), 1.5)
    denom = np.maximum(denom, 1e-8)
    return (numer / denom).astype(np.float32)


def _point_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    seg = end - start
    denom = float(np.dot(seg, seg))
    if denom <= 1e-8:
        return np.linalg.norm(points - start[None, :], axis=1).astype(np.float32)
    rel = points - start[None, :]
    alpha = np.clip((rel @ seg) / denom, 0.0, 1.0)
    proj = start[None, :] + alpha[:, None] * seg[None, :]
    return np.linalg.norm(points - proj, axis=1).astype(np.float32)


def _radial_signed_field(shape: str, points: np.ndarray) -> np.ndarray:
    theta = np.arctan2(points[:, 1], points[:, 0]).astype(np.float32)
    radius = np.linalg.norm(points, axis=1).astype(np.float32)
    boundary_radius = _radial_shape_radius(shape, theta)
    return (boundary_radius - radius).astype(np.float32)


def _serpentine_polyline() -> np.ndarray:
    return np.asarray(
        [
            [-0.92, -1.18],
            [0.96, -0.74],
            [-0.88, -0.18],
            [0.92, 0.38],
            [-0.82, 1.12],
        ],
        dtype=np.float32,
    )


def _serpentine_signed_field(points: np.ndarray) -> np.ndarray:
    polyline = _serpentine_polyline()
    min_dist = np.full(len(points), np.inf, dtype=np.float32)
    for start, end in zip(polyline[:-1], polyline[1:]):
        d = _point_segment_distance(points, start, end)
        min_dist = np.minimum(min_dist, d)
    tube_radius = 0.20
    return (tube_radius - min_dist).astype(np.float32)


def _ellipse_signed_field(
    points: np.ndarray,
    *,
    center: tuple[float, float],
    axes: tuple[float, float],
    angle: float = 0.0,
) -> np.ndarray:
    pts = points - np.asarray(center, dtype=np.float32)[None, :]
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    local = pts @ rot
    scaled = np.sqrt(
        (local[:, 0] / max(float(axes[0]), 1e-6)) ** 2 +
        (local[:, 1] / max(float(axes[1]), 1e-6)) ** 2
    )
    return (1.0 - scaled).astype(np.float32)


def _offset_radial_signed_field(
    points: np.ndarray,
    *,
    center: tuple[float, float],
    base_radius: float,
    ripple_scale: float,
    ripple_freq: float,
    ripple_phase: float,
    window_center: float | None = None,
    window_half_width: float | None = None,
) -> np.ndarray:
    centered = points - np.asarray(center, dtype=np.float32)[None, :]
    theta = np.arctan2(centered[:, 1], centered[:, 0]).astype(np.float32)
    radius = np.linalg.norm(centered, axis=1).astype(np.float32)
    ripple = ripple_scale * np.cos(ripple_freq * theta + ripple_phase)
    if window_center is not None and window_half_width is not None:
        ripple = ripple * _raised_cosine_window(theta, center=window_center, half_width=window_half_width)
    boundary_radius = np.maximum(base_radius + ripple, 0.08).astype(np.float32)
    return (boundary_radius - radius).astype(np.float32)


def _boundary_signed_field(shape: str, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    radial_shapes = {
        "flower",
        "rounded_star",
        "dumbbell",
        "windowed_ripple_ring",
        "corrugated_dumbbell",
    }
    if shape in radial_shapes:
        return _radial_signed_field(shape, points)
    if shape == "serpentine_tube":
        return _serpentine_signed_field(points)
    if shape == "multi_island":
        circle_a = 0.43 - np.linalg.norm(points - np.asarray([-0.92, 0.05], dtype=np.float32)[None, :], axis=1)
        island_b = _offset_radial_signed_field(
            points,
            center=(0.78, 0.44),
            base_radius=0.34,
            ripple_scale=0.08,
            ripple_freq=8.0,
            ripple_phase=-0.35,
            window_center=0.45,
            window_half_width=1.15,
        )
        ellipse_c = _ellipse_signed_field(
            points,
            center=(0.32, -0.78),
            axes=(0.25, 0.17),
            angle=0.35,
        )
        return np.maximum(np.maximum(circle_a.astype(np.float32), island_b), ellipse_c)
    raise ValueError(f"Unsupported boundary dataset {shape!r}.")


def _shape_support_radius(shape: str) -> float:
    if shape == "serpentine_tube":
        return 1.65
    if shape == "multi_island":
        return 1.55
    if shape in {"windowed_ripple_ring", "corrugated_dumbbell"}:
        return 1.6
    return 1.45


def _sample_shape_boundary(shape: str, n_boundary_points: int) -> tuple[np.ndarray, np.ndarray]:
    radial_shapes = {
        "flower",
        "rounded_star",
        "dumbbell",
        "windowed_ripple_ring",
        "corrugated_dumbbell",
    }
    if shape in radial_shapes:
        theta_boundary = np.linspace(-np.pi, np.pi, n_boundary_points, endpoint=False, dtype=np.float32)
        radius_boundary = _radial_shape_radius(shape, theta_boundary)
        boundary_points_low = np.column_stack(
            [
                radius_boundary * np.cos(theta_boundary),
                radius_boundary * np.sin(theta_boundary),
            ]
        ).astype(np.float32)
        return boundary_points_low, _closed_curve_curvature(boundary_points_low)
    return _sample_implicit_boundary(shape, n_boundary_points)


def _extract_closed_contours(
    field: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    *,
    min_points: int = 64,
) -> list[np.ndarray]:
    contours = find_contours(field, level=0.0)
    out: list[np.ndarray] = []
    for contour in contours:
        if len(contour) < min_points:
            continue
        row_idx = contour[:, 0]
        col_idx = contour[:, 1]
        x = np.interp(col_idx, np.arange(len(grid_x)), grid_x)
        y = np.interp(row_idx, np.arange(len(grid_y)), grid_y)
        pts = np.column_stack([x, y]).astype(np.float32)
        if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
            pts = np.vstack([pts, pts[:1]])
        out.append(pts.astype(np.float32))
    if not out:
        raise RuntimeError("Could not extract a boundary contour for the requested shape.")
    return out


def _curve_length(points: np.ndarray) -> float:
    pts = points if np.linalg.norm(points[0] - points[-1]) <= 1e-6 else np.vstack([points, points[:1]])
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _resample_closed_curve(points: np.ndarray, n_samples: int) -> np.ndarray:
    pts = points if np.linalg.norm(points[0] - points[-1]) <= 1e-6 else np.vstack([points, points[:1]])
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cumulative[-1])
    if total <= 1e-8:
        return np.repeat(pts[:1], n_samples, axis=0).astype(np.float32)
    samples = np.linspace(0.0, total, n_samples, endpoint=False, dtype=np.float32)
    x = np.interp(samples, cumulative, pts[:, 0]).astype(np.float32)
    y = np.interp(samples, cumulative, pts[:, 1]).astype(np.float32)
    return np.column_stack([x, y]).astype(np.float32)


def _sample_implicit_boundary(shape: str, n_boundary_points: int) -> tuple[np.ndarray, np.ndarray]:
    support = _shape_support_radius(shape)
    grid_size = int(np.clip(np.sqrt(max(n_boundary_points, 64)) * 18.0, 256, 768))
    grid_x = np.linspace(-support, support, grid_size, dtype=np.float32)
    grid_y = np.linspace(-support, support, grid_size, dtype=np.float32)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    query = np.column_stack([mesh_x.reshape(-1), mesh_y.reshape(-1)]).astype(np.float32)
    field = _boundary_signed_field(shape, query).reshape(grid_size, grid_size)
    contours = _extract_closed_contours(field, grid_x, grid_y)

    lengths = np.asarray([_curve_length(contour) for contour in contours], dtype=np.float32)
    weights = lengths / max(float(lengths.sum()), 1e-8)
    min_per_curve = min(64, max(16, n_boundary_points // max(4 * len(contours), 1)))
    allocations = np.maximum(min_per_curve, np.floor(weights * n_boundary_points).astype(int))
    while allocations.sum() < n_boundary_points:
        allocations[int(np.argmax(weights - allocations / max(n_boundary_points, 1)))] += 1
    while allocations.sum() > n_boundary_points:
        idx = int(np.argmax(allocations))
        if allocations[idx] <= min_per_curve:
            break
        allocations[idx] -= 1

    sampled_curves = [
        _resample_closed_curve(contour, int(count))
        for contour, count in zip(contours, allocations)
        if count > 0
    ]
    boundary_points_low = np.vstack(sampled_curves).astype(np.float32)
    true_curvature_low = np.concatenate(
        [_closed_curve_curvature(curve).astype(np.float32) for curve in sampled_curves],
        axis=0,
    ).astype(np.float32)
    return boundary_points_low, true_curvature_low


def make_boundary_classification(
    shape: str,
    *,
    n_samples: int = 2000,
    noise: float = 0.08,
    boundary_fraction: float = 0.55,
    n_boundary_points: int = 1024,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a binary classification dataset from a closed 2D boundary.

    Returns:
        X_low: [N, 2] sampled intrinsic coordinates
        y: [N] labels, with 1 indicating points inside the boundary
        boundary_points_low: [M, 2] dense boundary sample in intrinsic space
        true_curvature_low: [M] curvature magnitude along the boundary sample
    """
    rng = np.random.default_rng(random_state)
    boundary_points_low, true_curvature_low = _sample_shape_boundary(shape, n_boundary_points)

    n_focus = int(round(n_samples * boundary_fraction))
    n_uniform = max(n_samples - n_focus, 0)

    focus_idx = rng.integers(0, len(boundary_points_low), size=n_focus)
    X_focus = (
        boundary_points_low[focus_idx] +
        rng.normal(scale=noise, size=(n_focus, 2)).astype(np.float32)
    ).astype(np.float32)

    support_radius = float(np.max(np.abs(boundary_points_low)) + 0.40)
    X_uniform = rng.uniform(
        low=-support_radius,
        high=support_radius,
        size=(n_uniform, 2),
    ).astype(np.float32)

    X_low = np.vstack([X_focus, X_uniform]).astype(np.float32)
    y = (_boundary_signed_field(shape, X_low) >= 0.0).astype(np.int64)

    idx = rng.permutation(len(y))
    return X_low[idx], y[idx], boundary_points_low, true_curvature_low


def _apply_planar_transform(
    points: np.ndarray,
    *,
    center: tuple[float, float],
    scale: float = 1.0,
    rotation: float = 0.0,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32) * float(scale)
    c = float(np.cos(rotation))
    s = float(np.sin(rotation))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    pts = pts @ rot.T
    pts = pts + np.asarray(center, dtype=np.float32)[None, :]
    return pts.astype(np.float32)


def _inverse_planar_transform(
    points: np.ndarray,
    *,
    center: tuple[float, float],
    scale: float = 1.0,
    rotation: float = 0.0,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32) - np.asarray(center, dtype=np.float32)[None, :]
    c = float(np.cos(rotation))
    s = float(np.sin(rotation))
    rot_inv = np.asarray([[c, s], [-s, c]], dtype=np.float32)
    pts = pts @ rot_inv.T
    pts = pts / max(float(scale), 1e-6)
    return pts.astype(np.float32)


def _transformed_boundary_signed_field(
    shape: str,
    points: np.ndarray,
    *,
    center: tuple[float, float],
    scale: float = 1.0,
    rotation: float = 0.0,
) -> np.ndarray:
    local = _inverse_planar_transform(points, center=center, scale=scale, rotation=rotation)
    return _boundary_signed_field(shape, local).astype(np.float32)


def _multilabel_shape_specs(dataset: str) -> tuple[dict[str, object], ...]:
    if dataset == "overlap_triplet":
        return (
            {
                "label": "ring",
                "shape": "windowed_ripple_ring",
                "center": (0.0, 0.0),
                "scale": 1.0,
                "rotation": 0.0,
            },
            {
                "label": "tube",
                "shape": "serpentine_tube",
                "center": (-0.10, 0.08),
                "scale": 0.82,
                "rotation": 0.88,
            },
            {
                "label": "dumbbell",
                "shape": "corrugated_dumbbell",
                "center": (0.38, -0.22),
                "scale": 0.64,
                "rotation": -0.52,
            },
        )
    if dataset == "rough_overlap_triplet":
        return (
            {
                "label": "ring_a",
                "shape": "windowed_ripple_ring",
                "center": (-0.18, 0.02),
                "scale": 0.92,
                "rotation": 0.22,
            },
            {
                "label": "ring_b",
                "shape": "windowed_ripple_ring",
                "center": (0.34, -0.06),
                "scale": 0.74,
                "rotation": -0.68,
            },
            {
                "label": "tube",
                "shape": "serpentine_tube",
                "center": (0.10, 0.16),
                "scale": 0.74,
                "rotation": -0.94,
            },
        )
    if dataset == "rough_overlap_pentad":
        return (
            {
                "label": "ring_a",
                "shape": "windowed_ripple_ring",
                "center": (-0.20, 0.03),
                "scale": 0.90,
                "rotation": 0.18,
            },
            {
                "label": "ring_b",
                "shape": "windowed_ripple_ring",
                "center": (0.28, -0.10),
                "scale": 0.72,
                "rotation": -0.66,
            },
            {
                "label": "tube_a",
                "shape": "serpentine_tube",
                "center": (0.08, 0.20),
                "scale": 0.70,
                "rotation": -0.96,
            },
            {
                "label": "dumbbell",
                "shape": "corrugated_dumbbell",
                "center": (0.36, 0.26),
                "scale": 0.54,
                "rotation": 0.42,
            },
            {
                "label": "tube_b",
                "shape": "serpentine_tube",
                "center": (-0.30, -0.20),
                "scale": 0.56,
                "rotation": 0.52,
            },
        )
    raise ValueError(
        f"Unsupported multi-label dataset {dataset!r}. "
        "Choose 'overlap_triplet', 'rough_overlap_triplet', or 'rough_overlap_pentad'."
    )


def make_multilabel_shape_classification(
    dataset: str,
    *,
    n_samples: int = 2400,
    noise: float = 0.08,
    boundary_fraction: float = 0.60,
    n_boundary_points: int = 512,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Generate a 2D multi-label benchmark from overlapping transformed shapes."""
    rng = np.random.default_rng(random_state)
    specs = _multilabel_shape_specs(dataset)
    label_names = tuple(str(spec["label"]) for spec in specs)

    n_focus = int(round(n_samples * boundary_fraction))
    n_uniform = max(n_samples - n_focus, 0)
    per_label = max(n_focus // max(len(specs), 1), 1)
    focus_parts = []
    support_radius = 0.0
    for spec in specs:
        boundary_points, _ = _sample_shape_boundary(str(spec["shape"]), n_boundary_points)
        boundary_points = _apply_planar_transform(
            boundary_points,
            center=spec["center"],
            scale=float(spec["scale"]),
            rotation=float(spec["rotation"]),
        )
        idx = rng.integers(0, len(boundary_points), size=per_label)
        focus = boundary_points[idx] + rng.normal(scale=noise, size=(per_label, 2)).astype(np.float32)
        focus_parts.append(focus.astype(np.float32))
        support_radius = max(support_radius, float(np.max(np.abs(boundary_points))))

    X_focus = np.vstack(focus_parts).astype(np.float32)
    if len(X_focus) > n_focus:
        X_focus = X_focus[:n_focus]
    elif len(X_focus) < n_focus:
        extra_idx = rng.integers(0, len(X_focus), size=n_focus - len(X_focus))
        X_focus = np.vstack([X_focus, X_focus[extra_idx]]).astype(np.float32)

    support_radius += 0.45
    X_uniform = rng.uniform(
        low=-support_radius,
        high=support_radius,
        size=(n_uniform, 2),
    ).astype(np.float32)
    X_low = np.vstack([X_focus, X_uniform]).astype(np.float32)

    labels = []
    for spec in specs:
        signed = _transformed_boundary_signed_field(
            str(spec["shape"]),
            X_low,
            center=spec["center"],
            scale=float(spec["scale"]),
            rotation=float(spec["rotation"]),
        )
        labels.append((signed >= 0.0).astype(np.int64))
    Y = np.column_stack(labels).astype(np.int64)

    idx = rng.permutation(len(X_low))
    return X_low[idx], Y[idx], label_names


def make_embedded_multilabel_shape_benchmark(
    dataset: str = "overlap_triplet",
    *,
    observed_dim: int = 100,
    n_samples: int = 2400,
    noise: float = 0.08,
    boundary_fraction: float = 0.60,
    n_boundary_points: int = 512,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    random_state: int | None = None,
) -> EmbeddedMultiLabelBenchmark:
    """Create a high-D embedded multi-label benchmark from overlapping shapes."""
    X_low, Y, label_names = make_multilabel_shape_classification(
        dataset,
        n_samples=n_samples,
        noise=noise,
        boundary_fraction=boundary_fraction,
        n_boundary_points=n_boundary_points,
        random_state=random_state,
    )

    intrinsic_dim = X_low.shape[1]
    if observed_dim == intrinsic_dim:
        embedding = IdentityEmbedding(intrinsic_dim)
        X_high = X_low.astype(np.float32)
    else:
        X_high, embedding = embed_dataset_in_high_dimensional_space(
            X_low, target_dim=observed_dim, random_state=random_state
        )

    index = np.arange(len(Y))
    stratify = np.asarray([int("".join(map(str, row.tolist())), 2) for row in Y], dtype=np.int64)
    _, counts = np.unique(stratify, return_counts=True)
    if np.any(counts < 3):
        stratify = None
    train_idx, test_idx = train_test_split(
        index,
        test_size=test_fraction,
        random_state=random_state,
        stratify=stratify,
    )
    val_size = val_fraction / max(1.0 - test_fraction, 1e-8)
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size,
        random_state=random_state,
        stratify=None if stratify is None else stratify[train_idx],
    )

    def _split(ix: np.ndarray) -> EmbeddedMultiLabelSplit:
        return EmbeddedMultiLabelSplit(
            X_low=X_low[ix].astype(np.float32),
            X_high=X_high[ix].astype(np.float32),
            y=Y[ix].astype(np.int64),
        )

    return EmbeddedMultiLabelBenchmark(
        name=dataset,
        intrinsic_dim=intrinsic_dim,
        observed_dim=observed_dim,
        embedding=embedding,
        train=_split(train_idx),
        val=_split(val_idx),
        test=_split(test_idx),
        label_names=label_names,
    )


def make_embedded_boundary_classification_benchmark(
    dataset: str = "flower",
    *,
    observed_dim: int = 100,
    n_samples: int = 2400,
    noise: float = 0.08,
    boundary_fraction: float = 0.55,
    n_boundary_points: int = 1024,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    random_state: int | None = None,
) -> EmbeddedClassificationBenchmark:
    """Create an embedded boundary-defined classification benchmark."""
    X_low, y, boundary_points_low, true_curvature_low = make_boundary_classification(
        dataset,
        n_samples=n_samples,
        noise=noise,
        boundary_fraction=boundary_fraction,
        n_boundary_points=n_boundary_points,
        random_state=random_state,
    )

    intrinsic_dim = X_low.shape[1]
    if observed_dim == intrinsic_dim:
        embedding = IdentityEmbedding(intrinsic_dim)
        X_high = X_low.astype(np.float32)
    else:
        X_high, embedding = embed_dataset_in_high_dimensional_space(
            X_low, target_dim=observed_dim, random_state=random_state
        )

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_fraction,
        random_state=random_state,
        stratify=y,
    )
    val_size = val_fraction / max(1.0 - test_fraction, 1e-8)
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size,
        random_state=random_state,
        stratify=y[train_idx],
    )

    def _split(ix: np.ndarray) -> EmbeddedClassificationSplit:
        return EmbeddedClassificationSplit(
            X_low=X_low[ix].astype(np.float32),
            X_high=X_high[ix].astype(np.float32),
            y=y[ix].astype(np.int64),
        )

    return EmbeddedClassificationBenchmark(
        name=dataset,
        intrinsic_dim=intrinsic_dim,
        observed_dim=observed_dim,
        embedding=embedding,
        train=_split(train_idx),
        val=_split(val_idx),
        test=_split(test_idx),
        boundary_points_low=boundary_points_low.astype(np.float32),
        true_curvature_low=true_curvature_low.astype(np.float32),
    )
