"""Mesh loading and SDF point sampling for geometric compression experiments.

Provides utilities to load 3D meshes, normalize them, and sample points with
signed distance function (SDF) labels for training implicit neural representations.
"""

import torch
import trimesh
import numpy as np


def load_mesh(name: str = "torus") -> trimesh.Trimesh:
    """Load a named mesh and normalize to unit cube centered at origin.

    Args:
        name: mesh name. Built-in options:
              - "torus": torus (good SDF test — convex + concave regions)
              - "sphere": icosphere (simplest test case)
              - "capsule": capsule shape
              Otherwise treated as a file path to a .obj/.ply/.stl file.

    Returns:
        Normalized trimesh.Trimesh object.
    """
    procedural = {
        "torus": lambda: trimesh.creation.torus(major_radius=0.3, minor_radius=0.1, major_sections=64, minor_sections=32),
        "sphere": lambda: trimesh.creation.icosphere(subdivisions=3, radius=0.4),
        "capsule": lambda: trimesh.creation.capsule(height=0.5, radius=0.2, count=[32, 32]),
        "box": lambda: trimesh.creation.box(extents=(0.6, 0.6, 0.6)),
        "cylinder": lambda: trimesh.creation.cylinder(radius=0.25, height=0.6, sections=64),
    }
    if name in procedural:
        mesh = procedural[name]()
    else:
        mesh = trimesh.load(name)
        if hasattr(mesh, 'dump'):  # Scene from multi-mesh file
            mesh = mesh.dump(concatenate=True)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a Trimesh, got {type(mesh)}. Scene files not supported.")

    # Normalize to [-1, 1]^3 centered at origin
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    scale = np.abs(mesh.vertices).max()
    mesh.vertices /= scale

    return mesh


def sample_sdf_points(
    mesh: trimesh.Trimesh,
    n_points: int = 100_000,
    surface_ratio: float = 0.5,
    surface_noise: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample points with signed distance labels from a mesh.

    Sampling strategy:
    - surface_ratio fraction: points sampled ON the mesh surface + Gaussian noise
    - remaining fraction: uniform random in bounding volume (with margin)

    Args:
        mesh: normalized trimesh object (in [-1, 1]^3 after load_mesh)
        n_points: total number of points to sample
        surface_ratio: fraction of points near the surface (0 to 1)
        surface_noise: std of Gaussian noise added to surface points

    Returns:
        points: [N, 3] float32 tensor of 3D coordinates
        sdf: [N, 1] float32 tensor of signed distance values
    """
    n_surface = int(n_points * surface_ratio)
    n_volume = n_points - n_surface

    # Near-surface points: sample on surface then perturb
    surface_pts, _ = trimesh.sample.sample_surface(mesh, n_surface)
    surface_pts += np.random.randn(*surface_pts.shape) * surface_noise

    # Volume points: uniform in a cube that covers the mesh with margin.
    # Use max extent across all axes (not per-axis) so the cube matches the isosurface
    # evaluation grid — avoids out-of-distribution extrapolation in flat meshes (e.g. torus).
    vmin = mesh.vertices.min(axis=0)
    vmax = mesh.vertices.max(axis=0)
    center = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin).max() * 1.2  # isotropic cube
    lo, hi = center - half, center + half
    volume_pts = np.random.uniform(lo, hi, size=(n_volume, 3)).astype(np.float32)

    all_pts = np.concatenate([surface_pts, volume_pts], axis=0).astype(np.float32)

    # Compute signed distance
    sdf_values = _signed_distance(mesh, all_pts)

    points = torch.from_numpy(all_pts)
    sdf = torch.from_numpy(sdf_values).unsqueeze(-1)

    return points, sdf


def make_embedding_matrix(intrinsic_dim: int, embed_dim: int, seed: int = 42) -> torch.Tensor:
    """Orthogonal random projection [embed_dim, intrinsic_dim] via QR of random Gaussian.

    Maps intrinsic_dim-dimensional points into embed_dim-dimensional space.
    Preserves inner products up to scaling (embed_dim >= intrinsic_dim required for isometry).

    Args:
        intrinsic_dim: input dimensionality (e.g. 3 for 3D mesh points)
        embed_dim: output dimensionality (must be >= intrinsic_dim)
        seed: random seed for reproducibility

    Returns:
        A: [embed_dim, intrinsic_dim] orthonormal projection matrix (as float32 tensor)
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    G = torch.randn(embed_dim, intrinsic_dim, generator=rng)  # [embed_dim, intrinsic_dim]
    Q, _ = torch.linalg.qr(G)    # Q: [embed_dim, intrinsic_dim] with orthonormal cols
    return Q.float()              # [embed_dim, intrinsic_dim], Q.T @ Q = I_intrinsic


def _signed_distance(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Compute signed distance from points to mesh surface.

    Uses trimesh proximity for unsigned distance and winding number for sign.
    Negative = inside, positive = outside (standard SDF convention).

    Args:
        mesh: the mesh
        points: [N, 3] array

    Returns:
        [N] float32 array of signed distances
    """
    # Closest point distances (unsigned) — uses BVH tree, O(N log F)
    closest_pts, distances, face_ids = trimesh.proximity.closest_point(mesh, points)

    # Sign via face normal dot product: much faster than mesh.contains() (winding number)
    # Direction from surface to query point; positive dot = outside (normal points outward)
    face_normals = mesh.face_normals[face_ids]          # [N, 3] outward normals
    direction = points - closest_pts                     # [N, 3] surface→query
    dot = (direction * face_normals).sum(axis=-1)        # [N]
    signs = np.where(dot >= 0, 1.0, -1.0)

    return (signs * distances).astype(np.float32)
