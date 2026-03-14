"""Generic Jacobian and Hessian computation via torch.func.

Dimension-agnostic: works for any model f: R^n -> R^m.
Uses torch.func (jacrev, jacfwd, vmap) for efficient batched computation.
"""

import torch
from torch.func import jacrev, jacfwd, vmap, functional_call
from functools import partial


def _make_functional_call(model, x):
    """Helper: evaluate model as a pure function of input only."""
    return model(x.unsqueeze(0)).squeeze(0)


def jacobian(model: torch.nn.Module, x: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
    """Compute batched Jacobian df/dx at each point in x.

    Args:
        model: network f: R^n -> R^m
        x: [N, n] input points
        chunk_size: process this many points at a time (memory/speed tradeoff)

    Returns:
        [N, m, n] Jacobian tensor. For scalar output (m=1), shape is [N, 1, n].
    """
    model.eval()
    params = dict(model.named_parameters())

    def f_single(xi):
        return functional_call(model, params, xi.unsqueeze(0)).squeeze(0)

    jac_fn = vmap(jacrev(f_single))

    jacs = []
    for i in range(0, len(x), chunk_size):
        chunk = x[i:i + chunk_size]
        with torch.no_grad():
            j = jac_fn(chunk)
        jacs.append(j)

    return torch.cat(jacs, dim=0)


def hessian(model: torch.nn.Module, x: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """Compute batched Hessian d²f/dx² at each point in x.

    Only meaningful for scalar-output models (f: R^n -> R^1).
    For vector outputs, computes the Hessian of each output component separately.

    Args:
        model: network f: R^n -> R^1
        x: [N, n] input points
        chunk_size: process this many points at a time

    Returns:
        For scalar output: [N, n, n] Hessian tensor.
        For vector output: [N, m, n, n] tensor of per-output Hessians.
    """
    model.eval()
    params = dict(model.named_parameters())

    def f_scalar(xi):
        out = functional_call(model, params, xi.unsqueeze(0)).squeeze(0)
        return out.squeeze(-1) if out.dim() == 1 and out.shape[0] == 1 else out

    # jacrev of jacrev = Hessian. Use jacfwd for inner to avoid memory issues.
    hess_fn = vmap(jacrev(jacfwd(f_scalar)))

    hessians = []
    for i in range(0, len(x), chunk_size):
        chunk = x[i:i + chunk_size]
        with torch.no_grad():
            h = hess_fn(chunk)
        hessians.append(h)

    return torch.cat(hessians, dim=0)


def curvature_profile(model: torch.nn.Module, x: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """Compute Frobenius norm of Hessian at each point (scalar curvature measure).

    Note: for ReLU networks this is zero almost everywhere (piecewise linear).
    Use jacobian_variation() instead for ReLU networks.

    Args:
        model: scalar-output network f: R^n -> R^1
        x: [N, n] input points
        chunk_size: batch size for Hessian computation

    Returns:
        [N] tensor of ‖H(x)‖_F values.
    """
    H = hessian(model, x, chunk_size=chunk_size)
    # H is [N, n, n] for scalar output
    return torch.linalg.norm(H.flatten(start_dim=1), dim=1)


def jacobian_variation(
    model: torch.nn.Module,
    x: torch.Tensor,
    eps: float = 0.01,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Estimate local curvature via Jacobian variation in a neighborhood.

    For piecewise linear (ReLU) networks, the Hessian is zero almost everywhere.
    Instead, we estimate curvature by computing how much the Jacobian varies
    across small perturbations — this captures the density of linear region
    boundaries near each point.

    For each point x_i, computes:
        variation(x_i) = mean_d ||J(x_i + eps*e_d) - J(x_i)||_F

    where e_d are coordinate basis vectors.

    Args:
        model: the network
        x: [N, n] input points
        eps: perturbation size
        chunk_size: batch size for Jacobian computation

    Returns:
        [N] tensor of Jacobian variation values (effective curvature proxy).
    """
    n_in = x.shape[1]
    J_base = jacobian(model, x, chunk_size=chunk_size)  # [N, m, n]

    variation = torch.zeros(len(x), device=x.device)
    for d in range(n_in):
        x_pert = x.clone()
        x_pert[:, d] += eps
        J_pert = jacobian(model, x_pert, chunk_size=chunk_size)
        diff = (J_pert - J_base).flatten(start_dim=1)
        variation += torch.linalg.norm(diff, dim=1)

    return variation / (n_in * eps)  # normalize by step count and size


def relu_boundary_density(
    model: torch.nn.Module,
    x: torch.Tensor,
    eps: float = 0.01,
    n_perturbations: int = 16,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Count ReLU sign changes in a neighborhood to measure linear region boundary density.

    For a ReLU network, the function is piecewise linear. The "curvature" of the
    approximation is encoded in the density of linear region boundaries — i.e., how
    many ReLU neurons change activation sign when we perturb the input slightly.

    For each point x_i, we sample random perturbation directions, march along them
    by eps, and count how many ReLU pre-activations change sign. The average count
    across directions gives the boundary density.

    Args:
        model: ReLU network (must use nn.ReLU or nn.Sequential with ReLU layers)
        x: [N, n] input points
        eps: perturbation radius
        n_perturbations: number of random directions to sample per point
        chunk_size: batch size for forward passes

    Returns:
        [N] tensor of boundary density values (average sign changes per perturbation).
    """
    model.eval()
    relu_layers = _find_relu_layer_indices(model)
    if not relu_layers:
        raise ValueError("No ReLU layers found in model. Use jacobian_variation() instead.")

    N = len(x)
    density = torch.zeros(N, device=x.device)

    for _ in range(n_perturbations):
        # Random unit direction
        direction = torch.randn_like(x)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        x_pert = x + eps * direction

        # Count sign changes across all ReLU layers
        for i in range(0, N, chunk_size):
            chunk = x[i:i + chunk_size]
            chunk_pert = x_pert[i:i + chunk_size]
            signs_base = _relu_preact_signs(model, chunk, relu_layers)
            signs_pert = _relu_preact_signs(model, chunk_pert, relu_layers)

            # Count neurons that changed sign (each = one boundary crossed)
            changes = (signs_base != signs_pert).float().sum(dim=-1)  # [batch]
            density[i:i + chunk_size] += changes

    return density / n_perturbations


def _find_relu_layer_indices(model: torch.nn.Module) -> list[int]:
    """Find indices of layers just before ReLU in model.layers (Sequential)."""
    layers = None
    if hasattr(model, 'layers') and isinstance(model.layers, torch.nn.Sequential):
        layers = model.layers
    elif isinstance(model, torch.nn.Sequential):
        layers = model
    else:
        raise ValueError("Expected model with .layers Sequential or a Sequential model")

    indices = []
    for i, layer in enumerate(layers):
        if isinstance(layer, torch.nn.ReLU):
            indices.append(i - 1)  # the Linear layer before ReLU
    return indices


def _relu_preact_signs(
    model: torch.nn.Module,
    x: torch.Tensor,
    relu_layer_indices: list[int],
) -> torch.Tensor:
    """Get pre-activation signs at all ReLU layers. Returns [batch, total_neurons]."""
    layers = model.layers if hasattr(model, 'layers') else model
    signs = []

    h = x
    for i, layer in enumerate(layers):
        h = layer(h)
        if i in relu_layer_indices:
            # h is now the LINEAR output (pre-ReLU), next layer is ReLU
            signs.append((h > 0).float())

    if not signs:
        return torch.zeros(len(x), 0, device=x.device)
    return torch.cat(signs, dim=-1)


def gradient(model: torch.nn.Module, x: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    """Compute gradient of scalar-output model at each point.

    More efficient than full Jacobian when output is scalar.

    Args:
        model: scalar-output network f: R^n -> R^1
        x: [N, n] input points
        chunk_size: batch size

    Returns:
        [N, n] gradient tensor.
    """
    grads = []
    for i in range(0, len(x), chunk_size):
        chunk = x[i:i + chunk_size].detach().requires_grad_(True)
        out = model(chunk)
        if out.dim() == 2:
            out = out.sum(dim=-1)  # [B] for scalar output
        g = torch.autograd.grad(out.sum(), chunk, create_graph=False)[0]
        grads.append(g.detach())
    return torch.cat(grads, dim=0)


def verify_jacobian(
    model: torch.nn.Module,
    x: torch.Tensor,
    eps: float = 1e-4,
    n_check: int = 10,
) -> dict:
    """Verify Jacobian computation against finite differences.

    Args:
        model: the network
        x: input points (will check first n_check points)
        eps: finite difference step size
        n_check: number of points to check

    Returns:
        dict with 'max_error', 'mean_error', 'analytic', 'numerical' tensors.
    """
    model.eval()
    x_check = x[:n_check].detach()
    n_in = x_check.shape[1]

    # Analytic Jacobian
    J_analytic = jacobian(model, x_check)

    # Finite difference Jacobian
    J_numerical = torch.zeros_like(J_analytic)
    for d in range(n_in):
        x_plus = x_check.clone()
        x_minus = x_check.clone()
        x_plus[:, d] += eps
        x_minus[:, d] -= eps
        with torch.no_grad():
            f_plus = model(x_plus)
            f_minus = model(x_minus)
        J_numerical[:, :, d] = (f_plus - f_minus) / (2 * eps)

    error = (J_analytic - J_numerical).abs()
    return {
        'max_error': error.max().item(),
        'mean_error': error.mean().item(),
        'analytic': J_analytic,
        'numerical': J_numerical,
    }
