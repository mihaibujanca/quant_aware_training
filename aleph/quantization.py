import copy
import torch
import torch.nn as nn
from torch.ao.quantization import convert, prepare_qat


# =============================================================================
# Bit-width constants
# =============================================================================

SUPPORTED_BITS = {
    2: (-2, 1),
    4: (-8, 7),
    8: (-128, 127),
}


def get_quant_range(num_bits):
    """Get (quant_min, quant_max) for a given bit width."""
    if num_bits not in SUPPORTED_BITS:
        raise ValueError(f"Unsupported num_bits: {num_bits}. Supported: {list(SUPPORTED_BITS.keys())}")
    return SUPPORTED_BITS[num_bits]


# =============================================================================
# Fake quantization
# =============================================================================


class FakeQuantizeWithRounding(torch.ao.quantization.FakeQuantize):
    """FakeQuantize module with configurable rounding mode for QAT training."""

    def __init__(self, rounding_mode="nearest", **kwargs):
        super().__init__(**kwargs)
        self.rounding_mode = rounding_mode

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())

        if self.fake_quant_enabled[0] == 1:
            scale, zero_point = self.calculate_qparams()
            X = fake_quantize_with_rounding(
                X, scale, zero_point,
                self.quant_min, self.quant_max,
                self.rounding_mode,
                self.qscheme,
                self.ch_axis,
            )
        return X


def _reshape_qparams_for_broadcast(x, scale, zero_point, qscheme, ch_axis):
    if qscheme in (torch.per_channel_affine, torch.per_channel_symmetric):
        view_shape = [1] * x.dim()
        view_shape[ch_axis] = -1
        scale = scale.reshape(view_shape)
        zero_point = zero_point.reshape(view_shape)
    return scale, zero_point


def fake_quantize_with_rounding(x, scale, zero_point, quant_min, quant_max, rounding_mode, qscheme, ch_axis):
    """Fake quantize with custom rounding mode (differentiable via straight-through estimator)."""
    # Scale to quantized domain
    scale, zero_point = _reshape_qparams_for_broadcast(x, scale, zero_point, qscheme, ch_axis)
    x_scaled = x / scale + zero_point.float()

    # Apply rounding
    if rounding_mode == "nearest":
        x_rounded = torch.round(x_scaled)
    elif rounding_mode == "floor":
        x_rounded = torch.floor(x_scaled)
    elif rounding_mode == "ceil":
        x_rounded = torch.ceil(x_scaled)
    else:
        raise ValueError(f"Unknown rounding mode: {rounding_mode}")

    # Straight-through estimator: use rounded value in forward, but gradient flows through as if no rounding
    x_rounded = x_scaled + (x_rounded - x_scaled).detach()

    # Clamp to valid range
    x_clamped = torch.clamp(x_rounded, quant_min, quant_max)

    # Dequantize back to float
    x_dequant = (x_clamped - zero_point.float()) * scale

    return x_dequant


def fake_quantize(x, scale, zero_point, num_bits=8):
    """Fake quantize: quantize then dequantize (stays in float32)."""
    quant_min, quant_max = get_quant_range(num_bits)

    x_scaled = x / scale + zero_point
    x_rounded = torch.round(x_scaled)
    x_clamped = torch.clamp(x_rounded, quant_min, quant_max)
    x_dequant = (x_clamped - zero_point) * scale

    return x_dequant


def fake_quantize_with_error(x, scale, zero_point, num_bits=8):
    """Fake quantize with STE and return quantization error (pre - post)."""
    quant_min, quant_max = get_quant_range(num_bits)

    x_scaled = x / scale + zero_point
    x_rounded = torch.round(x_scaled)
    x_ste = x_scaled + (x_rounded - x_scaled).detach()
    x_clamped = torch.clamp(x_ste, quant_min, quant_max)
    x_dequant = (x_clamped - zero_point) * scale
    error = x - x_dequant

    return x_dequant, error


def compute_scales(activations, num_bits=8):
    """
    Compute scale factors and zero points from activation tensors.

    Args:
        activations: List of activation tensors at quantization points
        num_bits: Bit width for quantization

    Returns:
        scale_factors: List of scale factors (one per activation)
        zero_points: List of zero points (all 0.0 for symmetric quantization)
    """
    _, quant_max = get_quant_range(num_bits)

    scale_factors = []
    zero_points = []

    for acts in activations:
        abs_max = max(abs(acts.min().item()), abs(acts.max().item()))
        scale = abs_max / quant_max if abs_max > 0 else 1.0
        scale_factors.append(scale)
        zero_points.append(0.0)

    return scale_factors, zero_points


def calibrate_model(model, calibration_data, num_bits=8):
    """
    Generic calibration for models with get_quant_activations() method.

    Args:
        model: Model with get_quant_activations(x) method
        calibration_data: Input data for calibration
        num_bits: Bit width for quantization

    Returns:
        scale_factors, zero_points
    """
    model.eval()
    with torch.no_grad():
        activations = model.get_quant_activations(calibration_data)
    return compute_scales(activations, num_bits)


def calibrate_quantization(model, calibration_data, num_bits=8):
    """
    Calibrate scale/zero_point for MLP models (legacy, for backwards compatibility).

    Returns lists of scale factors and zero points, one per hidden layer.
    """
    model.eval()
    activations = []

    with torch.no_grad():
        x = calibration_data
        for layer in model.layers[:-1]:
            x = model.relu(layer(x))
            activations.append(x.clone())

    return compute_scales(activations, num_bits)


def prepare_qat_with_rounding(model, rounding_mode="nearest", backend="qnnpack", bits=8):
    """
    Prepare a model for QAT with a specific rounding mode.

    This patches the FakeQuantize modules to use custom rounding during training,
    so the model learns to be robust to that specific rounding behavior.

    Args:
        model: The model to prepare for QAT
        rounding_mode: "nearest", "floor", or "ceil"
        backend: Quantization backend ("qnnpack" for ARM, "x86" for Intel)
        bits: Bit width for quantization (4 or 8)

    Returns:
        The prepared model (modified in place)
    """
    from torch.ao.quantization import QConfig
    from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

    torch.backends.quantized.engine = backend
    quant_min, quant_max = get_quant_range(bits)

    model.qconfig = QConfig(
        activation=FakeQuantizeWithRounding.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=quant_min, quant_max=quant_max,
            dtype=torch.qint8, qscheme=torch.per_tensor_affine,
            rounding_mode=rounding_mode,
        ),
        weight=FakeQuantizeWithRounding.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=quant_min, quant_max=quant_max,
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric,
            rounding_mode=rounding_mode,
        ),
    )
    model.train()
    prepare_qat(model, inplace=True)

    return model


def _replace_fake_quantize_modules(module, rounding_mode):
    """Recursively replace FakeQuantize modules with custom rounding versions."""
    for name, child in module.named_children():
        if isinstance(child, torch.ao.quantization.FakeQuantize):
            # Create new FakeQuantize with custom rounding, copying settings
            new_fq = FakeQuantizeWithRounding(
                rounding_mode=rounding_mode,
                observer=child.activation_post_process.__class__,
                quant_min=child.quant_min,
                quant_max=child.quant_max,
                dtype=child.dtype,
                qscheme=child.qscheme,
            )
            # Copy observer state
            new_fq.activation_post_process.load_state_dict(
                child.activation_post_process.state_dict()
            )
            setattr(module, name, new_fq)
        else:
            _replace_fake_quantize_modules(child, rounding_mode)

    # Also check for fake_quant attributes on Linear layers
    if hasattr(module, 'weight_fake_quant') and isinstance(module.weight_fake_quant, torch.ao.quantization.FakeQuantize):
        old_fq = module.weight_fake_quant
        new_fq = FakeQuantizeWithRounding(
            rounding_mode=rounding_mode,
            observer=old_fq.activation_post_process.__class__,
            quant_min=old_fq.quant_min,
            quant_max=old_fq.quant_max,
            dtype=old_fq.dtype,
            qscheme=old_fq.qscheme,
        )
        new_fq.activation_post_process.load_state_dict(
            old_fq.activation_post_process.state_dict()
        )
        module.weight_fake_quant = new_fq

    if hasattr(module, 'activation_post_process') and isinstance(module.activation_post_process, torch.ao.quantization.FakeQuantize):
        old_fq = module.activation_post_process
        new_fq = FakeQuantizeWithRounding(
            rounding_mode=rounding_mode,
            observer=old_fq.activation_post_process.__class__,
            quant_min=old_fq.quant_min,
            quant_max=old_fq.quant_max,
            dtype=old_fq.dtype,
            qscheme=old_fq.qscheme,
        )
        new_fq.activation_post_process.load_state_dict(
            old_fq.activation_post_process.state_dict()
        )
        module.activation_post_process = new_fq


def quantize_model(model, inplace: bool = False, rounding_mode: str = "nearest"):
    """
    Convert a QAT-prepared model to a quantized model with configurable rounding.

    Args:
        model: A QAT-prepared model (after prepare_qat and training)
        inplace: If True, modify the model in place. If False, work on a copy.
        rounding_mode: Rounding mode for quantization - "nearest", "floor", or "ceil"

    Returns:
        The quantized model
    """
    if rounding_mode not in ("nearest", "floor", "ceil"):
        raise ValueError(f"rounding_mode must be 'nearest', 'floor', or 'ceil', got '{rounding_mode}'")

    if not inplace:
        model = copy.deepcopy(model)

    # For "nearest" rounding, use PyTorch's default convert
    if rounding_mode == "nearest":
        return convert(model, inplace=True)

    # For custom rounding modes, we need to manually re-quantize after conversion
    # First, save the original float weights and their quantization parameters
    float_weights = {}
    quant_params = {}
    _extract_weights_and_params(model, "", float_weights, quant_params)

    # Convert using default PyTorch conversion
    quantized_model = convert(model, inplace=True)

    # Re-quantize weights with custom rounding mode
    _apply_custom_rounding(quantized_model, "", float_weights, quant_params, rounding_mode)

    return quantized_model


def _extract_weights_and_params(module, prefix, float_weights, quant_params):
    """Extract floating point weights and quantization parameters before conversion."""
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # Check if this is a Linear layer with weight_fake_quant (QAT prepared)
        if isinstance(child, nn.Linear) and hasattr(child, "weight_fake_quant"):
            float_weights[full_name] = child.weight.detach().clone()
            if child.bias is not None:
                float_weights[f"{full_name}.bias"] = child.bias.detach().clone()

            # Get quantization parameters from the fake quant module
            wfq = child.weight_fake_quant
            scale, zero_point = wfq.calculate_qparams()
            quant_params[full_name] = {
                "scale": scale,
                "zero_point": zero_point,
                "quant_min": wfq.quant_min,
                "quant_max": wfq.quant_max,
                "dtype": wfq.dtype,
                "qscheme": wfq.qscheme,
                "ch_axis": wfq.ch_axis,
            }

        _extract_weights_and_params(child, full_name, float_weights, quant_params)


def _apply_custom_rounding(module, prefix, float_weights, quant_params, rounding_mode):
    """Apply custom rounding to quantized linear layers."""
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # Check if this is a quantized linear module (torch.ao.nn.quantized.modules.linear.Linear)
        is_quantized_linear = hasattr(child, "set_weight_bias") and full_name in float_weights
        if is_quantized_linear:
            params = quant_params[full_name]
            weight = float_weights[full_name]

            # Quantize weight with custom rounding
            quantized_weight = _quantize_tensor(
                weight,
                params["scale"],
                params["zero_point"],
                params["quant_min"],
                params["quant_max"],
                rounding_mode,
                params["qscheme"],
                params["ch_axis"],
            )

            # Get bias if present
            bias = float_weights.get(f"{full_name}.bias")

            # Update the packed parameters with re-quantized weights
            child.set_weight_bias(quantized_weight, bias)

        _apply_custom_rounding(child, full_name, float_weights, quant_params, rounding_mode)


def _quantize_tensor(tensor, scale, zero_point, quant_min, quant_max, rounding_mode, qscheme, ch_axis):
    """Quantize a tensor with custom rounding mode, returning a quantized tensor."""
    # Scale the tensor
    scale, zero_point = _reshape_qparams_for_broadcast(tensor, scale, zero_point, qscheme, ch_axis)
    scaled = tensor / scale + zero_point.float()

    # Apply rounding mode
    if rounding_mode == "nearest":
        rounded = torch.round(scaled)
    elif rounding_mode == "floor":
        rounded = torch.floor(scaled)
    elif rounding_mode == "ceil":
        rounded = torch.ceil(scaled)

    # Clamp to valid range
    clamped = torch.clamp(rounded, quant_min, quant_max)

    dequantized = (clamped - zero_point.float()) * scale

    # Convert to quantized tensor
    if qscheme in (torch.per_channel_affine, torch.per_channel_symmetric):
        return torch.quantize_per_channel(
            dequantized,
            scale.reshape(-1).float(),
            zero_point.reshape(-1).to(torch.int64),
            ch_axis,
            torch.qint8,
        )

    return torch.quantize_per_tensor(
        dequantized,
        scale.item(),
        zero_point.item(),
        torch.qint8,
    )
