"""
Aleph: Quantization-aware training and learned correction layers.
"""

# Submodules
from aleph import datasets, models, quantization, training, visualization

# Models
from aleph.models import (
    MLP,
    MLPWithCorrection,
    MLPWithLearnedCorrection,
    AutoencoderWithCorrection,
    TransformerWithCorrection,
    CorrectionNet,
    CorrectionMLP,
)

# Training
from aleph.training import train_with_qat

# Quantization utilities
from aleph.quantization import (
    get_quant_range,
    fake_quantize,
    fake_quantize_with_error,
    calibrate_model,
    compute_scales,
)

# Datasets
from aleph.datasets import (
    make_spirals,
    embed_dataset_in_high_dimensional_space,
    load_mnist_flat,
    load_shakespeare,
)

# Visualization
from aleph.visualization import (
    plot_decision_boundary,
    QuantizedWrapper,
    OracleWrapper,
    LearnedCorrectionWrapper,
)

__all__ = [
    # Submodules
    "datasets",
    "models",
    "quantization",
    "training",
    "visualization",
    # Models
    "MLP",
    "MLPWithCorrection",
    "MLPWithLearnedCorrection",
    "AutoencoderWithCorrection",
    "TransformerWithCorrection",
    "CorrectionNet",
    "CorrectionMLP",
    # Training
    "train_with_qat",
    # Quantization
    "get_quant_range",
    "fake_quantize",
    "fake_quantize_with_error",
    "calibrate_model",
    "compute_scales",
    # Datasets
    "make_spirals",
    "embed_dataset_in_high_dimensional_space",
    "load_mnist_flat",
    "load_shakespeare",
    # Visualization
    "plot_decision_boundary",
    "QuantizedWrapper",
    "OracleWrapper",
    "LearnedCorrectionWrapper",
]
