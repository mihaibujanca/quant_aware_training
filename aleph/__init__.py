"""
Aleph: neural-network geometry research utilities.

The package currently spans two connected lines of work:
- quantization-aware correction and quantization geometry
- geometric compression of trained networks
- high-dimensional-first ReLU compression
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
    EmbeddedClassificationBenchmark,
    EmbeddedClassificationSplit,
    EmbeddedMultiLabelBenchmark,
    EmbeddedMultiLabelSplit,
    IdentityEmbedding,
    make_boundary_classification,
    make_embedded_boundary_classification_benchmark,
    make_embedded_multilabel_shape_benchmark,
    make_multilabel_shape_classification,
    make_rough_hypershell,
    make_rough_spirals,
    make_spirals,
    make_embedded_classification_benchmark,
    embed_dataset_in_high_dimensional_space,
    load_mnist_flat,
    load_shakespeare,
)

from aleph.boundary_geometric_compression import (
    BoundaryCompressionConfig,
    TeacherBoundaryProfile,
    BoundaryChunk,
    HybridBoundaryModel,
    extract_teacher_projection,
    build_teacher_boundary_profile,
    build_hybrid_boundary_model,
    run_single_boundary_compression,
    run_boundary_compression_program,
    save_boundary_compression_results,
)

from aleph.relu_geometric_compression import (
    HighDReluCompressionConfig,
    RegionAtlas,
    MacroRegionAtlas,
    build_region_atlas,
    coarsen_region_atlas,
    derive_target_architecture,
    run_single_high_d_compression,
    run_high_d_first_program,
    save_high_d_program_results,
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
    "EmbeddedClassificationBenchmark",
    "EmbeddedClassificationSplit",
    "EmbeddedMultiLabelBenchmark",
    "EmbeddedMultiLabelSplit",
    "IdentityEmbedding",
    "make_boundary_classification",
    "make_embedded_boundary_classification_benchmark",
    "make_embedded_multilabel_shape_benchmark",
    "make_multilabel_shape_classification",
    "make_rough_hypershell",
    "make_rough_spirals",
    "make_spirals",
    "make_embedded_classification_benchmark",
    "embed_dataset_in_high_dimensional_space",
    "load_mnist_flat",
    "load_shakespeare",
    # Boundary compression
    "BoundaryCompressionConfig",
    "TeacherBoundaryProfile",
    "BoundaryChunk",
    "HybridBoundaryModel",
    "extract_teacher_projection",
    "build_teacher_boundary_profile",
    "build_hybrid_boundary_model",
    "run_single_boundary_compression",
    "run_boundary_compression_program",
    "save_boundary_compression_results",
    # High-D ReLU compression
    "HighDReluCompressionConfig",
    "RegionAtlas",
    "MacroRegionAtlas",
    "build_region_atlas",
    "coarsen_region_atlas",
    "derive_target_architecture",
    "run_single_high_d_compression",
    "run_high_d_first_program",
    "save_high_d_program_results",
    # Visualization
    "plot_decision_boundary",
    "QuantizedWrapper",
    "OracleWrapper",
    "LearnedCorrectionWrapper",
]
