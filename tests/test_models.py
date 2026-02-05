"""Tests for CorrectionNet bug fix and consolidated model APIs."""

import torch
import torch.nn as nn

from aleph.models import (
    CorrectionNet,
    CorrectionMLP,
    MLPWithCorrection,
    TransformerWithCorrection,
    AutoencoderWithCorrection,
)
from aleph.quantization import get_quant_range


class TestCorrectionNetBugFix:
    """Verify that CorrectionNet(hidden_size=0, quantize=True) actually quantizes output."""

    def test_linear_quantize_true_differs_from_float(self):
        """The core bug: linear corrections must quantize when quantize=True."""
        torch.manual_seed(42)
        net = CorrectionNet(size=64, hidden_size=0)

        # Train it slightly so weights are nonzero
        nn.init.normal_(net.layers.weight, std=0.1)
        nn.init.normal_(net.layers.bias, std=0.1)

        error = torch.randn(8, 64)

        # Calibrate
        net.calibrate(error, num_bits=4)
        assert 'output' in net._scales

        out_float = net(error, quantize=False)
        out_quant = net(error, quantize=True, num_bits=4)

        # Quantized output should differ from float (the bug was they were identical)
        assert not torch.allclose(out_float, out_quant), \
            "Bug: linear CorrectionNet quantize=True produces same output as quantize=False"

    def test_mlp_quantize_true_quantizes_hidden_and_output(self):
        """MLP corrections should quantize both hidden and output."""
        torch.manual_seed(42)
        net = CorrectionNet(size=64, hidden_size=32)

        # Give non-zero weights so output is non-trivial
        nn.init.normal_(net.layers[2].weight, std=0.1)
        nn.init.normal_(net.layers[2].bias, std=0.1)

        error = torch.randn(8, 64)

        net.calibrate(error, num_bits=4)
        assert 'hidden' in net._scales
        assert 'output' in net._scales

        out_float = net(error, quantize=False)
        out_quant = net(error, quantize=True, num_bits=4)

        assert not torch.allclose(out_float, out_quant), \
            "MLP CorrectionNet quantize=True should differ from quantize=False"

    def test_quantize_false_never_quantizes(self):
        """Control: quantize=False should never apply fake_quantize."""
        torch.manual_seed(42)

        for hidden_size in [0, 32]:
            net = CorrectionNet(size=64, hidden_size=hidden_size)
            error = torch.randn(8, 64)

            # Even with scales set, quantize=False should give pure float output
            net.calibrate(error, num_bits=4)
            out1 = net(error, quantize=False)
            net._scales = {}
            out2 = net(error, quantize=False)

            assert torch.allclose(out1, out2), \
                f"quantize=False should ignore _scales (hidden_size={hidden_size})"

    def test_correction_mlp_is_correction_net(self):
        """CorrectionMLP should be an alias for CorrectionNet."""
        assert CorrectionMLP is CorrectionNet


class TestMLPWithCorrection:
    """Test the consolidated MLPWithCorrection."""

    def _make_model(self):
        return MLPWithCorrection(
            input_size=10, hidden_size=32, output_size=2, depth=4,
            correction_every_n=2, correction_hidden=16,
        )

    def test_forward(self):
        model = self._make_model()
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 2)

    def test_calibrate_and_forward_quantized(self):
        model = self._make_model()
        x = torch.randn(8, 10)
        scales = model.calibrate(x, num_bits=8)
        assert len(scales) == 4  # one per backbone layer

        out = model.forward_quantized(x, scales, num_bits=8)
        assert out.shape == (8, 2)

    def test_forward_quantized_with_zero_points_none(self):
        """Experiment-style call: zero_points=None should work."""
        model = self._make_model()
        x = torch.randn(8, 10)
        scales = model.calibrate(x, num_bits=8)

        out = model.forward_quantized(x, scales, zero_points=None, num_bits=8)
        assert out.shape == (8, 2)

    def test_forward_with_correction(self):
        model = self._make_model()
        x = torch.randn(8, 10)
        scales = model.calibrate(x, num_bits=8)

        out = model.forward_with_correction(x, scales, num_bits=8)
        assert out.shape == (8, 2)

    def test_forward_with_correction_returns_intermediates(self):
        model = self._make_model()
        x = torch.randn(8, 10)
        scales = model.calibrate(x, num_bits=8)

        out, intermediates = model.forward_with_correction(
            x, scales, num_bits=8, return_intermediates=True,
        )
        assert out.shape == (8, 2)
        assert len(intermediates) == 2  # correction at layer 1 and 3

    def test_corrections_property(self):
        model = self._make_model()
        assert model.corrections is model.correction_layers

    def test_get_float_activations(self):
        model = self._make_model()
        x = torch.randn(8, 10)
        acts, logits = model.get_float_activations(x)
        assert logits.shape == (8, 2)
        assert len(acts) == 2

    def test_calibrate_corrections(self):
        model = self._make_model()
        x = torch.randn(8, 10)
        scales = model.calibrate(x, num_bits=4)
        model.calibrate_corrections(x, scales, num_bits=4)

        for corr in model.corrections.values():
            assert 'output' in corr._scales

    def test_backward_compat_alias(self):
        """forward_quantized_with_correction should work."""
        model = self._make_model()
        x = torch.randn(8, 10)
        scales = model.calibrate(x, num_bits=8)
        out = model.forward_quantized_with_correction(x, scales, num_bits=8)
        assert out.shape == (8, 2)

    def test_layers_attribute_for_legacy(self):
        """self.layers should contain backbone + head for calibrate_quantization compat."""
        model = self._make_model()
        assert len(model.layers) == 5  # 4 backbone + 1 head


class TestTransformerWithCorrection:
    """Test the consolidated TransformerWithCorrection."""

    def _make_model(self):
        return TransformerWithCorrection(
            vocab_size=50, d_model=32, n_heads=2, n_layers=2,
            d_ff=64, max_seq_len=16, dropout=0.0,
            correction_every_n=2, correction_hidden=16,
        )

    def test_forward(self):
        model = self._make_model()
        x = torch.randint(0, 50, (4, 16))
        out = model(x)
        assert out.shape == (4, 16, 50)

    def test_forward_with_correction(self):
        model = self._make_model()
        x = torch.randint(0, 50, (4, 16))
        scales = model.calibrate(x, num_bits=8)
        assert len(scales) == 4  # 2 layers * 2 quant points

        out = model.forward_with_correction(x, scales, num_bits=8)
        assert out.shape == (4, 16, 50)

    def test_corrections_property(self):
        model = self._make_model()
        assert model.corrections is model.correction_layers


class TestAutoencoderWithCorrection:
    """Test the consolidated AutoencoderWithCorrection."""

    def _make_model(self):
        return AutoencoderWithCorrection(
            input_size=28, hidden_sizes=[16, 8], latent_size=4,
            correction_every_n=1, correction_hidden=8,
        )

    def test_forward(self):
        model = self._make_model()
        x = torch.randn(4, 28)
        out = model(x)
        assert out.shape == (4, 28)

    def test_forward_with_correction(self):
        model = self._make_model()
        x = torch.randn(4, 28)
        scales = model.calibrate(x, num_bits=8)

        out = model.forward_with_correction(x, scales, num_bits=8)
        assert out.shape == (4, 28)

    def test_corrections_property(self):
        model = self._make_model()
        assert model.corrections is model.correction_layers

    def test_backward_compat_alias(self):
        """forward_quantized_with_correction should work."""
        model = self._make_model()
        x = torch.randn(4, 28)
        scales = model.calibrate(x, num_bits=8)
        out = model.forward_quantized_with_correction(x, scales, num_bits=8)
        assert out.shape == (4, 28)
