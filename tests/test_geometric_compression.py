import unittest

import torch

from aleph.geometric_compression import jacobian_rank_profile


class BilinearPlusLinear(torch.nn.Module):
    """f(x) = x0*x1 + x2 => grad = [x1, x0, 1, 0, ...] (rank 3 ensemble)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x[:, 0] * x[:, 1] + x[:, 2]
        return y.unsqueeze(-1)


class JacobianRankProfileTests(unittest.TestCase):
    def test_constant_gradient_has_rank_one(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(5, 1, bias=True)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, -2.0, 0.5, 0.0, 3.0]]))
            model.bias.zero_()

        x = torch.randn(256, 5)
        prof = jacobian_rank_profile(model, x, chunk_size=64)

        self.assertEqual(prof["input_dim"], 5)
        self.assertEqual(prof["n_samples"], 256)
        self.assertEqual(prof["rank_90"], 1)
        self.assertEqual(prof["rank_95"], 1)
        self.assertEqual(prof["rank_99"], 1)
        self.assertLess(abs(prof["effective_rank"] - 1.0), 1e-3)

    def test_bilinear_model_has_rank_three_gradient_ensemble(self):
        torch.manual_seed(0)
        model = BilinearPlusLinear()
        x = torch.randn(1024, 6)

        prof = jacobian_rank_profile(model, x, chunk_size=128)

        self.assertEqual(prof["input_dim"], 6)
        self.assertEqual(prof["rank_90"], 3)
        self.assertEqual(prof["rank_95"], 3)
        self.assertEqual(prof["rank_99"], 3)
        self.assertGreater(prof["effective_rank"], 2.0)
        self.assertLess(prof["effective_rank"], 3.1)

    def test_requires_multiple_samples(self):
        model = torch.nn.Linear(3, 1)
        x = torch.randn(1, 3)
        with self.assertRaises(ValueError):
            jacobian_rank_profile(model, x)


if __name__ == "__main__":
    unittest.main()
