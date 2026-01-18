import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub


class MLP(nn.Module):
    """Multi-Layer Perceptron compatible with PyTorch Quantization Aware Training."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        depth: int = 1,
        activation_fn: type = nn.ReLU,
    ):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        layers = []
        in_features = input_size
        for _ in range(depth):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(activation_fn())
            in_features = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        x = self.layers(x)
        x = self.dequant(x)
        return x
