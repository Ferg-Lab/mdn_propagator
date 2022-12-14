"""tests for modules"""

import pytest
import torch
from torch import nn

from mdn_propagator.modules import MLP


@pytest.mark.parametrize(
    "input_dim, output_dim, hidden_dim, n_hidden_layers, activation",
    [
        (5, 2, 128, 1, nn.SiLU),
        (10, 5, 256, 2, nn.ReLU),
        (15, 10, 64, 3, nn.Tanh),
    ],
)
def test_mlp(input_dim, output_dim, hidden_dim, n_hidden_layers, activation):
    mlp = MLP(input_dim, output_dim, hidden_dim, n_hidden_layers, activation)

    # Test the forward pass of the MLP
    x = torch.randn(3, input_dim)
    y = mlp(x)

    # Test the shape of the output
    assert y.shape == (3, output_dim)

    # Test the number of layers in the MLP
    expected_n_layers = 2 * n_hidden_layers + 3
    assert len(list(mlp.mlp.children())) == expected_n_layers
