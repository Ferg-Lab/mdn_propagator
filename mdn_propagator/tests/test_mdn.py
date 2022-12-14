"""Tests for mixture density networks"""

import pytest
import torch
from mdn_propagator.mdn import MixtureDensityNetwork


@pytest.mark.parametrize(
    "dim_in, dim_out, n_components, network_type",
    [
        (4, 4, 20, "mlp"),
        (6, 3, 5, "mlp"),
    ],
)
def test_mixture_density_network(dim_in, dim_out, n_components, network_type):
    mdn = MixtureDensityNetwork(
        dim_in, dim_out, n_components, network_type=network_type
    )

    # Test forward pass
    x = torch.randn(32, dim_in)
    pi, normal = mdn.forward(x)
    assert pi.probs.shape == (32, n_components)
    assert normal.loc.shape == (32, n_components, dim_out)
    assert normal.scale.shape == (32, n_components, dim_out)

    # Test loss calculation
    y = torch.randn(32, dim_out)
    loss = mdn.loss(x, y)
    assert loss.shape == (32,)

    # Test sampling from model
    samples = mdn.sample(x)
    assert samples.shape == (32, dim_out)
