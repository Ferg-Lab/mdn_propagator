"""Testing Propagator functionality"""

import torch

from mdn_propagator.propagator import Propagator


def test_Propagator():
    # create dummy data
    data = torch.randn(100, 2)
    lag = 1
    ln_dynamical_weight = None
    thermo_weight = None
    k = 1

    propagator = Propagator(dim=data.size(1))
    propagator.fit(data, lag, ln_dynamical_weight, thermo_weight, k, 10, 5)

    n = 5
    x = data[:n]
    y = propagator.propagate(x)
    assert y.shape == (n, 2)
    x = data[0][None]
    y = propagator.gen_synthetic_traj(x, int(1e2))
    assert y.shape == (1e2, 2)

    # test with k = 2
    data = [torch.randn(100, 2), torch.randn(100, 2)]
    lag = 1
    ln_dynamical_weight = [torch.zeros(100), torch.zeros(100)]
    thermo_weight = [torch.ones(100), torch.ones(100)]
    k = 2

    propagator = Propagator(dim=data[0].size(1), k=k)
    propagator.fit(data, lag, ln_dynamical_weight, thermo_weight, k, 10, 5)

    n = 5
    x = torch.randn(n, k * 2)
    y = propagator.propagate(x)
    assert y.shape == (n, 2)
    x = torch.randn(1, k * 2)
    y = propagator.gen_synthetic_traj(x, int(1e2))
    assert y.shape == (1e2, 2)
