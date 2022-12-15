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

    propagator = Propagator(dim = data.shape[1])

    # test with k = 2
    data = torch.randn(100, 2)
    lag = 1
    ln_dynamical_weight = None
    thermo_weight = None
    k = 2

    propagator = Propagator(dim = data.shape[1])
    propagator.fit(data, lag, ln_dynamical_weight, thermo_weight, k, 10, 5)
    propagator.fit(data, lag, ln_dynamical_weight, thermo_weight, k, 10, 5)

