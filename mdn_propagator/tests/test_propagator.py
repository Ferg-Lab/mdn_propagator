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

    propagator = Propagator(dim = data.size(1))
    propagator.fit(data, lag, ln_dynamical_weight, thermo_weight, k, 10, 5)

    x = data[0][None]
    propagator.propagate(x)
    propagator.gen_synthetic_traj(x, int(1E2))


    # test with k = 2
    data = [torch.randn(100, 2), torch.randn(100, 2)]
    lag = 1
    ln_dynamical_weight = [torch.zeros(100), torch.zeros(100)]
    thermo_weight = [torch.ones(100), torch.ones(100)]
    k = 2

    propagator = Propagator(dim = data[0].size(1), k  = k)
    propagator.fit(data, lag, ln_dynamical_weight, thermo_weight, k, 10, 5)

    #x = data[0][0][None]
    #propagator.propagate(x)
    #propagator.gen_synthetic_traj(x, int(1E2))
