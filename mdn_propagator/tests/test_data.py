import torch

from mdn_propagator.data import KStepDataset

def test_KStepDataset():
    # default
    data = torch.tensor([
       [1.0, 2.0],
       [3.0, 4.0],
       [5.0, 6.0],
       [7.0, 8.0] 
    ])
    lag = 1
    ln_dynamical_weight = None
    thermo_weight = None
    k = 1

    # initialize KStepDataset with dummy data
    dataset = KStepDataset(data, lag, ln_dynamical_weight, thermo_weight, k)

    # test that lag attribute is set correctly
    assert dataset.lag == lag
    # test that k attribute is set correctly
    assert dataset.k == k

    # test that ks attribute is set correctly
    assert len(dataset.ks) == k + 1

    # test expected output of ks
    expected_out = {0: torch.tensor([
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0]
                ]),
                  1: torch.tensor([
                    [3.0, 4.0],
                    [5.0, 6.0],
                    [7.0, 8.0]
                  ])}
    for k_ in dataset.ks.keys():
        assert torch.allclose(dataset.ks[k_], expected_out[k_])

    # test __getitem__
    assert torch.allclose(dataset[0][0][0],torch.tensor([1., 2.]))
    assert torch.allclose(dataset[0][0][1],torch.tensor([3., 4.]))
    assert torch.allclose(dataset[0][1],torch.tensor(1.))

    # k=2, thermo and dynamical weights
    data = torch.tensor([
       [1.0, 2.0],
       [3.0, 4.0],
       [5.0, 6.0],
       [7.0, 8.0] 
    ])
    lag = 1
    ln_dynamical_weight = torch.zeros(4)
    thermo_weight = torch.ones(4)
    k = 2

    # initialize KStepDataset with dummy data
    dataset = KStepDataset(data, lag, ln_dynamical_weight, thermo_weight, k)

    # test that lag attribute is set correctly
    assert dataset.lag == lag
    # test that k attribute is set correctly
    assert dataset.k == k

    # test that ks attribute is set correctly
    assert len(dataset.ks) == k + 1

    # test __getitem__
    assert torch.allclose(dataset[0][0][0],torch.tensor([1., 2.]))
    assert torch.allclose(dataset[0][0][1],torch.tensor([3., 4.]))
    assert torch.allclose(dataset[0][0][2],torch.tensor([5., 6.]))
    assert torch.allclose(dataset[0][1],torch.tensor(1.))