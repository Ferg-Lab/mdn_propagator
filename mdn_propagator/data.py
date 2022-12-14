""""Dataset for loading consequtive steps"""

import torch
from torch.utils.data import Dataset

from typing import Union

class KStepDataset(Dataset):
    """
    Custom dataset for Snrv class


    Parameters
    ----------
    data : float tensor (single traj) or list of float tensors (multi traj); dim 0 = steps, dim 1 = features
        time-continuous trajectories

    lag : int
        lag in steps to apply to data trajectory

    ln_dynamical_weight : torch.tensor or list[torch.tensor] or None, default = None
        accumulated sum of the log Girsanov path weights between frames in the trajectory;
        Girsanov theorem measure of the probability of the observed sample path under a target potential
        relative to that which was actually observed under the simulation potential;
        identically unity (no reweighting rqd) for target potential == simulation potential and code as None

    thermo_weight : torch.tensor or list[torch.tensor] or None, default = None
        thermodynamic weights for each trajectory frame

    k : int, default = 1
        length of the markov process, i.e. k=1 means constructing time-lagged pairs, while k=2 means constructing
        time-lagged triplets  

    Attributes
    ----------
    self.lag : int
        lag in steps

    self.ks : dict[float tensor], n_traj[n x dim], n = observations, dim = dimensionality of trajectory featurization
        time-lagged trajectories

    self.pathweight : float tensor, n = observations
        pathweights from Girsanov theorem between time lagged observations;
        identically unity (no reweighting rqd) for target potential == simulation potential;
        if ln_pathweight == None => pathweight == ones
    """

    def __init__(self,
                 data: Union[torch.Tensor, list],
                 lag: int,
                 ln_dynamical_weight: Union[torch.Tensor, list] = None,
                 thermo_weight: Union[torch.Tensor, list] = None,
                 k: int = 1):

        self.lag = lag
        self.k = k

        if type(data) is list:
            ks = {k_:list() for k_ in range(k + 1)}

            for ii in range(0, len(data)):
                assert type(data[ii]) is torch.Tensor

            if (ln_dynamical_weight is not None) and (thermo_weight is not None):
                assert type(ln_dynamical_weight) is list
                assert type(thermo_weight) is list
                assert len(data) == len(ln_dynamical_weight) == len(thermo_weight)
                for ii in range(len(ln_dynamical_weight)):
                    assert type(ln_dynamical_weight[ii]) is torch.Tensor
                    assert type(thermo_weight[ii]) is torch.Tensor
                    assert (
                        data[ii].size()[0]
                        == ln_dynamical_weight[ii].size()[0]
                        == thermo_weight[ii].size()[0]
                    )
            
            pathweight = list()

            for ii in range(len(data)):
                for k_ in ks.keys():
                    start = k_ * self.lag
                    end = - (k - k_) * self.lag if (k - k_) != 0 else None
                    ks[k_].append(data[ii][start:end])

                K = data[ii][k * self.lag :].size(0)
                pathweight_ii = torch.ones(K)
                if (ln_dynamical_weight is not None) and (thermo_weight is not None):
                    for jj in range(K):
                        arg = torch.sum(
                            ln_dynamical_weight[ii][jj + 1 : jj + k * self.lag + 1]
                        )
                        pathweight_ii[jj] = torch.exp(arg) * thermo_weight[ii][jj]
                pathweight.append(pathweight_ii)

            pathweight = torch.cat(pathweight, dim=0)
            ks = {k:torch.cat(v, dim=0) for k,v in ks.items()}

        elif type(data) is torch.Tensor:
            
            ks = dict()
            
            if (ln_dynamical_weight is not None) and (thermo_weight is not None):
                assert type(ln_dynamical_weight) is torch.Tensor
                assert type(thermo_weight) is torch.Tensor
                assert (
                    data.size()[0]
                    == ln_dynamical_weight.size()[0]
                    == thermo_weight.size()[0]
                )

            for k_ in range(k+1):
                start = k_ * self.lag
                end = - (k - k_) * self.lag if (k - k_) != 0 else None
                ks[k_] = data[start:end]

            K = ks[0].size(0)
            pathweight = torch.ones(K)
            if (ln_dynamical_weight is not None) and (thermo_weight is not None):
                for ii in range(K):
                    arg = torch.sum(ln_dynamical_weight[ii + 1 : ii + k * self.lag + 1])
                    pathweight[ii] = torch.exp(arg) * thermo_weight[ii]

        else:
            raise TypeError(
                "Data type %s is not supported; must be a float tensor (single traj) or list of float tensors (multi "
                "traj)" % type(data)
            )

        self.pathweight = pathweight
        self.ks = ks

    def __getitem__(self, index):
        pathweight = self.pathweight[index]
        return [[v[index] for v in self.ks.values()], pathweight]

    def __len__(self):
        return len(self.ks[0])