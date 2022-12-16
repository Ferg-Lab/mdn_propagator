"""Propagator module"""

import torch
from pytorch_lightning import LightningModule, Trainer
from typing import Union
from tqdm.autonotebook import tqdm
from mdn_propagator.mdn import MixtureDensityNetwork
from mdn_propagator.data import DataModule

class Propagator(LightningModule):
    """

    Propagator    

    Parameters
    ----------
    dim: int
        dimensionality data to learn the propagator over

    k : int, default = 1
        length of the markov process, i.e. k=1 means constructing time-lagged pairs, while k=2 means constructing
        time-lagged triplets  

    n_components: int
        number of components in the mixture model.

    network_type: str, default = 'mlp'
        type of network to use for the MDN. Currently, only "mlp" is supported.

    lr: float, default = 1e-3
        learning rate to use during training

    **kwargs
        other keyword arguments passed to the `network_type` constructor
    """
    def __init__(self, dim: int, k: int = 1, n_components: int = 25, network_type: str = 'mlp', lr: float = 1e-3, **kwargs):
        super(Propagator, self).__init__()
        self.save_hyperparameters()
        self.mdn = MixtureDensityNetwork(k * dim, dim, n_components, network_type=network_type, **kwargs)

        self.is_fit = False
    
    def forward(self, x):
        return self.mdn(x)
    
    def training_step(self, batch, batch_idx):
        frames, pathweights = batch
        if len(frames) > 2:
            x = torch.cat(frames[:-1], dim = -1)
        elif len(frames) == 2:
            x = frames[0]
        y = frames[-1]
        
        loss = self.mdn.loss(x, y)
        loss = (pathweights * loss).mean()
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return opt
    
    def fit(self,
            data: Union[torch.Tensor, list],
            lag: int,
            ln_dynamical_weight: Union[torch.Tensor, list] = None,
            thermo_weight: Union[torch.Tensor, list] = None,
            k: int = 1,
            batch_size: int = 1000,
            max_epochs: int = 100,
            **kwargs):
        """
        Datamodule for the k-step dataset

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

        batch_size : int, default = 1000
            training batch size

        max_epochs : int, default = 100
            maximum number of epochs to train for

        **kwargs: 
            additional keyword arguments to be passed to the the Lightning Trainer

        """
        datamodule = DataModule(data=data, lag=lag, ln_dynamical_weight=ln_dynamical_weight, thermo_weight=thermo_weight, k=k, batch_size=batch_size, **kwargs)
        self._scaler = datamodule.scaler

        trainer = Trainer(auto_select_gpus=True, max_epochs=max_epochs, logger=False, enable_checkpointing=False, **kwargs)
        trainer.fit(self, datamodule)

        self.is_fit = True
        return self
    
    def propagate(self, x: torch.Tensor):
        """
        Propagates sample(s) using the fit model  
        
        Assumes sample is in original data space and uses scaler to transform input and 
        inverse_transform the output

        Parameters
        ----------
        x : torch.Tensor
            sample(s) to be propagated with dimentionality [n, dim], where n = numebr of samples
            to propagate and dim = dimentionality of the input
        """

        assert self.is_fit, 'model must be fit to data first using `fit`'

        self.eval()
        x = torch.tensor(self._scaler.transform(x)).float()
        with torch.no_grad():
            x = self.mdn.sample(x).clip(0,1)
        x = self._scaler.inverse_transform(x)
        return x

    def gen_synthetic_traj(self, x_0: torch.Tensor, n_steps: int):
        """
        Generates a synthetic trajectory from an initial starting point `x_0`

        Parameters
        ----------
        x_0 : torch.Tensor
            starting point of the initial trajectory, with size [1, dim] 

        n_steps : int
            number of steps in the synthetic trajectory
        """

        assert self.is_fit, 'model must be fit to data first using `fit`'

        self.eval()
        with torch.no_grad():
            x = torch.tensor(self._scaler.transform(x_0)).float()
            xs = list()
            for _ in tqdm(range(int(n_steps))):
                x = self.mdn.sample(x).clip(0, 1)
                xs.append(x)
        xs = torch.cat(xs)
        xs = self._scaler.inverse_transform(xs)
        return xs
        

