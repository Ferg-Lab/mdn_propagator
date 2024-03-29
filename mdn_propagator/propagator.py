"""Propagator module"""

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from typing import Union
from tqdm.autonotebook import tqdm
from mdn_propagator.mdn import MixtureDensityNetwork
from mdn_propagator.data import DataModule
from mdn_propagator.utils import MinMaxScaler


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

    def __init__(
        self,
        dim: int,
        k: int = 1,
        n_components: int = 25,
        network_type: str = "mlp",
        lr: float = 1e-3,
        **kwargs,
    ):
        super(Propagator, self).__init__()
        self.save_hyperparameters()
        self.mdn = MixtureDensityNetwork(
            k * dim, dim, n_components, network_type=network_type, **kwargs
        )
        self._scaler = MinMaxScaler(dim)

        self.is_fit = False

    def forward(self, x):
        return self.mdn(x)

    def training_step(self, batch, batch_idx):
        frames, pathweights = batch
        if len(frames) > 2:
            x = torch.cat(frames[:-1], dim=-1)
        elif len(frames) == 2:
            x = frames[0]
        y = frames[-1]

        loss = self.mdn.loss(x, y)
        loss = (pathweights * loss).mean()
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return opt

    def fit(
        self,
        data: Union[torch.Tensor, list],
        lag: int,
        ln_dynamical_weight: Union[torch.Tensor, list] = None,
        thermo_weight: Union[torch.Tensor, list] = None,
        batch_size: int = 1000,
        max_epochs: int = 100,
        log: Union[str, bool] = False,
        **kwargs,
    ):
        """
        Fit the propagator on provided data

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

        batch_size : int, default = 1000
            training batch size

        max_epochs : int, default = 100
            maximum number of epochs to train for

        log : str or bool, default = False
            if the results of the training should be logged. If True logs are by default saved in CSV format
            to the directory `./mdn_propagator_logs/version_x/`, where `x` increments based on what has been
            logged already. If a string is passed the saving directory is created based on the provided name
            `./mdn_propagator_logs/{log}/`.

        **kwargs:
            additional keyword arguments to be passed to the the Lightning `Trainer` and/or the `DataModule`

        """
        kwargs.get("enable_checkpointing", False)
        datamodule = DataModule(
            data=data,
            lag=lag,
            ln_dynamical_weight=ln_dynamical_weight,
            thermo_weight=thermo_weight,
            k=self.hparams.k,
            batch_size=batch_size,
            **kwargs,
        )
        if self.is_fit:
            raise Warning(
                """The `fit` method was called more than once on the same `Propagator` instance,
                recreating data scaler on dataset from the most recent `fit` invocation. This warning
                can be safely ignored if the `Propagator` is being fit on the same data"""
            )
        self._scaler = datamodule.scaler

        if not hasattr(self, "trainer_"):
            self.trainer_ = Trainer(
                devices=1,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                max_epochs=max_epochs,
                logger=False
                if log is False
                else CSVLogger(
                    save_dir="./",
                    name="mdn_propagator_logs",
                    version=None if not isinstance(log, str) else log,
                ),
                **kwargs,
            )
            self.trainer_.fit(self, datamodule)
        else:
            self.trainer_.fit(self, datamodule)

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
            sample(s) to be propagated with dimentionality [n, k * dim], where
            n is the number of samples to propagate, k = order of the markov process
            and dim = dimentionality of the input
        """

        assert self.is_fit, "model must be fit to data first using `fit`"
        assert (
            x.size(1) == self.hparams.k * self.hparams.dim
        ), f"inconsistent dimensions, expecting {self.hparams.k} * {self.hparams.dim} dim"

        n = x.size(0)

        self.eval()
        if self.hparams.k == 1:
            x = self._scaler.transform(x).float()
        else:
            x = (
                self._scaler.transform(x.reshape(n * self.hparams.k, -1))
                .reshape(n, -1)
                .float()
            )

        with torch.no_grad():
            x = self.mdn.sample(x).clip(0, 1)
        x = self._scaler.inverse_transform(x)
        return x

    def gen_synthetic_traj(self, x_0: torch.Tensor, n_steps: int):
        """
        Generates a synthetic trajectory from an initial starting point `x_0`

        Parameters
        ----------
        x_0 : torch.Tensor
            starting point of the initial trajectory, with size [1, k * dim]
            where k = the order of the Markov process and dim = the dimensionality
            of the input data

        n_steps : int
            number of steps in the synthetic trajectory
        """

        assert self.is_fit, "model must be fit to data first using `fit`"
        assert (
            x_0.size(1) == self.hparams.k * self.hparams.dim
        ), f"inconsistent dimensions, expecting {self.hparams.k} * {self.hparams.dim} dim"

        self.eval()
        if self.hparams.k == 1:
            with torch.no_grad():
                x = self._scaler.transform(x_0).float()
                xs = list()
                for _ in tqdm(range(int(n_steps))):
                    x = self.mdn.sample(x).clip(0, 1)
                    xs.append(x)
            xs = torch.cat(xs)
            xs = self._scaler.inverse_transform(xs)
        elif self.hparams.k > 1:
            x = (
                self._scaler.transform(x_0.reshape(self.hparams.k, -1))
                .reshape(1, -1)
                .float()
            )
            with torch.no_grad():
                xs = list()
                for _ in tqdm(range(int(n_steps))):
                    x_next = self.mdn.sample(x).clip(0, 1)
                    xs.append(x_next)
                    x = torch.cat((x[:, self.hparams.dim :], x_next), dim=-1)
            xs = torch.cat(xs)
            xs = self._scaler.inverse_transform(xs)

        return xs

    def save(self, fname: str):
        """
        Generates a synthetic trajectory from an initial starting point `x_0`

        Parameters
        ----------
        fname : str
            file name for saving a model checkpoint
        """

        assert self.is_fit, "model must be fit to data first using `fit`"

        self.trainer_.save_checkpoint(fname)
