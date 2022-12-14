"""Mixture density network"""

import torch
from torch import nn
from torch.distributions import Normal, OneHotCategorical
import torch.nn.functional as F

from .modules import MLP


class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network [Bishop, 1994]

    A mixture density network (MDN) is a neural network that outputs a mixture of Gaussian distributions
    in order to model the distribution of the response variable. The MDN uses a mixture model, where each
    component of the mixture is a Gaussian distribution with its own mean and standard deviation.

    Parameters
    ----------
    dim_in: int
        dimensionality of the input covariates.

    dim_out: int
        dimensionality of the response variable.

    n_components: int
        number of components in the mixture model.

    network_type: str, default = 'mlp'
        type of network to use for the MDN. Currently, only "mlp" is supported.

    **kwargs
        other keyword arguments passed to the `network_type` constructor
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_components: int,
        network_type: str = "mlp",
        **kwargs
    ):
        super().__init__()
        self.n_components = n_components
        out_dim = n_components * (2 * dim_out + 1)

        if network_type.lower() == "mlp":
            self.network = MLP(input_dim=dim_in, output_dim=out_dim, **kwargs)
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward pass of the mixture density network.

        This method takes a tensor of inputs `x` and returns a tuple containing the mixture weights
        and the Gaussian distributions for each component of the mixture model.

        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, dim_in)

        Returns
        -------
        pi : torch.distributions.OneHotCategorical
            mixture weights for each component of the mixture model

        normal : torch.distributions.Normal
            Gaussian distributions for each component of the mixture model
        """

        out = self.network(x)

        # get Gaussian probabilities
        params_pi = out[..., : self.n_components]
        pi = OneHotCategorical(logits=params_pi)

        params = out[..., self.n_components :]
        mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        sd = torch.stack(sd.split(sd.shape[1] // self.n_components, 1))
        # replaced torch.exp(sd) with ELU plus to improve numerical stability
        # added epsilon to avoid zero scale
        # due to non associativity of floating point add, 1 and 1e-7 need to be added seperately
        return pi, Normal(
            torch.sigmoid(mean).transpose(0, 1), (F.elu(sd) + 1 + 1e-7).transpose(0, 1)
        )

    def loss(self, x, y):
        """Computes the negative log-likelihood loss of the model on input data.

        This method takes a tensor of inputs `x` and a tensor of targets `y` and computes the negative
        log-likelihood loss of the mixture density network on the given data.

        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, dim_in)

        y : torch.Tensor
            target tensor of shape (batch_size, dim_out)

        Returns
        -------
        loss : torch.Tensor
            negative log-likelihood loss of the mixture density network on the given data
        """
        pi, normal = self.forward(x)
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        # use pi.logits directly instead of torch.log(pi.probs) to
        # avoid numerical problem
        loss = -torch.logsumexp(pi.logits + loglik, dim=1)
        return loss

    def sample(self, x):
        """Generates samples from the mixture density network.

        This method takes a tensor of inputs `x` and generates samples from the mixture density network
        using the mixture weights and the Gaussian distributions predicted by the network.

        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, dim_in)

        Returns
        -------
        samples : torch.Tensor
            samples generated by the mixture density network
        """
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
        return samples
