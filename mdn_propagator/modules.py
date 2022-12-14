"""Neural network modules"""

import torch.nn as nn


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP)

    Parameters
    ----------
    input_dim : int
        input data dimensionality

    output_dim : int
        output data dimensionality

    hidden_dim : int, default = 128
        dimensionality of hidden layers

    n_hidden_layers : int, default = 1
        number of hidden layers

    activation : nn.Module, default = nn.SiLU
        activation function to use in the hidden layers
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 1,
        activation: nn.Module = nn.SiLU,
        **kwargs
    ):
        super(MLP, self).__init__()

        assert n_hidden_layers > 0, "number of hidden layers must be > 0"

        layers = list()
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the MLP

        This method takes a tensor of inputs `x` and returns the outputs of the MLP

        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, input_dim)

        Returns
        -------
        out : torch.Tensor
            output tensor of shape (batch_size, output_dim)
        """
        out = self.mlp(x)
        return out
