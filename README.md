Mixture Density Networks 
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/Ferg-Lab/mdn_propagator/workflows/CI/badge.svg)](https://github.com/Ferg-Lab/mdn_propagator/actions?query=workflow%3ACI)
<!-- [![codecov](https://codecov.io/gh/Ferg-Lab/MDN_Propagator/branch/main/graph/badge.svg)](https://codecov.io/gh/Ferg-Lab/MDN_Propagator/branch/main) -->


This package impliments Mixture Density Networks (MDNs) for learning simulation propagators. Given a trajectory $X=\{x_0, x_1, x_2, \cdots \, x_N}$ where $x_t \in \mathbb{R}^d$ we learn a propagator $f_{\theta}(x_t)$ as a MDN that predicts the system state $\hat{x}\_{t+\tau}$ after a lag time $\tau$ $$f_{\theta}(x_t) = \hat{x}_{t+\tau}$$ 

Getting Started
===============


Installation
------------
To use `mdn_propagator`, you will need an environment with the following packages:

* Python 3.7+
* [PyTorch](https://pytorch.org/get-started/locally/)
* [PyTorch Lightning](https://www.pytorchlightning.ai/)

For running and plotting examples:
* [NumPy](https://numpy.org/install/)
* [Matplotlib](https://matplotlib.org/stable/users/getting_started/)
* [PyEMMA](http://www.emma-project.org/latest/INSTALL.html)

Once you have these packages installed, you can install `mdn_propagator` in the same environment using

```
$ pip install -e .
```

Usage
-------
Once installed, you can use the package. This example generates a synthetic trajectory of Alanine Dipeptide (ADP) in the space of the backbone dihedral angles ($\phi , \psi$). More detailed examples can be found in the `examples` directory. 


```python
from mdn_propagator.propagator import Propagator
import torch
import numpy as np

# load data
dihedrals_data = np.load('examples/data/alanine-dipeptide-3x250ns-backbone-dihedrals.npz')
phi_psi_data = [dihedrals_data['arr_0'], dihedrals_data['arr_1'], dihedrals_data['arr_2']]
phi_psi_data = [torch.tensor(p).float() for p in phi_psi_data]

# ininstantiate the model
model = Propagator(dim = phi_psi_data[0].size(1))

# fit the model
model.fit(phi_psi_data, lag = 1, max_epochs=100)

# Generate synthetic trajectory
n_steps = int(1E6)
x = phi_psi_data[0][0][None]
syn_traj = model.gen_synthetic_traj(x, n_steps)

# Save model checkpoint
model.save('ADP.ckpt')

# Load from checkpoint
model = Propagator.load_from_checkpoint('ADP.ckpt')
```
![image](https://user-images.githubusercontent.com/40403472/208270555-e606079f-adf9-49f5-ae36-40b489b8fa35.png)



The defulat network used for the propagator is a simple MLP. Network hyperparameters can be defined in the `Propagator` constructor, also see [modules](mdn_propagator/modules.py) for more details:


```python
from mdn_propagator.propagator import Propagator
from torch import nn

model = Propagator(dim = 10, hidden_dim = 256, n_hidden_layers = 2, activation = nn.ReLU, lr = 1e-4)
```


### Copyright

Copyright (c) 2022, Kirill Shmilovich


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
