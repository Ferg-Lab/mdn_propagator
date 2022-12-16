MDN_Propagator
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/Ferg-Lab/mdn_propagator/workflows/CI/badge.svg)](https://github.com/Ferg-Lab/mdn_propagator/actions?query=workflow%3ACI)
<!-- [![codecov](https://codecov.io/gh/Ferg-Lab/MDN_Propagator/branch/main/graph/badge.svg)](https://codecov.io/gh/Ferg-Lab/MDN_Propagator/branch/main) -->


Mixture Density Networks for learning simulation propagators

Getting Started
===============


Installation
------------
To use mdn_propagator, you will need an environment with the following packages:

* Python 3.7+
* pytorch
* scikit-learn

Once you have these packages installed, you can install molecool in the same environment using
::

    pip install -e .

Usage
-------
Once installed, you can use the package. This example generates a synthetic trajectory of Alanine Dipeptide (ADP) in the backbone dihedral space ($\phi$, $\psi$). More examples can be found in the `examples` directory. 
::

    from mdn_propagator.propagator import Propagator

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


### Copyright

Copyright (c) 2022, Kirill Shmilovich


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
