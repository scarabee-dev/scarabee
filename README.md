# Scarabée
[![Documentation Status](https://readthedocs.org/projects/scarabee/badge/?version=latest)](https://scarabee.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-GPLv3-brightgreen)](https://github.com/HunterBelanger/scarabee/blob/master/LICENSE)

Scarabée is a simplified lattice physics code for light water reactor (LWR)
neutronics calculations. It currently has the following features:

* Resonance self-shielding according to Carlvik's 2-term rational approximation
* 1D Annular pin cell collision probabilities solver for fixed-source and k-eigenvalue problems
* 2D Method of characteristics solver for fixed-source and k-eigenvalue problems
* Finite difference diffusion solver for 1D, 2D, and 3D fixed-source and k-eigenvalue problems
* Nodal expansion method diffusion solver for 3D k-eigenvalue problems
* PWR lattice calculations with depletion

This project closely follows the methods outlined in *Methods of Steady-State
Reactor Physics in Nuclear Design* by Stamm'ler and Abbate, and *Lattice
Physics Computations* by Knott and Yamamoto, from the *Handbook of Nuclear
Engineering*, edited by D. Cacuci, and *Applied Reactor Physics* by Hébert.

Scarabée uses a custom HDF5 formated nuclear data library which is easy and
intuitive to understand. The `data` directory contains a helper library and
scripts to generate a multi-group nuclear data library with the FRENDY nuclear
data processing code. FRENDY is free/open-source software, and can be downloaded
[here](https://rpg.jaea.go.jp/main/en/program_frendy/). Currently, only FRENDY
is supported, due to its advanced multi-group processing features.
