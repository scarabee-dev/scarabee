Scarabée : A Light Water Reactor Lattice Physics Code
=====================================================

Scarabée is a Python library, primarily written in C++, which contains a number
of methods for solving the multi-group formalism of the neutron transport
equation.  These solvers can be used independently to analyze particular
problems of interest. Additionally, the library contains driver classes that
turn Scarabée into a full lattice physics code, automating the calculations
required to model 2D PWR assemblies. It uses a custom HDF5 formatted nuclear
data library which contains cross sections at different temperatures and
dilutions for all provided isotopes. Currently, equivalence in dilution is used
for self-shielding of cross sections.

Scarabée provides students an easy-to-use interface to model various LWR
problems, better understanding reactor physics and the core design process.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install/index
   theory/index 
   examples/index
   api/index 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
