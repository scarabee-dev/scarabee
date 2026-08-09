======================
Upcoming Release Notes
======================

.. currentmodule:: scarabee

---------------------
Important API Changes
---------------------

- Previously, when using the B1 or P1 leakage models, the flux spectrum obtained by the
  model would be used to condense fine-group diffusion coefficients obtained by
  homogenizing the assembly's transport cross section, completely disregarding the
  diffusion coefficients computed by the leakage model. Therefore, while the P1 leakage
  model was being used by default, it was not being used to compute the diffusion
  coefficients. This was an intentional design choice, but it is likely not what users
  expect. Now, when using the B1 or P1 leakage models, the diffusion coefficients
  produced for the assembly are computed with the fine-group diffusion coefficients
  calculated by the leakage model. Using the fundamental mode spectrum option, however,
  was always self consistent as this model directly uses the assembly homogenized
  transport cross sections to compute fine-group diffusion coefficients. 

- The default leakage model has been changed from P1 to Fundamental-Model.

- The keff attribute of the :class:`reseau.PWRAssembly` class is now a Numpy array, even
  if only a single transport calculation was performed. This change was made to make the
  API to access simulation results more consistent between the different simulation modes
  (with / without depletion).

- The diffusion_data, and form_factors attributes of the :class:`reseau.PWRAssembly`
  class are now lists, even if only a single transport calculation was performed. This
  change was made to make the API to access simulation results more consistent between
  the different simulation modes (with / without depletion).

------------
New Features
------------

- A new nodal diffusion solver based on the nodal CMFD method with 2-node current
  calculations has been added. This new solver, called :class:`NEM4DiffusionDriver` is
  approximately 10 times faster than the previous :class:`NEMDiffusionDriver` solver
  which was based on the method of interface currents. It has an identical interface to
  the previous solver (with a few added elements), and should work as a drop in
  replacement. As such, the previous :class:`NEMDiffusionDriver` has been deprecated. The
  motivation behind this new solver is not purely the better run time performance, but it
  is written in such a way that is will be drastically easier to add other nodal methods
  in the future, such as the Semi-Analytical Nodal Method and the Analytical Nodal Method.

- A new finite-difference diffusion solver, based on the new CMFD nodal kernel method
  above, has been added. This new solver is called :class:`FDNodalDiffusionDriver`.
  Currently, there are no plans to deprecate the previous :class:`FDDiffusionDriver`
  class, as that solver yields superior performance for finite-difference calculations.

- A new Semi-Analytical Nodal Method solver, called :class:`SANMDiffusionDriver` has been
  added, based on the new nodal diffusion solver shell. This solver uses a flux expansion
  based on a quadratic and hyperbolic functions.

- The :class:`NEMDiffusionDriver`, :class:`NEM4DiffusionDriver`, and
  :class:`SANMDiffusionDriver` classes can now detect when leakage corrections are
  present in a problem, and will use them automatically. They can, however, be disabled
  by the user after construction by setting the leakage_correction attribute to False.

- The :class:`reseau.PWRAssembly` class now has the new attributes moderator_xs and moc,
  to access the :class:`CrossSection` used for the moderator and the :class:`MOCDriver`
  used for the assembly calculation.

- The support scripts used to produce nuclear data libraries have been updated. ENDFtk
  and PapillonNDL are not longer required, but the
  `endf <https://github.com/paulromano/endf-python>`__ Python library is now needed.
  Library processing is now performed in parallel, greatly reducing the run times.
  Several bugs were also corrected (such as not saving IR-lambda factors). The default
  script to produce an ENDF/B-VIII.0 library now also requires TENDL ENDF files for
  several short lived nucleides which appear in the depletion chain.

---------
Bug Fixes
---------

- There was a bug where copying a DiffusionData instance in Python did not include the
  LeakageCorrections which may be present. This resulted in the copies not having a
  LeakageCorrections instance and gave incorrect results in nodal diffusion calculations.

