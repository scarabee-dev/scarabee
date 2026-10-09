======================
Upcoming Release Notes
======================

.. currentmodule:: scarabee

---------------------
Important API Changes
---------------------

- Instead of re-computing Dancoff corrections at each depletion step, they are now only
  calculated for the initial transport solve. As the cladding is not depleted, the fuel
  Dancoff corrections will not change with time. While the cladding Dancoff corrections
  could change slightly at each depletion step (due to changes in the potential cross
  section of the fuel), this effect should be very small, and the self-shielding of the
  cladding is already a relatively small effect.

- Classes like :class:`CrossSection`, :class:`DiffusionCrossSection`,
  :class:`DiffusionData`, :class:`MOCDriver`, etc., can no longer be saved/loaded
  into/from a binary file. You should now use pickles to save and load objects.

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

- The :class:`reseau.PWRAssembly` class now has support for control rods. A new
  :class:`reseau.ControlRod` class as been written which can be used as a fill for a
  :class:`reseau.GuideTube`. If a problem has control rods, two sets of Dancoff
  corrections are calculated for both the fuel and cladding: one set with control rods
  inserted, and one set with control rods removed. The control rods also have their own
  Dancoff corrections and are self-shielded. A control rod can be discretized into rings
  which will be self-shielded with the Stoker-Weiss method. Control rods are also
  depleted with the fuel if inserted when performing a depletion simulation. Control rods
  can also be removed/inserted on-the-fly between assembly solves; by default, the
  control rods start in an inserted position.

- All classes should now be picklable. If you find a class which is not picklable, this
  is a bug and should be reported as an issue.

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

- All of the solvers now check for interrupt signals (like Ctrl-c) from Python to stop
  a long running calculation. This makes it easier to force stop long simulations.

---------
Bug Fixes
---------

- The Dancoff correction (C) for guide tubes was incorrectly being replaced with the
  Dancoff factor (D = 1 - C). This was found and corrected when implementing the control
  rod model.

- There was a bug where copying a DiffusionData instance in Python did not include the
  LeakageCorrections which may be present. This resulted in the copies not having a
  LeakageCorrections instance and gave incorrect results in nodal diffusion calculations.

