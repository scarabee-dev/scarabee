======================
Upcoming Release Notes
======================

------------
New Features
------------

- A new nodal diffusion solver based on the nodal CMFD method with 2-node current
  calculations has been added. This new solver, called `NEM4DiffusionDriver` is
  approximately 10 times faster than the previous `NEMDiffusionDriver` solver which was
  based on the method of interface currents. It has an identical interface to the
  previous solver (with a few added elements), and should work as a drop in replacement.
  As such, the previous `NEMDiffusionDriver` has been deprecated. The motivation behind
  this new solver is not purely the better run time performance, but it is written in
  such a way that is will be drastically easier to add other nodal methods in the future,
  such as the Semi-Analytical Nodal Method and the Analytical Nodal Method.

- A new finite-difference diffusion solver, based on the new CMFD nodal kernel method
  above, has been added. This new solver is called `FDNodalDiffusionDriver`. Currently,
  there are no plans to deprecate the previous `FDDiffusionDriver` class, as that solver
  yields superior performance for finite-difference calculations.

---------
Bug Fixes
---------

- There was a bug where copying a DiffusionData instance in Python did not include the
  LeakageCorrections which may be present. This resulted in the copies not having a
  LeakageCorrections instance and gave incorrect results in nodal diffusion calculations.

