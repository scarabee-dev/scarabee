# Analytical Benchmark Test Set for Criticality Code and Verification
# Problem No.:32 [PUa-1-1-SL]
# Exact keff = 1.

from scarabee import *
import numpy as np
import matplotlib.pyplot as plt

Et = np.array([1.])
Ea = np.array([1. - 0.733333])
Es = np.array([[[0.733333]],
              [[0.333333]]])
Ef = np.array([0.266667])
vEf = np.array([2.5 * 0.266667])
chi = np.array([1.])

dtr = np.array([0.333333])

fuel = CrossSection(Et, dtr, Ea, Es, Ef, vEf, chi)

# Define Cells

thickness =  0.79606 * 2.
Nx = 500
# slab
dx = thickness / Nx
dy = thickness
slab = EmptyCell(fuel, dx, dy)
fuel_slab = Cartesian2D(Nx*[dx], [dy])
fuel_slab.set_tiles([slab] * Nx)

moc = MOCDriver(fuel_slab, anisotropic = True)
moc.x_min_bc = BoundaryCondition.Vacuum
moc.x_max_bc = BoundaryCondition.Vacuum
moc.y_min_bc = BoundaryCondition.Reflective
moc.y_max_bc = BoundaryCondition.Reflective

moc.cmfd = CMFD(10*[thickness/10], [thickness], [[0,0]])
moc.cmfd.check_neutron_balance = True
moc.generate_tracks(128, 0.005, YamamotoTabuchi6())
#moc.generate_tracks(128, 0.01, Legendre12())
moc.flux_tolerance = 1.E-15
moc.keff_tolerance = 1.E-15
set_logging_level(LogLevel.Debug)
moc.solve()

flux, x, y = moc.rasterize_flux(1000, 1)
plt.stairs(flux[0,0,:], edges=x)
plt.show()
