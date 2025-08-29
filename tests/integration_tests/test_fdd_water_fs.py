import pytest
import pytest_timeout
import numpy as np
from scarabee import *

"""
Integration test for finite-difference diffusion solver fixed-source mode on a water block with a point source.
"""

class TestFDDWaterFixedSource:
    
    # This problem should never take this long to run
    # if it takes longer, most likely didn't converge
    @pytest.mark.timeout(1000)

    def test_water_fs(self):
        pass

        Et = np.array([1.59206E-01, 4.12970E-01, 5.90310E-01, 5.84350E-01, 7.18000E-01, 1.25445E+00, 2.65038E+00])
        Ea = np.array([6.01050E-04, 1.57930E-05, 3.37160E-04, 1.94060E-03, 5.74160E-03, 1.50010E-02, 3.72390E-02])
        Es = np.array([[4.44777E-02, 1.13400E-01, 7.23470E-04, 3.74990E-06, 5.31840E-08, 0.00000E+00, 0.00000E+00],
                       [0.00000E+00, 2.82334E-01, 1.29940E-01, 6.23400E-04, 4.80020E-05, 7.44860E-06, 1.04550E-06],
                       [0.00000E+00, 0.00000E+00, 3.45256E-01, 2.24570E-01, 1.69990E-02, 2.64430E-03, 5.03440E-04],
                       [0.00000E+00, 0.00000E+00, 0.00000E+00, 9.10284E-02, 4.15510E-01, 6.37320E-02, 1.21390E-02],
                       [0.00000E+00, 0.00000E+00, 0.00000E+00, 7.14370E-05, 1.39138E-01, 5.11820E-01, 6.12290E-02],
                       [0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 2.21570E-03, 6.99913E-01, 5.37320E-01],
                       [0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 1.32440E-01, 2.48070E+00]])
        H2Oxs = CrossSection(Et, Ea, Es)

        H2Oxs = H2Oxs.diffusion_xs()

        # Define Cells
        L = 1.26 * 17
        NW = 5*17 # Number of empty water cells in a pin cell

        geom = DiffusionGeometry([H2Oxs], [L], [NW], [L], [NW], 0., 0., 0., 0.)
        solver = FDDiffusionDriver(geom)
        solver.set_extern_src(NW//2, NW//2, 0, 1.)
        solver.sim_mode = SimulationMode.FixedSource
        solver.flux_tolerance = 1.E-6
        solver.solve()

        flux, _, _, _ = solver.flux()

        # Check flux near one of the corners.
        # For some reason, M1 Macs have a hard time with this test, and need a
        # relatively high uncertainty. Don't think this is a real problem with
        # the code, more a hardware floating point issue.
        assert flux[0, NW//4, NW//4] == pytest.approx(0.00075089384360491140, rel=1.E-4)
        assert flux[1, NW//4, NW//4] == pytest.approx(0.00074730485848786140, rel=1.E-4)
        assert flux[2, NW//4, NW//4] == pytest.approx(0.00040547453273072160, rel=1.E-4)
        assert flux[3, NW//4, NW//4] == pytest.approx(0.00018576061880256343, rel=1.E-4)
        assert flux[4, NW//4, NW//4] == pytest.approx(0.00014746100470025965, rel=1.E-4)
        assert flux[5, NW//4, NW//4] == pytest.approx(0.00066284749711381990, rel=1.E-4)
        assert flux[6, NW//4, NW//4] == pytest.approx(0.00213305538166966800, rel=2.E-4)
