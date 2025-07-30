import pytest
import pytest_timeout
import numpy as np
from scarabee import *

"""
Integration test for finite-difference diffusion solver K-eigenvalue mode on IAEA-3D benchmark problem.
"""

class TestFDDIAEA3D:
    
    # This problem should never take this long to run
    # if it takes longer, most likely didn't converge
    @pytest.mark.timeout(1000)
    def test_iaea3d(self):

        # Outer Fuel
        Et = np.array([0.222222, 0.833333])
        D = 1. / (3. * Et)
        Ea = np.array([0.010, 0.080])
        vEf = np.array([0.0, 0.135])
        Ef = vEf
        chi = np.array([1., 0.])
        Es = np.array([[0.1922, 0.020],
                       [0.0,    0.7533]])
        a1 = DiffusionCrossSection(D, Ea, Es, Ef, vEf, chi, "A1")

        # Inner Fuel
        Et = np.array([0.222222, 0.833333])
        D = 1. / (3. * Et)
        Ea = np.array([0.010, 0.085])
        vEf = np.array([0.0, 0.135])
        Ef = vEf
        Es = np.array([[0.1922, 0.020],
                       [0.0,    0.7483]])
        a2 = DiffusionCrossSection(D, Ea, Es, Ef, vEf, chi, "A2")

        # Inner Fuel + Control Rods
        Et = np.array([0.222222, 0.833333])
        D = 1. / (3. * Et)
        Ea = np.array([0.0100, 0.1300])
        vEf = np.array([0.0, 0.135])
        Ef = vEf
        Es = np.array([[0.1922, 0.020],
                       [0.0,    0.7033]])
        a3 = DiffusionCrossSection(D, Ea, Es, Ef, vEf, chi, "A3")

        # Reflector
        Et = np.array([0.166667, 1.111111])
        D = 1. / (3. * Et)
        Ea = np.array([0.0, 0.01])
        Es = np.array([[0.1267, 0.04],
                       [0.0,    1.1011]])
        a4 = DiffusionCrossSection(D, Ea, Es, "A4")

        # Reflector + Control Rods
        Et = np.array([0.166667, 1.111111])
        D = 1. / (3. * Et)
        Ea = np.array([0.0, 0.055])
        Es = np.array([[0.0, 0.040],
                       [0.0, 0.000]])
        a5 = DiffusionCrossSection(D, Ea, Es, "A5")


        #        Half Assemblies down this column
        #        |
        #        V
        tiles = [# Top Reflector
                 a5, a4, a4, a4, a5, a4, a4, a4, a4, # <- Half Assemblies
                 a4, a4, a4, a4, a4, a4, a4, a4, a4,
                 a4, a4, a5, a4, a4, a4, a4, a4, a4,
                 a4, a4, a4, a4, a4, a4, a4, a4, a4,
                 a5, a4, a4, a4, a5, a4, a4, a4, 0.,
                 a4, a4, a4, a4, a4, a4, a4, a4, 0.,
                 a4, a4, a4, a4, a4, a4, a4, 0., 0.,
                 a4, a4, a4, a4, a4, a4, 0., 0., 0.,
                 a4, a4, a4, a4, 0., 0., 0., 0., 0.,

                 # Fuel + Control Rods
                 a3, a2, a2, a2, a3, a2, a2, a1, a4, # <- Half Assemblies
                 a2, a2, a2, a2, a2, a2, a2, a1, a4,
                 a2, a2, a3, a2, a2, a2, a1, a1, a4,
                 a2, a2, a2, a2, a2, a2, a1, a4, a4,
                 a3, a2, a2, a2, a3, a1, a1, a4, 0.,
                 a2, a2, a2, a2, a1, a1, a4, a4, 0.,
                 a2, a2, a1, a1, a1, a4, a4, 0., 0.,
                 a1, a1, a1, a4, a4, a4, 0., 0., 0.,
                 a4, a4, a4, a4, 0., 0., 0., 0., 0.,

                 # Fuel
                 a3, a2, a2, a2, a3, a2, a2, a1, a4, # <- Half Assemblies
                 a2, a2, a2, a2, a2, a2, a2, a1, a4,
                 a2, a2, a2, a2, a2, a2, a1, a1, a4,
                 a2, a2, a2, a2, a2, a2, a1, a4, a4,
                 a3, a2, a2, a2, a3, a1, a1, a4, 0.,
                 a2, a2, a2, a2, a1, a1, a4, a4, 0.,
                 a2, a2, a1, a1, a1, a4, a4, 0., 0.,
                 a1, a1, a1, a4, a4, a4, 0., 0., 0.,
                 a4, a4, a4, a4, 0., 0., 0., 0., 0.,
                
                 # Bottom Reflector
                 a4, a4, a4, a4, a4, a4, a4, a4, a4, # <- Half Assemblies
                 a4, a4, a4, a4, a4, a4, a4, a4, a4,
                 a4, a4, a4, a4, a4, a4, a4, a4, a4,
                 a4, a4, a4, a4, a4, a4, a4, a4, a4,
                 a4, a4, a4, a4, a4, a4, a4, a4, 0.,
                 a4, a4, a4, a4, a4, a4, a4, a4, 0.,
                 a4, a4, a4, a4, a4, a4, a4, 0., 0.,
                 a4, a4, a4, a4, a4, a4, 0., 0., 0.,
                 a4, a4, a4, a4, 0., 0., 0., 0., 0.
                ]

        dx = np.array([10., 20., 20., 20., 20., 20., 20., 20., 20.])
        nx = np.array([4,   8,   8,   8,   8,   8,   8,   8,   8])

        dy = np.array([20., 20., 20., 20., 20., 20., 20., 20., 10.])
        ny = np.array([8,   8,   8,   8,   8,   8,   8,   8,   4])

        dz = np.array([20., 13*20., 4*20., 20.])
        nz = 5*np.array([1,   13,  4,   1])

        geom = DiffusionGeometry(tiles, dx, nx, dy, ny, dz, nz, 1., 0., 0., 1., 0., 0.)
        solver = FDDiffusionDriver(geom)
        solver.solve()
        flux, _, _, _ = solver.flux()
        
        # Check flux at all corner points
        assert flux[0, 0, 0, 0] == pytest.approx(1.9845624413472607e-06, rel=1.E-5)
        assert flux[0, 0, 0, -1] == pytest.approx(1.4070637948186832e-06, rel=1.E-5)
        assert flux[0, 0, -1, 0] == pytest.approx(5.2278351543342474e-05, rel=1.E-5)
        assert flux[0, 0, -1, -1] == pytest.approx(2.277865538320556e-05, rel=1.E-5)
        assert flux[0, -1, 0, 0] == 0.0
        assert flux[0, -1, 0, -1] == 0.0
        assert flux[0, -1, -1, 0] == pytest.approx(1.984562441347691e-06, rel=1.E-5)
        assert flux[0, -1, -1, -1] == pytest.approx(1.4070637948182364e-06, rel=1.E-5)
        assert flux[1, 0, 0, 0] == pytest.approx(4.653675341160034e-06, rel=1.E-5)
        assert flux[1, 0, 0, -1] == pytest.approx(3.299276626607535e-06, rel=1.E-5)
        assert flux[1, 0, -1, 0] == pytest.approx(0.0001797643235206193, rel=1.E-5)
        assert flux[1, 0, -1, -1] == pytest.approx(1.6918463582272728e-05, rel=1.E-5)
        assert flux[1, -1, 0, 0] == 0.0
        assert flux[1, -1, 0, -1] == 0.0
        assert flux[1, -1, -1, 0] == pytest.approx(4.653675341158756e-06, rel=1.E-5)
        assert flux[1, -1, -1, -1] == pytest.approx(3.2992766266088287e-06, rel=1.E-5)

