from .._scarabee import DiffusionCrossSection
import numpy as np
from typing import Optional, Union


class NodalFlux1D:
    def __init__(
        self,
        x_min: float,
        x_max: float,
        keff: float,
        xs: DiffusionCrossSection,
        avg_flx: np.ndarray,
        j_neg: np.ndarray,
        j_pos: np.ndarray,
    ):
        """
        Performs the 1D nodal diffusion calculation using the Nodal Expansion
        Method (NEM) with reference currents and average fluxes. This
        represents the transverse integrated homogeneous nodal flux solution.
        The primary use of this class is for computing the homogeneous flux
        that is required to define discontinuity factors for generalized
        equivalence theory.

        Paramters
        ---------
        x_min : float
             Position of the negative surface of the node.
        x_max : float
             Position of the positive surface of the node.
        keff : float
             Multiplication factor from reference Sn calculation.
        xs : DiffusionCrossSection
             Diffusion cross sections homogenized for the node.
        avg_flx : np.ndarray
             Average flux in each group within the node from reference Sn
             calculation.
        j_neg : np.ndarray
             Reference net current in each group on the negative boundary
             from the reference calculation.
        j_pos : np.ndarray
             Reference net current in each group on the positive boundary
             from the reference calculation.
        """
        if x_min >= x_max:
            raise ValueError("The value of x_min must be < x_max.")
        self.x_min = x_min
        self.x_max = x_max
        dx = x_max - x_min  # Only needed locally to compute coefficients

        if keff <= 0.0 or 2.0 <= keff:
            raise ValueError("The value of keff must be in the interval (0, 2).")

        self.ngroups = xs.ngroups

        # Checks for all arrays
        if avg_flx.ndim != 1:
            raise ValueError("The array avg_flux must be 1D.")
        if j_neg.ndim != 1:
            raise ValueError("The array j_neg must be 1D.")
        if j_pos.ndim != 1:
            raise ValueError("The array j_pos must be 1D.")

        if avg_flx.size != xs.ngroups:
            raise ValueError(
                "The length of avg_flux does not agree with number of groups in xs."
            )
        if j_neg.size != xs.ngroups:
            raise ValueError(
                "The length of j_neg does not agree with number of groups in xs."
            )
        if j_pos.size != xs.ngroups:
            raise ValueError(
                "The length of j_neg does not agree with number of groups in xs."
            )

        # Perform the static nodal calculation for all groups
        NG = self.ngroups
        Na = NG * 4  # number of a coefficients to solve for
        invs_dx = 1.0 / dx
        A = np.zeros((Na, Na))  # Matrix to hold all coefficients
        b = np.zeros((Na))  # Results array

        # Load the coefficient matrix and results array
        j = 0  # Matrix row
        for g in range(NG):
            Dg = xs.D(g)
            Dg_dx = Dg * invs_dx
            # Use this for Et instead of 1/(3D), as that gave bad results.
            Et = xs.Ea(g) + xs.Es(g)
            chi_g_keff = xs.chi(g) / keff
            Erf_g = Et - xs.Es(g, g) - chi_g_keff * xs.vEf(g)

            # Each group has 4 equations. See reference [1].

            # Eq 2.54
            A[j, g * 4 + 2] -= 0.5 * Dg_dx * invs_dx
            A[j, g * 4 + 0] += Erf_g / 12.0
            A[j, g * 4 + 2] -= 0.1 * (Erf_g / 12.0)
            for gg in range(NG):
                if gg == g:
                    continue
                A[j, gg * 4 + 0] -= xs.Es(gg, g) / 12.0
                A[j, gg * 4 + 2] += 0.1 * (xs.Es(gg, g) / 12.0)
                A[j, gg * 4 + 0] -= chi_g_keff * (xs.vEf(gg) / 12.0)
                A[j, gg * 4 + 2] += 0.1 * chi_g_keff * (xs.vEf(gg) / 12.0)
            b[j] = 0.0
            j += 1

            # Eq 2.55
            A[j, g * 4 + 3] -= 0.2 * Dg_dx * invs_dx
            A[j, g * 4 + 1] += Erf_g / 20.0
            A[j, g * 4 + 3] -= Erf_g / (20.0 * 35.0)
            for gg in range(NG):
                if gg == g:
                    continue
                A[j, gg * 4 + 1] -= xs.Es(gg, g) / 20.0
                A[j, gg * 4 + 3] += xs.Es(gg, g) / (20.0 * 35.0)
                A[j, gg * 4 + 1] -= chi_g_keff * (xs.vEf(gg) / 20.0)
                A[j, gg * 4 + 3] += chi_g_keff * (xs.vEf(gg) / (20.0 * 35.0))
            b[j] = 0.0
            j += 1

            # Eq 2.57 and 2.58 have an error in [1]. They are both missing a
            # factor of 3 on the a2 term. The correct form can be found from
            # Lawrence in [2].
            # Eq 2.57
            A[j, g * 4 + 0] -= Dg_dx
            A[j, g * 4 + 1] += 3.0 * Dg_dx
            A[j, g * 4 + 2] -= 0.5 * Dg_dx
            A[j, g * 4 + 3] += 0.2 * Dg_dx
            b[j] = j_neg[g]
            j += 1

            # Eq 2.58
            A[j, g * 4 + 0] -= Dg_dx
            A[j, g * 4 + 1] -= 3.0 * Dg_dx
            A[j, g * 4 + 2] -= 0.5 * Dg_dx
            A[j, g * 4 + 3] -= 0.2 * Dg_dx
            b[j] = j_pos[g]
            j += 1

        a_tmp = np.linalg.solve(A, b)
        self.a = np.zeros((NG, 5))
        for g in range(NG):
            self.a[g, 0] = avg_flx[g]
            self.a[g, 1:] = a_tmp[g * 4 : g * 4 + 4]

    def __call__(self, x: float, g: int) -> float:
        if g < 0:
            raise RuntimeError("Group index g must be >= 0.")

        if g >= self.ngroups:
            raise RuntimeError("Group index g is out of range.")

        # Commented out to permit use with an array
        """
        if x < self.x_min:
            raise RuntimeError("x value is less than lower boundary.")
        
        if x > self.x_max:
            raise RuntimeError("x value is greater than upper boundary.")
        """

        u = ((x - self.x_min) / (self.x_max - self.x_min)) - 0.5
        u2 = u * u
        f1 = u
        f2 = 3.0 * u2 - 0.25
        f3 = (u2 - 0.25) * u
        f4 = (u2 - 0.25) * (u2 - 0.05)

        flx = (
            self.a[g, 0]
            + self.a[g, 1] * f1
            + self.a[g, 2] * f2
            + self.a[g, 3] * f3
            + self.a[g, 4] * f4
        )

        return flx

    def pos_surf_flux(self, g: Optional[int] = None) -> Union[np.ndarray, float]:
        if g is None:
            return self.a[:, 0] + 0.5 * self.a[:, 1] + 0.5 * self.a[:, 2]

        if g < 0:
            raise RuntimeError("Group index g must be >= 0.")

        if g >= self.ngroups:
            raise RuntimeError("Group index g is out of range.")

        return self.a[g, 0] + 0.5 * self.a[g, 1] + 0.5 * self.a[g, 2]

    def neg_surf_flux(self, g: Optional[int] = None) -> Union[np.ndarray, float]:
        if g is None:
            return self.a[:, 0] - 0.5 * self.a[:, 1] + 0.5 * self.a[:, 2]

        if g < 0:
            raise RuntimeError("Group index g must be >= 0.")

        if g >= self.ngroups:
            raise RuntimeError("Group index g is out of range.")

        return self.a[g, 0] - 0.5 * self.a[g, 1] + 0.5 * self.a[g, 2]


class NodalFlux2D:
    def __init__(
        self,
        dx: float,
        dy: float,
        keff: float,
        xs: DiffusionCrossSection,
        avg_flx: np.ndarray,
        j_x_neg: np.ndarray,
        j_x_pos: np.ndarray,
        j_y_neg: np.ndarray,
        j_y_pos: np.ndarray,
    ):
        """
        Performs the 2D nodal diffusion calculation using the Nodal Expansion
        Method (NEM) with reference currents and average fluxes. This
        represents the homogeneous nodal flux solution. The primary use of this
        class is for computing the homogeneous flux that is required to define
        corner discontinuity factors for flux reconstruction.

        By default, the coupled x-y terms are neglected, as this requires
        knowledge of the average corner fluxes based on the adjacent nodes. Due
        to this restriction, the coefficients for the coupled x-y terms must be
        computed separately. Examine the source code in
        NEMDriver::fit_node_recon_params_corners for an example of how this
        should be done. It follows the ANOVA-HDMR Decomposition method [3].

        Paramters
        ---------
        dx : float
             Width of the node along the x axis.
        dy : float
             Width of the node along the y axis.
        keff : float
             Multiplication factor from reference Sn calculation.
        xs : DiffusionCrossSection
             Diffusion cross sections homogenized for the node.
        avg_flx : np.ndarray
             Average flux in each group within the node from reference Sn
             calculation.
        j_x_neg : np.ndarray
             Reference net current in each group on the negative x boundary
             from the reference calculation.
        j_x_pos : np.ndarray
             Reference net current in each group on the positive x boundary
             from the reference calculation.
        j_y_neg : np.ndarray
             Reference net current in each group on the negative y boundary
             from the reference calculation.
        j_y_pos : np.ndarray
             Reference net current in each group on the positive y boundary
             from the reference calculation.
        """
        self.keff = keff
        self.ngroups = xs.ngroups
        NG = self.ngroups

        self.phi_0 = avg_flx  # f0
        self.eps = np.zeros(NG)

        # fx
        self.ax0 = np.zeros(NG)
        self.ax1 = np.zeros(NG)
        self.ax2 = np.zeros(NG)
        self.bx1 = np.zeros(NG)
        self.bx2 = np.zeros(NG)

        # fy
        self.ay0 = np.zeros(NG)
        self.ay1 = np.zeros(NG)
        self.ay2 = np.zeros(NG)
        self.by1 = np.zeros(NG)
        self.by2 = np.zeros(NG)

        # fxy
        self.cxy11 = np.zeros(NG)
        self.cxy12 = np.zeros(NG)
        self.cxy21 = np.zeros(NG)
        self.cxy22 = np.zeros(NG)

        self.invs_dx = 1.0 / dx
        self.invs_dy = 1.0 / dy
        self.zeta_x = np.zeros(NG)
        self.zeta_y = np.zeros(NG)

        self.flux_x = NodalFlux1D(
            -0.5 * dx, 0.5 * dx, self.keff, xs, avg_flx, j_x_neg, j_x_pos
        )
        self.flux_y = NodalFlux1D(
            -0.5 * dy, 0.5 * dy, self.keff, xs, avg_flx, j_y_neg, j_y_pos
        )

        self._initialize_params_no_cross_terms(xs, j_x_pos, j_x_neg, j_y_pos, j_y_neg)

    @property
    def dx(self) -> float:
        return 1.0 / self.invs_dx

    @property
    def dy(self) -> float:
        return 1.0 / self.invs_dy

    def _initialize_params_no_cross_terms(
        self, xs: DiffusionCrossSection, Jxp, Jxm, Jyp, Jym
    ) -> None:
        sinhc = lambda x: np.sinh(x) / x

        for g in range(self.ngroups):
            D = xs.D(g)
            Er = xs.Er(g)
            self.eps[g] = np.sqrt(Er / D)

            flx_avg = self.phi_0[g]
            flx_xp = self.flux_x.pos_surf_flux(g)
            flx_xm = self.flux_x.neg_surf_flux(g)
            flx_yp = self.flux_y.pos_surf_flux(g)
            flx_ym = self.flux_y.neg_surf_flux(g)

            # Initial base matrix for finding fx, fy, and fz coefficients
            M = np.array(
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, -1.0, 1.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 1.0, -3.0],
                ]
            )
            b = np.zeros(4)

            # Determine fx coefficients
            zeta_x = 0.5 * self.eps[g] * self.dx
            M[0, 0] = np.cosh(zeta_x) - sinhc(zeta_x)
            M[0, 1] = np.sinh(zeta_x)
            M[1, 0] = M[0, 0]
            M[1, 1] = -M[0, 1]
            M[2, 0] = zeta_x * np.sinh(zeta_x)
            M[2, 1] = zeta_x * np.cosh(zeta_x)
            M[3, 0] = -M[2, 0]
            M[3, 1] = M[2, 1]
            b[0] = flx_xp - flx_avg
            b[1] = flx_xm - flx_avg
            b[2] = -0.5 * Jxp[g] * self.dx / D
            b[3] = -0.5 * Jxm[g] * self.dx / D
            fu_coeffs = np.linalg.solve(M, b)
            self.ax1[g] = fu_coeffs[0]
            self.ax2[g] = fu_coeffs[1]
            self.bx1[g] = fu_coeffs[2]
            self.bx2[g] = fu_coeffs[3]
            self.ax0[g] = -self.ax1[g] * sinhc(zeta_x)
            self.zeta_x[g] = zeta_x

            # Determine fy coefficients
            zeta_y = 0.5 * self.eps[g] * self.dy
            M[0, 0] = np.cosh(zeta_y) - sinhc(zeta_y)
            M[0, 1] = np.sinh(zeta_y)
            M[1, 0] = M[0, 0]
            M[1, 1] = -M[0, 1]
            M[2, 0] = zeta_y * np.sinh(zeta_y)
            M[2, 1] = zeta_y * np.cosh(zeta_y)
            M[3, 0] = -M[2, 0]
            M[3, 1] = M[2, 1]
            b[0] = flx_yp - flx_avg
            b[1] = flx_ym - flx_avg
            b[2] = -0.5 * Jyp[g] * self.dy / D
            b[3] = -0.5 * Jym[g] * self.dy / D
            fu_coeffs = np.linalg.solve(M, b)
            self.ay1[g] = fu_coeffs[0]
            self.ay2[g] = fu_coeffs[1]
            self.by1[g] = fu_coeffs[2]
            self.by2[g] = fu_coeffs[3]
            self.ay0[g] = -self.ay1[g] * sinhc(zeta_y)
            self.zeta_y[g] = zeta_y

    def __call__(self, x, y) -> np.ndarray:
        return self.phi_0 + self.fx(x) + self.fy(y) + self.fxy(x, y)

    def flux_xy_no_cross(self, x, y) -> np.ndarray:
        return self.phi_0 + self.fx(x) + self.fy(y)

    def fx(self, x) -> np.ndarray:
        return (
            self.ax0
            + self.ax1 * np.cosh(self.eps * x)
            + self.ax2 * np.sinh(self.eps * x)
            + self.bx1 * self.p1(2.0 * x * self.invs_dx)
            + self.bx2 * self.p2(2.0 * x * self.invs_dx)
        )

    def fy(self, y) -> np.ndarray:
        return (
            self.ay0
            + self.ay1 * np.cosh(self.eps * y)
            + self.ay2 * np.sinh(self.eps * y)
            + self.by1 * self.p1(2.0 * y * self.invs_dy)
            + self.by2 * self.p2(2.0 * y * self.invs_dy)
        )

    def fxy(self, x, y) -> np.ndarray:
        x *= 2.0 * self.invs_dx
        y *= 2.0 * self.invs_dy
        p1x = self.p1(x)
        p2x = self.p2(x)
        p1y = self.p1(y)
        p2y = self.p2(y)
        return (
            self.cxy11 * p1x * p1y
            + self.cxy12 * p1x * p2y
            + self.cxy21 * p2x * p1y
            + self.cxy22 * p2x * p2y
        )

    def p1(self, xi) -> float:
        return xi

    def p2(self, xi) -> float:
        return 0.5 * (3.0 * xi * xi - 1.0)


# REFERENCES
#
# [1] S. Machach, “Étude des techniques d’équivalence nodale appliquées aux
#     modèles de réflecteurs dans les réacteurs à eau pressurisée,”
#     Polytechnique Montréal, 2022.
#
# [2] R. D. Lawrence, “Progress in nodal methods for the solution of the
#     neutron diffusion and transport equations,” Prog. Nucl. Energ., vol. 17,
#     no. 3, pp. 271–301, 1986, doi: 10.1016/0149-1970(86)90034-x.
#
# [3] P. M. Bokov, D. Botes, R. H. Prinsloo, and D. I. Tomašević, “A
#     Multigroup Homogeneous Flux Reconstruction Method Based on the
#     ANOVA-HDMR Decomposition,” Nucl. Sci. Eng., vol. 197, no. 2,
#     pp. 308–332, 2023, doi: 10.1080/00295639.2022.2108654.
