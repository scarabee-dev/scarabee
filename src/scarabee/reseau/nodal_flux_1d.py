from .._scarabee import DiffusionCrossSection
import numpy as np


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
             from the reference Sn calculation.
        j_pos : np.ndarray
             Reference net current in each group on the positive boundary
             from the reference Sn calculation.
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

    def pos_surf_flux(self, g: int) -> float:
        if g < 0:
            raise RuntimeError("Group index g must be >= 0.")

        if g >= self.ngroups:
            raise RuntimeError("Group index g is out of range.")

        return self.a[g, 0] + 0.5 * self.a[g, 1] + 0.5 * self.a[g, 2]

    def neg_surf_flux(self, g: int) -> float:
        if g < 0:
            raise RuntimeError("Group index g must be >= 0.")

        if g >= self.ngroups:
            raise RuntimeError("Group index g is out of range.")

        return self.a[g, 0] - 0.5 * self.a[g, 1] + 0.5 * self.a[g, 2]


# REFERENCES
#
# [1] S. Machach, “Étude des techniques d’équivalence nodale appliquées aux
#     modèles de réflecteurs dans les réacteurs à eau pressurisée,”
#     Polytechnique Montréal, 2022.
#
# [2] R. D. Lawrence, “Progress in nodal methods for the solution of the
#     neutron diffusion and transport equations,” Prog. Nucl. Energ., vol. 17,
#     no. 3, pp. 271–301, 1986, doi: 10.1016/0149-1970(86)90034-x.
