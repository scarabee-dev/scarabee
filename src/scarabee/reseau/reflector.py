from .._scarabee import (
    NDLibrary,
    Material,
    CrossSection,
    DiffusionCrossSection,
    DiffusionData,
    ReflectorSN,
    DiffusionGeometry,
    NEM4DiffusionDriver,
    FormFactors,
    set_logging_level,
    scarabee_log,
    LogLevel,
)
from .nodal_flux import NodalFlux1D
import numpy as np

# import matplotlib.pyplot as plt


class Reflector:
    """
    A Reflector instance is responsible for performing transport calculations
    necessary to produce few-group cross sections for the reflector of an LWR.
    The core baffle cross sections are self-shielded as an infinite slab,
    using the Roman two-term approximation. The calculation is performed using
    a 1D Sn simulation, in the group structure of the nuclear data library.
    This removes the need to obtain a fine-group spectrum that would be used
    to condense to an intermediate group structure.

    Parameters
    ----------
    fuel : CrossSection
        Homogenized cross section which is representative of a pin cell.
        This is typically obtained from a previous lattice calcualtion.
    moderator : CrossSection
        Material cross sections for the moderator at desired temperature
        and density.
    assembly_width : float
        Width of a single fuel assembly (and the reflector to be modeled).
    gap_width : float
        Width of the moderator gap between the assembly and the core baffle.
    baffle_width : float
        Width of the core baffle.
    baffle : Material
        Material for the core baffle at desired temperature and density.
    ndl : NDLibrary
        Nuclear data library for constructing the baffle cross sections.

    Attributes
    ----------
    condensation_scheme : list of pairs of ints
        Defines how the energy groups will be condensed to the few-group
        structure used in nodal calculations.
    fuel : CrossSection
        Cross sections for a homogenized fuel assembly.
    moderator : CrossSection
        Cross sections for the moderator.
    assembly_width : float
        Width of fuel assembly and reflector.
    gap_width : float
        Width of the moderator gap between a fuel assembly and the core baffle.
    baffle_width : float
        Width of the core baffle.
    baffle : CrossSection
        Self-shielded cross sections for the core baffle.
    nangles : int
        Number of angles used in the Sn solver. Default is 16. Must be one of:
        2, 4, 6, 8, 10, 12, 14, 16, 32, 64, 128.
    anisotropic : bool
        If True, the reflector calculation is performed with explicit
        anisotropic scattering. Otherwise, the transport correction with
        isotropic scattering is used. Default value is False.
    keff_tolerance : float
        Convergence criteria for keff. Default is 1.E-5.
    flux_tolerance : float
        Convergence criteria for the flux. Default is 1.E-5.
    diffusion_xs : DiffusionCrossSection
        The few-group diffusion group constants for the reflector.
    adf : ndarray
        The assembly discontinuity factors.
    diffusion_data : DiffusionData
        The few-group diffusion cross sections and ADFs for the reflector.
    """

    def __init__(
        self,
        fuel: CrossSection,
        moderator: CrossSection,
        assembly_width: float,
        gap_width: float,
        baffle_width: float,
        baffle: Material,
        ndl: NDLibrary,
    ):
        self.fuel = fuel
        self.fuel.name = "Fuel"
        self.moderator = moderator
        self.moderator.name = "Moderator"
        self.assembly_width = assembly_width
        self.gap_width = gap_width
        self.baffle_width = baffle_width
        self.condensation_scheme = ndl.condensation_scheme
        self.nangles = 16
        self.anisotropic = False
        self.adf = None
        self.diffusion_xs = None
        self.diffusion_data = None

        if self.baffle_width > 0.0:
            # No Dancoff correction, as looking at 1D isolated slab for baffle
            Ee = 1.0 / (2.0 * self.baffle_width)
            self.baffle = baffle.roman_xs(0.0, Ee, ndl)
            self.baffle.name = "Baffle"
        else:
            self.baffle = None

        self.keff_tolerance = 1.0e-6
        self.flux_tolerance = 1.0e-6

        if self.gap_width + self.baffle_width > self.assembly_width:
            raise RuntimeError(
                "The assembly width is smaller than the sum of the gap and baffle widths."
            )

    def _doctor_xs(self, xs: DiffusionCrossSection) -> DiffusionCrossSection:
        # First, get raw arrays from the cross section
        NG = xs.ngroups
        D = np.zeros(NG)
        Ea = np.zeros(NG)
        Ef = np.zeros(NG)
        vEf = np.zeros(NG)
        chi = np.zeros(NG)
        Es = np.zeros((NG, NG))
        for g in range(NG):
            D[g] = xs.D(g)
            Ea[g] = xs.Ea(g)
            Ef[g] = xs.Ef(g)
            vEf[g] = xs.vEf(g)
            chi[g] = xs.chi(g)
            for gg in range(NG):
                Es[g, gg] = xs.Es(g, gg)

        D *= 1.01

        return DiffusionCrossSection(D, Ea, Es, Ef, vEf, chi)

    def solve(self) -> None:
        """
        Runs a 1D problem to generate few group cross sections for the
        reflector, with the core baffle.
        """
        scarabee_log(LogLevel.Info, "Starting reflector calculation.")

        if self.condensation_scheme is None:
            raise RuntimeError(
                "Cannot perform reflector calculation without condensation scheme."
            )

        # We start by making a ReflectorSn to do 1D calculation
        dx = []
        mats = []

        # We add 1 fuel assembly worth of homogenized core
        NF = 15 * 17
        dr = self.assembly_width / float(NF)
        dx += [dr] * NF
        mats += [self.fuel] * NF

        # We now add the gap (if present)
        if self.gap_width == 0.0:
            NG = 0
        else:
            NG = int(self.gap_width / 0.1) + 1
            if NG < 5:
                NG = 5
            dr = self.gap_width / float(NG)
            dx += [dr] * NG
            mats += [self.moderator] * NG

        # Now we add regions for the baffle
        if self.baffle_width == 0.0:
            NB = 0
        else:
            NB = int(self.baffle_width / 0.1) + 1
            dr = self.baffle_width / float(NB)
            dx += [dr] * NB
            mats += [self.baffle] * NB

        # Now we add the outer water reflector regions
        ref_width = self.assembly_width - self.gap_width - self.baffle_width
        if ref_width <= 0.0:
            # Case of a heavy reflector
            NR = 0
        else:
            NR = int(ref_width / 0.1) + 1
            dr = ref_width / float(NR)
            dx += [dr] * NR
            mats += [self.moderator] * NR

        # Setup the Sn solver and run
        ref_sn = ReflectorSN(mats, dx, self.nangles, self.anisotropic)
        set_logging_level(LogLevel.Warning)
        ref_sn.solve()
        set_logging_level(LogLevel.Info)
        scarabee_log(LogLevel.Info, "")
        scarabee_log(LogLevel.Info, "SN  Keff: {:.5f}".format(ref_sn.keff))

        # Here we compute the homogenized and condensed cross sections
        ref_homog_xs = ref_sn.homogenize(list(range(NF, NF + NG + NB + NR)))
        ref_homog_spec = ref_sn.homogenize_flux_spectrum(
            list(range(NF, NF + NG + NB + NR))
        )
        ref_homog_diff_xs = ref_homog_xs.diffusion_xs()
        self.diffusion_xs = ref_homog_diff_xs.condense(
            self.condensation_scheme, ref_homog_spec
        )

        fuel_homog_xs = ref_sn.homogenize(list(range(0, NF)))
        fuel_homog_spec = ref_sn.homogenize_flux_spectrum(list(range(0, NF)))
        fuel_homog_diff_xs = fuel_homog_xs.diffusion_xs()
        fuel_diffusion_xs = fuel_homog_diff_xs.condense(
            self.condensation_scheme, fuel_homog_spec
        )

        # Condense the few-group flux along entire 1D geometry
        few_group_flux = np.zeros((len(self.condensation_scheme), NF + NG + NB + NR))
        for i in range(NF + NG + NB + NR):
            for G in range(len(self.condensation_scheme)):
                g_min = self.condensation_scheme[G][0]
                g_max = self.condensation_scheme[G][1]
                for g in range(g_min, g_max + 1):
                    few_group_flux[G, i] += ref_sn.flux(i, g)
        dx = np.array(dx)

        # Make an array with the mid-point x-values of each bin from Sn simulation
        x = np.zeros(len(dx))
        for i in range(len(dx)):
            if i == 0:
                x[0] = 0.5 * dx[0]
            else:
                x[i] = x[i - 1] + 0.5 * (dx[i - 1] + dx[i])

        # Obtain net currents at node boundaries and average flux
        avg_flx_ref = np.zeros(len(self.condensation_scheme))
        avg_flx_fuel = np.zeros(len(self.condensation_scheme))
        j_0 = np.zeros(len(self.condensation_scheme))
        j_mid = np.zeros(len(self.condensation_scheme))
        j_max = np.zeros(len(self.condensation_scheme))
        s_mid = NF
        s_max = ref_sn.nsurfaces - 1
        len_fuel = np.sum(dx[:NF])
        len_ref = np.sum(dx[NF:])
        for g in range(len(self.condensation_scheme)):
            avg_flx_fuel[g] = np.sum(few_group_flux[g, :NF] * dx[:NF]) / len_fuel
            avg_flx_ref[g] = np.sum(few_group_flux[g, NF:] * dx[NF:]) / len_ref

            g_min = self.condensation_scheme[g][0]
            g_max = self.condensation_scheme[g][1]
            for gg in range(g_min, g_max + 1):
                j_0[g] += ref_sn.current(0, gg)
                j_mid[g] += ref_sn.current(s_mid, gg)
                j_max[g] += ref_sn.current(s_max, gg)

        # Do nodal calculation to obtain homogeneous flux
        x_fuel = np.sum(dx[:NF])
        x_ref_end = np.sum(dx)

        fuel_node = NodalFlux1D(
            0.0, x_fuel, ref_sn.keff, fuel_diffusion_xs, avg_flx_fuel, j_0, j_mid
        )

        ref_node = NodalFlux1D(
            x_fuel, x_ref_end, ref_sn.keff, self.diffusion_xs, avg_flx_ref, j_mid, j_max
        )

        # I noticed that the initial fixed-source claculation from above can result in
        # a negative flux next to the vacuum BC, which is of course non-physical. Here,
        # we iteratively bump up the diffusion coefficients until the fluxes are
        # positive at the far right boundary in all groups.
        xmax_homog_fluxes = ref_node.pos_surf_flux()
        while np.min(xmax_homog_fluxes) < 0.0:
            self.diffusion_xs = self._doctor_xs(self.diffusion_xs)
            ref_node = NodalFlux1D(
                x_fuel,
                x_ref_end,
                ref_sn.keff,
                self.diffusion_xs,
                avg_flx_ref,
                j_mid,
                j_max,
            )
            xmax_homog_fluxes = ref_node.pos_surf_flux()

        # Compute DFs
        heter_flx = 0.5 * (few_group_flux[:, NF - 1] + few_group_flux[:, NF])
        homog_flx_fuel = fuel_node.pos_surf_flux()
        homog_flx_ref = ref_node.neg_surf_flux()

        # Get DF for fuel and reflector sides
        f_fuel = heter_flx / homog_flx_fuel
        f_ref = heter_flx / homog_flx_ref

        # As outlined by Smith in [1], we should normalize the reflector DFs by
        # The fuel DFs from the reflector calculation. Then, in the nodal
        # calculation, we can multiply by the DF of the actual fuel next to the
        # node to get better results.
        f_ref /= f_fuel
        f_fuel /= f_fuel

        nodal_flux = np.zeros((len(self.condensation_scheme), len(x)))
        self.adf = np.ones((len(self.condensation_scheme), 6))
        fuel_adf = np.ones((len(self.condensation_scheme), 6))
        for g in range(len(self.condensation_scheme)):
            nodal_flux[g, :NF] = fuel_node(x[:NF], g)
            nodal_flux[g, NF:] = ref_node(x[NF:], g)
            self.adf[g, :] = f_ref[g]
            fuel_adf[g, :] = f_fuel[g]

        # Create the diffusion data
        self.diffusion_data = DiffusionData(self.diffusion_xs)
        self.diffusion_data.adf = self.adf
        self.diffusion_data.reflector = True
        self.form_factors = FormFactors(
            np.array([[0.0]]),
            np.array([self.assembly_width]),
            np.array([self.assembly_width]),
        )

        # Do a nodal k-eff calulation
        fuel_diffision_data = DiffusionData(fuel_diffusion_xs)
        fuel_diffision_data.adf = fuel_adf

        dx = np.array([1 * self.assembly_width, self.assembly_width])
        nx = np.array([1, 1])
        dy = np.array([10.0])
        ny = np.array([1])

        nem_geom = DiffusionGeometry(
            [fuel_diffision_data, self.diffusion_data],
            dx,
            nx,
            dy,
            ny,
            dy,
            ny,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
        )
        nem_solver = NEM4DiffusionDriver(nem_geom)
        nem_solver.nonlinear_update_frequency = 1
        nem_solver.source_extrapolation_frequency = 100
        nem_solver.flux_tolerance = 1.0e-7
        set_logging_level(LogLevel.Warning)
        nem_solver.solve()
        set_logging_level(LogLevel.Info)
        scarabee_log(LogLevel.Info, "NEM Keff: {:.5f}".format(nem_solver.keff))

        nem_keff_flux = nem_solver.flux(x, [10.0], [10.0])
        flux_norm = np.sum(few_group_flux[:, :NF])
        nem_keff_flux *= flux_norm / np.sum(nem_keff_flux[:, :NF])

        # Plot flux for each group
        # for g in range(len(self.condensation_scheme)):
        #    plt.plot(x, few_group_flux[g, :], label="Sn")
        #    plt.plot(x, nodal_flux[g, :], label="NEM Fixed-Source")
        #    plt.plot(x, nem_keff_flux[g, :, 0, 0], label="NEM keff")
        #    plt.xlabel("x [cm]")
        #    plt.ylabel("Flux [Arb. Units]")
        #    plt.title("Group {:}".format(g))
        #    plt.legend().set_draggable(True)
        #    plt.tight_layout()
        #    plt.show()


# [1] K. S. Smith, “Nodal diffusion methods and lattice physics data in LWR
#     analyses: Understanding numerous subtle details,” Prog Nucl Energ,
#     vol. 101, pp. 360–369, 2017, doi: 10.1016/j.pnucene.2017.06.013.
