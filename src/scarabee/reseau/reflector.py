from .._scarabee import (
    NDLibrary,
    Material,
    CrossSection,
    DiffusionCrossSection,
    DiffusionData,
    ReflectorSN,
    set_logging_level,
    scarabee_log,
    LogLevel,
)
from .nodal_flux import NodalFlux1D

import numpy as np
#import matplotlib.pyplot as plt


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

        # No Dancoff correction, as looking at 1D isolated slab for baffle
        Ee = 1.0 / (2.0 * self.baffle_width)
        self.baffle = baffle.roman_xs(0.0, Ee, ndl)
        self.baffle.name = "Baffle"

        self.keff_tolerance = 1.0e-5
        self.flux_tolerance = 1.0e-5

        if self.gap_width + self.baffle_width >= self.assembly_width:
            raise RuntimeError(
                "The assembly width is smaller than the sum of the gap and baffle widths."
            )

    def solve(self) -> None:
        """
        Runs a 1D annular problem to generate few group cross sections for the
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

        # We add 2 fuel assemblies worth of homogenized core
        NF = 2 * 10 * 17
        dr = 2.0 * self.assembly_width / (float(NF))
        dx += [dr] * NF
        mats += [self.fuel] * NF

        # We now add one ring for the gap
        NG = 3
        dr = self.gap_width / (float(NG))
        dx += [dr] * NG
        mats += [self.moderator] * NG

        # Now we add 20 regions for the baffle
        NB = 20
        dr = self.baffle_width / float(NB)
        dx += [dr] * NB
        mats += [self.baffle] * NB

        # Now we add the outer water reflector regions
        ref_width = self.assembly_width - self.gap_width - self.baffle_width
        NR = int(ref_width / 0.02) + 1
        dr = ref_width / float(NR)
        dx += [dr] * NR
        mats += [self.moderator] * NR

        ref_sn = ReflectorSN(mats, dx, self.nangles, self.anisotropic)
        set_logging_level(LogLevel.Warning)
        ref_sn.solve()
        set_logging_level(LogLevel.Info)
        scarabee_log(LogLevel.Info, "")
        scarabee_log(LogLevel.Info, "Kinf: {:.5f}".format(ref_sn.keff))
        scarabee_log(LogLevel.Info, "")
        scarabee_log(LogLevel.Info, "Generating diffusion data.")

        few_group_flux = np.zeros((len(self.condensation_scheme), NF + NG + NB + NR))
        for i in range(NF + 1 + NB + NR):
            for G in range(len(self.condensation_scheme)):
                g_min = self.condensation_scheme[G][0]
                g_max = self.condensation_scheme[G][1]
                for g in range(g_min, g_max + 1):
                    few_group_flux[G, i] += ref_sn.flux(i, g)
        dx = np.array(dx)
        x = np.zeros(len(dx))
        for i in range(len(dx)):
            if i == 0:
                x[0] = 0.5 * dx[0]
            else:
                x[i] = x[i - 1] + 0.5 * (dx[i - 1] + dx[i])

        # Here we compute the cross sections
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

        # Obtain net currents at node boundaries and average flux
        avg_flx_ref = np.zeros(len(self.condensation_scheme))
        avg_flx_fuel = np.zeros(len(self.condensation_scheme))
        j_0 = np.zeros(len(self.condensation_scheme))
        j_mid = np.zeros(len(self.condensation_scheme))
        j_max = np.zeros(len(self.condensation_scheme))
        s_mid = NF
        s_max = ref_sn.nsurfaces - 1
        for g in range(len(self.condensation_scheme)):
            avg_flx_fuel[g] = np.mean(few_group_flux[g, :NF])
            avg_flx_ref[g] = np.mean(few_group_flux[g, NF:])

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

        # Plot reference Sn flux and nodal flux
        # for g in range(len(self.condensation_scheme)):
        #    nodal_flux = np.zeros(len(x))
        #    nodal_flux[:NF] = fuel_node(x[:NF], g)
        #    nodal_flux[NF:] = ref_node(x[NF:], g)
        #    plt.plot(x, few_group_flux[g, :], label="Sn")
        #    plt.plot(x, nodal_flux, label="Nodal")
        #    plt.xlabel("x [cm]")
        #    plt.ylabel("Flux [Arb. Units]")
        #    plt.title("Group {:}".format(g))
        #    plt.show()

        # Compute the ADFs
        self.adf = np.zeros((len(self.condensation_scheme), 6))
        for G in range(len(self.condensation_scheme)):
            heter_flx_fuel = few_group_flux[G, NF - 1]
            homog_flx_fuel = fuel_node.pos_surf_flux(G)

            heter_flx_ref = few_group_flux[G, NF]
            homog_flx_ref = ref_node.neg_surf_flux(G)

            f_fuel = heter_flx_fuel / homog_flx_fuel
            f_ref = heter_flx_ref / homog_flx_ref

            # Normalize to fuel DF
            f_ref = f_ref / f_fuel

            self.adf[G, :] = f_ref

        # Create the diffusion data
        self.diffusion_data = DiffusionData(self.diffusion_xs)
        self.diffusion_data.adf = self.adf
