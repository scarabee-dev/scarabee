from .._scarabee import (
    NDLibrary,
    DepletionChain,
    MaterialComposition,
    Material,
    CrossSection,
    MOCDriver,
    MixingFraction,
    DepletionChain,
    DepletionMatrix,
    mix_materials,
    build_depletion_matrix,
)
import numpy as np
from typing import Optional, List, Tuple
import copy


class ControlRod:
    """
    Represents a control rod in a PWR which is inserted into an empty
    guide tube.

    Parameters
    ----------
    absorber : Material
        Material at the center of the control rod which absorbs neutrons.
    gap : Material
        Material which describes the gap between the absorber and cladding.
    clad : Material
        Material which describes the cladding.
    absorber_radius : float
        Outer radius of the absorber in the center of the control rod.
    gap_radius : float
        Outer radius of the gap
    clad_radius : float
        Outer radius of the outer cladding of the control rod.
    num_rings : int, default 1
        Number of rings which should be used to discretize the absorber.
        Each ring will be self-shielded and depleted separately.

    Attributes
    ----------
    absorber : Material
        Material at the center of the control rod which absorbs neutrons.
    gap : Material
        Material which describes the gap between the absorber and cladding.
    clad : Material
        Material which describes the cladding.
    absorber_radius : float
        Outer radius of the absorber in the center of the control rod.
    gap_radius : float
        Outer radius of the gap
    clad_radius : float
        Outer radius of the outer cladding of the control rod.
    num_rings : int, default 1
        Number of rings which should be used to discretize the absorber.
        Each ring will be self-shielded and depleted separately.
    is_moderator : bool
        Indicates if the control rod has been replaced with moderator.
    absorber_dancoff_correction : float
        Dancoff correction for the absorber.
    """

    def __init__(
        self,
        absorber: Material,
        gap: Material,
        clad: Material,
        absorber_radius: float,
        gap_radius: float,
        clad_radius: float,
        num_rings: int = 1,
    ):
        # Make sure all radii are sorted
        tmp_radii_list = [
            absorber_radius,
            gap_radius,
            clad_radius,
        ]
        if not sorted(tmp_radii_list):
            raise ValueError("Control rod radii are not sorted.")
        if absorber_radius <= 0:
            raise ValueError("All radii must be > 0.")

        if num_rings <= 0:
            raise ValueError("Number of rings must be >= 1.")
        self._num_rings = num_rings

        # Set materials
        self._absorber = copy.deepcopy(absorber)
        self._gap = copy.deepcopy(gap)
        self._clad = copy.deepcopy(clad)

        # Set radii
        self._absorber_radius = absorber_radius
        self._gap_radius = gap_radius
        self._clad_radius = clad_radius

        self._is_moderator = False

        # ======================================================================
        # DANCOFF CORRECTION CALCULATION DATA
        # ----------------------------------------------------------------------
        self._absorber_dancoff_correction: float = 0.0

        self._absorber_dancoff_xs: CrossSection = CrossSection(
            np.array([self.absorber.potential_xs]),
            np.array([self.absorber.potential_xs]),
            np.array([[0.0]]),
            "CR Absorber",
        )

        self._clad_dancoff_xs: CrossSection = CrossSection(
            np.array([self.clad.potential_xs]),
            np.array([self.clad.potential_xs]),
            np.array([[0.0]]),
            "CR Clad",
        )

        self._gap_dancoff_xs: CrossSection = CrossSection(
            np.array([self.gap.potential_xs]),
            np.array([self.gap.potential_xs]),
            np.array([[0.0]]),
            "CR Gap",
        )

        self._absorber_isolated_dancoff_fsr_ids = []
        self._absorber_isolated_dancoff_fsr_inds = []
        self._gap_isolated_dancoff_fsr_ids = []
        self._gap_isolated_dancoff_fsr_inds = []
        self._clad_isolated_dancoff_fsr_ids = []
        self._clad_isolated_dancoff_fsr_inds = []

        self._absorber_full_dancoff_fsr_ids = []
        self._absorber_full_dancoff_fsr_inds = []
        self._gap_full_dancoff_fsr_ids = []
        self._gap_full_dancoff_fsr_inds = []
        self._clad_full_dancoff_fsr_ids = []
        self._clad_full_dancoff_fsr_inds = []

        # ======================================================================
        # TRANSPORT CALCULATION DATA
        # ----------------------------------------------------------------------
        # FSR IDs and indices.
        self._absorber_ring_fsr_ids: List[List[int]] = [
            [] for r in range(self.num_rings)
        ]
        self._absorber_ring_fsr_inds: List[List[int]] = [
            [] for r in range(self.num_rings)
        ]
        self._gap_fsr_ids: List[int] = []
        self._gap_fsr_inds: List[int] = []
        self._clad_fsr_ids: List[int] = []
        self._clad_fsr_inds: List[int] = []

        # Create list of the different radii for absorber
        self._absorber_radii = []
        if self.num_rings == 1:
            self._absorber_radii.append(self.absorber_radius)
        else:
            V = np.pi * self.absorber_radius * self.absorber_radius
            Vr = V / self.num_rings
            for ri in range(self.num_rings):
                Rin = 0.0
                if ri > 0:
                    Rin = self._absorber_radii[-1]
                Rout = np.sqrt((Vr + np.pi * Rin * Rin) / np.pi)
                if Rout > self.absorber_radius:
                    Rout = self.absorber_radius
                self._absorber_radii.append(Rout)

        # Initialize array of compositions for the absorber. This holds the
        # composition for each absorber ring and for each depletion step.
        self._absorber_ring_materials: List[List[Material]] = []
        for r in range(self.num_rings):
            # All rings initially start with the same composition
            self._absorber_ring_materials.append([copy.deepcopy(self.absorber)])

        # Initialize an array to hold the flux spectrum for each fuel ring.
        self._absorber_ring_flux_spectra: List[np.ndarray] = []
        for r in range(self.num_rings):
            # All rings initially start with empty flux spectrum list
            self._absorber_ring_flux_spectra.append(np.array([]))

        # Initialize an array to hold the depletion matrices for the previous and current steps.
        self._absorber_ring_prev_dep_mats: List[Optional[DepletionMatrix]] = []
        self._absorber_ring_current_dep_mats: List[Optional[DepletionMatrix]] = []
        for r in range(self.num_rings):
            # All rings initially start with empty matrix
            self._absorber_ring_prev_dep_mats.append(None)
            self._absorber_ring_current_dep_mats.append(None)

        # Holds all the CrossSection objects used for the real transport
        # calculation. These are NOT stored for each depletion step like with
        # the materials.
        self._absorber_ring_xs: List[Optional[CrossSection]] = [
            None for r in range(self.num_rings)
        ]
        self._gap_xs: Optional[CrossSection] = None
        self._clad_xs: Optional[CrossSection] = None

    @property
    def absorber(self) -> Material:
        return self._absorber

    @property
    def clad(self) -> Material:
        return self._clad

    @property
    def gap(self) -> Material:
        return self._gap

    @property
    def absorber_radius(self) -> float:
        return self._absorber_radius

    @property
    def clad_radius(self) -> float:
        return self._clad_radius

    @property
    def gap_radius(self) -> float:
        return self._gap_radius

    @property
    def num_rings(self) -> int:
        return self._num_rings

    @property
    def is_moderator(self) -> bool:
        return self._is_moderator

    @is_moderator.setter
    def is_moderator(self, val: bool) -> None:
        self._is_moderator = val

    @property
    def absorber_ring_materials(self) -> List[List[Material]]:
        return self._absorber_ring_materials

    @property
    def absorber_ring_flux_spectra(self) -> List[List[Material]]:
        return self._absorber_ring_materials

    @property
    def absorber_dancoff_correction(self) -> float:
        return self._absorber_dancoff_correction

    @absorber_dancoff_correction.setter
    def absorber_dancoff_correction(self, val: float) -> None:
        if val < 0.0 or val > 1.0:
            raise ValueError(
                f"Dancoff correction must be in interval [0,1]. Was provided {val}."
            )
        self._absorber_dancoff_correction = val

    def load_nuclides(self, ndl: NDLibrary) -> None:
        """
        Loads all the nuclides for all current materials into the data library.

        Parameters
        ----------
        ndl : NDLibrary
            Nuclear data library which should load the nuclides.
        """
        for ring_mats in self.absorber_ring_materials:
            ring_mats[-1].load_nuclides(ndl)
        self.clad.load_nuclides(ndl)
        self.gap.load_nuclides(ndl)

    # ==========================================================================
    # Dancoff Correction Related Methods

    def _make_dancoff_moc_cell(
        self, moderator_xs: CrossSection
    ) -> Tuple[List[float], List[CrossSection]]:
        """
        Returns the list of radii and list of cross sections for the control rod.
        """
        radii: List[float] = [
            self.absorber_radius,
            self.gap_radius,
            self.clad_radius,
        ]
        xs: List[CrossSection] = [
            self._absorber_dancoff_xs,
            self._gap_dancoff_xs,
            self._clad_dancoff_xs,
        ]

        return radii, xs

    def populate_dancoff_fsr_indexes(
        self, isomoc: MOCDriver, fullmoc: MOCDriver
    ) -> None:
        """
        Obtains the flat source region indexes for all of the flat source
        regions used in the Dancoff correction calculations.

        Parameters
        ----------
        isomoc : MOCDriver
            MOC simulation for the isolated pin.
        fullmoc : MOCDriver
            MOC simulation for the full geometry.
        """
        self._absorber_isolated_dancoff_fsr_inds = []
        self._gap_isolated_dancoff_fsr_inds = []
        self._clad_isolated_dancoff_fsr_inds = []

        self._absorber_full_dancoff_fsr_inds = []
        self._gap_full_dancoff_fsr_inds = []
        self._clad_full_dancoff_fsr_inds = []

        for id in self._absorber_isolated_dancoff_fsr_ids:
            self._absorber_isolated_dancoff_fsr_inds.append(isomoc.get_fsr_indx(id, 0))
        for id in self._gap_isolated_dancoff_fsr_ids:
            self._gap_isolated_dancoff_fsr_inds.append(isomoc.get_fsr_indx(id, 0))
        for id in self._clad_isolated_dancoff_fsr_ids:
            self._clad_isolated_dancoff_fsr_inds.append(isomoc.get_fsr_indx(id, 0))

        for id in self._absorber_full_dancoff_fsr_ids:
            self._absorber_full_dancoff_fsr_inds.append(fullmoc.get_fsr_indx(id, 0))
        for id in self._gap_full_dancoff_fsr_ids:
            self._gap_full_dancoff_fsr_inds.append(fullmoc.get_fsr_indx(id, 0))
        for id in self._clad_full_dancoff_fsr_ids:
            self._clad_full_dancoff_fsr_inds.append(fullmoc.get_fsr_indx(id, 0))

    def _set_dancoff_xs(
        self,
        abs_pot_xs: float,
        gap_pot_xs: float,
        clad_pot_xs: float,
        moderator: Material,
    ) -> None:
        self._absorber_dancoff_xs.set(
            CrossSection(
                np.array([abs_pot_xs]),
                np.array([abs_pot_xs]),
                np.array([[0.0]]),
                "CR Absorber",
            )
        )

        self._gap_dancoff_xs.set(
            CrossSection(
                np.array([gap_pot_xs]),
                np.array([gap_pot_xs]),
                np.array([[0.0]]),
                "CRR Gap",
            )
        )

        self._clad_dancoff_xs.set(
            CrossSection(
                np.array([clad_pot_xs]),
                np.array([clad_pot_xs]),
                np.array([[0.0]]),
                "CR Clad",
            )
        )

        if self.is_moderator:
            mod_name = "Moderator"
            self._absorber_dancoff_xs.name = mod_name
            self._gap_dancoff_xs.name = mod_name
            self._clad_dancoff_xs.name = mod_name

    def set_xs_for_dancoff_calculation(self, moderator: Material) -> None:
        """
        Sets the cross sections for the Dancoff calculations based on the
        initial control rod composition.

        Parameters
        ----------
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        """
        if not self.is_moderator:
            # Just use original abosrber info for dancoff calculation
            abs_pot_xs = self.absorber.potential_xs
            gap_pot_xs = self.gap.potential_xs
            clad_pot_xs = self.clad.potential_xs
        else:
            abs_pot_xs = moderator.potential_xs
            gap_pot_xs = moderator.potential_xs
            clad_pot_xs = moderator.potential_xs

        self._set_dancoff_xs(abs_pot_xs, gap_pot_xs, clad_pot_xs, moderator)

    def set_xs_for_control_rod_dancoff_calculation(self, moderator: Material) -> None:
        """
        Sets the cross sections for the control rod Dancoff calculations.

        Parameters
        ----------
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        """
        if not self.is_moderator:
            abs_pot_xs = 1.0e5
            gap_pot_xs = self.gap.potential_xs
            clad_pot_xs = self.clad.potential_xs
        else:
            abs_pot_xs = moderator.potential_xs
            gap_pot_xs = moderator.potential_xs
            clad_pot_xs = moderator.potential_xs

        self._set_dancoff_xs(abs_pot_xs, gap_pot_xs, clad_pot_xs, moderator)

    def set_isolated_dancoff_fuel_sources(
        self, isomoc: MOCDriver, moderator: Material
    ) -> None:
        """
        Initializes the fixed sources for the isolated MOC calculation required
        in computing Dancoff corrections. Sources are set for a fuel Dancoff
        correction calculation.

        Parameters
        ----------
        isomoc : MOCDriver
            MOC simulation for the isolated geometry.
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        """
        # If the control rod has been replaced with moderator (i.e. withdrawn)
        # then we need to use the moderator potential cross sections !
        if not self.is_moderator:
            abs_pot_xs = self.absorber.potential_xs
            gap_pot_xs = self.gap.potential_xs
            clad_pot_xs = self.clad.potential_xs
        else:
            abs_pot_xs = moderator.potential_xs
            gap_pot_xs = moderator.potential_xs
            clad_pot_xs = moderator.potential_xs

        for ind in self._absorber_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, abs_pot_xs)
        for ind in self._gap_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, gap_pot_xs)
        for ind in self._clad_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, clad_pot_xs)

    def set_isolated_dancoff_clad_sources(
        self, isomoc: MOCDriver, moderator: Material
    ) -> None:
        """
        Initializes the fixed sources for the isolated MOC calculation required
        in computing Dancoff corrections. Sources are set for a clad Dancoff
        correction calculation.

        The cladding of a control rod is not self-shielded. Therefore, this
        method is an alias to set_isolated_dancoff_fuel_sources.

        Parameters
        ----------
        isomoc : MOCDriver
            MOC simulation for the isolated geometry.
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        """
        self.set_isolated_dancoff_fuel_sources(isomoc, moderator)

    def set_isolated_dancoff_control_rod_sources(
        self, isomoc: MOCDriver, moderator: Material
    ) -> None:
        """
        Initializes the fixed sources for the full MOC calculation required
        in computing Dancoff corrections. Sources are set for a control rod
        Dancoff correction calculation.

        The cladding of a control rod is not self-shielded. Therefore, this
        method is an alias to set_isolated_dancoff_fuel_sources.

        Parameters
        ----------
        isomoc : MOCDriver
            MOC simulation for the isolated geometry.
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        """
        # If the control rod has been replaced with moderator (i.e. withdrawn)
        # then we need to use the moderator potential cross sections !
        if not self.is_moderator:
            abs_pot_xs = 0.0
            gap_pot_xs = self.gap.potential_xs
            clad_pot_xs = self.clad.potential_xs
        else:
            abs_pot_xs = moderator.potential_xs
            gap_pot_xs = moderator.potential_xs
            clad_pot_xs = moderator.potential_xs

        for ind in self._absorber_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, abs_pot_xs)
        for ind in self._gap_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, gap_pot_xs)
        for ind in self._clad_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, clad_pot_xs)

    def set_full_dancoff_fuel_sources(
        self, fullmoc: MOCDriver, moderator: Material
    ) -> None:
        """
        Initializes the fixed sources for the full MOC calculation required
        in computing Dancoff corrections. Sources are set for a fuel Dancoff
        correction calculation.

        Parameters
        ----------
        fullmoc : MOCDriver
            MOC simulation for the full geometry.
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        """
        # If the control rod has been replaced with moderator (i.e. withdrawn)
        # then we need to use the moderator potential cross sections !
        if not self.is_moderator:
            abs_pot_xs = self.absorber.potential_xs
            gap_pot_xs = self.gap.potential_xs
            clad_pot_xs = self.clad.potential_xs
        else:
            abs_pot_xs = moderator.potential_xs
            gap_pot_xs = moderator.potential_xs
            clad_pot_xs = moderator.potential_xs

        for ind in self._absorber_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, abs_pot_xs)
        for ind in self._gap_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, gap_pot_xs)
        for ind in self._clad_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, clad_pot_xs)

    def set_full_dancoff_clad_sources(
        self, fullmoc: MOCDriver, moderator: Material
    ) -> None:
        """
        Initializes the fixed sources for the full MOC calculation required
        in computing Dancoff corrections. Sources are set for a clad Dancoff
        correction calculation.

        The cladding of a control rod is not self-shielded. Therefore, this
        method is an alias to set_isolated_dancoff_fuel_sources.

        Parameters
        ----------
        fullmoc : MOCDriver
            MOC simulation for the full geometry.
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        """
        self.set_full_dancoff_fuel_sources(fullmoc, moderator)

    def set_full_dancoff_control_rod_sources(
        self, fullmoc: MOCDriver, moderator: Material
    ) -> None:
        """
        Initializes the fixed sources for the full MOC calculation required
        in computing Dancoff corrections. Sources are set for a control rod
        Dancoff correction calculation.

        Parameters
        ----------
        fullmoc : MOCDriver
            MOC simulation for the full geometry.
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        """
        # If the control rod has been replaced with moderator (i.e. withdrawn)
        # then we need to use the moderator potential cross sections !
        if not self.is_moderator:
            abs_pot_xs = 0.0
            gap_pot_xs = self.gap.potential_xs
            clad_pot_xs = self.clad.potential_xs
        else:
            abs_pot_xs = moderator.potential_xs
            gap_pot_xs = moderator.potential_xs
            clad_pot_xs = moderator.potential_xs

        for ind in self._absorber_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, abs_pot_xs)
        for ind in self._gap_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, gap_pot_xs)
        for ind in self._clad_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, clad_pot_xs)

    def compute_control_rod_dancoff_correction(
        self, isomoc: MOCDriver, fullmoc: MOCDriver
    ) -> float:
        """
        Computes the Dancoff correction for the control rod region.

        Parameters
        ----------
        isomoc : MOCDriver
            MOC simulation for the isolated geometry (previously solved).
        fullmoc : MOCDriver
            MOC simulation for the full geometry (previously solved).

        Returns
        -------
        float
            Dancoff correction for the cladding region.
        """
        iso_flux = isomoc.homogenize_flux_spectrum(
            self._absorber_isolated_dancoff_fsr_inds
        )[0]
        full_flux = fullmoc.homogenize_flux_spectrum(
            self._absorber_full_dancoff_fsr_inds
        )[0]
        C = (iso_flux - full_flux) / iso_flux
        # If the MOC discretization isn't fine enough, the Dancoff correction
        # can end up being negative. We might want a special check for this on
        # Control Rods one day.
        return C

    # ==========================================================================
    # Transport Calculation Related Methods

    def set_xs(self, t: int, moderator_xs: CrossSection, ndl: NDLibrary) -> None:
        """
        Sets or constructs the CrossSection objects for the control rod regions.
        If is_moderator is True, then all regions are set to the moderator_xs.
        Otherwise, the control rod material cross sections are used.

        Parameters
        ----------
        t : int
            Index for the depletion step.
        moderator_xs : CrossSection
            Cross sections for the moderator.
        ndl : NDLibrary
            Nuclear data library to use for cross sections.
        """
        if self.is_moderator:
            self._set_with_moderator(moderator_xs)
        else:
            self._set_with_control_rod(t, ndl)

    def _set_with_moderator(self, moderator_xs: CrossSection) -> None:
        """
        Sets or constructs the CrossSection objects for the control rod with
        cross sections for the moderator as if the control rod were withdrawn.

        Parameters
        ----------
        moderator_xs : CrossSection
            Cross sections for the moderator.
        """
        if not self._is_moderator:
            raise RuntimeError(
                "Cannot fill control rod with moderator as is_moderator is False."
            )

        def copy_or_set(xs):
            if xs is None:
                xs = copy.deepcopy(moderator_xs)
            else:
                xs.set(moderator_xs)
            return xs

        for r in range(self.num_rings):
            self._absorber_ring_xs[r] = copy_or_set(self._absorber_ring_xs[r])
        self._gap_xs = copy_or_set(self._gap_xs)
        self._clad_xs = copy_or_set(self._clad_xs)

    def _set_with_control_rod(self, t: int, ndl: NDLibrary) -> None:
        """
        Sets or constructs the CrossSection objects for the control rod at the
        specified depletion step.

        Parameters
        ----------
        t : int
            Index for the depletion step.
        ndl : NDLibrary
            Nuclear data library to use for cross sections.
        """
        if self._is_moderator:
            raise RuntimeError(
                "Cannot fill control rod with control rod as is_moderator is True."
            )

        self._set_absorber_xs_for_depletion_step(t, ndl)
        self._set_gap_xs(ndl)
        self._set_clad_xs(ndl)

    def _set_absorber_xs_for_depletion_step(self, t: int, ndl: NDLibrary) -> None:
        """
        Constructs the CrossSection object for the abosrber material at the
        center of the control rod at the specified depletion step.

        Parameters
        ----------
        t : int
            Index for the depletion step.
        ndl : NDLibrary
            Nuclear data library to use for cross sections.
        """
        if len(self._absorber_ring_xs) != self.num_rings:
            raise RuntimeError(
                "Number of absorber cross sections does not agree with the number of rings."
            )

        if self.num_rings == 1:
            # Compute escape xs
            Ee = 1.0 / (2.0 * self.absorber_radius)
            new_abs_xs = self._absorber_ring_materials[0][t].carlvik_xs(
                self._absorber_dancoff_correction, Ee, ndl
            )
            if self._absorber_ring_xs[0] is None:
                self._absorber_ring_xs[0] = new_abs_xs
            else:
                self._absorber_ring_xs[0].set(new_abs_xs)
            if self._absorber_ring_xs[0].name == "":
                self._absorber_ring_xs[0].name = "CR Absorber"
        else:
            # Do each ring
            for ri in range(self.num_rings):
                Rin = 0.0
                if ri > 0:
                    Rin = self._absorber_radii[ri - 1]
                Rout = self._absorber_radii[ri]
                new_ring_xs = self._absorber_ring_materials[ri][t].ring_carlvik_xs(
                    self._absorber_dancoff_correction,
                    self.absorber_radius,
                    Rin,
                    Rout,
                    ndl,
                )
                if self._absorber_ring_xs[ri] is None:
                    self._absorber_ring_xs[ri] = new_ring_xs
                else:
                    self._absorber_ring_xs[ri].set(new_ring_xs)
                if self._absorber_ring_xs[ri].name == "":
                    self._absorber_ring_xs[ri].name = "CR Absorber"

    def _set_gap_xs(self, ndl: NDLibrary) -> None:
        """
        Constructs the CrossSection object for the gap between the absorber and
        the cladding.

        Parameters
        ----------
        ndl : NDLibrary
            Nuclear data library to use for cross sections.
        """
        if self._gap_xs is None:
            self._gap_xs = self.gap.dilution_xs([1.0e10] * self.gap.size, ndl)
        else:
            self._gap_xs.set(self.gap.dilution_xs([1.0e10] * self.gap.size, ndl))

        if self._gap_xs.name == "":
            self._gap_xs.name = "CR Gap"

    def _set_clad_xs(self, ndl: NDLibrary) -> None:
        """
        Constructs the CrossSection object for the cladding of the control rod.

        The cladding of a control rod uses infinitely dilute cross sections.

        Parameters
        ----------
        ndl : NDLibrary
            Nuclear data library to use for cross sections.
        """
        if self._clad_xs is None:
            self._clad_xs = self.clad.dilution_xs([1.0e10] * self.clad.size, ndl)
        else:
            self._clad_xs.set(self.clad.dilution_xs([1.0e10] * self.clad.size, ndl))

        if self._clad_xs.name == "":
            self._clad_xs.name = "CR Clad"

    def _make_moc_cell(
        self, moderator_xs: CrossSection
    ) -> Tuple[List[float], List[CrossSection]]:
        """
        Returns the list of radii and list of cross sections for the control rod.
        """
        """
        Returns the list of radii and list of cross sections for the control rod.
        """
        if len(self._absorber_ring_xs) != self.num_rings:
            raise RuntimeError("Absorber cross sections have not yet been built.")
        if self.gap is not None and self._gap_xs is None:
            raise RuntimeError("Gap cross section has not yet been built.")
        if self._clad_xs is None:
            raise RuntimeError("Clad cross section has not yet been built.")

        # Initialize the radii and cross section lists with the fuel info
        radii = [r for r in self._absorber_radii]
        xss = [xs for xs in self._absorber_ring_xs]

        radii += [self.gap_radius, self.clad_radius]
        xss += [self._gap_xs, self._clad_xs]

        return radii, xss

    def populate_fsr_indexes(self, moc: MOCDriver) -> None:
        """
        Obtains the flat source region indexes for all of the flat source
        regions used in the full MOC calculations.

        Parameters
        ----------
        moc : MOCDriver
            MOC simulation for the full calculations.
        """
        self._absorber_ring_fsr_inds = [[] for r in range(self.num_rings)]
        self._gap_fsr_inds = []
        self._clad_fsr_inds = []

        for r in range(self.num_rings):
            for id in self._absorber_ring_fsr_ids[r]:
                self._absorber_ring_fsr_inds[r].append(moc.get_fsr_indx(id, 0))
        for id in self._gap_fsr_ids:
            self._gap_fsr_inds.append(moc.get_fsr_indx(id, 0))
        for id in self._clad_fsr_ids:
            self._clad_fsr_inds.append(moc.get_fsr_indx(id, 0))

    def obtain_flux_spectra(self, moc: MOCDriver) -> None:
        """
        Computes average flux spectrum in the poison from the MOC simulation.

        Parameters
        ----------
        moc : MOCDriver
            MOC simulation for the full calculations.
        """
        for r in range(self.num_rings):
            self._absorber_ring_flux_spectra[r] = moc.homogenize_flux_spectrum(
                self._absorber_ring_fsr_inds[r]
            )

    def normalize_flux_spectrum(self, f) -> None:
        """
        Applies a multiplicative factor to the flux spectra for the poison.
        This permits normalizing the flux to a known assembly power.

        Parameters
        ----------
        f : float
            Normalization factor.
        """
        if f <= 0.0:
            raise ValueError("Normalization factor must be > 0.")

        for r in range(self.num_rings):
            self._absorber_ring_flux_spectra[r] *= f

    def predict_depletion(
        self,
        chain: DepletionChain,
        ndl: NDLibrary,
        dt: float,
        dtm1: Optional[float] = None,
    ) -> None:
        """
        Performs the predictor in the integration of the Bateman equation.
        If the argument for the previous time step is not provided, CE/LI will
        be used. Otherwise, CE/LI is used on the first depletion step, and
        LE/QI is used for all subsequent time steps. The predicted material
        compositions are appended to the materials lists.

        Parameters
        ----------

        chain : DepletionChain
            Depletion chain to use for radioactive decay and transmutation.
        ndl : NDLibrary
            Nuclear data library.
        dt : float
            Durration of the time step in seconds.
        dtm1 : float, optional
            Durration of the previous time step in seconds. Default is None.
        """
        if dt <= 0:
            raise ValueError("Predictor time step must be > 0.")

        if self.is_moderator:
            raise RuntimeError("Cannot deplete control rod when set to moderator.")

        # Do the prediction step for each ring
        for r in range(self.num_rings):
            # Get the flux and initial material
            flux = self._absorber_ring_flux_spectra[r]
            mat = self._absorber_ring_materials[r][-1]  # Use last available mat !

            # Build depletion matrix for beginning of time step
            A0 = build_depletion_matrix(chain, mat, flux, ndl)

            # Save current matrix
            self._absorber_ring_current_dep_mats[r] = A0

            # At this point, we can clear the xs data from the last material as
            # depletion matrix is now built.
            mat.clear_all_micro_xs_data()

            # Initialize an array with the initial target number densities
            N = np.zeros(A0.size)
            nuclides = A0.nuclides
            for i, nuclide in enumerate(nuclides):
                N[i] = mat.atom_density(nuclide)

            if self._absorber_ring_prev_dep_mats[r] is None or dtm1 is None:
                # Use CE/LI
                A0 *= dt

                # Do the matrix exponential
                A0.exponential_product(N)

                # Undo multiplication by time step on the matrix
                A0 /= dt

            else:
                # Use LE/QI
                Am1 = self._absorber_ring_prev_dep_mats[r]

                F1 = (-dt / (12.0 * dtm1)) * Am1 + (
                    (6.0 * dtm1 + dt) / (12.0 * dtm1)
                ) * A0
                F1 *= dt

                F2 = (-5.0 * dt / (12.0 * dtm1)) * Am1 + (
                    (6.0 * dtm1 + 5.0 * dt) / (12.0 * dtm1)
                ) * A0
                F2 *= dt

                # Do the matrix exponentials
                F1.exponential_product(N)
                F2.exponential_product(N)

            # Now we can build a new material composition
            new_mat_comp = MaterialComposition(name=mat.name)
            for i, nuclide in enumerate(nuclides):
                if N[i] > 0.0:
                    new_mat_comp.add_nuclide(nuclide, N[i])

            # Make the new material
            new_mat = Material(new_mat_comp, mat.temperature, ndl)
            self._absorber_ring_materials[r].append(new_mat)

    def correct_depletion(
        self,
        chain: DepletionChain,
        ndl: NDLibrary,
        dt: float,
        dtm1: Optional[float] = None,
    ) -> None:
        """
        Performs the corrector in the integration of the Bateman equation.
        If the argument for the previous time step is not provided, CE/LI will
        be used. Otherwise, CE/LI is used on the first depletion step, and
        LE/QI is used for all subsequent time steps. The corrected material
        compositions replace the ones where were appended in the corrector step.

        Parameters
        ----------
        chain : DepletionChain
            Depletion chain to use for radioactive decay and transmutation.
        ndl : NDLibrary
            Nuclear data library.
        dt : float
            Durration of the time step in seconds.
        dtm1 : float, optional
            Durration of the previous time step in seconds. Default is None.
        """
        if dt <= 0:
            raise ValueError("Corrector time step must be > 0.")

        if self.is_moderator:
            raise RuntimeError("Cannot deplete control rod when set to moderator.")

        # Do the prediction step for each fuel ring
        for r in range(self.num_rings):
            # Get the flux and initial material
            flux = self._absorber_ring_flux_spectra[r]
            mat_pred = self._absorber_ring_materials[r][-1]  # Use last available mat !

            # Get depletion matrix for beginning of time step
            A0 = self._absorber_ring_current_dep_mats[r]

            # Build depletion matrix and multiply by time step
            Ap1 = build_depletion_matrix(chain, mat_pred, flux, ndl)

            # Initialize an array with the initial target number densities
            mat_old = self._absorber_ring_materials[r][-2]  # Go 2 steps back !!
            N = np.zeros(Ap1.size)
            nuclides = Ap1.nuclides
            for i, nuclide in enumerate(nuclides):
                N[i] = mat_old.atom_density(nuclide)

            if self._absorber_ring_prev_dep_mats[r] is None or dtm1 is None:
                # Use CE/LI
                F1 = (5.0 * dt / 12.0) * A0 + (dt / 12.0) * Ap1
                F2 = (dt / 12.0) * A0 + (5.0 * dt / 12.0) * Ap1

                F1.exponential_product(N)
                F2.exponential_product(N)

            else:
                # Use LE/QI

                # Get previous depletion matrix
                Am1 = self._absorber_ring_prev_dep_mats[r]

                F3 = (
                    (-dt * dt / (12.0 * dtm1 * (dtm1 + dt))) * Am1
                    + (
                        (5.0 * dtm1 * dtm1 + 6.0 * dtm1 * dt + dt * dt)
                        / (12.0 * dtm1 * (dtm1 + dt))
                    )
                    * A0
                    + (dtm1 / (12.0 * (dtm1 + dt))) * Ap1
                )
                F3 *= dt

                F4 = (
                    (-dt * dt / (12.0 * dtm1 * (dtm1 + dt))) * Am1
                    + (
                        (dtm1 * dtm1 + 2.0 * dtm1 * dt + dt * dt)
                        / (12.0 * dtm1 * (dtm1 + dt))
                    )
                    * A0
                    + ((5.0 * dtm1 + 4.0 * dt) / (12.0 * (dtm1 + dt))) * Ap1
                )
                F4 *= dt

                F3.exponential_product(N)
                F4.exponential_product(N)

            # Now we can build a new material composition
            new_mat_comp = MaterialComposition(name=mat_old.name)
            for i, nuclide in enumerate(nuclides):
                if N[i] > 0.0:
                    new_mat_comp.add_nuclide(nuclide, N[i])

            # Make the new material
            new_mat = Material(new_mat_comp, mat_pred.temperature, ndl)
            self._absorber_ring_materials[r][-1] = new_mat

            # Save the current matrix as previous matrix for next step !
            self._absorber_ring_prev_dep_mats[r] = A0
            self._absorber_ring_current_dep_mats[r] = None
