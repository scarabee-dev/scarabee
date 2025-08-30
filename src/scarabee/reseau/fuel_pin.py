from .._scarabee import (
    NDLibrary,
    MaterialComposition,
    Material,
    CrossSection,
    PinCellType,
    SimplePinCell,
    PinCell,
    MOCDriver,
    MixingFraction,
    DepletionChain,
    DepletionMatrix,
    mix_materials,
    build_depletion_matrix,
)
import numpy as np
from typing import Optional, List
import copy


class FuelPin:
    """
    Represents a generic fuel pin for a PWR.

    Parameters
    ----------
    fuel : Material
        Material which describes the fuel composition, density, and temperature.
    fuel_radius : float
        Outer radius of the fuel pellet region.
    gap : Material, optional
        Material which describes the composition, density, and temperature of
        the gap between the fuel pellet and the cladding, if present.
    gap_radius : float, optional
        Outer radius of the gap material, if present.
    clad : Material
        Material which describes the cladding composition, density, and
        temperature.
    clad_radius : float
        Outer radius of the cladding.
    num_fuel_rings : int, default 1
        Number of rings which should be used to discretize the fuel material.
        Each ring will be self-shielded and depleted separately.

    Attributes
    ----------
    fuel_radius : float
        Outer radius of the fuel pellet region.
    fuel_ring_materials : list of list of Material
        Contains the Material in each fuel ring for each depletion time step.
    fuel_ring_flux_spectra : list of list of ndarray
        Contains the average flux spectrum in each fuel ring for each depletion
        time step.
    fuel_dancoff_corrections : list of float
        Dancoff corrections to be used when self-shielding the fuel at each
        depletion time step.
    gap : Material, optional
        Material which describes the composition, density, and temperature of
        the gap between the fuel pellet and the cladding, if present.
    gap_radius : float, optional
        Outer radius of the gap material, if present.
    clad : Material
        Material which describes the cladding composition, density, and
        temperature.
    clad_radius : float
        Outer radius of the cladding.
    clad_dancoff_corrections : list of float
        Dancoff corrections to be used when self-shielding the cladding at each
        depletion time step.
    num_fuel_rings : int, default 1
        Number of rings which should be used to discretize the fuel material.
        Each ring will be self-shielded and depleted separately.
    """

    def __init__(
        self,
        fuel: Material,
        fuel_radius: float,
        clad: Material,
        clad_radius,
        gap: Optional[Material],
        gap_radius: Optional[float],
        num_fuel_rings: int = 1,
    ):
        if fuel_radius <= 0.0:
            raise ValueError("Fuel radius must be > 0.")
        self._fuel_radius = fuel_radius

        if num_fuel_rings <= 0:
            raise ValueError("Number of fuel rings must be >= 1.")
        self._num_fuel_rings = num_fuel_rings

        # Mass of fissionable matter in g / cm (i.e. per unit length)
        self._initial_fissionable_linear_mass = (
            fuel.fissionable_grams_per_cm3 * np.pi * self._fuel_radius**2.0
        )

        # Get gap related parameters
        if gap is None and gap_radius is not None:
            raise ValueError("Gap material is None but gap radius is defined.")
        elif gap is not None and gap_radius is None:
            raise ValueError("Gap material is defined but gap radius is None.")

        if gap_radius is not None and gap_radius <= fuel_radius:
            raise ValueError("Gap radius must be > fuel radius.")

        self._gap = copy.deepcopy(gap)
        self._gap_radius = gap_radius

        # Get cladding related parameters
        self._clad = copy.deepcopy(clad)

        if clad_radius <= 0.0 or clad_radius <= fuel_radius:
            raise ValueError("Clad radius must be > fuel radius.")
        elif gap_radius is not None and clad_radius <= gap_radius:
            raise ValueError("Clad radius must be > gap radius.")
        self._clad_radius = clad_radius

        # ======================================================================
        # DANCOFF CORRECTION CALCULATION DATA
        # ----------------------------------------------------------------------

        # Initialize empty list of Dancoff corrections for the fuel
        self._fuel_dancoff_corrections: List[float] = []

        # Initialize empty list of Dancoff corrections for the cladding
        self._clad_dancoff_corrections: List[float] = []

        # Initialize empty variables for Dancoff correction calculations.
        # These are all kept private.
        self._fuel_dancoff_xs: CrossSection = CrossSection(
            np.array([1.0e5]), np.array([1.0e5]), np.array([[0.0]]), "Fuel"
        )
        self._gap_dancoff_xs: Optional[CrossSection] = None
        if self.gap is not None:
            self._gap_dancoff_xs = CrossSection(
                np.array([self.gap.potential_xs]),
                np.array([self.gap.potential_xs]),
                np.array([[0.0]]),
                "Gap",
            )
        self._clad_dancoff_xs: CrossSection = CrossSection(
            np.array([self.clad.potential_xs]),
            np.array([self.clad.potential_xs]),
            np.array([[0.0]]),
            "Clad",
        )

        self._fuel_isolated_dancoff_fsr_ids = []
        self._gap_isolated_dancoff_fsr_ids = []
        self._clad_isolated_dancoff_fsr_ids = []
        self._mod_isolated_dancoff_fsr_ids = []

        self._fuel_full_dancoff_fsr_ids = []
        self._gap_full_dancoff_fsr_ids = []
        self._clad_full_dancoff_fsr_ids = []
        self._mod_full_dancoff_fsr_ids = []

        self._fuel_isolated_dancoff_fsr_inds = []
        self._gap_isolated_dancoff_fsr_inds = []
        self._clad_isolated_dancoff_fsr_inds = []
        self._mod_isolated_dancoff_fsr_inds = []

        self._fuel_full_dancoff_fsr_inds = []
        self._gap_full_dancoff_fsr_inds = []
        self._clad_full_dancoff_fsr_inds = []
        self._mod_full_dancoff_fsr_inds = []

        # ======================================================================
        # TRANSPORT CALCULATION DATA
        # ----------------------------------------------------------------------
        # Lists of the FSR IDs for each fuel ring, used to homogenize flux
        # spectra for depletion. These will be filled by make_moc_cell.
        self._fuel_ring_fsr_ids: List[List[int]] = []
        for r in range(self.num_fuel_rings):
            self._fuel_ring_fsr_ids.append([])
        self._gap_fsr_ids: List[int] = []
        self._clad_fsr_ids: List[int] = []
        self._mod_fsr_ids: List[int] = []

        self._fuel_ring_fsr_inds: List[List[int]] = []
        for r in range(self.num_fuel_rings):
            self._fuel_ring_fsr_inds.append([])
        self._gap_fsr_inds: List[int] = []
        self._clad_fsr_inds: List[int] = []
        self._mod_fsr_inds: List[int] = []

        # Create list of the different radii for fuel pellet
        self._fuel_radii = []
        if self.num_fuel_rings == 1:
            self._fuel_radii.append(self.fuel_radius)
        else:
            V = np.pi * self.fuel_radius * self.fuel_radius
            Vr = V / self.num_fuel_rings
            for ri in range(self.num_fuel_rings):
                Rin = 0.0
                if ri > 0:
                    Rin = self._fuel_radii[-1]
                Rout = np.sqrt((Vr + np.pi * Rin * Rin) / np.pi)
                if Rout > self.fuel_radius:
                    Rout = self.fuel_radius
                self._fuel_radii.append(Rout)

        # Initialize array of compositions for the fuel. This holds the
        # composition for each fuel ring and for each depletion step.
        self._fuel_ring_materials: List[List[Material]] = []
        for r in range(self.num_fuel_rings):
            # All rings initially start with the same composition
            self._fuel_ring_materials.append([copy.deepcopy(fuel)])

        # Initialize an array to hold the flux spectrum for each fuel ring.
        self._fuel_ring_flux_spectra: List[np.ndarray] = []
        for r in range(self.num_fuel_rings):
            # All rings initially start with empty flux spectrum list
            self._fuel_ring_flux_spectra.append(np.array([]))

        # Initialize an array to hold the depletion matrices for the previous and current steps.
        self._fuel_ring_prev_dep_mats: List[Optional[DepletionMatrix]] = []
        self._fuel_ring_current_dep_mats: List[Optional[DepletionMatrix]] = []
        for r in range(self.num_fuel_rings):
            # All rings initially start with empty matrix
            self._fuel_ring_prev_dep_mats.append(None)
            self._fuel_ring_current_dep_mats.append(None)

        # Holds all the CrossSection objects used for the real transport
        # calculation. These are NOT stored for each depletion step like with
        # the materials.
        self._fuel_ring_xs: List[CrossSection] = []
        self._gap_xs: Optional[CrossSection] = None
        self._clad_xs: Optional[CrossSection] = None

    @property
    def fuel_radius(self) -> float:
        return self._fuel_radius

    @property
    def num_fuel_rings(self) -> int:
        return self._num_fuel_rings

    @property
    def initial_fissionable_linear_mass(self) -> float:
        return self._initial_fissionable_linear_mass

    @property
    def fuel_ring_materials(self) -> List[List[Material]]:
        return self._fuel_ring_materials

    @property
    def fuel_ring_flux_spectra(self) -> List[List[Material]]:
        return self._fuel_ring_materials

    @property
    def fuel_dancoff_corrections(self) -> List[float]:
        return self._fuel_dancoff_corrections

    @property
    def gap(self) -> Optional[Material]:
        return self._gap

    @property
    def gap_radius(self) -> Optional[float]:
        return self._gap_radius

    @property
    def clad(self) -> Material:
        return self._clad

    @property
    def clad_radius(self) -> float:
        return self._clad_radius

    @property
    def clad_dancoff_corrections(self) -> List[float]:
        return self._clad_dancoff_corrections

    def _check_dx_dy(self, dx, dy, pintype):
        if pintype == PinCellType.Full:
            if dx < 2.0 * self.clad_radius:
                raise ValueError(
                    "The fuel pin cell x width must be > the diameter of the cladding."
                )
            if dy < 2.0 * self.clad_radius:
                raise ValueError(
                    "The fuel pin cell y width must be > the diameter of the cladding."
                )
        elif pintype in [PinCellType.XN, PinCellType.XP]:
            if dx < self.clad_radius:
                raise ValueError(
                    "The fuel pin cell x width must be > the radius of the cladding."
                )
            if dy < 2.0 * self.clad_radius:
                raise ValueError(
                    "The fuel pin cell y width must be > the diameter of the cladding."
                )
        elif pintype in [PinCellType.YN, PinCellType.YP]:
            if dy < self.clad_radius:
                raise ValueError(
                    "The fuel pin cell y width must be > the radius of the cladding."
                )
            if dx < 2.0 * self.clad_radius:
                raise ValueError(
                    "The fuel pin cell x width must be > the diameter of the cladding."
                )
        else:
            if dx < self.clad_radius:
                raise ValueError(
                    "The fuel pin cell x width must be > the radius of the cladding."
                )
            if dy < self.clad_radius:
                raise ValueError(
                    "The fuel pin cell y width must be > the radius of the cladding."
                )

    def load_nuclides(self, ndl: NDLibrary) -> None:
        """
        Loads all the nuclides for all current materials into the data library.

        Parameters
        ----------
        ndl : NDLibrary
            Nuclear data library which should load the nuclides.
        """
        for ring_mats in self.fuel_ring_materials:
            ring_mats[-1].load_nuclides(ndl)

        if self.gap is not None:
            self.gap.load_nuclides(ndl)

        self.clad.load_nuclides(ndl)

    # ==========================================================================
    # Interrogation Methods
    def get_fuel_material(self, t: int, r: int = 0) -> Material:
        """
        Returns the Material object for a desired fuel ring at a desired
        depletion time step. Ring index 0 is at the center of the pin.

        Parameters
        ----------
        t : int
            Depletion time step index.
        r : int
            Ring index. Default is 0.

        Returns
        -------
        Material
            Material defining the temperature, density, and composition for the
            desired fuel ring and depletion time step.
        """
        if r >= self.num_fuel_rings:
            raise IndexError(f"Fuel ring index {r} is out of range.")

        if t >= len(self._fuel_ring_materials[r]):
            raise IndexError(f"Fuel time step index {t} is out of range.")

        return self._fuel_ring_materials[r][t]

    def get_average_fuel_nuclide_density(self, t: int, nuclide: str) -> float:
        """
        Computes the average density of a nuclide within the fuel pellet at a
        single depletion time step.

        Parameters
        ----------
        t : int
            Depletion time step index.
        nuclide : str
            Name of the nuclide.

        Returns
        -------
        float
            Average density of the nuclide at depletion time step t across the
            fuel pellet in units of atoms per barn-cm.
        """
        sum_density = 0.0

        for r in range(self.num_fuel_rings):
            mat = self.get_fuel_material(t, r)
            sum_density += mat.atom_density(nuclide)

        return sum_density / self.num_fuel_rings

    # ==========================================================================
    # Dancoff Correction Related Methods
    def set_xs_for_fuel_dancoff_calculation(self) -> None:
        """
        Sets the 1-group cross sections to calculate the fuel Dancoff correction.
        """
        self._fuel_dancoff_xs.set(
            CrossSection(
                np.array([1.0e5]), np.array([1.0e5]), np.array([[0.0]]), "Fuel"
            )
        )

        if self._gap_dancoff_xs is not None and self.gap is not None:
            self._gap_dancoff_xs.set(
                CrossSection(
                    np.array([self.gap.potential_xs]),
                    np.array([self.gap.potential_xs]),
                    np.array([[0.0]]),
                    "Gap",
                )
            )

        self._clad_dancoff_xs.set(
            CrossSection(
                np.array([self.clad.potential_xs]),
                np.array([self.clad.potential_xs]),
                np.array([[0.0]]),
                "Clad",
            )
        )

    def set_xs_for_clad_dancoff_calculation(self, ndl: NDLibrary) -> None:
        """
        Sets the 1-group cross sections to calculate the clad Dancoff correction.

        Parameters
        ----------
        ndl : NDLibrary
            Nuclear data library for obtaining potential scattering cross
            sections.
        """
        # Create average fuel mixture
        fuel_mats = []
        fuel_vols = []
        for ring in self.fuel_ring_materials:
            fuel_mats.append(ring[-1])
            fuel_vols.append(1.0 / self.num_fuel_rings)
        avg_fuel: Material = mix_materials(
            fuel_mats, fuel_vols, MixingFraction.Volume, ndl
        )

        self._fuel_dancoff_xs.set(
            CrossSection(
                np.array([avg_fuel.potential_xs]),
                np.array([avg_fuel.potential_xs]),
                np.array([[0.0]]),
                "Fuel",
            )
        )

        if self._gap_dancoff_xs is not None and self.gap is not None:
            self._gap_dancoff_xs.set(
                CrossSection(
                    np.array([self.gap.potential_xs]),
                    np.array([self.gap.potential_xs]),
                    np.array([[0.0]]),
                    "Gap",
                )
            )

        self._clad_dancoff_xs.set(
            CrossSection(
                np.array([1.0e5]),
                np.array([1.0e5]),
                np.array([[0.0]]),
                "Clad",
            )
        )

    def make_dancoff_moc_cell(
        self,
        moderator_xs: CrossSection,
        dx: float,
        dy: float,
        pintype: PinCellType,
        isolated: bool,
    ) -> SimplePinCell:
        """
        Makes a simplified cell suitable for performing Dancoff correction
        calculations. The flat source region IDs are stored locally in the
        FuelPin object.

        Parameters
        ----------
        moderator_xs : CrossSection
            One group cross sections for the moderator. Total should equal
            absorption (i.e. no scattering) and should be equal to the
            macroscopic potential cross section.
        dx : float
            Width of the cell along x.
        dy : float
            Width of the cell along y.
        pintype : PinCellType
            How the pin cell should be split (along x, y, or only a quadrant).
        isolated : bool
            If True, the FSR IDs are stored for the isolated pin. Otherwise,
            they are stored for the full pin.

        Returns
        -------
        SimplifiedPinCell
            Pin cell object for MOC Dancoff correction calculation.
        """
        self._check_dx_dy(dx, dy, pintype)

        # First we create list of radii and materials
        radii = []
        xs = []

        radii.append(self.fuel_radius)
        xs.append(self._fuel_dancoff_xs)

        if self.gap is not None and self.gap_radius is not None:
            radii.append(self.gap_radius)
            xs.append(self._gap_dancoff_xs)

        radii.append(self.clad_radius)
        xs.append(self._clad_dancoff_xs)

        xs.append(moderator_xs)

        # Make the simple pin cell.
        cell = SimplePinCell(radii, xs, dx, dy, pintype)

        # Get the FSR IDs for the regions of interest
        cell_fsr_ids = list(cell.get_all_fsr_ids())
        cell_fsr_ids.sort()

        if isolated:
            self._fuel_isolated_dancoff_fsr_ids.append(cell_fsr_ids[0])
            if self.gap is None:
                self._clad_isolated_dancoff_fsr_ids.append(cell_fsr_ids[1])
                self._mod_isolated_dancoff_fsr_ids.append(cell_fsr_ids[2])
            else:
                self._gap_isolated_dancoff_fsr_ids.append(cell_fsr_ids[1])
                self._clad_isolated_dancoff_fsr_ids.append(cell_fsr_ids[2])
                self._mod_isolated_dancoff_fsr_ids.append(cell_fsr_ids[3])
        else:
            self._fuel_full_dancoff_fsr_ids.append(cell_fsr_ids[0])
            if self.gap is None:
                self._clad_full_dancoff_fsr_ids.append(cell_fsr_ids[1])
                self._mod_full_dancoff_fsr_ids.append(cell_fsr_ids[2])
            else:
                self._gap_full_dancoff_fsr_ids.append(cell_fsr_ids[1])
                self._clad_full_dancoff_fsr_ids.append(cell_fsr_ids[2])
                self._mod_full_dancoff_fsr_ids.append(cell_fsr_ids[3])

        return cell

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
        self._fuel_isolated_dancoff_fsr_inds = []
        self._gap_isolated_dancoff_fsr_inds = []
        self._clad_isolated_dancoff_fsr_inds = []
        self._mod_isolated_dancoff_fsr_inds = []

        self._fuel_full_dancoff_fsr_inds = []
        self._gap_full_dancoff_fsr_inds = []
        self._clad_full_dancoff_fsr_inds = []
        self._mod_full_dancoff_fsr_inds = []

        for id in self._fuel_isolated_dancoff_fsr_ids:
            self._fuel_isolated_dancoff_fsr_inds.append(isomoc.get_fsr_indx(id, 0))
        for id in self._gap_isolated_dancoff_fsr_ids:
            self._gap_isolated_dancoff_fsr_inds.append(isomoc.get_fsr_indx(id, 0))
        for id in self._clad_isolated_dancoff_fsr_ids:
            self._clad_isolated_dancoff_fsr_inds.append(isomoc.get_fsr_indx(id, 0))
        for id in self._mod_isolated_dancoff_fsr_ids:
            self._mod_isolated_dancoff_fsr_inds.append(isomoc.get_fsr_indx(id, 0))

        for id in self._fuel_full_dancoff_fsr_ids:
            self._fuel_full_dancoff_fsr_inds.append(fullmoc.get_fsr_indx(id, 0))
        for id in self._gap_full_dancoff_fsr_ids:
            self._gap_full_dancoff_fsr_inds.append(fullmoc.get_fsr_indx(id, 0))
        for id in self._clad_full_dancoff_fsr_ids:
            self._clad_full_dancoff_fsr_inds.append(fullmoc.get_fsr_indx(id, 0))
        for id in self._mod_full_dancoff_fsr_ids:
            self._mod_full_dancoff_fsr_inds.append(fullmoc.get_fsr_indx(id, 0))

    def set_isolated_dancoff_fuel_sources(
        self, isomoc: MOCDriver, moderator: Material
    ) -> None:
        """
        Initializes the fixed sources for the isolated MOC calculation required
        in computing Dancoff corrections. Sources are set for a fuel Dancoff
        corrections calculation.

        Parameters
        ----------
        isomoc : MOCDriver
            MOC simulation for the isolated geometry.
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        """
        # Fuel sources should all be zero !
        for ind in self._fuel_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, 0.0)

        # Gap sources should all be potential_xs
        if self.gap is not None:
            pot_xs = self.gap.potential_xs
            for ind in self._gap_isolated_dancoff_fsr_inds:
                isomoc.set_extern_src(ind, 0, pot_xs)

        # Clad sources should all be potential_xs
        pot_xs = self.clad.potential_xs
        for ind in self._clad_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, pot_xs)

        # Moderator sources should all be potential_xs
        pot_xs = moderator.potential_xs
        for ind in self._mod_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, pot_xs)

    def set_isolated_dancoff_clad_sources(
        self, isomoc: MOCDriver, moderator: Material, ndl: NDLibrary
    ) -> None:
        """
        Initializes the fixed sources for the isolated MOC calculation required
        in computing Dancoff corrections. Sources are set for a clad Dancoff
        correction calculation.

        Parameters
        ----------
        isomoc : MOCDriver
            MOC simulation for the isolated geometry.
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        ndl : NDLibrary
            Nuclear data library for obtaining potential scattering cross
            sections.
        """
        # Create average fuel mixture
        fuel_mats = []
        fuel_vols = []
        for ring in self.fuel_ring_materials:
            fuel_mats.append(ring[-1])
            fuel_vols.append(1.0 / self.num_fuel_rings)
        avg_fuel: Material = mix_materials(
            fuel_mats, fuel_vols, MixingFraction.Volume, ndl
        )

        # Fuel sources should all be potential_xs
        pot_xs = avg_fuel.potential_xs
        for ind in self._fuel_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, pot_xs)

        # Gap sources should all be potential_xs
        if self.gap is not None:
            pot_xs = self.gap.potential_xs
            for ind in self._gap_isolated_dancoff_fsr_inds:
                isomoc.set_extern_src(ind, 0, pot_xs)

        # Clad sources should all be zero !
        for ind in self._clad_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, 0.0)

        # Moderator sources should all be potential_xs
        pot_xs = moderator.potential_xs
        for ind in self._mod_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, pot_xs)

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
        # Fuel sources should all be zero !
        for ind in self._fuel_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, 0.0)

        # Gap sources should all be potential_xs
        if self.gap is not None:
            pot_xs = self.gap.potential_xs
            for ind in self._gap_full_dancoff_fsr_inds:
                fullmoc.set_extern_src(ind, 0, pot_xs)

        # Clad sources should all be potential_xs
        pot_xs = self.clad.potential_xs
        for ind in self._clad_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, pot_xs)

        # Moderator sources should all be potential_xs
        pot_xs = moderator.potential_xs
        for ind in self._mod_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, pot_xs)

    def set_full_dancoff_clad_sources(
        self, fullmoc: MOCDriver, moderator: Material, ndl: NDLibrary
    ) -> None:
        """
        Initializes the fixed sources for the full MOC calculation required
        in computing Dancoff corrections. Sources are set for a clad Dancoff
        correction calculation.

        Parameters
        ----------
        isomoc : MOCDriver
            MOC simulation for the isolated geometry.
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        ndl : NDLibrary
            Nuclear data library for obtaining potential scattering cross
            sections.
        """
        # Create average fuel mixture
        fuel_mats = []
        fuel_vols = []
        for ring in self.fuel_ring_materials:
            fuel_mats.append(ring[-1])
            fuel_vols.append(1.0 / self.num_fuel_rings)
        avg_fuel: Material = mix_materials(
            fuel_mats, fuel_vols, MixingFraction.Volume, ndl
        )

        # Fuel sources should all be potential_xs
        pot_xs = avg_fuel.potential_xs
        for ind in self._fuel_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, pot_xs)

        # Gap sources should all be potential_xs
        if self.gap is not None:
            pot_xs = self.gap.potential_xs
            for ind in self._gap_full_dancoff_fsr_inds:
                fullmoc.set_extern_src(ind, 0, pot_xs)

        # Clad sources should all be zero !
        for ind in self._clad_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, 0.0)

        # Moderator sources should all be potential_xs
        pot_xs = moderator.potential_xs
        for ind in self._mod_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, pot_xs)

    def compute_fuel_dancoff_correction(
        self, isomoc: MOCDriver, fullmoc: MOCDriver
    ) -> float:
        """
        Computes the Dancoff correction for the fuel region of the fuel pin.

        Parameters
        ----------
        isomoc : MOCDriver
            MOC simulation for the isolated geometry (previously solved).
        fullmoc : MOCDriver
            MOC simulation for the full geometry (previously solved).

        Returns
        -------
        float
            Dancoff correction for the fuel region.
        """
        iso_flux = isomoc.homogenize_flux_spectrum(
            self._fuel_isolated_dancoff_fsr_inds
        )[0]
        full_flux = fullmoc.homogenize_flux_spectrum(self._fuel_full_dancoff_fsr_inds)[
            0
        ]
        return (iso_flux - full_flux) / iso_flux

    def compute_clad_dancoff_correction(
        self, isomoc: MOCDriver, fullmoc: MOCDriver
    ) -> float:
        """
        Computes the Dancoff correction for the cladding region of the fuel pin.

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
            self._clad_isolated_dancoff_fsr_inds
        )[0]
        full_flux = fullmoc.homogenize_flux_spectrum(self._clad_full_dancoff_fsr_inds)[
            0
        ]
        return (iso_flux - full_flux) / iso_flux

    def append_fuel_dancoff_correction(self, C) -> None:
        """
        Saves new Dancoff correction for the fuel that will be used for all
        subsequent cross section updates.

        Parameters
        ----------
        C : float
            New Dancoff correction.
        """
        if C < 0.0 or C > 1.0:
            raise ValueError(
                f"Dancoff correction must be in range [0, 1]. Was provided {C}."
            )
        self._fuel_dancoff_corrections.append(C)

    def append_clad_dancoff_correction(self, C) -> None:
        """
        Saves new Dancoff correction for the cladding that will be used for all
        subsequent cross section updates.

        Parameters
        ----------
        C : float
            New Dancoff correction.
        """
        if C < 0.0 or C > 1.0:
            raise ValueError(
                f"Dancoff correction must be in range [0, 1]. Was provided {C}."
            )
        self._clad_dancoff_corrections.append(C)

    # ==========================================================================
    # Transport Calculation Related Methods
    def set_fuel_xs_for_depletion_step(self, t: int, ndl: NDLibrary) -> None:
        """
        Constructs the CrossSection object for all fuel rings of the pin at the
        specified depletion step.

        Parameters
        ----------
        t : int
            Index for the depletion step.
        ndl : NDLibrary
            Nuclear data library to use for cross sections.
        """
        # Do the fuel cross sections
        if len(self._fuel_ring_xs) == 0:
            # Create initial CrossSection objects
            if self.num_fuel_rings == 1:
                # Compute escape xs
                Ee = 1.0 / (2.0 * self.fuel_radius)
                self._fuel_ring_xs.append(
                    self._fuel_ring_materials[0][t].carlvik_xs(
                        self._fuel_dancoff_corrections[t], Ee, ndl
                    )
                )
                if self._fuel_ring_xs[-1].name == "":
                    self._fuel_ring_xs[-1].name = "Fuel"
            else:
                # Do each ring
                for ri in range(self.num_fuel_rings):
                    Rin = 0.0
                    if ri > 0:
                        Rin = self._fuel_radii[ri - 1]
                    Rout = self._fuel_radii[ri]
                    self._fuel_ring_xs.append(
                        self._fuel_ring_materials[ri][t].ring_carlvik_xs(
                            self._fuel_dancoff_corrections[t],
                            self.fuel_radius,
                            Rin,
                            Rout,
                            ndl,
                        )
                    )
                    if self._fuel_ring_xs[-1].name == "":
                        self._fuel_ring_xs[-1].name = "Fuel"

        elif len(self._fuel_ring_xs) == self.num_fuel_rings:
            # Reset XS values. Cannot reassign or pointers will be broken !
            if self.num_fuel_rings == 1:
                # Compute escape xs
                Ee = 1.0 / (2.0 * self.fuel_radius)
                self._fuel_ring_xs[0].set(
                    self._fuel_ring_materials[0][t].carlvik_xs(
                        self._fuel_dancoff_corrections[t], Ee, ndl
                    )
                )
                if self._fuel_ring_xs[0].name == "":
                    self._fuel_ring_xs[0].name = "Fuel"
            else:
                # Do each ring
                for ri in range(self.num_fuel_rings):
                    Rin = 0.0
                    if ri > 0:
                        Rin = self._fuel_radii[ri - 1]
                    Rout = self._fuel_radii[ri]
                    self._fuel_ring_xs[ri].set(
                        self._fuel_ring_materials[ri][t].ring_carlvik_xs(
                            self._fuel_dancoff_corrections[t],
                            self.fuel_radius,
                            Rin,
                            Rout,
                            ndl,
                        )
                    )
                    if self._fuel_ring_xs[ri].name == "":
                        self._fuel_ring_xs[ri].name = "Fuel"
        else:
            raise RuntimeError(
                "Number of fuel cross sections does not agree with the number of fuel rings."
            )

    def set_gap_xs(self, ndl: NDLibrary) -> None:
        """
        Constructs the CrossSection object for the gap between the fuel pellet
        and the cladding of the pin.

        Parameters
        ----------
        ndl : NDLibrary
            Nuclear data library to use for cross sections.
        """
        if self.gap is not None:
            if self._gap_xs is None:
                self._gap_xs = self.gap.dilution_xs([1.0e10] * self.gap.size, ndl)
            else:
                self._gap_xs.set(self.gap.dilution_xs([1.0e10] * self.gap.size, ndl))

            if self._gap_xs.name == "":
                self._gap_xs.name = "Gap"

    def set_clad_xs_for_depletion_step(self, t: int, ndl: NDLibrary) -> None:
        """
        Constructs the CrossSection object for the cladding of the pin at the
        specified depletion step. The depletion step only changes the Dancoff
        correction, not the cladding composition.

        Parameters
        ----------
        t : int
            Index for the depletion step.
        ndl : NDLibrary
            Nuclear data library to use for cross sections.
        """
        # Compute escape xs
        Ee = 0.0
        if self.gap_radius is not None:
            Ee = 1.0 / (2.0 * (self.clad_radius - self.gap_radius))
        else:
            Ee = 1.0 / (2.0 * (self.clad_radius - self.fuel_radius))

        # Get / set the xs
        if self._clad_xs is None:
            self._clad_xs = self.clad.roman_xs(
                self._clad_dancoff_corrections[t], Ee, ndl
            )
        else:
            self._clad_xs.set(
                self.clad.roman_xs(self._clad_dancoff_corrections[t], Ee, ndl)
            )

        if self._clad_xs.name == "":
            self._clad_xs.name = "Clad"

    def make_moc_cell(
        self,
        moderator_xs: CrossSection,
        dx: float,
        dy: float,
        pintype: PinCellType,
    ) -> PinCell:
        """
        Constructs the pin cell object used in for the global MOC simulation.

        Parameters
        ----------
        moderator_xs : CrossSection
            Cross sections to use for the moderator surrounding the fuel pin.
        dx : float
            Width of the cell along x.
        dy : float
            Width of the cell along y.
        pintype : PinCellType
            How the pin cell should be split (along x, y, or only a quadrant).

        Returns
        -------
        PinCell
            Pin cell suitable for the true MOC calculation.
        """
        if len(self._fuel_ring_xs) != self.num_fuel_rings:
            raise RuntimeError("Fuel cross sections have not yet been built.")
        if self.gap is not None and self._gap_xs is None:
            raise RuntimeError("Gap cross section has not yet been built.")
        if self._clad_xs is None:
            raise RuntimeError("Clad cross section has not yet been built.")
        self._check_dx_dy(dx, dy, pintype)

        # Initialize the radii and cross section lists with the fuel info
        radii = [r for r in self._fuel_radii]
        xss = [xs for xs in self._fuel_ring_xs]

        # Add the gap (if present)
        if self._gap_xs is not None:
            radii.append(self.gap_radius)
            xss.append(self._gap_xs)

        # Add cladding
        radii.append(self.clad_radius)
        xss.append(self._clad_xs)

        # Add another ring of moderator if possible
        if pintype == PinCellType.Full and min(dx, dy) > 2.0 * self.clad_radius:
            radii.append(0.5 * min(dx, dy))
            xss.append(moderator_xs)
        elif (
            pintype in [PinCellType.XN, PinCellType.XP]
            and dx > self.clad_radius
            and dy > 2.0 * self.clad_radius
        ):
            radii.append(min(dx, 0.5 * dy))
            xss.append(moderator_xs)
        elif (
            pintype in [PinCellType.YN, PinCellType.YP]
            and dy > self.clad_radius
            and dx > 2.0 * self.clad_radius
        ):
            radii.append(min(0.5 * dx, dy))
            xss.append(moderator_xs)
        elif dx > self.clad_radius and dy > self.clad_radius:
            radii.append(min(dx, dy))
            xss.append(moderator_xs)

        # Add moderator to the end of materials
        xss.append(moderator_xs)

        # Create the cell object
        cell = PinCell(radii, xss, dx, dy, pintype)

        # Get the FSR IDs for the regions of interest
        cell_fsr_ids = list(cell.get_all_fsr_ids())
        cell_fsr_ids.sort()

        # Number of angular divisions
        NA = 8
        if pintype in [PinCellType.XN, PinCellType.XP, PinCellType.YN, PinCellType.YP]:
            NA = 4
        elif pintype in [
            PinCellType.I,
            PinCellType.II,
            PinCellType.III,
            PinCellType.IV,
        ]:
            NA = 2

        I = 0  # Starting index for cell_fsr_inds
        # Go through all rings, and get FSR IDs
        for r in range(self.num_fuel_rings):
            for a in range(NA):
                self._fuel_ring_fsr_ids[r].append(cell_fsr_ids[I])
                I += 1

        # Get the FSRs for the gap, if present
        if self._gap_xs is not None:
            for a in range(NA):
                self._gap_fsr_ids.append(cell_fsr_ids[I])
                I += 1

        # Get the FSRs for the cladding
        for a in range(NA):
            self._clad_fsr_ids.append(cell_fsr_ids[I])
            I += 1

        # Everything else should be a moderator FSR
        self._mod_fsr_ids = list(cell_fsr_ids[I:])

        return cell

    def populate_fsr_indexes(self, moc: MOCDriver) -> None:
        """
        Obtains the flat source region indexes for all of the flat source
        regions used in the full MOC calculations.

        Parameters
        ----------
        moc : MOCDriver
            MOC simulation for the full calculations.
        """
        self._fuel_ring_fsr_inds: List[List[int]] = []
        for r in range(self.num_fuel_rings):
            self._fuel_ring_fsr_inds.append([])
        self._gap_fsr_inds: List[int] = []
        self._clad_fsr_inds: List[int] = []
        self._mod_fsr_inds: List[int] = []

        for r in range(self.num_fuel_rings):
            for id in self._fuel_ring_fsr_ids[r]:
                self._fuel_ring_fsr_inds[r].append(moc.get_fsr_indx(id, 0))
        for id in self._gap_fsr_ids:
            self._gap_fsr_inds.append(moc.get_fsr_indx(id, 0))
        for id in self._clad_fsr_ids:
            self._clad_fsr_inds.append(moc.get_fsr_indx(id, 0))
        for id in self._mod_fsr_ids:
            self._mod_fsr_inds.append(moc.get_fsr_indx(id, 0))

    def obtain_flux_spectra(self, moc: MOCDriver) -> None:
        """
        Computes the average flux spectrum for each fuel ring from the MOC
        simulation. Each ring's flux spectrum is volume averaged.

        Parameters
        ----------
        moc : MOCDriver
            MOC simulation for the full calculations.
        """
        for r in range(self.num_fuel_rings):
            self._fuel_ring_flux_spectra[r] = moc.homogenize_flux_spectrum(
                self._fuel_ring_fsr_inds[r]
            )

    def compute_pin_linear_power(self, ndl: NDLibrary):
        """
        Computes the linear power density of the fuel pin based on the current
        flux spectra, in units of w / cm. Does not consider the partial pin
        geometry at the assembly level (i.e. a half pin in a quarter assembly).

        Parameters
        ----------
        ndl : NDLibrary
            Nuclear data library for the fission energy release.

        Returns
        -------
        float
            Linear power density in w / cm.
        """
        power = 0.0
        A = np.pi * self.fuel_radius**2.0 / self.num_fuel_rings
        for r in range(self.num_fuel_rings):
            mat = self._fuel_ring_materials[r][-1]
            flux = self._fuel_ring_flux_spectra[r]
            power += A * mat.compute_fission_power_density(flux, ndl)
        # Convert from MeV/cm/s to J/cm/s = w/cm
        power *= 1.6021766339999e-13
        return power

    def normalize_flux_spectrum(self, f) -> None:
        """
        Applies a multiplicative factor to the flux spectra for the fuel rings.
        This permits normalizing the flux to a known assembly power.

        Parameters
        ----------
        f : float
            Normalization factor.
        """
        if f <= 0.0:
            raise ValueError("Normalization factor must be > 0.")

        for r in range(self.num_fuel_rings):
            self._fuel_ring_flux_spectra[r] *= f

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

        Paramters
        ---------

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

        # Do the prediction step for each fuel ring
        for r in range(self.num_fuel_rings):
            # Get the flux and initial material
            flux = self._fuel_ring_flux_spectra[r]
            mat = self._fuel_ring_materials[r][-1]  # Use last available mat !

            # Build depletion matrix for beginning of time step
            A0 = build_depletion_matrix(chain, mat, flux, ndl)

            # Save current matrix
            self._fuel_ring_current_dep_mats[r] = A0

            # At this point, we can clear the xs data from the last material as
            # depletion matrix is now built.
            mat.clear_all_micro_xs_data()

            # Initialize an array with the initial target number densities
            N = np.zeros(A0.size)
            nuclides = A0.nuclides
            for i, nuclide in enumerate(nuclides):
                N[i] = mat.atom_density(nuclide)

            if self._fuel_ring_prev_dep_mats[r] is None or dtm1 is None:
                # Use CE/LI
                A0 *= dt

                # Do the matrix exponential
                A0.exponential_product(N)

                # Undo multiplication by time step on the matrix
                A0 /= dt

            else:
                # Use LE/QI
                Am1 = self._fuel_ring_prev_dep_mats[r]

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
            new_mat_comp = MaterialComposition()
            for i, nuclide in enumerate(nuclides):
                if N[i] > 0.0:
                    new_mat_comp.add_nuclide(nuclide, N[i])

            # Make the new material
            new_mat = Material(new_mat_comp, mat.temperature, ndl)
            self._fuel_ring_materials[r].append(new_mat)

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

        # Do the prediction step for each fuel ring
        for r in range(self.num_fuel_rings):
            # Get the flux and initial material
            flux = self._fuel_ring_flux_spectra[r]
            mat_pred = self._fuel_ring_materials[r][-1]  # Use last available mat !

            # Get depletion matrix for beginning of time step
            A0 = self._fuel_ring_current_dep_mats[r]

            # Build depletion matrix and multiply by time step
            Ap1 = build_depletion_matrix(chain, mat_pred, flux, ndl)

            # Initialize an array with the initial target number densities
            mat_old = self._fuel_ring_materials[r][-2]  # Go 2 steps back !!
            N = np.zeros(Ap1.size)
            nuclides = Ap1.nuclides
            for i, nuclide in enumerate(nuclides):
                N[i] = mat_old.atom_density(nuclide)

            if self._fuel_ring_prev_dep_mats[r] is None or dtm1 is None:
                # Use CE/LI
                F1 = (5.0 * dt / 12.0) * A0 + (dt / 12.0) * Ap1
                F2 = (dt / 12.0) * A0 + (5.0 * dt / 12.0) * Ap1

                F1.exponential_product(N)
                F2.exponential_product(N)

            else:
                # Use LE/QI

                # Get previous depletion matrix
                Am1 = self._fuel_ring_prev_dep_mats[r]

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
            new_mat_comp = MaterialComposition()
            for i, nuclide in enumerate(nuclides):
                if N[i] > 0.0:
                    new_mat_comp.add_nuclide(nuclide, N[i])

            # Make the new material
            new_mat = Material(new_mat_comp, mat_pred.temperature, ndl)
            self._fuel_ring_materials[r][-1] = new_mat

            # Save the current matrix as previous matrix for next step !
            self._fuel_ring_prev_dep_mats[r] = A0
            self._fuel_ring_current_dep_mats[r] = None
