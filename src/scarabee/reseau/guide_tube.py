from .._scarabee import (
    NDLibrary,
    Material,
    CrossSection,
    PinCellType,
    SimplePinCell,
    PinCell,
    MOCDriver,
    DepletionChain,
)
from .burnable_poison_rod import BurnablePoisonRod
from .control_rod import ControlRod
import numpy as np
from typing import Optional, List
import copy


class GuideTube:
    """
    Represents an empty guide tube for a PWR.

    Parameters
    ----------
    clad : Material
        Material which describes the cladding composition, density, and
        temperature.
    inner_radius : float
        Inner radius of the guide tube.
    outer_radius : float
        Outer radius of the guide tube.
    fill : BurnablePoisonRod, optional
        Optional burnable poison rod which can be placed inside the guide tube.
        Default value is None.

    Attributes
    ----------
    clad : Material
        Material which describes the cladding composition, density, and
        temperature.
    inner_radius : float
        Inner radius of the guide tube.
    outer_radius : float
        Outer radius of the guide tube.
    fill : BurnablePoisonRod or ControlRod, optional
        Optional burnable poison rod or control rod which can be placed inside
        the guide tube.
    clad_dancoff_correction_no_control_rods : float
        Dancoff correction to be used when self-shielding the cladding when no
        control rods are present.
    clad_dancoff_correction_control_rods : float
        Dancoff correction to be used when self-shielding the cladding when
        control rods are present.
    clad_dancoff_correction : float
        Current Dancoff correction used when self-shielding the cladding.
    empty : bool
        True if fill is None and False otherwise.
    """

    def __init__(
        self,
        clad: Material,
        inner_radius: float,
        outer_radius: float,
        fill: Optional[BurnablePoisonRod | ControlRod] = None,
    ):
        if inner_radius >= outer_radius:
            raise ValueError("Inner radius must be > outer radius.")

        self._clad = copy.deepcopy(clad)
        self._inner_radius = inner_radius
        self._outer_radius = outer_radius
        self._fill = copy.deepcopy(fill)

        # Make sure the control rod of poison rod would actually fit inside the
        # guide tube. Do this by checking the radii.
        if self.fill is not None:
            if isinstance(self.fill, BurnablePoisonRod):
                if self.fill.outer_clad_radius >= self.inner_radius:
                    raise ValueError(
                        "The burnable poison rod is too large for the guide tube."
                    )
            elif isinstance(self.fill, ControlRod):
                if self.fill.clad_radius >= self.inner_radius:
                    raise ValueError("The control rod is too large for the guide tube.")
            else:
                raise TypeError("Unknown fill object placed in guide tube.")

        # ======================================================================
        # DANCOFF CORRECTION CALCULATION DATA
        # ----------------------------------------------------------------------

        # There could be 2 possible Dancoff corrections. One where the control
        # rods are inserted, and another where they are not.
        self._clad_dancoff_correction_no_control_rods: float = 0.0
        self._clad_dancoff_correction_control_rods: float = 0.0
        # This is the "active" Dancoff correction of the prior two
        self._clad_dancoff_correction: float = 0.0

        # Initialize empty variables for Dancoff correction calculations.
        # These are all kept private.
        self._clad_dancoff_xs: CrossSection = CrossSection(
            np.array([self.clad.potential_xs]),
            np.array([self.clad.potential_xs]),
            np.array([[0.0]]),
            "Clad",
        )

        self._clad_isolated_dancoff_fsr_ids = []
        self._clad_isolated_dancoff_fsr_inds = []
        self._mod_isolated_dancoff_fsr_ids = []
        self._mod_isolated_dancoff_fsr_inds = []

        self._clad_full_dancoff_fsr_ids = []
        self._clad_full_dancoff_fsr_inds = []
        self._mod_full_dancoff_fsr_ids = []
        self._mod_full_dancoff_fsr_inds = []

        # ======================================================================
        # TRANSPORT CALCULATION DATA
        # ----------------------------------------------------------------------
        self._clad_fsr_ids: List[int] = []
        self._mod_fsr_ids: List[int] = []

        self._clad_fsr_inds: List[int] = []
        self._mod_fsr_inds: List[int] = []

        # Holds the CrossSection object used for the real transport calc.
        self._clad_xs: Optional[CrossSection] = None

    @property
    def clad(self) -> Material:
        return self._clad

    @property
    def inner_radius(self) -> float:
        return self._inner_radius

    @property
    def outer_radius(self) -> float:
        return self._outer_radius

    @property
    def fill(self) -> Optional[BurnablePoisonRod | ControlRod]:
        return self._fill

    @property
    def empty(self) -> bool:
        return self.fill is None

    @property
    def clad_dancoff_correction_no_control_rods(self) -> float:
        return self._clad_dancoff_correction_no_control_rods

    @clad_dancoff_correction_no_control_rods.setter
    def clad_dancoff_correction_no_control_rods(self, val: float) -> None:
        if val < 0.0 or val > 1.0:
            raise ValueError(
                f"Dancoff correction must be in interval [0,1]. Was provided {val}."
            )
        self._clad_dancoff_correction_no_control_rods = val

    @property
    def clad_dancoff_correction_control_rods(self) -> float:
        return self._clad_dancoff_correction_control_rods

    @clad_dancoff_correction_control_rods.setter
    def clad_dancoff_correction_control_rods(self, val: float) -> None:
        if val < 0.0 or val > 1.0:
            raise ValueError(
                f"Dancoff correction must be in interval [0,1]. Was provided {val}."
            )
        self._clad_dancoff_correction_control_rods = val

    @property
    def clad_dancoff_correction(self) -> float:
        return self._clad_dancoff_correction

    def _check_dx_dy(self, dx, dy, pintype):
        if pintype == PinCellType.Full:
            if dx < 2.0 * self.outer_radius:
                raise ValueError(
                    "The fuel pin cell x width must be > the diameter of the cladding."
                )
            if dy < 2.0 * self.outer_radius:
                raise ValueError(
                    "The fuel pin cell y width must be > the diameter of the cladding."
                )
        elif pintype in [PinCellType.XN, PinCellType.XP]:
            if dx < self.outer_radius:
                raise ValueError(
                    "The fuel pin cell x width must be > the radius of the cladding."
                )
            if dy < 2.0 * self.outer_radius:
                raise ValueError(
                    "The fuel pin cell y width must be > the diameter of the cladding."
                )
        elif pintype in [PinCellType.YN, PinCellType.YP]:
            if dy < self.outer_radius:
                raise ValueError(
                    "The fuel pin cell y width must be > the radius of the cladding."
                )
            if dx < 2.0 * self.outer_radius:
                raise ValueError(
                    "The fuel pin cell x width must be > the diameter of the cladding."
                )
        else:
            if dx < self.outer_radius:
                raise ValueError(
                    "The fuel pin cell x width must be > the radius of the cladding."
                )
            if dy < self.outer_radius:
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
        self.clad.load_nuclides(ndl)

    def use_no_control_rod_dancoff_correction(self) -> None:
        """
        Will use the Dancoff correction for the cladding which assumes no
        control rods are present in the assembly.
        """
        self._clad_dancoff_correction = self._clad_dancoff_correction_no_control_rods

    def use_control_rod_dancoff_correction(self) -> None:
        """
        Will use the Dancoff correction for the cladding which assumes
        control rods are present in the assembly.
        """
        self._clad_dancoff_correction = self._clad_dancoff_correction_control_rods

    def remove_control_rod(self) -> None:
        """
        If the guide tube has a control rod fill, its cross sections will be
        set to that of the moderator.
        """
        if isinstance(self.fill, ControlRod):
            self.fill.is_moderator = True

    def insert_control_rod(self) -> None:
        """
        If the guide tube has a control rod fill, its cross sections will be
        set to that of the actual control rod.
        """
        if isinstance(self.fill, ControlRod):
            self.fill.is_moderator = False

    # ==========================================================================
    # Dancoff Correction Related Methods
    def set_xs_for_fuel_dancoff_calculation(self, moderator: Material) -> None:
        """
        Sets the 1-group cross sections to calculate the fuel Dancoff
        corrections.

        Parameters
        ----------
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        """
        self._clad_dancoff_xs.set(
            CrossSection(
                np.array([self.clad.potential_xs]),
                np.array([self.clad.potential_xs]),
                np.array([[0.0]]),
                "Clad",
            )
        )

        if isinstance(self.fill, BurnablePoisonRod):
            self.fill.set_xs_for_dancoff_calculation()
        elif isinstance(self.fill, ControlRod):
            self.fill.set_xs_for_dancoff_calculation(moderator)

    def set_xs_for_clad_dancoff_calculation(self, moderator: Material) -> None:
        """
        Sets the 1-group cross sections to calculate the clad Dancoff
        corrections.

        Parameters
        ----------
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        """
        self._clad_dancoff_xs.set(
            CrossSection(
                np.array([1.0e5]),
                np.array([1.0e5]),
                np.array([[0.0]]),
                "Clad",
            )
        )

        if isinstance(self.fill, BurnablePoisonRod):
            self.fill.set_xs_for_dancoff_calculation()
        elif isinstance(self.fill, ControlRod):
            self.fill.set_xs_for_dancoff_calculation(moderator)

    def set_xs_for_control_rod_dancoff_calculation(self, moderator: Material) -> None:
        """
        Sets the 1-group cross sections to calculate the control rod Dancoff
        corrections.

        Parameters
        ----------
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        """
        self._clad_dancoff_xs.set(
            CrossSection(
                np.array([self.clad.potential_xs]),
                np.array([self.clad.potential_xs]),
                np.array([[0.0]]),
                "Clad",
            )
        )

        if isinstance(self.fill, BurnablePoisonRod):
            self.fill.set_xs_for_dancoff_calculation()
        elif isinstance(self.fill, ControlRod):
            self.fill.set_xs_for_control_rod_dancoff_calculation(moderator)

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
        GuideTube object.

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
            How the cell should be split (along x, y, or only a quadrant).
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

        if isinstance(self.fill, BurnablePoisonRod) or isinstance(
            self.fill, ControlRod
        ):
            radii, xs = self.fill._make_dancoff_moc_cell(moderator_xs)
        FILL_OFFSET = len(radii)

        # This is either all the moderator inside the guide tube, or it is the
        # ring of moderator inside the guide tube, but outside the fill object.
        radii.append(self.inner_radius)
        xs.append(moderator_xs)

        radii.append(self.outer_radius)
        xs.append(self._clad_dancoff_xs)

        xs.append(moderator_xs)

        # Make the simple pin cell.
        cell = SimplePinCell(radii, xs, dx, dy, pintype)

        # Get the FSR IDs for the regions of interest
        cell_fsr_ids = cell.get_all_fsr_ids()

        if isolated:
            self._clad_isolated_dancoff_fsr_ids.append(cell_fsr_ids[FILL_OFFSET + 1])
            self._mod_isolated_dancoff_fsr_ids = [
                cell_fsr_ids[FILL_OFFSET + 0],
                cell_fsr_ids[FILL_OFFSET + 2],
            ]

            # Must set FSR IDs by hand for the fill
            if isinstance(self.fill, BurnablePoisonRod):
                if self.fill.center is None:
                    self._mod_isolated_dancoff_fsr_ids.append(cell_fsr_ids[0])
                else:
                    self.fill._center_isolated_dancoff_fsr_ids.append(cell_fsr_ids[0])

                self.fill._clad_isolated_dancoff_fsr_ids = [
                    cell_fsr_ids[1],
                    cell_fsr_ids[5],
                ]
                self.fill._gap_isolated_dancoff_fsr_ids = [
                    cell_fsr_ids[2],
                    cell_fsr_ids[4],
                ]
                self.fill._poison_isolated_dancoff_fsr_ids = [cell_fsr_ids[3]]
            elif isinstance(self.fill, ControlRod):
                self.fill._absorber_isolated_dancoff_fsr_ids.append(cell_fsr_ids[0])
                self.fill._gap_isolated_dancoff_fsr_ids.append(cell_fsr_ids[1])
                self.fill._clad_isolated_dancoff_fsr_ids.append(cell_fsr_ids[2])

        else:
            self._clad_full_dancoff_fsr_ids.append(cell_fsr_ids[FILL_OFFSET + 1])
            self._mod_full_dancoff_fsr_ids = [
                cell_fsr_ids[FILL_OFFSET + 0],
                cell_fsr_ids[FILL_OFFSET + 2],
            ]

            # Must set FSR IDs by hand for the fill
            if isinstance(self.fill, BurnablePoisonRod):
                if self.fill.center is None:
                    self._mod_full_dancoff_fsr_ids.append(cell_fsr_ids[0])
                else:
                    self.fill._center_full_dancoff_fsr_ids.append(cell_fsr_ids[0])

                self.fill._clad_full_dancoff_fsr_ids = [
                    cell_fsr_ids[1],
                    cell_fsr_ids[5],
                ]
                self.fill._gap_full_dancoff_fsr_ids = [cell_fsr_ids[2], cell_fsr_ids[4]]
                self.fill._poison_full_dancoff_fsr_ids = [cell_fsr_ids[3]]
            elif isinstance(self.fill, ControlRod):
                self.fill._absorber_full_dancoff_fsr_ids.append(cell_fsr_ids[0])
                self.fill._gap_full_dancoff_fsr_ids.append(cell_fsr_ids[1])
                self.fill._clad_full_dancoff_fsr_ids.append(cell_fsr_ids[2])

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
        self._clad_isolated_dancoff_fsr_inds = []
        self._mod_isolated_dancoff_fsr_inds = []

        self._clad_full_dancoff_fsr_inds = []
        self._mod_full_dancoff_fsr_inds = []

        for id in self._clad_isolated_dancoff_fsr_ids:
            self._clad_isolated_dancoff_fsr_inds.append(isomoc.get_fsr_indx(id, 0))
        for id in self._mod_isolated_dancoff_fsr_ids:
            self._mod_isolated_dancoff_fsr_inds.append(isomoc.get_fsr_indx(id, 0))

        for id in self._clad_full_dancoff_fsr_ids:
            self._clad_full_dancoff_fsr_inds.append(fullmoc.get_fsr_indx(id, 0))
        for id in self._mod_full_dancoff_fsr_ids:
            self._mod_full_dancoff_fsr_inds.append(fullmoc.get_fsr_indx(id, 0))

        if isinstance(self.fill, BurnablePoisonRod) or isinstance(
            self.fill, ControlRod
        ):
            self.fill.populate_dancoff_fsr_indexes(isomoc, fullmoc)

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
        # Clad sources should all be potential_xs
        pot_xs = self.clad.potential_xs
        for ind in self._clad_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, pot_xs)

        # Moderator sources should all be potential_xs
        pot_xs = moderator.potential_xs
        for ind in self._mod_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, pot_xs)

        if isinstance(self.fill, BurnablePoisonRod) or isinstance(
            self.fill, ControlRod
        ):
            self.fill.set_isolated_dancoff_fuel_sources(isomoc, moderator)

    def set_isolated_dancoff_clad_sources(
        self, isomoc: MOCDriver, moderator: Material
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
        """
        # Clad sources should all be zero !
        for ind in self._clad_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, 0.0)

        # Moderator sources should all be potential_xs
        pot_xs = moderator.potential_xs
        for ind in self._mod_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, pot_xs)

        if isinstance(self.fill, BurnablePoisonRod) or isinstance(
            self.fill, ControlRod
        ):
            self.fill.set_isolated_dancoff_clad_sources(isomoc, moderator)

    def set_isolated_dancoff_control_rod_sources(
        self, isomoc: MOCDriver, moderator: Material
    ) -> None:
        """
        Initializes the fixed sources for the isolated MOC calculation required
        in computing Dancoff corrections. Sources are set for a control rod
        Dancoff correction calculation.

        Parameters
        ----------
        isomoc : MOCDriver
            MOC simulation for the isolated geometry.
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        """
        # Clad sources should all be potential_xs
        pot_xs = self.clad.potential_xs
        for ind in self._clad_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, pot_xs)

        # Moderator sources should all be potential_xs
        pot_xs = moderator.potential_xs
        for ind in self._mod_isolated_dancoff_fsr_inds:
            isomoc.set_extern_src(ind, 0, pot_xs)

        if isinstance(self.fill, BurnablePoisonRod) or isinstance(
            self.fill, ControlRod
        ):
            self.fill.set_isolated_dancoff_control_rod_sources(isomoc, moderator)

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
        # Clad sources should all be potential_xs
        pot_xs = self.clad.potential_xs
        for ind in self._clad_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, pot_xs)

        # Moderator sources should all be potential_xs
        pot_xs = moderator.potential_xs
        for ind in self._mod_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, pot_xs)

        if isinstance(self.fill, BurnablePoisonRod) or isinstance(
            self.fill, ControlRod
        ):
            self.fill.set_full_dancoff_fuel_sources(fullmoc, moderator)

    def set_full_dancoff_clad_sources(
        self, fullmoc: MOCDriver, moderator: Material
    ) -> None:
        """
        Initializes the fixed sources for the full MOC calculation required
        in computing Dancoff corrections. Sources are set for a clad Dancoff
        correction calculation.

        Parameters
        ----------
        fullmoc : MOCDriver
            MOC simulation for the full geometry.
        moderator : Material
            Material definition for the moderator, used to obtain the potential
            scattering cross section.
        """
        # Clad sources should all be zero !
        for ind in self._clad_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, 0.0)

        # Moderator sources should all be potential_xs
        pot_xs = moderator.potential_xs
        for ind in self._mod_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, pot_xs)

        if isinstance(self.fill, BurnablePoisonRod) or isinstance(
            self.fill, ControlRod
        ):
            self.fill.set_full_dancoff_clad_sources(fullmoc, moderator)

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
        # Clad sources should all be potential_xs
        pot_xs = self.clad.potential_xs
        for ind in self._clad_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, pot_xs)

        # Moderator sources should all be potential_xs
        pot_xs = moderator.potential_xs
        for ind in self._mod_full_dancoff_fsr_inds:
            fullmoc.set_extern_src(ind, 0, pot_xs)

        if isinstance(self.fill, BurnablePoisonRod) or isinstance(
            self.fill, ControlRod
        ):
            self.fill.set_full_dancoff_control_rod_sources(fullmoc, moderator)

    def compute_clad_dancoff_correction(
        self, isomoc: MOCDriver, fullmoc: MOCDriver
    ) -> float:
        """
        Computes the Dancoff correction for the cladding region.

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

    # ==========================================================================
    # Transport Calculation Related Methods
    def set_clad_xs_for_depletion_step(self, t: int, ndl: NDLibrary) -> None:
        """
        Constructs the CrossSection object for the cladding of the guide tube
        at the specified depletion step. The depletion step only changes the
        Dancoff correction, not the cladding composition.

        Parameters
        ----------
        t : int
            Index for the depletion step.
        ndl : NDLibrary
            Nuclear data library to use for cross sections.
        """
        # Compute escape xs
        Ee = 1.0 / (2.0 * (self.outer_radius - self.inner_radius))
        new_clad_xs = self.clad.roman_xs(self._clad_dancoff_correction, Ee, ndl)

        # Set the xs
        if self._clad_xs is None:
            self._clad_xs = new_clad_xs
        else:
            self._clad_xs.set(new_clad_xs)

        if self._clad_xs.name == "":
            self._clad_xs.name = "Clad"

    def set_fill_xs_for_depletion_step(
        self, t: int, moderator_xs: CrossSection, ndl: NDLibrary
    ) -> None:
        """
        Constructs the CrossSection objects for the fill of the guide tube
        at the specified depletion step. The depletion step changes the poison
        composition, if filled with a burnable poison rod, or the control rod
        composition if filled with a control rod.

        Parameters
        ----------
        t : int
            Index for the depletion step.
        moderator_xs : CrossSection
            Cross sections for the moderator.
        ndl : NDLibrary
            Nuclear data library to use for cross sections.
        """
        if self.empty:
            return

        if isinstance(self.fill, BurnablePoisonRod):
            self.fill.set_center_xs(ndl)
            self.fill.set_gap_xs(ndl)
            self.fill.set_clad_xs(ndl)
            self.fill.set_poison_xs_for_depletion_step(t, ndl)
        elif isinstance(self.fill, ControlRod):
            self.fill.set_xs(t, moderator_xs, ndl)

    def make_moc_cell(
        self, moderator_xs: CrossSection, dx: float, dy: float, pintype: PinCellType
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
        """
        if self._clad_xs is None:
            raise RuntimeError("Clad cross section has not yet been built.")
        self._check_dx_dy(dx, dy, pintype)

        # Initialize empty lists
        radii = []
        xss = []
        FILL_OFFSET = 0
        NUM_MOD_RINGS = 3

        if self.empty:
            # For an empty guide tube, we use 3 rings of water
            # Create list of inner moderator radii
            V = np.pi * self.inner_radius * self.inner_radius
            Vr = V / 3.0
            for ri in range(3):
                Rin = 0.0
                if ri > 0:
                    Rin = radii[-1]
                Rout = np.sqrt((Vr + np.pi * Rin * Rin) / np.pi)
                if Rout > self.inner_radius:
                    Rout = self.inner_radius
                radii.append(Rout)

            # Initialize the cross section lists with the inner moderator xs
            xss += len(radii) * [moderator_xs]
        elif isinstance(self.fill, BurnablePoisonRod) or isinstance(
            self.fill, ControlRod
        ):
            radii, xss = self.fill._make_moc_cell(moderator_xs)
            FILL_OFFSET = len(radii)

            # Add the layer of moderator outside the poison rod
            radii.append(self.inner_radius)
            xss.append(moderator_xs)
            NUM_MOD_RINGS = 1
        else:
            # Should never get here due to check in constructor, but in any case
            raise TypeError("Unknown fill object placed in guide tube.")

        # Add cladding
        radii.append(self.outer_radius)
        xss.append(self._clad_xs)

        # Add another ring of moderator if possible
        if (
            pintype == PinCellType.Full
            and min(dx, dy) > 2.0 * self.outer_radius
            and 0.5 * min(dx, dy) - self.outer_radius >= 0.1
        ):
            radii.append(0.5 * min(dx, dy))
            xss.append(moderator_xs)
        elif (
            pintype in [PinCellType.XN, PinCellType.XP]
            and dx > self.outer_radius
            and dy > 2.0 * self.outer_radius
            and min(dx, 0.5 * dy) - self.outer_radius >= 0.1
        ):
            radii.append(min(dx, 0.5 * dy))
            xss.append(moderator_xs)
        elif (
            pintype in [PinCellType.YN, PinCellType.YP]
            and dy > self.outer_radius
            and dx > 2.0 * self.outer_radius
            and min(0.5 * dx, dy) - self.outer_radius >= 0.1
        ):
            radii.append(min(0.5 * dx, dy))
            xss.append(moderator_xs)
        elif (
            pintype in [PinCellType.I, PinCellType.II, PinCellType.III, PinCellType.IV]
            and dx > self.outer_radius
            and dy > self.outer_radius
            and min(dx, dy) - self.outer_radius >= 0.1
        ):
            radii.append(min(dx, dy))
            xss.append(moderator_xs)

        # Add moderator to the end of materials
        xss.append(moderator_xs)

        cell = PinCell(radii, xss, dx, dy, pintype)

        # Get the FSR IDs for the regions of interest
        cell_fsr_ids = cell.get_all_fsr_ids()

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

        # Get all the FSR ids for the poison rod
        if isinstance(self.fill, BurnablePoisonRod):
            if self.fill.center is not None:
                self.fill._center_fsr_ids += cell_fsr_ids[0:NA]
            self.fill._clad_fsr_ids += cell_fsr_ids[NA : 2 * NA]
            self.fill._gap_fsr_ids += cell_fsr_ids[2 * NA : 3 * NA]
            self.fill._poison_fsr_ids += cell_fsr_ids[3 * NA : 4 * NA]
            self.fill._gap_fsr_ids += cell_fsr_ids[4 * NA : 5 * NA]
            self.fill._clad_fsr_ids += cell_fsr_ids[5 * NA : 6 * NA]
        elif isinstance(self.fill, ControlRod):
            for r in range(self.fill.num_rings):
                self.fill._absorber_ring_fsr_ids[r] += cell_fsr_ids[
                    r * NA : (r + 1) * NA
                ]
            self.fill._gap_fsr_ids += cell_fsr_ids[
                self.fill.num_rings * NA : (self.fill.num_rings + 1) * NA
            ]
            self.fill._clad_fsr_ids += cell_fsr_ids[
                (self.fill.num_rings + 1) * NA : (self.fill.num_rings + 2) * NA
            ]

        I = 0  # Starting index for cell_fsr_inds
        # Go through all rings of moderator and get FSR IDs
        for a in range(NUM_MOD_RINGS * NA):
            self._mod_fsr_ids.append(cell_fsr_ids[FILL_OFFSET * NA + I])
            I += 1

        # If we have a poision rod with a water center, add those FSR IDs to the moderator list
        if isinstance(self.fill, BurnablePoisonRod):
            if self.fill.center is None:
                self._mod_fsr_ids += cell_fsr_ids[0:NA]

        # Get FSR IDs for the cladding
        for a in range(NA):
            self._clad_fsr_ids.append(cell_fsr_ids[FILL_OFFSET * NA + I])
            I += 1

        # Everything outside the clad should be moderator
        self._mod_fsr_ids = list(cell_fsr_ids[FILL_OFFSET * NA + I :])

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
        self._clad_fsr_inds: List[int] = []
        self._mod_fsr_inds: List[int] = []

        for id in self._clad_fsr_ids:
            self._clad_fsr_inds.append(moc.get_fsr_indx(id, 0))
        for id in self._mod_fsr_ids:
            self._mod_fsr_inds.append(moc.get_fsr_indx(id, 0))

        if isinstance(self.fill, BurnablePoisonRod) or isinstance(
            self.fill, ControlRod
        ):
            self.fill.populate_fsr_indexes(moc)

    def obtain_flux_spectra(self, moc: MOCDriver) -> None:
        """
        If the guide tube contains a burnable poison rod or a control rod, the
        average flux spectrum is obtained from the MOC simulation.

        Parameters
        ----------
        moc : MOCDriver
            MOC simulation for the full calculations.
        """
        if (
            not self.empty
            and isinstance(self.fill, BurnablePoisonRod)
            or isinstance(self.fill, ControlRod)
        ):
            self.fill.obtain_flux_spectra(moc)

    def normalize_flux_spectrum(self, f) -> None:
        """
        If the guide tube contains a burnable poison rod or a control rod, it
        applies a multiplicative factor to the flux spectra. This permits
        normalizing the flux to a known assembly power.

        Parameters
        ----------
        f : float
            Normalization factor.
        """
        if (
            not self.empty
            and isinstance(self.fill, BurnablePoisonRod)
            or isinstance(self.fill, ControlRod)
        ):
            self.fill.normalize_flux_spectrum(f)

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

        if not self.empty:
            if isinstance(self.fill, BurnablePoisonRod) or (
                isinstance(self.fill, ControlRod) and not self.fill.is_moderator
            ):
                self.fill.predict_depletion(chain, ndl, dt, dtm1)

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

        if not self.empty:
            if isinstance(self.fill, BurnablePoisonRod) or (
                isinstance(self.fill, ControlRod) and not self.fill.is_moderator
            ):
                self.fill.correct_depletion(chain, ndl, dt, dtm1)
