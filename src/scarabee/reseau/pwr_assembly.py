from scarabee.reseau.burnable_poison_rod import BurnablePoisonRod
from .fuel_pin import FuelPin
from .guide_tube import GuideTube
from .critical_leakage import CriticalLeakage
from ._ensleeve import (
    _ensleeve_quarter,
    _ensleeve_half_top,
    _ensleeve_half_right,
    _ensleeve_full,
)
from .nodal_flux import NodalFlux1D, NodalFlux2D
from .._scarabee import (
    borated_water,
    Material,
    CrossSection,
    NDLibrary,
    DepletionChain,
    PinCellType,
    Vector,
    Direction,
    Cartesian2D,
    MOCDriver,
    CMFD,
    BoundaryCondition,
    SimulationMode,
    YamamotoTabuchi6,
    P1CriticalitySpectrum,
    B1CriticalitySpectrum,
    ADF,
    CDF,
    DiffusionCrossSection,
    DiffusionData,
    set_logging_level,
    scarabee_log,
    LogLevel,
)
from enum import Enum
import numpy as np
from typing import Optional, List, Tuple, Union
import copy
from threading import Thread


class Symmetry(Enum):
    """
    Defines the symmetry to use when simulating a PWR fuel assembly.
    """

    Full = 1
    """
    The fuel assembly has no symmetry.
    """

    Half = 2
    """
    The fuel assembly has about either the x or y axis (but not both).
    """

    Quarter = 3
    """
    The fuel assembly has symmetry about the x and y axis.
    """


class PWRAssembly:
    """
    A PWRAssembly instance is responsible for performing all the lattice
    calculations necessary to produce few-group cross sections for a
    single PWR assembly.

    Parameters
    ----------
    shape : tuple of int
        The number of pin cells in the full assembly along x and y.
    pitch : float
        The spacing between fuel pins.
    ndl : NDLibrary
        Nuclear data library used for the calculation.
    cells : list of list of FuelPin or GuideTube
        All of the cells which describe the assembly geometry. Should be
        consistent with the symmetry argument.
    boron_ppm : float
        Moderator boron concentration in parts per million. Default is 800.
    moderator_temp : float
        Moderator temperature in Kelvin. Default is 570.
    moderator_pressure : float
        Moderator pressure in MPa. Default is 15.5.
    moderator_legendre_order : int
        Maximum Legendre order to load for the moderator's anisotropic
        scattering data. Default is 1.
    symmetry : Symmetry
        Symmetry of the fuel assembly. Default is Symmetry.Full.
    independent_quadrant : bool
        If symmetry is Symmetry.Quarter and this attribute is true, the quarter
        assembly is treated as an independent assembly, with ADFs generated for
        all sides and CDFs generated for all corners. Form factors are only
        generated for the pins present in the quarter assembly. This improves
        the modeling of asymmetric burnable absorber loadings in the nodal
        calculation. Default value is False.
    linear_power : float
        Linear power density of the full assembly in kW/cm. This value should
        not be reduced due to symmetry. Default is 42.
    assembly_pitch : optional float
        Spacing between fuel assemblies. If None, assembly pitch is calculated
        from the shape and pitch. Default value is None.
    spacer_grid_width : optional float
        Width of the spacer grid material between pin cells.
        Default value is None.
    spacer_grid : optional Material
        Material defining the spacer grid between pin cells.
    grid_sleeve_width : optional float
        Width of the grid sleeve around the assembly. Default value is None.
    grid_sleeve : optional Material
        Material defining the grid sleeve around the assembly.

    Attributes
    ----------
    shape : tuple of int
        The number of pin cells in the full assembly along x and y.
    pitch : float
        The spacing between fuel pins.
    symmetry : Symmetry
        Symmetry of the fuel assembly.
    independent_quadrant : bool
        If symmetry is Symmetry.Quarter and this attribute is true, the quarter
        assembly is treated as an independent assembly, with ADFs generated for
        all sides and CDFs generated for all corners. Form factors are only
        generated for the pins present in the quarter assembly. This improves
        the modeling of asymmetric burnable absorber loadings in the nodal
        calculation.
    assembly_pitch : float
        Spacing between fuel assemblies.
    fuel_volume_fraction : float
        Fraction of the assembly occupied by fuel.
    moderator_volume_fraction : float
        Fraction of the assembly occupied by moderator.
    linear_power : float
        Linear power density of the full assembly in kW/cm.
    initial_heavy_metal_linear_mass : float
        Initial linear density of heavy metal in the full assembly at the
        beginning of life in units of kg/cm.
    boron_ppm : float
        Moderator boron concentration in parts per million.
    moderator_temp : float
        Moderator temperature in Kelvin.
    moderator_pressure : float
        Moderator pressure in MPa.
    moderator_legendre_order : int
        Maximum Legendre order to load for the moderator's anisotropic
        scattering data.
    moderator : Material
        Material representing the assembly moderator.
    spacer_grid_width : optional float
        Width of the spacer grid material between pin cells.
    spacer_grid : optional Material
        Material defining the spacer grid between pin cells.
    grid_sleeve_width : optional float
        Width of the grid sleeve around the assembly.
    grid_sleeve : optional Material
        Material defining the grid sleeve around the assembly.
    dancoff_moc_track_spacing : float
        Spacing between tracks in the MOC calculations for determining Dancoff
        corrections. Default value is 0.05 cm.
    dancoff_moc_num_angles : int
        Number of azimuthal angles in the MOC calculations for determining
        Dancoff corrections. Default value is 32.
    dancoff_flux_tolerance : float
        Flux convergence tolerance for Dancoff correction calculations. Must be
        in range (0., 1.E-2). Default value is 1.E-5.
    moc_track_spacing : float
        Spacing between tracks in the assembly MOC calculations. Default value
        is 0.05 cm.
    moc_num_angles : int
        Number of azimuthal angles in the assembly MOC calculations. Default
        value is 32.
    flux_tolerance : float
        Flux convergence tolerance for assembly calculations. Must be in range
        (0., 1.E-2). Default value is 1.E-5.
    keff_tolerance : float
        Keff convergence tolerance for assembly calculations. Must be in range
        (0., 1.E-2). Default value is 1.E-5.
    anisotropic : bool
        If False, isotropic scattering with the transport correction is used in
        the assembly calculation. Otherwise, explicit anisotropic scattering is
        modeled. Default value is False. **If you wish to turn anisotropic
        scattering on, you must set this attribute before calling the solve
        method for the first time.**
    cmfd : bool
        If True, Coarse Mesh Finite Difference diffusion will be used to
        accelerate the assembly calculation. If True, a CMFD energy
        condensation scheme must be provided (see cmfd_condensation_scheme).
        Default value is True. **If you wish to turn CMFD off/on, you must set
        this attribute before calling the solve method for the first time.**
    cmfd_condensation_scheme : list of list of int
        Energy condensation scheme for the CMFD acceleration of the assembly
        calculation.
    condensation_scheme : list of list of int
        Energy condensation scheme to condense from the group structure of the
        library to the few-groups used in the core solver.
    prefer_moc_adf_cdf : bool
        If True, the MOC results will be used to generate the ADFs and CDFs,
        even if CMFD is being used. ADFs and CDFs that are generated in this
        manner are typically less accurate than those generated with the CMFD
        results. Default value is False.
    leakage_model : CriticalLeakage
        Model used to determine the critical leakage flux spectrum, also known
        as the fundamental mode. Default method is homogeneous P1.
    depletion_exposure_steps : optional ndarray
        1D Numpy array of assembly burn-up exposure steps, in units of MWd/kg.
        Default is None.
    depletion_time_steps : optional ndarray
        1D Numpy array of burn-up time steps, in units of days.
        Default is None.
    corrector_transport : bool
        If True, the corrector depletion step performs a transport calculation
        to obtain an updated solution for the flux based on the predictor
        nuclide concentrations, performing two transport calculations per time
        step. If False, the flux from the predictor transport calculation is
        used to perform the corrector step, performing only one transport
        calculation per time step. Default value is True.
    exposures : ndarray
        1D Numpy array of the total assembly burn-up exposures at which
        material information is available, in units of MWd/kg. Default value
        is an empty array before solve has been called.
    times : ndarray
        1D Numpy array of the total assembly burn-up times at which material
        information is available, in units of days. Default value is an empty
        array before solve has been called.
    keff : float or ndarray
        If depletion was not performed, this is a single float with keff for
        the infinite assembly. If depletion was performed, this is a 1D Numpy
        array for the values of keff at the tabulated burn-up exposures/times.
        Default value is 1 before solve has been called.
    depletion_data : optional DiffusionData or list of DiffusionData
        If a single assembly calculation is being performed without depletion,
        this attribute will the resulting DiffusionData instance based on the
        few-group condensation_scheme attribute. If a depletion calculation is
        performed, this attribute will be a list of DiffusionData instances,
        with one for each burn-up point. Before solve has been called, this
        attribute is None.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        pitch: float,
        ndl: NDLibrary,
        cells: List[List[Union[FuelPin, GuideTube]]],
        boron_ppm: float = 800.0,
        moderator_temp: float = 570.0,
        moderator_pressure: float = 15.5,
        moderator_legendre_order: int = 1,
        symmetry: Symmetry = Symmetry.Full,
        independent_quadrant: bool = False,
        linear_power: float = 42.0,
        assembly_pitch: Optional[float] = None,
        spacer_grid_width: Optional[float] = None,
        spacer_grid: Optional[Material] = None,
        grid_sleeve_width: Optional[float] = None,
        grid_sleeve: Optional[Material] = None,
    ):
        self._ndl: NDLibrary = ndl
        self._chain: DepletionChain = self._ndl.depletion_chain
        self._symmetry: Symmetry = symmetry
        self._independent_quadrant: bool = independent_quadrant

        if self.symmetry != Symmetry.Quarter and self.independent_quadrant:
            raise RuntimeError(
                "Cannot have independent_quadrant set to True when not using quarter symmetry."
            )

        if len(shape) != 2:
            raise ValueError("Shape must have 2 entries.")
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("Shape entries must be > 0.")
        if shape[0] != shape[1]:
            raise ValueError("Fuel assemblies must be square.")
        self._shape = shape

        # Compute the number of expected cells along each dimension [y][x]
        # based on the symmetry of the problem being solved.
        expx = self.shape[0]
        expy = self.shape[1]
        if self.symmetry == Symmetry.Half:
            expy = (expy // 2) + (expy % 2)
        elif self.symmetry == Symmetry.Quarter:
            expy = (expy // 2) + (expy % 2)
            expx = (expx // 2) + (expx % 2)
        self._simulated_shape = (expx, expy)

        self._initial_heavy_metal_linear_mass: float = 0.0

        self._cells_set = False
        self._cells: List[List[Union[FuelPin, GuideTube]]] = [[]]
        self._set_cells(cells)

        if pitch <= 0.5:
            raise ValueError("Pitch must be > 0.5.")
        self._pitch = pitch

        # Compute assembly pitch
        self._assembly_pitch: float = self.shape[0] * self.pitch
        if assembly_pitch is not None:
            try:
                assembly_pitch = float(assembly_pitch)
            except:
                raise TypeError(
                    "The assembly_pitch argument must be convertible to a float."
                )

            if assembly_pitch <= 0.0:
                raise RuntimeError("Assembly pitch must be > 0.")
            elif assembly_pitch <= self._assembly_pitch:
                raise RuntimeError(
                    "Provided assembly pitch is smaller than that indicated by the pitch and shape."
                )
            self._assembly_pitch = assembly_pitch

        # Can only compute the fuel volume fraction once the assembly pitch is known
        self._fuel_volume_fraction: float = 0.0
        self._compute_fuel_volume_fraction()

        # Check the spacer grid parameters
        self._spacer_grid_width: Optional[float] = None
        self._spacer_grid: Optional[Material] = None
        if spacer_grid is None and spacer_grid_width is not None:
            raise RuntimeError("Spacer grid width is provided but material is not.")
        elif spacer_grid is not None and spacer_grid_width is None:
            raise RuntimeError("Spacer grid material is provided but width is not.")
        else:
            self._spacer_grid_width = spacer_grid_width
            self._spacer_grid = spacer_grid

            if self.spacer_grid_width is not None:
                if self._spacer_grid_width <= 0.0:
                    raise ValueError("Spacer grid width must be > 0.")
                elif self._spacer_grid_width >= self._pitch:
                    raise ValueError("Spacer grid width must be < pitch.")

        # Check the grid sleeve parameters
        self._grid_sleeve_width: Optional[float] = None
        self._grid_sleeve: Optional[Material] = None
        if grid_sleeve is None and grid_sleeve_width is not None:
            raise RuntimeError("Grid sleeve width is provided but material is not.")
        elif grid_sleeve is not None and grid_sleeve_width is None:
            raise RuntimeError("Grid sleeve material is provided but width is not.")
        else:
            self._grid_sleeve_width = grid_sleeve_width
            self._grid_sleeve = grid_sleeve

            if self.grid_sleeve_width is not None:
                if self._grid_sleeve_width <= 0.0:
                    raise ValueError("Grid sleeve width must be > 0.")
                elif self._grid_sleeve_width >= 0.5 * (
                    self._assembly_pitch - self.shape[0] * self.pitch
                ):
                    raise ValueError(
                        "Grid sleeve width must be < the assembly gap width."
                    )

        # Can only be computed after spacer grid and grid sleeve processing
        self._moderator_volume_fraction: float = 0.0
        self._compute_moderator_volume_fraction()

        # Get moderator parameters
        if boron_ppm < 0.0:
            raise ValueError("Boron concentration must be >= 0.")
        self._boron_ppm = boron_ppm

        if moderator_temp <= 0.0:
            raise ValueError("Moderator temperature must be > 0.")
        self._moderator_temp = moderator_temp

        if moderator_pressure <= 0.0:
            raise ValueError("Moderator pressure must be > 0.")
        self._moderator_pressure = moderator_pressure

        if moderator_legendre_order < 1:
            raise ValueError("Moderator Legendre order must be >= 1.")
        self._moderator_legendre_order = moderator_legendre_order

        if linear_power <= 0.0:
            raise ValueError("Linear power must be > 0.")
        self._linear_power = linear_power

        # Make material for borated water
        self._moderator: Material = borated_water(
            self.boron_ppm, self.moderator_temp, self.moderator_pressure, self._ndl
        )
        self._moderator.name = f"Moderator ({self.boron_ppm} ppm boron)"
        self._moderator.max_legendre_order = self._moderator_legendre_order

        # Set initial boundary conditions
        self._x_min_bc = BoundaryCondition.Periodic
        self._x_max_bc = BoundaryCondition.Periodic
        self._y_min_bc = BoundaryCondition.Periodic
        self._y_max_bc = BoundaryCondition.Periodic

        if self.symmetry != Symmetry.Full:
            self._y_min_bc = BoundaryCondition.Reflective
            self._y_max_bc = BoundaryCondition.Reflective

        if self.symmetry == Symmetry.Quarter:
            self._x_min_bc = BoundaryCondition.Reflective
            self._x_max_bc = BoundaryCondition.Reflective

        # ======================================================================
        # DANCOFF CORRECTION CALCULATION DATA
        # ----------------------------------------------------------------------

        # Make water xs for dancoff calculation
        self._moderator_dancoff_xs: CrossSection = CrossSection(
            np.array([self.moderator.potential_xs]),
            np.array([self.moderator.potential_xs]),
            np.array([[0.0]]),
            "Moderator",
        )

        # Spacer grid and grid sleeve dancoff cross sections
        self._spacer_grid_dancoff_xs: Optional[CrossSection] = None
        self._grid_sleeve_dancoff_xs: Optional[CrossSection] = None
        if self.spacer_grid is not None:
            self._spacer_grid_dancoff_xs: CrossSection = CrossSection(
                np.array([self.spacer_grid.potential_xs]),
                np.array([self.spacer_grid.potential_xs]),
                np.array([[0.0]]),
                "Spacer Grid",
            )
        if self.grid_sleeve is not None:
            self._grid_sleeve_dancoff_xs: CrossSection = CrossSection(
                np.array([self.grid_sleeve.potential_xs]),
                np.array([self.grid_sleeve.potential_xs]),
                np.array([[0.0]]),
                "Grid Sleeve",
            )

        # Isolated cell geometry for Dancoff correction calculations
        self._isolated_dancoff_cells = []
        self._isolated_dancoff_mocs = []

        # Full geometry for Dancoff correction calculations
        self._full_dancoff_cells = []
        self._full_dancoff_geom = None
        self._full_dancoff_moc = None

        # Dancoff correction parameters
        self._dancoff_isolation_scale = 10.0
        self._dancoff_moc_track_spacing = 0.05
        self._dancoff_moc_num_angles = 32
        self._dancoff_flux_tolerance = 1.0e-5

        self._fuel_dancoff_corrections = np.zeros(
            (self._simulated_shape[1], self._simulated_shape[0])
        )
        self._clad_dancoff_corrections = np.zeros(
            (self._simulated_shape[1], self._simulated_shape[0])
        )

        # ======================================================================
        # TRANSPORT CALCULATION DATA
        # ----------------------------------------------------------------------

        # Make water xs for transport calculation
        self._moderator_xs: CrossSection = self.moderator.dilution_xs(
            self.moderator.size * [1.0e10], self._ndl
        )

        # Spacer grid and grid sleeve cross sections
        self._spacer_grid_xs: Optional[CrossSection] = None
        self._grid_sleeve_xs: Optional[CrossSection] = None
        if self.spacer_grid is not None:
            self._spacer_grid_xs: CrossSection = self.spacer_grid.dilution_xs(
                self.spacer_grid.size * [1.0e10], self._ndl
            )
        if self.grid_sleeve is not None:
            self._grid_sleeve_xs: CrossSection = self.grid_sleeve.dilution_xs(
                self.grid_sleeve.size * [1.0e10], self._ndl
            )

        self._moc_track_spacing: float = 0.05
        self._moc_num_angles: int = 64
        self._flux_tolerance: float = 1.0e-5
        self._keff_tolerance: float = 1.0e-5
        self._anisotropic: bool = False
        self._cmfd: bool = True
        self._prefer_moc_adf_cdf: bool = False

        self._asmbly_cells = []
        self._asmbly_geom: Optional[Cartesian2D] = None
        self._asmbly_moc: Optional[MOCDriver] = None

        self._leakage_model: CriticalLeakage = CriticalLeakage.P1
        self._infinite_flux_spectrum = (
            None  # To reset to infinite spectrum in MOC driver
        )

        # Depletion time steps in MWd/kg and depletion time steps in days.
        # Initially starts as None (should be provided by user)
        self._depletion_exposure_steps: Optional[np.ndarray] = None
        self._depletion_time_steps: Optional[np.ndarray] = None

        # Arrays for the depletion exposures (MWd/kg) and times (days)
        self._exposures: np.ndarray = np.array([])
        self._times: np.ndarray = np.array([])

        # Indicates if 2 (True) or 1 (False) transport calculation should be
        # performed per time step.
        self._corrector_transport: bool = True

        # Either a single value or list of values (for each depletion step)
        self._keff: Union[float, List[float]] = 1.0

        # Condensation scheme to make few-group cross sections
        self._condensation_scheme: Optional[List[List[int]]] = (
            self._ndl.condensation_scheme
        )

        # Condensation scheme to used for CMFD acceleration
        self._cmfd_condensation_scheme: Optional[List[List[int]]] = (
            self._ndl.cmfd_condensation_scheme
        )

        # If a single assembly calculation is performed, this attribute will
        # contain a single DiffusionData instance. If a depletion calculation
        # is performed, a DiffusionData instance will be generated for each
        # burn-up step.
        self._diffusion_data: Optional[Union[DiffusionData, List[DiffusionData]]] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def pitch(self) -> float:
        return self._pitch

    @property
    def symmetry(self) -> Symmetry:
        return self._symmetry

    @property
    def independent_quadrant(self) -> bool:
        return self._independent_quadrant

    @property
    def assembly_pitch(self) -> float:
        return self._assembly_pitch

    @property
    def fuel_volume_fraction(self) -> float:
        return self._fuel_volume_fraction

    @property
    def moderator_volume_fraction(self) -> float:
        return self._moderator_volume_fraction

    @property
    def spacer_grid_width(self) -> Optional[float]:
        return self._spacer_grid_width

    @property
    def spacer_grid(self) -> Optional[Material]:
        return self._spacer_grid

    @property
    def grid_sleeve_width(self) -> Optional[float]:
        return self._grid_sleeve_width

    @property
    def grid_sleeve(self) -> Optional[Material]:
        return self._grid_sleeve

    @property
    def linear_power(self) -> float:
        return self._linear_power

    @property
    def initial_heavy_metal_linear_mass(self) -> float:
        return self._initial_heavy_metal_linear_mass

    @property
    def boron_ppm(self) -> float:
        return self._boron_ppm

    @property
    def moderator_temp(self) -> float:
        return self._moderator_temp

    @property
    def moderator_pressure(self) -> float:
        return self._moderator_pressure

    @property
    def moderator_legendre_order(self) -> int:
        return self._moderator_legendre_order

    @property
    def moderator(self) -> Material:
        return self._moderator

    @property
    def dancoff_moc_track_spacing(self) -> float:
        return self._dancoff_moc_track_spacing

    @dancoff_moc_track_spacing.setter
    def dancoff_moc_track_spacing(self, dts: float) -> None:
        if dts <= 0.0 or dts > 0.1:
            raise ValueError("Dancoff track spacing must be in range (0, 0.1].")
        self._dancoff_moc_track_spacing = dts

    @property
    def dancoff_moc_num_angles(self) -> int:
        return self._dancoff_moc_num_angles

    @dancoff_moc_num_angles.setter
    def dancoff_moc_num_angles(self, dna: int) -> None:
        if dna % 4 != 0:
            raise ValueError(
                "Number of angles for Dancoff correction calculation must be a multiple of 4."
            )
        if dna < 4:
            raise ValueError(
                "Number of angles for Dancoff correction calculation must be > 4."
            )
        self._dancoff_moc_num_angles = int(dna)

    @property
    def dancoff_flux_tolerance(self) -> float:
        return self._dancoff_flux_tolerance

    @dancoff_flux_tolerance.setter
    def dancoff_flux_tolerance(self, tol: float) -> None:
        try:
            tol = float(tol)
        except:
            raise TypeError(
                "Dancoff flux tolerance must be a floating point value in (0,1.E-2)."
            )

        if tol <= 0.0 or tol >= 1.0e-2:
            raise ValueError("Dancoff flux tolerance must be in (0,1.E-2).")

        self._dancoff_flux_tolerance = tol

    @property
    def moc_track_spacing(self) -> float:
        return self._moc_track_spacing

    @moc_track_spacing.setter
    def moc_track_spacing(self, dts: float) -> None:
        if dts <= 0.0 or dts > 0.1:
            raise ValueError("Track spacing must be in range (0, 0.1].")
        self._moc_track_spacing = dts

    @property
    def moc_num_angles(self) -> int:
        return self._moc_num_angles

    @moc_num_angles.setter
    def moc_num_angles(self, dna: int) -> None:
        if dna % 4 != 0:
            raise ValueError(
                "Number of angles for MOC calculation must be a multiple of 4."
            )
        if dna < 4:
            raise ValueError("Number of angles for MOC calculation must be > 4.")
        self._moc_num_angles = int(dna)

    @property
    def flux_tolerance(self) -> float:
        return self._flux_tolerance

    @flux_tolerance.setter
    def flux_tolerance(self, tol: float) -> None:
        try:
            tol = float(tol)
        except:
            raise TypeError(
                "Flux tolerance must be a floating point value in (0,1.E-2)."
            )

        if tol <= 0.0 or tol >= 1.0e-2:
            raise ValueError("Flux tolerance must be in (0,1.E-2).")

        self._flux_tolerance = tol

    @property
    def keff_tolerance(self) -> float:
        return self._keff_tolerance

    @keff_tolerance.setter
    def keff_tolerance(self, tol: float) -> None:
        try:
            tol = float(tol)
        except:
            raise TypeError(
                "Keff tolerance must be a floating point value in (0,1.E-2)."
            )

        if tol <= 0.0 or tol >= 1.0e-2:
            raise ValueError("Keff tolerance must be in (0,1.E-2).")

        self._keff_tolerance = tol

    @property
    def anisotropic(self) -> bool:
        return self._anisotropic

    @anisotropic.setter
    def anisotropic(self, aniso: bool) -> None:
        self._anisotropic = aniso

    @property
    def cmfd(self) -> bool:
        return self._cmfd

    @cmfd.setter
    def cmfd(self, cmfd: bool) -> None:
        self._cmfd = cmfd

    @property
    def cells(self) -> List[List[Union[FuelPin, GuideTube]]]:
        return self._cells

    @property
    def leakage_model(self) -> CriticalLeakage:
        return self._leakage_model

    @leakage_model.setter
    def leakage_model(self, clm: CriticalLeakage) -> None:
        self._leakage_model = clm

    @property
    def depletion_exposure_steps(self) -> Optional[np.ndarray]:
        return self._depletion_exposure_steps

    @depletion_exposure_steps.setter
    def depletion_exposure_steps(self, steps: Optional[np.ndarray]) -> None:
        if steps is None:
            self._depletion_exposure_steps = None
            self._depletion_time_steps = None
            return

        if not isinstance(steps, np.ndarray):
            raise TypeError("Depletion exposure steps must be a 1D Numpy array.")
        if steps.ndim != 1:
            raise ValueError("Depletion exposure steps must be a 1D Numpy array.")
        if steps.size == 0:
            raise ValueError(
                "Depletion exposure steps must have at least one entry. Use None for no depletion."
            )
        for step in steps:
            if step <= 0.0:
                raise ValueError("Depletion exposure steps must be > 0.")
            elif step > 5.0:
                raise ValueError("Depletion exposure steps should be <= 5 MWd/kg.")
        self._depletion_exposure_steps = steps

        self._depletion_time_steps = self._depletion_exposure_steps.copy()
        self._depletion_time_steps *= (
            1.0e3 * (1.0 / self.linear_power) * self.initial_heavy_metal_linear_mass
        )

    @property
    def depletion_time_steps(self) -> Optional[np.ndarray]:
        return self._depletion_time_steps

    @depletion_time_steps.setter
    def depletion_time_steps(self, steps: Optional[np.ndarray]) -> None:
        if steps is None:
            self._depletion_time_steps = None
            self._depletion_exposure_steps = None
            return

        if not isinstance(steps, np.ndarray):
            raise TypeError("Depletion time steps must be a 1D Numpy array.")
        if steps.ndim != 1:
            raise ValueError("Depletion time steps must be a 1D Numpy array.")
        if steps.size == 0:
            raise ValueError(
                "Depletion time steps must have at least one entry. Use None for no depletion."
            )
        for step in steps:
            if step <= 0.0:
                raise ValueError("Depletion time steps must be > 0.")
            elif step > 100:
                raise ValueError("Depletion time steps should be <= 100 days.")
        self._depletion_time_steps = steps

        self._depletion_exposure_steps = self._depletion_time_steps.copy()
        self._depletion_exposure_steps /= (
            1.0e3 * (1.0 / self.linear_power) * self.initial_heavy_metal_linear_mass
        )

    @property
    def exposures(self) -> np.ndarray:
        return self._exposures

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def corrector_transport(self) -> bool:
        return self._corrector_transport

    @corrector_transport.setter
    def corrector_transport(self, val: bool) -> None:
        self._corrector_transport = val

    @property
    def keff(self) -> Union[float, List[float]]:
        return self._keff

    @property
    def condensation_scheme(self) -> Optional[List[List[int]]]:
        return self._condensation_scheme

    @condensation_scheme.setter
    def condensation_scheme(self, cs: Optional[List[List[int]]]) -> None:
        if cs is None:
            self._condensation_scheme = None
            return

        if not isinstance(cs, list):
            raise TypeError("Condensation scheme must be a list of lists of ints.")

        if len(cs) == 0:
            raise TypeError("Condensation scheme must have at least one group.")

        for G in range(len(cs)):
            if not isinstance(cs[G], list):
                raise TypeError("Condensation scheme must be a list of lists of ints.")

            if len(cs[G]) != 2:
                raise TypeError(
                    "Each entry in condensation scheme must have 2 entries."
                )

            try:
                cs[G][0] = int(cs[G][0])
                cs[G][1] = int(cs[G][1])
            except:
                raise TypeError(
                    f"Microgroup indices for macrogroup {G} are not convertible to ints."
                )

            if cs[G][1] < cs[G][0]:
                raise ValueError(
                    f"The microgroup indices in macrogroup {G} are not ordered."
                )

            if G == 0 and cs[G][0] != 0:
                raise ValueError(
                    "The first microgroup index of the 0 macrogroup must be 0."
                )
            elif G == len(cs) - 1 and cs[G][1] != self._ndl.ngroups - 1:
                NG = self._ndl.ngroups - 1
                raise ValueError(
                    f"The last microgroup index of the last macrogroup must be {NG}."
                )

            if G > 0:
                if cs[G][0] != cs[G - 1][1] + 1:
                    raise ValueError("The condensation scheme is not continuous")

        self._condensation_scheme = copy.deepcopy(cs)

    @property
    def prefer_moc_adf_cdf(self) -> bool:
        return self._prefer_moc_adf_cdf

    @prefer_moc_adf_cdf.setter
    def prefer_moc_adf_cdf(self, value: bool) -> None:
        self._prefer_moc_adf_cdf = value

    @property
    def cmfd_condensation_scheme(self) -> Optional[List[List[int]]]:
        return self._cmfd_condensation_scheme

    @cmfd_condensation_scheme.setter
    def cmfd_condensation_scheme(self, cs: Optional[List[List[int]]]) -> None:
        if cs is None:
            self._cmfd_condensation_scheme = None
            return

        if not isinstance(cs, list):
            raise TypeError("CMFD Condensation scheme must be a list of lists of ints.")

        if len(cs) == 0:
            raise TypeError("CMFD Condensation scheme must have at least one group.")

        for G in range(len(cs)):
            if not isinstance(cs[G], list):
                raise TypeError(
                    "CMFD Condensation scheme must be a list of lists of ints."
                )

            if len(cs[G]) != 2:
                raise TypeError(
                    "Each entry in CMFD condensation scheme must have 2 entries."
                )

            try:
                cs[G][0] = int(cs[G][0])
                cs[G][1] = int(cs[G][1])
            except:
                raise TypeError(
                    f"Microgroup indices for macrogroup {G} are not convertible to ints."
                )

            if cs[G][1] < cs[G][0]:
                raise ValueError(
                    f"The microgroup indices in macrogroup {G} are not ordered."
                )

            if G == 0 and cs[G][0] != 0:
                raise ValueError(
                    "The first microgroup index of the 0 macrogroup must be 0."
                )
            elif G == len(cs) - 1 and cs[G][1] != self._ndl.ngroups - 1:
                NG = self._ndl.ngroups - 1
                raise ValueError(
                    f"The last microgroup index of the last macrogroup must be {NG}."
                )

            if G > 0:
                if cs[G][0] != cs[G - 1][1] + 1:
                    raise ValueError("The CMFD condensation scheme is not continuous")

        self._cmfd_condensation_scheme = copy.deepcopy(cs)

    @property
    def diffusion_data(self) -> Optional[Union[DiffusionData, List[DiffusionData]]]:
        return self._diffusion_data

    def _set_cells(self, cells: List[List[Union[FuelPin, GuideTube]]]) -> None:
        if len(cells) != self._simulated_shape[1]:
            raise ValueError(
                "Shape along y of cells list does not agree with assembly shape and symmetry."
            )
        for j in range(len(cells)):
            if len(cells[j]) != self._simulated_shape[0]:
                raise ValueError(
                    "Shape along x of cells list does not agree with assembly shape and symmetry."
                )

        # Make deep copies of all cells provided by the user. This makes sure
        # that each duplicate FuelPin or GuideTube instance is unique.
        self._cells = []
        self._cells_set = False
        self._initial_heavy_metal_linear_mass = 0.0
        for j in range(len(cells)):
            self._cells.append([])
            for i in range(len(cells[j])):
                self._cells[-1].append(copy.deepcopy(cells[j][i]))

                if isinstance(cells[j][i], FuelPin):
                    lfm = cells[j][i].initial_fissionable_linear_mass

                    # First check for quarter symmetry and being corner pin
                    if (
                        self.symmetry == Symmetry.Quarter
                        and self.shape[0] % 2 == 1
                        and self.shape[1] % 2 == 1
                        and j == self._simulated_shape[1] - 1
                        and i == 0
                    ):
                        lfm *= 0.25
                    # Next, check for being on the side with a half pin in quarter symmetry
                    elif (
                        self.symmetry == Symmetry.Quarter
                        and self.shape[0] % 2 == 1
                        and i == 0
                    ):
                        lfm *= 0.5
                    # Next, check for being on the bottom row with a half pin
                    elif (
                        self.symmetry != Symmetry.Full
                        and self.shape[1] % 2 == 1
                        and j == self._simulated_shape[1] - 1
                    ):
                        lfm *= 0.5

                    self._initial_heavy_metal_linear_mass += lfm

        self._cells_set = True

        if self.symmetry == Symmetry.Half:
            self._initial_heavy_metal_linear_mass *= 2.0
        elif self.symmetry == Symmetry.Quarter:
            self._initial_heavy_metal_linear_mass *= 4.0

        # Convert HM mass from g to kg
        self._initial_heavy_metal_linear_mass *= 1.0e-3

    def _compute_fuel_volume_fraction(self) -> None:
        """
        Computes the fraction of the assembly which is fuel.
        """
        sum_volume = 0.0
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]

                if not isinstance(cell, FuelPin):
                    continue

                pin_volume = np.pi * cell.fuel_radius * cell.fuel_radius

                # First check for quarter symmetry and being corner pin
                if (
                    self.symmetry == Symmetry.Quarter
                    and self.shape[0] % 2 == 1
                    and self.shape[1] % 2 == 1
                    and j == self._simulated_shape[1] - 1
                    and i == 0
                ):
                    pin_volume *= 0.25
                # Next, check for being on the side with a half pin in quarter symmetry
                elif (
                    self.symmetry == Symmetry.Quarter
                    and self.shape[0] % 2 == 1
                    and i == 0
                ):
                    pin_volume *= 0.5
                # Next, check for being on the bottom row with a half pin
                elif (
                    self.symmetry != Symmetry.Full
                    and self.shape[1] % 2 == 1
                    and j == self._simulated_shape[1] - 1
                ):
                    pin_volume *= 0.5

                sum_volume += pin_volume

        if self.symmetry == Symmetry.Quarter:
            sum_volume *= 4.0
        elif self.symmetry == Symmetry.Half:
            sum_volume *= 2.0

        self._fuel_volume_fraction = sum_volume / (
            self.assembly_pitch * self.assembly_pitch
        )

    def _compute_moderator_volume_fraction(self) -> None:
        """
        Computes the fraction of the assembly which is moderator.
        """
        sum_volume = 0.0

        pin_cell_volume = self.pitch**2.0
        if self.spacer_grid_width is not None:
            pin_cell_volume = (self.pitch - 2.0 * self.spacer_grid_width) ** 2.0

        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]

                # Start with volume of moderator in a pin cell
                cell_mod_volume = pin_cell_volume

                if isinstance(cell, FuelPin):
                    # Remove volume of pin
                    cell_mod_volume -= np.pi * cell.clad_radius * cell.clad_radius
                else:
                    # Guide tube

                    # First, remove volume of the tube
                    cell_mod_volume -= np.pi * (
                        cell.outer_radius * cell.outer_radius
                        - cell.inner_radius * cell.inner_radius
                    )

                    # Remove volume of the burnable poison rod (if present)
                    if isinstance(cell.fill, BurnablePoisonRod):
                        cell_mod_volume -= (
                            np.pi
                            * cell.fill.outer_clad_radius
                            * cell.fill.outer_clad_radius
                        )

                        # Add center volume back if the center is moderator !
                        if cell.fill.center is not None:
                            cell_mod_volume += (
                                np.pi
                                * cell.fill.center_radius
                                * cell.fill.center_radius
                            )

                # First check for quarter symmetry and being corner pin
                if (
                    self.symmetry == Symmetry.Quarter
                    and self.shape[0] % 2 == 1
                    and self.shape[1] % 2 == 1
                    and j == self._simulated_shape[1] - 1
                    and i == 0
                ):
                    cell_mod_volume *= 0.25
                # Next, check for being on the side with a half pin in quarter symmetry
                elif (
                    self.symmetry == Symmetry.Quarter
                    and self.shape[0] % 2 == 1
                    and i == 0
                ):
                    cell_mod_volume *= 0.5
                # Next, check for being on the bottom row with a half pin
                elif (
                    self.symmetry != Symmetry.Full
                    and self.shape[1] % 2 == 1
                    and j == self._simulated_shape[1] - 1
                ):
                    cell_mod_volume *= 0.5

                sum_volume += cell_mod_volume

        if self.symmetry == Symmetry.Quarter:
            sum_volume *= 4.0
        elif self.symmetry == Symmetry.Half:
            sum_volume *= 2.0

        # Add volume for the gap around the assembly
        if self.grid_sleeve_width is None:
            sum_volume += self.assembly_pitch**2.0 - (self.shape[0] * self.pitch) ** 2.0
        else:
            sum_volume += (
                self.assembly_pitch**2.0
                - (self.shape[0] * self.pitch + 2.0 * self.grid_sleeve_width) ** 2.0
            )

        self._moderator_volume_fraction = sum_volume / (
            self.assembly_pitch * self.assembly_pitch
        )

    # ==========================================================================
    # Interrogation Methods

    def get_average_fuel_nuclide_density(self, t: int, nuclide: str) -> float:
        """
        Computes the average density of a nuclide within all the fuel pellets
        in the assembly at a single depletion time step.

        Parameters
        ----------
        t : int
            Depletion time step index.
        nuclide : str
            Name of the nuclide.

        Returns
        -------
        float
            Average density of the nuclide at depletion time step t across all
            the fuel pellets in units of atoms per barn-cm.
        """
        sum_density = 0.0
        sum_volume = 0.0
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]

                if not isinstance(cell, FuelPin):
                    continue

                pin_avg_density = cell.get_average_fuel_nuclide_density(t, nuclide)
                pin_volume = np.pi * cell.fuel_radius * cell.fuel_radius

                # First check for quarter symmetry and being corner pin
                if (
                    self.symmetry == Symmetry.Quarter
                    and self.shape[0] % 2 == 1
                    and self.shape[1] % 2 == 1
                    and j == self._simulated_shape[1] - 1
                    and i == 0
                ):
                    pin_volume *= 0.25
                # Next, check for being on the side with a half pin in quarter symmetry
                elif (
                    self.symmetry == Symmetry.Quarter
                    and self.shape[0] % 2 == 1
                    and i == 0
                ):
                    pin_volume *= 0.5
                # Next, check for being on the bottom row with a half pin
                elif (
                    self.symmetry != Symmetry.Full
                    and self.shape[1] % 2 == 1
                    and j == self._simulated_shape[1] - 1
                ):
                    pin_volume *= 0.5

                sum_density += pin_avg_density
                sum_volume += pin_volume

        return sum_density / sum_volume

    # ==========================================================================
    # Dancoff Correction Related Methods

    def _init_dancoff_components(self) -> None:
        scarabee_log(
            LogLevel.Info, "Initializing Dancoff correction calculation components."
        )
        set_logging_level(LogLevel.Warning)
        self._init_isolated_dancoff_components()
        self._init_full_dancoff_components()
        self._save_dancoff_fsr_indexes()
        set_logging_level(LogLevel.Info)

    def _init_isolated_dancoff_components(self) -> None:
        # Isolated pitch
        iso_pitch = self._dancoff_isolation_scale * self.pitch

        for j in range(len(self.cells)):
            self._isolated_dancoff_cells.append([])
            self._isolated_dancoff_mocs.append([])
            for i in range(len(self.cells[j])):
                cell = None
                geom = None
                moc = None

                x_min_bc = BoundaryCondition.Vacuum
                x_max_bc = BoundaryCondition.Vacuum
                y_min_bc = BoundaryCondition.Vacuum
                y_max_bc = BoundaryCondition.Vacuum

                # First check for quarter symmetry and being corner pin
                if (
                    self.symmetry == Symmetry.Quarter
                    and self.shape[0] % 2 == 1
                    and self.shape[1] % 2 == 1
                    and j == self._simulated_shape[1] - 1
                    and i == 0
                ):
                    cell = self.cells[j][i].make_dancoff_moc_cell(
                        self._moderator_dancoff_xs,
                        0.5 * iso_pitch,
                        0.5 * iso_pitch,
                        PinCellType.I,
                        True,
                    )
                    geom = Cartesian2D([0.5 * iso_pitch], [0.5 * iso_pitch])
                    geom.set_tiles([cell])
                    x_min_bc = BoundaryCondition.Reflective
                    y_min_bc = BoundaryCondition.Reflective
                # Next, check for being on the side with a half pin in quarter symmetry
                elif (
                    self.symmetry == Symmetry.Quarter
                    and self.shape[0] % 2 == 1
                    and i == 0
                ):
                    cell = self.cells[j][i].make_dancoff_moc_cell(
                        self._moderator_dancoff_xs,
                        0.5 * iso_pitch,
                        iso_pitch,
                        PinCellType.XP,
                        True,
                    )
                    geom = Cartesian2D([0.5 * iso_pitch], [iso_pitch])
                    geom.set_tiles([cell])
                    x_min_bc = BoundaryCondition.Reflective
                # Next, check for being on the bottom row with a half pin
                elif (
                    self.symmetry != Symmetry.Full
                    and self.shape[1] % 2 == 1
                    and j == self._simulated_shape[1] - 1
                ):
                    cell = self.cells[j][i].make_dancoff_moc_cell(
                        self._moderator_dancoff_xs,
                        iso_pitch,
                        0.5 * iso_pitch,
                        PinCellType.YP,
                        True,
                    )
                    geom = Cartesian2D([iso_pitch], [0.5 * iso_pitch])
                    geom.set_tiles([cell])
                    y_min_bc = BoundaryCondition.Reflective
                # Otherwise, we just make the full cell
                else:
                    cell = self.cells[j][i].make_dancoff_moc_cell(
                        self._moderator_dancoff_xs,
                        iso_pitch,
                        iso_pitch,
                        PinCellType.Full,
                        True,
                    )
                    geom = Cartesian2D([iso_pitch], [iso_pitch])
                    geom.set_tiles([cell])

                # Save to cells
                self._isolated_dancoff_cells[-1].append(cell)

                # Make the MOCDriver
                moc = MOCDriver(geom)
                moc.sim_mode = SimulationMode.FixedSource
                moc.x_min_bc = x_min_bc
                moc.x_max_bc = x_max_bc
                moc.y_min_bc = y_min_bc
                moc.y_max_bc = y_max_bc

                # Generate tracks in serial as each call will run with threads
                moc.generate_tracks(
                    self._dancoff_moc_num_angles,
                    self._dancoff_moc_track_spacing,
                    YamamotoTabuchi6(),
                )

                # Save the MOC
                self._isolated_dancoff_mocs[-1].append(moc)

    def _init_full_dancoff_components(self) -> None:
        pitch = self.pitch
        if self.spacer_grid_width is not None:
            pitch -= 2.0 * self.spacer_grid_width

        # Get all the cells
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                # Get the geometry bit from the cell
                cell = None

                # First check for quarter symmetry and being corner pin
                if (
                    self.symmetry == Symmetry.Quarter
                    and self.shape[0] % 2 == 1
                    and self.shape[1] % 2 == 1
                    and j == self._simulated_shape[1] - 1
                    and i == 0
                ):
                    cell = self.cells[j][i].make_dancoff_moc_cell(
                        self._moderator_dancoff_xs,
                        0.5 * pitch,
                        0.5 * pitch,
                        PinCellType.I,
                        False,
                    )
                    if self.spacer_grid_width is not None:
                        cell, _ = _ensleeve_quarter(
                            cell,
                            pitch,
                            self.spacer_grid_width,
                            self._spacer_grid_dancoff_xs,
                        )
                # Next, check for being on the side with a half pin in quarter symmetry
                elif (
                    self.symmetry == Symmetry.Quarter
                    and self.shape[0] % 2 == 1
                    and i == 0
                ):
                    cell = self.cells[j][i].make_dancoff_moc_cell(
                        self._moderator_dancoff_xs,
                        0.5 * pitch,
                        pitch,
                        PinCellType.XP,
                        False,
                    )
                    if self.spacer_grid_width is not None:
                        cell, _ = _ensleeve_half_right(
                            cell,
                            pitch,
                            self.spacer_grid_width,
                            self._spacer_grid_dancoff_xs,
                        )
                # Next, check for being on the bottom row with a half pin
                elif (
                    self.symmetry != Symmetry.Full
                    and self.shape[1] % 2 == 1
                    and j == self._simulated_shape[1] - 1
                ):
                    cell = self.cells[j][i].make_dancoff_moc_cell(
                        self._moderator_dancoff_xs,
                        pitch,
                        0.5 * pitch,
                        PinCellType.YP,
                        False,
                    )
                    if self.spacer_grid_width is not None:
                        cell, _ = _ensleeve_half_top(
                            cell,
                            pitch,
                            self.spacer_grid_width,
                            self._spacer_grid_dancoff_xs,
                        )
                # Otherwise, we just make the full cell
                else:
                    cell = self.cells[j][i].make_dancoff_moc_cell(
                        self._moderator_dancoff_xs,
                        pitch,
                        pitch,
                        PinCellType.Full,
                        False,
                    )
                    if self.spacer_grid_width is not None:
                        cell, _ = _ensleeve_full(
                            cell,
                            pitch,
                            self.spacer_grid_width,
                            self._spacer_grid_dancoff_xs,
                        )

                # Save to cells
                self._full_dancoff_cells.append(cell)

        # Construct the Cartesian2D geometry
        dx = self._simulated_shape[0] * [self.pitch]
        dy = self._simulated_shape[1] * [self.pitch]
        if self.symmetry != Symmetry.Full and self.shape[1] % 2 == 1:
            dy[0] *= 0.5
        if self.symmetry == Symmetry.Quarter and self.shape[0] % 2 == 1:
            dx[0] *= 0.5
        self._full_dancoff_geom = Cartesian2D(dx, dy)
        self._full_dancoff_geom.set_tiles(self._full_dancoff_cells)

        # Add gap if needed
        gap_width = 0.5 * (self.assembly_pitch - self.shape[0] * self.pitch)
        if self.grid_sleeve_width is not None:
            gap_width -= self.grid_sleeve_width
        if gap_width > 0.0:
            pins_geom = self._full_dancoff_geom

            if self.symmetry == Symmetry.Quarter:
                if self.grid_sleeve_width is not None:
                    pins_geom, _ = _ensleeve_quarter(
                        pins_geom,
                        self.pitch,
                        self._grid_sleeve_width,
                        self._grid_sleeve_dancoff_xs,
                    )
                self._full_dancoff_geom, _ = _ensleeve_quarter(
                    pins_geom, self.pitch, gap_width, self._moderator_dancoff_xs
                )
            elif self.symmetry == Symmetry.Half:
                if self.grid_sleeve_width is not None:
                    pins_geom, _ = _ensleeve_half_top(
                        pins_geom,
                        self.pitch,
                        self._grid_sleeve_width,
                        self._grid_sleeve_dancoff_xs,
                    )
                self._full_dancoff_geom, _ = _ensleeve_half_top(
                    pins_geom, self.pitch, gap_width, self._moderator_dancoff_xs
                )
            else:
                if self.grid_sleeve_width is not None:
                    pins_geom, _ = _ensleeve_full(
                        pins_geom,
                        self.pitch,
                        self._grid_sleeve_width,
                        self._grid_sleeve_dancoff_xs,
                    )
                self._full_dancoff_geom, _ = _ensleeve_full(
                    pins_geom, self.pitch, gap_width, self._moderator_dancoff_xs
                )

        # Construct the MOC
        self._full_dancoff_moc = MOCDriver(self._full_dancoff_geom)
        self._full_dancoff_moc.sim_mode = SimulationMode.FixedSource
        self._full_dancoff_moc.x_min_bc = self._x_min_bc
        self._full_dancoff_moc.x_max_bc = self._x_max_bc
        self._full_dancoff_moc.y_min_bc = self._y_min_bc
        self._full_dancoff_moc.y_max_bc = self._y_max_bc
        self._full_dancoff_moc.generate_tracks(
            self._dancoff_moc_num_angles,
            self._dancoff_moc_track_spacing,
            YamamotoTabuchi6(),
        )

    def _save_dancoff_fsr_indexes(self) -> None:
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]
                isomoc = self._isolated_dancoff_mocs[j][i]
                cell.populate_dancoff_fsr_indexes(isomoc, self._full_dancoff_moc)

    def _dancoff_components_initialized(self) -> bool:
        if self._full_dancoff_moc is None:
            return False
        return True

    def set_dancoff_moderator_xs(self) -> None:
        """
        Updates the moderator cross section for all Dancoff correction calculations.
        """
        self._moderator_dancoff_xs.set(
            CrossSection(
                np.array([self.moderator.potential_xs]),
                np.array([self.moderator.potential_xs]),
                np.array([[0.0]]),
                "Moderator",
            )
        )

    def set_dancoff_spacer_grid_sleeve_xs(self) -> None:
        """
        Updates the spacer grid and grid sleeve cross sections for all Dancoff
        correction calculations.
        """
        if self.spacer_grid is not None:
            self._spacer_grid_dancoff_xs.set(
                CrossSection(
                    np.array([self.spacer_grid.potential_xs]),
                    np.array([self.spacer_grid.potential_xs]),
                    np.array([[0.0]]),
                    "Spacer Grid",
                )
            )
        if self.grid_sleeve is not None:
            self._grid_sleeve_dancoff_xs.set(
                CrossSection(
                    np.array([self.grid_sleeve.potential_xs]),
                    np.array([self.grid_sleeve.potential_xs]),
                    np.array([[0.0]]),
                    "Grid Sleeve",
                )
            )

    def compute_fuel_dancoff_corrections(self) -> None:
        """
        Recomputes all Dancoff corrections for the fuel regions in the problem,
        using the most recent material definitions. All fuel is shelf-shielded
        together, regardless of wether or not is is UO2 or MOX.
        """
        scarabee_log(LogLevel.Info, "Computing Dancoff corrections for the fuel.")
        set_logging_level(LogLevel.Warning)
        if not self._dancoff_components_initialized():
            raise RuntimeError(
                "Dancoff calculation components have not been initialized."
            )

        # Set the xs and sources for all cells
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]
                isomoc = self._isolated_dancoff_mocs[j][i]

                isomoc.flux_tolerance = self.dancoff_flux_tolerance

                cell.set_xs_for_fuel_dancoff_calculation()

                cell.set_isolated_dancoff_fuel_sources(isomoc, self.moderator)

                cell.set_full_dancoff_fuel_sources(
                    self._full_dancoff_moc, self.moderator
                )
        self._full_dancoff_moc.flux_tolerance = self.dancoff_flux_tolerance

        # Solve all the MOCs in parallel
        threads = []
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]
                if isinstance(cell, FuelPin):
                    isomoc = self._isolated_dancoff_mocs[j][i]
                    threads.append(Thread(target=isomoc.solve))
                    threads[-1].start()
        threads.append(Thread(target=self._full_dancoff_moc.solve))
        threads[-1].start()
        for t in threads:
            t.join()

        # Go through and let each cell compute the Dancoff correction if it holds
        # a fuel pin.
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]
                isomoc = self._isolated_dancoff_mocs[j][i]

                if isinstance(cell, FuelPin):
                    C = cell.compute_fuel_dancoff_correction(
                        isomoc, self._full_dancoff_moc
                    )
                    self._fuel_dancoff_corrections[j, i] = C
        set_logging_level(LogLevel.Info)

    def compute_clad_dancoff_corrections(self) -> None:
        """
        Recomputes all Dancoff corrections for the fuel pin cladding regions
        in the problem, using the most recent material definitions. All
        cladding is shelf-shielded together.
        """
        scarabee_log(LogLevel.Info, "Computing Dancoff corrections for the cladding.")
        set_logging_level(LogLevel.Warning)
        if self._full_dancoff_moc is None:
            raise RuntimeError(
                "Dancoff calculation components have not been initialized."
            )

        # Set the xs and sources for all cells
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]
                isomoc = self._isolated_dancoff_mocs[j][i]
                isomoc.flux_tolerance = self.dancoff_flux_tolerance

                cell.set_xs_for_clad_dancoff_calculation(self._ndl)

                cell.set_isolated_dancoff_clad_sources(
                    isomoc, self.moderator, self._ndl
                )

                cell.set_full_dancoff_clad_sources(
                    self._full_dancoff_moc, self.moderator, self._ndl
                )
            self._full_dancoff_moc.flux_tolerance = self.dancoff_flux_tolerance

        # Solve all the MOCs in parallel
        threads = []
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]
                isomoc = self._isolated_dancoff_mocs[j][i]
                threads.append(Thread(target=isomoc.solve))
                threads[-1].start()
        threads.append(Thread(target=self._full_dancoff_moc.solve))
        threads[-1].start()
        for t in threads:
            t.join()

        # Go through and let each cell compute the Dancoff correction if it holds
        # a fuel pin.
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]
                isomoc = self._isolated_dancoff_mocs[j][i]

                C = cell.compute_clad_dancoff_correction(isomoc, self._full_dancoff_moc)
                self._clad_dancoff_corrections[j, i] = C
        set_logging_level(LogLevel.Info)

    def apply_dancoff_corrections(self) -> None:
        """
        Appends all fuel and cladding Dancoff corrections to the appropriate cell.
        """
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]
                if isinstance(cell, FuelPin):
                    cell.append_fuel_dancoff_correction(
                        self._fuel_dancoff_corrections[j, i]
                    )
                cell.append_clad_dancoff_correction(
                    self._clad_dancoff_corrections[j, i]
                )

    def self_shield_and_xs_update(self) -> None:
        """
        Computes a new set of Dancoff corrections for the fuel and the
        cladding.  After, these are applied to all the cells in the problem.
        """
        if not self._dancoff_components_initialized():
            self._init_dancoff_components()

        # Update the Dancoff cross sections held by the assembly
        self.set_dancoff_moderator_xs()
        self.set_dancoff_spacer_grid_sleeve_xs()

        # Compute Dancoff corrections
        self.compute_fuel_dancoff_corrections()
        self.compute_clad_dancoff_corrections()
        self.apply_dancoff_corrections()

    # ==========================================================================
    # Transport Calculation Related Methods

    def _init_moc(self) -> None:
        pitch = self.pitch
        if self.spacer_grid_width is not None:
            pitch -= 2.0 * self.spacer_grid_width

        # Get all the cells
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                # Get the geometry bit from the cell
                cell = None

                # First check for quarter symmetry and being corner pin
                if (
                    self.symmetry == Symmetry.Quarter
                    and self.shape[0] % 2 == 1
                    and self.shape[1] % 2 == 1
                    and j == self._simulated_shape[1] - 1
                    and i == 0
                ):
                    cell = self.cells[j][i].make_moc_cell(
                        self._moderator_xs,
                        0.5 * pitch,
                        0.5 * pitch,
                        PinCellType.I,
                    )
                    if self.spacer_grid_width is not None:
                        cell, _ = _ensleeve_quarter(
                            cell, pitch, self.spacer_grid_width, self._spacer_grid_xs
                        )
                # Next, check for being on the side with a half pin in quarter symmetry
                elif (
                    self.symmetry == Symmetry.Quarter
                    and self.shape[0] % 2 == 1
                    and i == 0
                ):
                    cell = self.cells[j][i].make_moc_cell(
                        self._moderator_xs, 0.5 * pitch, pitch, PinCellType.XP
                    )
                    if self.spacer_grid_width is not None:
                        cell, _ = _ensleeve_half_right(
                            cell, pitch, self.spacer_grid_width, self._spacer_grid_xs
                        )
                # Next, check for being on the bottom row with a half pin
                elif (
                    self.symmetry != Symmetry.Full
                    and self.shape[1] % 2 == 1
                    and j == self._simulated_shape[1] - 1
                ):
                    cell = self.cells[j][i].make_moc_cell(
                        self._moderator_xs, pitch, 0.5 * pitch, PinCellType.YP
                    )
                    if self.spacer_grid_width is not None:
                        cell, _ = _ensleeve_half_top(
                            cell, pitch, self.spacer_grid_width, self._spacer_grid_xs
                        )
                # Otherwise, we just make the full cell
                else:
                    cell = self.cells[j][i].make_moc_cell(
                        self._moderator_xs, pitch, pitch, PinCellType.Full
                    )
                    if self.spacer_grid_width is not None:
                        cell, _ = _ensleeve_full(
                            cell, pitch, self.spacer_grid_width, self._spacer_grid_xs
                        )

                # Save to cells
                self._asmbly_cells.append(cell)

        # Construct the Cartesian2D geometry
        dx = self._simulated_shape[0] * [self.pitch]
        dy = self._simulated_shape[1] * [self.pitch]
        if self.symmetry != Symmetry.Full and self.shape[1] % 2 == 1:
            dy[0] *= 0.5
        if self.symmetry == Symmetry.Quarter and self.shape[0] % 2 == 1:
            dx[0] *= 0.5
        self._asmbly_geom = Cartesian2D(dx, dy)
        self._asmbly_geom.set_tiles(self._asmbly_cells)

        # Add gap if needed
        gap_width = 0.5 * (self.assembly_pitch - self.shape[0] * self.pitch)
        if self.grid_sleeve_width is not None:
            gap_width -= self.grid_sleeve_width
        if gap_width > 0.0:
            pins_geom = self._asmbly_geom

            if self.symmetry == Symmetry.Quarter:
                if self.grid_sleeve_width is not None:
                    pins_geom, _ = _ensleeve_quarter(
                        pins_geom,
                        self.pitch,
                        self._grid_sleeve_width,
                        self._grid_sleeve_xs,
                    )
                self._asmbly_geom, _ = _ensleeve_quarter(
                    pins_geom, self.pitch, gap_width, self._moderator_xs
                )
            elif self.symmetry == Symmetry.Half:
                if self.grid_sleeve_width is not None:
                    pins_geom, _ = _ensleeve_half_top(
                        pins_geom,
                        self.pitch,
                        self._grid_sleeve_width,
                        self._grid_sleeve_xs,
                    )
                self._asmbly_geom, _ = _ensleeve_half_top(
                    pins_geom, self.pitch, gap_width, self._moderator_xs
                )
            else:
                if self.grid_sleeve_width is not None:
                    pins_geom, _ = _ensleeve_full(
                        pins_geom,
                        self.pitch,
                        self._grid_sleeve_width,
                        self._grid_sleeve_xs,
                    )
                self._asmbly_geom, _ = _ensleeve_full(
                    pins_geom, self.pitch, gap_width, self._moderator_xs
                )

        # Construct the MOC
        self._asmbly_moc = MOCDriver(self._asmbly_geom, anisotropic=self._anisotropic)
        self._asmbly_moc.x_min_bc = self._x_min_bc
        self._asmbly_moc.x_max_bc = self._x_max_bc
        self._asmbly_moc.y_min_bc = self._y_min_bc
        self._asmbly_moc.y_max_bc = self._y_max_bc

        # Apply CMFD if turned on
        if self.cmfd and self.cmfd_condensation_scheme is None:
            raise RuntimeError(
                "CMFD is enabled but no CMFD condensation scheme has been provided."
            )
        elif self.cmfd:
            extra_width = 0.5 * (self.assembly_pitch - self.shape[0] * self.pitch)
            dx_cmfd = self._simulated_shape[0] * [self.pitch]
            dy_cmfd = self._simulated_shape[1] * [self.pitch]
            if self.symmetry != Symmetry.Full and self.shape[1] % 2 == 1:
                dy_cmfd[0] *= 0.5
            if self.symmetry == Symmetry.Quarter and self.shape[0] % 2 == 1:
                dx_cmfd[0] *= 0.5

            if self.symmetry == Symmetry.Full:
                dx_cmfd[0] += extra_width
                dx_cmfd[-1] += extra_width
                dy_cmfd[0] += extra_width
                dy_cmfd[-1] += extra_width
            elif self.symmetry == Symmetry.Half:
                dx_cmfd[0] += extra_width
                dx_cmfd[-1] += extra_width
                dy_cmfd[-1] += extra_width
            else:
                dx_cmfd[-1] += extra_width
                dy_cmfd[-1] += extra_width

            self._asmbly_moc.cmfd = CMFD(
                dx_cmfd, dy_cmfd, self.cmfd_condensation_scheme
            )

        # Trace tracks
        self._asmbly_moc.generate_tracks(
            self._moc_num_angles,
            self._moc_track_spacing,
            YamamotoTabuchi6(),
        )

        self._save_fsr_indexes()

    def _save_fsr_indexes(self) -> None:
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]
                cell.populate_fsr_indexes(self._asmbly_moc)

    def plot(self) -> None:
        """
        Launches the graphical geometry plotter for the assembly calculation.
        """
        if self._asmbly_moc is None:
            raise RuntimeError(
                "Cannot launch plotter. MOCDriver has not been initialized."
            )
        self._asmbly_moc.plot()

    def set_moderator_xs(self) -> None:
        """
        Updates the moderator cross section for transport calculations.
        """
        self._moderator_xs.set(
            self.moderator.dilution_xs(self.moderator.size * [1.0e10], self._ndl)
        )

    def set_spacer_grid_sleeve_xs(self) -> None:
        """
        Updates the spacer grid and grid sleeve cross sections for transport
        calculations.
        """
        if self.spacer_grid is not None:
            self._spacer_grid_xs.set(
                self.spacer_grid.dilution_xs(
                    self.spacer_grid.size * [1.0e10], self._ndl
                )
            )
        if self.grid_sleeve is not None:
            self._grid_sleeve_xs.set(
                self.grid_sleeve.dilution_xs(
                    self.grid_sleeve.size * [1.0e10], self._ndl
                )
            )

    def recompute_all_xs(self) -> None:
        """
        Computes and applies all cross sections using the most recent material
        information and Dancoff corrections.
        """
        self.recompute_all_fuel_xs()
        self.recompute_all_gap_xs()
        self.recompute_all_clad_xs()

        self.recompute_all_guide_tube_fill_xs()

        self.set_moderator_xs()
        self.set_spacer_grid_sleeve_xs()

    def recompute_all_self_shielded_xs(self) -> None:
        """
        Computes and applies all fuel and caldding cross sections using the
        most recent material information and Dancoff corrections.
        """
        self.recompute_all_fuel_xs()
        self.recompute_all_clad_xs()

    def recompute_all_fuel_xs(self) -> None:
        """
        Computes and applies all fuel cross sections using the most recent
        material information and Dancoff corrections.
        """
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]
                if isinstance(cell, FuelPin):
                    cell.set_fuel_xs_for_depletion_step(-1, self._ndl)

    def recompute_all_clad_xs(self) -> None:
        """
        Computes and applies all cladding cross sections using the most recent
        material information and Dancoff corrections.
        """
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]
                cell.set_clad_xs_for_depletion_step(-1, self._ndl)

    def recompute_all_gap_xs(self) -> None:
        """
        Computes and applies all gap cross sections using the most recent
        material information.
        """
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]
                if isinstance(cell, FuelPin):
                    cell.set_gap_xs(self._ndl)

    def recompute_all_guide_tube_fill_xs(self) -> None:
        """
        Computes and applies all cross sections for the fill objects of guide
        tubes. These could be for burnable poison rods or control rods.
        """
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]
                if isinstance(cell, GuideTube):
                    cell.set_fill_xs_for_depletion_step(-1, self._ndl)

    def apply_leakage_model(self) -> None:
        """
        Applied the critical leakage model to the assembly, modifying the flux
        in the MOC simulation directly.
        """
        # If no leakage, just return
        if self.leakage_model == CriticalLeakage.NoLeakage:
            return

        scarabee_log(LogLevel.Info, "")

        self._infinite_flux_spectrum = self._asmbly_moc.homogenize_flux_spectrum()

        homogenized_moc = self._asmbly_moc.homogenize()

        if self.leakage_model == CriticalLeakage.P1:
            scarabee_log(
                LogLevel.Info, "Performing P1 criticality spectrum calculation"
            )
            critical_spectrum = P1CriticalitySpectrum(homogenized_moc)
        else:
            scarabee_log(
                LogLevel.Info, "Performing B1 criticality spectrum calculation"
            )
            critical_spectrum = B1CriticalitySpectrum(homogenized_moc)

        self._asmbly_moc.apply_criticality_spectrum(critical_spectrum.flux)

        scarabee_log(LogLevel.Info, "Kinf    : {:.5f}".format(critical_spectrum.k_inf))
        scarabee_log(
            LogLevel.Info, "Buckling: {:.5f}".format(critical_spectrum.buckling)
        )

    def apply_infinite_spectrum(self) -> None:
        """
        Undoes the critical flux spectrum adjustment to the MOCDriver.
        This permits subsequent transport calcualtions to converge much faster.
        """
        if self._infinite_flux_spectrum is not None and self._asmbly_moc is not None:
            self._asmbly_moc.apply_criticality_spectrum(self._infinite_flux_spectrum)

    def obtain_flux_spectra(self) -> None:
        """
        Computes the average flux spectrum for material regions whcih are
        depleted. This includes each fuel ring in fuel pins and the poison in
        burnable poison rods.
        """
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                self.cells[j][i].obtain_flux_spectra(self._asmbly_moc)

    def normalize_flux_to_power(self) -> None:
        """
        Normalizes the flux spectra based on the specified linear power density
        for the assembly. It assumes that all power comes from fission.
        """
        assembly_linear_power = 0.0
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]

                if not isinstance(cell, FuelPin):
                    continue

                pin_linear_power = cell.compute_pin_linear_power(self._ndl)

                # First check for quarter symmetry and being corner pin
                if (
                    self.symmetry == Symmetry.Quarter
                    and self.shape[0] % 2 == 1
                    and self.shape[1] % 2 == 1
                    and j == self._simulated_shape[1] - 1
                    and i == 0
                ):
                    pin_linear_power *= 0.25
                # Next, check for being on the side with a half pin in quarter symmetry
                elif (
                    self.symmetry == Symmetry.Quarter
                    and self.shape[0] % 2 == 1
                    and i == 0
                ):
                    pin_linear_power *= 0.5
                # Next, check for being on the bottom row with a half pin
                elif (
                    self.symmetry != Symmetry.Full
                    and self.shape[1] % 2 == 1
                    and j == self._simulated_shape[1] - 1
                ):
                    pin_linear_power *= 0.5

                assembly_linear_power += pin_linear_power

        if self.symmetry == Symmetry.Half:
            assembly_linear_power *= 2.0
        elif self.symmetry == Symmetry.Quarter:
            assembly_linear_power *= 4.0

        # Compute normalization factor.
        # Multiply by 10^3 to go from kW to W on linear power
        f = 1.0e3 * self.linear_power / assembly_linear_power

        # Normalize flux spectra
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i].normalize_flux_spectrum(f)

    def _compute_few_group_flux(self, r: Vector, u: Direction) -> List[float]:
        """
        Computes the flux in the few-group scheme at the given position
        and direction.

        Parameters
        ----------
        r : Vector
            Position where the flux is evaluated.
        u : Direction
            Direction vector to disambiguate the position.

        Returns
        -------
        list of float
            Values of the few-group flux at the given position.

        Raises
        ------
        RuntimeError
            If the condensation_scheme attribute is not set.
        """
        if self.condensation_scheme is None:
            raise RuntimeError("Energy condensation scheme not set.")

        flux = [0.0 for G in range(len(self.condensation_scheme))]

        for G in range(len(self.condensation_scheme)):
            gmin, gmax = self.condensation_scheme[G][:]

            for g in range(gmin, gmax + 1):
                flux[G] += self._asmbly_moc.flux(r, u, g)

        return flux

    def _compute_average_line_flux(
        self, segments: List[Tuple[int, float]]
    ) -> List[float]:
        """
        Computes the average flux along a set of line segments in the few-group
        structure.

        Paramters
        ---------
        segments : list of tuples of int and float
            List of flat source region index and segment length tuples.

        Returns
        -------
        list of float
            Values of the few-group flux alone the line.

        Raises
        ------
        RuntimeError
            If the condensation_scheme attribute is not set.
        """
        if self.condensation_scheme is None:
            raise RuntimeError("Energy condensation scheme not set.")

        total_length = 0.0
        for s in segments:
            total_length += s[1]
        invs_tot_length = 1.0 / total_length

        flux = np.zeros(len(self.condensation_scheme))

        for G in range(len(self.condensation_scheme)):
            gmin, gmax = self.condensation_scheme[G][:]
            for g in range(gmin, gmax + 1):
                for s in segments:
                    flux[G] += s[1] * self._asmbly_moc.flux(s[0], g)

        flux *= invs_tot_length

        return flux

    def _compute_adf_cdf_from_cmfd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the assembly and corner discontinuity factors in the few-group
        structure using the CMFD results. Currently, this method neglects the
        coupled x-y terms in the nodal construction when computing the CDFs.
        This simplifies the algorithm, but also ensures problems won't occur if
        trying to run single-pin cell "assembly" calculations.

        Returns
        -------
        ADF : ndarray
            Assembly Discontinuity Factors
        CDF : ndarray
            Corner Discontinuity Factors

        Raises
        ------
        RuntimeError
            If the condensation_scheme attribute is not set or if if is not
            possible to create a condensation scheme to go from the CMFD group
            structure to the few-group structure.
        """
        if self.condensation_scheme is None:
            raise RuntimeError("Energy condensation scheme not set.")

        NG = len(self.condensation_scheme)

        # Create the condensation scheme to go from CMFD to few group structure
        cond_scheme = []
        for G in range(len(self.condensation_scheme)):
            cmfd_grps = []

            # Get lower group
            for g in range(len(self.cmfd_condensation_scheme)):
                if (
                    self.cmfd_condensation_scheme[g][0]
                    == self.condensation_scheme[G][0]
                ):
                    cmfd_grps.append(g)
            if len(cmfd_grps) == 0:
                raise RuntimeError(
                    "Could not create condensation scheme from CMFD structure to few-group structure."
                )

            # Get upper group
            for g in range(len(self.cmfd_condensation_scheme)):
                if (
                    self.cmfd_condensation_scheme[g][-1]
                    == self.condensation_scheme[G][-1]
                ):
                    cmfd_grps.append(g)
            if len(cmfd_grps) == 1:
                raise RuntimeError(
                    "Could not create condensation scheme from CMFD structure to few-group structure."
                )

            cond_scheme.append(cmfd_grps)

        # Create empty arrays for ADFs and CDFs
        adf = np.ones((NG, 6))
        cdf = np.ones((NG, 4))

        # Compute the homogeneous flux for each few-group
        hom_flux = self._get_hom_flux_from_cmfd(cond_scheme)

        # Now we can go get the surface fluxes
        if self.symmetry in [Symmetry.Quarter, Symmetry.Half, Symmetry.Full]:
            het_flux_xp = self._get_het_flux_xp_cmfd(cond_scheme)
            het_flux_yp = self._get_het_flux_yp_cmfd(cond_scheme)
            het_flux_I = self._get_het_flux_I_cmfd(cond_scheme)

            adf[:, ADF.XP] = het_flux_xp / hom_flux.flux_x.pos_surf_flux()
            adf[:, ADF.XN] = adf[:, ADF.XP]
            adf[:, ADF.YP] = het_flux_yp / hom_flux.flux_y.pos_surf_flux()
            adf[:, ADF.YN] = adf[:, ADF.YP]

            cdf[:, CDF.I] = het_flux_I / hom_flux(0.5 * hom_flux.dx, 0.5 * hom_flux.dy)
            cdf[:, CDF.II] = cdf[:, CDF.I]
            cdf[:, CDF.III] = cdf[:, CDF.I]
            cdf[:, CDF.IV] = cdf[:, CDF.I]

        if self.symmetry in [Symmetry.Half, Symmetry.Full] or self.independent_quadrant:
            het_flux_xn = self._get_het_flux_xn_cmfd(cond_scheme)
            het_flux_II = self._get_het_flux_II_cmfd(cond_scheme)

            adf[:, ADF.XN] = het_flux_xn / hom_flux.flux_x.neg_surf_flux()

            cdf[:, CDF.II] = het_flux_II / hom_flux(
                -0.5 * hom_flux.dx, 0.5 * hom_flux.dy
            )
            cdf[:, CDF.III] = cdf[:, CDF.II]

        if self.symmetry == Symmetry.Full or self.independent_quadrant:
            het_flux_yn = self._get_het_flux_yn_cmfd(cond_scheme)
            het_flux_III = self._get_het_flux_III_cmfd(cond_scheme)
            het_flux_IV = self._get_het_flux_IV_cmfd(cond_scheme)

            adf[:, ADF.YN] = het_flux_yn / hom_flux.flux_y.neg_surf_flux()

            cdf[:, CDF.III] = het_flux_III / hom_flux(
                -0.5 * hom_flux.dx, -0.5 * hom_flux.dy
            )
            cdf[:, CDF.IV] = het_flux_IV / hom_flux(
                0.5 * hom_flux.dx, -0.5 * hom_flux.dy
            )

        return adf, cdf

    def _get_hom_flux_from_cmfd(self, cond_scheme: List[List[int]]) -> NodalFlux2D:
        NG = len(cond_scheme)
        moc = self._asmbly_moc
        keff = moc.keff
        cmfd = moc.cmfd

        dx = moc.x_max - moc.x_min
        dy = moc.y_max - moc.y_min

        # Compute average flux for assembly
        flux_spec = moc.homogenize_flux_spectrum()
        avg_flux = np.zeros(NG)
        for G in range(NG):
            for g in range(
                self.condensation_scheme[G][0], self.condensation_scheme[G][1] + 1
            ):
                avg_flux[G] += flux_spec[g]

        # Homogenize the xs and condense for assembly
        fine_xs = moc.homogenize()
        fine_diff_xs = fine_xs.diffusion_xs()
        few_diff_xs = fine_diff_xs.condense(self.condensation_scheme, flux_spec)

        # Condense currents for the assembly
        j_x_neg = self._get_avg_x_neg_current_cmfd(cond_scheme)
        j_x_pos = self._get_avg_x_pos_current_cmfd(cond_scheme)
        j_y_neg = self._get_avg_y_neg_current_cmfd(cond_scheme)
        j_y_pos = self._get_avg_y_pos_current_cmfd(cond_scheme)

        return NodalFlux2D(
            dx, dy, keff, few_diff_xs, avg_flux, j_x_neg, j_x_pos, j_y_neg, j_y_pos
        )

    def _get_het_flux_I_cmfd(self, cond_scheme: List[List[int]]) -> np.ndarray:
        cmfd = self._asmbly_moc.cmfd
        i = cmfd.nx - 1
        j = cmfd.ny - 1

        tile_nodal_flux = self._get_2d_nodal_flux_from_cmfd(cond_scheme, i, j)
        return tile_nodal_flux(0.5 * tile_nodal_flux.dx, 0.5 * tile_nodal_flux.dy)

    def _get_het_flux_II_cmfd(self, cond_scheme: List[List[int]]) -> np.ndarray:
        cmfd = self._asmbly_moc.cmfd
        i = 0
        j = cmfd.ny - 1

        tile_nodal_flux = self._get_2d_nodal_flux_from_cmfd(cond_scheme, i, j)
        return tile_nodal_flux(-0.5 * tile_nodal_flux.dx, 0.5 * tile_nodal_flux.dy)

    def _get_het_flux_III_cmfd(self, cond_scheme: List[List[int]]) -> np.ndarray:
        cmfd = self._asmbly_moc.cmfd
        i = 0
        j = 0

        tile_nodal_flux = self._get_2d_nodal_flux_from_cmfd(cond_scheme, i, j)
        return tile_nodal_flux(-0.5 * tile_nodal_flux.dx, -0.5 * tile_nodal_flux.dy)

    def _get_het_flux_IV_cmfd(self, cond_scheme: List[List[int]]) -> np.ndarray:
        cmfd = self._asmbly_moc.cmfd
        i = cmfd.nx - 1
        j = 0

        tile_nodal_flux = self._get_2d_nodal_flux_from_cmfd(cond_scheme, i, j)
        return tile_nodal_flux(0.5 * tile_nodal_flux.dx, -0.5 * tile_nodal_flux.dy)

    def _get_2d_nodal_flux_from_cmfd(
        self, cond_scheme: List[List[int]], i: int, j: int
    ) -> NodalFlux2D:
        NG = len(cond_scheme)
        moc = self._asmbly_moc
        keff = moc.keff
        cmfd = moc.cmfd

        dx = cmfd.dx[i]
        dy = cmfd.dy[j]

        # Get surface indices
        s_x_neg = cmfd.get_x_neg_surf(i, j)
        s_x_pos = cmfd.get_x_pos_surf(i, j)
        s_y_neg = cmfd.get_y_neg_surf(i, j)
        s_y_pos = cmfd.get_y_pos_surf(i, j)

        # Homogenize average flux for the tile
        cmfd_tile_fsrs = cmfd.tile_fsr_list(i, j)
        tile_flux_spec = moc.homogenize_flux_spectrum(cmfd_tile_fsrs)
        avg_flux = np.zeros(NG)
        for G in range(NG):
            for g in range(
                self.condensation_scheme[G][0], self.condensation_scheme[G][1] + 1
            ):
                avg_flux[G] += tile_flux_spec[g]

        # Homogenize the xs and condense
        fine_tile_xs = moc.homogenize(cmfd_tile_fsrs)
        fine_tile_diff_xs = fine_tile_xs.diffusion_xs()
        few_diff_xs = fine_tile_diff_xs.condense(
            self.condensation_scheme, tile_flux_spec
        )

        # Condense currents for the tile
        j_x_neg = np.zeros(NG)
        j_x_pos = np.zeros(NG)
        j_y_neg = np.zeros(NG)
        j_y_pos = np.zeros(NG)
        for G in range(NG):
            for g in range(cond_scheme[G][0], cond_scheme[G][1] + 1):
                j_x_neg[G] += cmfd.current(g, s_x_neg)
                j_x_pos[G] += cmfd.current(g, s_x_pos)
                j_y_neg[G] += cmfd.current(g, s_y_neg)
                j_y_pos[G] += cmfd.current(g, s_y_pos)

        return NodalFlux2D(
            dx, dy, keff, few_diff_xs, avg_flux, j_x_neg, j_x_pos, j_y_neg, j_y_pos
        )

    def _get_avg_x_pos_current_cmfd(self, cond_scheme: List[List[int]]) -> np.ndarray:
        NG = len(cond_scheme)
        moc = self._asmbly_moc
        cmfd = moc.cmfd

        current = np.zeros(NG)

        dlts = cmfd.dy

        i = cmfd.nx - 1
        for j in range(cmfd.ny):
            # Get surface indices
            s = cmfd.get_x_pos_surf(i, j)

            # Condense currents for the tile
            for G in range(NG):
                for g in range(cond_scheme[G][0], cond_scheme[G][1] + 1):
                    # Weight current contribution by length of segment
                    current[G] += cmfd.current(g, s) * dlts[j]

        # Normalize by length of xp surface
        current /= moc.y_max - moc.y_min
        return current

    def _get_avg_x_neg_current_cmfd(self, cond_scheme: List[List[int]]) -> np.ndarray:
        NG = len(cond_scheme)
        moc = self._asmbly_moc
        cmfd = moc.cmfd

        current = np.zeros(NG)

        dlts = cmfd.dy

        i = 0
        for j in range(cmfd.ny):
            # Get surface indices
            s = cmfd.get_x_neg_surf(i, j)

            # Condense currents for the tile
            for G in range(NG):
                for g in range(cond_scheme[G][0], cond_scheme[G][1] + 1):
                    # Weight current contribution by length of segment
                    current[G] += cmfd.current(g, s) * dlts[j]

        # Normalize by length of xp surface
        current /= moc.y_max - moc.y_min
        return current

    def _get_avg_y_pos_current_cmfd(self, cond_scheme: List[List[int]]) -> np.ndarray:
        NG = len(cond_scheme)
        moc = self._asmbly_moc
        cmfd = moc.cmfd

        current = np.zeros(NG)

        dlts = cmfd.dx

        j = cmfd.ny - 1
        for i in range(cmfd.nx):
            # Get surface indices
            s = cmfd.get_y_pos_surf(i, j)

            # Condense currents for the tile
            for G in range(NG):
                for g in range(cond_scheme[G][0], cond_scheme[G][1] + 1):
                    # Weight current contribution by length of segment
                    current[G] += cmfd.current(g, s) * dlts[i]

        # Normalize by length of xp surface
        current /= moc.x_max - moc.x_min
        return current

    def _get_avg_y_neg_current_cmfd(self, cond_scheme: List[List[int]]) -> np.ndarray:
        NG = len(cond_scheme)
        moc = self._asmbly_moc
        cmfd = moc.cmfd

        current = np.zeros(NG)

        dlts = cmfd.dx

        j = 0
        for i in range(cmfd.nx):
            # Get surface indices
            s = cmfd.get_y_neg_surf(i, j)

            # Condense currents for the tile
            for G in range(NG):
                for g in range(cond_scheme[G][0], cond_scheme[G][1] + 1):
                    # Weight current contribution by length of segment
                    current[G] += cmfd.current(g, s) * dlts[i]

        # Normalize by length of xp surface
        current /= moc.x_max - moc.x_min
        return current

    def _get_het_flux_xp_cmfd(self, cond_scheme: List[List[int]]) -> np.ndarray:
        NG = len(cond_scheme)
        moc = self._asmbly_moc
        keff = moc.keff
        cmfd = moc.cmfd

        dlts = cmfd.dy

        node_fluxes = []

        i = cmfd.nx - 1
        x_min = 0.0
        x_max = cmfd.dx[-1]
        for j in range(cmfd.ny):
            # For tile (i, j), get the FSR list
            cmfd_tile_fsrs = cmfd.tile_fsr_list(i, j)

            # Get surface indices
            s_neg = cmfd.get_x_neg_surf(i, j)
            s_pos = cmfd.get_x_pos_surf(i, j)

            # Homogenize average flux for the tile
            tile_flux_spec = moc.homogenize_flux_spectrum(cmfd_tile_fsrs)
            avg_flux = np.zeros(NG)
            for G in range(NG):
                for g in range(
                    self.condensation_scheme[G][0], self.condensation_scheme[G][1] + 1
                ):
                    avg_flux[G] += tile_flux_spec[g]

            # Homogenize the xs and condense
            fine_tile_xs = moc.homogenize(cmfd_tile_fsrs)
            fine_tile_diff_xs = fine_tile_xs.diffusion_xs()
            few_diff_xs = fine_tile_diff_xs.condense(
                self.condensation_scheme, tile_flux_spec
            )

            # Condense currents for the tile
            j_neg = np.zeros(NG)
            j_pos = np.zeros(NG)
            for G in range(NG):
                for g in range(cond_scheme[G][0], cond_scheme[G][1] + 1):
                    j_neg[G] += cmfd.current(g, s_neg)
                    j_pos[G] += cmfd.current(g, s_pos)

            # Create 1D nodal flux object
            node_fluxes.append(
                NodalFlux1D(x_min, x_max, keff, few_diff_xs, avg_flux, j_neg, j_pos)
            )

        # Now we need to get the average heterogeneous flux on the surface
        het_flux = np.zeros(NG)
        for G in range(NG):
            for j in range(cmfd.ny):
                het_flux[G] += node_fluxes[j].pos_surf_flux(G) * dlts[j]
        het_flux /= moc.y_max - moc.y_min

        return het_flux

    def _get_het_flux_xn_cmfd(self, cond_scheme: List[List[int]]) -> np.ndarray:
        NG = len(cond_scheme)
        moc = self._asmbly_moc
        keff = moc.keff
        cmfd = moc.cmfd

        dlts = cmfd.dy

        node_fluxes = []

        i = 0
        x_min = 0.0
        x_max = cmfd.dx[0]
        for j in range(cmfd.ny):
            # For tile (i, j), get the FSR list
            cmfd_tile_fsrs = cmfd.tile_fsr_list(i, j)

            # Get surface indices
            s_neg = cmfd.get_x_neg_surf(i, j)
            s_pos = cmfd.get_x_pos_surf(i, j)

            # Homogenize average flux for the tile
            tile_flux_spec = moc.homogenize_flux_spectrum(cmfd_tile_fsrs)
            avg_flux = np.zeros(NG)
            for G in range(NG):
                for g in range(
                    self.condensation_scheme[G][0], self.condensation_scheme[G][1] + 1
                ):
                    avg_flux[G] += tile_flux_spec[g]

            # Homogenize the xs and condense
            fine_tile_xs = moc.homogenize(cmfd_tile_fsrs)
            fine_tile_diff_xs = fine_tile_xs.diffusion_xs()
            few_diff_xs = fine_tile_diff_xs.condense(
                self.condensation_scheme, tile_flux_spec
            )

            # Condense currents for the tile
            j_neg = np.zeros(NG)
            j_pos = np.zeros(NG)
            for G in range(NG):
                for g in range(cond_scheme[G][0], cond_scheme[G][1] + 1):
                    j_neg[G] += cmfd.current(g, s_neg)
                    j_pos[G] += cmfd.current(g, s_pos)

            # Create 1D nodal flux object
            node_fluxes.append(
                NodalFlux1D(x_min, x_max, keff, few_diff_xs, avg_flux, j_neg, j_pos)
            )

        # Now we need to get the average heterogeneous flux on the surface
        het_flux = np.zeros(NG)
        for G in range(NG):
            for j in range(cmfd.ny):
                het_flux[G] += node_fluxes[j].neg_surf_flux(G) * dlts[j]
        het_flux /= moc.y_max - moc.y_min

        return het_flux

    def _get_het_flux_yp_cmfd(self, cond_scheme: List[List[int]]) -> np.ndarray:
        NG = len(cond_scheme)
        moc = self._asmbly_moc
        keff = moc.keff
        cmfd = moc.cmfd

        dlts = cmfd.dx

        node_fluxes = []

        j = cmfd.ny - 1
        y_min = 0.0
        y_max = cmfd.dy[-1]
        for i in range(cmfd.nx):
            # For tile (i, j), get the FSR list
            cmfd_tile_fsrs = cmfd.tile_fsr_list(i, j)

            # Get surface indices
            s_neg = cmfd.get_y_neg_surf(i, j)
            s_pos = cmfd.get_y_pos_surf(i, j)

            # Homogenize average flux for the tile
            tile_flux_spec = moc.homogenize_flux_spectrum(cmfd_tile_fsrs)
            avg_flux = np.zeros(NG)
            for G in range(NG):
                for g in range(
                    self.condensation_scheme[G][0], self.condensation_scheme[G][1] + 1
                ):
                    avg_flux[G] += tile_flux_spec[g]

            # Homogenize the xs and condense
            fine_tile_xs = moc.homogenize(cmfd_tile_fsrs)
            fine_tile_diff_xs = fine_tile_xs.diffusion_xs()
            few_diff_xs = fine_tile_diff_xs.condense(
                self.condensation_scheme, tile_flux_spec
            )

            # Condense currents for the tile
            j_neg = np.zeros(NG)
            j_pos = np.zeros(NG)
            for G in range(NG):
                for g in range(cond_scheme[G][0], cond_scheme[G][1] + 1):
                    j_neg[G] += cmfd.current(g, s_neg)
                    j_pos[G] += cmfd.current(g, s_pos)

            # Create 1D nodal flux object
            node_fluxes.append(
                NodalFlux1D(y_min, y_max, keff, few_diff_xs, avg_flux, j_neg, j_pos)
            )

        # Now we need to get the average heterogeneous flux on the surface
        het_flux = np.zeros(NG)
        for G in range(NG):
            for i in range(cmfd.nx):
                het_flux[G] += node_fluxes[i].pos_surf_flux(G) * dlts[i]
        het_flux /= moc.x_max - moc.x_min

        return het_flux

    def _get_het_flux_yn_cmfd(self, cond_scheme: List[List[int]]) -> np.ndarray:
        NG = len(cond_scheme)
        moc = self._asmbly_moc
        keff = moc.keff
        cmfd = moc.cmfd

        dlts = cmfd.dx
        node_fluxes = []

        j = 0
        y_min = 0.0
        y_max = cmfd.dy[0]
        for i in range(cmfd.nx):
            # For tile (i, j), get the FSR list
            cmfd_tile_fsrs = cmfd.tile_fsr_list(i, j)

            # Get surface indices
            s_neg = cmfd.get_y_neg_surf(i, j)
            s_pos = cmfd.get_y_pos_surf(i, j)

            # Homogenize average flux for the tile
            tile_flux_spec = moc.homogenize_flux_spectrum(cmfd_tile_fsrs)
            avg_flux = np.zeros(NG)
            for G in range(NG):
                for g in range(
                    self.condensation_scheme[G][0], self.condensation_scheme[G][1] + 1
                ):
                    avg_flux[G] += tile_flux_spec[g]

            # Homogenize the xs and condense
            fine_tile_xs = moc.homogenize(cmfd_tile_fsrs)
            fine_tile_diff_xs = fine_tile_xs.diffusion_xs()
            few_diff_xs = fine_tile_diff_xs.condense(
                self.condensation_scheme, tile_flux_spec
            )

            # Condense currents for the tile
            j_neg = np.zeros(NG)
            j_pos = np.zeros(NG)
            for G in range(NG):
                for g in range(cond_scheme[G][0], cond_scheme[G][1] + 1):
                    j_neg[G] += cmfd.current(g, s_neg)
                    j_pos[G] += cmfd.current(g, s_pos)

            # Create 1D nodal flux object
            node_fluxes.append(
                NodalFlux1D(y_min, y_max, keff, few_diff_xs, avg_flux, j_neg, j_pos)
            )

        # Now we need to get the average heterogeneous flux on the surface
        het_flux = np.zeros(NG)
        for G in range(NG):
            for i in range(cmfd.nx):
                het_flux[G] += node_fluxes[i].neg_surf_flux(G) * dlts[i]
        het_flux /= moc.x_max - moc.x_min

        return het_flux

    def _compute_adf_cdf_from_moc(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the assembly and corner discontinuity factors in the few-group
        structure using the MOC results.

        Returns
        -------
        ADF : ndarray
            Assembly Discontinuity Factors
        CDF : ndarray
            Corner Discontinuity Factors

        Raises
        ------
        RuntimeError
            If the condensation_scheme attribute is not set.
        """
        if self.condensation_scheme is None:
            raise RuntimeError("Energy condensation scheme not set.")

        NG = len(self.condensation_scheme)
        moc = self._asmbly_moc

        # First, compute the homogeneous flux
        homog_flux = np.zeros(NG)
        total_volume = 0.0
        for i in range(moc.nfsr):
            Vi = moc.volume(i)
            total_volume += Vi

            for G in range(NG):
                gmin, gmax = self.condensation_scheme[G][:]
                for g in range(gmin, gmax + 1):
                    homog_flux[G] += Vi * moc.flux(i, g)
        homog_flux /= total_volume

        # Create empty arrays for ADFs and CDFs
        adf = np.ones((NG, 6))
        cdf = np.zeros((NG, 4))

        if self.symmetry == Symmetry.Full or (
            self.symmetry == Symmetry.Quarter and self.independent_quadrant
        ):
            # Get flux along surfaces
            xn_segments = moc.trace_fsr_segments(
                Vector(moc.x_min + 0.001, moc.y_max), Direction(0.0, -1.0)
            )
            xp_segments = moc.trace_fsr_segments(
                Vector(moc.x_max - 0.001, moc.y_max), Direction(0.0, -1.0)
            )
            yn_segments = moc.trace_fsr_segments(
                Vector(moc.x_min, moc.y_min + 0.001), Direction(1.0, 0.0)
            )
            yp_segments = moc.trace_fsr_segments(
                Vector(moc.x_min, moc.y_max - 0.001), Direction(1.0, 0.0)
            )
            xn_flx = self._compute_average_line_flux(xn_segments)
            xp_flx = self._compute_average_line_flux(xp_segments)
            yn_flx = self._compute_average_line_flux(yn_segments)
            yp_flx = self._compute_average_line_flux(yp_segments)

            # Get flux at corners
            I_flx = self._compute_few_group_flux(
                Vector(moc.x_max - 0.001, moc.y_max - 0.001), Direction(-1.0, -1.0)
            )
            II_flx = self._compute_few_group_flux(
                Vector(moc.x_min + 0.001, moc.y_max - 0.001), Direction(1.0, -1.0)
            )
            III_flx = self._compute_few_group_flux(
                Vector(moc.x_min + 0.001, moc.y_min + 0.001), Direction(1.0, 1.0)
            )
            IV_flx = self._compute_few_group_flux(
                Vector(moc.x_max - 0.001, moc.y_min + 0.001), Direction(-1.0, 1.0)
            )

            for G in range(NG):
                adf[G, ADF.XN] = xn_flx[G] / homog_flux[G]
                adf[G, ADF.XP] = xp_flx[G] / homog_flux[G]
                adf[G, ADF.YN] = yn_flx[G] / homog_flux[G]
                adf[G, ADF.YP] = yp_flx[G] / homog_flux[G]
                # The ADFs on the +/- Z sides will be left at unity

                cdf[G, CDF.I] = I_flx[G] / homog_flux[G]
                cdf[G, CDF.II] = II_flx[G] / homog_flux[G]
                cdf[G, CDF.III] = III_flx[G] / homog_flux[G]
                cdf[G, CDF.IV] = IV_flx[G] / homog_flux[G]

        elif self.symmetry == Symmetry.Half:
            # Get flux along surfaces
            xn_segments = moc.trace_fsr_segments(
                Vector(moc.x_min + 0.001, moc.y_max), Direction(0.0, -1.0)
            )
            xp_segments = moc.trace_fsr_segments(
                Vector(moc.x_max - 0.001, moc.y_max), Direction(0.0, -1.0)
            )
            yp_segments = moc.trace_fsr_segments(
                Vector(moc.x_min, moc.y_max - 0.001), Direction(1.0, 0.0)
            )
            xn_flx = self._compute_average_line_flux(xn_segments)
            xp_flx = self._compute_average_line_flux(xp_segments)
            yp_flx = self._compute_average_line_flux(yp_segments)

            # Get flux at corners
            I_flx = self._compute_few_group_flux(
                Vector(moc.x_max - 0.001, moc.y_max - 0.001), Direction(-1.0, -1.0)
            )
            II_flx = self._compute_few_group_flux(
                Vector(moc.x_min + 0.001, moc.y_max - 0.001), Direction(1.0, -1.0)
            )

            for G in range(NG):
                adf[G, ADF.XN] = xn_flx[G] / homog_flux[G]
                adf[G, ADF.XP] = xp_flx[G] / homog_flux[G]
                adf[G, ADF.YP] = yp_flx[G] / homog_flux[G]
                adf[G, ADF.YN] = adf[G, ADF.YP]
                # The ADFs on the +/- Z sides will be left at unity

                cdf[G, CDF.I] = I_flx[G] / homog_flux[G]
                cdf[G, CDF.II] = II_flx[G] / homog_flux[G]
                cdf[G, CDF.III] = cdf[G, CDF.II]
                cdf[G, CDF.IV] = cdf[G, CDF.I]

        else:  # Quarter symmetry
            # Get flux along surfaces
            xp_segments = moc.trace_fsr_segments(
                Vector(moc.x_max - 0.001, moc.y_max), Direction(0.0, -1.0)
            )
            yp_segments = moc.trace_fsr_segments(
                Vector(moc.x_min, moc.y_max - 0.001), Direction(1.0, 0.0)
            )
            xp_flx = self._compute_average_line_flux(xp_segments)
            yp_flx = self._compute_average_line_flux(yp_segments)

            # Get flux at corners
            I_flx = self._compute_few_group_flux(
                Vector(moc.x_max - 0.001, moc.y_max - 0.001), Direction(-1.0, -1.0)
            )

            for G in range(NG):
                adf[G, ADF.XP] = xp_flx[G] / homog_flux[G]
                adf[G, ADF.XN] = adf[G, ADF.XP]
                adf[G, ADF.YP] = yp_flx[G] / homog_flux[G]
                adf[G, ADF.YN] = adf[G, ADF.YP]
                # The ADFs on the +/- Z sides will be left at unity

                cdf[G, CDF.I] = I_flx[G] / homog_flux[G]
                cdf[G, CDF.II] = cdf[G, CDF.I]
                cdf[G, CDF.III] = cdf[G, CDF.I]
                cdf[G, CDF.IV] = cdf[G, CDF.I]

        return adf, cdf

    def _compute_form_factors(self) -> np.ndarray:
        """
        Computes the one group pin power form factors for the full assembly.

        Returns
        -------
        ndarray
            A 2D Numpy array for the pin power form factors. First index is
            y (from high to low) and the second index is x (from low to high).
        """
        if self.symmetry == Symmetry.Quarter and self.independent_quadrant:
            ff = np.zeros((self._simulated_shape[1], self._simulated_shape[0]))

            for j in range(len(self.cells)):
                for i in range(len(self.cells[j])):
                    cell = self.cells[j][i]

                    if not isinstance(cell, FuelPin):
                        continue

                    # Pin Power
                    ff[j, i] = cell.compute_pin_linear_power(self._ndl)

            mean_ff = np.mean(ff)
            ff /= mean_ff

            return ff

        else:
            ff = np.zeros((self.shape[1], self.shape[0]))

            for j in range(len(self.cells)):
                for i in range(len(self.cells[j])):
                    cell = self.cells[j][i]

                    if not isinstance(cell, FuelPin):
                        continue

                    # Pin Power
                    pp = cell.compute_pin_linear_power(self._ndl)

                    if self.symmetry == Symmetry.Full:
                        ff[j, i] = pp

                    elif self.symmetry == Symmetry.Half:
                        ff[j, i] = pp
                        ff[-(j + 1), i] = pp

                    elif self.symmetry == Symmetry.Quarter:
                        ox = self.shape[1] // 2

                        ff[j, i + ox] = pp
                        ff[j, -(i + ox + 1)] = pp

                        ff[-(j + 1), i + ox] = pp
                        ff[-(j + 1), -(i + ox + 1)] = pp

            mean_ff = np.mean(ff)
            ff /= mean_ff

            return ff

    def _compute_few_group_xs(self) -> DiffusionCrossSection:
        """
        Computes the few-group diffusion cross sections for the problem.

        Returns
        -------
        DiffusionCrossSection
            Few-group diffusion cross sections for the assembly.

        Raises
        ------
        RuntimeError
            If the condensation_scheme attribute is not set.
        """
        if self.condensation_scheme is None:
            raise RuntimeError("Energy condensation scheme not set.")

        # According to Smith, one should do energy condensation on the
        # diffusion coefficients, and not on the transport cross sections which
        # one could then use to make diffusion coefficients [1]. This is in
        # contradiction to Lattice Physics Computations which states that
        # either method is acceptable [2]. In light of these comments, I have
        # chosen to go with Smith's recommendation of performing energy
        # condensation on the diffusion coefficients.

        homog_xs = self._asmbly_moc.homogenize()
        diff_xs = homog_xs.diffusion_xs()
        flux_spectrum = self._asmbly_moc.homogenize_flux_spectrum()
        return diff_xs.condense(self.condensation_scheme, flux_spectrum)

    def _compute_diffusion_data(self) -> DiffusionData:
        """
        Computes the nodal diffusion data for the assembly.

        Returns
        -------
        DiffusionData
            Few-group diffusion cross sections and discontinuity factors.
        """
        diff_xs = self._compute_few_group_xs()
        ff = self._compute_form_factors()

        if (
            self.cmfd
            and self.cmfd_condensation_scheme is not None
            and not self.prefer_moc_adf_cdf
        ):
            # If we are using CMFD, we should use the currents to generate the
            # ADFs and CDFs, as it is far more accurate, as explained by Smith [1].
            adf, cdf = self._compute_adf_cdf_from_cmfd()
        else:
            if not self.prefer_moc_adf_cdf:
                mssg = "CMFD is not on. Accurate discontinuity factors cannot be computed without CMFD."
                scarabee_log(LogLevel.Warning, mssg)
            adf, cdf = self._compute_adf_cdf_from_moc()

        return DiffusionData(diff_xs, ff, adf, cdf)

    def _run_assembly_calculation(
        self,
        self_shield: bool,
        apply_dancoff_corrections: bool = False,
        transport: bool = True,
    ) -> None:
        """
        Runs a single MOC calculation, applies critical leakage model, obtains
        the flux spectra for each fuel region, and then normalize the flux
        based on the linear assembly power. Self-shielding can be performed
        if desired. If not, the last computed Dancoff corrections can be
        applied to all fuel pins (to run a time step without explicit
        self-shielding step, using previously known Dancoff corrections).

        Paramters
        ---------
        self_shield : bool
            If True, self-shielding is performed for the fuel and cladding.
        apply_dancoff_corrections : bool, default False
            If self_shield is False and this option is True, the previously
            obtained Dancoff corrections are applied to all cells.
        transport : bool, default True
            If True, the MOC calculation is performed. Otherwise, the MOC
            calculation is not performed, but the leakage correction and flux
            normalization are.
        """
        if self_shield:
            # If we want self-shielding, do that stuff
            self.self_shield_and_xs_update()
        elif apply_dancoff_corrections:
            # Sets dancoff corrections, even if self-shielding wasn't performed
            self.apply_dancoff_corrections()

        self.recompute_all_xs()

        if self._asmbly_moc is None:
            self._init_moc()

        self._asmbly_moc.flux_tolerance = self.flux_tolerance
        self._asmbly_moc.keff_tolerance = self.keff_tolerance

        if transport:
            set_logging_level(LogLevel.Warning)
            self._asmbly_moc.solve()
            set_logging_level(LogLevel.Info)
            scarabee_log(LogLevel.Info, "")
            scarabee_log(LogLevel.Info, "Kinf: {:.5f}".format(self._asmbly_moc.keff))

        self.apply_leakage_model()
        self.obtain_flux_spectra()
        self.normalize_flux_to_power()
        self.apply_infinite_spectrum()

    def _predict_depletion(self, dt: float, dtm1: Optional[float]) -> None:
        # Do all depletions in parallel
        threads = []
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]
                threads.append(
                    Thread(
                        target=cell.predict_depletion,
                        args=(self._chain, self._ndl, dt, dtm1),
                    )
                )
                threads[-1].start()
        for t in threads:
            t.join()

    def _correct_depletion(self, dt: float, dtm1: Optional[float]) -> None:
        # Do all depletions in parallel
        threads = []
        for j in range(len(self.cells)):
            for i in range(len(self.cells[j])):
                cell = self.cells[j][i]
                threads.append(
                    Thread(
                        target=cell.correct_depletion,
                        args=(self._chain, self._ndl, dt, dtm1),
                    )
                )
                threads[-1].start()
        for t in threads:
            t.join()

    def _run_depletion_steps(self) -> None:
        if self._chain is None:
            raise RuntimeError(
                "No depletion chain is present. Cannot run depletion calculation."
            )

        self._keff = np.zeros(self.depletion_exposure_steps.size + 1)
        self._exposures = np.zeros(self.depletion_exposure_steps.size + 1)
        self._times = np.zeros(self.depletion_exposure_steps.size + 1)
        self._diffusion_data = []

        for t, dt in enumerate(self.depletion_time_steps):
            if t > 0:
                self._exposures[t] = (
                    self._exposures[t - 1] + self.depletion_exposure_steps[t - 1]
                )
                self._times[t] = self._times[t - 1] + self.depletion_time_steps[t - 1]

            scarabee_log(LogLevel.Info, "")
            scarabee_log(LogLevel.Info, 60 * "-")
            scarabee_log(LogLevel.Info, "Running Time Step {:}".format(t))
            scarabee_log(
                LogLevel.Info, "Exposure: {:.3E} MWd/kg".format(self._exposures[t])
            )
            scarabee_log(LogLevel.Info, "Time    : {:.3E} days".format(self._times[t]))
            scarabee_log(LogLevel.Info, "")
            # Convert days to seconds
            dt_sec = dt * 60.0 * 60.0 * 24.0

            dtm1_sec = None
            if t > 0:
                dtm1_sec = self.depletion_time_steps[t - 1] * 60.0 * 60.0 * 24.0

            scarabee_log(LogLevel.Info, "Predictor:")
            # Run initial calcualtion for this time step
            self._run_assembly_calculation(True)
            scarabee_log(LogLevel.Info, "")
            self._keff[t] = self._asmbly_moc.keff
            self._diffusion_data.append(self._compute_diffusion_data())

            # Predic isotopes at midpoint of step
            self._predict_depletion(dt_sec, dtm1_sec)

            scarabee_log(LogLevel.Info, "Corrector:")
            # Run the a new transport calcualtion to get rates
            self._run_assembly_calculation(False, transport=self.corrector_transport)

            # Do correction step for isotopes
            self._correct_depletion(dt_sec, dtm1_sec)

        # Run last step at the end to get keff for our final material compositions
        scarabee_log(LogLevel.Info, "")
        scarabee_log(LogLevel.Info, 60 * "-")
        self._exposures[-1] = self._exposures[-2] + self.depletion_exposure_steps[-1]
        self._times[-1] = self._times[-2] + dt
        scarabee_log(
            LogLevel.Info, "Running Time Step {:}".format(self._times.size - 1)
        )
        scarabee_log(
            LogLevel.Info, "Exposure: {:.3E} MWd/kg".format(self._exposures[-1])
        )
        scarabee_log(LogLevel.Info, "Time    : {:.3E} days".format(self._times[-1]))
        scarabee_log(LogLevel.Info, "")
        self._run_assembly_calculation(True)
        self._keff[-1] = self._asmbly_moc.keff
        self._diffusion_data.append(self._compute_diffusion_data())
        scarabee_log(LogLevel.Info, "")

    def solve(self) -> None:
        """
        Solves the assembly problem. If depletion_exposure_steps or
        depletion_time_steps are None, then a single k-eigenvalue problem will
        be run. Otherwise, the depletion calculation will be performed.
        """
        if self.depletion_exposure_steps is None:
            # Single one-off calulcation
            self._run_assembly_calculation(True)
            self._keff = self._asmbly_moc.keff
            self._diffusion_data = self._compute_diffusion_data()
        else:
            # Run depletion steps
            self._run_depletion_steps()


# REFERENCES
#
# [1] K. S. Smith, “Nodal diffusion methods and lattice physics data in LWR
#     analyses: Understanding numerous subtle details,” Prog Nucl Energ,
#     vol. 101, pp. 360–369, 2017, doi: 10.1016/j.pnucene.2017.06.013.
#
# [2] D. Knott and A. Yamamoto, "Lattice Physics Computations" in
#     Handbook of Nuclear Engineering, 2010, p 1226.
