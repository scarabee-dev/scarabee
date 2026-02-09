from typing import List, Optional
import numpy as np
from .pwr_case_matrix_options import CaseMatrix
from .._scarabee import DiffusionData

class AssemblyStatePoint:
    """ 
    The AssemblyStatePoint should store all the relevant data 
    from the lattice calculations within the PWRAssembly class. 
    It currently only stores a DiffusionData object 
    and the parameters of the lattice calculation, 
    but it will be extended to include 
    the base cross sections, pin-power form factors, and ADFs/CDFs etc.
    """
    def __init__(
        self,
        diffusion_data: DiffusionData,
        exposure: float,
        boron_ppm: float,
        moderator_temp: float,
        moderator_pressure: float,
        k_eff: float
    ):
        self.diffusion_data = diffusion_data
        self.exposure = exposure
        self.boron_ppm = boron_ppm
        self.moderator_temp = moderator_temp
        self.moderator_pressure = moderator_pressure
        self.keff = k_eff

    def __repr__(self) -> str:
        return (
            f"AssemblyStatePoint(E={self.exposure:.1f} MWd/kg, "
            f"B={self.boron_ppm:.0f} ppm, "
            f"Tm={self.moderator_temp:.0f}K, "
            f"Pm={self.moderator_pressure:.2f} MPa)"
        )

class AssemblySlice:
    """ 
    The AssemblySlice stores AssemblyStatePoint objects at different operating conditions, 
    facilitating a branching capability. In the future, it will be extended 
    to allow different AssemblySlices to be stacked on top of one another, 
    representing heterogeneities throughout the height of a 3D assembly. 
    """
    def __init__(
        self,
        case_matrix_options: CaseMatrix, 
        exposure_steps: Optional[List[float]]
    ):
        self.case_matrix_options = case_matrix_options

        # Converting the list of exposure steps into exposure values
        self.exposures = (np.array([0.0], dtype=float) if exposure_steps is None
            else np.concatenate(([0.0], np.cumsum(np.asarray(exposure_steps, dtype=float)))))
        self.num_burnup_steps = len(self.exposures)

        # Determine shape of multi-dimensional array
        # Shape: [burnup_steps, boron_values, temp_values, pressure_values]
        self.num_boron_values = len(case_matrix_options.boron_values) if case_matrix_options.branch_boron else 1
        self.num_temp_values = len(case_matrix_options.moderator_temps) if case_matrix_options.branch_moderator_temp else 1
        self.num_pressure_values = len(case_matrix_options.moderator_pressures) if case_matrix_options.branch_moderator_pressure else 1

        shape = (
            self.num_burnup_steps,
            self.num_boron_values,
            self.num_temp_values,
            self.num_pressure_values
        )

        # Initialize multi-dimensional array of state points
        # AssemblyStatePoint = state_points[burnup_step, boron_idx, temp_idx, pressure_idx]
        self.state_points = np.empty(shape, dtype=object)
        self.state_points.fill(None)

    def get_boron_index(self, boron_ppm: float) -> int:
        """Get the index for a given boron concentration."""
        if not self.case_matrix_options.branch_boron:
            return 0

        values = self.case_matrix_options.boron_values
        for i, b in enumerate(values):
            if abs(b - boron_ppm) < 1e-6:
                return i

        raise ValueError(f"Boron concentration {boron_ppm} not in configured values: {values}")

    def get_temp_index(self, moderator_temp: float) -> int:
        if not self.case_matrix_options.branch_moderator_temp:
            return 0

        values = self.case_matrix_options.moderator_temps
        for i, t in enumerate(values):
            if abs(t - moderator_temp) < 1e-6:
                return i

        raise ValueError(f"Moderator temperature {moderator_temp} not in configured values: {values}")

    def get_pressure_index(self, moderator_pressure: float) -> int:
        if not self.case_matrix_options.branch_moderator_pressure:
            return 0

        values = self.case_matrix_options.moderator_pressures
        for i, p in enumerate(values):
            if abs(p - moderator_pressure) < 1e-6:
                return i

        raise ValueError(f"Moderator pressure {moderator_pressure} not in configured values: {values}")

    def get_exposure_index(self, exposure: float) -> int:
        for i, exp in enumerate(self.exposures):
            if abs(exp - exposure) < 1e-6:
                return i

        raise ValueError(
            f"Exposure {exposure} MWd/kg not found in exposure steps. "
            f"Available: {self.exposures.tolist()}"
        )

    def add_state_point(
        self,
        burnup_step: int,
        state_point: AssemblyStatePoint,
        boron_idx: Optional[int] = None,
        temp_idx: Optional[int] = None,
        pressure_idx: Optional[int] = None
    ) -> None:
        if burnup_step < 0 or burnup_step >= self.num_burnup_steps:
            raise ValueError(f"Burnup step {burnup_step} out of range [0, {self.num_burnup_steps-1}]")

        # Auto-determine indices if not provided
        if boron_idx is None:
            boron_idx = self.get_boron_index(state_point.boron_ppm)
        if temp_idx is None:
            temp_idx = self.get_temp_index(state_point.moderator_temp)
        if pressure_idx is None:
            pressure_idx = self.get_pressure_index(state_point.moderator_pressure)

        if boron_idx < 0 or boron_idx >= self.num_boron_values:
            raise ValueError(f"Boron index {boron_idx} out of range [0, {self.num_boron_values-1}]")
        if temp_idx < 0 or temp_idx >= self.num_temp_values:
            raise ValueError(f"Temperature index {temp_idx} out of range [0, {self.num_temp_values-1}]")
        if pressure_idx < 0 or pressure_idx >= self.num_pressure_values:
            raise ValueError(f"Pressure index {pressure_idx} out of range [0, {self.num_pressure_values-1}]")

        self.state_points[burnup_step, boron_idx, temp_idx, pressure_idx] = state_point

    def get_state(
        self,
        burnup_step: int = 0,
        boron_idx: int = 0,
        temp_idx: int = 0,
        pressure_idx: int = 0
    ) -> Optional[AssemblyStatePoint]:
        return self.state_points[burnup_step, boron_idx, temp_idx, pressure_idx]

    def __repr__(self) -> str:
        num_filled = sum(sp is not None for sp in self.state_points.ravel())
        total = self.state_points.size

        s = f"AssemblySlice:\n"
        s += f" Burnup steps: {self.num_burnup_steps};\n"

        if self.case_matrix_options.branch_boron:
            s += f" Boron values: {self.case_matrix_options.boron_values};\n"
        else:
            s += f" Boron: spine only;\n"

        if self.case_matrix_options.branch_moderator_temp:
            s += f" Moderator temps: {self.case_matrix_options.moderator_temps};\n"
        else:
            s += f" Moderator temp: spine only;\n"

        if self.case_matrix_options.branch_moderator_pressure:
            s += f" Moderator pressures: {self.case_matrix_options.moderator_pressures};\n"
        else:
            s += f" Moderator pressure: spine only;\n"

        s += f" Shape: {self.state_points.shape}\n"
        s += f" State points: {num_filled}/{total} filled"

        return s