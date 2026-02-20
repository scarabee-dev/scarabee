from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class PWRCaseMatrix:
    """ 
    A dataclass to configure the branching capabilities within PWRAssembly class. 
    Currently able to branch over boron, the moderator temperature and the moderator pressure. 
    """
    boron_values: List[float] = field(default_factory=lambda: [0.0, 800.0, 1600.0])
    moderator_temps: List[float] = field(default_factory=lambda: [560.0, 580.0, 600.0])
    moderator_pressures: List[float] = field(default_factory=lambda: [15.0, 16.0])
    
    # Branch flags
    branch_boron: bool = False
    branch_moderator_temp: bool = False
    branch_moderator_pressure: bool = False
    
    def __post_init__(self) -> None:
        # Basic sanity
        self.boron_values = [float(x) for x in self.boron_values]
        self.moderator_temps = [float(x) for x in self.moderator_temps]
        self.moderator_pressures = [float(x) for x in self.moderator_pressures]

        if any(x < 0.0 for x in self.boron_values):
            raise ValueError("boron_values must be >= 0.")
        if any(x <= 0.0 for x in self.moderator_temps):
            raise ValueError("moderator_temps must be > 0.")
        if any(x <= 0.0 for x in self.moderator_pressures):
            raise ValueError("moderator_pressures must be > 0.")

        if self.branch_boron and len(self.boron_values) == 0:
            raise ValueError("branch_boron is True but boron_values is empty.")
        if self.branch_moderator_temp and len(self.moderator_temps) == 0:
            raise ValueError("branch_moderator_temp is True but moderator_temps is empty.")
        if self.branch_moderator_pressure and len(self.moderator_pressures) == 0:
            raise ValueError("branch_moderator_pressure is True but moderator_pressures is empty.")
