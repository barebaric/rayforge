from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple


@dataclass
class RuidaLayer:
    """
    Defines the parameters for a single Ruida 'color' or 'layer'.
    These settings are applied to all subsequent geometric commands
    associated with this layer's color_index.
    """

    color_index: int
    speed: float  # in mm/s
    power: float  # as a percentage (0-100)
    air_assist: bool = False
    # Ruida has min/max power for cornering, but we'll start simple.


@dataclass
class RuidaCommand:
    """
    Represents a single geometric or state command tagged with a layer index.
    """

    command_type: str  # e.g., 'Move_Abs', 'Cut_Abs', 'End'
    params: List[Any] = field(default_factory=list)
    # The layer these command parameters belong to.
    color_index: int = 0


@dataclass
class RuidaJob:
    """
    The complete logical representation of a job for a Ruida controller.
    This object is the bridge between the low-level binary file format
    and the application's internal models.
    """

    # A map of color indexes to their layer parameter definitions.
    layers: Dict[int, RuidaLayer] = field(default_factory=dict)

    # An ordered list of commands to be executed.
    commands: List[RuidaCommand] = field(default_factory=list)

    def get_extents(self) -> Tuple[float, float, float, float]:
        """
        Calculates the bounding box (min_x, min_y, max_x, max_y) in mm
        of all geometric commands in the job.
        """
        points = []
        for cmd in self.commands:
            if cmd.command_type in ("Move_Abs", "Cut_Abs") and cmd.params:
                points.append(cmd.params)

        if not points:
            return 0.0, 0.0, 0.0, 0.0

        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)

        return min_x, min_y, max_x, max_y
