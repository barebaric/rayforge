from __future__ import annotations
from dataclasses import dataclass, field
import uuid


@dataclass
class Tab:
    """Represents a single tab on a workpiece's geometry."""

    width: float  # The length of the tab along the path in mm
    segment_index: int  # The index of the Command in Geometry.commands
    pos: float  # Normalized position (0.0 to 1.0) along that segment
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
