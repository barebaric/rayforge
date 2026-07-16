"""
Parameter and data-class definitions for the Array / Pattern tool.

An "array" duplicates a multi-layer selection of items into a regular
pattern. Three arrangements are supported:

* GRID - copies laid out in a rows x columns grid.
* POINT_ROTATION - copies rotated in place around the selection's
  own center.
* CIRCULAR - copies placed along a circular arc around a center.

Each duplicate preserves the layer membership of its source item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class ArrayMode(Enum):
    """The geometric arrangement of the array."""

    GRID = "grid"
    POINT_ROTATION = "point_rotation"
    CIRCULAR = "circular"


class SpacingMode(Enum):
    """How the distance between adjacent copies is interpreted.

    DISPLACEMENT: center-to-center distance (independent of item size).
    GAP:          edge-to-edge distance; the pitch is computed from the
                 selection's collective bounding box plus the gap.
    """

    DISPLACEMENT = "displacement"
    GAP = "gap"


@dataclass
class GridArrayParams:
    """Parameters for a rows x columns grid array."""

    rows: int = 2
    cols: int = 2
    spacing_mode: SpacingMode = SpacingMode.GAP
    # Horizontal distance between adjacent columns.
    col_spacing_mm: float = 1.0
    # Vertical distance between adjacent rows.
    row_spacing_mm: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "spacing_mode": self.spacing_mode.value,
            "col_spacing_mm": self.col_spacing_mm,
            "row_spacing_mm": self.row_spacing_mm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridArrayParams":
        try:
            spacing = SpacingMode(data.get("spacing_mode", "gap"))
        except ValueError:
            spacing = SpacingMode.GAP
        return cls(
            rows=int(data.get("rows", 2)),
            cols=int(data.get("cols", 2)),
            spacing_mode=spacing,
            col_spacing_mm=float(data.get("col_spacing_mm", 1.0)),
            row_spacing_mm=float(data.get("row_spacing_mm", 1.0)),
        )


@dataclass
class PointRotationParams:
    """Parameters for a point-rotation array.

    Copies are rotated in place around the selection's own center and
    therefore share the same position; only their orientation differs.
    """

    count: int = 6
    total_angle_deg: float = 360.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "total_angle_deg": self.total_angle_deg,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PointRotationParams":
        return cls(
            count=int(data.get("count", 6)),
            total_angle_deg=float(data.get("total_angle_deg", 360.0)),
        )


@dataclass
class CircularArrayParams:
    """Parameters for a circular array.

    Copies are placed along a circular arc around ``center_mm`` at
    ``radius_mm``. The default radius is auto-computed from the center
    and the selection's center when the dialog opens; the guide circle
    is always drawn at this radius. With ``rotate_copies`` each copy is
    also spun by its angular offset around the selection's own center.
    """

    count: int = 6
    total_angle_deg: float = 360.0
    center_mm: Tuple[float, float] = (0.0, 0.0)
    radius_mm: float = 10.0
    rotate_copies: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "total_angle_deg": self.total_angle_deg,
            "center_mm": list(self.center_mm),
            "radius_mm": self.radius_mm,
            "rotate_copies": self.rotate_copies,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircularArrayParams":
        center = data.get("center_mm", [0.0, 0.0])
        return cls(
            count=int(data.get("count", 6)),
            total_angle_deg=float(data.get("total_angle_deg", 360.0)),
            center_mm=(float(center[0]), float(center[1])),
            radius_mm=float(data.get("radius_mm", 10.0)),
            rotate_copies=bool(data.get("rotate_copies", True)),
        )


@dataclass
class ArrayParams:
    """Top-level parameters for an array operation."""

    mode: ArrayMode = ArrayMode.GRID
    grid: GridArrayParams = field(default_factory=GridArrayParams)
    point_rotation: PointRotationParams = field(
        default_factory=PointRotationParams
    )
    circular: CircularArrayParams = field(default_factory=CircularArrayParams)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "grid": self.grid.to_dict(),
            "point_rotation": self.point_rotation.to_dict(),
            "circular": self.circular.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ArrayParams":
        if not data:
            return cls()
        try:
            mode = ArrayMode(data.get("mode", "grid"))
        except ValueError:
            mode = ArrayMode.GRID
        return cls(
            mode=mode,
            grid=GridArrayParams.from_dict(data.get("grid", {})),
            point_rotation=PointRotationParams.from_dict(
                data.get("point_rotation", {})
            ),
            circular=CircularArrayParams.from_dict(data.get("circular", {})),
        )
