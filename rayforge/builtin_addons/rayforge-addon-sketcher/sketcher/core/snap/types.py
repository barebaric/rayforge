from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any

from rayforge.core.geo.types import Point as GeoPoint
from rayforge.shared.util.colors import ColorRGBA
from ..types import EntityID


class SnapLineType(Enum):
    ENTITY_POINT = auto()
    ON_ENTITY = auto()
    INTERSECTION = auto()
    MIDPOINT = auto()
    EQUIDISTANT = auto()
    TANGENT = auto()
    CENTER = auto()

    @property
    def priority(self) -> int:
        priorities: Dict[SnapLineType, int] = {
            SnapLineType.ENTITY_POINT: 100,
            SnapLineType.MIDPOINT: 90,
            SnapLineType.ON_ENTITY: 80,
            SnapLineType.INTERSECTION: 70,
            SnapLineType.EQUIDISTANT: 60,
            SnapLineType.TANGENT: 40,
            SnapLineType.CENTER: 30,
        }
        return priorities.get(self, 0)


@dataclass(frozen=True)
class SnapLineStyle:
    color: ColorRGBA = (0.0, 0.6, 1.0, 0.8)
    dash: Optional[Tuple[float, ...]] = None
    line_width: float = 1.0


SNAP_LINE_STYLES: Dict[SnapLineType, SnapLineStyle] = {
    SnapLineType.ENTITY_POINT: SnapLineStyle(
        color=(0.2, 0.6, 1.0, 0.9),
        dash=(8, 4),
        line_width=2.0,
    ),
    SnapLineType.ON_ENTITY: SnapLineStyle(
        color=(1.0, 0.4, 0.8, 0.9),
        dash=(8, 4),
        line_width=2.0,
    ),
    SnapLineType.INTERSECTION: SnapLineStyle(
        color=(1.0, 0.2, 0.8, 0.9),
        dash=(8, 4),
        line_width=2.0,
    ),
    SnapLineType.MIDPOINT: SnapLineStyle(
        color=(0.2, 0.9, 0.3, 0.9),
        dash=(8, 4),
        line_width=2.0,
    ),
    SnapLineType.EQUIDISTANT: SnapLineStyle(
        color=(1.0, 0.6, 0.2, 0.9),
        dash=(8, 4),
        line_width=2.0,
    ),
    SnapLineType.TANGENT: SnapLineStyle(
        color=(0.7, 0.3, 0.9, 0.9),
        dash=(8, 4),
        line_width=2.0,
    ),
    SnapLineType.CENTER: SnapLineStyle(
        color=(1.0, 0.3, 0.3, 0.9),
        dash=(8, 4),
        line_width=2.0,
    ),
}


@dataclass(frozen=True)
class SnapPoint:
    x: float
    y: float
    line_type: SnapLineType
    source: Optional[Any] = None
    spacing: Optional[float] = None
    is_horizontal: bool = False
    pattern_coords: Optional[Tuple[float, ...]] = None
    axis_coord: Optional[float] = None

    @property
    def pos(self) -> GeoPoint:
        return (self.x, self.y)


@dataclass(frozen=True)
class SnapLine:
    is_horizontal: bool
    coordinate: float
    line_type: SnapLineType
    source: Optional[Any] = None

    @property
    def style(self) -> SnapLineStyle:
        return SNAP_LINE_STYLES.get(self.line_type, SnapLineStyle())

    def distance_to(self, x: float, y: float) -> float:
        if self.is_horizontal:
            return abs(y - self.coordinate)
        else:
            return abs(x - self.coordinate)

    def get_snap_position(self, x: float, y: float) -> GeoPoint:
        if self.is_horizontal:
            return (x, self.coordinate)
        else:
            return (self.coordinate, y)


@dataclass
class SnapResult:
    snapped: bool = False
    position: GeoPoint = (0.0, 0.0)
    snap_lines: List[SnapLine] = field(default_factory=list)
    snap_points: List[SnapPoint] = field(default_factory=list)
    primary_snap_line: Optional[SnapLine] = None
    primary_snap_point: Optional[SnapPoint] = None
    distance: float = float("inf")

    @classmethod
    def no_snap(cls, position: GeoPoint) -> "SnapResult":
        return cls(snapped=False, position=position)

    @classmethod
    def from_snap_line(
        cls,
        snap_line: SnapLine,
        original_pos: GeoPoint,
        distance: float,
    ) -> "SnapResult":
        snapped_pos = snap_line.get_snap_position(*original_pos)
        return cls(
            snapped=True,
            position=snapped_pos,
            snap_lines=[snap_line],
            primary_snap_line=snap_line,
            distance=distance,
        )

    @classmethod
    def from_snap_point(
        cls,
        snap_point: SnapPoint,
        distance: float,
        snap_lines: Optional[List[SnapLine]] = None,
    ) -> "SnapResult":
        return cls(
            snapped=True,
            position=snap_point.pos,
            snap_points=[snap_point],
            snap_lines=snap_lines or [],
            primary_snap_point=snap_point,
            distance=distance,
        )


class DragContext:
    def __init__(
        self,
        dragged_point_ids: Optional[Set[EntityID]] = None,
        dragged_entity_ids: Optional[Set[EntityID]] = None,
        initial_positions: Optional[Dict[EntityID, GeoPoint]] = None,
    ):
        self.dragged_point_ids: Set[EntityID] = dragged_point_ids or set()
        self.dragged_entity_ids: Set[EntityID] = dragged_entity_ids or set()
        self.initial_positions: Dict[EntityID, GeoPoint] = (
            initial_positions or {}
        )

    def is_point_dragged(self, point_id: EntityID) -> bool:
        return point_id in self.dragged_point_ids

    def is_entity_dragged(self, entity_id: EntityID) -> bool:
        return entity_id in self.dragged_entity_ids
