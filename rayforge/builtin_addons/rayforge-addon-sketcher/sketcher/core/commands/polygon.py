from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from gettext import gettext as _
from typing import TYPE_CHECKING

from raygeo.geo.types import Point as GeoPoint

from ..constraints import (
    AspectRatioConstraint,
    Constraint,
    EqualLengthConstraint,
    PointOnLineConstraint,
    RadiusConstraint,
)
from ..entities import Circle, Entity, Line, Point
from ..types import EntityID
from .base import PreviewState, SketchChangeCommand
from .dimension import DimensionData
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..registry import EntityRegistry
    from ..sketch import Sketch

MIN_SIDES = 3
MAX_SIDES = 100
DEFAULT_SIDES = 6
DEFAULT_STAR_RATIO = 0.5


def check_shape(sides: int, star_ratio: float | None) -> None:
    """Raises ValueError for a corner count or star ratio out of range."""
    if not MIN_SIDES <= sides <= MAX_SIDES:
        raise ValueError(
            f"A polygon needs {MIN_SIDES} to {MAX_SIDES} corners, got {sides}"
        )
    if star_ratio is not None and not 0.0 < star_ratio < 1.0:
        raise ValueError(f"Star ratio must be between 0 and 1: {star_ratio}")


def polygon_vertices(
    center: GeoPoint,
    first: GeoPoint,
    sides: int,
    star_ratio: float | None = None,
) -> list[GeoPoint]:
    """
    The corners of a regular polygon around ``center`` whose first
    corner is ``first``. A star alternates those outer corners with
    inner ones at ``star_ratio`` times the radius, halfway in between.
    """
    cx, cy = center
    radius = math.hypot(first[0] - cx, first[1] - cy)
    start = math.atan2(first[1] - cy, first[0] - cx)
    step = 2.0 * math.pi / sides
    vertices: list[GeoPoint] = []
    for k in range(sides):
        angle = start + k * step
        vertices.append(
            (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        )
        if star_ratio is not None:
            inner = radius * star_ratio
            angle += step / 2.0
            vertices.append(
                (cx + inner * math.cos(angle), cy + inner * math.sin(angle))
            )
    return vertices


def _ring(ids: list[EntityID]) -> Iterator[tuple[EntityID, EntityID]]:
    return zip(ids, ids[1:] + ids[:1])


class PolygonPreviewState(PreviewState):
    """Preview state for the polygon tool's 2-click workflow."""

    def __init__(
        self,
        center_id: EntityID,
        center_temp: bool,
        radius_id: EntityID,
        sides: int = DEFAULT_SIDES,
        star_ratio: float | None = None,
    ):
        self.center_id = center_id
        self.center_temp = center_temp
        self.radius_id = radius_id
        self.sides = sides
        self.star_ratio = star_ratio
        self.corner_ids: list[EntityID] = []
        self.line_ids: list[EntityID] = []
        self.locked_radius: float | None = None

    def get_preview_point_ids(self) -> set[EntityID]:
        """The preview's own points, which must not be snapped to."""
        return {self.radius_id, *self.corner_ids}

    def _vertices(self, registry: EntityRegistry) -> list[GeoPoint]:
        center = registry.get_point(self.center_id)
        first = registry.get_point(self.radius_id)
        return polygon_vertices(
            (center.x, center.y),
            (first.x, first.y),
            self.sides,
            self.star_ratio,
        )

    def _remove_outline(self, registry: EntityRegistry) -> None:
        corners = set(self.corner_ids)
        registry.remove_entities_by_id(self.line_ids)
        registry.remove_points(corners)
        self.line_ids = []
        self.corner_ids = []

    def rebuild(self, registry: EntityRegistry) -> None:
        """Recreates the corner points and lines for the current shape."""
        self._remove_outline(registry)
        vertices = self._vertices(registry)
        self.corner_ids = [registry.add_point(x, y) for x, y in vertices[1:]]
        ids = [self.radius_id, *self.corner_ids]
        self.line_ids = [registry.add_line(a, b) for a, b in _ring(ids)]

    def reposition(self, registry: EntityRegistry) -> None:
        """Moves the corner points to follow the radius point."""
        if len(self.corner_ids) + 1 != len(self._vertices(registry)):
            self.rebuild(registry)
            return
        for pid, (x, y) in zip(self.corner_ids, self._vertices(registry)[1:]):
            point = registry.get_point(pid)
            point.x, point.y = x, y

    def set_shape(
        self, registry: EntityRegistry, sides: int, star_ratio: float | None
    ) -> None:
        """Changes the corner count or star ratio of the preview."""
        check_shape(sides, star_ratio)
        self.sides = sides
        self.star_ratio = star_ratio
        self.rebuild(registry)

    def set_radius(self, registry: EntityRegistry, radius: float) -> None:
        """Locks the circumradius from numeric input."""
        self.locked_radius = radius
        center = registry.get_point(self.center_id)
        first = registry.get_point(self.radius_id)
        first.x, first.y = _at_radius(center, first.x, first.y, radius)
        self.reposition(registry)

    def get_dimensions(self, registry: EntityRegistry) -> list[DimensionData]:
        """Radius, corner count and star ratio, in dimension-field order."""
        try:
            center = registry.get_point(self.center_id)
            first = registry.get_point(self.radius_id)
        except IndexError:
            return []
        radius = math.hypot(first.x - center.x, first.y - center.y)
        ratio = "-" if self.star_ratio is None else f"{self.star_ratio:g}"
        return [
            DimensionData(
                label=f"R{DimensionData.format_length(radius)}",
                position=(
                    (center.x + first.x) / 2.0,
                    (center.y + first.y) / 2.0,
                ),
            ),
            DimensionData(
                label=f"{self.sides}", position=(center.x, center.y)
            ),
            DimensionData(label=ratio, position=(first.x, first.y)),
        ]


def _at_radius(center: Point, x: float, y: float, radius: float) -> GeoPoint:
    dx, dy = x - center.x, y - center.y
    length = math.hypot(dx, dy)
    if length < 1e-9:
        dx, dy, length = 1.0, 0.0, 1.0
    return (
        center.x + dx / length * radius,
        center.y + dy / length * radius,
    )


class PolygonCommand(SketchChangeCommand):
    """
    Creates a regular polygon or star as plain points and lines. The
    corners sit on construction circles around the center point and all
    edges have equal length, so the shape stays regular when edited.
    """

    def __init__(
        self,
        sketch: Sketch,
        center_id: EntityID,
        end_pos: GeoPoint,
        end_pid: EntityID | None = None,
        sides: int = DEFAULT_SIDES,
        star_ratio: float | None = None,
        is_center_temp: bool = False,
        fixed_radius: float | None = None,
    ):
        check_shape(sides, star_ratio)
        super().__init__(sketch, _("Add Polygon"))
        self.center_id = center_id
        self.end_pos = end_pos
        self.end_pid = end_pid if fixed_radius is None else None
        self.sides = sides
        self.star_ratio = star_ratio
        self.is_center_temp = is_center_temp
        self.fixed_radius = fixed_radius
        self.add_cmd: AddItemsCommand | None = None

    @staticmethod
    def start_preview(
        registry: EntityRegistry,
        x: float,
        y: float,
        snapped_pid: EntityID | None = None,
        sides: int = DEFAULT_SIDES,
        star_ratio: float | None = None,
        **kwargs,
    ) -> PolygonPreviewState:
        """Creates the preview: the center and the first corner."""
        if snapped_pid is not None:
            center_id, center_temp = snapped_pid, False
        else:
            center_id, center_temp = registry.add_point(x, y), True
        state = PolygonPreviewState(
            center_id,
            center_temp,
            registry.add_point(x, y),
            sides=sides,
            star_ratio=star_ratio,
        )
        state.rebuild(registry)
        return state

    @staticmethod
    def update_preview(
        registry: EntityRegistry,
        preview_state: PreviewState,
        x: float,
        y: float,
    ) -> None:
        """Moves the first corner, which sets radius and rotation."""
        if not isinstance(preview_state, PolygonPreviewState):
            raise TypeError("Expected PolygonPreviewState")
        try:
            center = registry.get_point(preview_state.center_id)
            first = registry.get_point(preview_state.radius_id)
        except IndexError:
            return
        if preview_state.locked_radius is not None:
            radius = preview_state.locked_radius
            first.x, first.y = _at_radius(center, x, y, radius)
        else:
            first.x, first.y = x, y
        preview_state.reposition(registry)

    @staticmethod
    def cleanup_preview(
        registry: EntityRegistry, preview_state: PreviewState
    ) -> None:
        """
        Removes the preview's corners and lines. Like the circle tool,
        it leaves a temporary center point for the tool to remove.
        """
        if not isinstance(preview_state, PolygonPreviewState):
            raise TypeError("Expected PolygonPreviewState")
        preview_state._remove_outline(registry)
        registry.remove_points([preview_state.radius_id])

    def _first_corner(self, center: Point) -> GeoPoint:
        x, y = self.end_pos
        if self.end_pid is not None:
            end = self.sketch.registry.get_point(self.end_pid)
            x, y = end.x, end.y
        if self.fixed_radius is not None:
            return _at_radius(center, x, y, self.fixed_radius)
        return x, y

    def _shape(
        self, center: Point, first: GeoPoint
    ) -> tuple[list[Point], list[Entity], list[Constraint]]:
        temp_ids = itertools.count(-2, -1)
        vertices = polygon_vertices(
            (center.x, center.y), first, self.sides, self.star_ratio
        )
        points = [Point(next(temp_ids), x, y) for x, y in vertices]
        ids = [p.id for p in points]
        if self.end_pid is not None:
            points.pop(0)
            ids[0] = self.end_pid
        lines = [Line(next(temp_ids), a, b) for a, b in _ring(ids)]
        stride = 1 if self.star_ratio is None else 2
        rings = [ids[0::stride]]
        if self.star_ratio is not None:
            rings.append(ids[1::2])
        circles = [
            Circle(next(temp_ids), center.id, ring[0], construction=True)
            for ring in rings
        ]
        constraints: list[Constraint] = [
            EqualLengthConstraint([line.id for line in lines])
        ]
        for circle, ring in zip(circles, rings):
            constraints += [
                PointOnLineConstraint(pid, circle.id) for pid in ring[1:]
            ]
        if self.star_ratio is not None:
            constraints.append(
                AspectRatioConstraint(
                    center.id, ids[1], center.id, ids[0], self.star_ratio
                )
            )
        if self.fixed_radius is not None:
            constraints.append(
                RadiusConstraint(circles[0].id, self.fixed_radius)
            )
        return points, [*lines, *circles], constraints

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._do_execute()
        registry = self.sketch.registry
        try:
            center = registry.get_point(self.center_id)
            first = self._first_corner(center)
        except IndexError:
            return
        radius = math.hypot(first[0] - center.x, first[1] - center.y)
        if radius < 1e-9 or self.end_pid == self.center_id:
            if self.is_center_temp:
                self.sketch.remove_point_if_unused(self.center_id)
            return
        points, entities, constraints = self._shape(center, first)
        if self.is_center_temp:
            registry.remove_points([center.id])
            points.append(center)
        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=points,
            entities=entities,
            constraints=constraints,
        )
        self.add_cmd._do_execute()

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
