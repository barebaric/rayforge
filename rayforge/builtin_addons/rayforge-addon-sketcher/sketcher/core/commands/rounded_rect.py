from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.geo.types import Point as GeoPoint
from raygeo.sketch.constraints.distance import DistanceConstraint
from raygeo.sketch.constraints.equal_distance import (
    EqualDistanceConstraint,
)
from raygeo.sketch.constraints.horizontal import HorizontalConstraint
from raygeo.sketch.constraints.tangent import TangentConstraint
from raygeo.sketch.constraints.vertical import VerticalConstraint
from raygeo.sketch.entities.arc import Arc

from .. import constraint_ops
from ..entity_ops import clone_entity
from ..types import EntityID
from .base import PreviewState, SketchChangeCommand
from .dimension import DimensionData
from .items import AddItemsCommand

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry

    from ..sketch import Sketch


class RoundedRectPreviewState(PreviewState):
    """Preview state for rounded rectangle tool's 2-click workflow."""

    def __init__(
        self,
        start_id: EntityID,
        start_temp: bool,
        p_end_id: EntityID,
        preview_ids: dict[str, EntityID],
        radius: float,
        center_on_start: bool = False,
    ):
        self.start_id = start_id
        self.start_temp = start_temp
        self.p_end_id = p_end_id
        self.preview_ids = preview_ids
        self.radius = radius
        self.locked_width: float | None = None
        self.locked_height: float | None = None
        self.locked_radius: float | None = None
        self.center_on_start = center_on_start
        self.constrain_square = False

    def get_preview_point_ids(self) -> set[EntityID]:
        """
        Returns IDs of temporary preview points that shouldn't be snapped to.

        Excludes the start point since that may be permanent.
        """
        result = {self.p_end_id}
        for key in ["t2", "t4", "t6", "t8", "c1", "c2", "c3", "c4"]:
            pid = self.preview_ids.get(key)
            if pid is not None:
                result.add(pid)
        return result

    def set_dimensions(
        self,
        registry: EntityRegistry,
        width: float | None = None,
        height: float | None = None,
        radius: float | None = None,
    ) -> None:
        """
        Sets the rounded rectangle dimensions from numeric input.

        Args:
            registry: The entity registry to modify.
            width: The width to apply (or None to keep current).
            height: The height to apply (or None to keep current).
            radius: The corner radius to apply (or None to keep current).
        """
        if width is not None:
            self.locked_width = width
        if height is not None:
            self.locked_height = height
        if radius is not None:
            self.locked_radius = radius
            self.radius = radius

        start_p = registry.get_point(self.start_id)
        end_p = registry.get_point(self.p_end_id)
        if start_p is None or end_p is None:
            return

        dx = end_p[0] - start_p[0]
        dy = end_p[1] - start_p[1]

        sign_x = 1.0 if dx >= 0 else -1.0
        sign_y = 1.0 if dy >= 0 else -1.0

        if self.center_on_start:
            new_dx = (
                self.locked_width / 2.0
                if self.locked_width is not None
                else abs(dx)
            )
            new_dy = (
                self.locked_height / 2.0
                if self.locked_height is not None
                else abs(dy)
            )
            end_pos = (
                start_p[0] + sign_x * new_dx,
                start_p[1] + sign_y * new_dy,
            )
        else:
            new_width = (
                self.locked_width if self.locked_width is not None else abs(dx)
            )
            new_height = (
                self.locked_height
                if self.locked_height is not None
                else abs(dy)
            )
            end_pos = (
                start_p[0] + sign_x * new_width,
                start_p[1] + sign_y * new_height,
            )
        registry.set_point_position(self.p_end_id, *end_pos)

        RoundedRectCommand.create_preview(
            registry,
            self.start_id,
            self.p_end_id,
            self.radius,
            preview_ids=self.preview_ids,
            center_on_start=self.center_on_start,
        )

    def get_dimensions(self, registry: EntityRegistry) -> list[DimensionData]:
        """
        Returns width, height and radius dimensions for preview.

        Args:
            registry: The entity registry to query for point positions.

        Returns:
            List containing DimensionData for width, height, and corner radius.
        """
        p1 = registry.get_point(self.start_id)
        p2 = registry.get_point(self.p_end_id)
        if p1 is None or p2 is None:
            return []
        if self.center_on_start:
            width = abs(p2[0] - p1[0]) * 2
            height = abs(p2[1] - p1[1]) * 2
            mid_x = p1[0]
            mid_y = p1[1]
            top_y = p1[1] - abs(p2[1] - p1[1])
            right_x = p1[0] + abs(p2[0] - p1[0])
        else:
            width = (
                self.locked_width
                if self.locked_width is not None
                else abs(p2[0] - p1[0])
            )
            height = (
                self.locked_height
                if self.locked_height is not None
                else abs(p2[1] - p1[1])
            )
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            top_y = min(p1[1], p2[1])
            right_x = max(p1[0], p2[0])
        dimensions = [
            DimensionData(
                label=DimensionData.format_length(width),
                position=(mid_x, top_y),
            ),
            DimensionData(
                label=DimensionData.format_length(height),
                position=(right_x, mid_y),
            ),
        ]
        if self.radius > 0:
            dimensions.append(
                DimensionData(
                    label=f"R{DimensionData.format_length(self.radius)}",
                    position=(p1[0] + self.radius / 2, p1[1]),
                )
            )
        return dimensions


class RoundedRectCommand(SketchChangeCommand):
    """A smart command to create a fully constrained rounded rectangle."""

    def __init__(
        self,
        sketch: Sketch,
        start_pid: EntityID,
        end_pos: GeoPoint,
        radius: float,
        is_start_temp: bool = False,
        fixed_width: float | None = None,
        fixed_height: float | None = None,
        fixed_radius: float | None = None,
        center_on_start: bool = False,
        constrain_square: bool = False,
    ):
        super().__init__(sketch, _("Add Rounded Rectangle"))
        self.start_pid = start_pid
        self.end_pos = end_pos
        self.radius = radius
        self.is_start_temp = is_start_temp
        self.fixed_width = fixed_width
        self.fixed_height = fixed_height
        self.fixed_radius = fixed_radius
        self.center_on_start = center_on_start
        self.constrain_square = constrain_square
        self.add_cmd: AddItemsCommand | None = None
        self._committed_end_id: EntityID | None = None

    @property
    def committed_end_id(self) -> EntityID | None:
        """
        The final end point ID after execute(), or None if not applicable.
        """
        return self._committed_end_id

    @staticmethod
    def _constrain_point_to_square(
        registry: EntityRegistry,
        start_id: EntityID,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """
        Adjusts a corner position so the shape defined by start_id and
        (x, y) is square, keeping the start point fixed.

        The smaller extent wins; the direction of each axis is kept.
        """
        start_p = registry.get_point(start_id)
        if start_p is None:
            return x, y

        size = min(abs(x - start_p[0]), abs(y - start_p[1]))
        sign_x = 1.0 if x >= start_p[0] else -1.0
        sign_y = 1.0 if y >= start_p[1] else -1.0
        return start_p[0] + sign_x * size, start_p[1] + sign_y * size

    @staticmethod
    @staticmethod
    def calculate_geometry(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        radius: float,
        fixed_width: float | None = None,
        fixed_height: float | None = None,
        fixed_radius: float | None = None,
        center_on_start: bool = False,
        constrain_square: bool = False,
    ) -> dict[str, Any] | None:
        """Pure geometry of the rounded rectangle: points, line/arc
        entities and constraints with temporary sequential point ids.
        Returns None for a degenerate rectangle."""
        from raygeo.sketch.constraints.equal_length import (
            EqualLengthConstraint,
        )
        from raygeo.sketch.constraints.radius import RadiusConstraint
        from raygeo.sketch.entities.line import Line
        from raygeo.sketch.entities.point import SketchPoint

        if constrain_square:
            size = min(abs(x2 - x1), abs(y2 - y1))
            sign_x = 1.0 if x2 >= x1 else -1.0
            sign_y = 1.0 if y2 >= y1 else -1.0
            x2 = x1 + sign_x * size
            y2 = y1 + sign_y * size

        if center_on_start:
            x1, y1 = x1 - (x2 - x1), y1 - (y2 - y1)

        width, height = abs(x2 - x1), abs(y2 - y1)
        if width < 1e-6 or height < 1e-6:
            return None

        radius = min(radius, width / 2.0, height / 2.0)
        sx, sy = (1 if x2 > x1 else -1), (1 if y2 > y1 else -1)

        coords = {
            "t1": (x1 + sx * radius, y1),
            "t2": (x2 - sx * radius, y1),
            "t3": (x2, y1 + sy * radius),
            "t4": (x2, y2 - sy * radius),
            "t5": (x2 - sx * radius, y2),
            "t6": (x1 + sx * radius, y2),
            "t7": (x1, y2 - sy * radius),
            "t8": (x1, y1 + sy * radius),
            "c1": (x1 + sx * radius, y1 + sy * radius),
            "c2": (x2 - sx * radius, y1 + sy * radius),
            "c3": (x2 - sx * radius, y2 - sy * radius),
            "c4": (x1 + sx * radius, y2 - sy * radius),
        }
        names = list(coords)
        pid = {name: i for i, name in enumerate(names)}
        points = [
            SketchPoint(pid[n], coords[n][0], coords[n][1]) for n in names
        ]

        is_cw = sx * sy < 0
        lines = [
            Line(pid["t1"], pid["t2"]),
            Line(pid["t3"], pid["t4"]),
            Line(pid["t5"], pid["t6"]),
            Line(pid["t7"], pid["t8"]),
        ]
        arcs = [
            Arc(pid["t8"], pid["t1"], pid["c1"], clockwise=is_cw),
            Arc(pid["t2"], pid["t3"], pid["c2"], clockwise=is_cw),
            Arc(pid["t4"], pid["t5"], pid["c3"], clockwise=is_cw),
            Arc(pid["t6"], pid["t7"], pid["c4"], clockwise=is_cw),
        ]
        entities = lines + arcs

        constraints: list[Any] = [
            HorizontalConstraint(pid["t1"], pid["t2"]),
            VerticalConstraint(pid["t3"], pid["t4"]),
            HorizontalConstraint(pid["t5"], pid["t6"]),
            VerticalConstraint(pid["t7"], pid["t8"]),
        ]
        for li, ai in [
            (0, 0),
            (3, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 3),
        ]:
            ln, arc = lines[li], arcs[ai]
            constraints.append(
                TangentConstraint(
                    ln.p1_idx, ln.p2_idx, arc.center_idx, arc.start_idx
                )
            )
        constraints.append(
            EqualLengthConstraint(
                [
                    (pid["c1"], pid["t8"], pid["c2"], pid["t2"]),
                    (pid["c1"], pid["t8"], pid["c3"], pid["t4"]),
                    (pid["c1"], pid["t8"], pid["c4"], pid["t6"]),
                ]
            )
        )
        constraints += [
            EqualDistanceConstraint(
                pid["c1"], pid["t8"], pid["c1"], pid["t1"]
            ),
            EqualDistanceConstraint(
                pid["c2"], pid["t2"], pid["c2"], pid["t3"]
            ),
            EqualDistanceConstraint(
                pid["c3"], pid["t4"], pid["c3"], pid["t5"]
            ),
            EqualDistanceConstraint(
                pid["c4"], pid["t6"], pid["c4"], pid["t7"]
            ),
        ]

        top_edge_y = min(y1, y2)
        right_edge_x = max(x1, x2)
        if top_edge_y == y1:
            top_edge_p1, top_edge_p2 = pid["t1"], pid["t2"]
        else:
            top_edge_p1, top_edge_p2 = pid["t5"], pid["t6"]
        if right_edge_x == x2:
            right_edge_p1, right_edge_p2 = pid["t3"], pid["t4"]
        else:
            right_edge_p1, right_edge_p2 = pid["t7"], pid["t8"]
        if fixed_width is not None:
            constraints.append(
                DistanceConstraint(top_edge_p1, top_edge_p2, fixed_width)
            )
        if fixed_height is not None:
            constraints.append(
                DistanceConstraint(right_edge_p1, right_edge_p2, fixed_height)
            )
        if fixed_radius is not None:
            constraints.append(
                RadiusConstraint(pid["c1"], pid["t8"], fixed_radius)
            )

        return {
            "points": points,
            "entities": entities,
            "constraints": constraints,
        }

    @staticmethod
    def create_preview(
        registry: EntityRegistry,
        start_pid: EntityID,
        end_pid: EntityID,
        radius: float,
        preview_ids: dict[str, EntityID] | None = None,
        center_on_start: bool = False,
    ) -> dict[str, EntityID] | None:
        """
        Creates or updates preview geometry in the registry.

        Args:
            registry: The entity registry to modify.
            start_pid: The ID of the start corner point.
            end_pid: The ID of the end corner point (preview corner).
            radius: The corner radius.
            preview_ids: Existing preview IDs to update, or None to create new.
            center_on_start: When True, the start point is the shape's
                center and the end point defines the extent; the actual
                corners are mirrored through the start point.

        Returns:
            Dict of preview IDs, or None if geometry is invalid.
        """
        start_p = registry.get_point(start_pid)
        end_p = registry.get_point(end_pid)
        if start_p is None or end_p is None:
            return None

        if center_on_start:
            x1, y1 = 2.0 * start_p[0] - end_p[0], 2.0 * start_p[1] - end_p[1]
            x2, y2 = end_p[0], end_p[1]
        else:
            x1, y1 = start_p[0], start_p[1]
            x2, y2 = end_p[0], end_p[1]
        width, height = abs(x2 - x1), abs(y2 - y1)

        if width > 1e-6 and height > 1e-6:
            radius = min(radius, width / 2.0, height / 2.0)
        else:
            radius = 0.0

        sx, sy = (1 if x2 > x1 else -1), (1 if y2 > y1 else -1)
        is_cw = sx * sy < 0

        coords = {
            "t1": (x1 + sx * radius, y1),
            "t2": (x2 - sx * radius, y1),
            "t3": (x2, y1 + sy * radius),
            "t4": (x2, y2 - sy * radius),
            "t5": (x2 - sx * radius, y2),
            "t6": (x1 + sx * radius, y2),
            "t7": (x1, y2 - sy * radius),
            "t8": (x1, y1 + sy * radius),
            "c1": (x1 + sx * radius, y1 + sy * radius),
            "c2": (x2 - sx * radius, y1 + sy * radius),
            "c3": (x2 - sx * radius, y2 - sy * radius),
            "c4": (x1 + sx * radius, y2 - sy * radius),
        }

        if preview_ids is None:
            # Create new preview geometry
            preview_ids = {}

            # Create all points
            for name, (px, py) in coords.items():
                preview_ids[name] = registry.add_point(px, py)

            # Lines
            preview_ids["line1"] = registry.add_line(
                preview_ids["t1"], preview_ids["t2"]
            )
            preview_ids["line2"] = registry.add_line(
                preview_ids["t3"], preview_ids["t4"]
            )
            preview_ids["line3"] = registry.add_line(
                preview_ids["t5"], preview_ids["t6"]
            )
            preview_ids["line4"] = registry.add_line(
                preview_ids["t7"], preview_ids["t8"]
            )

            # Arcs
            preview_ids["arc1"] = registry.add_arc(
                preview_ids["t8"],
                preview_ids["t1"],
                preview_ids["c1"],
                clockwise=is_cw,
            )
            preview_ids["arc2"] = registry.add_arc(
                preview_ids["t2"],
                preview_ids["t3"],
                preview_ids["c2"],
                clockwise=is_cw,
            )
            preview_ids["arc3"] = registry.add_arc(
                preview_ids["t4"],
                preview_ids["t5"],
                preview_ids["c3"],
                clockwise=is_cw,
            )
            preview_ids["arc4"] = registry.add_arc(
                preview_ids["t6"],
                preview_ids["t7"],
                preview_ids["c4"],
                clockwise=is_cw,
            )
        else:
            # Update existing preview geometry
            for name, (px, py) in coords.items():
                registry.set_point_position(preview_ids[name], px, py)

            # Update arc directions
            for key in ["arc1", "arc2", "arc3", "arc4"]:
                arc_entity = registry.get_entity_object(preview_ids[key])
                if isinstance(arc_entity, Arc):
                    registry.set_state(preview_ids[key], {"clockwise": is_cw})

        return preview_ids

    @staticmethod
    def start_preview(
        registry: EntityRegistry,
        x: float,
        y: float,
        snapped_pid: EntityID | None = None,
        radius: float = 10.0,
        center_on_start: bool = False,
        **kwargs,
    ) -> RoundedRectPreviewState:
        """
        Creates initial preview state with start and end points.

        Args:
            registry: The entity registry to modify.
            x, y: The initial coordinates.
            snapped_pid: An existing point ID to snap to, or None.
            radius: The corner radius.
            center_on_start: When True, the start point is the shape's
                center and the end point defines the extent.

        Returns:
            RoundedRectPreviewState for use with update_preview and
            cleanup_preview.
        """
        if snapped_pid is not None:
            start_id = snapped_pid
            start_temp = False
        else:
            start_id = registry.add_point(x, y)
            start_temp = True

        p_end_id = registry.add_point(x, y)

        preview_ids = RoundedRectCommand.create_preview(
            registry, start_id, p_end_id, radius
        )
        assert preview_ids is not None

        return RoundedRectPreviewState(
            start_id=start_id,
            start_temp=start_temp,
            p_end_id=p_end_id,
            preview_ids=preview_ids,
            radius=radius,
            center_on_start=center_on_start,
        )

    @staticmethod
    def update_preview(
        registry: EntityRegistry,
        preview_state: PreviewState,
        x: float,
        y: float,
        center_on_start: bool = False,
        constrain_square: bool = False,
    ) -> None:
        """
        Updates the end point position and refreshes preview geometry.

        Args:
            registry: The entity registry to modify.
            preview_state: The preview state from start_preview.
            x, y: The new end point coordinates.
            center_on_start: When True, the start point is the shape's
                center and the end point defines the extent.
            constrain_square: When True the end point is adjusted so
                both extents are equal, keeping the start point fixed.

        Raises:
            TypeError: If preview_state is not a RoundedRectPreviewState.
        """
        if not isinstance(preview_state, RoundedRectPreviewState):
            raise TypeError("Expected RoundedRectPreviewState")

        if constrain_square:
            x, y = RoundedRectCommand._constrain_point_to_square(
                registry, preview_state.start_id, x, y
            )
        preview_state.constrain_square = constrain_square
        preview_state.center_on_start = center_on_start

        p_end = registry.get_point(preview_state.p_end_id)
        p_start = registry.get_point(preview_state.start_id)
        if p_end is None or p_start is None:
            return

        if (
            preview_state.locked_width is not None
            or preview_state.locked_height is not None
        ):
            dx = p_end[0] - p_start[0]
            dy = p_end[1] - p_start[1]
            sign_x = 1.0 if dx >= 0 else -1.0
            sign_y = 1.0 if dy >= 0 else -1.0

            half = 2.0 if center_on_start else 1.0
            end_x = (
                p_start[0] + sign_x * preview_state.locked_width / half
                if preview_state.locked_width is not None
                else x
            )
            end_y = (
                p_start[1] + sign_y * preview_state.locked_height / half
                if preview_state.locked_height is not None
                else y
            )
        else:
            end_x, end_y = x, y
        registry.set_point_position(preview_state.p_end_id, end_x, end_y)

        RoundedRectCommand.create_preview(
            registry,
            preview_state.start_id,
            preview_state.p_end_id,
            preview_state.radius,
            preview_ids=preview_state.preview_ids,
            center_on_start=center_on_start,
        )

    @staticmethod
    def cleanup_preview(
        registry: EntityRegistry, preview_state: PreviewState
    ) -> None:
        """
        Removes all preview entities and points from the registry.

        Args:
            registry: The entity registry to modify.
            preview_state: The preview state from start_preview.

        Raises:
            TypeError: If preview_state is not a RoundedRectPreviewState.
        """
        if not isinstance(preview_state, RoundedRectPreviewState):
            raise TypeError("Expected RoundedRectPreviewState")
        preview_ids = preview_state.preview_ids
        p_end_id = preview_state.p_end_id

        point_ids = set(preview_ids.values())
        point_ids.add(p_end_id)

        entity_ids_to_remove = {
            eid
            for eid in registry.entity_ids()
            if any(
                pid in point_ids
                for pid in registry.entity_point_ids(eid) or []
            )
        }
        registry.remove_entities(list(entity_ids_to_remove))

        registry.remove_points(list(point_ids))

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._redo()

        reg = self.sketch.registry
        start_p = reg.get_point(self.start_pid)
        if start_p is None:
            return

        result = self.calculate_geometry(
            start_p[0],
            start_p[1],
            self.end_pos[0],
            self.end_pos[1],
            self.radius,
            fixed_width=self.fixed_width,
            fixed_height=self.fixed_height,
            fixed_radius=self.fixed_radius,
            center_on_start=self.center_on_start,
            constrain_square=self.constrain_square,
        )
        if not result:
            if self.is_start_temp:
                self.sketch.remove_point_if_unused(self.start_pid)
            return

        add_cmd = AddItemsCommand(self.sketch, "")
        pid_map = {
            point.id: add_cmd.add_point(point.x, point.y)
            for point in result["points"]
        }
        for entity in result["entities"]:
            add_cmd.add_entity(clone_entity(reg, entity, pid_map))
        for constraint in result["constraints"]:
            add_cmd.add_constraint(constraint_ops.rebuild(constraint, pid_map))

        # Unlike Rectangle, RoundedRect doesn't use the start_pid in
        # its final geometry, so a temporary start point is removed.
        if self.is_start_temp:
            reg.remove_points([self.start_pid])

        self.add_cmd = add_cmd
        self._committed_end_id = None

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
