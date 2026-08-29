from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.geo.types import Point as GeoPoint

from ..constraints import (
    DistanceConstraint,
    PerpendicularConstraint,
    SymmetryConstraint,
)
from ..entities import Line, Point
from ..types import EntityID
from .base import PreviewState, SketchChangeCommand
from .dimension import DimensionData
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..registry import EntityRegistry
    from ..sketch import Sketch


class RectanglePreviewState(PreviewState):
    """Preview state for rectangle tool's 2-click workflow."""

    def __init__(
        self,
        start_id: EntityID,
        start_temp: bool,
        p_end_id: EntityID,
        preview_ids: dict[str, EntityID],
        center_on_start: bool = False,
    ):
        self.start_id = start_id
        self.start_temp = start_temp
        self.p_end_id = p_end_id
        self.preview_ids = preview_ids
        self.locked_width: float | None = None
        self.locked_height: float | None = None
        self.center_on_start = center_on_start
        self.constrain_square = False

    def get_preview_point_ids(self) -> set[EntityID]:
        """
        Returns IDs of temporary preview points that shouldn't be snapped to.

        Excludes the start point since that may be permanent.
        """
        result = {self.p_end_id}
        for key in ["p2", "p4", "center", "c1", "c2"]:
            pid = self.preview_ids.get(key)
            if pid is not None:
                result.add(pid)
        return result

    def set_dimensions(
        self,
        registry: EntityRegistry,
        width: float | None = None,
        height: float | None = None,
    ) -> None:
        """
        Sets the rectangle dimensions from numeric input.

        Args:
            registry: The entity registry to modify.
            width: The width to apply (or None to keep current).
            height: The height to apply (or None to keep current).
        """
        if width is not None:
            self.locked_width = width
        if height is not None:
            self.locked_height = height

        try:
            start_p = registry.get_point(self.start_id)
            end_p = registry.get_point(self.p_end_id)
        except IndexError:
            return

        dx = end_p.x - start_p.x
        dy = end_p.y - start_p.y

        sign_x = 1.0 if dx >= 0 else -1.0
        sign_y = 1.0 if dy >= 0 else -1.0

        if self.center_on_start:
            # locked width/height are the full rectangle dimensions;
            # the end point defines a half-extent from the center.
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
            end_p.x = start_p.x + sign_x * new_dx
            end_p.y = start_p.y + sign_y * new_dy
        else:
            new_width = (
                self.locked_width if self.locked_width is not None else abs(dx)
            )
            new_height = (
                self.locked_height
                if self.locked_height is not None
                else abs(dy)
            )
            end_p.x = start_p.x + sign_x * new_width
            end_p.y = start_p.y + sign_y * new_height

        RectangleCommand.create_preview(
            registry,
            self.start_id,
            self.p_end_id,
            preview_ids=self.preview_ids,
            center_on_start=self.center_on_start,
        )

    def get_dimensions(self, registry: EntityRegistry) -> list[DimensionData]:
        """
        Returns width and height dimensions for preview.

        Args:
            registry: The entity registry to query for point positions.

        Returns:
            List containing DimensionData for width and height.
        """
        try:
            p1 = registry.get_point(self.start_id)
            p2 = registry.get_point(self.p_end_id)
        except IndexError:
            return []
        if self.center_on_start:
            width = abs(p2.x - p1.x) * 2
            height = abs(p2.y - p1.y) * 2
            mid_x = p1.x
            mid_y = p1.y
            top_y = p1.y - abs(p2.y - p1.y)
            right_x = p1.x + abs(p2.x - p1.x)
        else:
            width = abs(p2.x - p1.x)
            height = abs(p2.y - p1.y)
            mid_x = (p1.x + p2.x) / 2
            mid_y = (p1.y + p2.y) / 2
            top_y = min(p1.y, p2.y)
            right_x = max(p1.x, p2.x)
        return [
            DimensionData(
                label=DimensionData.format_length(width),
                position=(mid_x, top_y),
            ),
            DimensionData(
                label=DimensionData.format_length(height),
                position=(right_x, mid_y),
            ),
        ]


class RectangleCommand(SketchChangeCommand):
    """A smart command to create a fully constrained rectangle."""

    def __init__(
        self,
        sketch: Sketch,
        start_pid: EntityID,
        end_pos: GeoPoint,
        end_pid: EntityID | None = None,
        is_start_temp: bool = False,
        fixed_width: float | None = None,
        fixed_height: float | None = None,
        center_on_start: bool = False,
        constrain_square: bool = False,
    ):
        super().__init__(sketch, _("Add Rectangle"))
        self.start_pid = start_pid
        self.end_pos = end_pos
        self.end_pid = end_pid
        self.is_start_temp = is_start_temp
        self.fixed_width = fixed_width
        self.fixed_height = fixed_height
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
        Adjusts a corner position so the rectangle defined by start_id
        and (x, y) is square, keeping the start point fixed.

        The smaller extent wins; the direction of each axis is kept.
        """
        try:
            start_p = registry.get_point(start_id)
        except IndexError:
            return x, y

        size = min(abs(x - start_p.x), abs(y - start_p.y))
        sign_x = 1.0 if x >= start_p.x else -1.0
        sign_y = 1.0 if y >= start_p.y else -1.0
        return start_p.x + sign_x * size, start_p.y + sign_y * size

    @staticmethod
    def _calculate_rect_corners(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        center_on_start: bool,
        constrain_square: bool = False,
    ) -> tuple[float, float, float, float]:
        """
        Returns the diagonal corner coordinates (ax1, ay1, ax2, ay2) of
        the rectangle to create.

        When center_on_start is False (default) the start point (x1, y1)
        and the mouse point (x2, y2) are opposite corners.

        When center_on_start is True the start point is the rectangle's
        center and the mouse point defines the half-extent, so the
        rectangle is drawn symmetrically around the start point.

        When constrain_square is True the mouse point is adjusted so
        both extents are equal, keeping the start point fixed.
        """
        if constrain_square:
            size = min(abs(x2 - x1), abs(y2 - y1))
            sign_x = 1.0 if x2 >= x1 else -1.0
            sign_y = 1.0 if y2 >= y1 else -1.0
            x2 = x1 + sign_x * size
            y2 = y1 + sign_y * size

        if center_on_start:
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            return x1 - dx, y1 - dy, x1 + dx, y1 + dy
        return x1, y1, x2, y2

    @staticmethod
    def calculate_geometry(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        start_pid: EntityID,
        end_pid: EntityID | None,
        fixed_width: float | None = None,
        fixed_height: float | None = None,
        center_on_start: bool = False,
        constrain_square: bool = False,
    ) -> dict[str, Any] | None:
        """Calculates the points, entities, and constraints for a
        rectangle, including its auto-created center point.

        Square constraining is skipped when the end corner is a snapped
        point, since the coincidence then takes precedence."""
        ax1, ay1, ax2, ay2 = RectangleCommand._calculate_rect_corners(
            x1,
            y1,
            x2,
            y2,
            center_on_start,
            constrain_square=constrain_square and end_pid is None,
        )

        if abs(ax2 - ax1) < 1e-6 or abs(ay2 - ay1) < 1e-6:
            return None

        temp_id_counter = -1

        def next_temp_id():
            nonlocal temp_id_counter
            temp_id_counter -= 1
            return temp_id_counter

        p3_id = end_pid if end_pid is not None else next_temp_id()

        if center_on_start:
            # The start point is the rectangle's center, so the first
            # corner is a new point and the start point itself acts as
            # the symmetry center.
            p1 = Point(next_temp_id(), ax1, ay1)
            p1_id = p1.id
            center_id = start_pid
            center_point = None
        else:
            p1 = None
            p1_id = start_pid
            center_point = Point(
                next_temp_id(), (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
            )
            center_id = center_point.id

        points = {
            "p1_id": p1_id,
            "p1": p1,
            "p2": Point(next_temp_id(), ax2, ay1),
            "p3": Point(p3_id, ax2, ay2),
            "p4": Point(next_temp_id(), ax1, ay2),
            "center": center_point,
            "center_id": center_id,
        }

        entities = [
            Line(next_temp_id(), points["p1_id"], points["p2"].id),
            Line(next_temp_id(), points["p2"].id, points["p3"].id),
            Line(next_temp_id(), points["p3"].id, points["p4"].id),
            Line(next_temp_id(), points["p4"].id, points["p1_id"]),
        ]

        constraints: list[Any] = [
            PerpendicularConstraint(entities[0].id, entities[1].id),
            PerpendicularConstraint(entities[1].id, entities[2].id),
            PerpendicularConstraint(entities[2].id, entities[3].id),
            PerpendicularConstraint(entities[3].id, entities[0].id),
            SymmetryConstraint(
                points["p1_id"],
                points["p3"].id,
                center=center_id,
                user_visible=False,
            ),
        ]

        if fixed_width is not None:
            constraints.append(
                DistanceConstraint(
                    points["p1_id"], points["p2"].id, fixed_width
                )
            )

        if fixed_height is not None:
            constraints.append(
                DistanceConstraint(
                    points["p2"].id, points["p3"].id, fixed_height
                )
            )

        return {
            "points": points,
            "entities": entities,
            "constraints": constraints,
        }

    @staticmethod
    def _create_corner_preview(
        registry: EntityRegistry,
        start_pid: EntityID,
        end_pid: EntityID,
    ) -> dict[str, EntityID]:
        """
        Creates corner-mode preview geometry from scratch: two derived
        corner points (p2, p4), a center point, and four lines using
        the start/end points as the other two corners.
        """
        start_p = registry.get_point(start_pid)
        end_p = registry.get_point(end_pid)

        coords = {
            "p2": (end_p.x, start_p.y),
            "p4": (start_p.x, end_p.y),
            "center": (
                (start_p.x + end_p.x) / 2.0,
                (start_p.y + end_p.y) / 2.0,
            ),
        }
        preview_ids: dict[str, EntityID] = {}
        for name, (px, py) in coords.items():
            preview_ids[name] = registry.add_point(px, py)

        preview_ids["line1"] = registry.add_line(start_pid, preview_ids["p2"])
        preview_ids["line2"] = registry.add_line(preview_ids["p2"], end_pid)
        preview_ids["line3"] = registry.add_line(end_pid, preview_ids["p4"])
        preview_ids["line4"] = registry.add_line(preview_ids["p4"], start_pid)
        return preview_ids

    @staticmethod
    def _update_corner_preview(
        registry: EntityRegistry,
        preview_ids: dict[str, EntityID],
        start_pid: EntityID,
        end_pid: EntityID,
    ) -> None:
        """Updates positions of existing corner-mode preview points."""
        start_p = registry.get_point(start_pid)
        end_p = registry.get_point(end_pid)
        coords = {
            "p2": (end_p.x, start_p.y),
            "p4": (start_p.x, end_p.y),
            "center": (
                (start_p.x + end_p.x) / 2.0,
                (start_p.y + end_p.y) / 2.0,
            ),
        }
        for name, (px, py) in coords.items():
            p = registry.get_point(preview_ids[name])
            p.x, p.y = px, py

    @staticmethod
    def _create_centered_preview(
        registry: EntityRegistry,
        start_pid: EntityID,
        end_pid: EntityID,
    ) -> dict[str, EntityID]:
        """
        Creates center-on-start preview geometry from scratch: four
        corner points arranged symmetrically around the start point,
        plus a center point coincident with the start point's
        coordinates, and four lines forming the outline.
        """
        start_p = registry.get_point(start_pid)
        end_p = registry.get_point(end_pid)
        dx = abs(end_p.x - start_p.x)
        dy = abs(end_p.y - start_p.y)

        coords = {
            "c1": (start_p.x - dx, start_p.y - dy),
            "p2": (start_p.x + dx, start_p.y - dy),
            "c2": (start_p.x + dx, start_p.y + dy),
            "p4": (start_p.x - dx, start_p.y + dy),
            "center": (start_p.x, start_p.y),
        }
        preview_ids: dict[str, EntityID] = {}
        for name, (px, py) in coords.items():
            preview_ids[name] = registry.add_point(px, py)

        loop = ["c1", "p2", "c2", "p4"]
        loop_ids = [preview_ids[n] for n in loop]
        for i in range(4):
            preview_ids[f"line{i + 1}"] = registry.add_line(
                loop_ids[i], loop_ids[(i + 1) % 4]
            )
        return preview_ids

    @staticmethod
    def _update_centered_preview(
        registry: EntityRegistry,
        preview_ids: dict[str, EntityID],
        start_pid: EntityID,
        end_pid: EntityID,
    ) -> None:
        """Updates positions of existing center-on-start preview points."""
        start_p = registry.get_point(start_pid)
        end_p = registry.get_point(end_pid)
        dx = abs(end_p.x - start_p.x)
        dy = abs(end_p.y - start_p.y)

        coords = {
            "c1": (start_p.x - dx, start_p.y - dy),
            "p2": (start_p.x + dx, start_p.y - dy),
            "c2": (start_p.x + dx, start_p.y + dy),
            "p4": (start_p.x - dx, start_p.y + dy),
            "center": (start_p.x, start_p.y),
        }
        for name, (px, py) in coords.items():
            p = registry.get_point(preview_ids[name])
            p.x, p.y = px, py

    @staticmethod
    def create_preview(
        registry: EntityRegistry,
        start_pid: EntityID,
        end_pid: EntityID,
        preview_ids: dict[str, EntityID] | None = None,
        center_on_start: bool = False,
    ) -> dict[str, EntityID] | None:
        """
        Creates or updates preview geometry in the registry.

        Args:
            registry: The entity registry to modify.
            start_pid: The ID of the start point (a corner, or the center
                when center_on_start is True).
            end_pid: The ID of the end point (the opposite corner, or a
                corner defining the half-extent when center_on_start is
                True).
            preview_ids: Existing preview IDs to update, or None to create new.
            center_on_start: When True, the start point is the rectangle's
                center and the end point defines the half-extent.

        Returns:
            Dict of preview IDs, or None if geometry is invalid.
        """
        try:
            registry.get_point(start_pid)
            registry.get_point(end_pid)
        except IndexError:
            return None

        if center_on_start:
            if preview_ids is None:
                return RectangleCommand._create_centered_preview(
                    registry, start_pid, end_pid
                )
            RectangleCommand._update_centered_preview(
                registry, preview_ids, start_pid, end_pid
            )
        else:
            if preview_ids is None:
                return RectangleCommand._create_corner_preview(
                    registry, start_pid, end_pid
                )
            RectangleCommand._update_corner_preview(
                registry, preview_ids, start_pid, end_pid
            )
        return preview_ids

    @staticmethod
    def start_preview(
        registry: EntityRegistry,
        x: float,
        y: float,
        snapped_pid: EntityID | None = None,
        center_on_start: bool = False,
        **kwargs,
    ) -> RectanglePreviewState:
        """
        Creates initial preview state with start and end points.

        Args:
            registry: The entity registry to modify.
            x, y: The initial coordinates.
            snapped_pid: An existing point ID to snap to, or None.
            center_on_start: When True, the start point is the rectangle's
                center and the end point defines the half-extent.

        Returns:
            RectanglePreviewState for use with update_preview and
            cleanup_preview.
        """
        if snapped_pid is not None:
            start_id = snapped_pid
            start_temp = False
        else:
            start_id = registry.add_point(x, y)
            start_temp = True

        p_end_id = registry.add_point(x, y)

        preview_ids = RectangleCommand.create_preview(
            registry,
            start_id,
            p_end_id,
            center_on_start=center_on_start,
        )
        assert preview_ids is not None

        return RectanglePreviewState(
            start_id=start_id,
            start_temp=start_temp,
            p_end_id=p_end_id,
            preview_ids=preview_ids,
            center_on_start=center_on_start,
        )

    @staticmethod
    def _update_corner_preview_state(
        registry: EntityRegistry,
        preview_state: RectanglePreviewState,
        x: float,
        y: float,
    ) -> None:
        """
        Updates the end point and refreshes geometry in corner mode
        (start/end points are opposite corners).
        """
        try:
            p_end = registry.get_point(preview_state.p_end_id)
            p_start = registry.get_point(preview_state.start_id)
        except IndexError:
            return

        if (
            preview_state.locked_width is not None
            or preview_state.locked_height is not None
        ):
            dx = p_end.x - p_start.x
            dy = p_end.y - p_start.y
            sign_x = 1.0 if dx >= 0 else -1.0
            sign_y = 1.0 if dy >= 0 else -1.0
            p_end.x = (
                p_start.x + sign_x * preview_state.locked_width
                if preview_state.locked_width is not None
                else x
            )
            p_end.y = (
                p_start.y + sign_y * preview_state.locked_height
                if preview_state.locked_height is not None
                else y
            )
        else:
            p_end.x = x
            p_end.y = y

        RectangleCommand._update_corner_preview(
            registry,
            preview_state.preview_ids,
            preview_state.start_id,
            preview_state.p_end_id,
        )

    @staticmethod
    def _update_centered_preview_state(
        registry: EntityRegistry,
        preview_state: RectanglePreviewState,
        x: float,
        y: float,
    ) -> None:
        """
        Updates the end point and refreshes geometry in center-on-start
        mode (start point is the center; end point defines the
        half-extent).
        """
        try:
            p_end = registry.get_point(preview_state.p_end_id)
            p_start = registry.get_point(preview_state.start_id)
        except IndexError:
            return

        if (
            preview_state.locked_width is not None
            or preview_state.locked_height is not None
        ):
            dx = p_end.x - p_start.x
            dy = p_end.y - p_start.y
            sign_x = 1.0 if dx >= 0 else -1.0
            sign_y = 1.0 if dy >= 0 else -1.0
            new_dx = (
                preview_state.locked_width / 2.0
                if preview_state.locked_width is not None
                else abs(dx)
            )
            new_dy = (
                preview_state.locked_height / 2.0
                if preview_state.locked_height is not None
                else abs(dy)
            )
            p_end.x = p_start.x + sign_x * new_dx
            p_end.y = p_start.y + sign_y * new_dy
        else:
            p_end.x = x
            p_end.y = y

        RectangleCommand._update_centered_preview(
            registry,
            preview_state.preview_ids,
            preview_state.start_id,
            preview_state.p_end_id,
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
            center_on_start: When True, the start point is the rectangle's
                center and the end point defines the half-extent.
            constrain_square: When True the end point is adjusted so
                both extents are equal, keeping the start point fixed.

        Raises:
            TypeError: If preview_state is not a RectanglePreviewState.
        """
        if not isinstance(preview_state, RectanglePreviewState):
            raise TypeError("Expected RectanglePreviewState")

        if constrain_square:
            x, y = RectangleCommand._constrain_point_to_square(
                registry, preview_state.start_id, x, y
            )
        preview_state.constrain_square = constrain_square

        # If the mode changed, recreate the preview geometry so the
        # point/line topology matches the new interpretation.
        if preview_state.center_on_start != center_on_start:
            RectangleCommand._rebuild_preview_for_mode(
                registry, preview_state, center_on_start
            )

        if center_on_start:
            RectangleCommand._update_centered_preview_state(
                registry, preview_state, x, y
            )
        else:
            RectangleCommand._update_corner_preview_state(
                registry, preview_state, x, y
            )

    @staticmethod
    def _rebuild_preview_for_mode(
        registry: EntityRegistry,
        preview_state: RectanglePreviewState,
        center_on_start: bool,
    ) -> None:
        """
        Tears down and recreates preview geometry when center_on_start
        mode toggles, since the two modes have different point topologies.

        The start and end points are preserved (they define the
        rectangle); only the derived corner/center/line preview geometry
        is rebuilt.
        """
        preview_ids = preview_state.preview_ids
        point_ids = set(preview_ids.values())

        # Remove entities that reference the preview points
        entity_ids_to_remove = {
            e.id
            for e in registry.entities
            if any(pid in point_ids for pid in e.get_point_ids())
        }
        registry.remove_entities_by_id(list(entity_ids_to_remove))

        # Remove only the preview-derived points (not start/end)
        registry.points = [p for p in registry.points if p.id not in point_ids]

        preview_state.center_on_start = center_on_start
        preview_state.preview_ids = (
            RectangleCommand.create_preview(
                registry,
                preview_state.start_id,
                preview_state.p_end_id,
                center_on_start=center_on_start,
            )
            or {}
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
            TypeError: If preview_state is not a RectanglePreviewState.
        """
        if not isinstance(preview_state, RectanglePreviewState):
            raise TypeError("Expected RectanglePreviewState")
        preview_ids = preview_state.preview_ids
        p_end_id = preview_state.p_end_id

        # Collect all point IDs to remove
        point_ids = set(preview_ids.values())
        point_ids.add(p_end_id)

        # Find and remove entities that use these points
        entity_ids_to_remove = {
            e.id
            for e in registry.entities
            if any(pid in point_ids for pid in e.get_point_ids())
        }
        registry.remove_entities_by_id(list(entity_ids_to_remove))

        # Remove points
        registry.points = [p for p in registry.points if p.id not in point_ids]

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._do_execute()

        reg = self.sketch.registry
        try:
            start_p = reg.get_point(self.start_pid)
        except IndexError:
            return

        final_mx, final_my = self.end_pos
        if self.end_pid is not None:
            try:
                end_p = reg.get_point(self.end_pid)
                final_mx, final_my = end_p.x, end_p.y
            except IndexError:
                pass  # Use mouse coords if pid is invalid

        result = self.calculate_geometry(
            start_p.x,
            start_p.y,
            final_mx,
            final_my,
            self.start_pid,
            self.end_pid,
            fixed_width=self.fixed_width,
            fixed_height=self.fixed_height,
            center_on_start=self.center_on_start,
            constrain_square=self.constrain_square,
        )
        if not result:
            if self.is_start_temp:
                self.sketch.remove_point_if_unused(self.start_pid)
            return

        points_dict = result["points"]
        points_to_add = []
        # In center_on_start mode p1 is a new corner point; in corner
        # mode p1 is the existing start point.
        if points_dict["p1"] is not None:
            points_to_add.append(points_dict["p1"])
        # These points are always new
        points_to_add.extend([points_dict["p2"], points_dict["p4"]])

        # Add p3 only if it wasn't an existing snapped point
        if self.end_pid is None:
            points_to_add.append(points_dict["p3"])

        # In corner mode the center is a new point; in center_on_start
        # mode the start point itself is the symmetry center.
        if points_dict["center"] is not None:
            points_to_add.append(points_dict["center"])

        # If the start point was temporary, remove it from the registry
        # and add its object to the command to be re-added properly.
        if self.is_start_temp:
            reg.points.remove(start_p)
            points_to_add.append(start_p)

        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=points_to_add,
            entities=result["entities"],
            constraints=result["constraints"],
        )
        self.add_cmd._do_execute()
        self._committed_end_id = self.end_pid

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
