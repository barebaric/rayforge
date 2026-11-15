import logging
from typing import TYPE_CHECKING

import cairo
from raygeo.geo import Matrix

from ..core.types import HitTarget

if TYPE_CHECKING:
    from .sketchelement import SketchElement

from raygeo.sketch.constraints.coincident import CoincidentConstraint
from raygeo.sketch.constraints.point_on_curve import (
    PointOnCurveConstraint,
)
from raygeo.sketch.constraints.point_on_line import PointOnLineConstraint
from raygeo.sketch.entities.bezier import Bezier

from rayforge.ui_gtk.canvas.worldsurface import WorldSurface

from ..core import constraint_info
from ..core.entity_ops import text_box_fourth_corner

logger = logging.getLogger(__name__)

OVERLAY_HIT_THRESHOLD = 13.0


class SketchHitTester:
    """Handles geometric hit testing for sketch elements."""

    def __init__(self, snap_distance: float = 12.0):
        self.snap_distance = snap_distance

    def screen_to_model(
        self, wx: float, wy: float, element: "SketchElement"
    ) -> tuple[float, float]:
        """
        Converts world coordinates to Model coordinates
        (accounting for content_transform).
        World -> Local -> Model
        """
        try:
            # 1. World -> Local
            inv_world = element.get_world_transform().invert()
            lx, ly = inv_world.transform_point((wx, wy))

            # 2. Local -> Model (Inverse of content_transform)
            inv_content = element.content_transform.invert()
            mx, my = inv_content.transform_point((lx, ly))

            return mx, my
        except ValueError:
            return 0.0, 0.0

    def get_model_to_screen_transform(
        self, element: "SketchElement"
    ) -> Matrix:
        """
        Returns Matrix: Model -> Screen.
        """
        if not element.canvas:
            return cairo.Matrix()

        local_to_screen = (
            element.canvas.view_transform @ element.get_world_transform()
        )
        model_to_local = element.content_transform

        return local_to_screen @ model_to_local

    def get_hit_data(
        self, wx: float, wy: float, element: "SketchElement"
    ) -> tuple[str | None, HitTarget]:
        """
        Determines what was clicked using Model coordinates.
        Returns (type_string, object_id_or_index).
        Priorities: Points > Overlays (Constraints/Junctions) > Entities.
        """
        if not element.canvas:
            return None, None

        # 1. Points (most specific target)
        hit_pid = self._hit_test_points(wx, wy, element)
        if hit_pid is not None:
            return "point", hit_pid

        # 2. Overlays (Constraints and Junctions)
        hit_type, hit_obj = self._hit_test_overlays(wx, wy, element)
        if hit_type is not None:
            return hit_type, hit_obj

        # 3. Entities (Lines/Arcs/Circles)
        hit_entity = self._hit_test_entities(wx, wy, element)
        if hit_entity is not None:
            return "entity", hit_entity

        return None, None

    def get_objects_in_rect(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        element: "SketchElement",
        strict_containment: bool = False,
    ) -> tuple[list[int], list[int]]:
        """
        Finds all points and entities within a Model Space rectangle.

        Args:
            min_x, min_y, max_x, max_y: The rectangle in Model Space.
            element: The SketchElement.
            strict_containment:
                If True (Window Selection): Objects must be fully inside.
                If False (Crossing Selection): Objects can overlap or be
                  inside.

        Returns:
            A tuple of (list_of_point_ids, list_of_entity_ids).
        """
        registry = element.sketch.registry
        points_inside = []
        entities_inside = []
        rect = (min_x, min_y, max_x, max_y)

        # 1. Check Points
        for p in registry.points():
            x, y = p.x, p.y
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                points_inside.append(p.id)

        # 2. Check Entities
        for e in registry.entities():
            if element.sketch.flags.is_invisible(e.id):
                continue
            endpoints = registry.entity_endpoint_ids(e.id)
            is_match = False
            if not endpoints:
                continue
            if strict_containment:
                if registry.is_contained_by(e.id, rect):
                    is_match = True
            else:
                if registry.intersects_rect(e.id, rect):
                    is_match = True

            if is_match:
                entities_inside.append(e.id)

        return points_inside, entities_inside

    def _hit_test_points(self, wx, wy, element) -> int | None:
        """Precise point hit-testing in SCREEN coordinates.

        Derived points (whose position is owned by a master object,
        see ``Sketch.get_derived_point_ids``) lose to user geometry:
        where they coincide, the user geometry point is picked,
        because dragging a derived point is reverted by its master.
        A derived point is still picked when no user geometry point
        is nearby.
        """
        if not element.canvas:
            return None

        to_screen = self.get_model_to_screen_transform(element)
        cursor_sx, cursor_sy = element.canvas.view_transform.transform_point(
            (wx, wy)
        )
        threshold = element.point_radius + 2.0

        points = element.sketch.registry.points()
        derived = element.sketch.get_derived_point_ids()

        def closest(predicate) -> int | None:
            best_pid = None
            min_dist_sq = float("inf")
            for p in reversed(points):
                if not predicate(p.id):
                    continue
                pt_sx, pt_sy = to_screen.transform_point((p.x, p.y))
                dist_sq = (cursor_sx - pt_sx) ** 2 + (cursor_sy - pt_sy) ** 2
                if dist_sq < threshold**2 and dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_pid = p.id
            return best_pid

        hit = closest(lambda pid: pid not in derived)
        if hit is not None:
            return hit
        return closest(lambda pid: pid in derived)

    def _hit_test_overlays(
        self, wx, wy, element
    ) -> tuple[str | None, HitTarget]:
        if not element.canvas:
            return None, None
        to_screen = self.get_model_to_screen_transform(element)
        cursor_sx, cursor_sy = element.canvas.view_transform.transform_point(
            (wx, wy)
        )
        constraints = element.sketch.constraints
        text_box_point_ids = set()
        for entity in element.sketch.registry.entities():
            if entity.type_name() != "text_box":
                continue
            text_box_point_ids.update(entity.get_point_ids())
            fourth = text_box_fourth_corner(element.sketch.registry, entity)
            if fourth is not None:
                text_box_point_ids.add(fourth)
        for entity in element.sketch.registry.entities():
            if not isinstance(entity, Bezier):
                continue
            start_pt = element.sketch.registry.get_point(entity.start_idx)
            end_pt = element.sketch.registry.get_point(entity.end_idx)
            if start_pt is None or end_pt is None:
                continue
            if entity.cp1_offset is not None:
                cp1_abs = (
                    start_pt[0] + entity.cp1_offset[0],
                    start_pt[1] + entity.cp1_offset[1],
                )
                cp1_sx, cp1_sy = to_screen.transform_point(cp1_abs)
                dx = cursor_sx - cp1_sx
                dy = cursor_sy - cp1_sy
                dist_sq = dx * dx + dy * dy
                if dist_sq < OVERLAY_HIT_THRESHOLD**2:
                    return "control_point_out", (
                        start_pt.id
                        if hasattr(start_pt, "id")
                        else entity.start_idx,
                        entity.id,
                        1,
                    )
            if entity.cp2_offset is not None:
                cp2_abs = (
                    end_pt[0] + entity.cp2_offset[0],
                    end_pt[1] + entity.cp2_offset[1],
                )
                cp2_sx, cp2_sy = to_screen.transform_point(cp2_abs)
                dx = cursor_sx - cp2_sx
                dy = cursor_sy - cp2_sy
                dist_sq = dx * dx + dy * dy
                if dist_sq < OVERLAY_HIT_THRESHOLD**2:
                    return "control_point_in", (
                        end_pt.id if hasattr(end_pt, "id") else entity.end_idx,
                        entity.id,
                        2,
                    )
        for idx, constr in enumerate(constraints):
            if not constraint_info.user_visible(constr):
                continue
            if isinstance(constr, CoincidentConstraint):
                if (
                    constr.p1 in text_box_point_ids
                    or constr.p2 in text_box_point_ids
                ):
                    continue
            elif (
                isinstance(
                    constr, (PointOnCurveConstraint, PointOnLineConstraint)
                )
                and constr.point in text_box_point_ids
            ):
                continue
            if constr.is_hit(
                cursor_sx,
                cursor_sy,
                element.sketch.registry,
                to_screen.transform_point,
                element,
                element.point_radius,
            ):
                return "constraint", idx
        return None, None

    def _hit_test_entities(
        self, wx: float, wy: float, element: "SketchElement"
    ) -> HitTarget:
        mx, my = self.screen_to_model(wx, wy, element)
        scale = 1.0
        if isinstance(element.canvas, WorldSurface):
            scale_x, _ = element.canvas.get_view_scale()
            scale = scale_x if scale_x > 1e-9 else 1.0
        threshold = self.snap_distance / scale

        registry = element.sketch.registry
        entities = registry.entities()
        for entity in reversed(entities):
            if element.sketch.flags.is_invisible(entity.id):
                continue
            if entity.hit_test(mx, my, threshold, registry):
                return entity
        return None
