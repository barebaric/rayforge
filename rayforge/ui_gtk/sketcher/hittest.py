import logging
from typing import Any, List, Optional, Tuple

import cairo

from ...core.sketcher.constraints import (
    CoincidentConstraint,
    PointOnLineConstraint,
)
from ...core.sketcher.entities import (
    Entity,
    TextBoxEntity,
    Bezier,
)
from ..canvas.worldsurface import WorldSurface


logger = logging.getLogger(__name__)


class SketchHitTester:
    """Handles geometric hit testing for sketch elements."""

    def __init__(self, snap_distance: float = 12.0):
        self.snap_distance = snap_distance

    def screen_to_model(
        self, wx: float, wy: float, element: Any
    ) -> Tuple[float, float]:
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
        except Exception:
            return 0.0, 0.0

    def get_model_to_screen_transform(self, element: Any) -> Any:
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
        self, wx: float, wy: float, element: Any
    ) -> Tuple[Optional[str], Any]:
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
        element: Any,
        strict_containment: bool = False,
    ) -> Tuple[List[int], List[int]]:
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
        for p in registry.points:
            if p.is_in_rect(rect):
                points_inside.append(p.id)

        # 2. Check Entities
        for e in registry.entities:
            is_match = False
            if strict_containment:
                if e.is_contained_by(rect, registry):
                    is_match = True
            else:
                if e.intersects_rect(rect, registry):
                    is_match = True

            if is_match:
                entities_inside.append(e.id)

        return points_inside, entities_inside

    def _hit_test_points(self, wx, wy, element) -> Optional[int]:
        """Precise point hit-testing in SCREEN coordinates."""
        if not element.canvas:
            return None

        to_screen = self.get_model_to_screen_transform(element)
        cursor_sx, cursor_sy = element.canvas.view_transform.transform_point(
            (wx, wy)
        )
        threshold = element.point_radius + 2.0
        best_pid = None
        min_dist_sq = float("inf")

        points = element.sketch.registry.points or []
        for p in points:
            pt_sx, pt_sy = to_screen.transform_point((p.x, p.y))
            dist_sq = (cursor_sx - pt_sx) ** 2 + (cursor_sy - pt_sy) ** 2
            if dist_sq < threshold**2 and dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_pid = p.id
        return best_pid

    def _hit_test_overlays(self, wx, wy, element) -> Tuple[Optional[str], Any]:
        if not element.canvas:
            return None, None
        to_screen = self.get_model_to_screen_transform(element)
        cursor_sx, cursor_sy = element.canvas.view_transform.transform_point(
            (wx, wy)
        )
        constraints = element.sketch.constraints or []
        for constr in constraints:
            if not constr.user_visible:
                continue
        threshold = 13.0
        text_box_point_ids = set()
        for entity in element.sketch.registry.entities:
            if isinstance(entity, TextBoxEntity):
                text_box_point_ids.update(
                    entity.get_all_frame_point_ids(element.sketch.registry)
                )
        for entity in element.sketch.registry.entities:
            if isinstance(entity, Bezier):
                start_pt = element.sketch.registry.get_point(entity.start_idx)
                end_pt = element.sketch.registry.get_point(entity.end_idx)
                if start_pt is None or end_pt is None:
                    continue
                if entity.cp1 is not None:
                    cp1_abs = (
                        start_pt.x + entity.cp1[0],
                        start_pt.y + entity.cp1[1],
                    )
                    cp1_sx, cp1_sy = to_screen.transform_point(cp1_abs)
                    dx = cursor_sx - cp1_sx
                    dy = cursor_sy - cp1_sy
                    dist_sq = dx * dx + dy * dy
                    if dist_sq < threshold**2:
                        return "control_point_out", (
                            start_pt.id,
                            entity.id,
                            1,
                        )
                if entity.cp2 is not None:
                    cp2_abs = (
                        end_pt.x + entity.cp2[0],
                        end_pt.y + entity.cp2[1],
                    )
                    cp2_sx, cp2_sy = to_screen.transform_point(cp2_abs)
                    dx = cursor_sx - cp2_sx
                    dy = cursor_sy - cp2_sy
                    dist_sq = dx * dx + dy * dy
                    if dist_sq < threshold**2:
                        return "control_point_in", (
                            end_pt.id,
                            entity.id,
                            2,
                        )
        for idx, constr in enumerate(constraints):
            if not constr.user_visible:
                continue
            if isinstance(constr, CoincidentConstraint):
                if (
                    constr.p1 in text_box_point_ids
                    or constr.p2 in text_box_point_ids
                ):
                    continue
            elif isinstance(constr, PointOnLineConstraint):
                if constr.point_id in text_box_point_ids:
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

    def _hit_test_entities(self, wx, wy, element) -> Optional[Entity]:
        mx, my = self.screen_to_model(wx, wy, element)
        scale = 1.0
        if isinstance(element.canvas, WorldSurface):
            scale_x, _ = element.canvas.get_view_scale()
            scale = scale_x if scale_x > 1e-9 else 1.0
        threshold = self.snap_distance / scale

        registry = element.sketch.registry
        entities = registry.entities or []
        for entity in entities:
            if entity.hit_test(mx, my, threshold, registry):
                return entity
        return None
