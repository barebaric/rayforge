from __future__ import annotations
import math
from typing import TYPE_CHECKING, List, Tuple, Dict, Any
from .region import ElementRegion, get_region_rect, check_region_hit
from . import element
from ...core.matrix import Matrix


# Forward declaration for type hinting to avoid circular imports
if TYPE_CHECKING:
    from .canvas import Canvas
    from .element import CanvasElement


class MultiSelectionGroup:
    def __init__(self, elements: List[CanvasElement], canvas: Canvas):
        if not elements:
            raise ValueError(
                "MultiSelectionGroup cannot be initialized with an "
                "empty list of elements."
            )

        self.elements: List[CanvasElement] = elements
        self.canvas: Canvas = canvas
        self._bounding_box: Tuple[float, float, float, float] = (0, 0, 0, 0)
        self._center: Tuple[float, float] = (0, 0)
        self.angle: float = 0.0
        self.initial_states: List[Dict[str, Any]] = []
        self.initial_center: Tuple[float, float] = (0, 0)

        self._calculate_bounding_box()

    @property
    def x(self) -> float:
        return self._bounding_box[0]

    @property
    def y(self) -> float:
        return self._bounding_box[1]

    @property
    def width(self) -> float:
        return self._bounding_box[2]

    @property
    def height(self) -> float:
        return self._bounding_box[3]

    @property
    def center(self) -> Tuple[float, float]:
        return self._center

    def _calculate_bounding_box(self):
        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        for elem in self.elements:
            # We need the full world transform to correctly find the corners
            world_transform = elem.get_world_transform()
            w, h = elem.width, elem.height

            # The corners of the element in its own local space
            local_corners = [(0, 0), (w, 0), (w, h), (0, h)]

            for lx, ly in local_corners:
                wx, wy = world_transform.transform_point((lx, ly))
                min_x, min_y = min(min_x, wx), min(min_y, wy)
                max_x, max_y = max(max_x, wx), max(max_y, wy)

        self._bounding_box = (min_x, min_y, max_x - min_x, max_y - min_y)
        self._center = (min_x + self.width / 2, min_y + self.height / 2)

    def store_initial_states(self):
        """
        Stores the state of only the top-most elements in the selection.
        This prevents transformations from being applied to both a parent
        and its child, which would cause a "double transform".
        """
        self.initial_states.clear()
        self._calculate_bounding_box()
        self.initial_center = self.center

        selected_set = set(self.elements)
        top_level_elements = []

        for elem in self.elements:
            is_top_level = True
            if isinstance(elem.parent, element.CanvasElement):
                if elem.parent in selected_set:
                    is_top_level = False

            if is_top_level:
                top_level_elements.append(elem)

        for elem in top_level_elements:
            self.initial_states.append(
                {
                    "elem": elem,
                    "rect": elem.rect(),
                    "world_center": elem.get_world_center(),
                    "world_angle": elem.get_world_angle(),
                }
            )

    def get_region_rect(
        self,
        region: ElementRegion,
        base_handle_size: float,
        max_handle_size: float,
        scale_compensation: float = 1.0,
    ) -> Tuple[float, float, float, float]:
        return get_region_rect(
            region,
            self.width,
            self.height,
            base_handle_size,
            max_handle_size,
            scale_compensation,
        )

    def check_region_hit(self, x: float, y: float) -> ElementRegion:
        """
        Checks which region is hit by a point. The group's selection frame
        is always unrotated in world space, so we can use local coordinates.
        """
        # Convert absolute world coordinates to be local to the group's
        # bounding box.
        local_x = x - self.x
        local_y = y - self.y

        # For hit testing, use a consistent base size for a better feel.
        base_hit_size = 15.0
        return check_region_hit(
            local_x,
            local_y,
            self.width,
            self.height,
            base_hit_size,
        )

    def apply_move(self, dx: float, dy: float):
        """Moves all elements in the group by a delta, correctly handling
        different parent coordinate systems for each element.
        """
        screen_delta = (dx, dy)

        for state in self.initial_states:
            elem: CanvasElement = state["elem"]
            initial_x, initial_y, _, _ = state["rect"]

            parent_transform_inv = Matrix.identity()
            if isinstance(elem.parent, element.CanvasElement):
                parent_transform_inv = (
                    elem.parent.get_world_transform().invert()
                )

            local_dx, local_dy = parent_transform_inv.transform_vector(
                screen_delta
            )
            elem.set_pos(initial_x + local_dx, initial_y + local_dy)

    def apply_resize(
        self,
        new_box: Tuple[float, float, float, float],
        original_box: Tuple[float, float, float, float],
    ):
        orig_x, orig_y, orig_w, orig_h = original_box
        new_x, new_y, new_w, new_h = new_box

        if orig_w <= 1 or orig_h <= 1:
            return

        scale_x, scale_y = new_w / orig_w, new_h / orig_h

        for state in self.initial_states:
            elem: CanvasElement = state["elem"]
            initial_rect = state["rect"]
            initial_world_center = state["world_center"]
            initial_world_angle_deg = state["world_angle"]
            initial_w, initial_h = initial_rect[2], initial_rect[3]

            # --- Angle and Size calculations are correct and unchanged ---
            rel_center_x = (initial_world_center[0] - orig_x) / orig_w
            rel_center_y = (initial_world_center[1] - orig_y) / orig_h
            new_abs_center_x = new_x + (rel_center_x * new_w)
            new_abs_center_y = new_y + (rel_center_y * new_h)

            initial_world_angle_rad = math.radians(initial_world_angle_deg)
            cos_a, sin_a = (
                math.cos(initial_world_angle_rad),
                math.sin(initial_world_angle_rad),
            )
            vec_w_dir, vec_h_dir = (cos_a, sin_a), (-sin_a, cos_a)
            new_vec_w = (vec_w_dir[0] * scale_x, vec_w_dir[1] * scale_y)
            new_vec_h = (vec_h_dir[0] * scale_x, vec_h_dir[1] * scale_y)
            new_elem_w = math.hypot(*new_vec_w) * initial_w
            new_elem_h = math.hypot(*new_vec_h) * initial_h

            new_world_angle_rad = math.atan2(new_vec_w[1], new_vec_w[0])
            parent_world_angle = 0
            if isinstance(elem.parent, element.CanvasElement):
                parent_world_angle = elem.parent.get_world_angle()

            elem.set_angle(
                math.degrees(new_world_angle_rad) - parent_world_angle
            )
            elem.set_size(new_elem_w, new_elem_h)

            # Calculate the positional change of the element's center in world
            # space.
            world_delta_x = new_abs_center_x - initial_world_center[0]
            world_delta_y = new_abs_center_y - initial_world_center[1]

            # Convert this world-space delta into the parent's local-space
            # delta.
            parent_transform_inv = Matrix.identity()
            if isinstance(elem.parent, element.CanvasElement):
                parent_transform_inv = (
                    elem.parent.get_world_transform().invert()
                )
            local_delta_x, local_dy = parent_transform_inv.transform_vector(
                (world_delta_x, world_delta_y)
            )

            # Apply this local-space delta to the element's initial
            # local position.
            initial_x, initial_y = initial_rect[0], initial_rect[1]
            elem.set_pos(initial_x + local_delta_x, initial_y + local_dy)

    def apply_rotate(self, angle_delta: float):
        group_center_x, group_center_y = self.initial_center
        angle_rad = math.radians(angle_delta)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        for state in self.initial_states:
            elem: CanvasElement = state["elem"]
            initial_world_angle = state["world_angle"]
            initial_world_center = state["world_center"]
            initial_rect = state["rect"]

            # --- Angle calculation is correct and unchanged ---
            parent_world_angle = 0
            if isinstance(elem.parent, element.CanvasElement):
                parent_world_angle = elem.parent.get_world_angle()
            elem.set_angle(
                (initial_world_angle + angle_delta) - parent_world_angle
            )

            # Calculate the new world center by rotating the initial one
            # around the group center.
            ox = initial_world_center[0] - group_center_x
            oy = initial_world_center[1] - group_center_y
            new_ox, new_oy = ox * cos_a - oy * sin_a, ox * sin_a + oy * cos_a
            new_center_x, new_center_y = (
                group_center_x + new_ox,
                group_center_y + new_oy,
            )

            # Calculate the positional change of the element's center in
            # world space.
            world_delta_x = new_center_x - initial_world_center[0]
            world_delta_y = new_center_y - initial_world_center[1]

            # Convert this world-space delta into the parent's local-space
            # delta.
            parent_transform_inv = Matrix.identity()
            if isinstance(elem.parent, element.CanvasElement):
                parent_transform_inv = (
                    elem.parent.get_world_transform().invert()
                )
            local_delta_x, local_delta_y = (
                parent_transform_inv.transform_vector(
                    (world_delta_x, world_delta_y)
                )
            )

            # Apply this local-space delta to the element's initial
            # local position.
            initial_x, initial_y = initial_rect[0], initial_rect[1]
            elem.set_pos(initial_x + local_delta_x, initial_y + local_delta_y)
