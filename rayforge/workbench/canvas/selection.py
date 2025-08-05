from __future__ import annotations
import math
from typing import TYPE_CHECKING, List, Tuple, Dict, Any
from .region import ElementRegion, get_region_rect, check_region_hit
from . import element


# Forward declaration for type hinting to avoid circular imports
if TYPE_CHECKING:
    from .canvas import Canvas
    from .element import CanvasElement


class MultiSelectionGroup:
    """
    Helper class to manage interactions for multiple selected elements.
    It uses the generic region functions for its calculations.
    """

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
        self.handle_size: float = 30.0
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
            abs_x, abs_y, w, h = elem.rect_abs()
            angle_rad = math.radians(elem.get_angle())
            center_x, center_y = abs_x + w / 2, abs_y + h / 2
            corners_rel = [
                (-w / 2, -h / 2),
                (w / 2, -h / 2),
                (w / 2, h / 2),
                (-w / 2, h / 2),
            ]
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

            for rel_x, rel_y in corners_rel:
                rot_x = rel_x * cos_a - rel_y * sin_a
                rot_y = rel_x * sin_a + rel_y * cos_a
                abs_corner_x, abs_corner_y = center_x + rot_x, center_y + rot_y
                min_x, min_y = (
                    min(min_x, abs_corner_x),
                    min(min_y, abs_corner_y),
                )
                max_x, max_y = (
                    max(max_x, abs_corner_x),
                    max(max_y, abs_corner_y),
                )

        self._bounding_box = (min_x, min_y, max_x - min_x, max_y - min_y)
        self._center = (min_x + self.width / 2, min_y + self.height / 2)

    def store_initial_states(self):
        """Stores the state of the group and each element."""
        self.initial_states.clear()
        self._calculate_bounding_box()  # Recalculate just in case
        self.initial_center = self.center  # Store the stable, initial center
        for elem in self.elements:
            self.initial_states.append(
                {
                    "elem": elem,
                    "rect": elem.rect(),
                    "abs_pos": elem.pos_abs(),
                    "angle": elem.get_angle(),
                }
            )

    def get_region_rect(
        self, region: ElementRegion
    ) -> Tuple[float, float, float, float]:
        return get_region_rect(
            region, self.width, self.height, self.handle_size
        )

    def check_region_hit(self, x: float, y: float) -> ElementRegion:
        return check_region_hit(
            x,
            y,
            self.x,
            self.y,
            self.width,
            self.height,
            self.angle,
            self.center[0],
            self.center[1],
            self.handle_size,
        )

    def apply_move(self, dx: float, dy: float):
        """Moves all elements in the group by a delta."""
        for state in self.initial_states:
            elem = state["elem"]
            initial_x, initial_y, _, _ = state["rect"]
            elem.set_pos(initial_x + dx, initial_y + dy)

    def apply_resize(
        self,
        new_box: Tuple[float, float, float, float],
        original_box: Tuple[float, float, float, float],
    ):
        """
        Transforms all elements within the group based on the change from the
        original bounding box to the new one.
        """
        orig_x, orig_y, orig_w, orig_h = original_box
        new_x, new_y, new_w, new_h = new_box

        if orig_w <= 1 or orig_h <= 1:
            return

        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        for state in self.initial_states:
            elem: CanvasElement = state["elem"]
            initial_rect = state["rect"]
            initial_abs_pos = state["abs_pos"]
            initial_angle_deg = state["angle"]
            initial_w, initial_h = initial_rect[2], initial_rect[3]

            # 1. Transform element center
            initial_center_x = initial_abs_pos[0] + initial_w / 2
            initial_center_y = initial_abs_pos[1] + initial_h / 2

            rel_center_x = (initial_center_x - orig_x) / orig_w
            rel_center_y = (initial_center_y - orig_y) / orig_h

            new_abs_center_x = new_x + (rel_center_x * new_w)
            new_abs_center_y = new_y + (rel_center_y * new_h)

            # 2. Define and transform element's basis vectors
            initial_angle_rad = math.radians(initial_angle_deg)
            cos_a = math.cos(initial_angle_rad)
            sin_a = math.sin(initial_angle_rad)

            # Vector for width direction, and for height direction
            vec_w_dir = (cos_a, sin_a)
            vec_h_dir = (-sin_a, cos_a)

            new_vec_w = (vec_w_dir[0] * scale_x, vec_w_dir[1] * scale_y)
            new_vec_h = (vec_h_dir[0] * scale_x, vec_h_dir[1] * scale_y)

            # 3. New size is the magnitude of new vectors * original size
            new_elem_w = math.hypot(*new_vec_w) * initial_w
            new_elem_h = math.hypot(*new_vec_h) * initial_h

            # 4. New angle is the angle of the new width vector
            new_angle_rad = math.atan2(new_vec_w[1], new_vec_w[0])

            elem.set_angle(math.degrees(new_angle_rad))
            elem.set_size(new_elem_w, new_elem_h)

            # 5. Calculate new top-left from the new center
            new_abs_x = new_abs_center_x - new_elem_w / 2
            new_abs_y = new_abs_center_y - new_elem_h / 2

            # 6. Convert absolute back to parent-relative coordinates
            parent_abs_x, parent_abs_y = (0, 0)
            if isinstance(elem.parent, element.CanvasElement):
                parent_abs_x, parent_abs_y = elem.parent.pos_abs()

            elem.set_pos(
                new_abs_x - parent_abs_x,
                new_abs_y - parent_abs_y,
            )

    def apply_rotate(self, angle_delta: float):
        """Rotates all elements around the group's center."""
        group_center_x, group_center_y = self.initial_center
        angle_rad = math.radians(angle_delta)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        for state in self.initial_states:
            elem = state["elem"]
            initial_angle = state["angle"]
            elem.set_angle(initial_angle + angle_delta)

            # Rotate element's center around the group's center
            initial_abs_x, initial_abs_y = state["abs_pos"]
            initial_w, initial_h = state["rect"][2], state["rect"][3]

            elem_center_x = initial_abs_x + initial_w / 2
            elem_center_y = initial_abs_y + initial_h / 2

            # Translate to origin, rotate, translate back
            ox = elem_center_x - group_center_x
            oy = elem_center_y - group_center_y
            new_ox = ox * cos_a - oy * sin_a
            new_oy = ox * sin_a + oy * cos_a

            new_center_x = group_center_x + new_ox
            new_center_y = group_center_y + new_oy

            # Calculate new top-left from the new center
            new_abs_x = new_center_x - elem.width / 2
            new_abs_y = new_center_y - elem.height / 2

            # Convert back to parent-relative coordinates
            parent_abs_x, parent_abs_y = (0, 0)
            if isinstance(elem.parent, element.CanvasElement):
                parent_abs_x, parent_abs_y = elem.parent.pos_abs()

            elem.set_pos(
                new_abs_x - parent_abs_x,
                new_abs_y - parent_abs_y,
            )
