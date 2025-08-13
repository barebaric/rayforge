from __future__ import annotations
import math
import logging
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Union
from .region import ElementRegion, get_region_rect, check_region_hit
from . import element
from ...core.matrix import Matrix


# Forward declaration for type hinting to avoid circular imports
if TYPE_CHECKING:
    from .canvas import Canvas
    from .element import CanvasElement

logger = logging.getLogger(__name__)


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
        self.initial_states: List[Dict[str, Any]] = []
        self.initial_center: Tuple[float, float] = (0, 0)

        # The transformation matrix for the entire group, applied during a
        # drag operation.
        self.transform: Matrix = Matrix.identity()

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
        Stores the initial state of each top-level selected element.
        This includes its world transform and its parent's inverse world
        transform, which is crucial for recalculating its new local
        properties after a group transform.
        """
        self.initial_states.clear()
        self._calculate_bounding_box()
        self.initial_center = self.center
        self.transform = Matrix.identity()

        selected_set = set(self.elements)
        top_level_elements = []

        for elem in self.elements:
            is_top_level = True
            parent = elem.parent
            while isinstance(parent, element.CanvasElement):
                if parent in selected_set:
                    is_top_level = False
                    break
                parent = parent.parent
            if is_top_level:
                top_level_elements.append(elem)

        for elem in top_level_elements:
            parent_inv_world = Matrix.identity()
            if isinstance(elem.parent, element.CanvasElement):
                parent_inv_world = elem.parent.get_world_transform().invert()

            self.initial_states.append(
                {
                    "elem": elem,
                    "initial_world": elem.get_world_transform(),
                    "parent_inv_world": parent_inv_world,
                }
            )

    def _update_element_transforms(self):
        """
        Applies the group's `self.transform` to each element's initial
        state to calculate its new local transform matrix, which is then
        set directly on the element. This preserves shear.
        """
        for state in self.initial_states:
            elem: CanvasElement = state["elem"]

            # Calculate the element's new world transform by applying the
            # group's delta transform to its initial state.
            new_world_transform = self.transform @ state["initial_world"]

            # To get the new local transform, we must convert this new
            # world transform back into the element's parent-relative
            # coordinate space.
            new_transform_in_parent_space = (
                state["parent_inv_world"] @ new_world_transform
            )

            # Set the new matrix directly on the element. This avoids
            # destructive decomposition and preserves shear.
            elem.set_transform(new_transform_in_parent_space)

    def get_region_rect(
        self,
        region: ElementRegion,
        base_handle_size: float,
        scale_compensation: Union[float, Tuple[float, float]] = 1.0,
    ) -> Tuple[float, float, float, float]:
        return get_region_rect(
            region,
            self.width,
            self.height,
            base_handle_size,
            scale_compensation,
        )

    def check_region_hit(self, x: float, y: float) -> ElementRegion:
        # The group's bounding box is (min_x, min_y, width, height) in world
        # coords. Our handle geometry logic in get_region_rect adapts to the
        # coordinate system (Y-down vs Y-up) based on the view transform. We
        # must convert the world mouse coordinate (x,y) into the group's
        # local AABB coordinate space, preserving the Y-axis orientation.
        min_x, min_y, width, height = self._bounding_box

        # local_x is the distance from the left edge of the AABB.
        local_x = x - min_x

        # For local_y, we provide a coordinate relative to the AABB's origin
        # (min_x, min_y). - For a normal Y-down view, min_y is the top edge,
        # so this creates a Y-down local coordinate (distance from top). - For
        # a flipped Y-up view, min_y is the bottom edge, so this creates a
        # Y-up local coordinate (distance from bottom). This single
        # calculation produces the local coordinate space that
        # get_region_rect expects for both cases.
        local_y = y - min_y

        # Now we determine if the view is flipped to pass this info to the
        # geometry calculation function.
        m = self.canvas.view_transform.m
        det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]

        # The scale compensation must account for the canvas's view transform.
        sx = math.hypot(m[0, 0], m[1, 0])
        sy = math.hypot(m[0, 1], m[1, 1])
        if det < 0:
            sy = -sy  # Use signed scale for get_region_rect
        scale_compensation = (sx, sy)

        # check_region_hit from region.py will use the local coordinates and
        # the scale_compensation (which indicates if the view is flipped) to
        # correctly test against the handle geometry.
        return check_region_hit(
            local_x,
            local_y,
            self.width,
            self.height,
            self.canvas.BASE_HANDLE_SIZE,
            scale_compensation=scale_compensation,
        )

    def apply_move(self, dx: float, dy: float):
        """
        Sets the group transform to a simple translation and updates
        elements.
        """
        self.transform = Matrix.translation(dx, dy)
        self._update_element_transforms()

    def apply_resize(
        self,
        new_box: Tuple[float, float, float, float],
        original_box: Tuple[float, float, float, float],
    ):
        """
        Calculates a scale/translate transform that maps the original
        bounding box to the new one, and applies it to the group.
        """
        orig_x, orig_y, orig_w, orig_h = original_box
        new_x, new_y, new_w, new_h = new_box

        if orig_w <= 1e-6 or orig_h <= 1e-6:
            return

        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        # This transform scales the box around its top-left corner, then
        # translates it to the new position.
        # 1. Translate to origin
        t_to_origin = Matrix.translation(-orig_x, -orig_y)
        # 2. Scale
        s_around_origin = Matrix.scale(scale_x, scale_y)
        # 3. Translate to new top-left position
        t_to_new = Matrix.translation(new_x, new_y)

        # The combined transform is applied from right to left
        self.transform = t_to_new @ s_around_origin @ t_to_origin
        self._update_element_transforms()

    def apply_rotate(self, angle_delta: float):
        """
        Sets the group transform to a rotation around the group's initial
        center and updates elements.
        """
        self.transform = Matrix.rotation(angle_delta, self.initial_center)
        self._update_element_transforms()
