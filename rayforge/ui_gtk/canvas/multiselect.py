from __future__ import annotations
import math
import logging
from typing import (
    TYPE_CHECKING,
    List,
    Tuple,
    Dict,
    Any,
    Union,
    Optional,
    Set,
)
from .region import ElementRegion, get_region_rect, check_region_hit
from . import element, transform
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

    def check_region_hit(
        self,
        x: float,
        y: float,
        candidates: Optional[Set[ElementRegion]] = None,
    ) -> ElementRegion:
        # The group's bounding box is (min_x, min_y, width, height) in world
        # coords. We convert the world mouse coordinate (x,y) into the group's
        # local AABB coordinate space.
        min_x, min_y, _, _ = self._bounding_box
        local_x = x - min_x
        local_y = y - min_y

        # Use the get_scale() method which correctly returns signed scale
        # factors, implicitly handling the flip status for the geometry check.
        scale_compensation = self.canvas.view_transform.get_scale()

        return check_region_hit(
            local_x,
            local_y,
            self.width,
            self.height,
            self.canvas.BASE_HANDLE_SIZE,
            scale_compensation=scale_compensation,
            candidates=candidates,
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

        # To map a point from the old box to the new one, the correct matrix
        # is T_new * S * T_inv. Assuming post-multiplication (M' = M * Op),
        # the operations must be chained in the order they should appear in
        # the final matrix product.
        self.transform = (
            Matrix.identity()
            .post_translate(new_x, new_y)
            .post_scale(scale_x, scale_y)
            .post_translate(-orig_x, -orig_y)
        )
        self._update_element_transforms()

    def apply_rotate(
        self, angle_delta: float, center: Optional[Tuple[float, float]] = None
    ):
        """
        Sets the group transform to a rotation around the group's initial
        center and updates elements.
        """
        if center is None:
            center = self.initial_center
        self.transform = Matrix.rotation(angle_delta, center)
        self._update_element_transforms()

    def resize_from_drag(
        self,
        active_region: ElementRegion,
        offset_x: float,
        offset_y: float,
        active_origin: Tuple[float, float, float, float],
        ctrl_pressed: bool,
        shift_pressed: bool,
    ):
        """
        Calculates and applies the new group bounding box by calling the
        centralized logic in `transform.py`.
        """
        # 1. Define minimum size in world units.
        min_size_px = 20.0
        scale_x, scale_y = self.canvas.view_transform.get_abs_scale()
        min_size_world = (
            min_size_px / scale_x if scale_x > 1e-6 else 0.0,
            min_size_px / scale_y if scale_y > 1e-6 else 0.0,
        )

        # 2. Delegate the calculation, passing raw offsets and flip status.
        # The drag_delta for the world-aligned box is the world mouse offset.
        new_box = transform.calculate_resized_box(
            original_box=active_origin,
            active_region=active_region,
            drag_delta=(offset_x, offset_y),
            is_flipped=self.canvas.view_transform.is_flipped(),
            constrain_aspect=shift_pressed,
            from_center=ctrl_pressed,
            min_size=min_size_world,
        )

        # 3. Apply the result.
        self.apply_resize(new_box, active_origin)

    def rotate_from_drag(
        self,
        current_x: float,
        current_y: float,
        rotation_pivot: Tuple[float, float],
        drag_start_angle: float,
    ):
        """
        Rotates the entire selection group based on cursor drag.
        The coordinates are in WORLD space.
        """
        center_x, center_y = rotation_pivot
        current_angle = math.degrees(
            math.atan2(current_y - center_y, current_x - center_x)
        )
        angle_diff = current_angle - drag_start_angle
        # Temporarily override initial_center for the rotate call
        original_center = self.initial_center
        self.initial_center = rotation_pivot
        self.apply_rotate(angle_diff)
        self.initial_center = original_center

    def shear_from_drag(
        self,
        active_region: ElementRegion,
        world_dx: float,
        world_dy: float,
        active_origin: Tuple[float, float, float, float],
    ):
        """Shears the entire selection group."""
        shx, shy = 0.0, 0.0
        x, y, w, h = active_origin
        anchor_x, anchor_y = 0.0, 0.0

        is_view_flipped = self.canvas.view_transform.is_flipped()
        semantic_is_top = active_region == ElementRegion.SHEAR_TOP
        semantic_is_bottom = active_region == ElementRegion.SHEAR_BOTTOM
        semantic_is_left = active_region == ElementRegion.SHEAR_LEFT
        semantic_is_right = active_region == ElementRegion.SHEAR_RIGHT

        if semantic_is_top or semantic_is_bottom:
            # This logic is confirmed to work correctly in both Y-up and
            # Y-down.
            visual_top_y = y + h if is_view_flipped else y
            visual_bottom_y = y if is_view_flipped else y + h
            anchor_y = visual_bottom_y if semantic_is_top else visual_top_y
            anchor_x = x + w / 2

            y_diff = visual_bottom_y - visual_top_y
            if semantic_is_top:
                shx = -world_dx / y_diff if abs(y_diff) > 1e-6 else 0.0
            else:  # semantic_is_bottom
                shx = world_dx / y_diff if abs(y_diff) > 1e-6 else 0.0

        elif semantic_is_left or semantic_is_right:
            # Anchor is the edge opposite to the one being dragged.
            anchor_x = x if semantic_is_right else (x + w)
            anchor_y = y + h / 2

            # The vertical shear factor `shy` is derived from the transform:
            # `delta_y = shy * (x_dragged - x_anchor)`.
            # The `world_dy` already has the correct sign for the drag
            # regardless of whether the view is Y-up or Y-down. This single
            # set of formulas works for both cases without modification.
            if semantic_is_left:
                # Drag left edge, anchor is on right: x_dragged-x_anchor = -w
                # shy = delta_y / -w
                shy = -world_dy / w if w > 1e-6 else 0.0
            else:  # semantic_is_right
                # Drag right edge, anchor is on left: x_dragged-x_anchor = +w
                # shy = delta_y / w
                shy = world_dy / w if w > 1e-6 else 0.0

        # Construct delta shear matrix around world anchor
        self.transform = (
            Matrix.identity()
            .post_translate(anchor_x, anchor_y)
            .post_shear(shx, shy)
            .post_translate(-anchor_x, -anchor_y)
        )
        self._update_element_transforms()
