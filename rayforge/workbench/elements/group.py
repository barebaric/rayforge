import logging
import cairo
import math
from typing import TYPE_CHECKING, List, Optional, Tuple, cast
from ...core.matrix import Matrix
from ...core.item import DocItem
from ...core.workpiece import WorkPiece
from ...core.group import Group
from ..canvas.element import CanvasElement
from .workpiece import WorkPieceElement

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class GroupElement(CanvasElement):
    """
    A CanvasElement that represents a Group model.

    It does not render any content itself but provides a visual bounding box
    for its children and acts as a transformation parent for them in the
    canvas hierarchy. Its state is managed by its parent element (either a
    LayerElement or another GroupElement).
    """

    def __init__(self, group: "Group", **kwargs):
        # A GroupElement's local geometry is a fixed 1x1 unit square.
        # Its transformation matrix is solely responsible for its final
        # position and scale on the canvas.
        super().__init__(
            x=0,
            y=0,
            width=1,
            height=1,
            data=group,
            background=(0, 0, 0, 0),
            clip=False,
            **kwargs,
        )
        self.data: Group = group
        self.data.updated.connect(self.sync_with_model)
        self.data.transform_changed.connect(self._on_transform_changed)
        self.data.descendant_added.connect(self.sync_with_model)
        self.data.descendant_removed.connect(self.sync_with_model)

        # Set initial state from model
        self._on_transform_changed(self.data)
        self.sync_with_model(origin=None)

    def remove(self):
        """Disconnects signals before removing the element."""
        self.data.updated.disconnect(self.sync_with_model)
        self.data.transform_changed.disconnect(self._on_transform_changed)
        self.data.descendant_added.disconnect(self.sync_with_model)
        self.data.descendant_removed.disconnect(self.sync_with_model)
        super().remove()

    def _on_transform_changed(self, group: Group):
        """
        Handles transform changes from the model by directly applying the
        model's local matrix to this canvas element.
        """
        if not self.canvas:
            return

        self.set_transform(group.matrix)
        self.canvas.queue_draw()

    def recalculate_and_compensate_children(
        self,
    ) -> Optional[Tuple[Matrix, List[Tuple[CanvasElement, Matrix]]]]:
        """
        Calculates the group's new shrink-wrap transform and the necessary
        compensating local transforms for all its children.

        This ensures that when the group's transform changes, the children's
        world transforms remain fixed, preventing stretching or shifting.

        Returns:
            A tuple containing:
            - The new local transformation matrix for this group.
            - A list of tuples, each with a child element and its new
              compensating local matrix.
            Returns None if no update is needed.
        """
        if not self.children:
            return None

        # --- Step 1: Store the original state of all children ---
        child_original_world_transforms = {
            child: child.get_world_transform() for child in self.children
        }

        # --- Step 2: Calculate the group's new world bounding box ---
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        for child, world_transform in child_original_world_transforms.items():
            # Use the stored world transform to calculate the AABB
            x, y, w, h = world_transform.transform_rectangle(
                (0, 0, child.width, child.height)
            )
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        if not all(map(math.isfinite, [min_x, min_y, max_x, max_y])):
            return None

        new_world_w, new_world_h = max_x - min_x, max_y - min_y
        if new_world_w < 1e-9:
            new_world_w = 1.0
        if new_world_h < 1e-9:
            new_world_h = 1.0

        # --- Step 3: Calculate the group's new world & local transforms ---
        new_group_world_transform = Matrix.translation(
            min_x, min_y
        ) @ Matrix.scale(new_world_w, new_world_h)

        parent_transform = (
            cast(CanvasElement, self.parent).get_world_transform()
            if isinstance(self.parent, CanvasElement)
            else Matrix.identity()
        )
        try:
            new_group_local_transform = (
                parent_transform.invert() @ new_group_world_transform
            )
            inv_new_group_world = new_group_world_transform.invert()
        except Exception:
            return None

        # --- Step 4: Calculate compensating transforms for children ---
        compensations = []
        for (
            child,
            original_world_transform,
        ) in child_original_world_transforms.items():
            # To keep the child's world position the same, its new local
            # transform must be relative to the group's new transform.
            # new_local = inv(new_parent_world) * original_child_world
            new_child_local_transform = (
                inv_new_group_world @ original_world_transform
            )
            compensations.append((child, new_child_local_transform))

        return new_group_local_transform, compensations

    def sync_with_model(
        self, *args, origin: Optional[DocItem] = None, **kwargs
    ):
        """
        Reconciles child elements (WorkPieceElement, GroupElement) with the
        state of the Group model.
        """
        if not self.data or not self.canvas:
            return

        model_children = set(self.data.children)
        current_elements = self.children[:]
        current_element_data = {elem.data for elem in current_elements}

        # Remove elements for items no longer in the group
        for elem in current_elements:
            if elem.data not in model_children:
                elem.remove()

        # Add elements for new items in the group
        items_to_add = model_children - current_element_data
        for item_data in items_to_add:
            child_elem = None
            if isinstance(item_data, WorkPiece):
                child_elem = WorkPieceElement(
                    workpiece=item_data, canvas=self.canvas, selectable=True
                )
            elif isinstance(item_data, Group):
                child_elem = GroupElement(
                    group=item_data, canvas=self.canvas, selectable=True
                )

            if child_elem:
                self.add(child_elem)

        if self.canvas:
            self.canvas.queue_draw()

    def draw(self, ctx: cairo.Context):
        """
        Draws a crisp, dashed bounding box for the group directly in
        device space (pixels) to avoid scaling and alignment issues.
        """
        # Don't draw the bounding box if the group is selected, as the
        # canvas selection overlay will be drawn instead.
        if self.selected or not self.canvas:
            return

        # 1. Calculate the final transformation from the group's local 1x1
        #    space to the screen's pixel space.
        transform_to_screen = (
            self.canvas.view_transform @ self.get_world_transform()
        )

        # 2. Get the screen coordinates of the group's four corners.
        corners = [
            transform_to_screen.transform_point((0, 0)),
            transform_to_screen.transform_point((1, 0)),
            transform_to_screen.transform_point((1, 1)),
            transform_to_screen.transform_point((0, 1)),
        ]

        ctx.save()

        # 3. IMPORTANT: Reset the context's transformation matrix. We are now
        #    drawing directly in pixels, ignoring the element's transform.
        ctx.identity_matrix()

        # 4. Define line properties in simple, unscaled pixel units.
        ctx.set_source_rgba(0.5, 0.7, 1.0, 0.9)
        ctx.set_line_width(1.0)
        ctx.set_dash([4.0, 2.0])

        # 5. Build the path using the calculated screen coordinates.
        #    We add 0.5 to each coordinate to "pixel-snap" the line to the
        #    center of the pixels, preventing anti-aliasing artifacts.
        p1_x, p1_y = corners[0]
        ctx.move_to(round(p1_x) + 0.5, round(p1_y) + 0.5)

        for x, y in corners[1:]:
            ctx.line_to(round(x) + 0.5, round(y) + 0.5)

        ctx.close_path()
        ctx.stroke()

        ctx.restore()
