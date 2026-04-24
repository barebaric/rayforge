from __future__ import annotations
import logging
from typing import List, Dict, Optional, Sequence, TYPE_CHECKING, Any, Tuple
from dataclasses import dataclass
import cairo
from .item import DocItem
from .matrix import Matrix
from .workpiece import WorkPiece
from .geo import Rect

if TYPE_CHECKING:
    from .layer import Layer

logger = logging.getLogger(__name__)


@dataclass
class GroupingResult:
    """A container for the results of the group creation calculation."""

    new_group: "Group"
    child_matrices: Dict[str, Matrix]


class Group(DocItem):
    """
    A DocItem that acts as a container for other DocItems (WorkPieces or
    other Groups), allowing them to be treated as a single unit for
    transformations.
    """

    def __init__(self, name: str = "Group"):
        """Initializes a Group instance."""
        super().__init__(name=name)
        self.extra: Dict[str, Any] = {}

    @property
    def layer(self) -> Optional["Layer"]:
        """Traverses the hierarchy to find the parent Layer."""
        from .layer import Layer  # Local import to avoid circular dependency

        p = self.parent
        while p:
            if isinstance(p, Layer):
                return p
            p = p.parent
        return None

    @property
    def all_workpieces(self) -> List["WorkPiece"]:
        """
        Recursively finds and returns a flattened list of all WorkPiece
        objects contained within this layer, including those inside groups.
        """
        return self.get_descendants(of_type=WorkPiece)

    @property
    def natural_size(self) -> Tuple[float, float]:
        return self.matrix.get_abs_scale()

    def render_to_pixels(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """
        Render all children into a single composite surface.

        Delegates to each child's own render_to_pixels, so nested
        groups are handled recursively.
        """
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)
        ctx.set_source_rgba(0, 0, 0, 0)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)

        for child in self.children:
            T = child.matrix

            corners = [
                T.transform_point(c) for c in [(0, 0), (1, 0), (1, 1), (0, 1)]
            ]
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            apparent_w = max(xs) - min(xs)
            apparent_h = max(ys) - min(ys)

            child_pixel_w = max(int(apparent_w * width), 1)
            child_pixel_h = max(int(apparent_h * height), 1)

            if not isinstance(child, (Group, WorkPiece)):
                continue

            child_surface = child.render_to_pixels(
                child_pixel_w, child_pixel_h
            )
            if child_surface is None:
                continue

            ctx.save()
            # Map group (0-1) Y-up to surface pixels Y-down.
            ctx.translate(0, height)
            ctx.scale(width, -height)
            # Apply child-to-group transform.
            ctx.transform(cairo.Matrix(*T.for_cairo()))
            # Flip Y for the image: image (0,0) is top-left in
            # Y-down, but T expects Y-up child coordinates.
            ctx.translate(0, 1)
            ctx.scale(
                1.0 / child_surface.get_width(),
                -1.0 / child_surface.get_height(),
            )
            ctx.set_source_surface(child_surface, 0, 0)
            ctx.paint()
            ctx.restore()

        return surface

    def to_dict(self) -> Dict:
        """Serializes the Group and its children to a dictionary."""
        result = {
            "uid": self.uid,
            "type": "group",  # Discriminator for deserialization
            "name": self.name,
            "matrix": self.matrix.to_list(),
            "children": [child.to_dict() for child in self.children],
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "Group":
        """Deserializes a dictionary into a Group instance."""
        known_keys = {"uid", "type", "name", "matrix", "children"}
        extra = {k: v for k, v in data.items() if k not in known_keys}

        new_group = cls(name=data.get("name", "Group"))
        new_group.uid = data["uid"]
        new_group.matrix = Matrix.from_list(data["matrix"])
        new_group.extra = extra

        for child_data in data.get("children", []):
            child_type = child_data.get("type")
            child_item = None

            # Whitelist: Only process types that are allowed to be in a group.
            if child_type == "group":
                child_item = Group.from_dict(child_data)
            # Handle WorkPiece and legacy files with no type field.
            elif child_type == "workpiece" or child_type is None:
                child_item = WorkPiece.from_dict(child_data)

            # Any other type (like 'stockitem') is implicitly ignored.
            if child_item:
                new_group.add_child(child_item)

        return new_group

    @staticmethod
    def _calculate_world_bbox(
        items: Sequence[DocItem],
    ) -> Optional[Rect]:
        """
        Calculates the union of the world-space bounding boxes for a list
        of DocItems.
        """
        if not items:
            return None

        all_corners = []
        for item in items:
            unit_corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
            world_transform = item.get_world_transform()
            all_corners.extend(
                [world_transform.transform_point(c) for c in unit_corners]
            )

        min_x = min(p[0] for p in all_corners)
        min_y = min(p[1] for p in all_corners)
        max_x = max(p[0] for p in all_corners)
        max_y = max(p[1] for p in all_corners)

        return min_x, min_y, max_x - min_x, max_y - min_y

    @classmethod
    def create_from_items(
        cls, items_to_group: List[DocItem], parent: DocItem
    ) -> Optional[GroupingResult]:
        """
        Factory method to create a new Group sized and positioned to enclose
        a list of items. Only groupable items (WorkPiece, Group) are included.

        This is a pure calculation method; it does not modify the
        document tree.

        Args:
            items_to_group: The items that will be placed into the new group.
            parent: The prospective parent of the new group (e.g., a Layer).

        Returns:
            A GroupingResult object containing the configured (but not
            parented) new group and the calculated local matrices for its
            children, or None if no valid items are provided.
        """
        # Whitelist the types that are allowed to be grouped.
        valid_items_to_group = [
            item
            for item in items_to_group
            if isinstance(item, (WorkPiece, Group))
        ]

        if not valid_items_to_group:
            return None

        # 1. Capture original world transforms and calculate bounding box.
        original_world_transforms = {
            item.uid: item.get_world_transform()
            for item in valid_items_to_group
        }
        bbox = cls._calculate_world_bbox(valid_items_to_group)
        if not bbox:
            return None
        world_x, world_y, world_w, world_h = bbox

        # 2. Prevent zero-sized groups which are mathematically problematic.
        world_w = max(world_w, 1e-9)
        world_h = max(world_h, 1e-9)

        # 3. Determine the new group's world transform based on the bbox.
        group_world_transform = Matrix.translation(
            world_x, world_y
        ) @ Matrix.scale(world_w, world_h)

        # 4. Calculate the group's local matrix relative to its future parent.
        parent_inv_world = parent.get_world_transform().invert()
        group_local_matrix = parent_inv_world @ group_world_transform

        # 5. Calculate the new local matrices for all children relative
        # to the group.
        group_inv_world = group_world_transform.invert()
        new_child_matrices = {
            uid: group_inv_world @ world_transform
            for uid, world_transform in original_world_transforms.items()
        }

        # 6. Create and configure the new group instance.
        new_group = cls(name="Group")
        new_group.matrix = group_local_matrix

        return GroupingResult(
            new_group=new_group, child_matrices=new_child_matrices
        )
