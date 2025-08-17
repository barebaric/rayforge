from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .item import DocItem
from .matrix import Matrix
from .workpiece import WorkPiece


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

    @property
    def all_workpieces(self) -> List["WorkPiece"]:
        """
        Recursively finds and returns a flattened list of all WorkPiece
        objects contained within this layer, including those inside groups.
        """
        return self.get_descendants(of_type=WorkPiece)

    def to_dict(self) -> Dict:
        """Serializes the Group and its children to a dictionary."""
        return {
            "uid": self.uid,
            "type": "group",  # Discriminator for deserialization
            "name": self.name,
            "matrix": self.matrix.to_list(),
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Group":
        """Deserializes a dictionary into a Group instance."""
        new_group = cls(name=data.get("name", "Group"))
        new_group.uid = data["uid"]
        new_group.matrix = Matrix.from_list(data["matrix"])

        for child_data in data.get("children", []):
            # Use the 'type' to decide which class to instantiate
            if child_data.get("type") == "group":
                child_item = Group.from_dict(child_data)
            else:
                # Assume WorkPiece if type is not 'group' or is missing
                child_item = WorkPiece.from_dict(child_data)
            new_group.add_child(child_item)

        return new_group

    @staticmethod
    def _calculate_world_bbox(
        items: List[DocItem],
    ) -> Optional[Tuple[float, float, float, float]]:
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
        a list of items.

        This is a pure calculation method; it does not modify the
        document tree.

        Args:
            items_to_group: The items that will be placed into the new group.
            parent: The prospective parent of the new group (e.g., a Layer).

        Returns:
            A GroupingResult object containing the configured (but not
            parented) new group and the calculated local matrices for its
            children, or None if no items are provided.
        """
        if not items_to_group:
            return None

        # 1. Capture original world transforms and calculate bounding box.
        original_world_transforms = {
            item.uid: item.get_world_transform() for item in items_to_group
        }
        bbox = cls._calculate_world_bbox(items_to_group)
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
