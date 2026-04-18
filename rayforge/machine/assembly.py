import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np

from ..core.geo import Point3D
from ..core.ops.axis import Axis

if TYPE_CHECKING:
    from ..simulator.machine_state import MachineState


class JointType(Enum):
    FIXED = "fixed"
    PRISMATIC = "prismatic"
    REVOLUTE = "revolute"


class LinkRole(Enum):
    HEAD = "head"
    CHUCK = "chuck"


@dataclass
class Link:
    name: str
    parent: Optional[str]
    joint_type: JointType
    joint_axis: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    driver_axis: Optional[Axis] = None
    local_transform: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    model_id: Optional[str] = None
    model_transform: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    role: Optional[LinkRole] = None

    def __post_init__(self):
        if self.joint_type != JointType.FIXED and self.driver_axis is None:
            raise ValueError(
                f"Link '{self.name}': {self.joint_type.value} joint "
                f"requires a driver_axis"
            )
        if self.joint_type == JointType.FIXED and self.driver_axis is not None:
            raise ValueError(
                f"Link '{self.name}': FIXED joint cannot have a driver_axis"
            )


class Assembly:
    def __init__(self, links: List[Link]):
        if not links:
            raise ValueError("Assembly must have at least one link")
        self._links = links
        self._link_map: Dict[str, Link] = {}
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._roots: List[str] = []

        for link in links:
            if link.name in self._link_map:
                raise ValueError(f"Duplicate link name: '{link.name}'")
            self._link_map[link.name] = link

        for link in links:
            if link.parent is None:
                self._roots.append(link.name)
            elif link.parent not in self._link_map:
                raise ValueError(
                    f"Link '{link.name}' references unknown parent "
                    f"'{link.parent}'"
                )
            else:
                self._children[link.parent].append(link.name)

        if not self._roots:
            raise ValueError("Assembly must have exactly one root link")
        if len(self._roots) > 1:
            raise ValueError(
                f"Assembly must have exactly one root link, "
                f"found: {self._roots}"
            )

        self._validate_no_cycles()

        self._roles: Dict[LinkRole, List[str]] = {}
        self._chuck_diameters: Dict[str, float] = {}
        self._chuck_axis_offsets: Dict[str, np.ndarray] = {}
        self._index_roles()

    def _validate_no_cycles(self) -> None:
        visited: Set[str] = set()
        stack: Set[str] = set()

        def visit(name: str) -> None:
            if name in stack:
                raise ValueError(f"Cycle detected involving link '{name}'")
            if name in visited:
                return
            stack.add(name)
            for child in self._children.get(name, []):
                visit(child)
            stack.remove(name)
            visited.add(name)

        for root in self._roots:
            visit(root)

        all_names = set(self._link_map.keys())
        unreachable = all_names - visited
        if unreachable:
            raise ValueError(f"Unreachable links: {sorted(unreachable)}")

    def _index_roles(self) -> None:
        for link in self._links:
            if link.role is None:
                continue
            self._roles.setdefault(link.role, []).append(link.name)

    def set_chuck_diameter(self, chuck_name: str, diameter: float) -> None:
        self._chuck_diameters[chuck_name] = diameter

    def set_chuck_axis_offset(
        self, chuck_name: str, offset: np.ndarray
    ) -> None:
        self._chuck_axis_offsets[chuck_name] = offset.copy()

    @property
    def chuck_axis_offset(self) -> np.ndarray:
        chuck_names = self._roles.get(LinkRole.CHUCK, [])
        if not chuck_names:
            return np.zeros(3, dtype=np.float64)
        return self._chuck_axis_offsets.get(
            chuck_names[0], np.zeros(3, dtype=np.float64)
        )

    def set_rotary_diameter(self, diameter: float) -> None:
        for name in self._roles.get(LinkRole.CHUCK, []):
            self._chuck_diameters[name] = diameter

    @property
    def chuck_diameters(self) -> Dict[str, float]:
        return dict(self._chuck_diameters)

    @property
    def has_rotary(self) -> bool:
        return bool(self._roles.get(LinkRole.CHUCK))

    @property
    def rotary_diameter(self) -> Optional[float]:
        chuck_names = self._roles.get(LinkRole.CHUCK, [])
        if not chuck_names:
            return None
        return self._chuck_diameters.get(chuck_names[0])

    def get_link(self, name: str) -> Optional[Link]:
        return self._link_map.get(name)

    def get_links_by_role(self, role: LinkRole) -> List[Link]:
        names = self._roles.get(role, [])
        return [self._link_map[n] for n in names]

    def get_model_links(self) -> List[Link]:
        """Return all links that have a 3D model assigned."""
        return [link for link in self._links if link.model_id is not None]

    def model_world_transforms(
        self,
        state: "MachineState",
        wcs_offset: Point3D = (0.0, 0.0, 0.0),
    ) -> Dict[str, np.ndarray]:
        """Return a 4x4 world transform for each link with a 3D model.

        For prismatic joints, the transform includes the animated axis
        offset so models move with the gantry.  For revolute joints the
        transform is the static base pose (the joint rotation is
        typically visualized separately by a cylinder renderer).

        The link's ``model_transform`` is applied on top of the base
        pose and carries scale, rotation and offsets that are purely
        visual (e.g. focal-distance offset for laser heads).

        *wcs_offset* is applied in Z to non-rotary links (heads) so
        models sit at the correct work-coordinate height.
        """
        fk = self.forward_kinematics(state)
        chuck_names = set(self._roles.get(LinkRole.CHUCK, []))
        transforms: Dict[str, np.ndarray] = {}
        for link in self._links:
            if link.model_id is None:
                continue
            if link.joint_type == JointType.REVOLUTE:
                parent = self._link_map[link.parent] if link.parent else None
                if parent is not None and parent.name in fk:
                    pos_p, rot_p = fk[parent.name]
                    parent_t = np.eye(4, dtype=np.float64)
                    parent_t[:3, :3] = rot_p
                    parent_t[:3, 3] = [
                        pos_p[0],
                        pos_p[1],
                        pos_p[2],
                    ]
                    base = parent_t @ link.local_transform
                else:
                    base = link.local_transform.copy()
            else:
                pos, rot = fk[link.name]
                base = np.eye(4, dtype=np.float64)
                base[:3, :3] = rot
                base[:3, 3] = [pos[0], pos[1], pos[2]]
            t = base @ link.model_transform
            if link.name not in chuck_names:
                t[2, 3] += wcs_offset[2]
            transforms[link.name] = t
        return transforms

    def cylinder_base_transform(self) -> np.ndarray:
        """Return the static 4x4 cylinder base pose (no spin).

        Derives position and orientation from the rotary_base link's
        local_transform (= module.transform), with scale stripped and
        the stored chuck axis_offset applied.  Independent of machine
        state — the rotary_base is a FIXED child of the root.

        Returns:
            4x4 affine matrix (float64), rotation-only (no scale).
        """
        chuck_names = self._roles.get(LinkRole.CHUCK, [])
        if not chuck_names:
            return np.eye(4, dtype=np.float64)
        link = self._link_map.get(chuck_names[0])
        if link is None:
            return np.eye(4, dtype=np.float64)
        parent = self._link_map.get(link.parent) if link.parent else None
        if parent is not None:
            base = parent.local_transform.copy()
        else:
            base = link.local_transform.copy()

        rot3 = base[:3, :3].copy()
        for col in range(3):
            norm = np.linalg.norm(rot3[:, col])
            if norm > 1e-12:
                rot3[:, col] /= norm

        axis_offset = self.chuck_axis_offset
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = rot3
        result[:3, 3] = base[:3, 3] + rot3 @ axis_offset
        return result

    @property
    def cylinder_axis_index(self) -> int:
        """Index (0=X, 1=Y) of the cylinder axis in world frame.

        Derived from the rotary_base link's rotation matrix: the
        column corresponding to the chuck's driver axis gives the
        world-space cylinder direction.
        """
        chuck_names = self._roles.get(LinkRole.CHUCK, [])
        if not chuck_names:
            return 0
        chuck = self._link_map[chuck_names[0]]
        parent = self._link_map.get(chuck.parent) if chuck.parent else None
        if parent is not None:
            base = parent.local_transform
        else:
            base = chuck.local_transform
        rot3 = base[:3, :3].copy()
        for col in range(3):
            norm = np.linalg.norm(rot3[:, col])
            if norm > 1e-12:
                rot3[:, col] /= norm
        world_dir = rot3[:, 0]
        if abs(world_dir[1]) > abs(world_dir[0]):
            return 1
        return 0

    def head_rotary_positions(
        self,
        state: "MachineState",
        diameter: float,
        focal_distance: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Return head positions above the cylinder surface.

        Positions each HEAD link at the top of the cylinder at the
        correct along-cylinder offset, plus *focal_distance* above
        the surface.  Uses the static cylinder base pose (no spin).

        Args:
            state: Machine state (for FK head positions).
            diameter: Cylinder diameter.
            focal_distance: Height above the cylinder surface.

        Returns:
            Dict of {head_name: 3D world position (float64 array)}.
        """
        head_names = self._roles.get(LinkRole.HEAD, [])
        if not head_names:
            return {}

        chuck_names = self._roles.get(LinkRole.CHUCK, [])
        if not chuck_names:
            return {}

        radius = diameter / 2.0 if diameter > 0 else 0.0

        cyl_t = self.cylinder_base_transform()

        fk_heads = self.head_positions(state)

        result: Dict[str, np.ndarray] = {}
        for name in head_names:
            if name not in fk_heads:
                continue
            hx, hy, hz = fk_heads[name]

            local = np.array(
                [
                    hx,
                    0.0,
                    radius + focal_distance + hz,
                    1.0,
                ],
                dtype=np.float64,
            )
            world = cyl_t @ local
            result[name] = world[:3].copy()
        return result

    def cylinder_world_transform(
        self,
        state: "MachineState",
        axis_offset: np.ndarray,
    ) -> np.ndarray:
        """Return the 4x4 world transform for the rotary cylinder.

        The cylinder is a child of the chuck link in the kinematic
        chain.  The transform includes:

        1. Chuck base pose from FK (position + orientation, no scale)
        2. Axis offset in the chuck's local frame
        3. Revolute joint spin from the machine state

        Args:
            state: Current machine state (for FK + revolute angle).
            axis_offset: 3D offset from the module's mounting position.

        Returns:
            4x4 affine matrix (float64), rotation-only (no scale).
        """
        chuck_names = self._roles.get(LinkRole.CHUCK, [])
        if not chuck_names:
            return np.eye(4, dtype=np.float64)
        link = self._link_map.get(chuck_names[0])
        if link is None:
            return np.eye(4, dtype=np.float64)
        fk = self.forward_kinematics(state)
        parent = self._link_map.get(link.parent) if link.parent else None
        if parent is not None and parent.name in fk:
            pos_p, rot_p = fk[parent.name]
            parent_t = np.eye(4, dtype=np.float64)
            parent_t[:3, :3] = rot_p
            parent_t[:3, 3] = [pos_p[0], pos_p[1], pos_p[2]]
            base = parent_t @ link.local_transform
        else:
            base = link.local_transform.copy()

        result = np.eye(4, dtype=np.float64)
        rot3 = base[:3, :3].copy()
        for col in range(3):
            norm = np.linalg.norm(rot3[:, col])
            if norm > 1e-12:
                rot3[:, col] /= norm
        result[:3, :3] = rot3
        result[:3, 3] = base[:3, 3] + rot3 @ axis_offset

        if (
            link.driver_axis is not None
            and link.joint_type == JointType.REVOLUTE
        ):
            angle = state.axes.get(link.driver_axis, 0.0)
            angle_rad = math.radians(angle)
            axis = np.array(link.joint_axis, dtype=np.float64)
            rot = _rotation_matrix_4x4(axis, angle_rad)
            result = result @ rot

        return result

    def forward_kinematics(
        self, state: "MachineState"
    ) -> Dict[str, Tuple[Point3D, np.ndarray]]:
        result: Dict[str, Tuple[Point3D, np.ndarray]] = {}
        root = self._roots[0]
        root_link = self._link_map[root]
        root_transform = root_link.local_transform.copy()
        pos = root_transform[:3, 3]
        result[root] = (
            (float(pos[0]), float(pos[1]), float(pos[2])),
            root_transform[:3, :3].copy(),
        )
        self._fk_recursive(root, root_transform, state, result)
        return result

    def _fk_recursive(
        self,
        parent_name: str,
        parent_transform: np.ndarray,
        state: "MachineState",
        result: Dict[str, Tuple[Point3D, np.ndarray]],
    ) -> None:
        for child_name in self._children.get(parent_name, []):
            link = self._link_map[child_name]
            transform = parent_transform @ link.local_transform.copy()

            if link.joint_type == JointType.PRISMATIC:
                assert link.driver_axis is not None
                offset = state.axes.get(link.driver_axis, 0.0)
                axis = np.array(link.joint_axis, dtype=np.float64)
                transform[:3, 3] += offset * axis
            elif link.joint_type == JointType.REVOLUTE:
                assert link.driver_axis is not None
                angle = state.axes.get(link.driver_axis, 0.0)
                angle_rad = math.radians(angle)
                axis = np.array(link.joint_axis, dtype=np.float64)
                rot = _rotation_matrix_4x4(axis, angle_rad)
                transform = transform @ rot

            pos = transform[:3, 3]
            result[child_name] = (
                (float(pos[0]), float(pos[1]), float(pos[2])),
                transform[:3, :3].copy(),
            )
            self._fk_recursive(child_name, transform, state, result)

    def head_positions(
        self,
        state: "MachineState",
        wcs_offset: Point3D = (0.0, 0.0, 0.0),
    ) -> Dict[str, Point3D]:
        head_names = self._roles.get(LinkRole.HEAD, [])
        if not head_names:
            raise ValueError("Assembly has no links with role HEAD")
        poses = self.forward_kinematics(state)
        return {
            name: (
                poses[name][0][0],
                poses[name][0][1],
                poses[name][0][2] + wcs_offset[2],
            )
            for name in head_names
        }

    def chuck_angles(self, state: "MachineState") -> Dict[str, float]:
        chuck_names = self._roles.get(LinkRole.CHUCK, [])
        if not chuck_names:
            return {}
        result: Dict[str, float] = {}
        for name in chuck_names:
            chuck = self._link_map[name]
            assert chuck.driver_axis is not None
            angle = state.axes.get(chuck.driver_axis, 0.0)
            result[name] = math.radians(angle)
        return result


def _rotation_matrix_4x4(axis: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c
    x, y, z = axis
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = t * x * x + c
    m[0, 1] = t * x * y - s * z
    m[0, 2] = t * x * z + s * y
    m[1, 0] = t * x * y + s * z
    m[1, 1] = t * y * y + c
    m[1, 2] = t * y * z - s * x
    m[2, 0] = t * x * z - s * y
    m[2, 1] = t * y * z + s * x
    m[2, 2] = t * z * z + c
    return m
