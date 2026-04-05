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
from .driver.driver import Axis

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

    def get_links_by_role(self, role: LinkRole) -> List[Link]:
        names = self._roles.get(role, [])
        return [self._link_map[n] for n in names]

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
        chuck_names = set(self._roles.get(LinkRole.CHUCK, []))
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
                if child_name in chuck_names:
                    diameter = self._chuck_diameters.get(child_name)
                    if diameter:
                        circumference = diameter * math.pi
                        angle_rad = (angle / circumference) * 2 * math.pi
                    else:
                        angle_rad = math.radians(angle)
                else:
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

    def head_positions(self, state: "MachineState") -> Dict[str, Point3D]:
        head_names = self._roles.get(LinkRole.HEAD, [])
        if not head_names:
            raise ValueError("Assembly has no links with role HEAD")
        poses = self.forward_kinematics(state)
        return {name: poses[name][0] for name in head_names}

    def chuck_angles(self, state: "MachineState") -> Dict[str, float]:
        chuck_names = self._roles.get(LinkRole.CHUCK, [])
        if not chuck_names:
            return {}
        result: Dict[str, float] = {}
        for name in chuck_names:
            diameter = self._chuck_diameters.get(name)
            if not diameter:
                result[name] = 0.0
                continue
            circumference = diameter * math.pi
            chuck = self._link_map[name]
            assert chuck.driver_axis is not None
            y = state.axes.get(chuck.driver_axis, 0.0)
            result[name] = (y / circumference) * 2 * math.pi
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
