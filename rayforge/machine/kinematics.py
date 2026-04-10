from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from ..core.geo import Point3D
from ..core.ops.axis import Axis
from .assembly import Assembly, JointType, Link, LinkRole
from .models.axis import AxisSet

if TYPE_CHECKING:
    from ..simulator.machine_state import MachineState

HeadSpec = Tuple[Optional[str], np.ndarray]


def _axis_direction(axis_letter: Axis) -> Tuple[float, float, float]:
    mapping = {
        Axis.X: (1.0, 0.0, 0.0),
        Axis.Y: (0.0, 1.0, 0.0),
        Axis.Z: (0.0, 0.0, 1.0),
    }
    return mapping.get(axis_letter, (0.0, 0.0, 0.0))


def _joint_axis_for_rotary(axis_letter: Axis) -> Tuple[float, float, float]:
    mapping = {
        Axis.A: (1.0, 0.0, 0.0),
        Axis.B: (0.0, 1.0, 0.0),
        Axis.C: (0.0, 0.0, 1.0),
    }
    return mapping.get(axis_letter, (1.0, 0.0, 0.0))


def build_assembly(
    axis_set: AxisSet,
    head_specs: Optional[List[HeadSpec]] = None,
    rotary_modules=None,
) -> Assembly:
    if head_specs is None:
        head_specs = [(None, np.eye(4, dtype=np.float64))]
    links = [
        Link("base", parent=None, joint_type=JointType.FIXED),
    ]

    parent = "base"
    for axis_config in axis_set.linear_axes:
        if axis_config.letter == Axis.Z:
            continue
        name = f"gantry_{axis_config.letter.label}"
        links.append(
            Link(
                name,
                parent=parent,
                joint_type=JointType.PRISMATIC,
                joint_axis=_axis_direction(axis_config.letter),
                driver_axis=axis_config.letter,
            )
        )
        parent = name

    for i, (model_id, transform) in enumerate(head_specs):
        links.append(
            Link(
                f"head_{i}",
                parent=parent,
                joint_type=JointType.PRISMATIC,
                joint_axis=(0.0, 0.0, 1.0),
                driver_axis=Axis.Z,
                role=LinkRole.HEAD,
                model_id=model_id,
                model_transform=transform.copy(),
            )
        )

    if rotary_modules is not None:
        active_axes = {rm.axis for rm in rotary_modules.values()}
    else:
        active_axes = set()

    rotary_list = [
        ac for ac in axis_set.rotary_axes if ac.letter in active_axes
    ]
    for i, axis_config in enumerate(rotary_list):
        module = None
        if rotary_modules is not None:
            for rm in rotary_modules.values():
                if rm.axis == axis_config.letter:
                    module = rm
                    break
        base_name = f"rotary_base_{i}"
        chuck_name = f"rotary_chuck_{i}"
        links.append(
            Link(
                base_name,
                parent="base",
                joint_type=JointType.FIXED,
                local_transform=(
                    module.transform.copy()
                    if module
                    else np.eye(4, dtype=np.float64)
                ),
            )
        )
        links.append(
            Link(
                chuck_name,
                parent=base_name,
                joint_type=JointType.REVOLUTE,
                joint_axis=_joint_axis_for_rotary(axis_config.letter),
                driver_axis=axis_config.letter,
                role=LinkRole.CHUCK,
                model_id=module.model_id if module else None,
            )
        )

    asm = Assembly(links)
    for i, axis_config in enumerate(rotary_list):
        if axis_config.rotary_diameter:
            asm.set_chuck_diameter(
                f"rotary_chuck_{i}", axis_config.rotary_diameter
            )
    return asm


class Kinematics:
    """Translates MachineState into spatial data via an Assembly."""

    def __init__(self, assembly: Assembly):
        self._assembly = assembly

    @property
    def has_rotary(self) -> bool:
        return self._assembly.has_rotary

    @property
    def rotary_diameter(self) -> Optional[float]:
        return self._assembly.rotary_diameter

    def head_positions(self, state: "MachineState") -> Dict[str, Point3D]:
        return self._assembly.head_positions(state)

    def chuck_angles(self, state: "MachineState") -> Dict[str, float]:
        return self._assembly.chuck_angles(state)


def create_kinematics(
    axis_set: AxisSet,
    head_specs: Optional[List[HeadSpec]] = None,
    rotary_modules=None,
) -> Kinematics:
    return Kinematics(build_assembly(axis_set, head_specs, rotary_modules))
