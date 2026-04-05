from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from ..core.geo import Point3D
from .assembly import Assembly, JointType, Link, LinkRole
from .driver.driver import Axis

if TYPE_CHECKING:
    from ..simulator.machine_state import MachineState

RotarySpec = Tuple[Axis, float, np.ndarray, Optional[str]]
HeadSpec = Tuple[Optional[str], np.ndarray]


def build_cartesian_assembly(
    head_specs: Optional[List[HeadSpec]] = None,
) -> Assembly:
    if head_specs is None:
        head_specs = [(None, np.eye(4, dtype=np.float64))]
    links = [
        Link("base", parent=None, joint_type=JointType.FIXED),
        Link(
            "gantry_x",
            parent="base",
            joint_type=JointType.PRISMATIC,
            joint_axis=(1.0, 0.0, 0.0),
            driver_axis=Axis.X,
        ),
        Link(
            "gantry_y",
            parent="gantry_x",
            joint_type=JointType.PRISMATIC,
            joint_axis=(0.0, 1.0, 0.0),
            driver_axis=Axis.Y,
        ),
    ]
    for i, (model_id, transform) in enumerate(head_specs):
        links.append(
            Link(
                f"head_{i}",
                parent="gantry_y",
                joint_type=JointType.PRISMATIC,
                joint_axis=(0.0, 0.0, 1.0),
                driver_axis=Axis.Z,
                role=LinkRole.HEAD,
                model_id=model_id,
                model_transform=transform.copy(),
            )
        )
    return Assembly(links)


def build_rotary_assembly(
    rotary_diameter: float,
    head_specs: Optional[List[HeadSpec]] = None,
    rotary_specs: Optional[List[RotarySpec]] = None,
) -> Assembly:
    if rotary_specs is None:
        rotary_specs = [
            (Axis.Y, rotary_diameter, np.eye(4, dtype=np.float64), None)
        ]
    if head_specs is None:
        head_specs = [(None, np.eye(4, dtype=np.float64))]
    links = [
        Link("base", parent=None, joint_type=JointType.FIXED),
        Link(
            "gantry_x",
            parent="base",
            joint_type=JointType.PRISMATIC,
            joint_axis=(1.0, 0.0, 0.0),
            driver_axis=Axis.X,
        ),
        Link(
            "gantry_y",
            parent="gantry_x",
            joint_type=JointType.PRISMATIC,
            joint_axis=(0.0, 1.0, 0.0),
            driver_axis=Axis.Y,
        ),
    ]
    for i, (model_id, transform) in enumerate(head_specs):
        links.append(
            Link(
                f"head_{i}",
                parent="gantry_y",
                joint_type=JointType.PRISMATIC,
                joint_axis=(0.0, 0.0, 1.0),
                driver_axis=Axis.Z,
                role=LinkRole.HEAD,
                model_id=model_id,
                model_transform=transform.copy(),
            )
        )
    for j, (driver_axis, _, transform, model_id) in enumerate(rotary_specs):
        base_name = f"rotary_base_{j}"
        chuck_name = f"rotary_chuck_{j}"
        links.append(
            Link(
                base_name,
                parent="base",
                joint_type=JointType.FIXED,
                local_transform=transform.copy(),
            )
        )
        links.append(
            Link(
                chuck_name,
                parent=base_name,
                joint_type=JointType.REVOLUTE,
                joint_axis=(1.0, 0.0, 0.0),
                driver_axis=driver_axis,
                role=LinkRole.CHUCK,
                model_id=model_id,
            )
        )
    asm = Assembly(links)
    for j, (_, diameter, _, _) in enumerate(rotary_specs):
        asm.set_chuck_diameter(f"rotary_chuck_{j}", diameter)
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
    rotary_diameter: Optional[float] = None,
) -> Kinematics:
    if rotary_diameter is not None and rotary_diameter > 0:
        return Kinematics(build_rotary_assembly(rotary_diameter))
    return Kinematics(build_cartesian_assembly())
