from typing import TYPE_CHECKING, Dict, Optional

from ..core.geo import Point3D
from .assembly import Assembly, JointType, Link, LinkRole
from .driver.driver import Axis

if TYPE_CHECKING:
    from ..simulator.machine_state import MachineState


def _build_cartesian_assembly() -> Assembly:
    return Assembly(
        [
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
            Link(
                "laser_head",
                parent="gantry_y",
                joint_type=JointType.PRISMATIC,
                joint_axis=(0.0, 0.0, 1.0),
                driver_axis=Axis.Z,
                role=LinkRole.HEAD,
            ),
        ]
    )


def _build_rotary_assembly(rotary_diameter: float) -> Assembly:
    asm = Assembly(
        [
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
            Link(
                "laser_head",
                parent="gantry_y",
                joint_type=JointType.PRISMATIC,
                joint_axis=(0.0, 0.0, 1.0),
                driver_axis=Axis.Z,
                role=LinkRole.HEAD,
            ),
            Link("rotary_base", parent="base", joint_type=JointType.FIXED),
            Link(
                "rotary_chuck",
                parent="rotary_base",
                joint_type=JointType.REVOLUTE,
                joint_axis=(1.0, 0.0, 0.0),
                driver_axis=Axis.Y,
                role=LinkRole.CHUCK,
            ),
        ]
    )
    asm.set_rotary_diameter(rotary_diameter)
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
        return Kinematics(_build_rotary_assembly(rotary_diameter))
    return Kinematics(_build_cartesian_assembly())
