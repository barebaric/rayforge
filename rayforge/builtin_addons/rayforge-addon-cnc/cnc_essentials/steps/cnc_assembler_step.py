from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, Tuple, cast

from raygeo.cnc.execution.specs import ComputePayload
from raygeo.ops.assembly import Assembler
from raygeo.ops.part import Part

from rayforge.core.capability import MachineCapability
from rayforge.core.step import Step

from ..capabilities import MILL

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.machine import Machine


class CncAssemblerStep(Step):
    """Base class for CNC assembler-driven steps.

    Subclasses set ``ASSEMBLER_NAME`` and override
    ``build_spec`` to construct the matching raygeo spec.
    """

    CAPABILITIES = (MILL,)
    REQUIRED_MACHINE_CAPS = frozenset({MachineCapability.MILL})
    TYPELABEL = "CNC Step"

    @property
    def show_general_settings(self) -> bool:
        return False

    # Phase 3 turns this on for steps that consume predecessor state.
    uses_global_state: ClassVar[bool] = False

    def __init__(self, name=None, typelabel=None):
        self.tool_diameter: float = 6.0
        self.spindle_rpm: int = 12000
        self.plunge_speed: int = 200
        self.target_depth: float = -5.0
        self.depth_per_pass: float = 1.0
        self.safe_z: float = 2.0
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)

    @classmethod
    def create(
        cls,
        context: "RayforgeContext",
        name=None,
        **kwargs,
    ) -> "CncAssemblerStep":
        machine = context.machine
        step = cls(name=name)
        step.per_workpiece_transformers_dicts = []
        step.per_step_transformers_dicts = []
        if machine is not None:
            default_head = machine.get_default_head()
            step.selected_head_uid = default_head.uid
            step.max_cut_speed = machine.max_cut_speed
            step.max_travel_speed = machine.max_travel_speed
        else:
            step.selected_head_uid = None
        return step

    def set_tool_diameter(self, diameter: float):
        if self.tool_diameter != diameter:
            self.tool_diameter = float(diameter)
            self.updated.send(self)

    def set_spindle_rpm(self, rpm: int):
        if self.spindle_rpm != rpm:
            self.spindle_rpm = int(rpm)
            self.updated.send(self)

    def set_plunge_speed(self, speed: int):
        if self.plunge_speed != speed:
            self.plunge_speed = int(speed)
            self.updated.send(self)

    def set_target_depth(self, depth: float):
        if self.target_depth != depth:
            self.target_depth = float(depth)
            self.updated.send(self)

    def set_depth_per_pass(self, depth: float):
        if self.depth_per_pass != depth:
            self.depth_per_pass = float(depth)
            self.updated.send(self)

    def set_safe_z(self, z: float):
        if self.safe_z != z:
            self.safe_z = float(z)
            self.updated.send(self)

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "tool_diameter": self.tool_diameter,
                "spindle_rpm": self.spindle_rpm,
                "plunge_speed": self.plunge_speed,
                "target_depth": self.target_depth,
                "depth_per_pass": self.depth_per_pass,
                "safe_z": self.safe_z,
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CncAssemblerStep":
        step = cast("CncAssemblerStep", super().from_dict(data))
        step.tool_diameter = data.get("tool_diameter", step.tool_diameter)
        step.spindle_rpm = data.get("spindle_rpm", step.spindle_rpm)
        step.plunge_speed = data.get("plunge_speed", step.plunge_speed)
        step.target_depth = data.get("target_depth", step.target_depth)
        step.depth_per_pass = data.get("depth_per_pass", step.depth_per_pass)
        step.safe_z = data.get("safe_z", step.safe_z)
        return step

    @classmethod
    def _serialized_keys(cls) -> frozenset[str]:
        return super()._serialized_keys() | frozenset(
            {
                "tool_diameter",
                "spindle_rpm",
                "plunge_speed",
                "target_depth",
                "depth_per_pass",
                "safe_z",
            }
        )

    def build_spec(self, workpiece: "WorkPiece") -> Any:
        """Return the raygeo assembler spec for this step.

        Subclasses must override.
        """
        raise NotImplementedError

    def populate_payload(self, payload, machine: "Machine"):
        super().populate_payload(payload, machine)
        # The renderer colours ops by power and treats zero as a "no cut"
        # state. Express the spindle's power level as the fraction of its
        # max RPM so a running spindle renders as a cut at the right intensity.
        payload.power = self._spindle_power_fraction(machine)

    def _spindle_power_fraction(self, machine: "Machine") -> float:
        """CNC power level in ``[0, 1]`` from the spindle's RPM ratio."""
        max_rpm = getattr(self.get_selected_head(machine), "max_rpm", None)
        if max_rpm:
            return min(1.0, self.spindle_rpm / max_rpm)
        return 1.0

    def build_compute_payload(
        self,
        machine: "Machine",
        workpiece: "WorkPiece",
    ) -> Tuple[Part, ComputePayload]:
        part = workpiece.to_part()
        if part is None:
            part = Part(size_mm=workpiece.size)
        spec = self.build_spec(workpiece)
        return part, ComputePayload(
            assembler=Assembler(spec),
            cut_speed=self.cut_speed,
        )

    def assembler_token_params(
        self,
        machine: "Machine",
        workpiece: "WorkPiece",
    ) -> Dict[str, Any]:
        """Return a JSON-serialisable view of ``build_spec``'s params.

        Default implementation reflects the common CNC attributes;
        subclasses extend it with their own.
        """
        return {
            "tool_diameter": self.tool_diameter,
            "target_depth": self.target_depth,
            "depth_per_pass": self.depth_per_pass,
            "safe_z": self.safe_z,
        }
