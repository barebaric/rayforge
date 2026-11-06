from gettext import gettext as _
from typing import TYPE_CHECKING, Any, cast

from rayforge.core.capability import MachineCapability
from rayforge.core.varset import AngleVar, IntVar, LengthVar, VarSet

from ..transformers.tangential_knife_transformer import (
    TangentialKnifeTransformer,
)
from .knife_cut_step import KnifeCutStep

if TYPE_CHECKING:
    from raygeo.ops import Ops


class TangentialKnifeCutStep(KnifeCutStep):
    """Cuts vector contours with a rotary tangential knife.

    The knife's A axis follows the path heading; the blade lifts for
    turns beyond the angle tolerance and for arcs tighter than the
    radius tolerance.
    """

    REQUIRED_MACHINE_CAPS = frozenset({MachineCapability.TANGENTIAL_KNIFE})
    ASSEMBLER_NAME = "tangential_knife"
    KNIFE_TRANSFORMER_NAME = "TangentialKnifeTransformer"
    HEAD_CAPABILITY = MachineCapability.TANGENTIAL_KNIFE
    TYPELABEL = _("Tangential Knife Cut")
    ICON = "step-tangential-knife-symbolic"
    uses_global_state = True

    def __init__(self, name=None, typelabel=None):
        self.angle_tolerance_deg: float = 30.0
        self.radius_tolerance_mm: float = 1.0
        self.spindle_rpm: int = 10000
        super().__init__(name=name, typelabel=typelabel)

    @classmethod
    def _default_knife_transformer(cls) -> TangentialKnifeTransformer:
        return TangentialKnifeTransformer()

    def _knife_transformer_instance(self) -> TangentialKnifeTransformer:
        return TangentialKnifeTransformer(
            angle_tolerance_deg=self.angle_tolerance_deg,
            radius_tolerance_mm=self.radius_tolerance_mm,
            safe_z=self.safe_z,
        )

    def set_angle_tolerance_deg(self, value: float):
        value = min(180.0, max(0.0, float(value)))
        if self.angle_tolerance_deg == value:
            return
        self.angle_tolerance_deg = value
        self.sync_knife_transformer()
        self.updated.send(self)

    def set_radius_tolerance_mm(self, value: float):
        value = max(0.0, float(value))
        if self.radius_tolerance_mm == value:
            return
        self.radius_tolerance_mm = value
        self.sync_knife_transformer()
        self.updated.send(self)

    def set_spindle_rpm(self, value: int):
        value = int(value)
        if self.spindle_rpm == value:
            return
        self.spindle_rpm = value
        self.updated.send(self)

    def create_initial_ops(self) -> "Ops":
        ops = super().create_initial_ops()
        # The blade motor runs clockwise only (M3); M4 would reverse it.
        ops.set_spindle_rpm(self.spindle_rpm)
        return ops

    @classmethod
    def recipe_varset(cls) -> VarSet:
        return VarSet(
            vars=[
                *KnifeCutStep.recipe_varset().vars,
                AngleVar(
                    key="angle_tolerance_deg",
                    label=_("Angle Tolerance"),
                    description=_(
                        "Maximum heading change rotated with the "
                        "knife down; larger changes lift the knife"
                    ),
                    default=30.0,
                    min_val=0.0,
                    max_val=180.0,
                ),
                LengthVar(
                    key="radius_tolerance_mm",
                    label=_("Radius Tolerance"),
                    description=_(
                        "Arcs tighter than this radius are cut with "
                        "the knife lifted"
                    ),
                    default=1.0,
                    min_val=0.0,
                    max_val=50.0,
                ),
                IntVar(
                    key="spindle_rpm",
                    label=_("Spindle RPM"),
                    description=_("Blade motor speed (M3)"),
                    default=10000,
                    min_val=100,
                    max_val=60000,
                ),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "angle_tolerance_deg": self.angle_tolerance_deg,
                "radius_tolerance_mm": self.radius_tolerance_mm,
                "spindle_rpm": self.spindle_rpm,
            }
        )
        return result

    @classmethod
    def from_dict(cls, data) -> "TangentialKnifeCutStep":
        step = cast("TangentialKnifeCutStep", super().from_dict(data))
        step.angle_tolerance_deg = min(
            180.0,
            max(
                0.0,
                float(
                    data.get("angle_tolerance_deg", step.angle_tolerance_deg)
                ),
            ),
        )
        step.radius_tolerance_mm = max(
            0.0,
            float(data.get("radius_tolerance_mm", step.radius_tolerance_mm)),
        )
        step.spindle_rpm = int(data.get("spindle_rpm", step.spindle_rpm))
        return step

    @classmethod
    def _serialized_keys(cls) -> frozenset[str]:
        return super()._serialized_keys() | frozenset(
            {
                "angle_tolerance_deg",
                "radius_tolerance_mm",
                "spindle_rpm",
            }
        )

    def assembler_token_params(self, machine, workpiece) -> dict[str, Any]:
        return {
            **super().assembler_token_params(machine, workpiece),
            "angle_tolerance_deg": self.angle_tolerance_deg,
            "radius_tolerance_mm": self.radius_tolerance_mm,
        }
