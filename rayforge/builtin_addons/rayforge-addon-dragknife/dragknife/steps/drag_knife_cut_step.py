from gettext import gettext as _
from typing import Any, cast

from rayforge.core.capability import MachineCapability
from rayforge.core.varset import AngleVar, VarSet

from ..transformers.drag_knife_transformer import DragKnifeTransformer
from .knife_cut_step import KnifeCutStep


class DragKnifeCutStep(KnifeCutStep):
    """Cuts vector contours with a trailing-blade drag knife."""

    REQUIRED_MACHINE_CAPS = frozenset({MachineCapability.DRAG_KNIFE})
    ASSEMBLER_NAME = "drag_knife"
    KNIFE_TRANSFORMER_NAME = "DragKnifeTransformer"
    HEAD_CAPABILITY = MachineCapability.DRAG_KNIFE
    TYPELABEL = _("Drag Knife Cut")
    ICON = "step-drag-knife-symbolic"
    uses_global_state = True

    def __init__(self, name=None, typelabel=None):
        self.swivel_angle_deg: float = 45.0
        super().__init__(name=name, typelabel=typelabel)

    @classmethod
    def _default_knife_transformer(cls) -> DragKnifeTransformer:
        return DragKnifeTransformer()

    def _knife_transformer_instance(self) -> DragKnifeTransformer:
        return DragKnifeTransformer(
            offset_mm=self.offset_mm,
            swivel_angle_deg=self.swivel_angle_deg,
        )

    def set_swivel_angle_deg(self, value: float):
        value = min(180.0, max(0.0, float(value)))
        if self.swivel_angle_deg == value:
            return
        self.swivel_angle_deg = value
        self.sync_knife_transformer()
        self.updated.send(self)

    @classmethod
    def recipe_varset(cls) -> VarSet:
        return VarSet(
            vars=[
                *KnifeCutStep.recipe_varset().vars,
                AngleVar(
                    key="swivel_angle_deg",
                    label=_("Swivel Angle"),
                    description=_(
                        "Maximum corner angle pivoted with the blade "
                        "down; sharper corners lift the knife first"
                    ),
                    default=45.0,
                    min_val=0.0,
                    max_val=180.0,
                ),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({"swivel_angle_deg": self.swivel_angle_deg})
        return result

    @classmethod
    def from_dict(cls, data) -> "DragKnifeCutStep":
        step = cast("DragKnifeCutStep", super().from_dict(data))
        step.swivel_angle_deg = min(
            180.0,
            max(
                0.0, float(data.get("swivel_angle_deg", step.swivel_angle_deg))
            ),
        )
        return step

    @classmethod
    def _serialized_keys(cls) -> frozenset[str]:
        return super()._serialized_keys() | frozenset({"swivel_angle_deg"})

    def assembler_token_params(self, machine, workpiece) -> dict[str, Any]:
        return {
            **super().assembler_token_params(machine, workpiece),
            "swivel_angle_deg": self.swivel_angle_deg,
        }
