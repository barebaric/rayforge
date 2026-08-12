from gettext import gettext as _
from typing import Any, cast

from raygeo.ops.assembly.toroid import ToroidalClearSpec

from rayforge.core.varset import LengthVar, VarSet

from .cnc_assembler_step import CncAssemblerStep


class ToroidalClearStep(CncAssemblerStep):
    ASSEMBLER_NAME = "toroidal_clear"
    TYPELABEL = _("Toroidal Clear")
    uses_global_state = True

    @classmethod
    def recipe_varset(cls) -> VarSet:
        return VarSet(
            vars=[
                *CncAssemblerStep.recipe_varset().vars,
                LengthVar(
                    key="step_over",
                    label=_("Step Over"),
                    default=2.0,
                    min_val=0.1,
                ),
            ]
        )

    def __init__(self, name=None, typelabel=None):
        super().__init__(name=name, typelabel=typelabel)
        self.step_over: float = 2.0

    def build_spec(self, workpiece) -> ToroidalClearSpec:
        part = workpiece.to_part()
        if part is not None:
            sr = part.stock_region
            cx = sum(p[0] for p in sr.boundary) / len(sr.boundary)
            cy = sum(p[1] for p in sr.boundary) / len(sr.boundary)
        else:
            cx = workpiece.size[0] / 2.0
            cy = workpiece.size[1] / 2.0
        r = self.tool_diameter / 2.0 * 1.5
        carrier = [(cx, cy), (cx + self.step_over * 2, cy)]
        return ToroidalClearSpec(
            carrier=carrier,
            start=(cx, cy, 0.0),
            target_z=self.target_depth,
            tool_radius=r,
            step_over=self.step_over,
        )

    def assembler_token_params(self, machine, workpiece) -> dict[str, Any]:
        return {
            "tool_diameter": self.tool_diameter,
            "target_depth": self.target_depth,
            "depth_per_pass": self.depth_per_pass,
            "step_over": self.step_over,
        }

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["step_over"] = self.step_over
        return result

    @classmethod
    def from_dict(cls, data) -> "ToroidalClearStep":
        step = cast("ToroidalClearStep", super().from_dict(data))
        step.step_over = data.get("step_over", step.step_over)
        return step

    @classmethod
    def _serialized_keys(cls) -> frozenset[str]:
        return super()._serialized_keys() | frozenset({"step_over"})
