from gettext import gettext as _
from typing import Any, cast

from raygeo.ops.assembly.profile import ProfileSpec

from rayforge.core.varset import LengthVar, VarSet

from .cnc_assembler_step import CncAssemblerStep


class ProfileOuterStep(CncAssemblerStep):
    ASSEMBLER_NAME = "profile_outer"
    TYPELABEL = _("Profile Outer")
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
                LengthVar(
                    key="step_length",
                    label=_("Step Length"),
                    default=0.6,
                    min_val=0.1,
                ),
                LengthVar(
                    key="wall_margin",
                    label=_("Wall Margin"),
                    default=0.0,
                    min_val=0.0,
                ),
            ]
        )

    def __init__(self, name=None, typelabel=None):
        super().__init__(name=name, typelabel=typelabel)
        self.step_over: float = 2.0
        self.step_length: float = 0.6
        self.wall_margin: float = 0.0

    def build_spec(self, workpiece) -> ProfileSpec:
        return ProfileSpec(
            kind="outer",
            tool_radius=self.tool_diameter / 2.0,
            step_over=self.step_over,
            step_length=self.step_length,
            target_z=self.target_depth,
            safe_z=self.safe_z,
            wall_margin=self.wall_margin,
        )

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "step_over": self.step_over,
                "step_length": self.step_length,
                "wall_margin": self.wall_margin,
            }
        )
        return result

    @classmethod
    def from_dict(cls, data) -> "ProfileOuterStep":
        step = cast("ProfileOuterStep", super().from_dict(data))
        step.step_over = data.get("step_over", step.step_over)
        step.step_length = data.get("step_length", step.step_length)
        step.wall_margin = data.get("wall_margin", step.wall_margin)
        return step

    @classmethod
    def _serialized_keys(cls) -> frozenset[str]:
        return super()._serialized_keys() | frozenset(
            {
                "step_over",
                "step_length",
                "wall_margin",
            }
        )
