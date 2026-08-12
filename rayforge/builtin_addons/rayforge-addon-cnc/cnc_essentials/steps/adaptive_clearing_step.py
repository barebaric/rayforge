from gettext import gettext as _
from typing import TYPE_CHECKING, Any, cast

from raygeo.cnc.execution.specs import ComputePayload
from raygeo.ops.assembly.adaptive import AdaptiveClearingSpec
from raygeo.ops.part import Part

from rayforge.core.varset import FloatVar, LengthVar, VarSet

from .cnc_assembler_step import CncAssemblerStep

if TYPE_CHECKING:
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.machine import Machine


class AdaptiveClearStep(CncAssemblerStep):
    ASSEMBLER_NAME = "adaptive_clearing"  # matches PlanStep.kind
    TYPELABEL = _("Adaptive Clear")
    uses_global_state = True  # consumes predecessor cleared-area

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
                FloatVar(
                    key="max_deflection_deg",
                    label=_("Max Deflection"),
                    default=30.0,
                    min_val=0.0,
                    max_val=90.0,
                ),
                LengthVar(
                    key="wall_margin",
                    label=_("Wall Margin"),
                    default=0.0,
                    min_val=0.0,
                ),
                FloatVar(
                    key="area_tolerance",
                    label=_("Area Tolerance"),
                    default=1.0,
                    min_val=0.0,
                ),
            ]
        )

    def __init__(self, name=None, typelabel=None):
        super().__init__(name=name, typelabel=typelabel)
        self.step_over: float = 2.0
        self.step_length: float = 0.6
        self.max_deflection_deg: float = 30.0
        self.wall_margin: float = 0.0
        self.area_tolerance: float = 1.0

    def build_spec(self, workpiece) -> AdaptiveClearingSpec:
        return AdaptiveClearingSpec(
            tool_radius=self.tool_diameter / 2,
            step_over=self.step_over,
            step_length=self.step_length,
            target_z=self.target_depth,
            safe_z=self.safe_z,
            max_deflection_deg=self.max_deflection_deg,
            wall_margin=self.wall_margin,
            area_tolerance=self.area_tolerance,
        )

    def build_compute_payload(
        self,
        machine: "Machine",
        workpiece: "WorkPiece",
    ) -> tuple[Part, ComputePayload]:
        # Multi-pocket handling is done on the Rust side:
        # Part.from_geometry_multi_face exposes each pocket as a face,
        # the compute stage iterates them, and AdaptiveClearingSpec
        # splits each pocket into regions via find_regions.  No
        # Python-side seeding needed.
        return super().build_compute_payload(machine, workpiece)

    def assembler_token_params(self, machine, workpiece) -> dict[str, Any]:
        params = super().assembler_token_params(machine, workpiece)
        params.update(
            {
                "step_over": self.step_over,
                "step_length": self.step_length,
                "max_deflection_deg": self.max_deflection_deg,
                "wall_margin": self.wall_margin,
                "area_tolerance": self.area_tolerance,
            }
        )
        return params

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "step_over": self.step_over,
                "step_length": self.step_length,
                "max_deflection_deg": self.max_deflection_deg,
                "wall_margin": self.wall_margin,
                "area_tolerance": self.area_tolerance,
            }
        )
        return result

    @classmethod
    def from_dict(cls, data) -> "AdaptiveClearStep":
        step = cast("AdaptiveClearStep", super().from_dict(data))
        step.step_over = data.get("step_over", step.step_over)
        step.step_length = data.get("step_length", step.step_length)
        step.max_deflection_deg = data.get(
            "max_deflection_deg", step.max_deflection_deg
        )
        step.wall_margin = data.get("wall_margin", step.wall_margin)
        step.area_tolerance = data.get("area_tolerance", step.area_tolerance)
        return step

    @classmethod
    def _serialized_keys(cls) -> frozenset[str]:
        return super()._serialized_keys() | frozenset(
            {
                "step_over",
                "step_length",
                "max_deflection_deg",
                "wall_margin",
                "area_tolerance",
            }
        )
