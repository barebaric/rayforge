from gettext import gettext as _
from typing import Any

from raygeo.ops.assembly.ramp import RampSpec

from .cnc_assembler_step import CncAssemblerStep


class RampEntryStep(CncAssemblerStep):
    ASSEMBLER_NAME = "ramp"
    TYPELABEL = _("Ramp Entry")

    def build_spec(self, workpiece) -> RampSpec:
        part = workpiece.to_part()
        if part is not None:
            sr = part.stock_region
            cx = sum(p[0] for p in sr.boundary) / len(sr.boundary)
            cy = sum(p[1] for p in sr.boundary) / len(sr.boundary)
        else:
            cx = workpiece.size[0] / 2.0
            cy = workpiece.size[1] / 2.0
        r = self.tool_diameter / 2.0
        return RampSpec(
            start=(cx - r, cy),
            end=(cx + r, cy),
            z_start=0.0,
            z_end=self.target_depth,
        )

    def assembler_token_params(self, machine, workpiece) -> dict[str, Any]:
        return {
            "tool_diameter": self.tool_diameter,
            "target_depth": self.target_depth,
        }
