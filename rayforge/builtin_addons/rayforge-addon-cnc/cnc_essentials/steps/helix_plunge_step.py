from gettext import gettext as _
from typing import Any

from raygeo.ops.assembly.helix import HelixSpec

from .cnc_assembler_step import CncAssemblerStep


class HelixPlungeStep(CncAssemblerStep):
    ASSEMBLER_NAME = "helix"
    TYPELABEL = _("Helix Plunge")

    def build_spec(self, workpiece) -> HelixSpec:
        part = workpiece.to_part()
        if part is not None:
            sr = part.stock_region
            cx = sum(p[0] for p in sr.boundary) / len(sr.boundary)
            cy = sum(p[1] for p in sr.boundary) / len(sr.boundary)
        else:
            cx = workpiece.size[0] / 2.0
            cy = workpiece.size[1] / 2.0
        return HelixSpec(
            center=(cx, cy),
            start_radius=self.tool_diameter / 2.0 * 1.5,
            z_start=0.0,
            z_end=self.target_depth,
            pitch=abs(self.target_depth / max(self.depth_per_pass, 0.1)),
        )

    def assembler_token_params(self, machine, workpiece) -> dict[str, Any]:
        return {
            "tool_diameter": self.tool_diameter,
            "target_depth": self.target_depth,
            "depth_per_pass": self.depth_per_pass,
        }
