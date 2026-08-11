from __future__ import annotations

from gettext import gettext as _
from typing import Any

from raygeo.ops.assembly.spiral import SpiralSpec

from .cnc_assembler_step import CncAssemblerStep


class FlatSpiralStep(CncAssemblerStep):
    ASSEMBLER_NAME = "spiral"
    TYPELABEL = _("Flat Spiral")

    def build_spec(self, workpiece) -> SpiralSpec:
        part = workpiece.to_part()
        if part is not None:
            sr = part.stock_region
            cx = sum(p[0] for p in sr.boundary) / len(sr.boundary)
            cy = sum(p[1] for p in sr.boundary) / len(sr.boundary)
            max_r = max(
                ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5
                for p in sr.boundary
            )
        else:
            cx = workpiece.size[0] / 2.0
            cy = workpiece.size[1] / 2.0
            max_r = min(cx, cy) * 0.8
        return SpiralSpec(
            center=(cx, cy),
            z=self.target_depth,
            start_radius=self.tool_diameter / 2.0 * 1.5,
            end_radius=max_r * 0.9,
            revolutions=3.0,
        )

    def assembler_token_params(self, machine, workpiece) -> dict[str, Any]:
        return {
            "tool_diameter": self.tool_diameter,
            "target_depth": self.target_depth,
        }
