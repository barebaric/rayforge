from __future__ import annotations

from gettext import gettext as _
from typing import Any

from raygeo.ops.assembly.slot import SlotSpec

from .cnc_assembler_step import CncAssemblerStep


class SlotStep(CncAssemblerStep):
    ASSEMBLER_NAME = "slot"
    TYPELABEL = _("Slot")

    def build_spec(self, workpiece) -> SlotSpec:
        part = workpiece.to_part()
        if part is not None:
            sr = part.stock_region
            cx = sum(p[0] for p in sr.boundary) / len(sr.boundary)
            cy = sum(p[1] for p in sr.boundary) / len(sr.boundary)
        else:
            cx = workpiece.size[0] / 2.0
            cy = workpiece.size[1] / 2.0
        half_len = self.tool_diameter * 2
        return SlotSpec(
            carrier=[(cx - half_len, cy), (cx + half_len, cy)],
            tool_radius=self.tool_diameter / 2.0,
            target_z=self.target_depth,
        )

    def assembler_token_params(self, machine, workpiece) -> dict[str, Any]:
        return {
            "tool_diameter": self.tool_diameter,
            "target_depth": self.target_depth,
        }
