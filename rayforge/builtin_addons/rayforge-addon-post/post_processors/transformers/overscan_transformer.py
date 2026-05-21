from __future__ import annotations

import logging
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from raygeo.ops import Ops
from raygeo.ops.types import CommandCategory, CommandType, SectionType

from rayforge.pipeline.transformer.base import ExecutionPhase, OpsTransformer
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from raygeo import Geometry

    from rayforge.core.workpiece import WorkPiece

logger = logging.getLogger(__name__)


class OverscanTransformer(OpsTransformer):
    """
    Intelligently rewrites raster line patterns to include overscan for
    machine acceleration and deceleration, ensuring constant engraving
    velocity.

    This transformer operates only on commands within a `RASTER_FILL` section.
    It identifies a raster line (a MoveTo followed by cutting commands) and
    replaces it with a physically correct toolpath that includes lead-in and
    lead-out moves at zero power.
    """

    def __init__(
        self, enabled: bool = True, distance_mm: float = 2.0, auto: bool = True
    ):
        super().__init__(enabled=enabled)
        self._distance_mm: float = 0.0
        self.distance_mm = distance_mm
        self._auto: bool = auto

    @staticmethod
    def calculate_auto_distance(
        step_speed: int, max_acceleration: int
    ) -> float:
        """
        Calculate the optimal overscan distance based on step speed and machine
        acceleration with a safety factor of 2.

        Formula: distance = (speed²) / (2 * acceleration * safety_factor)
        Where safety_factor = 2 for additional safety margin

        Args:
            step_speed: The cutting speed in mm/min
            max_acceleration: The maximum machine acceleration in mm/s²

        Returns:
            The calculated overscan distance in millimeters
        """
        # Convert speed from mm/min to mm/s for the calculation
        speed_mm_per_sec = step_speed / 60.0

        # Safety factor of 2 as specified in requirements
        safety_factor = 2.0

        # Calculate distance using physics formula with safety factor
        # d = v² / (2 * a * safety_factor)
        distance_mm = (speed_mm_per_sec**2) / (
            2 * max_acceleration * safety_factor
        )

        # Ensure minimum distance for practical purposes
        return max(0.5, distance_mm)

    @property
    def execution_phase(self) -> ExecutionPhase:
        """
        Overscan must run before path optimization to ensure travel moves
        are planned between the final, extended endpoints of the toolpaths.
        """
        return ExecutionPhase.POST_PROCESSING

    @property
    def distance_mm(self) -> float:
        return self._distance_mm

    @distance_mm.setter
    def distance_mm(self, value: float):
        new_value = max(0.0, float(value))
        if not math.isclose(self._distance_mm, new_value):
            self._distance_mm = new_value
            self.changed.send(self)

    @property
    def auto(self) -> bool:
        return self._auto

    @auto.setter
    def auto(self, value: bool):
        if self._auto != bool(value):
            self._auto = bool(value)
            self.changed.send(self)

    @property
    def label(self) -> str:
        return _("Overscan")

    @property
    def description(self) -> str:
        return _("Extends raster lines to ensure constant engraving speed.")

    def run(
        self,
        ops: Ops,
        workpiece: Optional[WorkPiece] = None,
        context: Optional[ProgressContext] = None,
        stock_geometries: Optional[List["Geometry"]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or math.isclose(self.distance_mm, 0.0):
            return

        # Preload state only when we are actually going to process commands.
        ops.preload_state()

        new_ops = Ops()
        line_buffer: List[int] = []
        in_raster_section = False

        def _process_buffer():
            nonlocal line_buffer
            if line_buffer:
                self._rewrite_buffered_line(new_ops, ops, line_buffer)
                line_buffer = []

        for i in range(ops.len()):
            ct = ops.command_type(i)

            is_start = ct == CommandType.OPS_SECTION_START
            if is_start:
                sec_type, _ = ops.section_params(i)
                is_start = sec_type == SectionType.RASTER_FILL

            is_end = ct == CommandType.OPS_SECTION_END
            if is_end:
                sec_type, _ = ops.section_params(i)
                is_end = sec_type == SectionType.RASTER_FILL

            if is_start:
                _process_buffer()
                in_raster_section = True
                new_ops.transfer_command_from(ops, i)
            elif is_end:
                _process_buffer()
                in_raster_section = False
                new_ops.transfer_command_from(ops, i)
            elif not in_raster_section:
                new_ops.transfer_command_from(ops, i)
            else:  # Inside raster section
                if ct == CommandType.MOVE_TO:
                    _process_buffer()
                    line_buffer = [i]
                elif line_buffer:
                    # If a line has been started, append subsequent commands
                    # (could be state changes or cutting moves)
                    line_buffer.append(i)
                else:
                    # This command appeared without a preceding MoveTo
                    _process_buffer()
                    new_ops.transfer_command_from(ops, i)

        _process_buffer()
        ops.replace_with(new_ops)

    def _rewrite_buffered_line(
        self,
        new_ops: Ops,
        old_ops: Ops,
        indices: List[int],
    ) -> None:
        """
        Replaces a simple raster line pattern with a full toolpath
        including overscan lead-in and lead-out.
        """
        moving_indices = [
            j for j in indices if old_ops.category(j) == CommandCategory.MOVING
        ]

        if len(moving_indices) < 2 or (
            old_ops.command_type(moving_indices[0]) != CommandType.MOVE_TO
        ):
            for j in indices:
                new_ops.transfer_command_from(old_ops, j)
            return

        content_start_3d = old_ops.endpoint(moving_indices[0])
        content_end_3d = old_ops.endpoint(moving_indices[-1])

        start_x, start_y = content_start_3d[0], content_start_3d[1]
        end_x, end_y = content_end_3d[0], content_end_3d[1]

        if start_x == end_x and start_y == end_y:
            for j in indices:
                new_ops.transfer_command_from(old_ops, j)
            return

        dx = end_x - start_x
        dy = end_y - start_y
        original_length = math.hypot(dx, dy)
        if original_length < 1e-9:
            for j in indices:
                new_ops.transfer_command_from(old_ops, j)
            return

        v_dir_norm_x = dx / original_length
        v_dir_norm_y = dy / original_length

        overscan_start_x = start_x - self.distance_mm * v_dir_norm_x
        overscan_start_y = start_y - self.distance_mm * v_dir_norm_y
        overscan_end_x = end_x + self.distance_mm * v_dir_norm_x
        overscan_end_y = end_y + self.distance_mm * v_dir_norm_y

        overscan_start_3d = (
            overscan_start_x,
            overscan_start_y,
            content_start_3d[2],
        )
        overscan_end_3d = (
            overscan_end_x,
            overscan_end_y,
            content_end_3d[2],
        )

        # Case 1: Variable Power ScanLine - Handled by padding its data
        if len(indices) == 2 and (
            old_ops.command_type(indices[1]) == CommandType.SCAN_LINE
        ):
            scan_idx = indices[1]
            old_pv = bytes(old_ops.scanline_data(scan_idx))
            pixels_per_mm = (
                len(old_pv) / original_length if original_length > 0 else 0
            )
            num_pad_pixels = round(self.distance_mm * pixels_per_mm)
            pad_bytes = bytearray([0] * num_pad_pixels)

            padded_pv = pad_bytes + old_pv + pad_bytes

            new_ops.move_to(*overscan_start_3d)
            new_ops.scan_to(*overscan_end_3d, power_values=padded_pv)
            return

        # Case 2: Constant Power LineTo(s) - Handled by wrapping
        else:
            first_cut_idx = None
            for j in moving_indices[1:]:
                if old_ops.command_type(j) == CommandType.LINE_TO:
                    first_cut_idx = j
                    break

            if first_cut_idx is None:
                for j in indices:
                    new_ops.transfer_command_from(old_ops, j)
                return

            original_power = old_ops.preloaded_state(first_cut_idx).power

            new_ops.move_to(*overscan_start_3d)
            new_ops.set_power(0)
            new_ops.line_to(*content_start_3d)

            content_indices = indices[1:]

            if not content_indices or (
                old_ops.command_type(content_indices[0])
                != CommandType.SET_POWER
            ):
                new_ops.set_power(original_power)

            for j in content_indices:
                new_ops.transfer_command_from(old_ops, j)

            new_ops.set_power(0)
            new_ops.line_to(*overscan_end_3d)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "distance_mm": self.distance_mm,
            "auto": self.auto,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OverscanTransformer":
        return cls(
            enabled=data.get("enabled", True),
            distance_mm=data.get("distance_mm", 2.0),
            auto=data.get("auto", True),
        )
