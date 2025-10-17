from __future__ import annotations
import math
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Sequence, TYPE_CHECKING
from copy import deepcopy

from .base import OpsTransformer, ExecutionPhase
from ...core.ops import (
    Ops,
    Command,
    MovingCommand,
    MoveToCommand,
    LineToCommand,
    ScanLinePowerCommand,
    SetPowerCommand,
    SectionType,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
)

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...shared.tasker.proxy import BaseExecutionContext

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
        context: Optional[BaseExecutionContext] = None,
    ) -> None:
        if not self.enabled or math.isclose(self.distance_mm, 0.0):
            return

        # Preload state only when we are actually going to process commands.
        ops.preload_state()

        new_commands: List[Command] = []
        line_buffer: List[Command] = []
        in_raster_section = False

        def _process_buffer():
            nonlocal line_buffer
            if line_buffer:
                rewritten_line = self._rewrite_buffered_line(line_buffer)
                new_commands.extend(rewritten_line)
                line_buffer = []

        for cmd in ops:
            is_start = (
                isinstance(cmd, OpsSectionStartCommand)
                and cmd.section_type == SectionType.RASTER_FILL
            )
            is_end = (
                isinstance(cmd, OpsSectionEndCommand)
                and cmd.section_type == SectionType.RASTER_FILL
            )

            if is_start:
                _process_buffer()
                in_raster_section = True
                new_commands.append(cmd)
            elif is_end:
                _process_buffer()
                in_raster_section = False
                new_commands.append(cmd)
            elif not in_raster_section:
                new_commands.append(cmd)
            else:  # Inside raster section
                if isinstance(cmd, MoveToCommand):
                    _process_buffer()
                    line_buffer = [cmd]
                elif line_buffer:
                    # If a line has been started, append subsequent commands
                    # (could be state changes or cutting moves)
                    line_buffer.append(cmd)
                else:
                    # This command appeared without a preceding MoveTo
                    _process_buffer()
                    new_commands.append(cmd)

        _process_buffer()
        ops.commands = new_commands

    def _rewrite_buffered_line(
        self, buffer: List[Command]
    ) -> Sequence[Command]:
        """
        Replaces a simple raster line pattern with a full toolpath
        including overscan lead-in and lead-out.
        """
        moving_cmds = [c for c in buffer if isinstance(c, MovingCommand)]

        if len(moving_cmds) < 2 or not isinstance(
            moving_cmds[0], MoveToCommand
        ):
            return buffer

        content_start_3d = moving_cmds[0].end
        content_end_3d = moving_cmds[-1].end

        p_start = np.array(content_start_3d[:2])
        p_end = np.array(content_end_3d[:2])

        if np.allclose(p_start, p_end):
            return buffer

        v_dir = p_end - p_start
        original_length = np.linalg.norm(v_dir)
        if original_length < 1e-9:
            return buffer
        v_dir_norm = v_dir / original_length

        overscan_start_2d = p_start - self.distance_mm * v_dir_norm
        overscan_end_2d = p_end + self.distance_mm * v_dir_norm

        overscan_start_3d = (
            overscan_start_2d[0],
            overscan_start_2d[1],
            content_start_3d[2],
        )
        overscan_end_3d = (
            overscan_end_2d[0],
            overscan_end_2d[1],
            content_end_3d[2],
        )

        # Case 1: Variable Power ScanLine - Handled by padding its data
        if len(buffer) == 2 and isinstance(buffer[1], ScanLinePowerCommand):
            scan_cmd = deepcopy(buffer[1])
            pixels_per_mm = (
                len(scan_cmd.power_values) / original_length
                if original_length > 0
                else 0
            )
            num_pad_pixels = round(self.distance_mm * pixels_per_mm)
            pad_bytes = bytearray([0] * num_pad_pixels)

            scan_cmd.power_values = (
                pad_bytes + scan_cmd.power_values + pad_bytes
            )
            scan_cmd.end = overscan_end_3d

            return [MoveToCommand(overscan_start_3d), scan_cmd]

        # Case 2: Constant Power LineTo(s) - Handled by wrapping
        else:
            first_cut_cmd = next(
                (cmd for cmd in moving_cmds if cmd.is_cutting_command()), None
            )

            if not first_cut_cmd or not first_cut_cmd.state:
                return buffer

            original_power = first_cut_cmd.state.power
            rewritten_commands: List[Command] = [
                MoveToCommand(overscan_start_3d),
            ]
            rewritten_commands.extend(
                [SetPowerCommand(0), LineToCommand(content_start_3d)]
            )

            content_cmds = buffer[1:]

            if not content_cmds or not isinstance(
                content_cmds[0], SetPowerCommand
            ):
                rewritten_commands.append(SetPowerCommand(original_power))

            rewritten_commands.extend(content_cmds)

            rewritten_commands.extend(
                [SetPowerCommand(0), LineToCommand(overscan_end_3d)]
            )
            return rewritten_commands

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
