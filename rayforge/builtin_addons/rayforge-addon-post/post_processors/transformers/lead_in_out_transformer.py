from __future__ import annotations

import math
import logging
from typing import Optional, List, Dict, Any, Sequence, TYPE_CHECKING
from gettext import gettext as _

from rayforge.core.ops import (
    Ops,
    Command,
    MovingCommand,
    MoveToCommand,
    LineToCommand,
    SetPowerCommand,
    SectionType,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
)
from rayforge.pipeline.transformer.base import OpsTransformer, ExecutionPhase
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from rayforge.core.geo import Geometry
    from rayforge.core.workpiece import WorkPiece

logger = logging.getLogger(__name__)


class LeadInOutTransformer(OpsTransformer):
    """
    Adds zero-power lead-in and lead-out moves to vector contour paths.

    For each contour within a VECTOR_OUTLINE section, this transformer
    computes the tangent direction at the start and end of the path using
    the geometry module, then extends the toolpath with lead-in (before
    the cut starts) and lead-out (after the cut ends) segments at zero
    laser power. This allows the laser head to reach constant velocity
    before the actual cut begins and to decelerate after the cut ends,
    improving cut quality at start/end points.
    """

    def __init__(
        self,
        enabled: bool = True,
        lead_in_mm: float = 2.0,
        lead_out_mm: float = 2.0,
        auto: bool = True,
    ):
        super().__init__(enabled=enabled)
        self._lead_in_mm: float = 0.0
        self.lead_in_mm = lead_in_mm
        self._lead_out_mm: float = 0.0
        self.lead_out_mm = lead_out_mm
        self._auto: bool = auto

    @staticmethod
    def calculate_auto_distance(
        step_speed: int, max_acceleration: int
    ) -> float:
        """
        Calculate the optimal lead-in/out distance based on step speed
        and machine acceleration with a safety factor of 2.

        Formula: distance = (speed^2) / (2 * acceleration * safety_factor)
        Where safety_factor = 2 for additional safety margin.

        Args:
            step_speed: The cutting speed in mm/min
            max_acceleration: The maximum machine acceleration in mm/s^2

        Returns:
            The calculated distance in millimeters
        """
        speed_mm_per_sec = step_speed / 60.0
        safety_factor = 2.0
        distance_mm = (speed_mm_per_sec**2) / (
            2 * max_acceleration * safety_factor
        )
        return max(0.5, distance_mm)

    @property
    def execution_phase(self) -> ExecutionPhase:
        return ExecutionPhase.POST_PROCESSING

    @property
    def lead_in_mm(self) -> float:
        return self._lead_in_mm

    @lead_in_mm.setter
    def lead_in_mm(self, value: float):
        new_value = max(0.0, float(value))
        if not math.isclose(self._lead_in_mm, new_value):
            self._lead_in_mm = new_value
            self.changed.send(self)

    @property
    def lead_out_mm(self) -> float:
        return self._lead_out_mm

    @lead_out_mm.setter
    def lead_out_mm(self, value: float):
        new_value = max(0.0, float(value))
        if not math.isclose(self._lead_out_mm, new_value):
            self._lead_out_mm = new_value
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
        return _("Lead-In/Out")

    @property
    def description(self) -> str:
        return _(
            "Adds zero-power lead-in and lead-out moves to vector contours."
        )

    def run(
        self,
        ops: Ops,
        workpiece: Optional[WorkPiece] = None,
        context: Optional[ProgressContext] = None,
        stock_geometries: Optional[List["Geometry"]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        has_lead_in = self.enabled and not math.isclose(self.lead_in_mm, 0.0)
        has_lead_out = self.enabled and not math.isclose(self.lead_out_mm, 0.0)
        if not has_lead_in and not has_lead_out:
            return

        ops.preload_state()

        new_commands: List[Command] = []
        line_buffer: List[Command] = []
        in_vector_section = False

        def _process_buffer():
            nonlocal line_buffer
            if line_buffer:
                rewritten = self._rewrite_buffered_contour(line_buffer)
                new_commands.extend(rewritten)
                line_buffer = []

        for cmd in ops:
            is_start = (
                isinstance(cmd, OpsSectionStartCommand)
                and cmd.section_type == SectionType.VECTOR_OUTLINE
            )
            is_end = (
                isinstance(cmd, OpsSectionEndCommand)
                and cmd.section_type == SectionType.VECTOR_OUTLINE
            )

            if is_start:
                _process_buffer()
                in_vector_section = True
                new_commands.append(cmd)
            elif is_end:
                _process_buffer()
                in_vector_section = False
                new_commands.append(cmd)
            elif not in_vector_section:
                new_commands.append(cmd)
            else:
                if isinstance(cmd, MoveToCommand):
                    _process_buffer()
                    line_buffer = [cmd]
                elif line_buffer:
                    line_buffer.append(cmd)
                else:
                    _process_buffer()
                    new_commands.append(cmd)

        _process_buffer()
        ops.replace_all(new_commands)

    def _get_tangent_at_start(self, buffer: List[Command]) -> Optional[tuple]:
        """
        Compute the normalized tangent direction at the start of a
        contour using the geometry module. Converts the ops buffer to a
        Geometry object and queries the tangent at t=0 of the first
        drawing segment.

        Returns None if the first segment has zero length.
        """
        sub_ops = Ops()
        sub_ops.replace_all(buffer)
        geo = sub_ops.to_geometry()
        if geo.data is None or len(geo.data) < 2:
            return None
        seg_start_x, seg_start_y = geo.data[0, 1], geo.data[0, 2]
        seg_end_x, seg_end_y = geo.data[1, 1], geo.data[1, 2]
        seg_len = math.hypot(seg_end_x - seg_start_x, seg_end_y - seg_start_y)
        if seg_len < 1e-9:
            return None
        result = geo.get_point_and_tangent_at(1, 0.0)
        if result is None:
            return None
        _, tangent = result
        return tangent

    def _get_tangent_at_end(self, buffer: List[Command]) -> Optional[tuple]:
        """
        Compute the normalized tangent direction at the end of a contour
        using the geometry module. Converts the ops buffer to a Geometry
        object and queries the tangent at t=1 of the last segment.

        Returns None if the last segment has zero length.
        """
        sub_ops = Ops()
        sub_ops.replace_all(buffer)
        geo = sub_ops.to_geometry()
        if geo.data is None or len(geo.data) < 2:
            return None
        last_idx = len(geo.data) - 1
        prev_x, prev_y = geo.data[last_idx - 1, 1], geo.data[last_idx - 1, 2]
        end_x, end_y = geo.data[last_idx, 1], geo.data[last_idx, 2]
        seg_len = math.hypot(end_x - prev_x, end_y - prev_y)
        if seg_len < 1e-9:
            return None
        result = geo.get_point_and_tangent_at(last_idx, 1.0)
        if result is None:
            return None
        _, tangent = result
        return tangent

    def _rewrite_buffered_contour(
        self, buffer: List[Command]
    ) -> Sequence[Command]:
        """
        Rewrites a single contour path to include lead-in and/or lead-out
        zero-power segments, computed from the proper tangent direction
        at the path start and end.
        """
        moving_cmds = [c for c in buffer if isinstance(c, MovingCommand)]

        if len(moving_cmds) < 2 or not isinstance(
            moving_cmds[0], MoveToCommand
        ):
            return buffer

        has_lead_in = not math.isclose(self.lead_in_mm, 0.0)
        has_lead_out = not math.isclose(self.lead_out_mm, 0.0)

        if not has_lead_in and not has_lead_out:
            return buffer

        lead_in_tangent = None
        if has_lead_in:
            lead_in_tangent = self._get_tangent_at_start(buffer)
            if lead_in_tangent is None:
                has_lead_in = False

        lead_out_tangent = None
        if has_lead_out:
            lead_out_tangent = self._get_tangent_at_end(buffer)
            if lead_out_tangent is None:
                has_lead_out = False

        if not has_lead_in and not has_lead_out:
            return buffer

        first_cut = next(
            (c for c in moving_cmds[1:] if c.is_cutting_command() and c.state),
            None,
        )
        if not first_cut:
            return buffer

        assert first_cut.state is not None
        original_power = first_cut.state.power
        start_3d = moving_cmds[0].end
        end_3d = moving_cmds[-1].end

        rewritten: List[Command] = []

        if has_lead_in and lead_in_tangent is not None:
            tx, ty = lead_in_tangent
            lead_in_start_3d = (
                start_3d[0] - tx * self.lead_in_mm,
                start_3d[1] - ty * self.lead_in_mm,
                start_3d[2],
            )
            rewritten.append(MoveToCommand(lead_in_start_3d))
            rewritten.extend([SetPowerCommand(0), LineToCommand(start_3d)])
        else:
            rewritten.append(buffer[0])

        content_cmds = buffer[1:]
        if not content_cmds or not isinstance(
            content_cmds[0], SetPowerCommand
        ):
            rewritten.append(SetPowerCommand(original_power))
        rewritten.extend(content_cmds)

        if has_lead_out and lead_out_tangent is not None:
            tx, ty = lead_out_tangent
            lead_out_end_3d = (
                end_3d[0] + tx * self.lead_out_mm,
                end_3d[1] + ty * self.lead_out_mm,
                end_3d[2],
            )
            rewritten.extend(
                [SetPowerCommand(0), LineToCommand(lead_out_end_3d)]
            )

        return rewritten

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "lead_in_mm": self.lead_in_mm,
            "lead_out_mm": self.lead_out_mm,
            "auto": self.auto,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LeadInOutTransformer":
        return cls(
            enabled=data.get("enabled", True),
            lead_in_mm=data.get("lead_in_mm", 2.0),
            lead_out_mm=data.get("lead_out_mm", 2.0),
            auto=data.get("auto", True),
        )
