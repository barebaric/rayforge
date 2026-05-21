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

        new_ops = Ops()
        line_buffer: List[int] = []
        in_vector_section = False

        def _process_buffer():
            nonlocal line_buffer
            if line_buffer:
                self._rewrite_buffered_contour(new_ops, ops, line_buffer)
                line_buffer = []

        for i in range(ops.len()):
            ct = ops.command_type(i)

            is_start = ct == CommandType.OPS_SECTION_START
            if is_start:
                sec_type, _ = ops.section_params(i)
                is_start = sec_type == SectionType.VECTOR_OUTLINE

            is_end = ct == CommandType.OPS_SECTION_END
            if is_end:
                sec_type, _ = ops.section_params(i)
                is_end = sec_type == SectionType.VECTOR_OUTLINE

            if is_start:
                _process_buffer()
                in_vector_section = True
                new_ops.transfer_command_from(ops, i)
            elif is_end:
                _process_buffer()
                in_vector_section = False
                new_ops.transfer_command_from(ops, i)
            elif not in_vector_section:
                new_ops.transfer_command_from(ops, i)
            else:
                if ct == CommandType.MOVE_TO:
                    _process_buffer()
                    line_buffer = [i]
                elif line_buffer:
                    line_buffer.append(i)
                else:
                    _process_buffer()
                    new_ops.transfer_command_from(ops, i)

        _process_buffer()
        ops.replace_with(new_ops)

    def _get_tangent_at_start(
        self, old_ops: Ops, indices: List[int]
    ) -> Optional[tuple]:
        """
        Compute the normalized tangent direction at the start of a
        contour using the geometry module. Converts the ops buffer to a
        Geometry object and queries the tangent at t=0 of the first
        drawing segment.

        Returns None if the first segment has zero length.
        """
        sub_ops = Ops()
        for j in indices:
            sub_ops.transfer_command_from(old_ops, j)
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

    def _get_tangent_at_end(
        self, old_ops: Ops, indices: List[int]
    ) -> Optional[tuple]:
        """
        Compute the normalized tangent direction at the end of a contour
        using the geometry module. Converts the ops buffer to a Geometry
        object and queries the tangent at t=1 of the last segment.

        Returns None if the last segment has zero length.
        """
        sub_ops = Ops()
        for j in indices:
            sub_ops.transfer_command_from(old_ops, j)
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
        self,
        new_ops: Ops,
        old_ops: Ops,
        indices: List[int],
    ) -> None:
        """
        Rewrites a single contour path to include lead-in and/or lead-out
        zero-power segments, computed from the proper tangent direction
        at the path start and end.
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

        has_lead_in = not math.isclose(self.lead_in_mm, 0.0)
        has_lead_out = not math.isclose(self.lead_out_mm, 0.0)

        if not has_lead_in and not has_lead_out:
            for j in indices:
                new_ops.transfer_command_from(old_ops, j)
            return

        lead_in_tangent = None
        if has_lead_in:
            lead_in_tangent = self._get_tangent_at_start(old_ops, indices)
            if lead_in_tangent is None:
                has_lead_in = False

        lead_out_tangent = None
        if has_lead_out:
            lead_out_tangent = self._get_tangent_at_end(old_ops, indices)
            if lead_out_tangent is None:
                has_lead_out = False

        if not has_lead_in and not has_lead_out:
            for j in indices:
                new_ops.transfer_command_from(old_ops, j)
            return

        first_cut_idx = None
        for j in moving_indices[1:]:
            if old_ops.command_type(j) == CommandType.LINE_TO:
                state = old_ops.preloaded_state(j)
                if state.power is not None:
                    first_cut_idx = j
                    break

        if first_cut_idx is None:
            for j in indices:
                new_ops.transfer_command_from(old_ops, j)
            return

        original_power = old_ops.preloaded_state(first_cut_idx).power
        start_3d = old_ops.endpoint(moving_indices[0])
        end_3d = old_ops.endpoint(moving_indices[-1])

        if has_lead_in and lead_in_tangent is not None:
            tx, ty = lead_in_tangent
            lead_in_start_3d = (
                start_3d[0] - tx * self.lead_in_mm,
                start_3d[1] - ty * self.lead_in_mm,
                start_3d[2],
            )
            new_ops.move_to(*lead_in_start_3d)
            new_ops.set_power(0)
            new_ops.line_to(*start_3d)
        else:
            new_ops.transfer_command_from(old_ops, indices[0])

        content_indices = indices[1:]
        if not content_indices or (
            old_ops.command_type(content_indices[0]) != CommandType.SET_POWER
        ):
            new_ops.set_power(original_power)
        for j in content_indices:
            new_ops.copy_command_from(old_ops, j)

        if has_lead_out and lead_out_tangent is not None:
            tx, ty = lead_out_tangent
            lead_out_end_3d = (
                end_3d[0] + tx * self.lead_out_mm,
                end_3d[1] + ty * self.lead_out_mm,
                end_3d[2],
            )
            new_ops.set_power(0)
            new_ops.line_to(*lead_out_end_3d)

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
