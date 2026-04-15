from __future__ import annotations

from typing import Dict, List, Optional, Any

from ...core.geo.types import Point3D
from ...core.ops.axis import Axis
from ...core.ops.commands import (
    ArcToCommand,
    BezierToCommand,
    MoveToCommand,
    MovingCommand,
    QuadraticBezierToCommand,
)
from ...core.ops import Ops
from ...core.workpiece import WorkPiece
from ...machine.kinematic_math import KinematicMath
from ...machine.models.rotary_module import RotaryMode, RotaryType
from ...shared.tasker.progress import ProgressContext
from .base import ExecutionPhase, OpsTransformer

_AXIS_INDEX = {Axis.X: 0, Axis.Y: 1, Axis.Z: 2}


class AxisMapper(OpsTransformer):
    def __init__(
        self,
        source_axis: Axis = Axis.Y,
        rotary_axis: Axis = Axis.A,
        rotary_diameter: float = 25.0,
        has_physical_source: bool = False,
        mode: RotaryMode = RotaryMode.TRUE_4TH_AXIS,
        mu_per_rotation: float = 0.0,
        rotary_type: RotaryType = RotaryType.JAWS,
        roller_diameter: float = 0.0,
        reverse_axis: bool = False,
        enabled: bool = True,
        **kwargs,
    ):
        super().__init__(enabled=enabled, **kwargs)
        self.source_axis = source_axis
        self.rotary_axis = rotary_axis
        self.rotary_diameter = rotary_diameter
        self.has_physical_source = has_physical_source
        self.mode = mode
        self.mm_per_rotation = mu_per_rotation
        self.rotary_type = rotary_type
        self.roller_diameter = roller_diameter
        self.reverse_axis = reverse_axis

    @property
    def label(self) -> str:
        return "Axis Mapper"

    @property
    def description(self) -> str:
        return "Maps source axis values to a rotary axis"

    @property
    def execution_phase(self) -> ExecutionPhase:
        return ExecutionPhase.GEOMETRY_REFINEMENT

    def _mu_to_degrees(self, mu: float, z: float) -> float:
        eff_d = KinematicMath.effective_diameter(self.rotary_diameter, z)
        ratio = KinematicMath.gear_ratio(
            self.rotary_type == RotaryType.ROLLERS,
            self.rotary_diameter,
            self.roller_diameter,
        )
        return KinematicMath.mu_to_degrees(
            mu, eff_d, gear_ratio=ratio, reverse=self.reverse_axis
        )

    def _source_index(self) -> int:
        idx = _AXIS_INDEX.get(self.source_axis)
        if idx is None:
            raise ValueError(
                f"Source axis {self.source_axis} is not a linear (XYZ) axis"
            )
        return idx

    def _replace_source(self, end: Point3D, si: int, value: float) -> Point3D:
        parts = list(end)
        parts[si] = value
        return (parts[0], parts[1], parts[2])

    def run(
        self,
        ops: Ops,
        workpiece: Optional[WorkPiece] = None,
        context: Optional[ProgressContext] = None,
        stock_geometries: Optional[List] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        si = self._source_index()
        commands = ops._commands

        if self.mode == RotaryMode.AXIS_REPLACEMENT:
            self._run_replacement(commands, si)
        else:
            self._run_true_4th_axis(commands, si)

    def _run_replacement(self, commands: list, si: int) -> None:
        for cmd in commands:
            if not isinstance(cmd, MovingCommand):
                continue
            if self.mm_per_rotation <= 0:
                continue
            src_val = cmd.end[si]
            z_val = cmd.end[2]
            degrees = self._mu_to_degrees(src_val, z_val)
            cmd.extra_axes[self.source_axis] = degrees
            cmd.end = self._replace_source(cmd.end, si, 0.0)

            if isinstance(cmd, ArcToCommand):
                src_offset = cmd.center_offset[si]
                src_deg = self._mu_to_degrees(src_offset, z_val)
                new_offset = list(cmd.center_offset)
                new_offset[si] = src_deg
                cmd.center_offset = (
                    new_offset[0],
                    new_offset[1],
                )
            elif isinstance(cmd, BezierToCommand):
                for attr in ("control1", "control2"):
                    cp = list(getattr(cmd, attr))
                    src_deg = self._mu_to_degrees(cp[si], z_val)
                    cp[si] = src_deg
                    setattr(cmd, attr, tuple(cp))
            elif isinstance(cmd, QuadraticBezierToCommand):
                cp = list(cmd.control)
                src_deg = self._mu_to_degrees(cp[si], z_val)
                cp[si] = src_deg
                cmd.control = (cp[0], cp[1], cp[2])

    def _run_true_4th_axis(self, commands: list, si: int) -> None:
        insert_pos = 0
        park_inserted = False

        for cmd in commands:
            if not isinstance(cmd, MovingCommand):
                continue

            if not park_inserted and self.has_physical_source:
                park_end = self._replace_source(cmd.end, si, 0.0)
                park_cmd = MoveToCommand(
                    park_end,
                    extra_axes=dict(cmd.extra_axes),
                )
                park_cmd.state = cmd.state
                commands.insert(insert_pos, park_cmd)
                insert_pos += 1
                park_inserted = True

            src_val = cmd.end[si]
            z_val = cmd.end[2]
            degrees = self._mu_to_degrees(src_val, z_val)

            cmd.extra_axes[self.rotary_axis] = degrees
            cmd.end = self._replace_source(cmd.end, si, 0.0)

            if isinstance(cmd, ArcToCommand):
                src_offset = cmd.center_offset[si]
                src_deg = self._mu_to_degrees(src_offset, z_val)
                new_center_offset = list(cmd.center_offset)
                new_center_offset[si] = src_deg
                cmd.center_offset = (
                    new_center_offset[0],
                    new_center_offset[1],
                )

            elif isinstance(cmd, BezierToCommand):
                for attr in ("control1", "control2"):
                    cp = list(getattr(cmd, attr))
                    src_deg = self._mu_to_degrees(cp[si], z_val)
                    cp[si] = src_deg
                    setattr(cmd, attr, tuple(cp))

            elif isinstance(cmd, QuadraticBezierToCommand):
                cp = list(cmd.control)
                src_deg = self._mu_to_degrees(cp[si], z_val)
                cp[si] = src_deg
                cmd.control = (cp[0], cp[1], cp[2])

            insert_pos += 1

    @staticmethod
    def degrees_to_scaled_mu_pass(
        commands: list,
        source_axis: "Axis",
        mu_per_rotation: float,
    ) -> None:
        si = _AXIS_INDEX.get(source_axis)
        if si is None or mu_per_rotation <= 0:
            return
        for cmd in commands:
            if not isinstance(cmd, MovingCommand):
                continue
            degrees = cmd.extra_axes.pop(source_axis, None)
            if degrees is None:
                continue
            scaled = KinematicMath.degrees_to_scaled_mu(
                degrees, mu_per_rotation
            )
            end = list(cmd.end)
            end[si] = scaled
            cmd.end = (end[0], end[1], end[2])

            if isinstance(cmd, ArcToCommand):
                cp_deg = cmd.center_offset[si]
                cp_scaled = KinematicMath.degrees_to_scaled_mu(
                    cp_deg, mu_per_rotation
                )
                new_offset = list(cmd.center_offset)
                new_offset[si] = cp_scaled
                cmd.center_offset = (
                    new_offset[0],
                    new_offset[1],
                )
            elif isinstance(cmd, BezierToCommand):
                for attr in ("control1", "control2"):
                    cp = list(getattr(cmd, attr))
                    cp_scaled = KinematicMath.degrees_to_scaled_mu(
                        cp[si], mu_per_rotation
                    )
                    cp[si] = cp_scaled
                    setattr(cmd, attr, tuple(cp))
            elif isinstance(cmd, QuadraticBezierToCommand):
                cp = list(cmd.control)
                cp_scaled = KinematicMath.degrees_to_scaled_mu(
                    cp[si], mu_per_rotation
                )
                cp[si] = cp_scaled
                cmd.control = (cp[0], cp[1], cp[2])
