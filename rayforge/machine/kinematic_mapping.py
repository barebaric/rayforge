from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from ..core.geo.types import Point3D
from ..core.layer import Layer
from ..core.ops.axis import Axis
from ..core.ops.commands import (
    ArcToCommand,
    BezierToCommand,
    LayerStartCommand,
    MovingCommand,
    QuadraticBezierToCommand,
)
from ..core.ops import Ops
from .kinematic_math import KinematicMath
from .models.rotary_module import RotaryMode, RotaryType

if TYPE_CHECKING:
    from .models.rotary_module import RotaryModule
    from ..core.doc import Doc
    from .models.machine import Machine


class KinematicMapping:
    """Applies rotary kinematic mapping to world-space ops.

    Converts Y-axis mu values to degrees and stores them in
    extra_axes.  The source axis is always Y.

    Operates on world-space ops, before the world→machine transform.
    """

    def __init__(
        self,
        rotary_axis: Axis,
        diameter: float,
        gear_ratio: float = 1.0,
        reverse: bool = False,
        axis_position: float = 0.0,
        axis_position_3d: Optional[np.ndarray] = None,
        cylinder_dir: Optional[np.ndarray] = None,
        replaced_axis: Optional[Axis] = None,
    ):
        self.rotary_axis = rotary_axis
        self.diameter = diameter
        self.gear_ratio = gear_ratio
        self.reverse = reverse
        self.replaced_axis = replaced_axis
        self.axis_position = axis_position
        if axis_position_3d is not None:
            self.axis_position_3d = axis_position_3d.astype(np.float64)
        else:
            v = np.zeros(3, dtype=np.float64)
            v[1] = axis_position
            self.axis_position_3d = v
        if cylinder_dir is not None:
            self.cylinder_dir = cylinder_dir.astype(np.float64)
        else:
            self.cylinder_dir = np.array([1.0, 0.0, 0.0])

    @classmethod
    def from_rotary_module(
        cls,
        module: RotaryModule,
        diameter: float,
    ) -> Optional[KinematicMapping]:
        if module.mode == RotaryMode.TRUE_4TH_AXIS:
            rotary_axis = module.axis
            replaced_axis = None
        else:
            rotary_axis = Axis.Y
            replaced_axis = module.axis

        ratio = KinematicMath.gear_ratio(
            module.rotary_type == RotaryType.ROLLERS,
            diameter,
            module.roller_diameter,
        )

        rot3 = module.transform[:3, :3].astype(np.float64).copy()
        for col in range(3):
            norm = np.linalg.norm(rot3[:, col])
            if norm > 1e-12:
                rot3[:, col] /= norm

        mod_pos = module.transform[:3, 3].astype(np.float64)
        axis_position_3d = mod_pos + rot3 @ module.axis_position
        cylinder_dir = rot3[:, 0].copy()
        norm = np.linalg.norm(cylinder_dir)
        if norm > 1e-12:
            cylinder_dir /= norm

        axis_position = float(axis_position_3d[1])

        return cls(
            rotary_axis=rotary_axis,
            diameter=diameter,
            gear_ratio=ratio,
            reverse=module.reverse_axis,
            axis_position=axis_position,
            axis_position_3d=axis_position_3d,
            cylinder_dir=cylinder_dir,
            replaced_axis=replaced_axis,
        )

    def _mu_to_degrees(self, mu: float) -> float:
        return KinematicMath.mu_to_degrees(
            mu, self.diameter,
            gear_ratio=self.gear_ratio, reverse=self.reverse,
        )

    def _cylinder_center_y(self, cmd: MovingCommand) -> float:
        d = cmd.end[0]
        return float(
            self.axis_position_3d[1] + d * self.cylinder_dir[1]
        )

    _AXIS_TO_INDEX = {Axis.X: 0, Axis.Y: 1, Axis.Z: 2}

    def _null_replaced_axis(self, end: Point3D) -> Point3D:
        if self.replaced_axis is None:
            return end
        idx = self._AXIS_TO_INDEX.get(self.replaced_axis)
        if idx is None:
            return end
        parts = list(end)
        parts[idx] = 0.0
        return (parts[0], parts[1], parts[2])

    def apply(self, ops: Ops) -> None:
        for cmd in ops._commands:
            if not isinstance(cmd, MovingCommand):
                continue

            src_val = cmd.end[1]
            degrees = self._mu_to_degrees(src_val)

            cmd.extra_axes[self.rotary_axis] = degrees
            cmd.end = self._replace_y(
                cmd.end, self._cylinder_center_y(cmd)
            )
            cmd.end = self._null_replaced_axis(cmd.end)

            if isinstance(cmd, ArcToCommand):
                src_offset = cmd.center_offset[1]
                src_deg = self._mu_to_degrees(src_offset)
                new_offset = list(cmd.center_offset)
                new_offset[1] = src_deg
                cmd.center_offset = (
                    new_offset[0],
                    new_offset[1],
                )
            elif isinstance(cmd, BezierToCommand):
                for attr in ("control1", "control2"):
                    cp = list(getattr(cmd, attr))
                    cp_deg = self._mu_to_degrees(cp[1])
                    cp[1] = cp_deg
                    setattr(cmd, attr, tuple(cp))
            elif isinstance(cmd, QuadraticBezierToCommand):
                cp = list(cmd.control)
                cp_deg = self._mu_to_degrees(cp[1])
                cp[1] = cp_deg
                cmd.control = (cp[0], cp[1], cp[2])

    @staticmethod
    def _replace_y(end: Point3D, value: float) -> Point3D:
        parts = list(end)
        parts[1] = value
        return (parts[0], parts[1], parts[2])

    @staticmethod
    def apply_to_job_ops(
        ops: Ops,
        doc: Doc,
        machine: Machine,
    ) -> None:
        commands = ops._commands
        i = 0
        while i < len(commands):
            cmd = commands[i]
            if not isinstance(cmd, LayerStartCommand):
                i += 1
                continue

            descendant = doc.find_descendant_by_uid(cmd.layer_uid)
            if not isinstance(descendant, Layer):
                i += 1
                continue

            if (
                not descendant.rotary_module_uid
                or not descendant.rotary_enabled
            ):
                i += 1
                continue

            module = machine.rotary_modules.get(descendant.rotary_module_uid)
            if module is None:
                i += 1
                continue

            diameter = descendant.rotary_diameter

            mapping = KinematicMapping.from_rotary_module(module, diameter)
            if mapping is None:
                i += 1
                continue

            layer_cmds = []
            i += 1
            while i < len(commands) and not isinstance(
                commands[i], LayerStartCommand
            ):
                if not commands[i].is_marker():
                    layer_cmds.append(commands[i])
                i += 1

            layer_ops = Ops()
            layer_ops._commands = layer_cmds
            mapping.apply(layer_ops)

    @staticmethod
    def degrees_to_scaled_mu_pass(
        commands: list,
        mu_per_rotation: float,
        target_axis: Axis = Axis.Y,
    ) -> None:
        idx = KinematicMapping._AXIS_TO_INDEX.get(target_axis, 1)
        cp_idx = idx
        null_source = (
            target_axis != Axis.Y
            and target_axis in KinematicMapping._AXIS_TO_INDEX
        )
        for cmd in commands:
            if not isinstance(cmd, MovingCommand):
                continue
            degrees = cmd.extra_axes.pop(Axis.Y, None)
            if degrees is None:
                continue
            scaled = KinematicMath.degrees_to_scaled_mu(
                degrees, mu_per_rotation
            )
            end = list(cmd.end)
            end[idx] = scaled
            if null_source:
                end[1] = 0.0
            cmd.end = (end[0], end[1], end[2])

            if isinstance(cmd, ArcToCommand):
                if cp_idx < len(cmd.center_offset):
                    cp_deg = cmd.center_offset[cp_idx]
                    cp_scaled = KinematicMath.degrees_to_scaled_mu(
                        cp_deg, mu_per_rotation
                    )
                    new_offset = list(cmd.center_offset)
                    new_offset[cp_idx] = cp_scaled
                    if null_source:
                        new_offset[1] = 0.0
                    cmd.center_offset = (
                        new_offset[0],
                        new_offset[1],
                    )
            elif isinstance(cmd, BezierToCommand):
                for attr in ("control1", "control2"):
                    cp = list(getattr(cmd, attr))
                    cp_scaled = KinematicMath.degrees_to_scaled_mu(
                        cp[cp_idx], mu_per_rotation
                    )
                    cp[cp_idx] = cp_scaled
                    if null_source:
                        cp[1] = 0.0
                    setattr(cmd, attr, tuple(cp))
            elif isinstance(cmd, QuadraticBezierToCommand):
                cp = list(cmd.control)
                cp_scaled = KinematicMath.degrees_to_scaled_mu(
                    cp[cp_idx], mu_per_rotation
                )
                cp[cp_idx] = cp_scaled
                if null_source:
                    cp[1] = 0.0
                cmd.control = (cp[0], cp[1], cp[2])
