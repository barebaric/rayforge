from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from ..core.geo.types import Point3D
from ..core.ops.axis import Axis
from ..core.ops.commands import (
    ArcToCommand,
    BezierToCommand,
    MovingCommand,
    QuadraticBezierToCommand,
)
from ..core.ops import Ops
from .kinematic_math import KinematicMath

if TYPE_CHECKING:
    from .models.rotary_module import RotaryModule
    from ..core.doc import Doc
    from .models.machine import Machine

_AXIS_INDEX = {Axis.X: 0, Axis.Y: 1, Axis.Z: 2}


class KinematicMapping:
    """Applies rotary kinematic mapping to world-space ops.

    Converts source axis mu values to degrees and stores them in
    extra_axes.  The mode distinction (TRUE_4TH_AXIS vs AXIS_REPLACEMENT)
    is irrelevant here — degrees are the canonical intermediate
    representation.

    Operates on world-space ops, before the world→machine transform.
    """

    def __init__(
        self,
        source_axis: Axis,
        rotary_axis: Axis,
        diameter: float,
        gear_ratio: float = 1.0,
        reverse: bool = False,
        axis_position: float = 0.0,
    ):
        self.source_axis = source_axis
        self.rotary_axis = rotary_axis
        self.diameter = diameter
        self.gear_ratio = gear_ratio
        self.reverse = reverse
        self.axis_position = axis_position

    @classmethod
    def from_rotary_module(
        cls,
        module: RotaryModule,
        diameter: float,
    ) -> Optional[KinematicMapping]:
        """Factory: reads kinematics parameters from a RotaryModule.

        Args:
            module: The rotary module configuration.
            diameter: The effective rotary diameter (from the layer).

        Returns:
            A KinematicMapping instance, or None if the source axis
            is not a linear axis.
        """
        from .models.rotary_module import RotaryMode

        source_axis = module.source_axis
        if source_axis not in _AXIS_INDEX:
            return None

        if module.mode == RotaryMode.TRUE_4TH_AXIS:
            rotary_axis = module.axis
        else:
            rotary_axis = source_axis

        from .models.rotary_module import RotaryType

        ratio = KinematicMath.gear_ratio(
            module.rotary_type == RotaryType.ROLLERS,
            diameter,
            module.roller_diameter,
        )

        si = _AXIS_INDEX[source_axis]
        axis_position = float(module.transform[si, 3]) + float(
            module.axis_position[si]
        )

        return cls(
            source_axis=source_axis,
            rotary_axis=rotary_axis,
            diameter=diameter,
            gear_ratio=ratio,
            reverse=module.reverse_axis,
            axis_position=axis_position,
        )

    def _mu_to_degrees(self, mu: float, z: float) -> float:
        eff_d = KinematicMath.effective_diameter(self.diameter, z)
        return KinematicMath.mu_to_degrees(
            mu, eff_d, gear_ratio=self.gear_ratio, reverse=self.reverse
        )

    def _source_index(self) -> int:
        return _AXIS_INDEX[self.source_axis]

    @staticmethod
    def _replace_source(end: Point3D, si: int, value: float) -> Point3D:
        parts = list(end)
        parts[si] = value
        return (parts[0], parts[1], parts[2])

    def apply(self, ops: Ops) -> None:
        """Map source axis mu values to degrees in-place on ops.

        Converts source-axis arc lengths to degrees, stores the result
        in extra_axes, and parks the source axis at axis_position.

        Arc center_offset is a relative displacement and is converted
        directly.  Bezier control points are converted the same way.
        """
        si = self._source_index()
        for cmd in ops._commands:
            if not isinstance(cmd, MovingCommand):
                continue

            src_val = cmd.end[si]
            z_val = cmd.end[2]
            degrees = self._mu_to_degrees(src_val, z_val)

            cmd.extra_axes[self.rotary_axis] = degrees
            cmd.end = self._replace_source(cmd.end, si, self.axis_position)

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
                    cp_deg = self._mu_to_degrees(cp[si], z_val)
                    cp[si] = cp_deg
                    setattr(cmd, attr, tuple(cp))
            elif isinstance(cmd, QuadraticBezierToCommand):
                cp = list(cmd.control)
                cp_deg = self._mu_to_degrees(cp[si], z_val)
                cp[si] = cp_deg
                cmd.control = (cp[0], cp[1], cp[2])

    @staticmethod
    def apply_to_job_ops(
        ops: Ops,
        doc: Doc,
        machine: Machine,
    ) -> None:
        """Walk job ops and apply the appropriate mapping per layer.

        This replaces the manual loop in Machine._apply_rotary_axis_mapping().
        """
        from ..core.layer import Layer
        from ..core.ops.commands import LayerStartCommand

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
        source_axis: Axis,
        mu_per_rotation: float,
    ) -> None:
        """Convert degrees in extra_axes back to scaled-mu on the source axis.

        This is the AXIS_REPLACEMENT downstream pass, run after world→machine
        so the scaled values land in machine-coordinate commands.
        """
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
