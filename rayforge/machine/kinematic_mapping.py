from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from raygeo.ops import Ops
from raygeo.ops.axis import Axis

from ..core.layer import Layer
from .kinematic_math import KinematicMath
from .models.rotary_module import RotaryMode, RotaryType

if TYPE_CHECKING:
    from ..core.doc import Doc
    from .models.machine import Machine
    from .models.rotary_module import RotaryModule


def _resolve_rotary_layer_by_uid(layer_uid: str, doc):
    """Return the Layer if it has rotary enabled."""
    descendant = doc.find_descendant_by_uid(layer_uid)
    if not isinstance(descendant, Layer):
        return None
    if not descendant.rotary_module_uid or not descendant.rotary_enabled:
        return None
    return descendant


def _is_valid_replacement_module(module):
    """Check whether an AXIS_REPLACEMENT module is valid for mapping.

    Modules in AXIS_REPLACEMENT mode are only valid when they have a
    positive ``mu_per_rotation`` *or* their target axis is one of the
    standard XYZ axes (which can accept degree values directly).
    """
    if module.mode != RotaryMode.AXIS_REPLACEMENT:
        return True
    if module.mu_per_rotation > 0:
        return True
    return module.axis in KinematicMapping._AXIS_TO_INDEX


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
        apply_gear_ratio: bool = True,
    ) -> Optional[KinematicMapping]:
        if module.mode == RotaryMode.TRUE_4TH_AXIS:
            rotary_axis = module.axis
            replaced_axis = None
        else:
            rotary_axis = Axis.Y
            replaced_axis = module.axis

        if apply_gear_ratio:
            ratio = KinematicMath.gear_ratio(
                module.rotary_type == RotaryType.ROLLERS,
                diameter,
                module.roller_diameter,
            )
        else:
            ratio = 1.0

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
            mu,
            self.diameter,
            gear_ratio=self.gear_ratio,
            reverse=self.reverse,
        )

    _AXIS_TO_INDEX: Dict[Axis, int] = {
        Axis.X: 0,
        Axis.Y: 1,
        Axis.Z: 2,
    }

    def apply(self, ops: Ops) -> None:
        replaced_idx = (
            self._AXIS_TO_INDEX.get(self.replaced_axis)
            if self.replaced_axis is not None
            else None
        )

        def on_endpoint(
            end: List[float],
            extra_axes: Dict[Axis, float],
        ) -> None:
            degrees = self._mu_to_degrees(end[1])
            extra_axes[self.rotary_axis] = degrees
            end[1] = float(
                self.axis_position_3d[1] + end[0] * self.cylinder_dir[1]
            )
            if replaced_idx is not None:
                end[replaced_idx] = 0.0

        def on_aux_point(point: List[float]) -> None:
            point[1] = self._mu_to_degrees(point[1])

        ops.transform_moving(on_endpoint, on_aux_point)

    @staticmethod
    def apply_to_job_ops(
        ops: Ops,
        doc: Doc,
        machine: Machine,
        apply_scaled_mu: bool = False,
        apply_gear_ratio: bool = True,
    ) -> None:
        """Apply per-layer rotary axis mapping to a full job's ops.

        Walks the command stream looking for ``LayerStartCommand``
        markers.  For each layer that has rotary enabled, resolves the
        layer's ``RotaryModule`` and applies a ``KinematicMapping``
        (Y→degrees) to that layer's movement commands in-place.

        This is the single entry point for both the UI path (3D canvas /
        OpPlayer via ``job_compute.py``) and the G-code encoding path
        (via ``Machine.encode_ops()``).

        Args:
            ops: Assembled ops in world-space coordinates.  Modified
                in-place.
            doc: The document that owns the layers (used to look up
                per-layer rotary configuration).
            machine: The machine whose ``rotary_modules`` dict is used
                to resolve module UIDs.
            apply_scaled_mu: When *True*, also convert degrees to
                scaled machine units for ``AXIS_REPLACEMENT`` layers.
                The G-code path passes ``False`` and defers this step
                until *after* the world→machine coordinate transform
                (see ``Machine._apply_replacement_downstream()``).
                The UI path passes ``False`` because the 3D canvas
                works in world-space degrees.
            apply_gear_ratio: When *True*, the roller gear ratio is
                included in the degrees-to-mu conversion.  Pass
                ``False`` for visualisation paths that should represent
                the object geometry directly, without the roller
                encoding.
        """

        def _on_layer(layer_uid: str, layer_ops: Ops) -> None:
            layer = _resolve_rotary_layer_by_uid(layer_uid, doc)
            if layer is None:
                return
            module = machine.rotary_modules.get(layer.rotary_module_uid or "")
            if module is None or not _is_valid_replacement_module(module):
                return
            mapping = KinematicMapping.from_rotary_module(
                module,
                layer.rotary_diameter,
                apply_gear_ratio=apply_gear_ratio,
            )
            if mapping is None:
                return
            mapping.apply(layer_ops)
            if apply_scaled_mu and module.mode == RotaryMode.AXIS_REPLACEMENT:
                KinematicMapping.degrees_to_scaled_mu_pass(
                    layer_ops,
                    module.mu_per_rotation,
                    target_axis=module.axis,
                )

        ops.transform_layers(_on_layer)

    @staticmethod
    def degrees_to_scaled_mu_pass(
        ops: Ops,
        mu_per_rotation: float,
        target_axis: Axis = Axis.Y,
    ) -> None:
        idx = KinematicMapping._AXIS_TO_INDEX.get(target_axis, 1)
        null_source = (
            target_axis != Axis.Y
            and target_axis in KinematicMapping._AXIS_TO_INDEX
        )

        def on_endpoint(
            end: List[float],
            extra_axes: Dict[Axis, float],
        ) -> None:
            degrees = extra_axes.pop(Axis.Y, None)
            if degrees is None:
                return
            end[idx] = KinematicMath.degrees_to_scaled_mu(
                degrees, mu_per_rotation
            )
            if null_source:
                end[1] = 0.0

        def on_aux_point(point: List[float]) -> None:
            if idx < len(point):
                point[idx] = KinematicMath.degrees_to_scaled_mu(
                    point[idx], mu_per_rotation
                )
            if null_source:
                point[1] = 0.0

        ops.transform_moving(on_endpoint, on_aux_point)
