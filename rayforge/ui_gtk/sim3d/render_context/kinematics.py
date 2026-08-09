"""Machine kinematics section for the render context.

A single :class:`KinematicsContext` carries both flat and rotary state;
``mvp_for()`` and ``cylinder_mesh_mvp()`` branch on the current rotary
configuration, so renderers do not need to distinguish the two cases
themselves.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ....core.color import hex_to_rgba
from ....machine.models.laser import LaserHead
from ....simulator.machine_state import MachineState
from ..gl_utils import rotation_4x4

if TYPE_CHECKING:
    from raygeo.ops.axis import Axis

    from ....core.doc import Doc
    from ....machine.models.machine import Machine
    from ....simulator.op_player import OpPlayer
    from .base import FrameInputs
    from .camera import CameraContext
    from .viewport import ViewportContext

_DEFAULT_FOCAL_DISTANCE = 50.0
_VIS_ROT_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)


@dataclass
class HeadConfig:
    """Beam visual config for one laser head."""

    beam_height: float
    beam_color: tuple
    valid: bool = True


class KinematicsContext:
    """Pre-computed machine kinematics for the current frame.

    ``model_world_transforms`` / ``head_positions`` / ``head_configs``
    are populated by :meth:`update` from the machine assembly.  Flat
    frames use the plain UI MVP; rotary frames additionally expose a
    rotated toolpath MVP and the cylinder mesh MVP.

    The plain constructor leaves the section empty; call :meth:`update`
    each frame to recompute it from the current state.
    """

    def __init__(
        self,
        *,
        mvp_ui: np.ndarray | None = None,
        mvp_rot: np.ndarray | None = None,
        cyl_mesh_mvp: np.ndarray | None = None,
        model_world_transforms: dict[str, np.ndarray] | None = None,
        head_positions: dict[str, tuple] | None = None,
        head_configs: dict[str, HeadConfig] | None = None,
        rotary_head_positions: dict[str, np.ndarray] | None = None,
        focused_rotary_head_positions: dict[str, np.ndarray] | None = None,
        has_rotary: bool = False,
        rotary_axis: Optional["Axis"] = None,
    ):
        identity = np.eye(4, dtype=np.float32)
        self._mvp_ui = identity if mvp_ui is None else mvp_ui
        self._mvp_rot = mvp_rot
        self._cyl_mesh_mvp = cyl_mesh_mvp
        self.model_world_transforms = model_world_transforms or {}
        self.head_positions = head_positions or {}
        self.head_configs = head_configs or {}
        self.rotary_head_positions = rotary_head_positions or {}
        self.focused_rotary_head_positions = (
            focused_rotary_head_positions or {}
        )
        self.has_rotary = has_rotary
        self.rotary_axis = rotary_axis
        self.laser_light_pos: np.ndarray | None = None

    @property
    def is_rotary(self) -> bool:
        """True when a rotary axis is active for this frame."""
        return self.rotary_axis is not None and self.has_rotary

    def mvp_for(self, renderer_is_rotary: bool) -> np.ndarray:
        """MVP for a toolpath renderer, rotary or flat layer."""
        if renderer_is_rotary and self._mvp_rot is not None:
            return self._mvp_rot
        return self._mvp_ui

    def cylinder_mesh_mvp(self) -> np.ndarray | None:
        """MVP for the rotary cylinder mesh, or None when not rotary."""
        return self._cyl_mesh_mvp

    def update(
        self,
        frame: "FrameInputs",
        *,
        camera: "CameraContext",
        viewport: "ViewportContext",
    ) -> None:
        """Recomputes the kinematics section from the current frame."""
        self.laser_light_pos = None
        mvp_ui = camera.mvp_ui
        machine = frame.machine
        asm = frame.playback_assembly
        if asm is None and machine is not None:
            asm = machine.assembly
        if asm is None:
            self._apply_flat(
                mvp_ui,
                model_world_transforms={},
                head_positions={},
                head_configs={},
                has_rotary=False,
            )
            return

        op_player = frame.op_player
        state = (
            op_player.render_state()
            if op_player is not None
            else MachineState()
        )
        wcs = viewport.wcs_offset_mm
        model_world_transforms = asm.model_world_transforms(
            state, wcs_offset=wcs
        )
        try:
            head_positions = asm.head_positions(state, wcs_offset=wcs)
        except ValueError:
            head_positions = {}
        head_configs = {
            name: _head_config(machine, name) for name in head_positions
        }
        has_rotary = asm.has_rotary

        if not frame.had_rotary_layers:
            self._apply_flat(
                mvp_ui,
                model_world_transforms=model_world_transforms,
                head_positions=head_positions,
                head_configs=head_configs,
                has_rotary=has_rotary,
            )
            return

        op_player = frame.op_player
        rotary_axis = op_player.rotary_axis if op_player else None
        diameter = (
            _current_rotary_diameter(op_player, frame.doc)
            if op_player
            else 0.0
        )
        rotary_head_positions = asm.head_rotary_positions(state, diameter)
        focused = _focused_rotary_head_positions(machine, asm, state, diameter)

        cyl_angle = 0.0
        if op_player and rotary_axis is not None and has_rotary:
            cyl_angle = math.radians(state.axes.get(rotary_axis, 0.0))

        physical_to_visual = (
            viewport.margin_shift @ viewport.native_to_workspace
        )
        cylinder_transform = (
            frame.cylinder_transform
            if frame.cylinder_transform is not None
            else np.eye(4, dtype=np.float64)
        )
        cyl_base_mvp = (
            mvp_ui.astype(np.float64)
            @ physical_to_visual.astype(np.float64)
            @ cylinder_transform
        )
        rot_4x4 = rotation_4x4(_VIS_ROT_AXIS, cyl_angle)
        mvp_rot = (cyl_base_mvp @ rot_4x4).astype(np.float32)
        cyl_mesh_mvp = (
            mvp_ui @ physical_to_visual @ cylinder_transform @ rot_4x4
        ).astype(np.float32)

        self._mvp_ui = mvp_ui
        self._mvp_rot = mvp_rot
        self._cyl_mesh_mvp = cyl_mesh_mvp
        self.model_world_transforms = model_world_transforms
        self.head_positions = head_positions
        self.head_configs = head_configs
        self.rotary_head_positions = rotary_head_positions
        self.focused_rotary_head_positions = focused
        self.has_rotary = has_rotary
        self.rotary_axis = rotary_axis

    def _apply_flat(
        self,
        mvp_ui: np.ndarray,
        *,
        model_world_transforms: dict[str, np.ndarray],
        head_positions: dict[str, tuple],
        head_configs: dict[str, HeadConfig],
        has_rotary: bool,
    ) -> None:
        self._mvp_ui = mvp_ui
        self._mvp_rot = None
        self._cyl_mesh_mvp = None
        self.model_world_transforms = model_world_transforms
        self.head_positions = head_positions
        self.head_configs = head_configs
        self.rotary_head_positions = {}
        self.focused_rotary_head_positions = {}
        self.has_rotary = has_rotary
        self.rotary_axis = None


def _current_rotary_diameter(
    op_player: "OpPlayer", doc: Optional["Doc"]
) -> float:
    """Return the current layer's rotary diameter, or 0.0 if none."""
    if doc is None:
        return 0.0
    current_layer = op_player.get_current_layer(doc)
    if current_layer is None:
        return 0.0
    return current_layer.rotary_diameter or 0.0


def _head_focal_distance(
    machine: Optional["Machine"], head_name: str
) -> float:
    """Return the focal distance of the named laser head."""
    if machine is None or not head_name.startswith("head_"):
        return _DEFAULT_FOCAL_DISTANCE
    try:
        idx = int(head_name.split("_")[1])
        laser = machine.heads[idx]
    except (ValueError, IndexError, TypeError, AttributeError):
        return _DEFAULT_FOCAL_DISTANCE
    if not isinstance(laser, LaserHead):
        return _DEFAULT_FOCAL_DISTANCE
    if laser.focal_distance and laser.focal_distance > 0:
        return laser.focal_distance
    return _DEFAULT_FOCAL_DISTANCE


def _focused_rotary_head_positions(
    machine: Optional["Machine"],
    asm,
    state: MachineState,
    diameter: float,
) -> dict[str, np.ndarray]:
    """Rotary head positions with each head's focal distance applied.

    Only links with HEAD role appear in the result, so it doubles as the
    set of model links that should be placed above the cylinder.
    """
    if not asm.has_rotary:
        return {}
    result: dict[str, np.ndarray] = {}
    for name in asm.head_rotary_positions(state, diameter):
        focal = _head_focal_distance(machine, name)
        focused = asm.head_rotary_positions(
            state, diameter, focal_distance=focal
        )
        if name in focused:
            result[name] = focused[name]
    return result


def _head_config(machine: Optional["Machine"], head_name: str) -> HeadConfig:
    """Return the beam config for the named head link."""
    default = HeadConfig(
        beam_height=_DEFAULT_FOCAL_DISTANCE,
        beam_color=(1.0, 0.3, 0.1, 1.0),
    )
    if machine is None or not head_name.startswith("head_"):
        return default
    try:
        idx = int(head_name.split("_")[1])
        laser = machine.heads[idx]
    except (ValueError, IndexError, TypeError, AttributeError):
        return default
    if not isinstance(laser, LaserHead):
        return HeadConfig(
            beam_height=_DEFAULT_FOCAL_DISTANCE,
            beam_color=(1.0, 0.3, 0.1, 1.0),
            valid=False,
        )
    beam_height = (
        laser.focal_distance
        if laser.focal_distance and laser.focal_distance > 0
        else _DEFAULT_FOCAL_DISTANCE
    )
    return HeadConfig(
        beam_height=beam_height, beam_color=hex_to_rgba(laser.cut_color)
    )
