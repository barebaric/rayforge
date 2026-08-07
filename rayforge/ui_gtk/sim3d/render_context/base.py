"""Composite render context and its per-frame input bundle."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from .camera import CameraContext
from .kinematics import KinematicsContext
from .playback import PlaybackContext
from .viewport import ViewportContext

if TYPE_CHECKING:
    from ....core.color import ColorSet
    from ....core.doc import Doc
    from ....machine.models.machine import Machine
    from ....simulator.op_player import OpPlayer
    from ....simulator.scene3d import CompiledSceneArtifact
    from ..camera import Camera
    from ..viewport import ViewportConfig


@dataclass
class FrameInputs:
    """Raw per-frame inputs consumed by the section contexts.

    ``cylinder_transform`` and ``had_rotary_layers`` are provided by the
    canvas from the scene state, so the contexts never reach into the
    SceneRenderer themselves.
    """

    camera: "Camera"
    viewport: "ViewportConfig"
    color_set: "ColorSet"
    op_player: Optional["OpPlayer"] = None
    machine: Optional["Machine"] = None
    compiled_artifact: Optional["CompiledSceneArtifact"] = None
    doc: Optional["Doc"] = None
    cylinder_transform: Optional[np.ndarray] = None
    had_rotary_layers: bool = False
    show_travel_moves: bool = False
    show_grid: bool = True
    show_nogo_zones: bool = True
    show_models: bool = True


class RenderContext:
    """Composite per-frame rendering state, sectioned by concern.

    Matrices are row-major (NumPy convention).  ``Shader.set_mat4`` /
    ``Shader.set_mat3`` transpose to column-major at the GL boundary, so
    renderers pass row-major matrices directly.

    Sections:
      - ``camera``: view/projection matrices, colours, line width and
        display toggles shared by all renderers.
      - ``viewport``: grid/world transforms derived from the viewport.
      - ``kinematics``: pre-computed machine head positions, model
        transforms and rotary matrices.
      - ``playback``: the op player, compiled artifact and per-frame
        execution counters.

    Each section refreshes itself in place from a :class:`FrameInputs`
    bundle via :meth:`update`, so a single context can be reused across
    frames.
    """

    def __init__(
        self,
        camera: Optional[CameraContext] = None,
        viewport: Optional[ViewportContext] = None,
        kinematics: Optional[KinematicsContext] = None,
        playback: Optional[PlaybackContext] = None,
    ):
        self.camera = camera if camera is not None else CameraContext()
        self.viewport = viewport if viewport is not None else ViewportContext()
        self.kinematics = (
            kinematics if kinematics is not None else KinematicsContext()
        )
        self.playback = playback if playback is not None else PlaybackContext()

    def update(self, frame: FrameInputs) -> None:
        """Refreshes every section from the given frame inputs.

        Sections are updated in dependency order: camera and viewport
        first, then kinematics (which consumes both), then playback.
        """
        self.camera.update(frame)
        self.viewport.update(frame)
        self.kinematics.update(
            frame, camera=self.camera, viewport=self.viewport
        )
        self.playback.update(frame)
