"""
A collection of utility classes and functions for simplifying common
PyOpenGL tasks, such as shader compilation and buffer management.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol

import numpy as np
from OpenGL import GL

from ...core.color import ColorSet

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...machine.models.machine import Machine
    from ...simulator.op_player import OpPlayer
    from ...simulator.scene3d.compiled_scene import CompiledSceneArtifact
    from .shader.base import Shader
    from .viewport import ViewportConfig


def rotation_4x4(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Build a 4x4 rotation matrix from an axis and angle (Rodrigues).

    Returns the identity if *angle* is near zero.
    """
    if abs(angle) < 1e-9:
        return np.eye(4, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return np.eye(4, dtype=np.float64)
    ax = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c
    x, y, z = ax
    rot = np.eye(4, dtype=np.float64)
    rot[:3, :3] = [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
    return rot


def set_line_width(requested: float) -> None:
    try:
        width_range = GL.glGetFloatv(GL.GL_ALIASED_LINE_WIDTH_RANGE)
    except GL.GLError:
        width_range = None

    if width_range is None or len(width_range) < 2:
        GL.glLineWidth(requested)
        return

    min_width = float(width_range[0])
    max_width = float(width_range[1])
    clamped = max(min_width, min(requested, max_width))
    GL.glLineWidth(clamped)


@dataclass
class RenderContext:
    """
    Frame-level rendering state shared across all renderers.

    Matrices are row-major (NumPy convention).  Renderers that need
    column-major for OpenGL should transpose the matrix.

    The frame-state fields below default to ``None``/sentinel so that
    legacy callers that only populate the original geometry fields keep
    working during the polymorphic-renderer migration.  Renderers
    introduced or migrated under the new ``LayerRenderer`` protocol
    read these instead of receiving extra positional arguments.
    """

    proj_matrix: np.ndarray
    view_matrix: np.ndarray
    mvp_ui: np.ndarray
    mvp_scene: np.ndarray
    margin_shift: np.ndarray
    model_matrix: np.ndarray
    viewport_height: int
    camera_position: np.ndarray
    color_set: ColorSet
    show_travel_moves: bool = False
    line_width: float = 1.0

    # --- Frame-state extension (all optional) -----------------------
    machine: "Optional[Machine]" = None
    doc: "Optional[Doc]" = None
    op_player: "Optional[OpPlayer]" = None
    compiled_artifact: "Optional[CompiledSceneArtifact]" = None
    viewport: "Optional[ViewportConfig]" = None
    rotary_axis: Optional[str] = None
    executed_vertex_count: int = -1
    executed_travel_vertex_count: int = -1
    alpha_pending: float = 0.2
    reached_count: Optional[int] = None
    mvp_flat_gl: Optional[np.ndarray] = None
    mvp_rot_gl: Optional[np.ndarray] = None
    cyl_mesh_mvp_gl: Optional[np.ndarray] = None
    laser_light_pos: Optional[np.ndarray] = None
    rot_4x4: Optional[np.ndarray] = None
    show_grid: bool = True
    show_nogo_zones: bool = True
    show_models: bool = True
    had_rotary_layers: bool = False


@dataclass
class ShaderSet:
    """
    Bag of shaders passed to ``LayerRenderer.render``.

    Each renderer picks the program it needs (``main`` / ``text`` /
    ``texture``) instead of receiving a bespoke positional ``shader``
    argument.  Fields default to ``None`` so partial populations are
    valid during migration.
    """

    main: Optional["Shader"] = None
    text: Optional["Shader"] = None
    texture: Optional["Shader"] = None


class LayerRenderer(Protocol):
    """
    Polymorphic renderer contract.

    ``prepare`` performs per-frame state setup (the work formerly done
    by each renderer's divergent ``update_from_state``); ``render``
    performs the GL draw.  Both pull everything they need from the
    shared :class:`RenderContext`.  This is a ``Protocol`` (duck-typed);
    renderers are not required to inherit from it.
    """

    def prepare(self, ctx: RenderContext) -> None: ...

    def render(self, ctx: RenderContext, shaders: ShaderSet) -> None: ...

    def init_gl(self) -> None: ...
