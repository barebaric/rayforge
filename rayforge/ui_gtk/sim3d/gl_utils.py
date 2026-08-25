"""
A collection of utility classes and functions for simplifying common
PyOpenGL tasks, such as shader compilation and buffer management.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from OpenGL import GL

if TYPE_CHECKING:
    from .shader.base import Shader


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


# Toolpath and scanline-trail lines lie exactly on the workpiece
# surface, so a plain depth test would z-fight with the raster texture
# (and the faceted cylinder mesh).  The line renderers use the
# LineDepthBiasShader, whose fragment shader subtracts this constant
# from gl_FragCoord.z: a window-space offset of a few depth-buffer
# LSBs that wins ties against coplanar geometry (and the stock fill,
# which glPolygonOffset pushes away by one unit) while staying far
# below the depth separation of the laser head model, so the head
# still occludes the lines.  Unlike a clip-space bias it changes
# depth ordering only — the lines keep their exact projected
# position in perspective and ortho alike, and the offset does not
# degrade with viewing distance.
#
# 2.4e-7 is ~4 LSB of a 24-bit fixed-point depth buffer.
LINE_DEPTH_WINDOW_BIAS = 2.4e-7


@dataclass
class ShaderSet:
    """
    Bag of shaders passed to ``render``.

    Each renderer picks the program it needs (``main`` / ``text`` /
    ``stock``) instead of receiving a bespoke positional ``shader``
    argument.  Fields default to ``None`` so partial populations are
    valid during migration.
    """

    main: Shader | None = None
    main_lines: Shader | None = None
    text: Shader | None = None
    background: Shader | None = None
    stock: Shader | None = None
    image: Shader | None = None
