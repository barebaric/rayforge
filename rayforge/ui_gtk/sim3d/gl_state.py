"""
OpenGL pipeline-state save/restore context managers.

The :func:`gl_state` context manager brackets a renderer's ``render``
body so that state mutations — depth test, blend, blend function,
depth mask, depth function, line width, pixel unpack alignment, and
texture bindings — are restored on exit, including the exceptional
path.

The :func:`render_pass` context manager combines ``gl_state`` with the
``Shader`` context-manager protocol (``with shader:`` snapshots and
restores that shader's uniforms) for a single renderer draw call.

Usage::

    with render_pass(self.main_shader):
        self.zone_renderer.render(ctx, self.shader_set)
"""

from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from typing import Optional

import numpy as np
from OpenGL import GL

from .shader.base import Shader


def _get_int(name: int) -> int:
    value = GL.glGetIntegerv(name)
    if value is None:
        return 0
    flat = np.asarray(value).reshape(-1)
    if flat.size == 0:
        return 0
    return int(flat.item(0))


def _get_float(name: int) -> float:
    value = GL.glGetFloatv(name)
    if value is None:
        return 0.0
    flat = np.asarray(value).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(flat.item(0))


def _is_enabled(name: int) -> bool:
    return bool(GL.glIsEnabled(name))


_TEXTURE_UNITS = (GL.GL_TEXTURE0, GL.GL_TEXTURE1)


@contextmanager
def gl_state(
    *,
    save_depth_test: bool = True,
    save_blend: bool = True,
    save_depth_mask: bool = True,
    save_depth_func: bool = True,
    save_line_width: bool = True,
    save_unpack_alignment: bool = True,
    save_texture_bindings: bool = True,
) -> Generator[None, None, None]:
    """
    Snapshot a set of GL pipeline states on entry, restore on exit.

    Each ``save_*`` flag toggles whether a given state is snapshotted.
    Renderers that are known not to touch a state can skip its
    save/restore to avoid extra GL queries.
    """
    snap_depth_test: Optional[bool] = None
    snap_blend: Optional[bool] = None
    snap_blend_src: Optional[int] = None
    snap_blend_dst: Optional[int] = None
    snap_depth_mask: Optional[bool] = None
    snap_depth_func: Optional[int] = None
    snap_line_width: Optional[float] = None
    snap_unpack_alignment: Optional[int] = None
    snap_active_texture: Optional[int] = None
    snap_texture_bindings: dict = {}

    try:
        if save_depth_test:
            snap_depth_test = _is_enabled(GL.GL_DEPTH_TEST)
        if save_blend:
            snap_blend = _is_enabled(GL.GL_BLEND)
            snap_blend_src = _get_int(GL.GL_BLEND_SRC_RGB)
            snap_blend_dst = _get_int(GL.GL_BLEND_DST_RGB)
        if save_depth_mask:
            snap_depth_mask = bool(_get_int(GL.GL_DEPTH_WRITEMASK))
        if save_depth_func:
            snap_depth_func = _get_int(GL.GL_DEPTH_FUNC)
        if save_line_width:
            snap_line_width = _get_float(GL.GL_LINE_WIDTH)
        if save_unpack_alignment:
            snap_unpack_alignment = _get_int(GL.GL_UNPACK_ALIGNMENT)
        if save_texture_bindings:
            snap_active_texture = _get_int(GL.GL_ACTIVE_TEXTURE)
            for unit in _TEXTURE_UNITS:
                GL.glActiveTexture(unit)
                snap_texture_bindings[unit] = _get_int(
                    GL.GL_TEXTURE_BINDING_2D
                )
            if snap_active_texture is not None:
                GL.glActiveTexture(snap_active_texture)
        yield
    finally:
        if snap_depth_test is not None:
            if snap_depth_test:
                GL.glEnable(GL.GL_DEPTH_TEST)
            else:
                GL.glDisable(GL.GL_DEPTH_TEST)
        if snap_blend is not None:
            if snap_blend:
                GL.glEnable(GL.GL_BLEND)
            else:
                GL.glDisable(GL.GL_BLEND)
        if snap_blend_src is not None and snap_blend_dst is not None:
            GL.glBlendFunc(snap_blend_src, snap_blend_dst)
        if snap_depth_mask is not None:
            GL.glDepthMask(GL.GL_TRUE if snap_depth_mask else GL.GL_FALSE)
        if snap_depth_func is not None:
            GL.glDepthFunc(snap_depth_func)
        if snap_line_width is not None:
            GL.glLineWidth(snap_line_width)
        if snap_unpack_alignment is not None:
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, snap_unpack_alignment)
        if snap_texture_bindings:
            for unit, binding in snap_texture_bindings.items():
                GL.glActiveTexture(unit)
                GL.glBindTexture(GL.GL_TEXTURE_2D, binding)
            if snap_active_texture is not None:
                GL.glActiveTexture(snap_active_texture)


@contextmanager
def render_pass(*shaders: Optional[Shader]) -> Generator[None, None, None]:
    """
    Bracket a single renderer draw call with state isolation.

    Saves and restores GL pipeline state and, for each shader provided,
    snapshots and restores that shader's uniforms via its context-manager
    protocol.  A renderer wrapped by this context manager cannot leak GL
    state or uniform changes to subsequent renderers, even on exception.

    Example::

        with render_pass(self.main_shader, self.text_shader):
            self.axis_renderer.render(ctx, self.shader_set)
    """
    with gl_state():
        if not shaders:
            yield
            return
        with ExitStack() as stack:
            for shader in shaders:
                if shader is not None:
                    stack.enter_context(shader)
            yield
