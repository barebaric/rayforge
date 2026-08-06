"""
OpenGL pipeline-state save/restore context managers.

Used by renderers conforming to the ``LayerRenderer`` protocol to
bracket their ``render`` body so that state mutations — depth test,
blend, blend function, depth mask, depth function, line width, and
pixel unpack alignment — are restored on exit, including the
exceptional path.

Usage::

    with gl_state():
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        # ... draw ...

The :func:`uniform_block` companion snapshots and restores the
per-uniform values a ``Shader`` has tracked via ``set_*``.
"""

from contextlib import contextmanager
from typing import Generator, Optional

from OpenGL import GL


def _get_int(name: int) -> int:
    val = GL.glGetIntegerv(name)
    if val is None or len(val) == 0:
        return 0
    return int(val[0])


def _get_float(name: int) -> float:
    val = GL.glGetFloatv(name)
    if val is None or len(val) == 0:
        return 0.0
    return float(val[0])


def _is_enabled(name: int) -> bool:
    return bool(GL.glIsEnabled(name))


@contextmanager
def gl_state(
    *,
    save_depth_test: bool = True,
    save_blend: bool = True,
    save_depth_mask: bool = True,
    save_depth_func: bool = True,
    save_line_width: bool = True,
    save_unpack_alignment: bool = True,
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

    try:
        if save_depth_test:
            snap_depth_test = _is_enabled(GL.GL_DEPTH_TEST)
        if save_blend:
            snap_blend = _is_enabled(GL.GL_BLEND)
            blend_val = GL.glGetIntegerv(GL.GL_BLEND_SRC_ALPHA)
            if blend_val is not None and len(blend_val) >= 2:
                snap_blend_src = int(blend_val[0])
                snap_blend_dst = int(blend_val[1])
        if save_depth_mask:
            snap_depth_mask = bool(_get_int(GL.GL_DEPTH_WRITEMASK))
        if save_depth_func:
            snap_depth_func = _get_int(GL.GL_DEPTH_FUNC)
        if save_line_width:
            snap_line_width = _get_float(GL.GL_LINE_WIDTH)
        if save_unpack_alignment:
            snap_unpack_alignment = _get_int(GL.GL_UNPACK_ALIGNMENT)
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


@contextmanager
def uniform_block(shader) -> Generator[dict, None, None]:
    """
    Snapshot ``shader`` uniform values on entry, restore on exit.

    Pairs with :meth:`Shader.save` / :meth:`Shader.restore` (see
    ``gl_utils.py``).  Renderers enter this around their uniform-sets
    so a partial draw that throws cannot leave stale values for the
    next renderer.

    Yields the snapshot dict so the caller can inspect what was saved.
    """
    snapshot = shader.save()
    try:
        yield snapshot
    finally:
        shader.restore(snapshot)
