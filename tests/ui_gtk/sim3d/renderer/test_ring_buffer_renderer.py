"""Tests for RingBufferRenderer prepare/render exec-count handling."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rayforge.core.color import ColorSet
from rayforge.ui_gtk.sim3d.gl_utils import ShaderSet
from rayforge.ui_gtk.sim3d.render_context import (
    CameraContext,
    KinematicsContext,
    PlaybackContext,
    RenderContext,
)
from rayforge.ui_gtk.sim3d.renderer.ring_buffer_renderer import (
    RingBufferRenderer,
)

_SET_LINE_WIDTH = (
    "rayforge.ui_gtk.sim3d.renderer.ring_buffer_renderer.set_line_width"
)


class _FakePlayer:
    def __init__(self, current_index):
        self.current_index = current_index

    def playback_progress(self):
        return (self.current_index + 1, 0.0)


def _make_ctx(op_player=None, ring_vertex_count=0):
    renderer_ctx = RenderContext(
        camera=CameraContext(
            color_set=ColorSet({"zero_power": (0.0, 0.0, 1.0, 1.0)}),
        ),
        kinematics=KinematicsContext(mvp_ui=np.eye(4, dtype=np.float32)),
        playback=PlaybackContext(op_player=op_player),
    )
    ring = RingBufferRenderer()
    ring.vertex_count = ring_vertex_count
    return renderer_ctx, ring


@pytest.mark.ui
def test_prepare_computes_ring_exec_count():
    ctx, ring = _make_ctx(op_player=_FakePlayer(0), ring_vertex_count=8)
    ring.ring_offsets = np.array([0, 1, 2], dtype=np.int32)

    ring.prepare(ctx)

    assert ring._exec_ring == 1


@pytest.mark.ui
def test_prepare_without_player_defaults_negative():
    ctx, ring = _make_ctx(ring_vertex_count=8)
    ring.ring_offsets = np.array([0, 1, 2], dtype=np.int32)

    ring.prepare(ctx)

    assert ring._exec_ring == -1


@pytest.mark.ui
def test_render_publishes_ring_exec_count():
    ctx, ring = _make_ctx(op_player=_FakePlayer(0), ring_vertex_count=8)
    ring.ring_offsets = np.array([0, 1, 2], dtype=np.int32)
    ring.prepare(ctx)
    assert ring._exec_ring == 1

    shader = MagicMock()
    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glBindTexture"),
        patch("OpenGL.GL.glActiveTexture"),
        patch("OpenGL.GL.glDepthFunc") as mock_depth_func,
        patch("OpenGL.GL.glDepthMask"),
        patch("OpenGL.GL.glDrawArrays"),
        patch(_SET_LINE_WIDTH),
    ):
        ring.render(ctx, ShaderSet(main=shader))

    assert ctx.playback.executed_vertex_count == 1
    # The trail must depth-test against the laser head model instead of
    # always drawing on top; a small bias keeps it above the coplanar
    # raster texture.  0x0203 is GL.GL_LEQUAL (no GL import in tests).
    assert mock_depth_func.call_args[0][0] == 0x0203
    shader.set_float.assert_any_call("uDepthBias", 2.0)


@pytest.mark.ui
def test_render_noop_when_empty():
    ctx, ring = _make_ctx(ring_vertex_count=0)
    ring.ring_offsets = np.array([0, 1, 2], dtype=np.int32)

    shader = MagicMock()
    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glBindTexture"),
        patch("OpenGL.GL.glActiveTexture"),
        patch("OpenGL.GL.glDepthFunc"),
        patch("OpenGL.GL.glDepthMask"),
        patch("OpenGL.GL.glDrawArrays") as mock_draw,
        patch(_SET_LINE_WIDTH),
    ):
        ring.render(ctx, ShaderSet(main=shader))

    mock_draw.assert_not_called()
