"""
Tests for the OpsRenderer class.
"""

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
from rayforge.ui_gtk.sim3d.renderer.ops_renderer import OpsRenderer


class _FakePlayer:
    def __init__(self, current_index):
        self.current_index = current_index


@pytest.fixture
def renderer():
    return OpsRenderer()


@pytest.fixture
def colors():
    return ColorSet(
        {
            "cut": np.zeros((256, 4)),
            "engrave": np.zeros((256, 4)),
            "travel": (0.0, 1.0, 0.0, 1.0),
            "zero_power": (0.0, 0.0, 1.0, 1.0),
        }
    )


def _make_ctx(colors, show_travel_moves=False, op_player=None):
    return RenderContext(
        camera=CameraContext(
            color_set=colors,
            show_travel_moves=show_travel_moves,
        ),
        kinematics=KinematicsContext(mvp_ui=np.eye(4, dtype=np.float32)),
        playback=PlaybackContext(op_player=op_player),
    )


def _make_shaders(shader):
    return ShaderSet(main=shader)


def _init_renderer(renderer):
    with (
        patch.object(renderer, "_create_vbo", return_value=1),
        patch.object(renderer, "_create_vao", return_value=1),
        patch.object(renderer, "_create_texture", return_value=1),
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glVertexAttribPointer"),
        patch("OpenGL.GL.glEnableVertexAttribArray"),
    ):
        renderer.init_gl()
    return renderer


@pytest.mark.ui
def test_init_gl_creates_buffers(renderer):
    with (
        patch.object(renderer, "_create_vbo", return_value=1) as mock_vbo,
        patch.object(renderer, "_create_vao", return_value=1) as mock_vao,
        patch.object(renderer, "_create_texture", return_value=1),
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glVertexAttribPointer"),
        patch("OpenGL.GL.glEnableVertexAttribArray"),
    ):
        renderer.init_gl()

    assert mock_vbo.call_count == 3
    assert mock_vao.call_count == 2


def _make_attrib(n: int) -> np.ndarray:
    a = np.zeros((n, 4), dtype=np.float32)
    a[:, 3] = 1.0
    return a


@pytest.mark.ui
def test_update_from_vertex_data_sets_counts(renderer):
    _init_renderer(renderer)

    powered_verts = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    powered_attrib = _make_attrib(2)
    travel_verts = np.array([2, 2, 2, 3, 3, 3], dtype=np.float32)

    with (
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
    ):
        renderer.update_from_vertex_data(
            powered_verts, powered_attrib, travel_verts
        )

    assert renderer.powered_vertex_count == 2
    assert renderer.travel_vertex_count == 2


@pytest.mark.ui
def test_clear_resets_counts(renderer):
    _init_renderer(renderer)

    powered_verts = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    powered_attrib = _make_attrib(2)
    travel_verts = np.array([2, 2, 2, 3, 3, 3], dtype=np.float32)

    with (
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
    ):
        renderer.update_from_vertex_data(
            powered_verts, powered_attrib, travel_verts
        )
    assert renderer.powered_vertex_count == 2

    with (
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
    ):
        renderer.clear()
    assert renderer.powered_vertex_count == 0
    assert renderer.travel_vertex_count == 0


@pytest.mark.ui
def test_render_clamps_executed_count_to_vertex_count(renderer, colors):
    _init_renderer(renderer)

    powered_verts = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    powered_attrib = _make_attrib(2)
    travel_verts = np.array([], dtype=np.float32)

    with (
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
    ):
        renderer.update_from_vertex_data(
            powered_verts, powered_attrib, travel_verts
        )

    shader = MagicMock()
    mvp = np.eye(4, dtype=np.float32)
    renderer.powered_offsets = np.array([0, 999], dtype=np.int32)
    renderer.travel_offsets = np.array([0, 999], dtype=np.int32)
    ctx = _make_ctx(colors, show_travel_moves=True, op_player=_FakePlayer(0))
    ctx.kinematics._mvp_ui = mvp
    renderer.prepare(ctx)

    # The fractional exec mapping clamps to the uploaded vertex count
    # instead of overrunning the buffer.
    assert renderer._exec_powered == 2
    assert renderer._partial_powered_id == -1

    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glActiveTexture"),
        patch("OpenGL.GL.glBindTexture"),
        patch("OpenGL.GL.glDepthFunc"),
        patch("OpenGL.GL.glDepthMask"),
        patch("OpenGL.GL.glDrawArrays"),
        patch("rayforge.ui_gtk.sim3d.renderer.ops_renderer.set_line_width"),
    ):
        renderer.render(ctx, _make_shaders(shader))


@pytest.mark.ui
def test_render_draws_powered_and_travel(renderer, colors):
    _init_renderer(renderer)

    powered_verts = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    powered_attrib = _make_attrib(2)
    travel_verts = np.array([2, 2, 2, 3, 3, 3], dtype=np.float32)

    with (
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
    ):
        renderer.update_from_vertex_data(
            powered_verts, powered_attrib, travel_verts
        )

    shader = MagicMock()
    mvp = np.eye(4, dtype=np.float32)
    ctx = _make_ctx(colors, show_travel_moves=True)
    ctx.kinematics._mvp_ui = mvp

    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glActiveTexture"),
        patch("OpenGL.GL.glBindTexture"),
        patch("OpenGL.GL.glDepthFunc"),
        patch("OpenGL.GL.glDepthMask"),
        patch("OpenGL.GL.glDrawArrays") as mock_draw,
        patch("rayforge.ui_gtk.sim3d.renderer.ops_renderer.set_line_width"),
    ):
        renderer.render(ctx, _make_shaders(shader))

    assert mock_draw.call_count == 2
    shader.use.assert_called_once()
    shader.set_mat4.assert_called_once_with("uMVP", mvp)


@pytest.mark.ui
def test_render_hides_travel_when_disabled(renderer, colors):
    _init_renderer(renderer)

    powered_verts = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    powered_attrib = _make_attrib(2)
    travel_verts = np.array([2, 2, 2, 3, 3, 3], dtype=np.float32)

    with (
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
    ):
        renderer.update_from_vertex_data(
            powered_verts, powered_attrib, travel_verts
        )

    shader = MagicMock()
    mvp = np.eye(4, dtype=np.float32)
    ctx = _make_ctx(colors, show_travel_moves=False)
    ctx.kinematics._mvp_ui = mvp

    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glActiveTexture"),
        patch("OpenGL.GL.glBindTexture"),
        patch("OpenGL.GL.glDepthFunc"),
        patch("OpenGL.GL.glDepthMask"),
        patch("OpenGL.GL.glDrawArrays") as mock_draw,
        patch("rayforge.ui_gtk.sim3d.renderer.ops_renderer.set_line_width"),
    ):
        renderer.render(ctx, _make_shaders(shader))

    assert mock_draw.call_count == 1


@pytest.mark.ui
def test_render_noop_when_empty(renderer, colors):
    _init_renderer(renderer)

    shader = MagicMock()
    mvp = np.eye(4, dtype=np.float32)
    ctx = _make_ctx(colors, show_travel_moves=True)
    ctx.kinematics._mvp_ui = mvp

    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glDepthFunc"),
        patch("OpenGL.GL.glDepthMask"),
        patch("OpenGL.GL.glDrawArrays") as mock_draw,
        patch("rayforge.ui_gtk.sim3d.renderer.ops_renderer.set_line_width"),
    ):
        renderer.render(ctx, _make_shaders(shader))

    mock_draw.assert_not_called()


@pytest.mark.ui
def test_prepare_publishes_exec_counts_to_render():
    """prepare computes powered/travel exec counts from the playhead and
    render publishes them back into ctx for the draw."""
    renderer = OpsRenderer()
    _init_renderer(renderer)

    powered_verts = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    powered_attrib = _make_attrib(2)
    travel_verts = np.array([2, 2, 2, 3, 3, 3], dtype=np.float32)
    with (
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
    ):
        renderer.update_from_vertex_data(
            powered_verts, powered_attrib, travel_verts
        )

    renderer.powered_offsets = np.array([0, 2, 4], dtype=np.int32)
    renderer.travel_offsets = np.array([0, 2], dtype=np.int32)

    shader = MagicMock()
    mvp = np.eye(4, dtype=np.float32)
    ctx = _make_ctx(
        ColorSet(), show_travel_moves=True, op_player=_FakePlayer(0)
    )
    ctx.kinematics._mvp_ui = mvp
    renderer.prepare(ctx)

    assert renderer._exec_powered == 2
    assert renderer._exec_travel == 2

    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glActiveTexture"),
        patch("OpenGL.GL.glBindTexture"),
        patch("OpenGL.GL.glDepthFunc"),
        patch("OpenGL.GL.glDepthMask"),
        patch("OpenGL.GL.glDrawArrays"),
        patch("rayforge.ui_gtk.sim3d.renderer.ops_renderer.set_line_width"),
    ):
        renderer.render(ctx, _make_shaders(shader))

    assert ctx.playback.executed_vertex_count == 2
    assert ctx.playback.executed_travel_vertex_count == 2
