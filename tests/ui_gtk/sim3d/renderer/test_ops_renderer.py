"""
Tests for the OpsRenderer class.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rayforge.core.color import ColorSet
from rayforge.ui_gtk.sim3d.gl_utils import RenderContext, ShaderSet
from rayforge.ui_gtk.sim3d.renderer.ops_renderer import OpsRenderer


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


def _make_ctx(colors, show_travel_moves=False):
    return RenderContext(
        proj_matrix=np.eye(4, dtype=np.float32),
        view_matrix=np.eye(4, dtype=np.float32),
        mvp_ui=np.eye(4, dtype=np.float32),
        mvp_scene=np.eye(4, dtype=np.float32),
        margin_shift=np.eye(4, dtype=np.float32),
        model_matrix=np.eye(4, dtype=np.float32),
        viewport_height=800,
        camera_position=np.zeros(3),
        color_set=colors,
        show_travel_moves=show_travel_moves,
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
def test_render_raises_on_invalid_executed_count(renderer, colors):
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
    ctx = _make_ctx(colors, show_travel_moves=True)
    ctx.mvp_flat_gl = mvp
    ctx.executed_vertex_count = 999

    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glDrawArrays"),
        patch("rayforge.ui_gtk.sim3d.renderer.ops_renderer.set_line_width"),
    ):
        with pytest.raises(ValueError, match="executed_vertex_count"):
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
    ctx.mvp_flat_gl = mvp

    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glActiveTexture"),
        patch("OpenGL.GL.glBindTexture"),
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
    ctx.mvp_flat_gl = mvp

    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glActiveTexture"),
        patch("OpenGL.GL.glBindTexture"),
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
    ctx.mvp_flat_gl = mvp

    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glDepthMask"),
        patch("OpenGL.GL.glDrawArrays") as mock_draw,
        patch("rayforge.ui_gtk.sim3d.renderer.ops_renderer.set_line_width"),
    ):
        renderer.render(ctx, _make_shaders(shader))

    mock_draw.assert_not_called()
