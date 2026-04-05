"""
Tests for the OpsRenderer class.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from rayforge.ui_gtk.sim3d.canvas3d.ops_renderer import OpsRenderer
from rayforge.shared.util.colors import ColorSet


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


def _init_renderer(renderer):
    with (
        patch.object(renderer, "_create_vbo", return_value=1),
        patch.object(renderer, "_create_vao", return_value=1),
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
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glVertexAttribPointer"),
        patch("OpenGL.GL.glEnableVertexAttribArray"),
    ):
        renderer.init_gl()

    assert mock_vbo.call_count == 4
    assert mock_vao.call_count == 2


@pytest.mark.ui
def test_update_from_vertex_data_sets_counts(renderer):
    _init_renderer(renderer)

    powered_verts = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    powered_colors = np.array([1, 0, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    travel_verts = np.array([2, 2, 2, 3, 3, 3], dtype=np.float32)

    with (
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
    ):
        renderer.update_from_vertex_data(
            powered_verts, powered_colors, travel_verts
        )

    assert renderer.powered_vertex_count == 2
    assert renderer.travel_vertex_count == 2


@pytest.mark.ui
def test_clear_resets_counts(renderer):
    _init_renderer(renderer)

    powered_verts = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    powered_colors = np.array([1, 0, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    travel_verts = np.array([2, 2, 2, 3, 3, 3], dtype=np.float32)

    with (
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
    ):
        renderer.update_from_vertex_data(
            powered_verts, powered_colors, travel_verts
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
    powered_colors = np.array([1, 0, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    travel_verts = np.array([], dtype=np.float32)

    with (
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
    ):
        renderer.update_from_vertex_data(
            powered_verts, powered_colors, travel_verts
        )

    shader = MagicMock()
    mvp = np.eye(4, dtype=np.float32)

    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glDrawArrays"),
    ):
        with pytest.raises(ValueError, match="executed_vertex_count"):
            renderer.render(
                shader,
                mvp,
                colors,
                show_travel_moves=True,
                executed_vertex_count=999,
            )


@pytest.mark.ui
def test_render_draws_powered_and_travel(renderer, colors):
    _init_renderer(renderer)

    powered_verts = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    powered_colors = np.array([1, 0, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    travel_verts = np.array([2, 2, 2, 3, 3, 3], dtype=np.float32)

    with (
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
    ):
        renderer.update_from_vertex_data(
            powered_verts, powered_colors, travel_verts
        )

    shader = MagicMock()
    mvp = np.eye(4, dtype=np.float32)

    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glDrawArrays") as mock_draw,
    ):
        renderer.render(shader, mvp, colors, show_travel_moves=True)

    assert mock_draw.call_count == 2
    shader.use.assert_called_once()
    shader.set_mat4.assert_called_once_with("uMVP", mvp)


@pytest.mark.ui
def test_render_hides_travel_when_disabled(renderer, colors):
    _init_renderer(renderer)

    powered_verts = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    powered_colors = np.array([1, 0, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    travel_verts = np.array([2, 2, 2, 3, 3, 3], dtype=np.float32)

    with (
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
    ):
        renderer.update_from_vertex_data(
            powered_verts, powered_colors, travel_verts
        )

    shader = MagicMock()
    mvp = np.eye(4, dtype=np.float32)

    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glDrawArrays") as mock_draw,
    ):
        renderer.render(shader, mvp, colors, show_travel_moves=False)

    assert mock_draw.call_count == 1


@pytest.mark.ui
def test_render_noop_when_empty(renderer, colors):
    _init_renderer(renderer)

    shader = MagicMock()
    mvp = np.eye(4, dtype=np.float32)

    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glDrawArrays") as mock_draw,
    ):
        renderer.render(shader, mvp, colors, show_travel_moves=True)

    mock_draw.assert_not_called()
