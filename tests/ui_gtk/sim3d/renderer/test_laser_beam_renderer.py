"""
Tests for the LaserBeamRenderer class.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rayforge.ui_gtk.sim3d.gl_utils import ShaderSet
from rayforge.ui_gtk.sim3d.render_context import CameraContext, RenderContext
from rayforge.ui_gtk.sim3d.renderer.laser_beam_renderer import (
    LaserBeamRenderer,
)

# GL constants without importing OpenGL in tests.
GL_DEPTH_TEST = 0x0B71
GL_LEQUAL = 0x0203
GL_FALSE = 0


@pytest.fixture
def renderer():
    return LaserBeamRenderer()


def _make_ctx():
    proj = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return RenderContext(
        camera=CameraContext(
            proj_matrix=proj,
            view_matrix=np.eye(4, dtype=np.float32),
            viewport_height=800,
        ),
    )


def _init_renderer(renderer):
    with (
        patch.object(renderer, "_create_vbo", return_value=1),
        patch.object(renderer, "_create_vao", return_value=1),
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
        patch("OpenGL.GL.glVertexAttribPointer"),
        patch("OpenGL.GL.glEnableVertexAttribArray"),
    ):
        renderer.init_gl()
    return renderer


@pytest.mark.ui
def test_init_gl_creates_geometry(renderer):
    _init_renderer(renderer)
    assert renderer.vao == 1
    assert renderer.vertex_count > 0


@pytest.mark.ui
def test_render_noop_without_vao():
    renderer = LaserBeamRenderer()
    ctx = _make_ctx()
    shader = MagicMock()
    shaders = ShaderSet(main=shader)
    renderer.render(ctx, shaders)
    shader.use.assert_not_called()


@pytest.mark.ui
def test_render_depth_tests_against_scene(renderer):
    """The beam must be occluded by the laser head model instead of
    always drawing on top, so depth testing stays enabled with LEQUAL
    and depth writes off."""
    _init_renderer(renderer)
    renderer._beams = [
        (
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            50.0,
            (1.0, 0.3, 0.1, 1.0),
        )
    ]
    shader = MagicMock()
    shaders = ShaderSet(main=shader)
    ctx = _make_ctx()

    with (
        patch("OpenGL.GL.glEnable") as mock_enable,
        patch("OpenGL.GL.glDepthFunc") as mock_depth_func,
        patch("OpenGL.GL.glDepthMask") as mock_depth_mask,
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glDrawArrays"),
    ):
        renderer.render(ctx, shaders)

    assert GL_DEPTH_TEST in [
        call.args[0] for call in mock_enable.call_args_list
    ]
    assert mock_depth_func.call_args[0][0] == GL_LEQUAL
    assert mock_depth_mask.call_args[0][0] == GL_FALSE


@pytest.mark.ui
def test_render_uses_additive_blending(renderer):
    _init_renderer(renderer)
    renderer._beams = [
        (
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            50.0,
            (1.0, 0.3, 0.1, 1.0),
        )
    ]
    shader = MagicMock()
    shaders = ShaderSet(main=shader)
    ctx = _make_ctx()

    with (
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glDepthFunc"),
        patch("OpenGL.GL.glDepthMask"),
        patch("OpenGL.GL.glBlendFunc") as mock_blend,
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glDrawArrays") as mock_draw,
    ):
        renderer.render(ctx, shaders)

    # One additive-blend pass per glow layer.
    assert mock_blend.call_count == 16
    assert mock_draw.call_count == 16
