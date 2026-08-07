"""Tests for the row-major -> column-major matrix convention boundary.

The chosen boundary is: ``RenderContext`` carries row-major (NumPy
convention) matrices, and ``Shader.set_mat4``/``set_mat3`` transpose to
column-major at the GL boundary (``GL_TRUE``).  These tests guard that
boundary so renderers must keep passing row-major matrices.
"""

from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from OpenGL import GL

from rayforge.core.color import ColorSet
from rayforge.ui_gtk.sim3d.gl_utils import ShaderSet
from rayforge.ui_gtk.sim3d.render_context import (
    CameraContext,
    KinematicsContext,
    RenderContext,
    ViewportContext,
)
from rayforge.ui_gtk.sim3d.renderer.plane_renderer import PlaneRenderer
from rayforge.ui_gtk.sim3d.renderer.texture_renderer import (
    TextureArtifactRenderer,
)
from rayforge.ui_gtk.sim3d.shader.base import Shader


def _make_shader() -> Shader:
    with (
        patch("OpenGL.GL.glGetString", return_value=b"4.6"),
        patch("OpenGL.GL.shaders.compileShader", return_value=1),
        patch("OpenGL.GL.shaders.compileProgram", return_value=42),
    ):
        return Shader("void main(){}", "void main(){}")


def _make_ctx(
    mvp_ui: Optional[np.ndarray] = None,
    model_matrix: Optional[np.ndarray] = None,
    cyl_mesh_mvp: Optional[np.ndarray] = None,
) -> RenderContext:
    identity = np.eye(4, dtype=np.float32)
    return RenderContext(
        camera=CameraContext(
            mvp_ui=mvp_ui if mvp_ui is not None else identity,
            viewport_height=800,
            camera_position=np.zeros(3),
            color_set=ColorSet(),
        ),
        viewport=ViewportContext(
            model_matrix=(
                model_matrix if model_matrix is not None else identity
            )
        ),
        kinematics=KinematicsContext(
            mvp_ui=mvp_ui if mvp_ui is not None else identity,
            cyl_mesh_mvp=cyl_mesh_mvp,
        ),
    )


@pytest.mark.ui
def test_set_mat4_requests_gl_transpose():
    """set_mat4 uploads row-major data and asks GL to transpose it."""
    shader = _make_shader()
    row_major = np.arange(16, dtype=np.float32).reshape(4, 4)

    with (
        patch("OpenGL.GL.glGetUniformLocation", return_value=0),
        patch("OpenGL.GL.glUniformMatrix4fv") as mock_upload,
    ):
        shader.set_mat4("uMVP", row_major)

    mock_upload.assert_called_once()
    args = mock_upload.call_args.args
    assert args[2] == GL.GL_TRUE
    np.testing.assert_array_equal(args[3], row_major)


@pytest.mark.ui
def test_set_mat3_requests_gl_transpose():
    """set_mat3 uploads row-major data and asks GL to transpose it."""
    shader = _make_shader()
    row_major = np.arange(9, dtype=np.float32).reshape(3, 3)

    with (
        patch("OpenGL.GL.glGetUniformLocation", return_value=0),
        patch("OpenGL.GL.glUniformMatrix3fv") as mock_upload,
    ):
        shader.set_mat3("uBillboard", row_major)

    mock_upload.assert_called_once()
    args = mock_upload.call_args.args
    assert args[2] == GL.GL_TRUE
    np.testing.assert_array_equal(args[3], row_major)


@pytest.mark.ui
def test_plane_renderer_passes_row_major_mvp():
    """Renderers hand row-major matrices to set_mat4 (GL transposes)."""
    renderer = PlaneRenderer(
        width=100.0, height=80.0, color=(1.0, 0.0, 0.0, 1.0)
    )
    renderer.vao = 1

    mvp_ui = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 5.0, 6.0],
            [0.0, 0.0, 1.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    model_matrix = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ctx = _make_ctx(mvp_ui=mvp_ui, model_matrix=model_matrix)

    shader = MagicMock()
    with patch("rayforge.ui_gtk.sim3d.renderer.plane_renderer.GL"):
        renderer.render(ctx, ShaderSet(main=shader))

    shader.set_mat4.assert_called_once()
    name, mat = shader.set_mat4.call_args.args
    assert name == "uMVP"
    np.testing.assert_allclose(mat, mvp_ui @ model_matrix)


@pytest.mark.ui
def test_texture_renderer_cylinder_mvp_kept_row_major():
    """The cylinder MVP is stored row-major with no double transpose."""
    renderer = TextureArtifactRenderer()
    cyl_mvp = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=np.float32,
    )
    ctx = _make_ctx(cyl_mesh_mvp=cyl_mvp)

    renderer.prepare(ctx)

    assert renderer._cyl_mvp is cyl_mvp
    assert renderer._flat_mvp is ctx.camera.mvp_ui
