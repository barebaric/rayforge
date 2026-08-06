"""Tests for CylinderRenderer.update_from_state."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rayforge.ui_gtk.sim3d.renderer.cylinder_renderer import CylinderRenderer


@pytest.fixture
def renderer():
    r = CylinderRenderer(diameter=20.0, length=50.0)
    with (
        patch.object(r, "_create_vao", return_value=1),
        patch.object(r, "_create_vbo", return_value=1),
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
        patch("OpenGL.GL.glVertexAttribPointer"),
        patch("OpenGL.GL.glEnableVertexAttribArray"),
    ):
        r.init_gl()
    return r


@pytest.mark.ui
def test_render_noop_without_update_from_state(renderer):
    shader = MagicMock()
    with (
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glDrawArrays"),
    ):
        renderer.render(MagicMock(), shader)
    shader.use.assert_not_called()


@pytest.mark.ui
def test_render_uses_cached_mvp(renderer):
    mvp = np.eye(4, dtype=np.float32)
    renderer.update_from_state(mvp)

    shader = MagicMock()
    with (
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glDrawArrays"),
        patch("OpenGL.GL.glDisable"),
    ):
        renderer.render(MagicMock(), shader)

    shader.use.assert_called_once()
    shader.set_mat4.assert_called_once_with("uMVP", mvp)
