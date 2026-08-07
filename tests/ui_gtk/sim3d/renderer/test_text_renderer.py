"""
Tests for the TextRenderer GL resource lifecycle.
"""

from unittest.mock import MagicMock, patch

import pytest

from rayforge.ui_gtk.sim3d.renderer.text_renderer import TextRenderer


@pytest.mark.ui
def test_init_gl_reuploads_atlas_after_cleanup():
    """A second init_gl after cleanup must recreate the texture and re-upload
    the atlas instead of sampling from a stale texture."""
    renderer = TextRenderer()

    ids = iter(range(1, 100))

    def gen_id():
        return next(ids)

    gl_tex_image = MagicMock()

    with (
        patch.object(renderer, "_create_texture", side_effect=gen_id),
        patch.object(renderer, "_create_vao", side_effect=gen_id),
        patch.object(renderer, "_create_vbo", side_effect=gen_id),
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
        patch("OpenGL.GL.glEnableVertexAttribArray"),
        patch("OpenGL.GL.glVertexAttribPointer"),
        patch("OpenGL.GL.glBindTexture"),
        patch("OpenGL.GL.glGetIntegerv"),
        patch("OpenGL.GL.glPixelStorei"),
        patch("OpenGL.GL.glTexImage2D", gl_tex_image),
        patch("OpenGL.GL.glTexParameteri"),
    ):
        renderer.init_gl()
        first_texture = renderer.texture_id
        assert first_texture != 0

        renderer.cleanup()
        assert renderer.texture_id == 0

        renderer.init_gl()
        assert renderer.texture_id != 0
        assert renderer.texture_id != first_texture
        assert renderer.vao != 0
        assert renderer.vbo != 0

    assert gl_tex_image.call_count == 2
