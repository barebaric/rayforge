"""
Tests for the gl_state / uniform_block context managers.

These tests are headless-safe: no GTK or real GL context is required.
All ``OpenGL.GL`` calls are patched via ``unittest.mock``.
"""

from unittest.mock import MagicMock, patch

import pytest
from OpenGL import GL

from rayforge.ui_gtk.sim3d.gl_state import gl_state, uniform_block


def test_gl_state_restores_depth_test_on_exit():
    enabled_calls = []
    disabled_calls = []

    def fake_enable(cap):
        enabled_calls.append(cap)

    def fake_disable(cap):
        disabled_calls.append(cap)

    with (
        patch("rayforge.ui_gtk.sim3d.gl_state._is_enabled", return_value=True),
        patch(
            "rayforge.ui_gtk.sim3d.gl_state.GL.glGetIntegerv",
            return_value=[GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA],
        ),
        patch("rayforge.ui_gtk.sim3d.gl_state._get_int", return_value=1),
        patch("rayforge.ui_gtk.sim3d.gl_state._get_float", return_value=1.0),
        patch(
            "rayforge.ui_gtk.sim3d.gl_state.GL.glEnable",
            side_effect=fake_enable,
        ),
        patch(
            "rayforge.ui_gtk.sim3d.gl_state.GL.glDisable",
            side_effect=fake_disable,
        ),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glBlendFunc"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDepthMask"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDepthFunc"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glLineWidth"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glPixelStorei"),
    ):
        with gl_state():
            pass

    assert GL.GL_DEPTH_TEST in enabled_calls


def test_gl_state_restores_depth_test_when_disabled_snapshot():
    enabled = []
    disabled = []

    with (
        patch(
            "rayforge.ui_gtk.sim3d.gl_state._is_enabled",
            return_value=False,
        ),
        patch(
            "rayforge.ui_gtk.sim3d.gl_state.GL.glGetIntegerv",
            return_value=[GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA],
        ),
        patch("rayforge.ui_gtk.sim3d.gl_state._get_int", return_value=0),
        patch("rayforge.ui_gtk.sim3d.gl_state._get_float", return_value=1.0),
        patch(
            "rayforge.ui_gtk.sim3d.gl_state.GL.glEnable",
            side_effect=lambda c: enabled.append(c),
        ),
        patch(
            "rayforge.ui_gtk.sim3d.gl_state.GL.glDisable",
            side_effect=lambda c: disabled.append(c),
        ),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glBlendFunc"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDepthMask"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDepthFunc"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glLineWidth"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glPixelStorei"),
    ):
        with gl_state():
            pass

    assert GL.GL_DEPTH_TEST in disabled


def test_gl_state_restores_on_exception():
    blend_func = MagicMock()
    line_width = MagicMock()
    pixel_store = MagicMock()

    with (
        patch("rayforge.ui_gtk.sim3d.gl_state._is_enabled", return_value=True),
        patch(
            "rayforge.ui_gtk.sim3d.gl_state.GL.glGetIntegerv",
            return_value=[GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA],
        ),
        patch("rayforge.ui_gtk.sim3d.gl_state._get_int", return_value=1),
        patch("rayforge.ui_gtk.sim3d.gl_state._get_float", return_value=2.5),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glEnable"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDisable"),
        patch(
            "rayforge.ui_gtk.sim3d.gl_state.GL.glBlendFunc",
            side_effect=blend_func,
        ),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDepthMask"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDepthFunc"),
        patch(
            "rayforge.ui_gtk.sim3d.gl_state.GL.glLineWidth",
            side_effect=line_width,
        ),
        patch(
            "rayforge.ui_gtk.sim3d.gl_state.GL.glPixelStorei",
            side_effect=pixel_store,
        ),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            with gl_state():
                raise RuntimeError("boom")

    line_width.assert_called_with(2.5)
    pixel_store.assert_called_with(GL.GL_UNPACK_ALIGNMENT, 1)
    blend_func.assert_called_with(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)


def test_gl_state_skip_flags_avoid_queries():
    with (
        patch("rayforge.ui_gtk.sim3d.gl_state._is_enabled") as mock_is_enabled,
        patch("rayforge.ui_gtk.sim3d.gl_state._get_float") as mock_get_float,
        patch("rayforge.ui_gtk.sim3d.gl_state._get_int") as mock_get_int,
        patch(
            "rayforge.ui_gtk.sim3d.gl_state.GL.glGetIntegerv"
        ) as mock_get_integerv,
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glEnable"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDisable"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glBlendFunc"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDepthMask"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDepthFunc"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glLineWidth"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glPixelStorei"),
    ):
        with gl_state(
            save_depth_test=False,
            save_blend=False,
            save_depth_mask=False,
            save_depth_func=False,
            save_line_width=False,
            save_unpack_alignment=False,
        ):
            pass

    mock_is_enabled.assert_not_called()
    mock_get_float.assert_not_called()
    mock_get_int.assert_not_called()
    mock_get_integerv.assert_not_called()


def test_uniform_block_snapshots_and_restores():
    shader = MagicMock()
    shader.save.return_value = {"uMVP": ("mat4", "matrix-A")}

    with uniform_block(shader) as snap:
        assert snap == {"uMVP": ("mat4", "matrix-A")}

    shader.save.assert_called_once()
    shader.restore.assert_called_once_with({"uMVP": ("mat4", "matrix-A")})


def test_uniform_block_restores_on_exception():
    shader = MagicMock()
    shader.save.return_value = {"uColor": ("vec3", "c")}

    with pytest.raises(ValueError, match="mid"):
        with uniform_block(shader):
            raise ValueError("mid")

    shader.restore.assert_called_once_with({"uColor": ("vec3", "c")})
