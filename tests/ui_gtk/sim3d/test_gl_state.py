"""
Tests for the gl_state / render_pass context managers and the Shader
context-manager protocol.

These tests are headless-safe: no GTK or real GL context is required.
All ``OpenGL.GL`` calls are patched via ``unittest.mock``.
"""

from unittest.mock import MagicMock, patch

import pytest
from OpenGL import GL

from rayforge.ui_gtk.sim3d.gl_state import gl_state, render_pass
from rayforge.ui_gtk.sim3d.shader.base import Shader


def test_gl_state_restores_depth_test_on_exit():
    enabled_calls = []
    disabled_calls = []

    def fake_enable(cap):
        enabled_calls.append(cap)

    def fake_disable(cap):
        disabled_calls.append(cap)

    with (
        patch("rayforge.ui_gtk.sim3d.gl_state._is_enabled", return_value=True),
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
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glBindTexture"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glActiveTexture"),
        gl_state(),
    ):
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
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glBindTexture"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glActiveTexture"),
        gl_state(),
    ):
        pass

    assert GL.GL_DEPTH_TEST in disabled


def test_gl_state_restores_on_exception():
    blend_func = MagicMock()
    line_width = MagicMock()
    pixel_store = MagicMock()

    with (
        patch("rayforge.ui_gtk.sim3d.gl_state._is_enabled", return_value=True),
        patch(
            "rayforge.ui_gtk.sim3d.gl_state._get_int",
            side_effect=lambda name: {
                GL.GL_BLEND_SRC_RGB: GL.GL_SRC_ALPHA,
                GL.GL_BLEND_DST_RGB: GL.GL_ONE_MINUS_SRC_ALPHA,
                GL.GL_DEPTH_WRITEMASK: 1,
                GL.GL_DEPTH_FUNC: GL.GL_LEQUAL,
                GL.GL_UNPACK_ALIGNMENT: 1,
                GL.GL_ACTIVE_TEXTURE: GL.GL_TEXTURE0,
                GL.GL_TEXTURE_BINDING_2D: 0,
            }.get(name, 0),
        ),
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
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glBindTexture"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glActiveTexture"),
        pytest.raises(RuntimeError, match="boom"),
        gl_state(),
    ):
        raise RuntimeError("boom")

    line_width.assert_called_with(2.5)
    pixel_store.assert_called_with(GL.GL_UNPACK_ALIGNMENT, 1)
    blend_func.assert_called_with(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)


def test_gl_state_skip_flags_avoid_queries():
    with (
        patch("rayforge.ui_gtk.sim3d.gl_state._is_enabled") as mock_is_enabled,
        patch("rayforge.ui_gtk.sim3d.gl_state._get_float") as mock_get_float,
        patch("rayforge.ui_gtk.sim3d.gl_state._get_int") as mock_get_int,
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glEnable"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDisable"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glBlendFunc"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDepthMask"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDepthFunc"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glLineWidth"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glPixelStorei"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glBindTexture"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glActiveTexture"),
        gl_state(
            save_depth_test=False,
            save_blend=False,
            save_depth_mask=False,
            save_depth_func=False,
            save_line_width=False,
            save_unpack_alignment=False,
            save_texture_bindings=False,
        ),
    ):
        pass

    mock_is_enabled.assert_not_called()
    mock_get_float.assert_not_called()
    mock_get_int.assert_not_called()


def _make_shader():
    shader = object.__new__(Shader)
    shader._uniform_values = {}
    shader._uniform_snapshots = []
    return shader


def test_shader_context_protocol_snapshots_and_restores():
    shader = _make_shader()
    snapshot = {"uMVP": ("mat4", "matrix-A")}
    with (
        patch.object(shader, "save", return_value=snapshot) as mock_save,
        patch.object(shader, "restore") as mock_restore,
        shader,
    ):
        pass

    mock_save.assert_called_once()
    mock_restore.assert_called_once_with(snapshot)


def test_shader_context_protocol_restores_on_exception():
    shader = _make_shader()
    snapshot = {"uColor": ("vec3", "c")}
    with (
        patch.object(shader, "save", return_value=snapshot),
        patch.object(shader, "restore") as mock_restore,
        pytest.raises(ValueError, match="mid"),
        shader,
    ):
        raise ValueError("mid")

    mock_restore.assert_called_once_with(snapshot)


def test_render_pass_restores_gl_state_and_uniforms():
    shader = _make_shader()
    snapshot = {"uEmissive": ("float", 0.0)}
    with (
        patch.object(shader, "save", return_value=snapshot) as mock_save,
        patch.object(shader, "restore") as mock_restore,
        patch("rayforge.ui_gtk.sim3d.gl_state._is_enabled", return_value=True),
        patch("rayforge.ui_gtk.sim3d.gl_state._get_int", return_value=1),
        patch("rayforge.ui_gtk.sim3d.gl_state._get_float", return_value=1.0),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glEnable"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDisable"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glBlendFunc"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDepthMask"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDepthFunc"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glLineWidth"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glPixelStorei"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glBindTexture"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glActiveTexture"),
        render_pass(shader),
    ):
        pass

    mock_save.assert_called_once()
    mock_restore.assert_called_once_with(snapshot)


def test_render_pass_without_shader_only_restores_gl_state():
    with (
        patch("rayforge.ui_gtk.sim3d.gl_state._is_enabled", return_value=True),
        patch("rayforge.ui_gtk.sim3d.gl_state._get_int", return_value=1),
        patch("rayforge.ui_gtk.sim3d.gl_state._get_float", return_value=1.0),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glEnable"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDisable"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glBlendFunc"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDepthMask"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glDepthFunc"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glLineWidth"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glPixelStorei"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glBindTexture"),
        patch("rayforge.ui_gtk.sim3d.gl_state.GL.glActiveTexture"),
        render_pass(),
    ):
        pass
