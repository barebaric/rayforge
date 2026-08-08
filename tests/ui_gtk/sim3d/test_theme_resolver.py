"""UI tests for the ThemeResolver 3D colour derivation."""

# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rayforge.core.color import ColorSet
from rayforge.ui_gtk.shared.color_lut_provider import ColorLutProvider
from rayforge.ui_gtk.sim3d.theme_resolver import ThemeResolver


def _theme_color_set() -> ColorSet:
    return ColorSet(
        {
            "cut": (1.0, 0.0, 0.0, 1.0),
            "engrave": np.full((256, 4), 0.5, dtype=np.float32),
            "travel": (0.0, 1.0, 0.0, 1.0),
            "zero_power": (0.0, 0.0, 1.0, 1.0),
        }
    )


def _make_service(color_set=None, dirty=False):
    service = MagicMock()
    type(service).color_set = property(lambda self: color_set)
    type(service).dirty = property(lambda self: dirty)
    service.color_lut_provider.return_value = (
        ColorLutProvider(color_set, {}) if color_set is not None else None
    )
    return service


def _make_resolver(service, gl_initialized=True):
    widget = MagicMock()
    widget.get_style_context.return_value = MagicMock()
    scene = MagicMock()
    scene.axis_renderer = MagicMock()
    scene.texture_renderer = MagicMock()
    rendered = []

    resolver = ThemeResolver(
        widget=widget,
        scene=scene,
        get_machine=lambda: MagicMock(heads=[]),
        get_gl_initialized=lambda: gl_initialized,
        request_render=lambda: rendered.append(True),
    )
    return resolver, widget, scene, rendered


@pytest.mark.ui
def test_color_set_delegates_to_service(ui_context_initializer):
    from rayforge.context import get_context

    color_set = _theme_color_set()
    service = _make_service(color_set=color_set)
    get_context()._theme_service = service
    resolver, _, _, _ = _make_resolver(service)
    assert resolver.color_set is color_set


@pytest.mark.ui
def test_theme_is_dirty_delegates_to_service(ui_context_initializer):
    from rayforge.context import get_context

    service = _make_service(dirty=True)
    get_context()._theme_service = service
    resolver, _, _, _ = _make_resolver(service)
    assert resolver.theme_is_dirty is True


@pytest.mark.ui
def test_mark_dirty_delegates_to_service(ui_context_initializer):
    from rayforge.context import get_context

    service = _make_service()
    get_context()._theme_service = service
    resolver, _, _, _ = _make_resolver(service)
    resolver.mark_dirty()
    service.mark_dirty.assert_called_once()


@pytest.mark.ui
def test_on_style_changed_marks_dirty_and_requests_render(
    ui_context_initializer,
):
    from rayforge.context import get_context

    service = _make_service()
    get_context()._theme_service = service
    resolver, _, _, rendered = _make_resolver(service)
    resolver.on_style_changed(None, None)
    service.mark_dirty.assert_called_once()
    assert rendered == [True]


@pytest.mark.ui
def test_update_theme_and_colors_keeps_gl_derivation(
    ui_context_initializer,
):
    from rayforge.context import get_context

    service = _make_service(color_set=_theme_color_set())
    get_context()._theme_service = service
    resolver, widget, scene, _ = _make_resolver(service)

    style = widget.get_style_context()
    style.lookup_color.side_effect = [
        (True, _rgba(0.2, 0.2, 0.2, 1.0)),  # theme_bg_color
        (True, _rgba(0.8, 0.8, 0.8, 1.0)),  # view_fg_color
    ]

    with patch("rayforge.ui_gtk.sim3d.theme_resolver.GL.glClearColor"):
        resolver.update_theme_and_colors()

    service.set_machine.assert_called_once()
    scene.apply_background_colors.assert_called_once()
    scene.apply_axis_colors.assert_called_once()
    scene.update_color_luts.assert_called_once()


@pytest.mark.ui
def test_update_theme_and_colors_skipped_without_color_set(
    ui_context_initializer,
):
    from rayforge.context import get_context

    service = _make_service(color_set=None)
    get_context()._theme_service = service
    resolver, _, scene, _ = _make_resolver(service)

    resolver.update_theme_and_colors()

    scene.apply_background_colors.assert_not_called()


@pytest.mark.ui
def test_update_renderer_color_luts_skipped_when_not_initialized(
    ui_context_initializer,
):
    from rayforge.context import get_context

    service = _make_service()
    get_context()._theme_service = service
    resolver, _, scene, _ = _make_resolver(service, gl_initialized=False)
    resolver.update_renderer_color_luts()
    scene.update_color_luts.assert_not_called()


@pytest.mark.ui
def test_update_renderer_color_luts_fans_out_to_scene(
    ui_context_initializer,
):
    from rayforge.context import get_context

    service = _make_service(color_set=_theme_color_set())
    get_context()._theme_service = service
    resolver, _, scene, _ = _make_resolver(service)
    resolver.update_renderer_color_luts()
    scene.update_color_luts.assert_called_once()


def _rgba(r, g, b, a):
    rgba = MagicMock()
    rgba.red, rgba.green, rgba.blue, rgba.alpha = r, g, b, a
    return rgba
