# flake8: noqa: E402
"""Tests for loading icons as pixbufs."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import gi

gi.require_version("GdkPixbuf", "2.0")
gi.require_version("Gtk", "4.0")

import pytest
from gi.repository import GdkPixbuf, GLib

from rayforge.ui_gtk.icons import get_icon_pixbuf

pytestmark = pytest.mark.ui

# Exit code the helper process uses when its GTK has no SVG renderer.
_NO_SVG_RENDERER = 77


@pytest.fixture(autouse=True)
def clear_pixbuf_cache():
    get_icon_pixbuf.cache_clear()
    yield
    get_icon_pixbuf.cache_clear()


def test_icon_load_falls_back_to_gtk(monkeypatch, mocker):
    def fail(*args, **kwargs):
        raise GLib.Error("SVG loader module is missing")

    monkeypatch.setattr(GdkPixbuf.Pixbuf, "new_from_file_at_scale", fail)
    render = mocker.patch(
        "rayforge.ui_gtk.icons._load_svg_pixbuf_with_gtk",
        return_value="rendered",
    )

    assert get_icon_pixbuf("sketch-rect-symbolic", 24) == "rendered"
    render.assert_called_once()


def test_gtk_fallback_rasterizes_the_svg():
    """
    The fallback must produce pixels without gdk-pixbuf's SVG loader,
    since that loader is what is missing in the affected builds.

    Runs in a subprocess so the pixi typelibs win over the system ones
    that the UI test task puts first.
    """
    typelib_dir = Path(sys.prefix) / "lib" / "girepository-1.0"
    if not (typelib_dir / "Gtk-4.0.typelib").is_file():
        pytest.skip("no pixi GTK typelib available")

    script = textwrap.dedent(
        """
        import sys

        import gi

        gi.require_version("GdkPixbuf", "2.0")
        gi.require_version("Gtk", "4.0")
        from gi.repository import GdkPixbuf, GLib, Gtk

        if not hasattr(Gtk, "Svg"):
            sys.exit(77)

        from rayforge.ui_gtk.icons import get_icon_pixbuf

        def fail(*args, **kwargs):
            raise GLib.Error("SVG loader module is missing")

        GdkPixbuf.Pixbuf.new_from_file_at_scale = fail
        pixbuf = get_icon_pixbuf("sketch-rect-symbolic", 24)
        if pixbuf is None or pixbuf.get_width() != 24:
            sys.exit(1)
        """
    )
    env = os.environ.copy()
    env["GI_TYPELIB_PATH"] = str(typelib_dir)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[3],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == _NO_SVG_RENDERER:
        pytest.skip("this GTK has no built-in SVG renderer")
    assert result.returncode == 0, result.stderr
