import importlib
import os
import sys

from rayforge import app


def test_app_uses_simple_gtk_input_module_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("GTK_IM_MODULE", raising=False)

    importlib.reload(app)

    assert os.environ["GTK_IM_MODULE"] == "simple"


def test_app_preserves_windows_gtk_input_module_override(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("GTK_IM_MODULE", "custom")

    importlib.reload(app)

    assert os.environ["GTK_IM_MODULE"] == "custom"


def test_app_does_not_set_gtk_input_module_on_other_platforms(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("GTK_IM_MODULE", raising=False)

    importlib.reload(app)

    assert "GTK_IM_MODULE" not in os.environ


def test_app_uses_cairo_gsk_renderer_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("GSK_RENDERER", raising=False)

    importlib.reload(app)

    assert os.environ["GSK_RENDERER"] == "cairo"


def test_app_preserves_windows_gsk_renderer_override(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("GSK_RENDERER", "gl")

    importlib.reload(app)

    assert os.environ["GSK_RENDERER"] == "gl"


def test_app_does_not_set_gsk_renderer_on_other_platforms(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("GSK_RENDERER", raising=False)

    importlib.reload(app)

    assert "GSK_RENDERER" not in os.environ
