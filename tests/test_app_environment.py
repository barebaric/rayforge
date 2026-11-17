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
