import os
import sys

from rayforge import gtk_environment


def test_configure_gtk_environment_uses_simple_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("GTK_IM_MODULE", raising=False)

    gtk_environment.configure_gtk_environment()

    assert os.environ["GTK_IM_MODULE"] == "simple"


def test_configure_gtk_environment_preserves_windows_override(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("GTK_IM_MODULE", "custom")

    gtk_environment.configure_gtk_environment()

    assert os.environ["GTK_IM_MODULE"] == "custom"


def test_configure_gtk_environment_does_not_change_other_platforms(
    monkeypatch,
):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("GTK_IM_MODULE", raising=False)

    gtk_environment.configure_gtk_environment()

    assert "GTK_IM_MODULE" not in os.environ
