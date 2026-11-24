import sys

import pytest

from rayforge.shared.util import platform


@pytest.fixture(autouse=True)
def no_bundle(monkeypatch):
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)


class TestIsWindows:
    def test_windows(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert platform.is_windows()

    def test_linux(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert not platform.is_windows()


class TestIsFrozen:
    def test_bundle(self, monkeypatch):
        monkeypatch.setattr(sys, "_MEIPASS", "/tmp/rayforge", raising=False)
        assert platform.is_frozen()

    def test_source_checkout(self):
        assert not platform.is_frozen()


class TestSupportsSelfUpdate:
    def test_windows_bundle(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(sys, "_MEIPASS", "/tmp/rayforge", raising=False)
        assert platform.supports_self_update()

    def test_windows_source_checkout(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert not platform.supports_self_update()

    def test_linux_bundle(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(sys, "_MEIPASS", "/tmp/rayforge", raising=False)
        assert not platform.supports_self_update()
