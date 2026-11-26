import ctypes
import sys

import pytest

from rayforge.shared.util import platform


class _FakeKernel32:
    def __init__(self, result):
        self.result = result

    def GetCurrentPackageFullName(self, length, name):
        return self.result


class _FakeWindll:
    def __init__(self, result):
        self.kernel32 = _FakeKernel32(result)


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


class TestIsMsix:
    def test_not_windows(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert not platform.is_msix()

    def test_unpackaged(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            ctypes,
            "windll",
            _FakeWindll(platform._APPMODEL_ERROR_NO_PACKAGE),
            raising=False,
        )
        assert not platform.is_msix()

    def test_packaged(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(ctypes, "windll", _FakeWindll(122), raising=False)
        assert platform.is_msix()

    def test_api_unavailable(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.delattr(ctypes, "windll", raising=False)
        assert not platform.is_msix()


class TestSupportsSelfUpdate:
    def test_windows_bundle(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(sys, "_MEIPASS", "/tmp/rayforge", raising=False)
        monkeypatch.setattr(platform, "is_msix", lambda: False)
        assert platform.supports_self_update()

    def test_windows_bundle_msix(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(sys, "_MEIPASS", "/tmp/rayforge", raising=False)
        monkeypatch.setattr(platform, "is_msix", lambda: True)
        assert not platform.supports_self_update()

    def test_windows_source_checkout(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert not platform.supports_self_update()

    def test_linux_bundle(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(sys, "_MEIPASS", "/tmp/rayforge", raising=False)
        assert not platform.supports_self_update()
