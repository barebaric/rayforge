# flake8: noqa: E402
"""UI tests for the SerialPortAdapter (USB-first ordering, port pinning)."""

import os
import sys
from contextlib import ExitStack
from typing import cast
from unittest.mock import patch

import pytest

if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if not os.environ.get("DISPLAY"):
        pytest.skip(
            "DISPLAY not set on Linux, skipping UI tests.",
            allow_module_level=True,
        )

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gtk

from rayforge.core.varset import SerialPortVar
from rayforge.machine.transport.serial import SerialPortInfo, SerialTransport
from rayforge.ui_gtk.varset.adapter import create_row_for_var
from rayforge.ui_gtk.varset.adapter.combo import SerialPortAdapter

pytestmark = pytest.mark.ui

NULL_LABEL = "None Selected"


def _create_row(var: SerialPortVar) -> tuple[Adw.ComboRow, SerialPortAdapter]:
    row, adapter = create_row_for_var(var, "value")
    assert adapter is not None
    return cast(Adw.ComboRow, row), cast(SerialPortAdapter, adapter)


def _model_strings(row: Adw.ComboRow) -> list:
    model = row.get_model()
    assert isinstance(model, Gtk.StringList)
    return [model.get_string(i) for i in range(model.get_n_items())]


def _make_infos() -> list:
    return [
        SerialPortInfo("/dev/ttyS0", "ttyS0"),
        SerialPortInfo("/dev/ttyS3", "ttyS3"),
        SerialPortInfo("/dev/ttyACM0", "ttyACM0"),
        SerialPortInfo("/dev/ttyUSB0", "CH340"),
    ]


def _scan_patches(infos: list):
    """
    Returns context managers patching the port scan on a POSIX system.
    USB detection behaves differently on non-POSIX platforms, where
    every port is treated as USB.
    """
    return [
        patch.object(SerialTransport, "list_port_info", return_value=infos),
        patch("os.name", "posix"),
    ]


def test_usb_ports_come_first(ui_context_initializer):
    """USB adapters are listed before hardware ports."""
    var = SerialPortVar(key="port", label="Port")
    with ExitStack() as stack:
        for p in _scan_patches(_make_infos()):
            stack.enter_context(p)
        row, _adapter = _create_row(var)
        assert _model_strings(row) == [
            NULL_LABEL,
            "/dev/ttyACM0",
            "/dev/ttyUSB0",
            "/dev/ttyS0",
            "/dev/ttyS3",
        ]


def test_descriptions_collected_for_factory(ui_context_initializer):
    """Known descriptions are stored for the two-line factory."""
    var = SerialPortVar(key="port", label="Port")
    with ExitStack() as stack:
        for p in _scan_patches(_make_infos()):
            stack.enter_context(p)
        _row, adapter = _create_row(var)
        assert adapter._descriptions == {
            "/dev/ttyS0": "ttyS0",
            "/dev/ttyS3": "ttyS3",
            "/dev/ttyACM0": "ttyACM0",
            "/dev/ttyUSB0": "CH340",
        }


def test_device_paths_round_trip(ui_context_initializer):
    """Selecting an entry yields the raw device path and back."""
    var = SerialPortVar(key="port", label="Port")
    with ExitStack() as stack:
        for p in _scan_patches(_make_infos()):
            stack.enter_context(p)
        _row, adapter = _create_row(var)
        adapter.set_value("/dev/ttyUSB0")
        assert adapter.get_value() == "/dev/ttyUSB0"
        adapter.set_value(None)
        assert adapter.get_value() is None


def test_configured_port_pinned_when_not_plugged_in(ui_context_initializer):
    """
    A configured port that is absent from the live scan must not be
    relegated to the end of the list; it stays on top.
    """
    var = SerialPortVar(key="port", label="Port", value="/dev/ttyUSB0")
    with ExitStack() as stack:
        for p in _scan_patches([]):
            stack.enter_context(p)
        row, adapter = _create_row(var)
        strings = _model_strings(row)
        assert strings == [NULL_LABEL, "/dev/ttyUSB0"]
        assert row.get_selected() == 1
        assert adapter.get_value() == "/dev/ttyUSB0"


def test_rescan_preserves_configured_value(ui_context_initializer):
    """
    Re-scanning (dropdown open) keeps the selection when the configured
    port reappears in the live scan.
    """
    var = SerialPortVar(key="port", label="Port", value="/dev/ttyUSB0")
    with ExitStack() as stack:
        for p in _scan_patches([]):
            stack.enter_context(p)
        _row, adapter = _create_row(var)

    with ExitStack() as stack:
        for p in _scan_patches(
            [
                SerialPortInfo("/dev/ttyUSB0", "CH340"),
                SerialPortInfo("/dev/ttyS0", "ttyS0"),
            ]
        ):
            stack.enter_context(p)
        adapter._refresh(adapter.get_value())
        assert adapter.get_value() == "/dev/ttyUSB0"
