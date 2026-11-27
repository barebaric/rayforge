# flake8: noqa: E402
"""UI tests for the BluetoothAddressAdapter (paired-device picker)."""

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
from raygeo.driver.core.bluetooth import BluetoothDeviceInfo

from rayforge.core.varset import BluetoothAddressVar
from rayforge.ui_gtk.varset.adapter import create_row_for_var
from rayforge.ui_gtk.varset.adapter.bluetooth import (
    MANUAL_ADDRESS_LABEL,
    BluetoothAddressAdapter,
)

pytestmark = pytest.mark.ui

NULL_LABEL = "None Selected"


def _create_row(
    var: BluetoothAddressVar,
) -> tuple[Adw.ComboRow, BluetoothAddressAdapter]:
    row, adapter = create_row_for_var(var, "value")
    assert adapter is not None
    return cast(Adw.ComboRow, row), cast(BluetoothAddressAdapter, adapter)


def _model_strings(row: Adw.ComboRow) -> list:
    model = row.get_model()
    assert isinstance(model, Gtk.StringList)
    return [model.get_string(i) for i in range(model.get_n_items())]


def _devices() -> list[BluetoothDeviceInfo]:
    return [
        BluetoothDeviceInfo("AA:BB:CC:DD:EE:01", "Laser B"),
        BluetoothDeviceInfo("AA:BB:CC:DD:EE:02", "Laser A"),
    ]


def _scan_patches(devices: list):
    return [
        patch(
            "rayforge.ui_gtk.varset.adapter.bluetooth.list_paired_devices",
            return_value=devices,
        )
    ]


def test_lists_paired_devices(ui_context_initializer):
    var = BluetoothAddressVar(key="address", label="Bluetooth device")
    with ExitStack() as stack:
        for p in _scan_patches(_devices()):
            stack.enter_context(p)
        row, adapter = _create_row(var)
        assert _model_strings(row) == [
            NULL_LABEL,
            "AA:BB:CC:DD:EE:02",
            "AA:BB:CC:DD:EE:01",
            MANUAL_ADDRESS_LABEL,
        ]
        assert adapter._descriptions == {
            "AA:BB:CC:DD:EE:02": "Laser A",
            "AA:BB:CC:DD:EE:01": "Laser B",
        }


def test_value_round_trip(ui_context_initializer):
    var = BluetoothAddressVar(key="address", label="Bluetooth device")
    with ExitStack() as stack:
        for p in _scan_patches(_devices()):
            stack.enter_context(p)
        _row, adapter = _create_row(var)
        adapter.set_value("aa-bb-cc-dd-ee-01")
        assert adapter.get_value() == "AA:BB:CC:DD:EE:01"
        adapter.set_value(None)
        assert adapter.get_value() is None


def test_configured_address_pinned_when_not_listed(
    ui_context_initializer,
):
    var = BluetoothAddressVar(
        key="address",
        label="Bluetooth device",
        value="AA:BB:CC:DD:EE:FF",
    )
    with ExitStack() as stack:
        for p in _scan_patches([]):
            stack.enter_context(p)
        row, adapter = _create_row(var)
        assert _model_strings(row) == [
            NULL_LABEL,
            "AA:BB:CC:DD:EE:FF",
            MANUAL_ADDRESS_LABEL,
        ]
        assert row.get_selected() == 1
        assert adapter.get_value() == "AA:BB:CC:DD:EE:FF"
