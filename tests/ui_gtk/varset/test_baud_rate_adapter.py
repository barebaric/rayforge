# flake8: noqa: E402
"""UI tests for the BaudRateAdapter (standard rates + custom entry)."""

import os
import sys
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

from rayforge.core.varset import BaudrateVar
from rayforge.ui_gtk.varset.adapter import create_row_for_var
from rayforge.ui_gtk.varset.adapter.combo import (
    CUSTOM_BAUD_ENTRY_LABEL,
    BaudRateAdapter,
)

pytestmark = pytest.mark.ui


def _create(
    var: BaudrateVar,
) -> tuple[Adw.ComboRow, BaudRateAdapter]:
    row, adapter = create_row_for_var(var, "value")
    assert isinstance(row, Adw.ComboRow)
    assert isinstance(adapter, BaudRateAdapter)
    return row, adapter


def _strings(row: Adw.ComboRow) -> list:
    model = row.get_model()
    assert isinstance(model, Gtk.StringList)
    return [model.get_string(i) for i in range(model.get_n_items())]


def test_standard_choices_plus_custom_entry(ui_context_initializer):
    var = BaudrateVar(
        key="baudrate", label="Baud Rate", choices=[9600, 115200]
    )

    row, adapter = _create(var)

    assert _strings(row) == [
        "9600",
        "115200",
        CUSTOM_BAUD_ENTRY_LABEL,
    ]
    assert row.get_selected() == 1
    assert adapter.get_value() == 115200


def test_custom_value_shown_in_numeric_order(ui_context_initializer):
    var = BaudrateVar(key="baudrate", label="Baud Rate", value=250000)

    row, adapter = _create(var)

    strings = _strings(row)
    assert strings[-1] == CUSTOM_BAUD_ENTRY_LABEL
    assert strings.index("230400") < strings.index("250000")
    assert strings.index("250000") < strings.index("460800")
    assert row.get_selected() == strings.index("250000")
    assert adapter.get_value() == 250000


def test_apply_custom_value(ui_context_initializer):
    var = BaudrateVar(key="baudrate", label="Baud Rate")
    row, adapter = _create(var)
    changes = []
    adapter.changed.connect(lambda sender: changes.append(True), weak=False)

    adapter._apply_custom_value(500000)

    strings = _strings(row)
    assert row.get_selected() == strings.index("500000")
    assert adapter.get_value() == 500000
    assert changes


def test_switching_back_to_standard_rate_drops_custom_entry(
    ui_context_initializer,
):
    var = BaudrateVar(key="baudrate", label="Baud Rate")
    row, adapter = _create(var)

    adapter._apply_custom_value(250000)
    adapter.set_value(115200)

    strings = _strings(row)
    assert "250000" not in strings
    assert row.get_selected() == strings.index("115200")
    assert adapter.get_value() == 115200


def test_selecting_custom_entry_opens_prompt(ui_context_initializer):
    var = BaudrateVar(key="baudrate", label="Baud Rate")
    row, adapter = _create(var)

    with patch.object(BaudRateAdapter, "_prompt_custom_value") as prompt:
        row.set_selected(_strings(row).index(CUSTOM_BAUD_ENTRY_LABEL))

    prompt.assert_called_once_with()
    assert adapter.get_value() == 115200


def test_selecting_standard_entry_does_not_prompt(ui_context_initializer):
    var = BaudrateVar(key="baudrate", label="Baud Rate")
    row, adapter = _create(var)

    with patch.object(BaudRateAdapter, "_prompt_custom_value") as prompt:
        row.set_selected(_strings(row).index("9600"))

    prompt.assert_not_called()
    assert adapter.get_value() == 9600
