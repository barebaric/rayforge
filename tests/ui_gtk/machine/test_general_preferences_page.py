"""Tests for the error display on the machine settings General page."""

import pytest

from rayforge.machine.models.machine import Machine
from rayforge.ui_gtk.machine.general_preferences_page import (
    GeneralPreferencesPage,
)

pytestmark = pytest.mark.ui

SNAP_ERROR = (
    "Serial ports found, but none are accessible. Please ensure your "
    "Snap has the 'serial-port' interface connected by running:\n"
    "\n"
    "sudo snap set system experimental.hotplug=true\n"
    "sudo snap connect rayforge:serial-port"
)


def _make_page(context):
    """A registered machine with its General settings page attached."""
    machine = Machine(context)
    context.machine_mgr.add_machine(machine)
    return machine, GeneralPreferencesPage(machine)


def test_precheck_error_commands_become_copy_rows(ui_context_initializer):
    machine, page = _make_page(ui_context_initializer)

    machine.set_precheck_error(SNAP_ERROR)
    machine.changed.send(machine)

    banner = page.error_banner.get_title()
    assert banner.endswith("connected by running:")
    assert "sudo snap" not in banner
    assert [r._command for r in page._command_rows] == [
        "sudo snap set system experimental.hotplug=true",
        "sudo snap connect rayforge:serial-port",
    ]


def test_a_clear_precheck_takes_the_rows_with_it(ui_context_initializer):
    machine, page = _make_page(ui_context_initializer)

    machine.set_precheck_error(SNAP_ERROR)
    machine.changed.send(machine)
    assert page._command_rows

    machine.set_precheck_error(None)
    machine.changed.send(machine)

    assert page._command_rows == []
    assert not page.error_banner.get_revealed()


def test_an_error_without_commands_shows_only_the_banner(
    ui_context_initializer,
):
    machine, page = _make_page(ui_context_initializer)

    machine.set_precheck_error("No USB serial ports found.")
    machine.changed.send(machine)

    assert page._command_rows == []
    assert "No USB serial ports found." in page.error_banner.get_title()
