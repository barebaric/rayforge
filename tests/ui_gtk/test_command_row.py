"""Tests for rendering the terminal commands inside error messages."""

import pytest
from gi.repository import Adw

from rayforge.ui_gtk.shared.command_row import (
    CommandRow,
    set_command_rows,
    split_commands,
)

pytestmark = pytest.mark.ui

# The message the serial transport raises when a Snap cannot reach the
# ports, exactly as written there: prose, a blank line, then the
# commands.
SNAP_ERROR = (
    "Serial ports found, but none are accessible. Please ensure your "
    "Snap has the 'serial-port' interface connected by running:\n"
    "\n"
    "sudo snap set system experimental.hotplug=true\n"
    "sudo snap connect rayforge:serial-port"
)


def test_split_commands_takes_the_trailing_block():
    prose, commands = split_commands(SNAP_ERROR)

    assert prose == (
        "Serial ports found, but none are accessible. Please ensure "
        "your Snap has the 'serial-port' interface connected by running:"
    )
    assert commands == [
        "sudo snap set system experimental.hotplug=true",
        "sudo snap connect rayforge:serial-port",
    ]


def test_a_message_without_a_blank_line_is_left_alone():
    prose, commands = split_commands("No USB serial ports found.")

    assert prose == "No USB serial ports found."
    assert commands == []


def test_a_sentence_after_a_blank_line_stays_in_the_message():
    message = (
        "Could not access any serial ports. On Linux, ensure your "
        "user is in the 'dialout' group.\n"
        "\n"
        "Then log out and back in."
    )

    prose, commands = split_commands(message)

    assert prose == message
    assert commands == []


def test_a_capitalised_line_is_left_alone():
    message = "Something went wrong.\n\nThen restart Rayforge."

    prose, commands = split_commands(message)

    assert prose == message
    assert commands == []


def test_command_row_keeps_the_exact_text(ui_context_initializer):
    row = CommandRow("sudo snap connect rayforge:serial-port")

    assert row._command == "sudo snap connect rayforge:serial-port"
    assert "monospace" in row.get_title()


def test_set_command_rows_replaces_the_previous_rows(ui_context_initializer):
    group = Adw.PreferencesGroup()
    rows = set_command_rows(group, [], ["sudo a", "sudo b"])
    assert [r._command for r in rows] == ["sudo a", "sudo b"]

    previous = rows[0]
    rows = set_command_rows(group, rows, ["sudo c"])

    assert previous.get_parent() is None
    assert [r._command for r in rows] == ["sudo c"]

    rows = set_command_rows(group, rows, [])
    assert rows == []
