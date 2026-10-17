"""Terminal commands that error messages hand the user.

Two pieces live here: :class:`CommandRow`, which renders one command
as a single-click copy row, and :func:`split_commands`, which pulls
the command block out of a free-form error message so the prose can
stay where it was while the commands become rows.
"""

import re
from collections.abc import Sequence
from typing import Any

from gi.repository import Adw, GLib, Gtk

from ..icons import get_icon

_COPY_FEEDBACK_MS = 2000

# First word of a line that is meant to be typed: a program name, or
# something that starts a shell expression.
_COMMAND_HEAD = re.compile(r"[a-z0-9_$|#(]")

# Sentence punctuation a line of prose ends on.
_PROSE_ENDING = (".", "?", "!")


def split_commands(message: str) -> tuple[str, list[str]]:
    """Separate a message's prose from its trailing commands.

    Messages put the commands they tell the user to run after a blank
    line. Such a block is only taken when *every* line of it looks like
    something to type into a terminal: a line that starts like prose or
    that ends a sentence leaves the message untouched, because a
    message shown wrongly is worse than one without a copy button.

    Returns ``(prose, commands)``; ``prose`` is the whole message when
    there is nothing to split off.
    """
    head, separator, tail = message.rpartition("\n\n")
    if not separator or not head:
        return message, []
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if not lines:
        return message, []
    for line in lines:
        if line.endswith(_PROSE_ENDING):
            return message, []
        first = line[0]
        if not _COMMAND_HEAD.match(first):
            return message, []
    return head, lines


class CommandRow(Adw.ActionRow):
    """A terminal command that copies itself on a single click.

    The command text is the row title (monospace via Pango markup);
    a copy icon sits in the suffix slot and flips to a check mark
    briefly after copying. Use it wherever an error or a hint hands
    the user something to type into a terminal, so the command is
    never re-typed by hand.
    """

    def __init__(self, command: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._command = command
        self._feedback_source_id: int | None = None

        escaped = GLib.markup_escape_text(command, -1)
        self.set_use_markup(True)
        self.set_title(f'<span font_family="monospace">{escaped}</span>')
        self.set_activatable(True)

        self._icon = get_icon("copy-symbolic")
        self._icon.set_valign(Gtk.Align.CENTER)
        self.add_suffix(self._icon)

        self.connect("activated", self._on_activated)
        self.connect("destroy", self._on_destroy)

    def _on_activated(self, _row: Adw.ActionRow) -> None:
        display = self.get_display()
        if display is None:
            return
        display.get_clipboard().set(self._command)
        self._icon.set_from_icon_name("check-symbolic")
        if self._feedback_source_id is not None:
            GLib.source_remove(self._feedback_source_id)
        self._feedback_source_id = GLib.timeout_add(
            _COPY_FEEDBACK_MS, self._restore_icon
        )

    def _restore_icon(self) -> bool:
        self._feedback_source_id = None
        self._icon.set_from_icon_name("copy-symbolic")
        return GLib.SOURCE_REMOVE

    def _on_destroy(self, _widget: Gtk.Widget) -> None:
        if self._feedback_source_id is not None:
            GLib.source_remove(self._feedback_source_id)
            self._feedback_source_id = None


def set_command_rows(
    group: Adw.PreferencesGroup,
    rows: Sequence[CommandRow],
    commands: Sequence[str],
) -> list[CommandRow]:
    """Replace a group's copy rows with one row per command.

    Returns the new rows so the caller can hold on to them; the old
    ones are removed from ``group`` and disposed of.
    """
    for row in rows:
        group.remove(row)
    new_rows = [CommandRow(command) for command in commands]
    for row in new_rows:
        group.add(row)
    return new_rows


__all__ = ["CommandRow", "set_command_rows", "split_commands"]
