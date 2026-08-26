from __future__ import annotations

import time
from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.core.undo.command import Command
from rayforge.core.undo.history import COALESCE_THRESHOLD

from ..entities.text_box import TextBoxEntity
from ..types import EntityID

if TYPE_CHECKING:
    from ..sketch import Sketch


class LiveTextEditCommand(Command):
    """
    Session-scoped undo for the text box editor.

    Tracks keystroke-level history while a text edit session is active.
    It is owned by the tool and must never be pushed onto the global
    undo history: the committed model change is recorded by
    ModifyTextPropertyCommand when the session ends.
    """

    def __init__(
        self,
        sketch: Sketch,
        text_entity_id: EntityID,
    ):
        super().__init__(_("Edit Text"))
        self.text_entity_id = text_entity_id
        self._sketch = sketch
        # History stores tuples of (content, cursor_pos, timestamp)
        self.history: list[tuple[str, int, float]] = []
        self.current_index = -1
        # Maintained for attribute compatibility with tests
        self.cursor_pos = 0
        self._last_capture_time: float = 0.0

    def execute(self) -> None:
        entity = self._sketch.registry.get_entity(self.text_entity_id)
        if not isinstance(entity, TextBoxEntity):
            return

        # Initialize history with the earliest known state, but never
        # reset an existing session history: doing so would drop the
        # pre-edit state and turn this command into an undo dead entry.
        if self.history:
            return

        self.history = [(entity.content, 0, time.time())]
        self.current_index = 0
        self._last_capture_time = time.time()

    def should_skip_undo(self) -> bool:
        # This command is session-scoped only and is never a valid
        # global-history entry.
        return True

    def undo(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self._restore_state(self.current_index)
            # Force a break in coalescing so the next type action creates a
            # new entry rather than overwriting the state we just undid to.
            self._last_capture_time = 0.0

    def redo(self) -> None:
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            self._restore_state(self.current_index)
            # Force a break in coalescing on redo as well
            self._last_capture_time = 0.0

    def _restore_state(self, index: int) -> None:
        if 0 <= index < len(self.history):
            content, _, _ = self.history[index]
            entity = self._sketch.registry.get_entity(self.text_entity_id)
            if isinstance(entity, TextBoxEntity):
                entity.content = content

    def capture_state(self, content: str, cursor_pos: int) -> None:
        now = time.time()

        # 1. Handle Branching (The Fix for Duplicates)
        # If we have undid some actions and are now typing, we must discard
        # the old "future".
        if self.current_index < len(self.history) - 1:
            self.history = self.history[: self.current_index + 1]

        time_delta = now - self._last_capture_time

        # 2. Coalescing Logic
        # If typing is fast enough, update the current tip of history
        # in-place - but never the initial (index 0) entry: it is the
        # pre-edit baseline that the first undo must be able to restore.
        if (
            time_delta < COALESCE_THRESHOLD
            and self.current_index > 0
            and self.history
        ):
            self.history[self.current_index] = (content, cursor_pos, now)
        else:
            self.history.append((content, cursor_pos, now))
            self.current_index = len(self.history) - 1

        self._last_capture_time = now

    def update_cursor(self, cursor_pos: int) -> None:
        """Syncs the cursor of the current history tip after a pure cursor
        movement (click or arrow keys). Cursor moves are not undoable
        actions, so they must not create history entries - but the tip
        must know the position the next edit will start from, otherwise
        undoing that edit restores a stale cursor position."""
        if not self.history or not 0 <= self.current_index < len(self.history):
            return
        content, _, timestamp = self.history[self.current_index]
        self.history[self.current_index] = (content, cursor_pos, timestamp)

    def get_current_content(self) -> str:
        if 0 <= self.current_index < len(self.history):
            return self.history[self.current_index][0]
        return ""

    def get_current_cursor_pos(self) -> int:
        if 0 <= self.current_index < len(self.history):
            return self.history[self.current_index][1]
        return 0
