from __future__ import annotations
import time
from typing import TYPE_CHECKING, List, Tuple
from gettext import gettext as _
from ...undo.command import Command
from ...undo.history import COALESCE_THRESHOLD
from ..entities.text_box import TextBoxEntity

if TYPE_CHECKING:
    from ..sketch import Sketch


class LiveTextEditCommand(Command):
    def __init__(
        self,
        sketch: Sketch,
        text_entity_id: int,
    ):
        super().__init__(_("Edit Text"))
        self.text_entity_id = text_entity_id
        self._sketch = sketch
        # History stores tuples of (content, cursor_pos, timestamp)
        self.history: List[Tuple[str, int, float]] = []
        self.current_index = -1
        # Maintained for attribute compatibility with tests
        self.cursor_pos = 0
        self._last_capture_time: float = 0.0

    def execute(self) -> None:
        entity = self._sketch.registry.get_entity(self.text_entity_id)
        if not isinstance(entity, TextBoxEntity):
            return

        # Initialize history with the current (initial) state
        self.history = [(entity.content, 0, time.time())]
        self.current_index = 0
        self._last_capture_time = time.time()

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
        # If typing is fast enough, update the current tip of history in-place.
        # This includes index 0 if the user starts typing immediately after
        # execute.
        if time_delta < COALESCE_THRESHOLD and self.history:
            self.history[self.current_index] = (content, cursor_pos, now)
        else:
            self.history.append((content, cursor_pos, now))
            self.current_index = len(self.history) - 1

        self._last_capture_time = now

    def get_current_content(self) -> str:
        if 0 <= self.current_index < len(self.history):
            return self.history[self.current_index][0]
        return ""

    def get_current_cursor_pos(self) -> int:
        if 0 <= self.current_index < len(self.history):
            return self.history[self.current_index][1]
        return 0
