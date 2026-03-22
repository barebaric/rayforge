from __future__ import annotations

import locale
from dataclasses import dataclass, field
from gettext import gettext as _
from typing import Callable, List, Optional, Tuple, Union, cast


@dataclass
class DimensionInputHandler:
    """
    Handles locale-aware numeric input for sketch tool dimension editing.

    Supports multiple fields with Tab navigation between them.
    Respects the system locale for decimal separator (comma vs dot).
    """

    field_count: int = 1
    field_labels: Optional[List[str]] = None
    decimal_sep: str = field(default=".", init=False)
    buffers: List[str] = field(default_factory=list, init=False)
    current_field: int = field(default=0, init=False)
    _is_active: bool = field(default=False, init=False)

    def __post_init__(self):
        self._detect_decimal_separator()
        self._init_buffers()

    def _detect_decimal_separator(self):
        try:
            self.decimal_sep = locale.localeconv().get("decimal_point", ".")
        except (locale.Error, AttributeError):
            self.decimal_sep = "."

    def _init_buffers(self):
        self.buffers = [""] * self.field_count
        self.current_field = 0

    def is_active(self) -> bool:
        return self._is_active

    def start(self) -> None:
        self._init_buffers()
        self._is_active = True

    def cancel(self) -> None:
        self._init_buffers()
        self._is_active = False

    def get_display_text(
        self, field_index: Optional[int] = None
    ) -> Optional[str]:
        """
        Returns the current buffer text for display.

        Args:
            field_index: If provided, returns text for that specific field.
                         Returns buffer content if any, or None if empty.

        Returns:
            The buffer text (or None if empty), or None if field_index is
            out of range.
        """
        if field_index is not None:
            if field_index < 0 or field_index >= len(self.buffers):
                return None
            buf = self.buffers[field_index]
            return buf if buf else None

        if self.field_count == 1:
            return self.buffers[0]

        parts = []
        for i, buf in enumerate(self.buffers):
            label = (
                self.field_labels[i]
                if self.field_labels and i < len(self.field_labels)
                else str(i + 1)
            )
            if i == self.current_field:
                parts.append(f"[{label}: {buf or '_'}]")
            else:
                parts.append(f"{label}: {buf or '_'}")
        return " | ".join(parts)

    def handle_text_input(self, text: str) -> bool:
        if not self._is_active:
            return False

        for char in text:
            if self._is_valid_char(char):
                self.buffers[self.current_field] += char

        return True

    def _is_valid_char(self, char: str) -> bool:
        if char.isdigit():
            return True
        if char == self.decimal_sep or char == "." or char == ",":
            buf = self.buffers[self.current_field]
            if self.decimal_sep not in buf:
                dot_count = buf.count(".")
                comma_count = buf.count(",")
                if dot_count + comma_count == 0:
                    return True
            return False
        if char == " ":
            if (
                self.field_count > 1
                and self.current_field < self.field_count - 1
            ):
                self.current_field += 1
                return True
        return False

    def handle_backspace(self) -> bool:
        if not self._is_active:
            return False

        if self.buffers[self.current_field]:
            self.buffers[self.current_field] = self.buffers[
                self.current_field
            ][:-1]
            return True

        if self.field_count > 1 and self.current_field > 0:
            self.current_field -= 1
            return True

        return False

    def handle_delete(self) -> bool:
        return self.handle_backspace()

    def handle_tab(
        self, shift: bool = False
    ) -> Tuple[bool, bool, Optional[int]]:
        """
        Handle Tab key press.

        Returns:
            Tuple of (handled, should_apply, committed_field):
            - handled: True if the key was consumed
            - should_apply: True if the input should be committed
              (Tab on last field or single-field mode)
            - committed_field: The field index that was tabbed out of
              (for applying immediate constraints), or None
        """
        if not self._is_active:
            return (False, False, None)

        if self.field_count <= 1:
            return (True, True, 0)

        if shift:
            if self.current_field > 0:
                committed_field = self.current_field
                self.current_field -= 1
                return (True, False, committed_field)
        else:
            if self.current_field < self.field_count - 1:
                committed_field = self.current_field
                self.current_field += 1
                return (True, False, committed_field)
            else:
                return (True, True, self.current_field)

        return (False, False, None)

    def get_field_value(self, field_index: int) -> Optional[float]:
        """
        Parse and return the value for a specific field.

        Returns None if the field is empty or invalid.
        """
        if field_index < 0 or field_index >= len(self.buffers):
            return None
        buf = self.buffers[field_index]
        if not buf.strip():
            return None
        try:
            normalized = buf.strip().replace(",", ".")
            value = float(normalized)
            if value < 0:
                return None
            return value
        except ValueError:
            return None

    def parse_values(self) -> Optional[List[float]]:
        values = []

        for buf in self.buffers:
            if not buf.strip():
                values.append(None)
                continue
            try:
                normalized = buf.strip().replace(",", ".")
                value = float(normalized)
                if value < 0:
                    return None
                values.append(value)
            except ValueError:
                return None

        if all(v is None for v in values):
            return None

        return values

    def commit(self) -> Optional[Tuple[Optional[float], ...]]:
        values = self.parse_values()
        self._is_active = False
        self._init_buffers()
        if values is None:
            return None
        return tuple(values)

    def get_active_shortcuts(
        self,
    ) -> List[Tuple[Union[str, List[str]], str, Optional[Callable[[], bool]]]]:
        if not self._is_active:
            return []

        shortcuts = [
            ("Enter", _("Apply"), None),
            ("Esc", _("Cancel"), None),
            ("Del", _("Delete"), None),
        ]

        if self.field_count > 1:
            shortcuts.append(("Tab", _("Next field"), None))
            shortcuts.append(("⇧Tab", _("Prev field"), None))

        return cast(
            List[
                Tuple[Union[str, List[str]], str, Optional[Callable[[], bool]]]
            ],
            shortcuts,
        )
