"""
Validation helpers for user-editable G-code fields in dialogs.
"""

import re
from gettext import gettext as _

_NUMBER_ONLY_PATTERN = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)$")


def is_number_only(text: str) -> bool:
    """
    Returns True if the text consists solely of a numeric literal,
    e.g. "6000", "-296.429" or ".5".
    """
    return bool(_NUMBER_ONLY_PATTERN.match(text.strip()))


def find_number_only_line(text: str) -> tuple[int, str] | None:
    """
    Finds the first line in *text* that consists solely of a numeric
    literal.

    Returns a (line number, stripped line) tuple with a 1-based line
    number, or None if no such line exists.
    """
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if is_number_only(stripped):
            return lineno, stripped
    return None


def format_number_only_warning(value: str, lineno: int | None = None) -> str:
    """
    Builds the user-facing warning for a G-code field (or line) that
    contains only a number and would be sent to the machine as-is.
    """
    if lineno is None:
        return _(
            '"{value}" is only a number and would be sent to the '
            "machine as-is. A G-code command must start with a "
            'letter (e.g. "M4 S{value}").'
        ).format(value=value)
    return _(
        'Line {lineno} is only the number "{value}" and would be '
        "sent to the machine as-is. A G-code command must start "
        'with a letter (e.g. "M4 S{value}").'
    ).format(lineno=lineno, value=value)
