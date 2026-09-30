from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from .textareavar import TextAreaVar


class CodeVar(TextAreaVar):
    """
    A Var subclass for multi-line source-code blocks.

    Like :class:`TextAreaVar`, but hints to the UI that the content is
    code: the editor is monospace, does not wrap long lines, and is
    rendered as an always-expanded, taller block without an expander
    header.
    """

    display_name = _("Code (Multi-Line)")

    def __init__(
        self,
        key: str,
        label: str,
        description: str | None = None,
        default: str | None = None,
        value: str | None = None,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
    ):
        super().__init__(
            key=key,
            label=label,
            description=description,
            default=default,
            value=value,
            visible_when=visible_when,
        )
