from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from .textareavar import TextAreaVar


class CodeVar(TextAreaVar):
    """
    A Var subclass for multi-line source-code blocks.

    Like :class:`TextAreaVar`, but hints to the UI that the content is
    code: the editor is monospace, does not wrap long lines, and is
    rendered as an always-expanded, taller block with toolbars for
    inserting template placeholders and macro includes.
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
        variable_context_level: str = "job",
        macros_provider: Callable[[], list] | None = None,
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
        #: Context level for the placeholder documentation popover.
        self.variable_context_level = variable_context_level
        #: Returns the macros offered for inclusion, or ``None`` to
        #: show no macros.
        self.macros_provider = macros_provider
