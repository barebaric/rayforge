from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from .var import Var


class BoolVar(Var[bool]):
    """A variable that represents a boolean value."""

    display_name = _("Boolean (Switch)")

    def __init__(
        self,
        key: str,
        label: str,
        description: str | None = None,
        default: bool | None = None,
        value: bool | None = None,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
        sensitive_when: "Callable[[dict[str, Any]], bool] | None" = None,
    ):
        """
        Initializes a new BoolVar instance.

        Args:
            key: The unique machine-readable identifier.
            label: The human-readable name for the UI.
            description: A longer, human-readable description.
            default: The default value.
            value: The initial value. If provided, it overrides the default.
            visible_when: Optional callable that receives a dict of all
                          current var values in the widget and returns True
                          when this var's row should be visible.
            sensitive_when: Optional callable that receives a dict of all
                            current var values in the widget and returns
                            True when this var's row should be interactive.
        """
        super().__init__(
            key=key,
            label=label,
            var_type=bool,
            description=description,
            default=default,
            value=value,
            visible_when=visible_when,
            sensitive_when=sensitive_when,
        )
