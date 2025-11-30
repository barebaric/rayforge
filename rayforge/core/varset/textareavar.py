from typing import Optional
from .var import Var


class TextAreaVar(Var[str]):
    """
    A Var subclass for multi-line string values that hints to the UI
    that it should be represented by a text area (Gtk.TextView) rather than
    a single-line entry.
    """

    def __init__(
        self,
        key: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        value: Optional[str] = None,
    ):
        super().__init__(
            key=key,
            label=label,
            var_type=str,
            description=description,
            default=default,
            value=value,
        )
