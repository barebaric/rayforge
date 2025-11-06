from __future__ import annotations
from typing import Optional
from .var import Var


class BoolVar(Var[bool]):
    """A variable that represents a boolean value."""

    def __init__(
        self,
        key: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[bool] = None,
        value: Optional[bool] = None,
    ):
        """
        Initializes a new BoolVar instance.

        Args:
            key: The unique machine-readable identifier.
            label: The human-readable name for the UI.
            description: A longer, human-readable description.
            default: The default value.
            value: The initial value. If provided, it overrides the default.
        """
        super().__init__(
            key=key,
            label=label,
            var_type=bool,
            description=description,
            default=default,
            value=value,
        )
