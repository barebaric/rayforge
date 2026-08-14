from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from .var import Var


class TupleVar(Var[tuple]):
    """
    A Var whose value is a fixed-size tuple (e.g. a min/max range).

    Composite step attributes like ``power_range`` or
    ``grid_dimensions`` are stored as one tuple per step key, so the
    recipe editor keeps a single YAML-safe entry per range. The UI
    renders the components via a dedicated adapter.
    """

    display_name = _("Tuple")

    def __init__(
        self,
        key: str,
        label: str,
        item_labels: tuple[str, str],
        item_subtitles: tuple[str, str] | None = None,
        description: str | None = None,
        default: tuple | None = None,
        value: tuple | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
        digits: int | None = None,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
        sensitive_when: "Callable[[dict[str, Any]], bool] | None" = None,
    ):
        """
        Args:
            item_labels: The row titles for the tuple components (e.g.
                ("Min", "Max") or ("Columns", "Rows")).
            item_subtitles: Optional subtitles for the component rows.
            min_val/max_val: Bounds applied to every component.
            digits: Number of decimal digits for the component rows.
        """
        self.item_labels = item_labels
        self.item_subtitles = item_subtitles
        self.min_val = min_val
        self.max_val = max_val
        self.digits = digits
        super().__init__(
            key=key,
            label=label,
            var_type=tuple,
            description=description,
            default=default,
            value=value,
            visible_when=visible_when,
            sensitive_when=sensitive_when,
        )

    def to_dict(self, include_value: bool = False) -> dict[str, Any]:
        data = super().to_dict(include_value=include_value)
        data.update(
            {
                "item_labels": list(self.item_labels),
                "item_subtitles": (
                    list(self.item_subtitles)
                    if self.item_subtitles is not None
                    else None
                ),
                "min_val": self.min_val,
                "max_val": self.max_val,
                "digits": self.digits,
            }
        )
        return data
