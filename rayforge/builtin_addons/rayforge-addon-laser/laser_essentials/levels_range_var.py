"""The raster brightness-range (black point) VarSet variable."""

from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from rayforge.core.varset import IntVar


class LevelsRangeVar(IntVar):
    """An IntVar for the raster ``black_point`` setting (0-255).

    Hints the UI to render the histogram (brightness range) row whose
    :class:`HistogramPreview` drags the black/white points. The
    adapter manages the related ``white_point`` key through the same
    row via :meth:`~RowAdapter.related_keys`.
    """

    def __init__(
        self,
        key: str = "black_point",
        label: str = _("Brightness Range"),
        description: str | None = None,
        default: int = 0,
        value: int | None = None,
        min_val: int = 0,
        max_val: int = 255,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
    ):
        super().__init__(
            key=key,
            label=label,
            description=description,
            default=default,
            value=value,
            min_val=min_val,
            max_val=max_val,
            visible_when=visible_when,
        )
