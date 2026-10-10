from __future__ import annotations

import logging
from collections.abc import Callable
from gettext import gettext as _
from typing import ClassVar

from gi.repository import Adw

from ..shared.pref_rows.length_spin_row import LengthSpinRow

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE_MM = 0.1


class PathToleranceDialog(Adw.AlertDialog):
    """
    Asks for the tolerance of a path clean-up operation and runs it on
    confirmation. The last value used per operation is remembered for
    the rest of the session.
    """

    _last_values: ClassVar[dict[str, float]] = {}

    def __init__(
        self,
        heading: str,
        body: str,
        apply_label: str,
        key: str,
        on_apply: Callable[[float], None],
    ):
        super().__init__()
        self._key = key
        self._on_apply = on_apply

        self.set_heading(heading)
        self.set_body(body)
        self.add_response("cancel", _("Cancel"))
        self.add_response("apply", apply_label)
        self.set_response_appearance("apply", Adw.ResponseAppearance.SUGGESTED)
        self.set_default_response("apply")
        self.set_close_response("cancel")

        self.tolerance_row = LengthSpinRow(
            _("Tolerance"),
            lower=0.001,
            upper=10.0,
            step_increment=0.01,
            digits=3,
            value_in_base=self._last_values.get(key, DEFAULT_TOLERANCE_MM),
        )
        group = Adw.PreferencesGroup()
        group.add(self.tolerance_row)
        self.set_extra_child(group)

        self.connect("response", self._on_response)

    def get_tolerance_mm(self) -> float:
        return self.tolerance_row.get_value_in_base_units()

    def _on_response(self, _dialog, response: str):
        if response != "apply":
            return
        tolerance = self.get_tolerance_mm()
        self._last_values[self._key] = tolerance
        self._on_apply(tolerance)
