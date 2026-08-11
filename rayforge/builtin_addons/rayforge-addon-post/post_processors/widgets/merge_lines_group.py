from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.post_processor.groups import (
    ExpanderHost,
    TransformerSettingsGroup,
)
from rayforge.ui_gtk.shared.pref_rows import LengthSpinRow

from ..transformers import MergeLinesTransformer

if TYPE_CHECKING:
    from rayforge.core.step import Step


class MergeLinesSettingsGroup(DebounceMixin, TransformerSettingsGroup):
    """UI for configuring the MergeLinesTransformer."""

    def __init__(
        self,
        title: str,
        transformer: MergeLinesTransformer,
        page: ExpanderHost,
        *,
        step: "Step | None" = None,
        **kwargs,
    ):
        super().__init__(title, transformer, page, step=step, **kwargs)

        self.tolerance_row = LengthSpinRow(
            _("Tolerance"),
            _("Maximum distance for lines to be considered overlapping"),
            lower=0.01,
            upper=10.0,
            value_in_base=transformer.tolerance,
        )
        self.tolerance_row.value_changed.connect(
            lambda r: self._debounce(self._on_tolerance_changed, r)
        )
        self.add(self.tolerance_row)

    def _on_tolerance_changed(self, row: LengthSpinRow) -> None:
        new_value = row.get_value_in_base_units()
        self.param_changed.send(
            self,
            key="tolerance",
            value=new_value,
            name=_("Change merge tolerance"),
        )
