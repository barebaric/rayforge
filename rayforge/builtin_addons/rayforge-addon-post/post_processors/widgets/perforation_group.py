from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.post_processor.groups import (
    ExpanderHost,
    TransformerSettingsGroup,
)
from rayforge.ui_gtk.shared.pref_rows import LengthSpinRow

from ..transformers import PerforationTransformer

if TYPE_CHECKING:
    from rayforge.core.step import Step


class PerforationSettingsGroup(DebounceMixin, TransformerSettingsGroup):
    """UI for configuring the PerforationTransformer."""

    def __init__(
        self,
        title: str,
        transformer: PerforationTransformer,
        page: ExpanderHost,
        *,
        step: "Step | None" = None,
        **kwargs,
    ):
        super().__init__(title, transformer, page, step=step, **kwargs)

        self.cut_row = LengthSpinRow(
            _("Cut Length"),
            _("Distance the laser fires before each gap"),
            lower=0.01,
            upper=1000.0,
            value_in_base=transformer.cut_length,
        )
        self.cut_row.value_changed.connect(
            lambda r: self._debounce(self._on_cut_changed, r)
        )
        self.add(self.cut_row)

        self.skip_row = LengthSpinRow(
            _("Skip Length"),
            _("Distance travelled with the laser off between cuts"),
            lower=0.01,
            upper=1000.0,
            value_in_base=transformer.skip_length,
        )
        self.skip_row.value_changed.connect(
            lambda r: self._debounce(self._on_skip_changed, r)
        )
        self.add(self.skip_row)

    def _on_cut_changed(self, row: LengthSpinRow) -> None:
        self.param_changed.send(
            self,
            key="cut_length",
            value=row.get_value_in_base_units(),
            name=_("Change perforation cut length"),
        )

    def _on_skip_changed(self, row: LengthSpinRow) -> None:
        self.param_changed.send(
            self,
            key="skip_length",
            value=row.get_value_in_base_units(),
            name=_("Change perforation skip length"),
        )
