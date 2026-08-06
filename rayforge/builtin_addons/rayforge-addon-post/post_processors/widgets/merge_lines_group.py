from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw

from rayforge.core.undo import DictItemCommand
from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.step_settings.groups import (
    TransformerSettingsGroup,
)
from rayforge.ui_gtk.shared.pref_rows.length_spin_row import LengthSpinRow

from ..transformers import MergeLinesTransformer

if TYPE_CHECKING:
    from rayforge.core.step import Step
    from rayforge.doceditor.editor import DocEditor


class MergeLinesSettingsGroup(DebounceMixin, TransformerSettingsGroup):
    """UI for configuring the MergeLinesTransformer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        transformer: MergeLinesTransformer,
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        super().__init__(
            editor,
            title,
            component=transformer,
            page=page,
            step=step,
            description=transformer.description,
            **kwargs,
        )

        self.tolerance_row = LengthSpinRow(
            _("Tolerance"),
            _("Maximum distance for lines to be considered overlapping"),
            lower=0.01,
            upper=10.0,
            min_value_in_base=0.01,
            max_value_in_base=10.0,
            value_in_base=transformer.tolerance,
        )
        self.tolerance_row.value_changed.connect(
            lambda r: self._debounce(self._on_tolerance_changed, r)
        )
        self.add(self.tolerance_row)

    def _on_tolerance_changed(self, row):
        new_value = row.get_value_in_base_units()
        command = DictItemCommand(
            target_dict=self.target_dict,
            key="tolerance",
            new_value=new_value,
            name=_("Change merge tolerance"),
            on_change_callback=self.step.per_step_transformer_changed.send,
        )
        self.history_manager.execute(command)
