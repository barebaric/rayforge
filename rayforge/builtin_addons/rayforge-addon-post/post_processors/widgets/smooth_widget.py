from typing import TYPE_CHECKING

from gi.repository import Adw, Gtk
from gettext import gettext as _

from rayforge.core.undo import DictItemCommand
from ..transformers import Smooth
from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.step_settings.base import (
    StepComponentSettingsWidget,
)
from rayforge.ui_gtk.shared.adwfix import get_spinrow_int
from rayforge.ui_gtk.shared.slider import create_slider_row

if TYPE_CHECKING:
    from rayforge.core.step import Step
    from rayforge.doceditor.editor import DocEditor


class SmoothSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
    """UI for configuring the Smooth transformer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        transformer: Smooth,
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

        amount_adj = Gtk.Adjustment(
            lower=0, upper=100, step_increment=1, page_increment=10
        )
        amount_adj.set_value(transformer.amount)
        amount_row, amount_scale = create_slider_row(
            title=_("Smoothness"),
            subtitle=_("Higher values produce smoother curves"),
            adjustment=amount_adj,
            digits=0,
            on_value_changed=lambda s: self._debounce(
                self._on_amount_changed, s
            ),
        )
        self.add(amount_row)

        # Corner Angle Threshold Setting
        corner_adj = Gtk.Adjustment(
            lower=0, upper=179, step_increment=1, page_increment=10
        )
        corner_row = Adw.SpinRow(
            title=_("Corner Angle Threshold"),
            subtitle=_(
                "Angles sharper than this are kept as corners (degrees)"
            ),
            adjustment=corner_adj,
        )
        corner_adj.set_value(transformer.corner_angle_threshold)
        self.add(corner_row)

        corner_row.connect(
            "changed",
            lambda spin_row: self._debounce(
                self._on_corner_angle_changed, spin_row
            ),
        )

    def _on_amount_changed(self, scale):
        new_value = int(scale.get_value())
        command = DictItemCommand(
            target_dict=self.target_dict,
            key="amount",
            new_value=new_value,
            name=_("Change smoothness"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)

    def _on_corner_angle_changed(self, spin_row):
        new_value = get_spinrow_int(spin_row)
        command = DictItemCommand(
            target_dict=self.target_dict,
            key="corner_angle_threshold",
            new_value=new_value,
            name=_("Change corner angle"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)
