from typing import Any, TYPE_CHECKING
from gi.repository import Adw, Gtk
from .base import StepComponentSettingsWidget
from ....core.undo import DictItemCommand
from ....shared.util.glib import DebounceMixin

if TYPE_CHECKING:
    from ....doceditor.editor import DocEditor


class RasterizerSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        target_dict: Any,
        page: Adw.PreferencesPage,
        step: Any,
        **kwargs,
    ):
        super().__init__(
            editor,
            title,
            target_dict=target_dict,
            page=page,
            step=step,
            **kwargs,
        )

        params = self.target_dict.setdefault("params", {})

        # Threshold slider
        threshold_row = Adw.ActionRow(
            title=_("Threshold"),
            subtitle=_("Pixel brightness value to consider black (0-255)"),
        )
        threshold_adjustment = Gtk.Adjustment(
            lower=0,
            upper=255,
            step_increment=1,
            page_increment=10,
            value=params.get("threshold", 128),
        )
        threshold_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=threshold_adjustment,
            digits=0,
            draw_value=True,
        )
        threshold_scale.set_size_request(200, -1)
        threshold_scale.connect(
            "value-changed",
            lambda scale: self._debounce(
                self._on_param_changed,
                "threshold",
                int(scale.get_value()),
                _("Change Raster Threshold"),
            ),
        )
        threshold_row.add_suffix(threshold_scale)
        self.add(threshold_row)

        self.cross_hatch_row = Adw.SwitchRow(
            title=_("Cross-Hatch"),
            subtitle=_(
                "Perform a second pass at 90 degrees for a denser fill"
            ),
        )
        self.cross_hatch_row.set_active(params.get("cross_hatch", False))
        self.cross_hatch_row.connect(
            "notify::active",
            lambda w, pspec: self._on_param_changed(
                "cross_hatch", w.get_active(), _("Toggle Cross-Hatch")
            ),
        )
        self.add(self.cross_hatch_row)

    def _on_param_changed(self, key: str, new_value: Any, name: str):
        params_dict = self.target_dict.setdefault("params", {})
        if new_value == params_dict.get(key):
            return

        command = DictItemCommand(
            target_dict=params_dict,
            key=key,
            new_value=new_value,
            name=name,
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)
