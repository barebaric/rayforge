from typing import Any, TYPE_CHECKING
from gi.repository import Adw, Gtk
from ....core.undo import DictItemCommand
from ....shared.util.glib import DebounceMixin
from .base import StepComponentSettingsWidget
from .direction_preview import DirectionPreview

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
        current_direction = params.get("direction_degrees", 0)

        self.direction_preview = DirectionPreview(current_direction)
        preview_row = Adw.ActionRow(title=_("Direction Preview"))
        preview_row.add_suffix(self.direction_preview)
        self.add(preview_row)

        direction_row = Adw.ActionRow(
            title=_("Direction"),
            subtitle=_(
                "Raster direction in degrees (0°=horizontal, 90°=vertical)"
            ),
        )
        direction_adjustment = Gtk.Adjustment(
            lower=0,
            upper=179,
            step_increment=1,
            page_increment=15,
            value=current_direction,
        )
        direction_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=direction_adjustment,
            digits=0,
            draw_value=True,
        )
        direction_scale.set_size_request(200, -1)
        direction_scale.connect(
            "value-changed",
            self._on_direction_changed,
        )
        direction_row.add_suffix(direction_scale)
        self.add(direction_row)

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

        self.invert_row = Adw.SwitchRow(
            title=_("Invert"),
            subtitle=_("Engrave white areas instead of black areas"),
        )
        self.invert_row.set_active(params.get("invert", False))
        self.invert_row.connect(
            "notify::active",
            lambda w, pspec: self._on_param_changed(
                "invert", w.get_active(), _("Toggle Invert")
            ),
        )
        self.add(self.invert_row)

        line_interval_adj = Gtk.Adjustment(
            lower=0.01,
            upper=10.0,
            step_increment=0.01,
            value=params.get("line_interval_mm") or 0.1,
        )
        self.line_interval_row = Adw.SpinRow(
            title=_("Line Interval"),
            subtitle=_(
                "Distance between scan lines in mm. Leave at 0 to use laser "
                "spot size."
            ),
            adjustment=line_interval_adj,
            digits=2,
        )
        self.line_interval_row.connect(
            "changed",
            lambda r: self._debounce(
                self._on_line_interval_changed,
                r,
            ),
        )
        self.add(self.line_interval_row)

    def _on_line_interval_changed(self, spinrow):
        value = spinrow.get_value()
        if value <= 0:
            value = None
        self._on_param_changed(
            "line_interval_mm", value, _("Change Line Interval")
        )

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

    def _on_direction_changed(self, scale):
        value = float(scale.get_value())
        self.direction_preview.update(value)
        self._debounce(
            self._on_param_changed,
            "direction_degrees",
            value,
            _("Change Raster Direction"),
        )
