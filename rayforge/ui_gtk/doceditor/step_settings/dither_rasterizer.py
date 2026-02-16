from typing import Any, TYPE_CHECKING
from gi.repository import Adw, Gtk
from .base import StepComponentSettingsWidget
from ....core.undo import DictItemCommand
from ....shared.util.glib import DebounceMixin

if TYPE_CHECKING:
    from ....doceditor.editor import DocEditor


DITHER_ALGORITHMS = [
    ("floyd_steinberg", "Floyd-Steinberg"),
    ("bayer2", "Bayer 2x2"),
    ("bayer4", "Bayer 4x4"),
    ("bayer8", "Bayer 8x8"),
]

DITHER_ALGORITHM_KEYS = [key for key, _label in DITHER_ALGORITHMS]


class DitherRasterizerSettingsWidget(
    DebounceMixin, StepComponentSettingsWidget
):
    show_general_settings = True

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

        algorithm_row = Adw.ComboRow(
            title=_("Dither Algorithm"),
            subtitle=_("Algorithm to use for dithering"),
        )
        string_list = Gtk.StringList.new(
            [label for _key, label in DITHER_ALGORITHMS]
        )
        algorithm_row.set_model(string_list)

        current_algorithm = params.get("dither_algorithm", "floyd_steinberg")
        for i, (key, _label) in enumerate(DITHER_ALGORITHMS):
            if key == current_algorithm:
                algorithm_row.set_selected(i)
                break

        algorithm_row.connect(
            "notify::selected",
            self._on_algorithm_changed,
        )
        self.add(algorithm_row)

        self.invert_row = Adw.SwitchRow(
            title=_("Invert"),
            subtitle=(_("Engrave white areas instead of black areas")),
        )
        self.invert_row.set_active(params.get("invert", False))
        self.invert_row.connect(
            "notify::active",
            lambda w, pspec: self._on_param_changed(
                "invert", w.get_active(), _("Toggle Invert")
            ),
        )
        self.add(self.invert_row)

        self.bidirectional_row = Adw.SwitchRow(
            title=_("Bidirectional"),
            subtitle=(_("Scan alternate rows in opposite directions")),
        )
        self.bidirectional_row.set_active(params.get("bidirectional", True))
        self.bidirectional_row.connect(
            "notify::active",
            lambda w, pspec: self._on_param_changed(
                "bidirectional", w.get_active(), _("Toggle Bidirectional")
            ),
        )
        self.add(self.bidirectional_row)

    def _on_algorithm_changed(self, row, pspec):
        selected_index = row.get_selected()
        algorithm_key = DITHER_ALGORITHMS[selected_index][0]
        self._on_param_changed(
            "dither_algorithm",
            algorithm_key,
            _("Change Dither Algorithm"),
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
