from gi.repository import Adw
from .base import StepComponentSettingsWidget


class RasterizerSettingsWidget(StepComponentSettingsWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cross_hatch_row = Adw.SwitchRow(
            title=_("Cross-Hatch"),
            subtitle=_(
                "Perform a second pass at 90 degrees for a denser fill"
            ),
        )
        self.cross_hatch_row.set_active(
            self.target_dict.get("params", {}).get("cross_hatch", False)
        )
        self.cross_hatch_row.connect(
            "notify::active", self._on_cross_hatch_changed
        )
        self.add(self.cross_hatch_row)

    def _on_cross_hatch_changed(self, switch, _):
        if "params" not in self.target_dict:
            self.target_dict["params"] = {}
        self.target_dict["params"]["cross_hatch"] = switch.get_active()
        self.step.updated.send(self.step)
