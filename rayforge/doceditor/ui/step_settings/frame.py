from typing import Dict, Any, TYPE_CHECKING, cast
from gi.repository import Gtk, Adw
from .base import StepComponentSettingsWidget
from ....shared.util.adwfix import get_spinrow_float
from ....shared.util.glib import DebounceMixin
from ....pipeline.producer.base import OpsProducer
from ....pipeline.producer.frame import FrameProducer
from ....undo import DictItemCommand

if TYPE_CHECKING:
    from ....core.step import Step
    from ....undo import HistoryManager


class FrameProducerSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
    """UI for configuring the FrameProducer."""

    def __init__(
        self,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        history_manager: "HistoryManager",
        **kwargs,
    ):
        producer = cast(FrameProducer, OpsProducer.from_dict(target_dict))

        super().__init__(
            target_dict=target_dict,
            page=page,
            step=step,
            history_manager=history_manager,
            **kwargs,
        )

        # Offset setting
        offset_adj = Gtk.Adjustment(
            lower=-100.0,
            upper=100.0,
            step_increment=0.1,
            page_increment=1.0,
        )
        offset_row = Adw.SpinRow(
            title=_("Offset"),
            subtitle=_("Distance from content boundary (mm)."),
            adjustment=offset_adj,
            digits=2,
        )
        offset_adj.set_value(producer.offset)
        self.add(offset_row)

        # Connect signals
        offset_row.connect(
            "changed", lambda r: self._debounce(self._on_offset_changed, r)
        )

    def _on_offset_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        # The producer logic lives in the 'params' sub-dictionary
        params_dict = self.target_dict.setdefault("params", {})

        if abs(new_value - params_dict.get("offset", 0.0)) < 1e-6:
            return

        command = DictItemCommand(
            target_dict=params_dict,
            key="offset",
            new_value=new_value,
            name=_("Change Frame Offset"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)
