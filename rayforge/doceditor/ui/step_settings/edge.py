from typing import Dict, Any, TYPE_CHECKING, cast
from gi.repository import Adw
from .base import StepComponentSettingsWidget
from ....pipeline.producer.base import OpsProducer
from ....pipeline.producer.edge import EdgeTracer
from ....undo import DictItemCommand

if TYPE_CHECKING:
    from ....core.step import Step
    from ....undo import HistoryManager


class EdgeTracerSettingsWidget(StepComponentSettingsWidget):
    """UI for configuring the EdgeTracer."""

    def __init__(
        self,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        history_manager: "HistoryManager",
        **kwargs,
    ):
        producer = cast(EdgeTracer, OpsProducer.from_dict(target_dict))

        super().__init__(
            target_dict=target_dict,
            page=page,
            step=step,
            history_manager=history_manager,
            **kwargs,
        )

        # Toggle switch for removing inner paths
        switch_row = Adw.SwitchRow(
            title=_("Remove Inner Paths"),
            subtitle=_("If enabled, only trace the outer outline of shapes."),
        )
        switch_row.set_active(producer.remove_inner_paths)
        self.add(switch_row)

        # Connect signals
        switch_row.connect("notify::active", self._on_toggle)

    def _on_toggle(self, row, pspec):
        new_value = row.get_active()
        # The producer logic lives in the 'params' sub-dictionary
        params_dict = self.target_dict.setdefault("params", {})

        if new_value == params_dict.get("remove_inner_paths"):
            return

        command = DictItemCommand(
            target_dict=params_dict,
            key="remove_inner_paths",
            new_value=new_value,
            name=_("Toggle Remove Inner Paths"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)
