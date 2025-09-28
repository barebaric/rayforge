from typing import Dict, Any, TYPE_CHECKING, cast
from gi.repository import Gtk, Adw
from .base import StepComponentSettingsWidget
from ....shared.util.glib import DebounceMixin
from ....pipeline.producer.base import OpsProducer
from ....pipeline.producer.shrinkwrap import ShrinkWrapProducer
from ....undo import DictItemCommand

if TYPE_CHECKING:
    from ....core.step import Step
    from ....undo import HistoryManager


class ShrinkWrapProducerSettingsWidget(
    DebounceMixin, StepComponentSettingsWidget
):
    """UI for configuring the HullProducer."""

    def __init__(
        self,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        history_manager: "HistoryManager",
        **kwargs,
    ):
        producer = cast(ShrinkWrapProducer, OpsProducer.from_dict(target_dict))

        super().__init__(
            target_dict=target_dict,
            page=page,
            step=step,
            history_manager=history_manager,
            **kwargs,
        )

        # Gravity setting (Slider)
        gravity_row = Adw.ActionRow(
            title=_("Gravity"),
            subtitle=_("Pulls the hull inward. 0.0 is a standard convex hull"),
        )
        gravity_adj = Gtk.Adjustment(
            lower=0.0, upper=1.0, step_increment=0.01, page_increment=0.1
        )
        gravity_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=gravity_adj,
            digits=2,
            draw_value=True,
        )
        gravity_adj.set_value(producer.gravity)
        gravity_scale.set_size_request(200, -1)
        gravity_row.add_suffix(gravity_scale)
        self.add(gravity_row)

        # Connect signals
        gravity_scale.connect(
            "value-changed",
            lambda scale: self._debounce(self._on_gravity_changed, scale),
        )

    def _on_gravity_changed(self, scale):
        new_value = scale.get_value()
        # The producer logic lives in the 'params' sub-dictionary
        params_dict = self.target_dict.setdefault("params", {})

        if abs(new_value - params_dict.get("gravity", 0.0)) < 1e-6:
            return

        command = DictItemCommand(
            target_dict=params_dict,
            key="gravity",
            new_value=new_value,
            name=_("Change Shrinkwrap Gravity"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)
