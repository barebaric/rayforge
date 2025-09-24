from typing import Dict, Any, TYPE_CHECKING, cast, Tuple
from gi.repository import Gtk, Adw, GLib
from .base import StepComponentSettingsWidget
from ....pipeline.producer.base import OpsProducer
from ....pipeline.producer.shrinkwrap import HullProducer
from ....undo import DictItemCommand

if TYPE_CHECKING:
    from ....core.step import Step
    from ....undo import HistoryManager


class HullProducerSettingsWidget(StepComponentSettingsWidget):
    """UI for configuring the HullProducer."""

    def __init__(
        self,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        history_manager: "HistoryManager",
        **kwargs,
    ):
        producer = cast(HullProducer, OpsProducer.from_dict(target_dict))

        super().__init__(
            target_dict=target_dict,
            page=page,
            step=step,
            history_manager=history_manager,
            **kwargs,
        )

        self._debounce_timer = 0
        self._debounced_callback = None
        self._debounced_args: Tuple = ()

        # Gravity setting (Slider)
        gravity_row = Adw.ActionRow(
            title=_("Gravity"),
            subtitle=_(
                "Pulls the hull inward. 0.0 is a standard convex hull"
            ),
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

    def _debounce(self, callback, *args):
        if self._debounce_timer > 0:
            GLib.source_remove(self._debounce_timer)
        self._debounced_callback = callback
        self._debounced_args = args
        self._debounce_timer = GLib.timeout_add(
            150, self._commit_debounced_change
        )

    def _commit_debounced_change(self):
        if self._debounced_callback:
            self._debounced_callback(*self._debounced_args)
        self._debounce_timer = 0
        return GLib.SOURCE_REMOVE

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
