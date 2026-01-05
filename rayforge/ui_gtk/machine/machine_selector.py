import logging
from typing import Optional, cast
from gi.repository import Gtk, Gio, GObject
from blinker import Signal
from ...context import get_context
from ...machine.models.machine import Machine

logger = logging.getLogger(__name__)


# This allows the plain Python Machine object to be stored in a Gio.ListStore.
class MachineListItem(GObject.Object):
    __gtype_name__ = "MachineListItem"

    def __init__(self, machine: Machine):
        super().__init__()
        self.machine = machine


class MachineSelector(Gtk.DropDown):
    """
    A self-contained dropdown widget for selecting the active machine.

    This widget is fully autonomous. It listens to the `machine_mgr` to
    keep its list of machines up-to-date, and it listens to `config.changed`
    to keep its selection synchronized with the global application state.

    It emits a blinker signal `machine_selected` when the user actively
    chooses a new machine from the dropdown.
    """

    __gtype_name__ = "MachineSelector"

    # Blinker signal emitted when a user selects a machine.
    # sender: MachineSelector instance
    # machine: The selected Machine object
    machine_selected: Signal = Signal()

    def __init__(self, **kwargs):
        self._model = Gio.ListStore.new(MachineListItem)

        expression = Gtk.ClosureExpression.new(
            str,
            lambda item: item.machine.name if item else _("Select Machine"),
            None,
        )

        super().__init__(model=self._model, expression=expression, **kwargs)

        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._on_factory_setup)
        factory.connect("bind", self.on_factory_bind)
        self.set_factory(factory)

        self.set_tooltip_text(_("Select active machine"))
        self._selection_changed_handler_id = self.connect(
            "notify::selected-item", self._on_user_selection_changed
        )

        # Listen to signals to keep the widget's state up-to-date.
        context = get_context()
        context.machine_mgr.machine_added.connect(
            self.update_model_and_selection
        )
        context.machine_mgr.machine_removed.connect(
            self.update_model_and_selection
        )
        context.machine_mgr.machine_updated.connect(
            self.update_model_and_selection
        )
        context.config.changed.connect(self.update_model_and_selection)

        # Initial population and selection sync.
        self.update_model_and_selection()

    def _on_factory_setup(self, factory, list_item):
        """Setup a list item for the machine dropdown."""
        box = Gtk.Box(spacing=6)
        label = Gtk.Label()
        box.append(label)
        list_item.set_child(box)

    def on_factory_bind(self, factory, list_item):
        """Bind a machine object to a list item."""
        box = list_item.get_child()
        label = box.get_first_child()
        list_item_obj: Optional[MachineListItem] = list_item.get_item()
        if list_item_obj:
            label.set_text(list_item_obj.machine.name)

    def update_model_and_selection(self, *args, **kwargs):
        """
        Repopulates the model from the machine manager and syncs the
        selection with the current config. This is the central method for
        ensuring the widget's state is correct.
        """
        logger.debug("Syncing machine selector model and selection.")
        context = get_context()
        machines = sorted(
            context.machine_mgr.machines.values(), key=lambda m: m.name
        )

        # Block the GTK signal while we modify the list and selection
        # to prevent an infinite loop of signals.
        self.handler_block(self._selection_changed_handler_id)

        try:
            # Only update if the list of machines or the selected machine
            # differs
            self._model.remove_all()
            selected_index = -1
            for i, machine in enumerate(machines):
                self._model.append(MachineListItem(machine))
                if context.machine and machine.id == context.machine.id:
                    selected_index = i

            # Only set selected if it's a valid index (>= 0)
            if selected_index >= 0 and self.get_selected() != selected_index:
                self.set_selected(selected_index)
            elif selected_index < 0 and self.get_selected() >= 0:
                # If no machine is selected (selected_index is -1) but there's
                # currently a selection, clear it by setting to 0 if possible
                # or leave it as is if the model is empty
                if len(self._model) > 0:
                    self.set_selected(0)

        finally:
            self.handler_unblock(self._selection_changed_handler_id)

    def _on_user_selection_changed(self, dropdown, param):
        """
        Handles user-driven selection changes and emits the blinker signal.
        This is only fired by direct user interaction, not programmatic
        changes.
        """
        selected_list_item = cast(
            Optional[MachineListItem], self.get_selected_item()
        )

        if selected_list_item:
            logger.info(
                f"User selected '{selected_list_item.machine.name}'. "
                "Emitting 'machine_selected' signal."
            )
            self.machine_selected.send(
                self, machine=selected_list_item.machine
            )
