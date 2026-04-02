import logging
from typing import Optional, cast
from gettext import gettext as _
from gi.repository import Gtk, Gio, GObject, Pango
from blinker import Signal
from ...context import get_context
from ...machine.driver.driver import (
    DEVICE_STATUS_LABELS,
    DeviceStatus,
)
from ...machine.driver.dummy import NoDeviceDriver
from ...machine.models.machine import Machine
from ...machine.transport.transport import TransportStatus
from ...shared.util.time_format import format_seconds
from ..icons import get_icon

logger = logging.getLogger(__name__)


class MachineListItem(GObject.Object):
    __gtype_name__ = "MachineListItem"

    def __init__(self, machine: Machine):
        super().__init__()
        self.machine = machine


def _get_connection_icon_name(status: TransportStatus) -> str:
    if status == TransportStatus.UNKNOWN:
        return "question-box-symbolic"
    elif status == TransportStatus.IDLE:
        return "status-idle-symbolic"
    elif status == TransportStatus.CONNECTING:
        return "status-connecting-symbolic"
    elif status == TransportStatus.CONNECTED:
        return "status-connected-symbolic"
    elif status == TransportStatus.ERROR:
        return "error-symbolic"
    elif status == TransportStatus.CLOSING:
        return "status-offline-symbolic"
    elif status == TransportStatus.DISCONNECTED:
        return "status-offline-symbolic"
    elif status == TransportStatus.SLEEPING:
        return "sleep-symbolic"
    else:
        return "status-offline-symbolic"


def _get_status_text(
    machine: Machine, eta_seconds: Optional[float] = None
) -> str:
    is_nodriver = isinstance(machine.driver, NoDeviceDriver)
    if is_nodriver:
        return _("No driver")
    status = machine.device_state.status
    text = DEVICE_STATUS_LABELS.get(status, _("Unknown"))
    if (
        status == DeviceStatus.RUN
        and eta_seconds is not None
        and eta_seconds > 0
    ):
        text = f"{text} · {format_seconds(eta_seconds)}"
    return text


def _get_connection_status(machine: Machine) -> TransportStatus:
    if isinstance(machine.driver, NoDeviceDriver):
        return TransportStatus.DISCONNECTED
    return machine.connection_status


class MachineDropdown(Gtk.DropDown):
    """
    A dropdown for selecting the active machine, showing connection state
    and machine status in each entry.
    """

    __gtype_name__ = "MachineDropdown"

    def __init__(self, **kwargs):
        self.machine_selected = Signal()
        self._model = Gio.ListStore.new(MachineListItem)
        self._eta_seconds: Optional[float] = None
        self._status_label_refs: dict = {}

        expression = Gtk.ClosureExpression.new(
            str,
            lambda item: item.machine.name if item else _("Select Machine"),
            None,
        )

        super().__init__(model=self._model, expression=expression, **kwargs)

        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._on_factory_setup)
        factory.connect("bind", self._on_factory_bind)
        factory.connect("unbind", self._on_factory_unbind)
        self.set_factory(factory)

        self.set_tooltip_text(_("Select active machine"))
        self._selection_changed_handler_id = self.connect(
            "notify::selected-item", self._on_user_selection_changed
        )

        self._signal_refs = []

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

        self.update_model_and_selection()

    def _on_factory_setup(self, factory, list_item):
        box = Gtk.Box(spacing=8)

        icon_box = Gtk.Box(valign=Gtk.Align.CENTER)
        box.append(icon_box)

        text_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, valign=Gtk.Align.CENTER
        )
        name_label = Gtk.Label(
            xalign=0,
            ellipsize=Pango.EllipsizeMode.END,
        )
        status_label = Gtk.Label(
            xalign=0,
            ellipsize=Pango.EllipsizeMode.END,
        )
        status_label.add_css_class("caption")
        status_label.add_css_class("dim-label")

        text_box.append(name_label)
        text_box.append(status_label)
        box.append(text_box)

        list_item.set_child(box)
        list_item._signal_refs = []

    def _on_factory_bind(self, factory, list_item):
        box = list_item.get_child()
        icon_box = box.get_first_child()
        text_box = icon_box.get_next_sibling()
        name_label = text_box.get_first_child()
        status_label = name_label.get_next_sibling()

        list_item_obj: Optional[MachineListItem] = list_item.get_item()
        if not list_item_obj:
            return

        machine = list_item_obj.machine
        name_label.set_text(machine.name)
        status_label.set_text(_get_status_text(machine))

        conn_status = _get_connection_status(machine)
        icon_name = _get_connection_icon_name(conn_status)
        new_img = get_icon(icon_name)
        child = icon_box.get_first_child()
        if child:
            icon_box.remove(child)
        icon_box.append(new_img)

        for ref in list_item._signal_refs:
            try:
                ref[0].disconnect(ref[1])
            except (TypeError, Exception):
                pass

        refs = []

        def on_state_changed(m, state, lbl=status_label):
            lbl.set_text(_get_status_text(m, self._get_eta_for_machine(m)))

        def on_conn_changed(m, status, message=None, ibox=icon_box):
            conn = _get_connection_status(m)
            img = get_icon(_get_connection_icon_name(conn))
            old = ibox.get_first_child()
            if old:
                ibox.remove(old)
            ibox.append(img)

        machine.state_changed.connect(on_state_changed)
        refs.append((machine, on_state_changed))

        machine.connection_status_changed.connect(on_conn_changed)
        refs.append((machine, on_conn_changed))

        self._status_label_refs[id(machine)] = status_label

        list_item._signal_refs = refs

    def _on_factory_unbind(self, factory, list_item):
        list_item_obj: Optional[MachineListItem] = list_item.get_item()
        if list_item_obj:
            self._status_label_refs.pop(id(list_item_obj.machine), None)
        for ref in list_item._signal_refs:
            try:
                ref[0].disconnect(ref[1])
            except (TypeError, Exception):
                pass
        list_item._signal_refs = []

    def _get_eta_for_machine(self, machine: Machine) -> Optional[float]:
        context = get_context()
        if context.config.machine and context.config.machine.id == machine.id:
            return self._eta_seconds
        return None

    def update_eta(self, eta_seconds: Optional[float]):
        """Update the ETA for the active machine's status label."""
        self._eta_seconds = eta_seconds
        context = get_context()
        machine = context.config.machine
        if not machine:
            return
        label = self._status_label_refs.get(id(machine))
        if label:
            label.set_text(_get_status_text(machine, eta_seconds))

    def update_model_and_selection(self, *args, **kwargs):
        logger.debug("Syncing machine dropdown model and selection.")
        context = get_context()
        machines = sorted(
            context.machine_mgr.machines.values(), key=lambda m: m.name
        )

        self.handler_block(self._selection_changed_handler_id)

        try:
            self._model.remove_all()
            selected_index = -1
            for i, machine in enumerate(machines):
                self._model.append(MachineListItem(machine))
                if context.machine and machine.id == context.machine.id:
                    selected_index = i

            if selected_index >= 0 and self.get_selected() != selected_index:
                self.set_selected(selected_index)
            elif selected_index < 0 and self.get_selected() >= 0:
                if len(self._model) > 0:
                    self.set_selected(0)

        finally:
            self.handler_unblock(self._selection_changed_handler_id)

    def _on_user_selection_changed(self, dropdown, param):
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
