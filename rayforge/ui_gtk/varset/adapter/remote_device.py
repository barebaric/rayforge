"""Adapter for a device picker backed by a raydriver daemon.

The choices are queried live from the daemon identified by the
sibling endpoint vars. The daemon resolves each device's protocol;
the adapter records it under a related key so the driver can derive
capabilities from raydriver's metadata without naming a protocol.
"""

import logging
from gettext import gettext as _
from typing import Any

from gi.repository import Adw, Gtk
from raydriver.remote.client import list_devices

from ....core.varset import RemoteDeviceVar, Var
from ....shared.tasker import task_mgr
from .base import NULL_CHOICE_LABEL, RowAdapter, escape_title, register_adapter

logger = logging.getLogger(__name__)

_ENDPOINT_KEYS = (
    "remote_host",
    "remote_port",
    "token",
    "tls_ca",
    "tls_hostname",
)


@register_adapter(RemoteDeviceVar)
class RemoteDeviceAdapter(RowAdapter):
    """A combo row listing the devices a daemon hosts."""

    related_keys = ("_raydriver_protocol",)

    def __init__(self, row: Adw.ComboRow, var: Var) -> None:
        super().__init__()
        self._row = row
        self._var = var
        self._value: str | None = None
        self._protocols: dict[str, str] = {}
        self._labels: dict[str, str] = {}
        self._endpoint: dict[str, Any] | None = None
        self._row.connect(
            "notify::selected-item", self._on_selected_item_changed
        )

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[Adw.PreferencesRow, "RemoteDeviceAdapter"]:
        initial_val = getattr(var, target_property)
        row = Adw.ComboRow(title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        adapter = cls(row, var)
        adapter._value = str(initial_val) if initial_val else None

        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", adapter._on_factory_setup)
        factory.connect("bind", adapter._on_factory_bind)
        row.set_factory(factory)
        adapter._rebuild([])

        def on_open(
            gesture: Gtk.GestureClick, n_press: int, x: float, y: float
        ) -> None:
            adapter._query_devices()

        click_controller = Gtk.GestureClick.new()
        click_controller.connect("pressed", on_open)
        row.add_controller(click_controller)
        return row, adapter

    def _on_factory_setup(
        self, factory: Gtk.SignalListItemFactory, list_item: Gtk.ListItem
    ) -> None:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        name_label = Gtk.Label(xalign=0)
        subtitle_label = Gtk.Label(xalign=0)
        subtitle_label.add_css_class("dim-label")
        box.append(name_label)
        box.append(subtitle_label)
        list_item.set_child(box)

    def _on_factory_bind(
        self, factory: Gtk.SignalListItemFactory, list_item: Gtk.ListItem
    ) -> None:
        item = list_item.get_item()
        assert isinstance(item, Gtk.StringObject)
        ref = item.get_string()
        box = list_item.get_child()
        assert box is not None
        name_label = box.get_first_child()
        assert isinstance(name_label, Gtk.Label)
        sibling = name_label.get_next_sibling()
        assert isinstance(sibling, Gtk.Label)
        name_label.set_label(ref)
        label = self._labels.get(ref)
        sibling.set_visible(bool(label))
        if label:
            sibling.set_label(label)

    def update_from_values(self, values: dict[str, Any]) -> None:
        endpoint = {key: values.get(key) for key in _ENDPOINT_KEYS}
        if endpoint != self._endpoint:
            self._endpoint = endpoint
            if endpoint.get("remote_host"):
                self._query_devices()

    def _query_devices(self) -> None:
        endpoint = self._endpoint
        if not endpoint or not endpoint.get("remote_host"):
            return
        self._row.set_subtitle(_("Searching for devices…"))

        async def _coroutine(exec_ctx: Any) -> Any:
            # Return errors instead of raising so the task manager does
            # not log them as unhandled; the picker shows them inline.
            try:
                return await list_devices(dict(endpoint))
            except (ConnectionError, OSError) as err:
                return err

        task_mgr.add_coroutine(
            _coroutine,
            key=f"remote-device-list-{id(self)}",
            when_done=self._on_devices,
        )

    def _on_devices(self, task: Any) -> None:
        def _update() -> None:
            result = task.result()
            if isinstance(result, Exception):
                logger.debug("Listing daemon devices failed: %s", result)
                self._row.set_subtitle(
                    _(
                        "Could not list devices — check the hostname and "
                        "token (the daemon may require a token)."
                    )
                )
                return
            devices = result
            self._protocols = {
                d["device"]: d.get("protocol", "") for d in devices
            }
            self._labels = {d["device"]: d.get("label", "") for d in devices}
            self._rebuild([d["device"] for d in devices])
            if devices:
                self._row.set_subtitle(self._var.description or "")
            else:
                self._row.set_subtitle(
                    _("The daemon has no devices configured.")
                )

        task_mgr.schedule_on_main_thread(_update)

    def _rebuild(self, refs: list[str]) -> None:
        items = [NULL_CHOICE_LABEL]
        for ref in refs:
            if ref not in items:
                items.append(ref)
        # Keep a configured device visible even when the daemon is
        # offline or no longer lists it.
        if self._value and self._value not in items:
            items.append(self._value)
        model = Gtk.StringList.new(items)
        self._row.set_model(model)
        selected = 0
        if self._value and self._value in items:
            selected = items.index(self._value)
        self._row.set_selected(selected)

    def _on_selected_item_changed(self, row: Any, param: Any) -> None:
        self.changed.send(self)

    def get_value(self) -> Any | None:
        selected = self._row.get_selected_item()
        if isinstance(selected, Gtk.StringObject):
            value = selected.get_string()
            if value != NULL_CHOICE_LABEL:
                self._value = value
                return value
        return None

    def get_value_for_key(self, key: str) -> Any | None:
        if key == "_raydriver_protocol":
            ref = self.get_value()
            return self._protocols.get(ref) if ref else None
        return self.get_value()

    def set_value(self, value: Any) -> None:
        self._value = str(value) if value else None
        model = self._row.get_model()
        strings: list[str] = []
        if isinstance(model, Gtk.StringList):
            strings = [
                s
                for i in range(model.get_n_items())
                if (s := model.get_string(i)) is not None
            ]
        if self._value and self._value not in strings:
            self._rebuild(strings[1:] if strings else [])
            return
        selected = 0
        if self._value and self._value in strings:
            selected = strings.index(self._value)
        self._row.set_selected(selected)
