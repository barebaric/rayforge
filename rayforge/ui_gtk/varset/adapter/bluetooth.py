from gettext import gettext as _
from typing import Any, cast

from gi.repository import Adw, Gtk
from raygeo.driver.core.bluetooth import (
    BluetoothDeviceInfo,
    list_paired_devices,
    normalize_address,
)

from ....core.varset import BluetoothAddressVar, Var
from .base import (
    NULL_CHOICE_LABEL,
    escape_title,
    register_adapter,
)
from .combo import ComboAdapter

MANUAL_ADDRESS_LABEL = _("Enter Bluetooth address…")


@register_adapter(BluetoothAddressVar)
class BluetoothAddressAdapter(ComboAdapter):
    """
    A two-line paired-device selector: the device name on the first
    line and its address (if known) as a dimmed second line, plus a
    manual-entry fallback for devices the scan cannot see.
    """

    def __init__(self, row: Adw.ComboRow, var: Var) -> None:
        super().__init__(row, var)
        self._descriptions: dict[str, str] = {}
        self._value: str | None = None
        self._row.connect(
            "notify::selected-item", self._on_selected_item_changed
        )

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[Adw.PreferencesRow, "BluetoothAddressAdapter"]:
        initial_val = getattr(var, target_property)
        row = Adw.ComboRow(title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        adapter = cls(row, var)

        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", adapter._on_factory_setup)
        factory.connect("bind", adapter._on_factory_bind)
        row.set_factory(factory)

        adapter._refresh(initial_val)

        def on_open(
            gesture: Gtk.GestureClick, n_press: int, x: float, y: float
        ) -> None:
            adapter._refresh(adapter.get_value())

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
        address = item.get_string()
        box = list_item.get_child()
        assert box is not None
        name_label = box.get_first_child()
        assert isinstance(name_label, Gtk.Label)
        sibling = name_label.get_next_sibling()
        assert isinstance(sibling, Gtk.Label)

        name = self._descriptions.get(address)
        name_label.set_label(name or address)
        sibling.set_visible(bool(name))
        if name:
            sibling.set_label(address)

    @staticmethod
    def _scan_devices(
        extra_value: str | None,
    ) -> list[BluetoothDeviceInfo]:
        devices: list[BluetoothDeviceInfo] = []
        seen: set[str] = set()
        for info in list_paired_devices():
            if info.address in seen:
                continue
            seen.add(info.address)
            devices.append(info)
        devices.sort(key=lambda i: ((i.name or "").lower(), i.address))
        if extra_value and extra_value not in seen:
            devices.insert(0, BluetoothDeviceInfo(extra_value))
        return devices

    def _refresh(self, current_value: Any | None) -> None:
        value_str = str(current_value) if current_value else None
        self._value = value_str
        devices = self._scan_devices(value_str)
        self._descriptions = {d.address: d.name for d in devices if d.name}
        items = (
            [NULL_CHOICE_LABEL]
            + [d.address for d in devices]
            + [MANUAL_ADDRESS_LABEL]
        )
        self._row.set_model(Gtk.StringList.new(items))
        selected = 0
        if value_str and value_str in items:
            selected = items.index(value_str)
        self._row.set_selected(selected)

    def get_value(self) -> Any | None:
        selected = self._row.get_selected_item()
        if (
            isinstance(selected, Gtk.StringObject)
            and selected.get_string() == MANUAL_ADDRESS_LABEL
        ):
            return self._value
        value = super().get_value()
        if value is not None:
            self._value = str(value)
        return value

    def set_value(self, value: Any) -> None:
        if value is None:
            self._value = None
            super().set_value(None)
            return
        value_str = str(value)
        try:
            value_str = normalize_address(value_str)
        except ValueError:
            pass
        self._value = value_str
        model = self._row.get_model()
        strings: list[str | None] = []
        if isinstance(model, Gtk.StringList):
            strings = [model.get_string(i) for i in range(model.get_n_items())]
        if value_str not in strings:
            self._refresh(value_str)
            return
        super().set_value(value_str)

    def _on_selected_item_changed(
        self, row: Adw.ComboRow, _pspec: Any
    ) -> None:
        selected = row.get_selected_item()
        if not isinstance(selected, Gtk.StringObject):
            return
        if selected.get_string() != MANUAL_ADDRESS_LABEL:
            return
        self._prompt_address()
        self.set_value(self._value)

    def _prompt_address(self) -> None:
        entry = Adw.EntryRow(title=_("Bluetooth address (AA:BB:CC:DD:EE:FF)"))
        if self._value:
            entry.set_text(self._value)
        dialog = Adw.MessageDialog(
            transient_for=cast(Gtk.Window, self._row.get_root()),
            modal=True,
            heading=_("Enter Bluetooth address"),
            body=_(
                "Use this when the device is paired but not listed. "
                "Find its address in the operating system's Bluetooth "
                "settings."
            ),
        )
        dialog.set_extra_child(entry)
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("apply", _("Apply"))
        dialog.set_response_appearance(
            "apply", Adw.ResponseAppearance.SUGGESTED
        )
        dialog.set_default_response("apply")
        dialog.set_close_response("cancel")

        def on_validate(_entry: Adw.EntryRow, _pspec: Any) -> None:
            try:
                normalize_address(entry.get_text())
            except ValueError:
                entry.add_css_class("error")
            else:
                entry.remove_css_class("error")

        entry.connect("notify::text", on_validate)

        def on_response(d: Adw.MessageDialog, response: str) -> None:
            if response != "apply":
                return
            try:
                normalized = normalize_address(entry.get_text())
            except ValueError:
                return
            self.set_value(normalized)

        dialog.connect("response", on_response)
        dialog.present()
