from typing import Any

from gi.repository import Adw, Gtk

from ....core.varset import (
    BaudrateVar,
    ChoiceVar,
    SerialPortVar,
    Var,
)
from ....machine.transport.serial import (
    SerialPortInfo,
    SerialTransport,
    is_usb_serial_port,
    natural_key,
)
from ...shared.adwfix import ensure_row_min_width
from .base import (
    NULL_CHOICE_LABEL,
    RowAdapter,
    escape_title,
    register_adapter,
)


@register_adapter(ChoiceVar)
class ComboAdapter(RowAdapter):
    def __init__(self, row: Adw.ComboRow, var: Var) -> None:
        super().__init__()
        self._row = row
        self._var = var
        self._disabled_displays: set[str] = set()
        self._disabled_factory_attached: bool = False
        self._row.connect(
            "notify::selected-item",
            lambda r, p: self.changed.send(self),
        )

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[Adw.PreferencesRow, "ComboAdapter"]:
        assert isinstance(var, ChoiceVar)
        null_label = var.null_label or NULL_CHOICE_LABEL
        choices: list[str] = (
            [null_label] + var.choices if var.allow_none else list(var.choices)
        )
        store = Gtk.StringList.new(choices)
        row = Adw.ComboRow(model=store, title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        ensure_row_min_width(row)
        initial_val = getattr(var, target_property)
        if initial_val:
            display_str = var.get_display_for_value(str(initial_val))
            if display_str in choices:
                row.set_selected(choices.index(display_str))
            else:
                row.set_selected(0)
        else:
            row.set_selected(0)
        return row, cls(row, var)

    def set_disabled_displays(self, displays: set[str]) -> None:
        """
        Disables individual dropdown entries, keyed by display string.

        Disabled entries stay visible but cannot be activated, so a
        value that the current machine cannot handle remains readable.
        The popup factory is attached lazily so combo rows without
        disabled entries keep their default appearance.
        """
        if displays == self._disabled_displays:
            return
        self._disabled_displays = set(displays)
        if self._disabled_displays and not self._disabled_factory_attached:
            factory = Gtk.SignalListItemFactory()
            factory.connect("setup", self._on_item_setup)
            factory.connect("bind", self._on_item_bind)
            self._row.set_factory(factory)
            self._disabled_factory_attached = True

    def _on_item_setup(
        self, _factory: Gtk.SignalListItemFactory, list_item: Gtk.ListItem
    ) -> None:
        list_item.set_child(Gtk.Label(xalign=0))

    def _on_item_bind(
        self, _factory: Gtk.SignalListItemFactory, list_item: Gtk.ListItem
    ) -> None:
        item = list_item.get_item()
        assert isinstance(item, Gtk.StringObject)
        label = list_item.get_child()
        assert isinstance(label, Gtk.Label)
        display = item.get_string()
        label.set_text(display)
        label.set_sensitive(display not in self._disabled_displays)

    def get_value(self) -> Any | None:
        selected = self._row.get_selected_item()
        display_str = ""
        if selected:
            display_str = selected.get_string()  # type: ignore

        null_label = (
            getattr(self._var, "null_label", None) or NULL_CHOICE_LABEL
        )
        if display_str == null_label:
            return None
        if isinstance(self._var, ChoiceVar):
            return self._var.get_value_for_display(display_str)
        return display_str

    def set_value(self, value: Any) -> None:
        model = self._row.get_model()
        if not isinstance(model, Gtk.StringList):
            return
        null_label = (
            getattr(self._var, "null_label", None) or NULL_CHOICE_LABEL
        )
        display_str = null_label
        if value is not None:
            if isinstance(self._var, ChoiceVar):
                display_str = self._var.get_display_for_value(
                    str(value)
                ) or str(value)
            else:
                display_str = str(value)
        for i in range(model.get_n_items()):
            if model.get_string(i) == display_str:
                self._row.set_selected(i)
                break

    def needs_rebuild(self, old_var: Var, new_var: Var) -> bool:
        if super().needs_rebuild(old_var, new_var):
            return True
        if isinstance(old_var, ChoiceVar) and isinstance(new_var, ChoiceVar):
            return old_var.choices != new_var.choices
        return False

    def update_from_var(self, var: Var):
        if var.label:
            self._row.set_title(escape_title(var.label))
        if var.description:
            self._row.set_subtitle(var.description)


@register_adapter(BaudrateVar)
class BaudRateAdapter(ComboAdapter):
    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[Adw.PreferencesRow, "BaudRateAdapter"]:
        assert isinstance(var, BaudrateVar)
        choices_str = [str(rate) for rate in var.choices]
        store = Gtk.StringList.new(choices_str)
        row = Adw.ComboRow(model=store, title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        initial_val = getattr(var, target_property)
        if initial_val is not None and str(initial_val) in choices_str:
            row.set_selected(choices_str.index(str(initial_val)))
        return row, cls(row, var)


@register_adapter(SerialPortVar)
class SerialPortAdapter(ComboAdapter):
    """
    A two-line port selector, mirroring the WCS selector: the device
    path on the first line and the USB description (if known) as a
    dimmed second line. The model holds raw device paths, so value
    mapping is handled by ComboAdapter directly.
    """

    def __init__(self, row: Adw.ComboRow, var: Var) -> None:
        super().__init__(row, var)
        self._descriptions: dict[str, str] = {}

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[Adw.PreferencesRow, "SerialPortAdapter"]:
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
        path = item.get_string()
        box = list_item.get_child()
        assert box is not None
        name_label = box.get_first_child()
        assert isinstance(name_label, Gtk.Label)
        sibling = name_label.get_next_sibling()
        assert isinstance(sibling, Gtk.Label)
        name_label.set_label(path)
        description = self._descriptions.get(path)
        sibling.set_visible(bool(description))
        if description:
            sibling.set_label(description)

    @staticmethod
    def _scan_ports(extra_value: str | None) -> list[SerialPortInfo]:
        """
        Returns the available ports. USB adapters come first; a
        configured port that is not currently plugged in is pinned
        to the top so it stays immediately visible.
        """
        ports: list[SerialPortInfo] = []
        seen: set[str] = set()
        for info in SerialTransport.list_port_info():
            if info.device in seen:
                continue
            seen.add(info.device)
            ports.append(info)
        ports.sort(
            key=lambda i: (
                not is_usb_serial_port(i.device),
                natural_key(i.device),
            )
        )
        if extra_value and extra_value not in seen:
            ports.insert(0, SerialPortInfo(extra_value))
        return ports

    def _refresh(self, current_value: Any | None) -> None:
        """Re-scans ports and rebuilds the dropdown contents."""
        value_str = str(current_value) if current_value else None
        ports = self._scan_ports(value_str)
        self._descriptions = {
            p.device: p.description for p in ports if p.description
        }
        devices = [p.device for p in ports]
        model = Gtk.StringList.new([NULL_CHOICE_LABEL] + devices)
        self._row.set_model(model)
        selected = 0
        if value_str and value_str in devices:
            selected = devices.index(value_str) + 1
        self._row.set_selected(selected)

    def set_value(self, value: Any) -> None:
        if value is None:
            super().set_value(None)
            return
        value_str = str(value)
        model = self._row.get_model()
        strings: list[str | None] = []
        if isinstance(model, Gtk.StringList):
            strings = [model.get_string(i) for i in range(model.get_n_items())]
        if value_str not in strings:
            # Unknown port (e.g. just plugged in or not seen by the
            # last scan): rebuild the list so it becomes selectable.
            self._refresh(value_str)
            return
        super().set_value(value_str)
