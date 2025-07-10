import inspect
import re
from gi.repository import Gtk, Adw
from blinker import Signal
from ..util.adwfix import get_spinrow_int
from ..transport.serial import SerialTransport, SerialPort
from typing import Any


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


class DynamicPreferencesGroup(Adw.PreferencesGroup):
    def __init__(self, *args, **kwargs):
        """
        Params is a dict of items as returned by
        inspect.signature.parameters.items()
        """
        super().__init__(*args, **kwargs)
        self.widget_map = {}
        self.data_changed = Signal()

    def clear(self):
        for row, _ in self.widget_map.values():
            self.remove(row)
        self.widget_map = {}

    def create_params(self, params):
        self.clear()

        # Get constructor parameters
        for name, param in params:
            if name == 'self':
                continue

            annotation = param.annotation
            isempty = param.default == inspect.Parameter.empty
            default = param.default if not isempty else None

            # Create appropriate row based on type
            if annotation is SerialPort:
                row = self._create_port_selection_row(name, default)
            elif annotation is str:
                row = self._create_string_row(name, default)
            elif annotation is bool:
                row = self._create_boolean_row(name, default)
            elif annotation is int:
                row = self._create_integer_row(name, default)
            else:
                continue  # Skip unsupported types

            self.add(row)
            self.widget_map[name] = (row, annotation)

    def _create_string_row(self, name, default):
        row = Adw.EntryRow(title=name.capitalize())
        if default is not None:
            row.set_text(str(default))
        row.connect("changed", lambda e: self.data_changed.send(e))
        return row

    def _create_boolean_row(self, name, default):
        row = Adw.ActionRow(title=name.capitalize())
        switch = Gtk.Switch()
        switch.set_active(default if default is not None else False)
        switch.set_valign(Gtk.Align.CENTER)
        row.add_suffix(switch)
        row.activatable_widget = switch
        row.switch = switch  # Store reference
        return row

    def _create_integer_row(self, name, default):
        adjustment = Gtk.Adjustment(
            value=default if default is not None else 0,
            lower=-2147483648,
            upper=2147483647,
            step_increment=1
        )
        row = Adw.SpinRow(title=name.capitalize(), adjustment=adjustment)
        row.connect("changed", lambda e: self.data_changed.send(e))
        return row

    def _create_port_selection_row(self, name: str, default: Any):
        ports = sorted(SerialTransport.list_ports(), key=natural_sort_key)
        store = Gtk.StringList.new(ports)
        row = Adw.ComboRow(title=name.capitalize(), model=store)
        if default and default in ports:
            row.set_selected(ports.index(default))
        row.connect("notify::selected-item",
                    lambda e, p: self.data_changed.send(e))
        return row

    def get_values(self):
        values = {}
        for name, (row, annotation) in self.widget_map.items():
            if annotation is SerialPort:
                selected_item = row.get_selected_item()
                if selected_item:
                    values[name] = selected_item.get_string()
                else:
                    values[name] = ""
            elif annotation is int:
                values[name] = get_spinrow_int(row)
            elif annotation is bool:
                values[name] = row.switch.get_active()
            elif annotation is str:
                values[name] = row.get_text()
        return values

    def set_values(self, values):
        for name, value in values.items():
            item = self.widget_map.get(name)
            if item is None:
                continue
            row, annotation = item

            if annotation is SerialPort:
                ports = SerialTransport.list_ports()
                try:
                    index = ports.index(str(value))
                    row.set_selected(index)
                except ValueError:
                    row.set_selected(0)  # Select the first item as a fallback
            elif annotation is str:
                row.set_text(str(value))
            elif annotation is int:
                row.set_value(int(value))
            elif annotation is bool:
                row.switch.set_active(bool(value))
        return values
