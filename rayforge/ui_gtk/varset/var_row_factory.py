import re
import logging
from typing import Callable, Dict, Type

from gi.repository import Adw, Gtk

from ...core.varset import (
    BaudrateVar,
    BoolVar,
    ChoiceVar,
    FloatVar,
    HostnameVar,
    IntVar,
    PortVar,
    SerialPortVar,
    SliderFloatVar,
    TextAreaVar,
    Var,
)
from ...machine.transport.serial import SerialTransport

logger = logging.getLogger(__name__)
NULL_CHOICE_LABEL = _("None Selected")


def escape_title(text: str) -> str:
    """Escape ampersands for GTK/Adwaita title display."""
    return text.replace("&", "&&")


def natural_sort_key(s: str) -> list:
    """Sorts strings containing numbers in a natural order."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


class VarRowFactory:
    """
    A factory for creating Adwaita preference rows from Var objects.

    This class acts as a pure "builder". It constructs and configures the
    appropriate Adw.PreferencesRow for a given Var, but does not connect
    any signals that would modify the Var's state. The caller is responsible
    for wiring up the row's signals.
    """

    def __init__(self):
        self._CREATOR_MAP: Dict[Type[Var], Callable] = {
            TextAreaVar: self._create_textarea_row,
            SliderFloatVar: self._create_slider_row,
            BaudrateVar: self._create_baud_rate_row,
            PortVar: self._create_integer_row,
            SerialPortVar: self._create_port_selection_row,
            HostnameVar: self._create_hostname_row,
            ChoiceVar: self._create_choice_row,
            FloatVar: self._create_float_row,
            IntVar: self._create_integer_row,
            BoolVar: self._create_boolean_row,
        }

    def create_row_for_var(
        self, var: Var, target_property: str = "value"
    ) -> Adw.PreferencesRow:
        row: Adw.PreferencesRow | None = None

        # 1. Try to find a creator for the specific subclass
        for var_class, creator_method in self._CREATOR_MAP.items():
            if isinstance(var, var_class):
                row = creator_method(var, target_property)
                break

        # 2. If no specific creator found, dispatch based on value type
        if row is None:
            if var.var_type is int:
                row = self._create_integer_row(var, target_property)
            elif var.var_type is float:
                row = self._create_float_row(var, target_property)
            elif var.var_type is bool:
                row = self._create_boolean_row(var, target_property)
            elif var.var_type is str:
                row = self._create_string_row(var, target_property)

        if row is None:
            logger.warning(
                f"No UI widget defined for Var with key '{var.key}' "
                f"and type {type(var)}"
            )
            row = Adw.ActionRow(
                title=escape_title(var.label),
                subtitle=_("Unsupported type: {t}").format(
                    t=type(var).__name__
                ),
                sensitive=False,
            )
        return row

    def _create_string_row(self, var: Var, target_property: str):
        row = Adw.EntryRow(title=escape_title(var.label))
        if var.description:
            row.set_tooltip_text(var.description)
        initial_val = getattr(var, target_property)
        if initial_val is not None:
            row.set_text(str(initial_val))
        return row

    def _create_hostname_row(self, var: HostnameVar, target_property: str):
        row = Adw.EntryRow(title=escape_title(var.label))
        if var.description:
            row.set_tooltip_text(var.description)
        row.set_show_apply_button(True)
        initial_val = getattr(var, target_property)
        if initial_val is not None:
            row.set_text(str(initial_val))
        return row

    def _create_boolean_row(self, var: Var, target_property: str):
        row = Adw.ActionRow(title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        switch = Gtk.Switch(valign=Gtk.Align.CENTER)
        initial_val = getattr(var, target_property)
        switch.set_active(
            bool(initial_val) if initial_val is not None else False
        )
        row.add_suffix(switch)
        row.set_activatable_widget(switch)
        return row

    def _create_integer_row(self, var: Var, target_property: str):
        # Use getattr to safely handle generic Vars that don't have min/max
        min_val = getattr(var, "min_val", None)
        max_val = getattr(var, "max_val", None)
        lower = min_val if min_val is not None else -2147483647
        upper = max_val if max_val is not None else 2147483647

        initial_val = getattr(var, target_property)

        adj = Gtk.Adjustment(
            value=int(initial_val) if initial_val is not None else 0,
            lower=lower,
            upper=upper,
            step_increment=1,
        )
        row = Adw.SpinRow(adjustment=adj, title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        return row

    def _create_float_row(self, var: Var, target_property: str):
        # Use getattr to safely handle generic Vars that don't have min/max
        min_val = getattr(var, "min_val", None)
        max_val = getattr(var, "max_val", None)
        lower = min_val if min_val is not None else -1.0e12
        upper = max_val if max_val is not None else 1.0e12

        initial_val = getattr(var, target_property)

        adj = Gtk.Adjustment(
            value=float(initial_val) if initial_val is not None else 0.0,
            lower=lower,
            upper=upper,
            step_increment=0.1,
        )
        row = Adw.SpinRow(
            adjustment=adj,
            digits=3,
            title=escape_title(var.label),
        )
        if var.description:
            row.set_subtitle(var.description)
        return row

    def _create_slider_row(self, var: SliderFloatVar, target_property: str):
        row = Adw.ActionRow(title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)

        min_val = var.min_val if var.min_val is not None else 0.0
        max_val = var.max_val if var.max_val is not None else 1.0
        val = getattr(var, target_property)
        if val is None:
            val = min_val

        # Calculate initial percentage for the slider
        initial_percent = 0.0
        range_size = max_val - min_val
        if range_size > 1e-9:
            initial_percent = ((val - min_val) / range_size) * 100.0

        adj = Gtk.Adjustment(
            value=initial_percent,
            lower=0.0,
            upper=100.0,
            step_increment=1,
            page_increment=10,
        )
        scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=adj,
            digits=0,
            draw_value=var.show_value,
            hexpand=True,
        )
        scale.set_size_request(200, -1)
        row.add_suffix(scale)
        row.set_activatable_widget(scale)
        return row

    def _create_choice_row(self, var: ChoiceVar, target_property: str):
        choices = [NULL_CHOICE_LABEL] + var.choices
        store = Gtk.StringList.new(choices)
        row = Adw.ComboRow(model=store, title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        initial_val = getattr(var, target_property)
        if initial_val and initial_val in choices:
            row.set_selected(choices.index(initial_val))
        else:
            row.set_selected(0)
        return row

    def _create_baud_rate_row(self, var: BaudrateVar, target_property: str):
        choices_str = [str(rate) for rate in SerialTransport.list_baud_rates()]
        store = Gtk.StringList.new(choices_str)
        row = Adw.ComboRow(model=store, title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        initial_val = getattr(var, target_property)
        if initial_val is not None and str(initial_val) in choices_str:
            row.set_selected(choices_str.index(str(initial_val)))
        return row

    def _create_port_selection_row(
        self, var: SerialPortVar, target_property: str
    ):
        initial_val = getattr(var, target_property)
        port_set = set(SerialTransport.list_ports())
        if initial_val:
            port_set.add(initial_val)
        sorted_ports = sorted(list(port_set), key=natural_sort_key)
        choices = [NULL_CHOICE_LABEL] + sorted_ports
        store = Gtk.StringList.new(choices)
        row = Adw.ComboRow(model=store, title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        if initial_val and initial_val in choices:
            row.set_selected(choices.index(initial_val))

        def on_open(gesture, n_press, x, y):
            selected_obj = row.get_selected_item()
            current_sel = None
            if selected_obj:
                current_sel = selected_obj.get_string()  # type: ignore

            new_ports = SerialTransport.list_ports()
            port_set = set(new_ports)
            if current_sel and current_sel != NULL_CHOICE_LABEL:
                port_set.add(current_sel)
            new_sorted = sorted(list(port_set), key=natural_sort_key)
            new_choices = [NULL_CHOICE_LABEL] + new_sorted

            model = row.get_model()
            if isinstance(model, Gtk.StringList):
                model.splice(0, model.get_n_items(), new_choices)
                if current_sel in new_choices:
                    row.set_selected(new_choices.index(current_sel))

        click_controller = Gtk.GestureClick.new()
        click_controller.connect("pressed", on_open)
        row.add_controller(click_controller)
        return row

    def _create_textarea_row(self, var: TextAreaVar, target_property: str):
        row = Adw.ExpanderRow(title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        text_view = Gtk.TextView(
            monospace=True, wrap_mode=Gtk.WrapMode.WORD_CHAR
        )
        scroller = Gtk.ScrolledWindow(
            child=text_view,
            min_content_height=100,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
        )
        row.add_row(scroller)
        initial_val = getattr(var, target_property)
        if initial_val is not None:
            text_view.get_buffer().set_text(str(initial_val))
        row.core_widget = text_view  # type: ignore
        return row
