import logging
import re
from typing import (
    Any,
    Optional,
    Iterable,
    Type,
    TYPE_CHECKING,
    Literal,
    Tuple,
    cast,
)
from gi.repository import Gtk, Adw, Gdk
from blinker import Signal
from ...core.undo.property_cmd import ChangePropertyCommand
from ...core.varset import (
    Var,
    VarSet,
    IntVar,
    BoolVar,
    FloatVar,
    SliderFloatVar,
    ChoiceVar,
    HostnameVar,
    SerialPortVar,
    TextAreaVar,
)
from ..icons import get_icon
from ..settings.preferences_group import PreferencesGroupWithButton
from .var_row_factory import VarRowFactory, NULL_CHOICE_LABEL

if TYPE_CHECKING:
    from ...core.undo import HistoryManager

logger = logging.getLogger(__name__)


def adjust_value(
    min_val: Optional[float],
    max_val: Optional[float],
    value: float,
    keep: Literal["min", "max", "value"],
) -> Tuple[Optional[float], Optional[float], float]:
    """
    Adjusts min, max, and value to be consistent, keeping one value fixed.
    Returns a tuple of (final_min, final_max, final_value).
    """
    if keep == "value":
        if min_val is not None and value < min_val:
            min_val = value
        if max_val is not None and value > max_val:
            max_val = value
    elif keep == "min":
        if min_val is not None:
            if max_val is not None and min_val > max_val:
                max_val = min_val
            if value < min_val:
                value = min_val
    elif keep == "max":
        if max_val is not None:
            if min_val is not None and max_val < min_val:
                min_val = max_val
            if value > max_val:
                value = max_val
    return min_val, max_val, value


class VarDefinitionRowWidget(Adw.ExpanderRow):
    """
    A widget for displaying and editing the definition of a single Var.
    Supports Drag and Drop reordering and Undo/Redo.
    """

    def __init__(
        self,
        var: Var,
        factory: VarRowFactory,
        undo_manager: Optional["HistoryManager"] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.var = var
        self._factory = factory
        self.undo_manager = undo_manager
        self._in_update = False
        self._updating_key_from_label = False

        # Only auto-update key if it looks like a freshly added parameter.
        # Existing variables (loaded from files) should never auto-update key
        # based on label to prevent breaking references.
        is_temp_key = self.var.key.startswith("new_parameter")
        self._key_manually_edited = not is_temp_key

        # Define signals as INSTANCE attributes for proper scoping
        self.delete_clicked = Signal()
        self.reorder_requested = Signal()

        self._update_header()

        # --- Prefix Area (Drag Handle & Delete) ---
        prefix_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        prefix_box.set_margin_end(8)

        # Drag Handle
        drag_handle = get_icon("drag-handle-symbolic")
        drag_handle.set_tooltip_text(_("Drag to reorder"))
        drag_handle.add_css_class("dim-label")
        drag_handle.set_cursor(Gdk.Cursor.new_from_name("grab", None))
        prefix_box.append(drag_handle)

        # Delete Button
        delete_button = Gtk.Button(
            child=get_icon("delete-symbolic"),
            tooltip_text=_("Delete Variable"),
            valign=Gtk.Align.CENTER,
        )
        delete_button.add_css_class("flat")
        delete_button.connect(
            "clicked", lambda b: self.delete_clicked.send(self)
        )
        prefix_box.append(delete_button)

        self.add_prefix(prefix_box)

        # --- Drag Source Setup (On the Handle) ---
        drag_source = Gtk.DragSource(actions=Gdk.DragAction.MOVE)
        drag_source.connect("prepare", self._on_drag_prepare)
        drag_handle.add_controller(drag_source)

        # --- Drop Target Setup (On the Row) ---
        # We accept strings (the variable key)
        drop_target = Gtk.DropTarget.new(type=str, actions=Gdk.DragAction.MOVE)
        drop_target.connect("drop", self._on_drop)
        self.add_controller(drop_target)

        self._build_content_rows()

    def _update_header(self):
        """Updates the row title and subtitle based on var state."""
        self.set_title(self.var.label)
        self.set_subtitle(f"{self.var.key} ({type(self.var).__name__})")

    def _derive_key_from_label(self, label: str) -> str:
        """Converts a label into a snake_case key."""
        s = label.lower().strip().replace(" ", "_")
        return re.sub(r"[^a-z0-9_]", "", s)

    def _on_drag_prepare(self, source, x, y):
        """
        Called when dragging starts. Returns content provider with the key.
        """
        return Gdk.ContentProvider.new_for_value(self.var.key)

    def _on_drop(self, target, value, x, y):
        """Called when something is dropped onto this row."""
        if isinstance(value, str) and value != self.var.key:
            # Emit signal: moved 'value' (key) to 'self.var.key' (target)
            self.reorder_requested.send(
                self, source_key=value, target_key=self.var.key
            )
            return True
        return False

    def _build_content_rows(self):
        """Creates and wires up the editor rows for the Var's properties."""

        # 1. Label Row (First)
        self.label_row = Adw.EntryRow(title=_("Label"))
        if self.var.label:
            self.label_row.set_text(self.var.label)
        self.label_row.connect("changed", self._on_label_changed)
        self.add_row(self.label_row)

        # 2. Key Row (Second, derived from Label unless edited)
        self.key_row = Adw.EntryRow(title=_("Key"))
        self.key_row.set_text(self.var.key)
        self.key_row.connect("changed", self._on_key_changed)
        self.add_row(self.key_row)

        self.desc_row = Adw.EntryRow(title=_("Description"))
        if self.var.description:
            self.desc_row.set_tooltip_text(self.var.description)
            self.desc_row.set_text(self.var.description)
        self.desc_row.connect("changed", self._on_description_changed)
        self.add_row(self.desc_row)

        self.default_row = self._factory.create_row_for_var(
            self.var, target_property="default"
        )
        self.default_row.set_title(_("Default Value"))
        self._wire_up_default_row()
        self.add_row(self.default_row)

        if isinstance(self.var, (IntVar, FloatVar)):
            is_slider = isinstance(self.var, SliderFloatVar)
            bound_var_instance = (
                FloatVar(key=self.var.key, label=self.var.label)
                if is_slider
                else self.var
            )
            default_val = (
                self.var.default if self.var.default is not None else 0
            )

            # --- Minimum / Start Value Row ---
            self.min_val_row = self._factory.create_row_for_var(
                bound_var_instance, "min_val"
            )
            if isinstance(self.min_val_row, Adw.SpinRow):
                self.min_val_row.set_title(
                    _("Start Value") if is_slider else _("Minimum Value")
                )

                if not is_slider:
                    self.min_toggle = Gtk.Switch(valign=Gtk.Align.CENTER)
                    self.min_toggle.connect(
                        "state-set",
                        self._on_bound_toggle,
                        self.min_val_row,
                        "min_val",
                    )
                    self.min_val_row.add_prefix(self.min_toggle)
                    has_min = self.var.min_val is not None
                    self.min_toggle.set_active(has_min)
                    self.min_val_row.set_editable(has_min)
                    if not has_min:
                        self.min_val_row.set_value(default_val)
                else:
                    # Sliders always have active bounds, no toggle needed
                    self.min_val_row.set_editable(True)
                    if self.var.min_val is None:
                        self.min_val_row.set_value(0.0)

                self._wire_up_bound_row(self.min_val_row, "min_val")
                self.add_row(self.min_val_row)

            # --- Maximum / End Value Row ---
            self.max_val_row = self._factory.create_row_for_var(
                bound_var_instance, "max_val"
            )
            if isinstance(self.max_val_row, Adw.SpinRow):
                self.max_val_row.set_title(
                    _("End Value") if is_slider else _("Maximum Value")
                )

                if not is_slider:
                    self.max_toggle = Gtk.Switch(valign=Gtk.Align.CENTER)
                    self.max_toggle.connect(
                        "state-set",
                        self._on_bound_toggle,
                        self.max_val_row,
                        "max_val",
                    )
                    self.max_val_row.add_prefix(self.max_toggle)
                    has_max = self.var.max_val is not None
                    self.max_toggle.set_active(has_max)
                    self.max_val_row.set_editable(has_max)
                    if not has_max:
                        min_val_from_ui = (
                            self.min_val_row.get_value()
                            if hasattr(self, "min_val_row")
                            and isinstance(self.min_val_row, Adw.SpinRow)
                            and self.min_val_row.get_editable()
                            else default_val
                        )
                        self.max_val_row.set_value(
                            max(default_val, min_val_from_ui)
                        )
                else:
                    self.max_val_row.set_editable(True)
                    if self.var.max_val is None:
                        self.max_val_row.set_value(1.0)

                self._wire_up_bound_row(self.max_val_row, "max_val")
                self.add_row(self.max_val_row)

    def _wire_up_default_row(self):
        widget = getattr(
            self.default_row, "get_activatable_widget", lambda: None
        )()
        widget = widget or self.default_row

        if isinstance(self.default_row, Adw.SpinRow):
            self.default_row.connect(
                "notify::value", self._on_default_changed_spinrow
            )
        elif isinstance(self.default_row, Adw.EntryRow):
            self.default_row.connect("changed", self._on_default_changed_entry)
        elif isinstance(self.default_row, Adw.ComboRow):
            self.default_row.connect(
                "notify::selected-item", self._on_default_changed_combo
            )
        elif isinstance(widget, Gtk.Switch):
            widget.connect("state-set", self._on_default_changed_switch)
        elif isinstance(widget, Gtk.Scale):
            widget.connect("value-changed", self._on_default_changed_scale)

    def _wire_up_bound_row(self, row: Adw.PreferencesRow, property_name: str):
        if isinstance(row, Adw.SpinRow):
            row.connect(
                "notify::value", self._on_bound_changed_spinrow, property_name
            )

    def _sync_prop(self, row, prop_name, sync_header=False):
        val = getattr(self.var, prop_name) or ""
        if row.get_text() != val:
            row.set_text(val)
        if sync_header:
            self._update_header()

    def _sync_bound(self, row, toggle: Optional[Gtk.Switch], prop_name):
        if not isinstance(self.var, (IntVar, FloatVar)) or not isinstance(
            row, Adw.SpinRow
        ):
            return
        self._in_update = True
        try:
            val = getattr(self.var, prop_name)
            has_val = val is not None

            if toggle:
                if toggle.get_active() != has_val:
                    toggle.set_active(has_val)
                row.set_editable(has_val)
            else:
                row.set_editable(True)

            if has_val and row.get_value() != val:
                row.set_value(val)
        finally:
            self._in_update = False

    def _sync_default(self):
        val = self.var.default
        row = self.default_row
        widget = getattr(row, "get_activatable_widget", lambda: None)()

        # Prevent signal recursion during sync
        self._in_update = True
        try:
            if isinstance(row, Adw.SpinRow):
                if row.get_value() != val:
                    row.set_value(val if val is not None else 0)
            elif isinstance(row, Adw.EntryRow):
                text_val = str(val or "")
                if row.get_text() != text_val:
                    row.set_text(text_val)
            elif isinstance(widget, Gtk.Switch):
                active_val = bool(val)
                if widget.get_active() != active_val:
                    widget.set_active(active_val)
            elif isinstance(widget, Gtk.Scale) and isinstance(
                self.var, SliderFloatVar
            ):
                min_val = (
                    self.var.min_val if self.var.min_val is not None else 0.0
                )
                max_val = (
                    self.var.max_val if self.var.max_val is not None else 1.0
                )
                range_size = max_val - min_val
                percent = 0.0
                if val is not None and range_size > 1e-9:
                    percent = ((float(val) - min_val) / range_size) * 100.0
                if abs(widget.get_value() - percent) > 1e-6:
                    widget.set_value(percent)
            elif isinstance(row, Adw.ComboRow):
                model = row.get_model()
                if isinstance(model, Gtk.StringList):
                    display_str = NULL_CHOICE_LABEL
                    if val is not None:
                        display_str = (
                            self.var.get_display_for_value(str(val))
                            or str(val)
                            if isinstance(self.var, ChoiceVar)
                            else str(val)
                        )
                    for i in range(model.get_n_items()):
                        if model.get_string(i) == display_str:
                            if row.get_selected() != i:
                                row.set_selected(i)
                            break
        finally:
            self._in_update = False

    def _on_change_generic(self, row, prop, sync_header=False):
        new_val = row.get_text()
        if getattr(self.var, prop) == new_val:
            return

        def sync_callback():
            self._sync_prop(row, prop, sync_header)

        if self.undo_manager:
            cmd = ChangePropertyCommand(
                self.var, prop, new_val, on_change_callback=sync_callback
            )
            self.undo_manager.execute(cmd)
        else:
            setattr(self.var, prop, new_val)
            sync_callback()

    def _on_key_changed(self, row):
        if not self._updating_key_from_label:
            # User manually typed in the key row; stop auto-updates
            self._key_manually_edited = True

        self._on_change_generic(row, "key", sync_header=True)

    def _on_label_changed(self, row):
        # Update the actual Label property
        self._on_change_generic(row, "label", sync_header=True)

        # If not manually edited, auto-update the Key
        if not self._key_manually_edited:
            new_key = self._derive_key_from_label(row.get_text())

            # Use flag to prevent _on_key_changed from marking this as manual
            self._updating_key_from_label = True
            self.key_row.set_text(new_key)
            self._updating_key_from_label = False

    def _on_description_changed(self, row):
        self._on_change_generic(row, "description")

    def _commit_property_change(self, prop_name: str, new_val: Any):
        if (
            not isinstance(self.var, (IntVar, FloatVar))
            and prop_name != "default"
        ):
            return

        def sync_callback():
            if prop_name == "default":
                self._sync_default()
            else:
                row = getattr(self, f"{prop_name}_row")
                toggle = None
                if prop_name == "min_val":
                    toggle = getattr(self, "min_toggle", None)
                elif prop_name == "max_val":
                    toggle = getattr(self, "max_toggle", None)

                self._sync_bound(row, toggle, prop_name)

        if self.undo_manager:
            cmd = ChangePropertyCommand(
                self.var, prop_name, new_val, on_change_callback=sync_callback
            )
            self.undo_manager.execute(cmd)
        else:
            setattr(self.var, prop_name, new_val)
            sync_callback()

    def _on_bound_toggle(
        self,
        switch: Gtk.Switch,
        state: bool,
        spin_row: Adw.SpinRow,
        prop_name: str,
    ):
        if self._in_update:
            return False
        spin_row.set_editable(state)
        new_val = spin_row.get_value() if state else None
        self._commit_property_change(prop_name, new_val)
        if state:
            self._on_bound_changed_spinrow(spin_row, None, prop_name)
        return False

    def _commit_numeric_changes(
        self,
        default: Optional[float],
        min_val: Optional[float],
        max_val: Optional[float],
        keep: Literal["min", "max", "value"],
    ):
        if (
            not isinstance(self.var, (IntVar, FloatVar))
            or not self.undo_manager
        ):
            return

        with self.undo_manager.transaction(_("Adjust Value")):
            final_min, final_max, final_default = adjust_value(
                min_val, max_val, default or 0.0, keep
            )
            if self.var.default != final_default:
                self._commit_property_change("default", final_default)
            if self.var.min_val != final_min:
                self._commit_property_change("min_val", final_min)
            if self.var.max_val != final_max:
                self._commit_property_change("max_val", final_max)

    def _on_bound_changed_spinrow(self, spin_row, _pspec, prop_name):
        if self._in_update:
            return
        self._in_update = True
        try:
            new_bound_val = spin_row.get_value()

            if isinstance(self.var, SliderFloatVar):
                # For sliders, we want to keep the relative percentage fixed
                # rather than the absolute value when bounds change.
                self._update_slider_bounds(prop_name, new_bound_val)
            else:
                default_val = (
                    self.default_row.get_value()
                    if isinstance(self.default_row, Adw.SpinRow)
                    else 0.0
                )
                min_val = (
                    self.min_val_row.get_value()
                    if hasattr(self, "min_val_row")
                    and isinstance(self.min_val_row, Adw.SpinRow)
                    and self.min_val_row.get_editable()
                    else None
                )
                max_val = (
                    self.max_val_row.get_value()
                    if hasattr(self, "max_val_row")
                    and isinstance(self.max_val_row, Adw.SpinRow)
                    and self.max_val_row.get_editable()
                    else None
                )

                if prop_name == "min_val":
                    self._commit_numeric_changes(
                        default_val, new_bound_val, max_val, keep="min"
                    )
                else:
                    self._commit_numeric_changes(
                        default_val, min_val, new_bound_val, keep="max"
                    )
        finally:
            self._in_update = False

    def _update_slider_bounds(self, prop_name: str, new_val: float):
        """
        Special logic for SliderFloatVar: changing bounds keeps the slider's
        relative position (percentage) constant, recalculating default value.
        """
        scale_widget = getattr(
            self.default_row, "get_activatable_widget", lambda: None
        )()
        if not isinstance(scale_widget, Gtk.Scale):
            return

        var = cast(SliderFloatVar, self.var)

        # 1. Get current percentage [0..1]
        percent = scale_widget.get_value() / 100.0

        # 2. Determine new bounds
        min_val = var.min_val if var.min_val is not None else 0.0
        max_val = var.max_val if var.max_val is not None else 1.0

        if prop_name == "min_val":
            min_val = new_val
        else:
            max_val = new_val

        # 3. Calculate new default to preserve percentage
        new_default = min_val + percent * (max_val - min_val)

        # 4. Commit changes
        def apply():
            if prop_name == "min_val":
                self._commit_property_change("min_val", min_val)
            else:
                self._commit_property_change("max_val", max_val)
            self._commit_property_change("default", new_default)

        if self.undo_manager:
            with self.undo_manager.transaction(_("Adjust Slider Range")):
                apply()
        else:
            apply()

    def _on_default_changed_spinrow(self, spin_row: Adw.SpinRow, _pspec):
        if self._in_update:
            return
        self._in_update = True
        try:
            new_default = spin_row.get_value()
            min_val = (
                self.min_val_row.get_value()
                if hasattr(self, "min_val_row")
                and isinstance(self.min_val_row, Adw.SpinRow)
                and self.min_val_row.get_editable()
                else None
            )
            max_val = (
                self.max_val_row.get_value()
                if hasattr(self, "max_val_row")
                and isinstance(self.max_val_row, Adw.SpinRow)
                and self.max_val_row.get_editable()
                else None
            )
            self._commit_numeric_changes(
                new_default, min_val, max_val, keep="value"
            )
        finally:
            self._in_update = False

    def _on_default_changed_entry(self, entry_row: Adw.EntryRow):
        self._commit_property_change("default", entry_row.get_text())

    def _on_default_changed_switch(self, switch: Gtk.Switch, state: bool):
        self._commit_property_change("default", state)

    def _on_default_changed_scale(self, scale: Gtk.Scale, _pspec=None):
        if self._in_update or not isinstance(self.var, SliderFloatVar):
            return
        self._in_update = True
        try:
            # For sliders, min_val and max_val are assumed valid/present
            min_val = self.var.min_val if self.var.min_val is not None else 0.0
            max_val = self.var.max_val if self.var.max_val is not None else 1.0

            percent = scale.get_value() / 100.0
            new_default = min_val + percent * (max_val - min_val)
            self._commit_numeric_changes(
                new_default, min_val, max_val, keep="value"
            )
        finally:
            self._in_update = False

    def _on_default_changed_combo(self, combo_row: Adw.ComboRow, _pspec):
        selected = combo_row.get_selected_item()
        display_str = selected.get_string() if selected else ""  # type: ignore
        val = (
            self.var.get_value_for_display(display_str)
            if isinstance(self.var, ChoiceVar)
            and display_str != NULL_CHOICE_LABEL
            else (None if display_str == NULL_CHOICE_LABEL else display_str)
        )
        self._commit_property_change("default", val)


class VarSetEditorWidget(PreferencesGroupWithButton):
    """
    A widget for interactively defining a VarSet, styled to integrate
    seamlessly with an 'Add' button at the bottom.
    """

    # Define available types and their labels
    _ALL_VAR_TYPES = [
        (_("Integer"), IntVar),
        (_("Boolean (Switch)"), BoolVar),
        (_("Floating Point"), FloatVar),
        (_("Slider (0-100%)"), SliderFloatVar),
        (_("Text (Single Line)"), Var),
        (_("Text (Multi-Line)"), TextAreaVar),
        (_("Choice"), ChoiceVar),
        (_("Hostname / IP"), HostnameVar),
        (_("Serial Port"), SerialPortVar),
    ]

    def __init__(
        self,
        vartypes: Optional[Iterable[Type[Var]]] = None,
        undo_manager: Optional["HistoryManager"] = None,
        **kwargs,
    ):
        """
        Args:
            vartypes: A set or list of Var classes allowed to be added.
            undo_manager: Optional HistoryManager for undo/redo support.
        """
        self._allowed_types = set(vartypes) if vartypes else None
        self._undo_manager = undo_manager

        # Pass a dummy label; we override the button creation entirely.
        super().__init__(button_label="", **kwargs)
        self._var_set = VarSet()
        self._factory = VarRowFactory()
        self._add_counter = 0

    @property
    def undo_manager(self) -> Optional["HistoryManager"]:
        return self._undo_manager

    @undo_manager.setter
    def undo_manager(self, value: Optional["HistoryManager"]):
        self._undo_manager = value
        # Propagate to existing rows
        i = 0
        while row := self.list_box.get_row_at_index(i):
            widget = row.get_child()
            if isinstance(widget, VarDefinitionRowWidget):
                widget.undo_manager = value
            i += 1

    def _create_add_button(self, button_label: str) -> Gtk.Widget:
        """Overrides the base class to create a Gtk.MenuButton."""
        add_button = Gtk.MenuButton()

        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        button_box.set_margin_top(10)
        button_box.set_margin_end(12)
        button_box.set_margin_bottom(10)
        button_box.set_margin_start(12)
        button_box.append(get_icon("add-symbolic"))
        lbl = Gtk.Label(label=_("Add Parameter"))
        button_box.append(lbl)
        add_button.set_child(button_box)

        menu = Gtk.PopoverMenu()
        add_button.set_popover(menu)

        # Create a box to hold the menu item buttons
        menu_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        menu.set_child(menu_box)

        for label, var_class in self._ALL_VAR_TYPES:
            # Filter types if a restriction list is provided
            if self._allowed_types and var_class not in self._allowed_types:
                continue

            item_button = Gtk.Button(label=label)
            item_button.add_css_class("flat")
            item_button.set_halign(Gtk.Align.FILL)
            item_button.connect(
                "clicked", self._on_add_var_activated, var_class, menu
            )
            menu_box.append(item_button)

        return add_button

    def create_row_widget(self, item: Var) -> Gtk.Widget:
        row_widget = VarDefinitionRowWidget(
            item, self._factory, undo_manager=self._undo_manager
        )
        row_widget.delete_clicked.connect(self._on_delete_var_clicked)
        row_widget.reorder_requested.connect(self._on_reorder_requested)
        return row_widget

    def _on_add_var_activated(self, button, var_class, popover: Gtk.Popover):
        """Handler for when a user selects a Var type to add."""
        self._add_counter += 1
        # Create a placeholder label/key that matches the validation rules
        label = "New Parameter"
        key = "new_parameter"

        new_var: Var
        if var_class is ChoiceVar:
            new_var = var_class(
                key=key, label=label, choices=["Option 1", "Option 2"]
            )
        elif var_class is Var:
            new_var = var_class(key=key, label=label, var_type=str)
        elif var_class is FloatVar:
            # Use a non-zero default to avoid geometry collapse on assignment
            new_var = var_class(key=key, label=label, default=10.0)
        elif var_class is IntVar:
            # Use a non-zero default to avoid geometry collapse on assignment
            new_var = var_class(key=key, label=label, default=10)
        else:
            new_var = var_class(key=key, label=label)

        base_key = key
        counter = 1
        while base_key in self._var_set.get_values():
            base_key = f"{key}_{counter}"
            counter += 1
        new_var.key = base_key

        self._var_set.add(new_var)
        self.populate(self._var_set)

        popover.popdown()

    def _on_delete_var_clicked(self, sender: VarDefinitionRowWidget):
        """Handler for when a row's delete button is clicked."""
        var_to_delete = sender.var
        self._var_set.remove(var_to_delete.key)
        self.populate(self._var_set)

    def _on_reorder_requested(self, sender, source_key: str, target_key: str):
        """
        Handler for when a row is dropped onto another row.
        Reorders the VarSet and refreshes the UI.
        """
        # Determine current index of target
        try:
            target_index = -1
            for i, var in enumerate(self._var_set.vars):
                if var.key == target_key:
                    target_index = i
                    break

            if target_index != -1:
                self._var_set.move_var(source_key, target_index)
                self.populate(self._var_set)
        except Exception as e:
            logger.error(f"Failed to reorder vars: {e}")

    def populate(self, var_set: VarSet):
        """Populates the editor with an existing VarSet."""
        self._var_set = var_set
        if var_set.title:
            self.set_title(var_set.title)
        if var_set.description:
            self.set_description(var_set.description)
        self.set_items(self._var_set.vars)

    def get_var_set(self) -> VarSet:
        return self._var_set
