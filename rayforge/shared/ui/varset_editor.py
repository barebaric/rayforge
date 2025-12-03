import logging
from typing import Optional, Iterable, Type, Any, TYPE_CHECKING
from gi.repository import Gtk, Adw, Gdk
from blinker import Signal
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
from ...icons import get_icon
from .preferences_group import PreferencesGroupWithButton
from ...undo.models.property_cmd import ChangePropertyCommand
from .var_row_factory import VarRowFactory, NULL_CHOICE_LABEL

if TYPE_CHECKING:
    from ...undo import HistoryManager

logger = logging.getLogger(__name__)


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

        # Define signals as INSTANCE attributes for proper scoping
        self.delete_clicked = Signal()
        self.reorder_requested = Signal()

        self._update_header()

        # --- Prefix Area (Drag Handle & Delete) ---
        prefix_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        prefix_box.set_margin_end(8)

        # Drag Handle
        drag_handle = Gtk.Image(
            icon_name="list-drag-handle-symbolic",
            tooltip_text=_("Drag to reorder"),
        )
        drag_handle.add_css_class("dim-label")
        drag_handle.set_cursor(Gdk.Cursor.new_from_name("grab", None))
        prefix_box.append(drag_handle)

        # Delete Button
        delete_button = Gtk.Button(
            icon_name="user-trash-symbolic",
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

        self.key_row = Adw.EntryRow(title=_("Key"))
        self.key_row.set_text(self.var.key)
        self.key_row.connect("changed", self._on_key_changed)
        self.add_row(self.key_row)

        self.label_row = Adw.EntryRow(title=_("Label"))
        if self.var.label:
            self.label_row.set_text(self.var.label)
        self.label_row.connect("changed", self._on_label_changed)
        self.add_row(self.label_row)

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

    def _wire_up_default_row(self):
        widget = getattr(
            self.default_row, "get_activatable_widget", lambda: None
        )()
        widget = widget or self.default_row

        if isinstance(self.default_row, Adw.EntryRow):
            # Use 'changed' for entry rows if supported
            self.default_row.connect("changed", self._on_default_changed_entry)
        elif isinstance(self.default_row, Adw.SpinRow):
            self.default_row.connect(
                "changed", self._on_default_changed_spinrow
            )
        elif isinstance(self.default_row, Adw.ComboRow):
            self.default_row.connect(
                "notify::selected-item", self._on_default_changed_combo
            )
        elif isinstance(widget, Gtk.Switch):
            widget.connect("state-set", self._on_default_changed_switch)
        elif isinstance(widget, Gtk.Scale):
            widget.connect("value-changed", self._on_default_changed_scale)

    def _validate_and_set_default(self, value: Any, widget: Gtk.Widget):
        """
        Attempts to set the default value. If a validator exists and fails,
        marks the widget with the 'error' style class.
        """
        try:
            # 1. Run validator if it exists
            if self.var.validator:
                self.var.validator(value)

            # 2. Update via UndoManager or directly.
            if self.undo_manager:
                cmd = ChangePropertyCommand(self.var, "default", value)
                self.undo_manager.execute(cmd)
            else:
                self.var.default = value

            # 3. Clear error style
            widget.remove_css_class("error")

        except Exception as e:
            logger.warning(f"Validation failed for default value: {e}")
            widget.add_css_class("error")

    # --- Sync Callbacks for Undo/Redo ---
    def _sync_key(self):
        if self.key_row.get_text() != self.var.key:
            self.key_row.set_text(self.var.key)
        self._update_header()

    def _sync_label(self):
        val = self.var.label or ""
        if self.label_row.get_text() != val:
            self.label_row.set_text(val)
        self._update_header()

    def _sync_desc(self):
        val = self.var.description or ""
        if self.desc_row.get_text() != val:
            self.desc_row.set_text(val)

    # --- Change Handlers ---

    def _on_key_changed(self, entry_row: Adw.EntryRow):
        new_val = entry_row.get_text()
        if self.var.key == new_val:
            return

        if self.undo_manager:
            cmd = ChangePropertyCommand(
                self.var, "key", new_val, on_change_callback=self._sync_key
            )
            self.undo_manager.execute(cmd)
        else:
            self.var.key = new_val
            self._sync_key()

    def _on_label_changed(self, entry_row: Adw.EntryRow):
        new_val = entry_row.get_text()
        if self.var.label == new_val:
            return

        if self.undo_manager:
            cmd = ChangePropertyCommand(
                self.var, "label", new_val, on_change_callback=self._sync_label
            )
            self.undo_manager.execute(cmd)
        else:
            self.var.label = new_val
            self._update_header()

    def _on_description_changed(self, entry_row: Adw.EntryRow):
        new_val = entry_row.get_text()
        if self.var.description == new_val:
            return

        if self.undo_manager:
            cmd = ChangePropertyCommand(
                self.var,
                "description",
                new_val,
                on_change_callback=self._sync_desc,
            )
            self.undo_manager.execute(cmd)
        else:
            self.var.description = new_val

    def _on_default_changed_entry(self, entry_row: Adw.EntryRow):
        self._validate_and_set_default(entry_row.get_text(), entry_row)

    def _on_default_changed_spinrow(self, spin_row: Adw.SpinRow):
        self._validate_and_set_default(spin_row.get_value(), spin_row)

    def _on_default_changed_switch(self, switch: Gtk.Switch, state: bool):
        self._validate_and_set_default(state, self.default_row)

    def _on_default_changed_scale(self, scale: Gtk.Scale):
        val = scale.get_value() / 100.0
        self._validate_and_set_default(val, self.default_row)

    def _on_default_changed_combo(self, combo_row: Adw.ComboRow, _):
        selected = combo_row.get_selected_item()
        display_str = selected.get_string() if selected else ""  # type: ignore

        if display_str == NULL_CHOICE_LABEL:
            val = None
        elif isinstance(self.var, ChoiceVar):
            val = self.var.get_value_for_display(display_str)
        else:
            val = display_str

        self._validate_and_set_default(val, combo_row)


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
        lbl = Gtk.Label()
        lbl.set_markup(_("Add Parameter"))
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
            item,
            self._factory,
            undo_manager=self._undo_manager,
        )
        row_widget.delete_clicked.connect(self._on_delete_var_clicked)
        row_widget.reorder_requested.connect(self._on_reorder_requested)
        return row_widget

    def _on_add_var_activated(self, button, var_class, popover: Gtk.Popover):
        """Handler for when a user selects a Var type to add."""
        self._add_counter += 1
        key = f"new_var_{self._add_counter}"
        label = button.get_label()

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
        vars_in_order = self._var_set.vars
        try:
            target_index = -1
            for i, var in enumerate(vars_in_order):
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
        self.set_items(self._var_set.vars)

    def get_var_set(self) -> VarSet:
        return self._var_set
