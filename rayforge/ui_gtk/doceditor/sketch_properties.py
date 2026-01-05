import logging
from typing import Optional, List, TYPE_CHECKING, Any

from gi.repository import Adw, Gtk

from ...core.item import DocItem
from ...core.sketcher import Sketch
from ...core.workpiece import WorkPiece
from ...core.varset import ChoiceVar, SliderFloatVar, TextAreaVar, Var
from ..shared.adwfix import get_spinrow_int
from ..shared.expander import Expander
from ..varset.var_row_factory import VarRowFactory
from ..varset.varsetwidget import NULL_CHOICE_LABEL
from ..sketcher.cmd import UpdateSketchCommand

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class SketchPropertiesWidget(Expander):
    """
    An expander widget that displays and allows editing of input parameters
    for a selected sketch-based WorkPiece. Renders rows directly without a
    nested group for a consistent UI appearance.
    """

    def __init__(self, editor: "DocEditor", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.editor = editor
        self.workpiece: Optional[WorkPiece] = None
        self.sketch: Optional[Sketch] = None
        self._in_update = False

        self.set_title(_("Sketch Parameters"))
        self.set_expanded(True)

        self._list_box = Gtk.ListBox()
        self._list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self.set_child(self._list_box)

        self._factory = VarRowFactory()
        self.widget_map: dict[str, tuple[Adw.PreferencesRow, Var]] = {}

    def set_items(self, items: Optional[List[DocItem]]):
        """
        Sets the items to be displayed. The widget will show if the selection
        is a single, valid sketch-based workpiece with parameters.
        """
        if self.workpiece:
            self.workpiece.updated.disconnect(self._on_workpiece_updated)

        self.workpiece = None
        self.sketch = None

        items = items or []
        if len(items) == 1 and isinstance(items[0], WorkPiece):
            item = items[0]
            if item.sketch_uid:
                sketch_asset = self.editor.doc.get_asset_by_uid(
                    item.sketch_uid
                )
                if isinstance(sketch_asset, Sketch):
                    self.workpiece = item
                    self.sketch = sketch_asset

        has_params = (
            self.sketch
            and self.sketch.input_parameters.vars is not None
            and len(self.sketch.input_parameters.vars) > 0
        )
        if self.workpiece and self.sketch and has_params:
            self.set_visible(True)
            self._populate()
            self.workpiece.updated.connect(self._on_workpiece_updated)
        else:
            self.set_visible(False)
            self._clear()

    def _clear(self):
        """Removes all rows from the list box."""
        child = self._list_box.get_first_child()
        while child:
            self._list_box.remove(child)
            child = self._list_box.get_first_child()
        self.widget_map.clear()

    def _populate(self):
        """Builds rows from the current sketch's VarSet."""
        if not self.sketch:
            return

        # Update title and subtitle from the VarSet
        varset = self.sketch.input_parameters
        self.set_title(varset.title or _("Sketch Parameters"))
        self.set_subtitle(varset.description or "")

        self._clear()

        for var in self.sketch.input_parameters:
            row = self._factory.create_row_for_var(var, "value")
            if row:
                self._wire_up_row(row, var)
                self._list_box.append(row)
                self.widget_map[var.key] = (row, var)

        self._update_ui_from_model()

    def _wire_up_row(self, row: Adw.PreferencesRow, var: Var):
        """Connects signals for the row to trigger an update."""
        widget = getattr(row, "get_activatable_widget", lambda: None)() or row

        if isinstance(row, Adw.EntryRow):
            row.connect("apply", self._on_ui_value_changed)
        elif isinstance(row, Adw.SpinRow):
            row.connect("notify::value", self._on_ui_value_changed)
        elif isinstance(row, Adw.ComboRow):
            row.connect("notify::selected-item", self._on_ui_value_changed)
        elif isinstance(widget, Gtk.Switch):
            widget.connect("state-set", self._on_ui_value_changed)
        elif isinstance(widget, Gtk.Scale):
            widget.connect("value-changed", self._on_ui_value_changed)

    def _on_ui_value_changed(self, widget, *args):
        """
        Called when the user changes a value. This triggers an undoable
        command to update the sketch with all current UI values.
        """
        if self._in_update or not self.sketch or not self.workpiece:
            return

        new_ui_values = self._get_values_from_ui()

        sketch_dict = self.sketch.to_dict(include_input_values=False)
        for var_data in sketch_dict["input_parameters"]["vars"]:
            if var_data["key"] in new_ui_values:
                var_data["value"] = new_ui_values[var_data["key"]]

        cmd = UpdateSketchCommand(
            doc=self.editor.doc,
            sketch_uid=self.sketch.uid,
            new_sketch_dict=sketch_dict,
            name=_("Change Sketch Parameter"),
        )
        self.editor.history_manager.execute(cmd)

    def _on_workpiece_updated(self, workpiece: WorkPiece):
        """Handles data changes from the WorkPiece model (e.g., undo/redo)."""
        logger.debug("Workpiece updated, refreshing sketch properties UI.")
        # Re-fetch sketch as it might have been replaced by undo/redo
        if workpiece.sketch_uid:
            sketch_asset = self.editor.doc.get_asset_by_uid(
                workpiece.sketch_uid
            )
            if isinstance(sketch_asset, Sketch):
                # If the parameters definition changed, we need a full rebuild
                if (
                    self.sketch is None
                    or self.sketch.input_parameters.keys()
                    != sketch_asset.input_parameters.keys()
                ):
                    self.sketch = sketch_asset
                    self._populate()
                else:  # Otherwise, just sync the values
                    self.sketch = sketch_asset
                    self._update_ui_from_model()
                return

        # If we fall through, it's no longer a valid sketch workpiece
        self.sketch = None
        self.set_visible(False)
        self._clear()

    def _update_ui_from_model(self):
        """Populates the UI widgets from the current sketch model values."""
        if not self.sketch or not self.sketch.input_parameters:
            return

        self._in_update = True
        try:
            values = self.sketch.input_parameters.get_values()
            self._set_values_in_ui(values)
        finally:
            self._in_update = False

    def _get_values_from_ui(self) -> dict[str, Any]:
        """Reads all current values from the UI widgets."""
        values = {}
        for key, (row, var) in self.widget_map.items():
            value = None
            widget = getattr(row, "get_activatable_widget", lambda: None)()
            if isinstance(var, TextAreaVar):
                text_view = getattr(row, "core_widget", None)
                if isinstance(text_view, Gtk.TextView):
                    buffer = text_view.get_buffer()
                    start, end = buffer.get_start_iter(), buffer.get_end_iter()
                    value = buffer.get_text(start, end, True)
            elif isinstance(var, SliderFloatVar) and isinstance(
                widget, Gtk.Scale
            ):
                min_val = var.min_val if var.min_val is not None else 0.0
                max_val = var.max_val if var.max_val is not None else 1.0
                percent = widget.get_value() / 100.0
                value = min_val + percent * (max_val - min_val)
            elif isinstance(widget, Gtk.Switch):
                value = widget.get_active()
            elif isinstance(row, Adw.EntryRow):
                value = row.get_text()
            elif isinstance(row, Adw.SpinRow):
                value = (
                    get_spinrow_int(row)
                    if var.var_type is int
                    else row.get_value()
                )
            elif isinstance(row, Adw.ComboRow):
                selected = row.get_selected_item()
                display_str = ""
                if selected:
                    display_str = selected.get_string()  # type: ignore

                if display_str == NULL_CHOICE_LABEL:
                    value = None
                elif isinstance(var, ChoiceVar):
                    value = var.get_value_for_display(display_str)
                else:
                    value = display_str

            values[key] = value
        return values

    def _set_values_in_ui(self, values: dict[str, Any]):
        """Sets the UI widgets from a dictionary of values."""
        for key, value in values.items():
            if key not in self.widget_map or value is None:
                continue

            row, var = self.widget_map[key]
            widget = getattr(row, "get_activatable_widget", lambda: None)()
            if isinstance(var, TextAreaVar):
                text_view = getattr(row, "core_widget", None)
                if isinstance(text_view, Gtk.TextView):
                    buffer = text_view.get_buffer()
                    if buffer.get_text(
                        buffer.get_start_iter(), buffer.get_end_iter(), True
                    ) != str(value):
                        buffer.set_text(str(value))
            elif isinstance(var, SliderFloatVar) and isinstance(
                widget, Gtk.Scale
            ):
                min_val = var.min_val if var.min_val is not None else 0.0
                max_val = var.max_val if var.max_val is not None else 1.0
                range_size = max_val - min_val
                percent = 0.0
                if range_size > 1e-9:
                    percent = ((float(value) - min_val) / range_size) * 100.0
                if abs(widget.get_value() - percent) > 1e-6:
                    widget.set_value(percent)
            elif isinstance(widget, Gtk.Switch):
                if widget.get_active() != bool(value):
                    widget.set_active(bool(value))
            elif isinstance(row, Adw.EntryRow):
                if row.get_text() != str(value):
                    row.set_text(str(value))
            elif isinstance(row, Adw.SpinRow):
                if abs(row.get_value() - float(value)) > 1e-6:
                    row.set_value(float(value))
            elif isinstance(row, Adw.ComboRow):
                model = row.get_model()
                if isinstance(model, Gtk.StringList):
                    display_str = NULL_CHOICE_LABEL
                    if value is not None:
                        display_str = (
                            var.get_display_for_value(str(value)) or str(value)
                            if isinstance(var, ChoiceVar)
                            else str(value)
                        )
                    for i in range(model.get_n_items()):
                        if model.get_string(i) == display_str:
                            if row.get_selected() != i:
                                row.set_selected(i)
                            break
