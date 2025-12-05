import logging
from typing import List, cast, Any, Set, TYPE_CHECKING

from gi.repository import Adw, Gtk, GLib

from ....core.item import DocItem
from ....core.sketcher.sketch import Sketch
from ....core.varset import VarSet
from ....core.workpiece import WorkPiece
from ....shared.ui.varsetwidget import VarSetWidget
from .base import PropertyProvider

if TYPE_CHECKING:
    from ...editor import DocEditor

logger = logging.getLogger(__name__)

DEBOUNCE_DELAY_MS = 300


class SketchPropertyProvider(PropertyProvider):
    """Provides a VarSetWidget to configure a Sketch's input parameters."""

    def can_handle(self, items: List[DocItem]) -> bool:
        """
        Handles the selection if all items are WorkPieces derived from the
        *same* sketch definition.
        """
        if not items:
            return False

        first_sketch_uid = None
        for item in items:
            if not isinstance(item, WorkPiece) or not item.sketch_uid:
                return False  # All items must be sketch-based workpieces

            if first_sketch_uid is None:
                first_sketch_uid = item.sketch_uid
            elif item.sketch_uid != first_sketch_uid:
                return False  # All workpieces must share the same sketch UID

        return first_sketch_uid is not None

    def create_widgets(self) -> List[Gtk.Widget]:
        """Creates the VarSetWidget for sketch parameters."""
        # The VarSetWidget dynamically creates its own rows, so we create
        # it here but leave it empty. It will be populated in update_widgets.
        self.varset_widget = VarSetWidget(title=_("Sketch Parameters"))
        self.varset_widget.data_changed.connect(self._on_params_changed)

        # State for debouncing user input
        self._debounce_timer_id = 0
        self._pending_changes: dict[str, Any] = {}

        return [self.varset_widget]

    def update_widgets(self, editor: "DocEditor", items: List[DocItem]):
        """Populates and updates the VarSetWidget based on the selection."""
        self._in_update = True
        try:
            self.editor = editor
            self.items = items
            workpieces = cast(List[WorkPiece], self.items)
            first_wp = workpieces[0]

            sketch = first_wp.get_sketch_definition()
            if not sketch or not sketch.input_parameters:
                # To clear the widget, populate it with an empty VarSet.
                self.varset_widget.populate(VarSet())
                return

            # 1. (Re)populate the VarSetWidget from the sketch definition.
            # This builds the UI with the default values for each parameter.
            self.varset_widget.populate(sketch.input_parameters)

            # 2. Update the UI to reflect the actual values from the selection,
            # handling mixed values appropriately.
            self._update_widget_for_mixed_state(sketch, workpieces)
        finally:
            self._in_update = False

    def _update_widget_for_mixed_state(
        self, sketch: Sketch, workpieces: List[WorkPiece]
    ):
        """
        Adjusts the UI controls to show common values or indicate a
        mixed state when multiple items are selected.
        """
        if len(workpieces) == 1:
            # For a single selection, just set the current values.
            self.varset_widget.set_values(workpieces[0].sketch_params)
            return

        # For multiple selections, check each parameter for mixed values.
        for key, (row, var) in self.varset_widget.widget_map.items():
            all_values: Set[Any] = set()
            for wp in workpieces:
                # Get the instance's value, falling back to the sketch default.
                value = wp.sketch_params.get(key, var.default)
                all_values.add(value)

            if len(all_values) == 1:
                # All workpieces have the same override value for this key.
                common_value = all_values.pop()
                self.varset_widget.set_values({key: common_value})
            else:
                # Values are mixed. Update UI to reflect this.
                if isinstance(row, Adw.EntryRow):
                    row.set_text("")
                elif isinstance(row, Adw.SpinRow):
                    row.set_subtitle(_("Mixed Values"))
                elif isinstance(row, Adw.ComboRow):
                    row.set_selected(0)  # "None Selected"
                    row.set_subtitle(_("Mixed Values"))
                elif isinstance(
                    getattr(row, "get_activatable_widget", lambda: None)(),
                    Gtk.Switch,
                ):
                    row.set_sensitive(False)
                    if isinstance(row, Adw.ActionRow):
                        row.set_subtitle(_("Mixed Values"))

    def _on_params_changed(self, sender: VarSetWidget, key: str):
        """
        Handles the raw signal from the VarSetWidget. Instead of applying
        changes immediately, it schedules a debounced update.
        """
        if self._in_update or not self.items:
            return

        # Cancel any previously scheduled update
        if self._debounce_timer_id > 0:
            GLib.source_remove(self._debounce_timer_id)
            self._debounce_timer_id = 0

        # Store the latest value for the changed key
        all_values = sender.get_values()
        self._pending_changes[key] = all_values.get(key)

        # Schedule the actual update to happen after a short delay
        self._debounce_timer_id = GLib.timeout_add(
            DEBOUNCE_DELAY_MS, self._apply_debounced_changes
        )

    def _apply_debounced_changes(self) -> bool:
        """
        Applies the collected changes to the model. This is called by the
        GLib timer after the user has paused input.
        """
        if not self._pending_changes or not self.items:
            self._debounce_timer_id = 0
            self._pending_changes.clear()
            return GLib.SOURCE_REMOVE  # Stop the timer

        # Make a copy of the changes and clear the pending state
        changes_to_apply = self._pending_changes.copy()
        self._pending_changes.clear()
        self._debounce_timer_id = 0

        logger.debug(
            f"Applying debounced sketch params to {len(self.items)} "
            f"items: {changes_to_apply}"
        )

        workpieces = cast(List[WorkPiece], self.items)
        self.editor.sketch.set_workpiece_parameters(
            workpieces, changes_to_apply
        )

        # Returning GLib.SOURCE_REMOVE ensures the timer does not repeat
        return GLib.SOURCE_REMOVE
