import logging
from typing import List, cast, Any, Set, TYPE_CHECKING

from gi.repository import Adw, Gtk, GLib

from ....core.item import DocItem
from ....core.varset import VarSet
from ....core.workpiece import WorkPiece
from ...varset.varsetwidget import VarSetWidget
from .base import PropertyProvider

if TYPE_CHECKING:
    from ....doceditor.editor import DocEditor

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
        logger.debug("Creating sketch property widgets.")
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
        logger.debug(
            f"Updating sketch property widgets for {len(items)} items."
        )
        self._in_update = True
        try:
            self.editor = editor
            self.items = items
            workpieces = cast(List[WorkPiece], self.items)
            first_wp = workpieces[0]

            sketch = first_wp.get_sketch_definition()
            if not sketch or not sketch.input_parameters:
                # To clear the widget, populate it with an empty VarSet.
                logger.debug("No sketch or params found, clearing widget.")
                self.varset_widget.populate(VarSet())
                return

            # 1. (Re)populate the VarSetWidget from a CLEAN copy of the sketch
            # definition. This prevents state from one instance's UI session
            # from leaking into another's.
            logger.debug(
                "Populating VarSetWidget from a clean sketch definition copy."
            )
            clean_varset_def = sketch.input_parameters.to_dict(
                include_value=False
            )
            clean_varset = VarSet.from_dict(clean_varset_def)
            self.varset_widget.populate(clean_varset)

            # 2. Update the UI to reflect the actual values from the selection,
            # handling mixed values appropriately.
            self._update_widget_for_mixed_state(clean_varset, workpieces)
        finally:
            self._in_update = False
            logger.debug("Finished updating sketch property widgets.")

    def _update_widget_for_mixed_state(
        self, base_varset: VarSet, workpieces: List[WorkPiece]
    ):
        """
        Adjusts the UI controls to show common values or indicate a
        mixed state when multiple items are selected. This is done by
        calculating the final state (defaults + overrides) and setting it.
        """
        if len(workpieces) == 1:
            wp = workpieces[0]
            # Build the final set of values: start with defaults, then
            # override.
            final_values = base_varset.get_values()
            final_values.update(wp.sketch_params)
            logger.debug(
                f"Single item selected, setting final values: {final_values}"
            )
            self.varset_widget.set_values(final_values)
            return

        # For multiple selections, check each parameter for mixed values.
        logger.debug("Multiple items selected, checking for mixed values.")
        for key, (row, var) in self.varset_widget.widget_map.items():
            all_values: Set[Any] = set()
            for wp in workpieces:
                # Get the instance's value, falling back to the Var's default.
                value = wp.sketch_params.get(key, var.default)
                all_values.add(value)

            if len(all_values) == 1:
                # All workpieces have the same override value for this key.
                common_value = all_values.pop()
                self.varset_widget.set_values({key: common_value})
            else:
                # Values are mixed. Update UI to reflect this.
                logger.debug(f"Parameter '{key}' has mixed values.")
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
        logger.debug(
            f"_on_params_changed called for key '{key}'. "
            f"_in_update={self._in_update}."
        )
        if self._in_update or not self.items:
            return

        # Cancel any previously scheduled update
        if self._debounce_timer_id > 0:
            logger.debug("Cancelling previous debounce timer.")
            GLib.source_remove(self._debounce_timer_id)
            self._debounce_timer_id = 0

        # Store the latest value for the changed key
        all_values = sender.get_values()
        self._pending_changes[key] = all_values.get(key)
        logger.debug(f"Pending changes: {self._pending_changes}")

        # Schedule the actual update to happen after a short delay
        logger.debug("Scheduling debounced update.")
        self._debounce_timer_id = GLib.timeout_add(
            DEBOUNCE_DELAY_MS, self._apply_debounced_changes
        )

    def _apply_debounced_changes(self) -> bool:
        """
        Applies the collected changes to the model. This is called by the
        GLib timer after the user has paused input.
        """
        logger.debug("Debounced timer fired.")
        if not self._pending_changes or not self.items:
            self._debounce_timer_id = 0
            self._pending_changes.clear()
            logger.debug("No pending changes to apply, stopping timer.")
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
