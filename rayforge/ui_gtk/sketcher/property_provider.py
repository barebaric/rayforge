import logging
from gettext import gettext as _
from typing import Any, List, Set, TYPE_CHECKING, cast

from gi.repository import Adw, GLib, Gtk

from ...core.item import DocItem
from ...core.varset import VarSet
from ...core.workpiece import WorkPiece
from ..doceditor.property_providers.base import PropertyProvider
from ..varset.varsetwidget import VarSetWidget

if TYPE_CHECKING:
    from ...core.sketcher.sketch import Sketch
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)

DEBOUNCE_DELAY_MS = 300


class SketchPropertyProvider(PropertyProvider):
    """Provides a VarSetWidget to configure a Sketch's input parameters."""

    priority = 100

    def can_handle(self, items: List[DocItem]) -> bool:
        """
        Handles the selection if all items are WorkPieces derived from the
        *same* sketch definition.
        """
        if not items:
            return False

        first_sketch_uid = None
        for item in items:
            if (
                not isinstance(item, WorkPiece)
                or not item.geometry_provider_uid
            ):
                return False

            if first_sketch_uid is None:
                first_sketch_uid = item.geometry_provider_uid
            elif item.geometry_provider_uid != first_sketch_uid:
                return False

        return first_sketch_uid is not None

    def create_widgets(self) -> List[Gtk.Widget]:
        """Creates the VarSetWidget for sketch parameters."""
        logger.debug("Creating sketch property widgets.")
        self.varset_widget = VarSetWidget(title=_("Sketch Parameters"))
        self.varset_widget.data_changed.connect(self._on_params_changed)

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

            sketch = cast("Sketch", first_wp.get_geometry_provider())
            if not sketch or not sketch.input_parameters:
                logger.debug("No sketch or params found, clearing widget.")
                self.varset_widget.populate(VarSet())
                return

            logger.debug(
                "Populating VarSetWidget from a clean sketch definition copy."
            )
            clean_varset_def = sketch.input_parameters.to_dict(
                include_value=False
            )
            clean_varset = VarSet.from_dict(clean_varset_def)
            self.varset_widget.populate(clean_varset)

            self._update_widget_for_mixed_state(clean_varset, workpieces)
        finally:
            self._in_update = False
            logger.debug("Finished updating sketch property widgets.")

    def _update_widget_for_mixed_state(
        self, base_varset: VarSet, workpieces: List[WorkPiece]
    ):
        """
        Adjusts the UI controls to show common values or indicate a
        mixed state when multiple items are selected.
        """
        if len(workpieces) == 1:
            wp = workpieces[0]
            final_values = base_varset.get_values()
            final_values.update(wp.geometry_provider_params)
            logger.debug(
                f"Single item selected, setting final values: {final_values}"
            )
            self.varset_widget.set_values(final_values)
            return

        logger.debug("Multiple items selected, checking for mixed values.")
        for key, (row, var) in self.varset_widget.widget_map.items():
            all_values: Set[Any] = set()
            for wp in workpieces:
                value = wp.geometry_provider_params.get(key, var.default)
                all_values.add(value)

            if len(all_values) == 1:
                common_value = all_values.pop()
                self.varset_widget.set_values({key: common_value})
            else:
                logger.debug(f"Parameter '{key}' has mixed values.")
                if isinstance(row, Adw.EntryRow):
                    row.set_text("")
                elif isinstance(row, Adw.SpinRow):
                    row.set_subtitle(_("Mixed Values"))
                elif isinstance(row, Adw.ComboRow):
                    row.set_selected(0)
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

        if self._debounce_timer_id > 0:
            logger.debug("Cancelling previous debounce timer.")
            GLib.source_remove(self._debounce_timer_id)
            self._debounce_timer_id = 0

        all_values = sender.get_values()
        self._pending_changes[key] = all_values.get(key)
        logger.debug(f"Pending changes: {self._pending_changes}")

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
            return GLib.SOURCE_REMOVE

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

        return GLib.SOURCE_REMOVE
