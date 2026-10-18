"""Read-only view of the workpieces a step processes."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, Gtk

from ....icons import get_icon
from .base import StepSettingsPage

if TYPE_CHECKING:
    from .....doceditor.editor import DocEditor


class WorkpiecesPage(StepSettingsPage):
    """Lists the workpieces a step processes on its layer.

    A step's workpiece selection is data set by generators (e.g. the
    Material Test Grid links its step to the workpiece it created);
    this page makes that selection visible, it does not edit it. It
    refreshes on step and layer changes so it stays current while the
    dialog is open.
    """

    show_identity = False

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__(editor, step)
        self._group = Adw.PreferencesGroup(
            title=_("Workpieces"),
            description=_("The workpieces this step processes."),
        )
        self.add(self._group)
        self._rows: list[Gtk.Widget] = []
        self._layer = None
        self._connect_layer()
        self._refresh()

    def _connect_layer(self):
        layer = self.step.layer
        if layer is None:
            return
        self._layer = layer
        layer.descendant_added.connect(self._on_layer_changed)
        layer.descendant_removed.connect(self._on_layer_changed)
        layer.descendant_updated.connect(self._on_layer_changed)

    def _disconnect_layer(self):
        layer = self._layer
        if layer is None:
            return
        self._layer = None
        for signal in (
            "descendant_added",
            "descendant_removed",
            "descendant_updated",
        ):
            try:
                getattr(layer, signal).disconnect(self._on_layer_changed)
            except KeyError:
                pass

    def _on_layer_changed(self, *args, **kwargs):
        self._refresh()

    def _sync_widgets_to_model(self, *args):
        super()._sync_widgets_to_model()
        if self.step.layer is not self._layer:
            self._disconnect_layer()
            self._connect_layer()
        self._refresh()

    def _refresh(self):
        """Rebuild the rows from the step's current filter and layer."""
        self._clear_rows()
        if self._layer is None:
            return
        workpieces = self._layer.all_workpieces
        uids = set(self.step.workpiece_uids)
        if not uids:
            summary = _("All workpieces on this layer")
        else:
            included = sum(1 for wp in workpieces if wp.uid in uids)
            summary = _("Processes {count} of {total} workpieces").format(
                count=included, total=len(workpieces)
            )
        self._add_row(Adw.ActionRow(title=summary))
        for wp in workpieces:
            self._add_row(self._workpiece_row(wp, uids))

    def _add_row(self, row: Adw.ActionRow):
        self._rows.append(row)
        self._group.add(row)

    @staticmethod
    def _workpiece_row(wp, uids: set[str]) -> Adw.ActionRow:
        """One non-interactive row per workpiece of the layer."""
        row = Adw.ActionRow(title=wp.name)
        if uids and wp.uid not in uids:
            row.set_sensitive(False)
            icon = get_icon("action-unavailable-symbolic")
        else:
            icon = get_icon("object-select-symbolic")
        icon.set_valign(Gtk.Align.CENTER)
        row.add_suffix(icon)
        return row

    def _clear_rows(self):
        for row in self._rows:
            self._group.remove(row)
        self._rows = []

    def _cleanup(self):
        self._disconnect_layer()
        super()._cleanup()
