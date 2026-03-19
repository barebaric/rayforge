import logging
from typing import Optional, TYPE_CHECKING
from gettext import gettext as _
from gi.repository import Adw, Gtk
from ...core.sketcher.commands import RemoveItemsCommand
from ..icons import get_icon

if TYPE_CHECKING:
    from .sketchelement import SketchElement

logger = logging.getLogger(__name__)


class ConflictingConstraintsWidget(Adw.PreferencesGroup):
    """
    A widget that displays conflicting constraints in the sketch studio.
    Shows a list of constraints that have CONFLICTING status after solving.
    """

    def __init__(self):
        super().__init__()
        self._sketch_element: Optional["SketchElement"] = None

        self.set_title(_("Conflicting Constraints"))
        self.set_description(
            _("These constraints cannot be satisfied simultaneously")
        )
        self.set_visible(False)

        self._rows: list[Adw.ActionRow] = []
        self._constraint_indices: list[int] = []

    def set_sketch_element(self, element: Optional["SketchElement"]):
        """
        Sets the sketch element to monitor for conflicts.
        """
        self._sketch_element = element
        self._update_conflicts()

    def _update_conflicts(self):
        """
        Updates the list of conflicting constraints.
        """
        for row in self._rows:
            self.remove(row)
        self._rows.clear()
        self._constraint_indices.clear()

        if self._sketch_element is None:
            self.set_visible(False)
            return

        sketch = self._sketch_element.sketch
        conflicts = sketch.conflicting_constraints

        if not conflicts:
            self.set_visible(False)
            return

        self.set_visible(True)

        for constraint in conflicts:
            idx = sketch.constraints.index(constraint)
            self._constraint_indices.append(idx)

            row = Adw.ActionRow()
            row.set_title(constraint.get_title())
            row.set_activatable(True)

            subtitle = constraint.get_subtitle(sketch.registry)
            if subtitle:
                row.set_subtitle(subtitle)

            delete_btn = Gtk.Button(child=get_icon("delete-symbolic"))
            delete_btn.add_css_class("flat")
            delete_btn.add_css_class("circular")
            delete_btn.set_valign(Gtk.Align.CENTER)
            delete_btn.set_tooltip_text(_("Delete constraint"))
            delete_btn.connect("clicked", self._on_delete_clicked, idx)
            row.add_suffix(delete_btn)

            icon = get_icon("warning-symbolic")
            icon.add_css_class("error")
            icon.set_valign(Gtk.Align.CENTER)
            row.add_suffix(icon)

            row.connect("activated", self._on_row_activated, idx)

            hover_ctrl = Gtk.EventControllerMotion()
            hover_ctrl.connect("enter", self._on_row_enter, idx)
            hover_ctrl.connect("leave", self._on_row_leave)
            row.add_controller(hover_ctrl)

            self.add(row)
            self._rows.append(row)

    def _on_row_enter(self, ctrl, x, y, idx: int):
        """Highlights the constraint on the canvas when hovering."""
        if self._sketch_element:
            self._sketch_element.external_hovered_constraint_idx = idx
            self._sketch_element.mark_dirty()

    def _on_row_leave(self, ctrl):
        """Removes highlight when leaving the row."""
        if self._sketch_element:
            self._sketch_element.external_hovered_constraint_idx = None
            self._sketch_element.mark_dirty()

    def _on_row_activated(self, row, idx: int):
        """Selects the constraint when the row is clicked."""
        if self._sketch_element:
            self._sketch_element.selection.select_constraint(
                idx, is_multi=False
            )

    def _on_delete_clicked(self, btn, idx: int):
        """Deletes the constraint."""
        if self._sketch_element is None:
            return

        sketch = self._sketch_element.sketch
        if idx < 0 or idx >= len(sketch.constraints):
            return

        constraint = sketch.constraints[idx]
        cmd = RemoveItemsCommand(
            sketch=sketch,
            name=_("Delete Constraint"),
            constraints=[constraint],
        )
        self._sketch_element.execute_command(cmd)
