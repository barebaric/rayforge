from __future__ import annotations

import logging
import math
from gettext import gettext as _
from typing import ClassVar

import cairo
from gi.repository import Adw, Gtk

from rayforge.ui_gtk.shared.pref_rows import LengthSpinRow, SpinRow

from ...core.arrays import (
    CurveAlongArray,
    CurveAlongArrayStrategy,
    path_length,
    sample_path,
)
from ...core.commands import CreateArrayCommand
from ...core.entities import Arc, Bezier, Line
from ...core.entity_group import EntityGroup
from .array_base import ArrayToolBase

logger = logging.getLogger(__name__)


def _is_path_entity(entity) -> bool:
    """A Line, Arc or Bezier can act as a guide path."""
    return isinstance(entity, (Line, Arc, Bezier))


def _entity_label(entity) -> str:
    if isinstance(entity, Line):
        return _("Line")
    if isinstance(entity, Arc):
        return _("Arc")
    if isinstance(entity, Bezier):
        return _("Bezier")
    return type(entity).__name__


class CurveAlongArrayTool(ArrayToolBase):
    """
    Creates and edits "array along curve" arrays.

    The selection order defines the roles: the **first** selected entity
    is the guide path (a Line, Arc or Bezier to distribute along); the
    remaining selected entities form the template that gets repeated.
    Copies are oriented along the path tangent (when "align to tangent"
    is on) and are static: editing the path does not live-update the
    copies until the array is re-applied. Double-clicking the guide
    path re-opens this dialog.
    """

    ICON = "sketch-array-symbolic"
    LABEL = _("Array Along Curve")
    SHORTCUTS: ClassVar[list[str]] = ["gw"]

    ARRAY_TYPE = CurveAlongArray
    DIALOG_TITLE = _("Array Along Curve")
    EDIT_DIALOG_TITLE = _("Edit Array Along Curve")
    GROUP_TITLE = _("Along Curve")
    GROUP_DESCRIPTION = _(
        "Distributes copies along a guide path, oriented to its tangent."
    )

    PATH_HIGHLIGHT_COLOR = (0.3, 0.5, 0.8, 0.9)

    #: Minimum number of selected entities: one guide path + one
    #: template entity.
    MIN_SELECTION = 2

    def __init__(self, element):
        super().__init__(element)
        self._path_entity_id: int = -1

    def is_available(self, target, target_type) -> bool:
        return len(self.element.selection.entity_ids) >= self.MIN_SELECTION

    # ------------------------------------------------------------------
    # Parameter construction
    # ------------------------------------------------------------------

    def _make_default_strategy(self) -> CurveAlongArrayStrategy:
        return CurveAlongArrayStrategy(
            count=6,
            path_entity_id=self._path_entity_id,
            align_to_tangent=True,
            offset_to_start=0.0,
            spacing=0.0,
        )

    def _make_strategy_from_target(self) -> CurveAlongArrayStrategy:
        assert self._edit_target is not None
        assert isinstance(self._edit_target, CurveAlongArray)
        self._path_entity_id = self._edit_target.path_entity_id
        return CurveAlongArrayStrategy(
            count=self._edit_target.count,
            path_entity_id=self._edit_target.path_entity_id,
            align_to_tangent=self._edit_target.align_to_tangent,
            offset_to_start=self._edit_target.offset_to_start,
            spacing=self._edit_target.spacing,
        )

    # ------------------------------------------------------------------
    # Activation: split selection into guide + template
    # ------------------------------------------------------------------

    def on_activate(self):
        if self._edit_target is not None:
            if not self._begin_edit():
                self.element.set_tool("select")
            return

        registry = self.element.sketch.registry
        selected = [
            eid
            for eid in self.element.selection.entity_ids
            if registry.get_entity(eid) is not None
        ]
        if len(selected) < self.MIN_SELECTION:
            self.element.set_tool("select")
            return

        self._path_entity_id = selected[0]
        self._template_entity_ids = selected[1:]
        self._strategy = self._make_default_strategy()
        self._show_dialog()

    # ------------------------------------------------------------------
    # Dialog rows
    # ------------------------------------------------------------------

    def _build_mode_rows(self, group: Adw.PreferencesGroup):
        strategy = self._strategy
        registry = self.element.sketch.registry

        # Guide path (read-only summary of the first selected entity).
        path_entity = registry.get_entity(strategy.path_entity_id)
        path_row = Adw.ActionRow()
        path_row.set_title(_("Guide path"))
        path_row.set_subtitle(
            _("First selected entity; the rest are repeated.")
        )
        if path_entity is not None:
            path_row.add_suffix(
                Gtk.Label(
                    label=f"{_entity_label(path_entity)} "
                    f"#{strategy.path_entity_id}"
                )
            )
        group.add(path_row)

        self._count_row = SpinRow(
            _("Count"),
            lower=2,
            upper=360,
            digits=0,
            value=strategy.count,
        )

        self._spacing_row = LengthSpinRow(
            _("Spacing"),
            subtitle=_(
                "Distance between copies. 0 derives count from the path "
                "length; >0 sets the count from the usable length."
            ),
            lower=0.0,
            upper=100000.0,
            value_in_base=strategy.spacing,
        )

        self._offset_row = LengthSpinRow(
            _("Start offset"),
            lower=0.0,
            upper=100000.0,
            value_in_base=strategy.offset_to_start,
        )

        align_row = Adw.ActionRow()
        align_row.set_title(_("Align to tangent"))
        self._align_switch = Gtk.Switch()
        self._align_switch.set_active(strategy.align_to_tangent)
        self._align_switch.set_valign(Gtk.Align.CENTER)
        self._align_switch.connect(
            "notify::active", lambda *a: self._sync_params()
        )
        align_row.add_suffix(self._align_switch)
        align_row.set_activatable_widget(self._align_switch)

        self._count_row.value_changed.connect(
            lambda *a: self._on_count_changed()
        )
        self._spacing_row.value_changed.connect(
            lambda *a: self._on_spacing_changed()
        )
        self._offset_row.value_changed.connect(lambda *a: self._sync_params())

        group.add(self._count_row)
        group.add(self._spacing_row)
        group.add(self._offset_row)
        group.add(align_row)

    # ------------------------------------------------------------------
    # Count <-> spacing linkage
    # ------------------------------------------------------------------

    def _on_count_changed(self):
        """Editing count clears spacing (count takes over)."""
        if self._updating_rows or self._strategy is None:
            return
        self._strategy.spacing = 0.0
        self._strategy.count = self._count_row.get_int_value()
        self._strategy.offset_to_start = (
            self._offset_row.get_value_in_base_units()
        )
        self._strategy.align_to_tangent = self._align_switch.get_active()
        self._updating_rows = True
        try:
            self._spacing_row.set_value_in_base_units(0.0)
        finally:
            self._updating_rows = False
        self.element.mark_dirty()

    def _on_spacing_changed(self):
        """Editing spacing drives the count from the path length."""
        if self._updating_rows or self._strategy is None:
            return
        # Sync spacing/offset into params first, then derive count.
        self._strategy.spacing = self._spacing_row.get_value_in_base_units()
        self._strategy.offset_to_start = (
            self._offset_row.get_value_in_base_units()
        )
        if (
            self._strategy.spacing > 1e-9
            and self._strategy.path_entity_id >= 0
        ):
            total = path_length(
                self.element.sketch.registry, self._strategy.path_entity_id
            )
            offset = min(self._strategy.offset_to_start, total)
            usable = max(total - offset, 0.0)
            if usable > 0.0:
                count = max(
                    1, min(360, int(usable / self._strategy.spacing) + 1)
                )
            else:
                count = 1
            self._strategy.count = count
            self._updating_rows = True
            try:
                self._count_row.set_value(count)
            finally:
                self._updating_rows = False
        self._strategy.align_to_tangent = self._align_switch.get_active()
        self.element.mark_dirty()

    def _sync_params(self):
        if self._updating_rows or self._strategy is None:
            return
        self._strategy.count = self._count_row.get_int_value()
        self._strategy.spacing = self._spacing_row.get_value_in_base_units()
        self._strategy.offset_to_start = (
            self._offset_row.get_value_in_base_units()
        )
        self._strategy.align_to_tangent = self._align_switch.get_active()
        self.element.mark_dirty()

    # ------------------------------------------------------------------
    # Command assembly
    # ------------------------------------------------------------------

    def _make_create_command(self) -> CreateArrayCommand | None:
        if self._strategy is None or self._strategy.path_entity_id < 0:
            return None
        return CreateArrayCommand(
            self.element.sketch,
            self._strategy,
            list(self._template_entity_ids),
        )

    # ------------------------------------------------------------------
    # Preview overlay
    # ------------------------------------------------------------------

    def _draw_guide(self, ctx: cairo.Context, model_to_screen, strategy):
        """Highlights the selected guide path on top of the canvas."""
        if self._strategy is None or self._strategy.path_entity_id < 0:
            return
        registry = self.element.sketch.registry
        entity = registry.get_entity(self._strategy.path_entity_id)
        if entity is None:
            return
        polyline = self._path_polyline(registry, entity)
        if len(polyline) < 2:
            return
        ctx.save()
        ctx.set_line_width(2.0)
        ctx.set_dash([6.0, 4.0])
        ctx.set_source_rgba(*self.PATH_HIGHLIGHT_COLOR)
        ctx.new_sub_path()
        sx, sy = model_to_screen.transform_point(*polyline[0])
        ctx.move_to(sx, sy)
        for x, y in polyline[1:]:
            sx, sy = model_to_screen.transform_point(x, y)
            ctx.line_to(sx, sy)
        ctx.stroke()

        # Sample markers along the path where copies will land.
        # Slot j's copy lands on sample j; sample 0 is position 0,
        # where the template sits, so it gets no marker.
        count = max(
            int(
                self._count_row.get_int_value()
                if self._count_row is not None
                else self._strategy.count
            ),
            1,
        )
        samples = sample_path(
            registry,
            self._strategy.path_entity_id,
            count,
            self._strategy.offset_to_start,
        )
        ctx.set_dash([])
        ctx.set_line_width(1.0)
        for point, _angle in samples[1:]:
            sx, sy = model_to_screen.transform_point(*point)
            ctx.arc(sx, sy, 3.0, 0, 2 * math.pi)
            ctx.stroke()
        ctx.restore()

    def _path_polyline(self, registry, entity):
        polylines = EntityGroup(registry, [entity.id]).polylines()
        return polylines[0] if polylines else []
