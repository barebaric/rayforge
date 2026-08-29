from __future__ import annotations

import logging
import math
from gettext import gettext as _
from typing import ClassVar

from gi.repository import Adw, Gtk

from rayforge.ui_gtk.shared.pref_rows import (
    AngleSpinRow,
    LengthSpinRow,
    SpinRow,
)

from ...core.arrays import (
    CircularArray,
    CircularArrayStrategy,
    resolve_template_center,
)
from ...core.commands import CreateArrayCommand
from ...core.entities import Circle
from .array_base import ArrayToolBase

logger = logging.getLogger(__name__)


class CircularArrayTool(ArrayToolBase):
    """
    Creates and edits circular (polar) arrays of sketch entities.

    Instances are equal static copies distributed along a circular arc.
    The dashed construction guide circle is the array's master: it can
    be double-clicked to re-open this dialog, and applying an edit
    regenerates missing or moved instances from the first surviving
    member.
    """

    ICON = "sketch-array-symbolic"
    LABEL = _("Circular Array")
    SHORTCUTS: ClassVar[list[str]] = ["gy"]

    ARRAY_TYPE = CircularArray
    DIALOG_TITLE = _("Circular Array")
    EDIT_DIALOG_TITLE = _("Edit Circular Array")
    GROUP_TITLE = _("Circular")
    GROUP_DESCRIPTION = _(
        "Places copies along a circular arc around a centre."
    )

    def is_available(self, target, target_type) -> bool:
        return len(self.element.selection.entity_ids) > 0

    def _make_default_strategy(self) -> CircularArrayStrategy:
        center = self._default_center()
        return CircularArrayStrategy(
            count=6,
            total_angle_deg=360.0,
            center=center,
            radius=self._default_radius(center),
            rotate_copies=True,
        )

    def _make_strategy_from_target(self) -> CircularArrayStrategy:
        """Pre-fills parameters from the edit target's master geometry."""
        assert self._edit_target is not None
        assert isinstance(self._edit_target, CircularArray)
        registry = self.element.sketch.registry
        circle = registry.get_entity(self._edit_target.guide_circle_id)
        center = (0.0, 0.0)
        radius = 10.0
        if isinstance(circle, Circle):
            try:
                center_pt = registry.get_point(circle.center_idx)
                radius_pt = registry.get_point(circle.radius_pt_idx)
                center = (center_pt.x, center_pt.y)
                radius = math.hypot(
                    radius_pt.x - center_pt.x, radius_pt.y - center_pt.y
                )
                logger.info(
                    "ArrayTool: master circle read back center=%r radius=%.3f",
                    center,
                    radius,
                )
            except IndexError:
                pass
        strategy = CircularArrayStrategy(
            count=self._edit_target.count,
            total_angle_deg=self._edit_target.total_angle_deg,
            center=center,
            radius=radius,
            rotate_copies=self._edit_target.rotate_copies,
        )
        logger.info("ArrayTool: prefilled strategy %r", strategy)
        return strategy

    def _default_center(self) -> tuple[float, float]:
        """Defaults to the sketch origin, like a clock/speaker center."""
        return (0.0, 0.0)

    def _default_radius(self, center: tuple[float, float]) -> float:
        """
        Derives the guide circle radius from the template's center —
        exactly like CreateArrayCommand does — so the previewed circle
        matches the applied one without any jump.
        """
        registry = self.element.sketch.registry
        pids = CreateArrayCommand.collect_template_point_ids(
            registry, self._template_entity_ids
        )
        if not pids:
            return 10.0
        template_points = [registry.get_point(pid) for pid in pids]
        tcx, tcy = resolve_template_center(
            registry, self._template_entity_ids, template_points
        )
        radius = math.hypot(tcx - center[0], tcy - center[1])
        return radius if radius > 1e-6 else 10.0

    def _build_mode_rows(self, group: Adw.PreferencesGroup):
        strategy = self._strategy

        self._count_row = SpinRow(
            _("Count"),
            lower=1,
            upper=360,
            digits=0,
            value=strategy.count,
        )
        self._angle_row = AngleSpinRow(
            _("Total angle (deg)"),
            lower=1.0,
            upper=360.0,
            value=strategy.total_angle_deg,
        )
        self._center_x_row = LengthSpinRow(
            _("Center X"),
            lower=-10000,
            upper=10000,
            value_in_base=strategy.center[0],
        )
        self._center_y_row = LengthSpinRow(
            _("Center Y"),
            lower=-10000,
            upper=10000,
            value_in_base=strategy.center[1],
        )
        self._radius_row = LengthSpinRow(
            _("Radius"),
            subtitle=self._radius_row_subtitle(),
            lower=0.0,
            upper=10000,
            value_in_base=strategy.radius,
        )

        # At creation time the radius is derived from where the anchor
        # point sits, so editing it here would silently be overridden.
        if not self._is_editing:
            self._radius_row.set_sensitive(False)

        for row in (
            self._count_row,
            self._angle_row,
            self._center_x_row,
            self._center_y_row,
            self._radius_row,
        ):
            row.value_changed.connect(lambda *a: self._sync_params())
            group.add(row)

        rotate_row = Adw.ActionRow()
        rotate_row.set_title(_("Rotate copies"))
        self._rotate_switch = Gtk.Switch()
        self._rotate_switch.set_active(strategy.rotate_copies)
        self._rotate_switch.set_valign(Gtk.Align.CENTER)
        self._rotate_switch.connect(
            "notify::active", lambda *a: self._sync_params()
        )
        rotate_row.add_suffix(self._rotate_switch)
        rotate_row.set_activatable_widget(self._rotate_switch)
        group.add(rotate_row)

    def _radius_row_subtitle(self) -> str:
        if self._is_editing:
            return _("Resizes the whole array.")
        return _("Derived from the template's anchor point.")

    def _sync_params(self):
        if self._updating_rows or self._strategy is None:
            return
        self._strategy.count = self._count_row.get_int_value()
        self._strategy.total_angle_deg = self._angle_row.get_value()
        cx = self._center_x_row.get_value_in_base_units()
        cy = self._center_y_row.get_value_in_base_units()
        self._strategy.center = (cx, cy)
        self._strategy.radius = self._radius_row.get_value_in_base_units()
        if not self._is_editing:
            # At creation the circle always passes through the template's
            # anchor point; keep the row in sync so preview matches
            # the applied result exactly.
            derived = self._default_radius((cx, cy))
            self._strategy.radius = derived
            self._updating_rows = True
            try:
                self._radius_row.set_value_in_base_units(derived)
            finally:
                self._updating_rows = False
        self._strategy.rotate_copies = self._rotate_switch.get_active()
        self.element.mark_dirty()

    def _make_create_command(self) -> CreateArrayCommand | None:
        if self._strategy is None:
            return None
        return CreateArrayCommand(
            self.element.sketch,
            self._strategy,
            list(self._template_entity_ids),
        )
