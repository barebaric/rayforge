"""
Non-modal, movable dialogs for the Array / Pattern tool.

There are three separate dialogs: Grid, Point Rotation and Circular.
Each shares a common base that manages the preview overlay, canvas
drag signals, and model events, but the content and parameter model
are fully independent.
"""

import logging
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional

from gi.repository import Adw, Gtk

from ..core.group import Group
from ..core.item import DocItem
from ..core.workpiece import WorkPiece
from ..context import get_context
from ..doceditor.array import (
    ArrayMode,
    ArrayParams,
    CircularArrayParams,
    GridArrayParams,
    PointRotationParams,
    SpacingMode,
    make_array_strategy,
)
from ..doceditor.array_cmd import ArrayCmd
from .canvas2d.elements.crosshair import CrosshairElement
from .canvas2d.elements.outline import OutlineElement
from .shared.adwfix import get_spinrow_float, get_spinrow_int
from .shared.patched_dialog_window import PatchedDialogWindow

if TYPE_CHECKING:
    from ..doceditor.editor import DocEditor
    from .canvas2d.surface import WorkSurface

logger = logging.getLogger(__name__)

_DISPLACEMENT = 0
_GAP = 1


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------
class _BaseArrayDialog(PatchedDialogWindow):
    """Common infrastructure for all array dialogs.

    Subclasses must implement :meth:`_mode_content` and
    :meth:`_current_params`.
    """

    def __init__(
        self,
        parent: Gtk.Window,
        editor: "DocEditor",
        surface: "WorkSurface",
        items: List[DocItem],
        mode: ArrayMode,
        title: str,
    ):
        super().__init__(transient_for=parent)
        self._mode = mode
        self._editor = editor
        self._surface = surface
        self._items = list(items)
        self._updating = False

        self.set_title(title)
        self.set_default_size(570, -1)
        self.set_modal(False)
        self.set_resizable(True)

        # Preview overlay.
        self._preview = OutlineElement()
        self._surface.root.add(self._preview)

        # Mode-specific initialisation (centre, crosshair, etc.)
        self._init_mode()

        # Canvas drag -> live preview.
        self._surface.transform_moved.connect(self._on_canvas_transform)

        # Model update -> finalise preview.
        for item in self._items:
            item.transform_changed.connect(self._on_item_moved)

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)
        main_box.append(self._build_header(title))
        main_box.append(self._mode_content())

        self.connect("close-request", self._on_close_request)

        self._initial_sync()
        self._update_preview()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def _mode_content(self) -> Gtk.Widget:
        raise NotImplementedError

    def _current_params(self) -> ArrayParams:
        raise NotImplementedError

    def _init_mode(self) -> None:
        """Called before _mode_content — override to set up mode-specific
        state such as the crosshair and centre/radius defaults."""

    def _initial_sync(self) -> None:
        """Called once after init, before the first preview."""

    def _sync_params_live(self, bbox, params: ArrayParams) -> None:
        """Called on each canvas-drag frame to keep derived values
        (e.g. the circular radius) in sync."""

    def _guide_for(self, params: ArrayParams, bbox) -> Optional[tuple]:
        """Return a ``(center, radius)`` guide circle or ``None``."""

    # ------------------------------------------------------------------
    # Shared UI helpers
    # ------------------------------------------------------------------
    def _build_header(self, title: str) -> Adw.HeaderBar:
        header = Adw.HeaderBar()
        header.set_title_widget(
            Adw.WindowTitle.new(title, _("Copies keep their original layers."))
        )
        apply_btn = Gtk.Button(label=_("_Apply"), use_underline=True)
        apply_btn.add_css_class("suggested-action")
        apply_btn.connect("clicked", self._on_apply_clicked)
        header.pack_end(apply_btn)
        return header

    def _make_spin_row(
        self,
        title: str,
        lower,
        upper,
        step,
        value,
        digits: int = 3,
        change_callback=None,
    ) -> Adw.SpinRow:
        row = Adw.SpinRow.new_with_range(float(lower), float(upper), step)
        row.set_title(title)
        row.set_value(float(value))
        row.set_numeric(True)
        row.set_digits(digits)
        cb = change_callback or self._update_preview
        row.get_adjustment().connect("value-changed", lambda *a: cb())
        row.connect("notify::text", lambda *a: cb())
        return row

    @staticmethod
    def _canvas_center():
        """World-space centre of the machine work area (canvas)."""
        machine = get_context().machine
        if not machine:
            return (0.0, 0.0)
        wa = machine.work_area
        wa_w, wa_h = float(wa[2]), float(wa[3])
        ref_x, ref_y = machine.get_reference_position_world()
        space = machine.get_coordinate_space()
        origin = space.world_position_from_origin(ref_x, ref_y, (wa_w, wa_h))
        return (origin[0] + wa_w / 2.0, origin[1] + wa_h / 2.0)

    # ------------------------------------------------------------------
    # Preview data
    # ------------------------------------------------------------------
    def _items_shapes_and_bbox(self, moved_elements=None):
        """Returns ``(shapes, bbox)`` — see ``ArrayDialog``."""
        live = {}
        if moved_elements:
            for ce in moved_elements:
                d = getattr(ce, "data", None)
                if d is not None and hasattr(d, "uid"):
                    live[d.uid] = ce

        min_x = min_y = math.inf
        max_x = max_y = -math.inf
        shapes = []
        items = ArrayCmd._get_top_level_items(self._items)
        for item in items:
            if not isinstance(item, (WorkPiece, Group)):
                continue
            live_elem = live.get(item.uid)
            transform = (
                live_elem.get_world_transform()
                if live_elem is not None
                else item.get_world_transform()
            )
            corners = [
                transform.transform_point(p)
                for p in [(0, 0), (1, 0), (1, 1), (0, 1)]
            ]
            shapes.append(corners)
            for x, y in corners:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        if math.isinf(min_x):
            return shapes, None
        return shapes, (min_x, min_y, max_x, max_y)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------
    def _on_item_moved(self, sender=None, **kwargs) -> None:
        if self._updating:
            return
        self._update_preview()

    def _on_canvas_transform(self, sender, **kwargs) -> None:
        if self._updating:
            return
        elements = kwargs.get("elements")
        shapes, bbox = self._items_shapes_and_bbox(elements)
        if not bbox:
            return
        params = self._current_params()
        self._sync_params_live(bbox, params)
        params = self._current_params()  # re-read if sync changed them
        deltas = make_array_strategy(bbox, params).calculate_placements()
        guide = self._guide_for(params, bbox)
        self._preview.set_outlines(shapes, deltas, guide_circle=guide)

    def _update_preview(self) -> None:
        if self._updating:
            return
        shapes, bbox = self._items_shapes_and_bbox()
        params = self._current_params()
        if bbox:
            deltas = make_array_strategy(bbox, params).calculate_placements()
        else:
            deltas = []
        guide = self._guide_for(params, bbox)
        self._preview.set_outlines(shapes, deltas, guide_circle=guide)

    # ------------------------------------------------------------------
    # Commit / cleanup
    # ------------------------------------------------------------------
    def _on_apply_clicked(self, button) -> None:
        params = self._current_params()
        created = self._editor.array.create_array(self._items, params)
        if created:
            self._surface.select_items(list(self._items) + created)
        self._detach_preview()
        self.close()

    def _on_close_request(self, window) -> bool:
        self._detach_preview()
        return False

    def _detach_preview(self) -> None:
        self._preview.clear()
        self._surface.transform_moved.disconnect(self._on_canvas_transform)
        for item in self._items:
            item.transform_changed.disconnect(self._on_item_moved)
        if self._preview.parent is not None:
            self._preview.remove()
        self._preview.canvas = None


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
class GridArrayDialog(_BaseArrayDialog):
    """Creates a rows x columns grid of the selection."""

    def __init__(self, parent, editor, surface, items):
        super().__init__(
            parent,
            editor,
            surface,
            items,
            ArrayMode.GRID,
            _("Grid Array"),
        )

    def _mode_content(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_top(18)
        box.set_margin_bottom(18)
        box.set_margin_start(18)
        box.set_margin_end(18)
        group = Adw.PreferencesGroup()
        group.set_title(_("Grid"))

        self._rows_row = self._make_spin_row(_("Rows"), 1, 100, 1, 2, 0)
        self._cols_row = self._make_spin_row(_("Columns"), 1, 100, 1, 2, 0)
        self._spacing_mode_row = Adw.ComboRow(
            model=Gtk.StringList.new([_("Displacement"), _("Gap")])
        )
        self._spacing_mode_row.set_title(_("Spacing"))
        self._spacing_mode_row.set_subtitle(
            _("Displacement is center-to-center; gap is edge-to-edge.")
        )
        self._spacing_mode_row.set_selected(_GAP)
        self._spacing_mode_row.connect(
            "notify::selected", self._on_spacing_mode_changed
        )
        self._col_spacing_row = self._make_spin_row(
            _("Column spacing"), -10000, 10000, 0.1, 1.0
        )
        self._row_spacing_row = self._make_spin_row(
            _("Row spacing"), -10000, 10000, 0.1, 1.0
        )

        for r in (
            self._rows_row,
            self._cols_row,
            self._spacing_mode_row,
            self._col_spacing_row,
            self._row_spacing_row,
        ):
            group.add(r)
        box.append(group)
        return box

    def _current_params(self) -> ArrayParams:
        spacing = (
            SpacingMode.GAP
            if self._spacing_mode_row.get_selected() == _GAP
            else SpacingMode.DISPLACEMENT
        )
        return ArrayParams(
            mode=ArrayMode.GRID,
            grid=GridArrayParams(
                rows=get_spinrow_int(self._rows_row),
                cols=get_spinrow_int(self._cols_row),
                spacing_mode=spacing,
                col_spacing_mm=get_spinrow_float(self._col_spacing_row),
                row_spacing_mm=get_spinrow_float(self._row_spacing_row),
            ),
        )

    def _initial_sync(self) -> None:
        self._last_spacing_mode = self._spacing_mode_row.get_selected()

    def _on_spacing_mode_changed(self, *args) -> None:
        if self._updating:
            return
        new_mode = self._spacing_mode_row.get_selected()
        old_mode = self._last_spacing_mode
        if new_mode != old_mode:
            self._translate_spacing(old_mode, new_mode)
        self._last_spacing_mode = new_mode
        self._update_preview()

    def _translate_spacing(self, from_mode: int, to_mode: int) -> None:
        bbox = self._editor.array.get_selection_bbox(self._items)
        if not bbox:
            return
        unit_w = bbox[2] - bbox[0]
        unit_h = bbox[3] - bbox[1]
        col = get_spinrow_float(self._col_spacing_row)
        row = get_spinrow_float(self._row_spacing_row)
        sign = -1.0 if to_mode == _GAP else 1.0
        self._updating = True
        try:
            self._col_spacing_row.set_value(col + sign * unit_w)
            self._row_spacing_row.set_value(row + sign * unit_h)
        finally:
            self._updating = False


# ---------------------------------------------------------------------------
# Point Rotation
# ---------------------------------------------------------------------------
class PointRotationArrayDialog(_BaseArrayDialog):
    """Rotates copies around the selection's own centre."""

    def __init__(self, parent, editor, surface, items):
        super().__init__(
            parent,
            editor,
            surface,
            items,
            ArrayMode.POINT_ROTATION,
            _("Point Rotation Array"),
        )

    def _mode_content(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_top(18)
        box.set_margin_bottom(18)
        box.set_margin_start(18)
        box.set_margin_end(18)
        group = Adw.PreferencesGroup()
        group.set_title(_("Point Rotation"))
        group.set_description(
            _("Rotates copies in place around the selection's centre.")
        )
        self._pr_count_row = self._make_spin_row(_("Count"), 1, 360, 1, 6, 0)
        self._pr_angle_row = self._make_spin_row(
            _("Total angle (deg)"), -360, 360, 1, 360.0
        )
        for r in (self._pr_count_row, self._pr_angle_row):
            group.add(r)
        box.append(group)
        return box

    def _current_params(self) -> ArrayParams:
        return ArrayParams(
            mode=ArrayMode.POINT_ROTATION,
            point_rotation=PointRotationParams(
                count=get_spinrow_int(self._pr_count_row),
                total_angle_deg=get_spinrow_float(self._pr_angle_row),
            ),
        )


# ---------------------------------------------------------------------------
# Circular
# ---------------------------------------------------------------------------
class CircularArrayDialog(_BaseArrayDialog):
    """Places copies along a circular arc around a centre."""

    def __init__(self, parent, editor, surface, items):
        super().__init__(
            parent,
            editor,
            surface,
            items,
            ArrayMode.CIRCULAR,
            _("Circular Array"),
        )

    def _init_mode(self) -> None:
        """Set up centre default and draggable crosshair."""
        bbox = self._editor.array.get_selection_bbox(self._items)
        sel_center = (
            ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            if bbox
            else (0.0, 0.0)
        )
        centre = self._canvas_center()
        self._crosshair = CrosshairElement(on_drag=self._on_center_dragged)
        self._crosshair.move_to(centre[0], centre[1])
        self._surface.root.add(self._crosshair)
        self._centre = centre
        self._default_radius = (
            math.hypot(centre[0] - sel_center[0], centre[1] - sel_center[1])
            if bbox
            else 10.0
        )

    def _mode_content(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_top(18)
        box.set_margin_bottom(18)
        box.set_margin_start(18)
        box.set_margin_end(18)
        group = Adw.PreferencesGroup()
        group.set_title(_("Circular"))
        group.set_description(
            _("Places copies along a circular arc around a centre.")
        )
        self._c_count_row = self._make_spin_row(_("Count"), 1, 360, 1, 6, 0)
        self._c_angle_row = self._make_spin_row(
            _("Total angle (deg)"), -360, 360, 1, 360.0
        )
        self._c_center_x_row = self._make_spin_row(
            _("Center X"),
            -10000,
            10000,
            0.1,
            self._centre[0],
            change_callback=self._on_center_changed,
        )
        self._c_center_y_row = self._make_spin_row(
            _("Center Y"),
            -10000,
            10000,
            0.1,
            self._centre[1],
            change_callback=self._on_center_changed,
        )
        r = self._default_radius if self._default_radius > 0.0 else 10.0
        self._c_radius_row = self._make_spin_row(
            _("Radius"),
            0,
            10000,
            0.1,
            r,
            change_callback=self._on_radius_changed,
        )
        self._c_rotate_row = Adw.ActionRow()
        self._c_rotate_row.set_title(_("Rotate copies"))
        self._c_rotate_switch = Gtk.Switch()
        self._c_rotate_switch.set_active(True)
        self._c_rotate_switch.set_valign(Gtk.Align.CENTER)
        self._c_rotate_switch.connect(
            "notify::active", lambda *a: self._update_preview()
        )
        self._c_rotate_row.add_suffix(self._c_rotate_switch)
        self._c_rotate_row.set_activatable_widget(self._c_rotate_switch)

        for r in (
            self._c_count_row,
            self._c_angle_row,
            self._c_center_x_row,
            self._c_center_y_row,
            self._c_radius_row,
            self._c_rotate_row,
        ):
            group.add(r)
        box.append(group)
        return box

    def _current_params(self) -> ArrayParams:
        return ArrayParams(
            mode=ArrayMode.CIRCULAR,
            circular=CircularArrayParams(
                count=get_spinrow_int(self._c_count_row),
                total_angle_deg=get_spinrow_float(self._c_angle_row),
                center_mm=(
                    get_spinrow_float(self._c_center_x_row),
                    get_spinrow_float(self._c_center_y_row),
                ),
                radius_mm=get_spinrow_float(self._c_radius_row),
                rotate_copies=self._c_rotate_switch.get_active(),
            ),
        )

    def _sync_params_live(self, bbox, params) -> None:
        """Keep the radius matching the current centre & workpiece."""
        ux = (bbox[0] + bbox[2]) / 2.0
        uy = (bbox[1] + bbox[3]) / 2.0
        cx = get_spinrow_float(self._c_center_x_row)
        cy = get_spinrow_float(self._c_center_y_row)
        r = math.hypot(ux - cx, uy - cy)
        self._updating = True
        try:
            self._c_radius_row.set_value(r)
        finally:
            self._updating = False

    def _guide_for(self, params, bbox):
        if not bbox:
            return None
        return (params.circular.center_mm, params.circular.radius_mm)

    def _initial_sync(self) -> None:
        self._sync_radius()

    def _sync_radius(self) -> None:
        _, bbox = self._items_shapes_and_bbox()
        if not bbox:
            return
        ux = (bbox[0] + bbox[2]) / 2.0
        uy = (bbox[1] + bbox[3]) / 2.0
        cx = get_spinrow_float(self._c_center_x_row)
        cy = get_spinrow_float(self._c_center_y_row)
        r = math.hypot(ux - cx, uy - cy)
        was = self._updating
        self._updating = True
        try:
            self._c_radius_row.set_value(r)
        finally:
            self._updating = was

    def _on_center_changed(self) -> None:
        if self._updating:
            return
        self._sync_radius()
        self._crosshair_sync()
        self._update_preview()

    def _on_center_dragged(self, pos) -> None:
        self._updating = True
        try:
            self._c_center_x_row.set_value(pos[0])
            self._c_center_y_row.set_value(pos[1])
            if self._crosshair is not None:
                self._crosshair.move_to(pos[0], pos[1])
            self._sync_radius()
        finally:
            self._updating = False
        self._update_preview()

    def _crosshair_sync(self) -> None:
        if self._crosshair is not None:
            self._crosshair.move_to(
                self._c_center_x_row.get_value(),
                self._c_center_y_row.get_value(),
            )

    def _on_radius_changed(self) -> None:
        if self._updating:
            return
        _, bbox = self._items_shapes_and_bbox()
        if not bbox:
            return
        ux = (bbox[0] + bbox[2]) / 2.0
        uy = (bbox[1] + bbox[3]) / 2.0
        cx = get_spinrow_float(self._c_center_x_row)
        cy = get_spinrow_float(self._c_center_y_row)
        new_r = get_spinrow_float(self._c_radius_row)
        dx = cx - ux
        dy = cy - uy
        cur_d = math.hypot(dx, dy)
        if cur_d > 1e-9:
            was = self._updating
            self._updating = True
            try:
                self._c_center_x_row.set_value(ux + (dx / cur_d) * new_r)
                self._c_center_y_row.set_value(uy + (dy / cur_d) * new_r)
            finally:
                self._updating = was
        self._update_preview()

    def _detach_preview(self) -> None:
        if self._crosshair is not None:
            if self._crosshair.parent is not None:
                self._crosshair.remove()
        self._crosshair = None
        super()._detach_preview()

    def _update_preview(self) -> None:
        self._crosshair_sync()
        super()._update_preview()
