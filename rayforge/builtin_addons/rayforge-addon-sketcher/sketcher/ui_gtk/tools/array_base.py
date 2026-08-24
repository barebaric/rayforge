from __future__ import annotations

import logging
import math
from collections.abc import Callable
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, ClassVar

import cairo
from gi.repository import Adw, Gtk

from ...core.commands import CreatePatternCommand, EditPatternCommand
from ...core.entities import Arc, Bezier, Circle, Ellipse, Line
from ...core.patterns import make_pattern_strategy
from .base import SketcherKey, SketchTool

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...core.entities import Entity
    from ...core.patterns import PatternDefinition, SketchArrayMode
    from ...core.registry import EntityRegistry


class ArrayToolBase(SketchTool):
    """
    Base class for pattern/array tools.

    Handles seed capture, a non-modal parameter dialog with live preview,
    canvas interaction for placing the pattern anchor, and committing a
    CreatePatternCommand. When an edit target is set before activation,
    the tool edits an existing pattern definition instead: parameters are
    pre-filled from its master geometry and applying regenerates the
    instances via an EditPatternCommand.

    Subclasses provide mode-specific parameters, dialog rows, and command
    assembly.
    """

    MODE: ClassVar[SketchArrayMode]
    DIALOG_TITLE: ClassVar[str]
    EDIT_DIALOG_TITLE: ClassVar[str] = ""
    GROUP_TITLE: ClassVar[str] = ""
    GROUP_DESCRIPTION: ClassVar[str] = ""

    PREVIEW_COLOR = (0.6, 0.3, 0.8, 0.85)
    GUIDE_COLOR = (0.3, 0.5, 0.8, 0.9)

    def __init__(self, element):
        super().__init__(element)
        self._dialog: Adw.Window | None = None
        self._params: Any = None
        self._edit_target: PatternDefinition | None = None
        self._seed_entity_ids: list[int] = []
        self._seed_points: list[tuple[float, float]] = []
        self._updating_rows = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_available(self, target, target_type) -> bool:
        return bool(self.element.selection.entity_ids)

    def set_edit_target(self, pattern: PatternDefinition) -> None:
        """
        Arms the tool for editing an existing pattern. Must be called
        before the tool is activated (e.g. before set_tool()).
        """
        self._edit_target = pattern

    def is_available_for_edit(self, pattern: PatternDefinition) -> bool:
        return self.MODE == pattern.mode

    def on_activate(self):
        if self._edit_target is not None:
            if not self._begin_edit():
                self.element.set_tool("select")
            return

        registry = self.element.sketch.registry
        self._seed_entity_ids = [
            eid
            for eid in self.element.selection.entity_ids
            if registry.get_entity(eid) is not None
        ]
        if not self._seed_entity_ids:
            self.element.set_tool("select")
            return

        self._capture_seed_geometry(registry)
        self._params = self._make_default_params()
        self._show_dialog()

    def _begin_edit(self) -> bool:
        """Prepares state for editing the current _edit_target."""
        assert self._edit_target is not None
        registry = self.element.sketch.registry
        living = self._edit_target.living_members(registry)
        if not living:
            return False
        _template_slot, template_eids = living[0]
        self._seed_entity_ids = list(template_eids)
        self._capture_seed_geometry(registry)
        self._params = self._make_params_from_target()
        logger.info(
            "ArrayTool: begin edit uid=%s living=%r params=%r",
            self._edit_target.uid[:8],
            living,
            self._params,
        )
        self._show_dialog()
        return True

    def on_deactivate(self):
        self._close_dialog()
        self._params = None
        self._edit_target = None

    def _capture_seed_geometry(self, registry: EntityRegistry):
        seed_pids = CreatePatternCommand.collect_seed_point_ids(
            registry, self._seed_entity_ids
        )
        self._seed_points = []
        for pid in seed_pids:
            pt = registry.get_point(pid)
            self._seed_points.append((pt.x, pt.y))

    def _seed_bbox_center(self) -> tuple[float, float]:
        if not self._seed_points:
            return (0.0, 0.0)
        xs = [p[0] for p in self._seed_points]
        ys = [p[1] for p in self._seed_points]
        return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)

    @property
    def _is_editing(self) -> bool:
        return self._edit_target is not None

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _make_default_params(self) -> Any:
        """Creates initial parameters for creating a new pattern."""
        raise NotImplementedError

    def _make_params_from_target(self) -> Any:
        """Creates initial parameters from the current edit target."""
        raise NotImplementedError

    def _build_mode_rows(self, group: Adw.PreferencesGroup) -> None:
        raise NotImplementedError

    def _sync_params_from_rows(self) -> None:
        """Reads current widget values into self._params."""

    def _make_create_command(self) -> CreatePatternCommand | None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Dialog
    # ------------------------------------------------------------------

    def _show_dialog(self):
        editor = self.element.editor
        if not editor or not editor.parent_window:
            self.element.set_tool("select")
            return

        title = (
            self.EDIT_DIALOG_TITLE if self._is_editing else self.DIALOG_TITLE
        )
        parent_window = editor.parent_window
        dialog = Adw.Window(
            transient_for=parent_window,
            modal=False,
            destroy_with_parent=True,
            title=title,
        )
        dialog.set_default_size(380, -1)

        header = Adw.HeaderBar()
        cancel_btn = Gtk.Button(label=_("_Cancel"), use_underline=True)
        cancel_btn.connect("clicked", lambda *_: self._on_cancel())
        header.pack_start(cancel_btn)

        apply_btn = Gtk.Button(label=_("_Apply"), use_underline=True)
        apply_btn.add_css_class("suggested-action")
        apply_btn.connect("clicked", lambda *_: self._on_apply())
        header.pack_end(apply_btn)

        group = Adw.PreferencesGroup()
        if self.GROUP_TITLE:
            group.set_title(self.GROUP_TITLE)
        if self.GROUP_DESCRIPTION:
            group.set_description(self.GROUP_DESCRIPTION)
        self._build_mode_rows(group)

        page = Adw.PreferencesPage()
        page.add(group)

        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(header)
        toolbar.set_content(page)

        dialog.set_content(toolbar)
        dialog.connect("close-request", self._on_close_request)
        self._dialog = dialog
        dialog.present()

    def _close_dialog(self):
        if self._dialog is not None:
            dialog = self._dialog
            self._dialog = None
            dialog.destroy()

    def _on_apply(self):
        cmd = self._collect_command()
        if cmd is None:
            logger.warning("ArrayTool: apply skipped, no command built")
            return
        logger.info(
            "ArrayTool: applying %s with %r", type(cmd).__name__, cmd.params
        )
        self.element.execute_command(cmd)
        if cmd.created_entity_ids:
            self._select_entities(cmd.created_entity_ids)
        self._close_dialog()
        self.element.mark_dirty()
        self.element.set_tool("select")

    def _on_cancel(self):
        self.element.set_tool("select")

    def _on_close_request(self, *_args) -> bool:
        """
        Handles the window manager closing the dialog (X button).
        Returns False so the window is allowed to close.
        """
        if self._dialog is not None:
            self._dialog = None
            self.element.set_tool("select")
        return False

    def _collect_command(self):
        if self._is_editing:
            assert self._params is not None
            assert self._edit_target is not None
            return EditPatternCommand(
                self.element.sketch, self._edit_target, self._params
            )
        return self._make_create_command()

    def _select_entities(self, entity_ids: list[int]):
        registry = self.element.sketch.registry
        selection = self.element.selection
        selection.clear()
        for eid in entity_ids:
            entity = registry.get_entity(eid)
            if entity is not None:
                selection.select_entity(entity, False)

    # ------------------------------------------------------------------
    # Canvas interaction
    # ------------------------------------------------------------------

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        """
        Canvas clicks are deliberately ignored: an accidental click or
        pan attempt must never relocate the pattern center. The center
        is set numerically (and stays draggable afterwards via the guide
        circle's own center point).
        """
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def handle_key_event(
        self, key: SketcherKey, shift: bool = False, ctrl: bool = False
    ) -> bool:
        if key == SketcherKey.ESCAPE and self._dialog is not None:
            self.element.set_tool("select")
            return True
        return False

    def get_active_shortcuts(
        self,
    ) -> list[tuple[str | list[str], str, Callable[[], bool] | None]]:
        return []

    # ------------------------------------------------------------------
    # Live preview
    # ------------------------------------------------------------------

    def draw_overlay(self, ctx: cairo.Context):
        if self._dialog is None or self._params is None:
            return
        registry = self.element.sketch.registry
        strategy = self._make_strategy()
        placements = strategy.calculate_placements(self._seed_bbox_center())
        # Preview must mirror what Apply will do: when the parameters
        # changed, everything but the template is re-distributed, so
        # ghost every non-template slot. Otherwise only truly missing
        # slots are filled and occupied ones stay untouched.
        occupied_slots: set[int] = set()
        if self._edit_target is not None:
            if self._would_full_regen():
                occupied_slots = {0}
            else:
                occupied_slots = self._edit_target.occupied_slots(registry)

        model_to_screen = self.element.hittester.get_model_to_screen_transform(
            self.element
        )

        ctx.save()
        ctx.set_line_width(1.5)
        ctx.set_dash([5.0, 4.0])
        ctx.set_source_rgba(*self.PREVIEW_COLOR)

        polylines = _collect_seed_polylines(registry, self._seed_entity_ids)
        for index, placement in enumerate(placements):
            if index + 1 in occupied_slots:
                continue
            for polyline in polylines:
                transformed = [
                    placement.transform_point(x, y) for x, y in polyline
                ]
                _stroke_polyline(ctx, model_to_screen, transformed)

        self._draw_guide(ctx, model_to_screen, strategy)
        ctx.restore()

    def _would_full_regen(self) -> bool:
        """True if applying now would re-distribute all members."""
        target = self._edit_target
        assert self._params is not None
        if target is None:
            return False
        return (
            self._params.count != target.count
            or self._params.total_angle_deg != target.total_angle_deg
            or self._params.rotate_copies != target.rotate_copies
        )

    def _make_strategy(self):
        assert self._params is not None
        return make_pattern_strategy(self.MODE, self._params)

    def _draw_guide(self, ctx: cairo.Context, model_to_screen, strategy):
        """Draws the guide circle and center marker."""
        if not strategy.needs_center_point:
            return
        center = self._params.center
        sx, sy = model_to_screen.transform_point(*center)
        ctx.save()
        cross = 8.0
        ctx.set_source_rgba(*self.GUIDE_COLOR)
        ctx.set_line_width(1.0)
        ctx.move_to(sx - cross, sy)
        ctx.line_to(sx + cross, sy)
        ctx.move_to(sx, sy - cross)
        ctx.line_to(sx, sy + cross)
        ctx.stroke()

        radius = self._params.radius
        if radius > 0.0:
            radius_pt = model_to_screen.transform_point(
                center[0] + radius, center[1]
            )
            screen_radius = math.hypot(radius_pt[0] - sx, radius_pt[1] - sy)
            ctx.set_dash([5.0, 4.0])
            ctx.new_sub_path()
            ctx.arc(sx, sy, screen_radius, 0, 2 * math.pi)
            ctx.stroke()
        ctx.restore()


# ----------------------------------------------------------------------
# Preview geometry sampling helpers
# ----------------------------------------------------------------------


def _collect_seed_polylines(
    registry: EntityRegistry, entity_ids: list[int]
) -> list[list[tuple[float, float]]]:
    polylines: list[list[tuple[float, float]]] = []
    for eid in entity_ids:
        entity = registry.get_entity(eid)
        if entity is None:
            continue
        polylines.extend(_entity_polylines(entity, registry))
    return polylines


def _entity_polylines(
    entity: Entity, registry: EntityRegistry
) -> list[list[tuple[float, float]]]:

    def point(pid):
        pt = registry.get_point(pid)
        return (pt.x, pt.y)

    if isinstance(entity, Line):
        return [[point(entity.p1_idx), point(entity.p2_idx)]]

    if isinstance(entity, Circle):
        c = point(entity.center_idx)
        r_pt = point(entity.radius_pt_idx)
        radius = math.hypot(r_pt[0] - c[0], r_pt[1] - c[1])
        return [_sample_arc(c, radius, 0.0, 2 * math.pi, clockwise=False)]

    if isinstance(entity, Arc):
        start = point(entity.start_idx)
        end = point(entity.end_idx)
        c = point(entity.center_idx)
        radius = math.hypot(start[0] - c[0], start[1] - c[1])
        start_a = math.atan2(start[1] - c[1], start[0] - c[0])
        end_a = math.atan2(end[1] - c[1], end[0] - c[0])
        return [
            _sample_arc(c, radius, start_a, end_a, clockwise=entity.clockwise)
        ]

    if isinstance(entity, Ellipse):
        c = point(entity.center_idx)
        rx_pt = point(entity.radius_x_pt_idx)
        ry_pt = point(entity.radius_y_pt_idx)
        rx = math.hypot(rx_pt[0] - c[0], rx_pt[1] - c[1])
        ry = math.hypot(ry_pt[0] - c[0], ry_pt[1] - c[1])
        rotation = math.atan2(rx_pt[1] - c[1], rx_pt[0] - c[0])
        return [_sample_ellipse(c, rx, ry, rotation)]

    if isinstance(entity, Bezier):
        start = point(entity.start_idx)
        end = point(entity.end_idx)
        cp1 = start
        if entity.cp1 is not None:
            cp1 = (start[0] + entity.cp1[0], start[1] + entity.cp1[1])
        cp2 = end
        if entity.cp2 is not None:
            cp2 = (end[0] + entity.cp2[0], end[1] + entity.cp2[1])
        return [_sample_bezier(start, cp1, cp2, end)]

    return []


def _sample_arc(
    center: tuple[float, float],
    radius: float,
    start_a: float,
    end_a: float,
    clockwise: bool,
) -> list[tuple[float, float]]:
    two_pi = 2 * math.pi
    sweep = end_a - start_a
    if clockwise:
        while sweep >= 0:
            sweep -= two_pi
    else:
        while sweep <= 0:
            sweep += two_pi

    segments = max(8, int(abs(sweep) / two_pi * 48))
    return [
        (
            center[0] + radius * math.cos(start_a + sweep * i / segments),
            center[1] + radius * math.sin(start_a + sweep * i / segments),
        )
        for i in range(segments + 1)
    ]


def _sample_ellipse(
    center: tuple[float, float],
    rx: float,
    ry: float,
    rotation: float,
) -> list[tuple[float, float]]:
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    segments = 48
    result = []
    for i in range(segments + 1):
        t = 2 * math.pi * i / segments
        ex = rx * math.cos(t)
        ey = ry * math.sin(t)
        result.append(
            (
                center[0] + ex * cos_r - ey * sin_r,
                center[1] + ex * sin_r + ey * cos_r,
            )
        )
    return result


def _sample_bezier(p0, p1, p2, p3):
    segments = 24
    result = []
    for i in range(segments + 1):
        t = i / segments
        u = 1.0 - t
        x = (
            u * u * u * p0[0]
            + 3 * u * u * t * p1[0]
            + 3 * u * t * t * p2[0]
            + t * t * t * p3[0]
        )
        y = (
            u * u * u * p0[1]
            + 3 * u * u * t * p1[1]
            + 3 * u * t * t * p2[1]
            + t * t * t * p3[1]
        )
        result.append((x, y))
    return result


def _stroke_polyline(ctx: cairo.Context, model_to_screen, points) -> None:
    if len(points) < 2:
        return
    ctx.new_sub_path()
    sx, sy = model_to_screen.transform_point(*points[0])
    ctx.move_to(sx, sy)
    for x, y in points[1:]:
        sx, sy = model_to_screen.transform_point(x, y)
        ctx.line_to(sx, sy)
    ctx.stroke()
