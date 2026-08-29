from __future__ import annotations

import logging
import math
from collections.abc import Callable
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, ClassVar

import cairo
from gi.repository import Adw, Gtk

from ...core.arrays import (
    CircularArray,
    CurveAlongArray,
    InstancePlacement,
)
from ...core.commands import CreateArrayCommand, EditArrayCommand
from ...core.entity_group import EntityGroup
from .base import SketcherKey, SketchTool

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...core.arrays import Array


class ArrayToolBase(SketchTool):
    """
    Base class for array tools.

    Handles template capture, a non-modal parameter dialog with live preview,
    canvas interaction for placing the array anchor, and committing a
    CreateArrayCommand. When an edit target is set before activation,
    the tool edits an existing array definition instead: parameters are
    pre-filled from its master geometry and applying regenerates the
    instances via an EditArrayCommand.

    Subclasses provide mode-specific parameters, dialog rows, and command
    assembly.
    """

    ARRAY_TYPE: ClassVar[type[Array]]
    DIALOG_TITLE: ClassVar[str]
    EDIT_DIALOG_TITLE: ClassVar[str] = ""
    GROUP_TITLE: ClassVar[str] = ""
    GROUP_DESCRIPTION: ClassVar[str] = ""

    PREVIEW_COLOR = (0.6, 0.3, 0.8, 0.85)
    GUIDE_COLOR = (0.3, 0.5, 0.8, 0.9)

    def __init__(self, element):
        super().__init__(element)
        self._dialog: Adw.Window | None = None
        self._strategy: Any = None
        self._edit_target: Array | None = None
        self._template_entity_ids: list[int] = []
        self._updating_rows = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_available(self, target, target_type) -> bool:
        return bool(self.element.selection.entity_ids)

    def set_edit_target(self, array_def: Array) -> None:
        """
        Arms the tool for editing an existing array. Must be called
        before the tool is activated (e.g. before set_tool()).
        """
        self._edit_target = array_def

    def is_available_for_edit(self, array_def: Array) -> bool:
        return isinstance(array_def, self.ARRAY_TYPE)

    def on_activate(self):
        if self._edit_target is not None:
            if not self._begin_edit():
                self.element.set_tool("select")
            return

        registry = self.element.sketch.registry
        self._template_entity_ids = [
            eid
            for eid in self.element.selection.entity_ids
            if registry.get_entity(eid) is not None
        ]
        if not self._template_entity_ids:
            self.element.set_tool("select")
            return

        self._strategy = self._make_default_strategy()
        self._show_dialog()

    def _begin_edit(self) -> bool:
        """Prepares state for editing the current _edit_target."""
        assert self._edit_target is not None
        registry = self.element.sketch.registry
        living = self._edit_target.living_members(registry)
        if not living:
            return False
        _template_slot, template_eids = living[0]
        self._template_entity_ids = list(template_eids)
        self._strategy = self._make_strategy_from_target()
        logger.info(
            "ArrayTool: begin edit uid=%s living=%r strategy=%r",
            self._edit_target.uid[:8],
            living,
            self._strategy,
        )
        self._show_dialog()
        return True

    def on_deactivate(self):
        self._close_dialog()
        self._strategy = None
        self._edit_target = None

    def _captured_template_center(self) -> tuple[float, float]:
        """Resolves the template center with the exact same logic the
        CreateArrayCommand uses, so the preview places its ghosts
        exactly where Apply will place the members."""
        registry = self.element.sketch.registry
        if not self._template_entity_ids:
            return (0.0, 0.0)
        return EntityGroup(registry, self._template_entity_ids).center()

    @property
    def _is_editing(self) -> bool:
        return self._edit_target is not None

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _make_default_strategy(self) -> Any:
        """Creates initial parameters for creating a new array."""
        raise NotImplementedError

    def _make_strategy_from_target(self) -> Any:
        """Creates initial parameters from the current edit target."""
        raise NotImplementedError

    def _build_mode_rows(self, group: Adw.PreferencesGroup) -> None:
        raise NotImplementedError

    def _sync_params_from_rows(self) -> None:
        """Reads current widget values into self._strategy."""

    def _make_create_command(self) -> CreateArrayCommand | None:
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
            "ArrayTool: applying %s with %r", type(cmd).__name__, cmd.strategy
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
            assert self._strategy is not None
            assert self._edit_target is not None
            return EditArrayCommand(
                self.element.sketch, self._edit_target, self._strategy
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
        pan attempt must never relocate the array center. The center
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
        if self._dialog is None or self._strategy is None:
            return
        registry = self.element.sketch.registry
        strategy = self._strategy
        template_center = self._captured_template_center()
        placements = strategy.member_placements(template_center, registry)
        # In create mode the template has not been placed on the guide
        # yet; ghost its position-0 destination too, using the exact
        # placement Apply will move it by.
        slot0: InstancePlacement | None = None
        if self._edit_target is None:
            slot0 = strategy.template_placement(template_center, registry)
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

        polylines = EntityGroup(
            registry, self._template_entity_ids
        ).polylines()
        if slot0 is not None:
            # Create mode: the template is still at its drawn
            # position. Every member — including slot 0 — derives
            # from the PLACED template, so compose the position-0
            # placement into the ghosts.
            polylines = [
                [slot0.transform_point(x, y) for x, y in polyline]
                for polyline in polylines
            ]
            for polyline in polylines:
                _stroke_polyline(ctx, model_to_screen, polyline)
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
        assert self._strategy is not None
        if target is None:
            return False
        if self._strategy.count != target.count:
            return True
        if isinstance(target, CircularArray):
            return (
                self._strategy.rotate_copies != target.rotate_copies
                or self._strategy.total_angle_deg != target.total_angle_deg
            )
        if isinstance(target, CurveAlongArray):
            return (
                self._strategy.path_entity_id != target.path_entity_id
                or self._strategy.align_to_tangent != target.align_to_tangent
                or self._strategy.offset_to_start != target.offset_to_start
                or self._strategy.spacing != target.spacing
            )
        return False

    def _draw_guide(self, ctx: cairo.Context, model_to_screen, strategy):
        """Draws the guide circle and center marker."""
        if not strategy.needs_center_point:
            return
        center = strategy.center
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

        radius = strategy.radius
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
# Preview geometry helpers
# ----------------------------------------------------------------------


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
