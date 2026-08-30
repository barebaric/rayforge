import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, ClassVar, Union

import cairo
from gi.repository import Adw, Gtk

from rayforge.ui_gtk.shared.pref_rows import SpinRow

from ...core.commands import OffsetCommand
from ...core.contour import build_offset_items
from ...core.entities import Entity, Point
from .base import SketchTool

if TYPE_CHECKING:
    from ...core.constraints import Constraint

logger = logging.getLogger(__name__)


class OffsetTool(SketchTool):
    """Dialog tool that adds offset copies of the selected contours."""

    ICON = "sketch-offset-symbolic"
    LABEL = _("Offset")
    SHORTCUTS: ClassVar[list[str]] = ["of"]

    DEFAULT_DISTANCE = 2.0
    MAX_DISTANCE = 500.0

    def __init__(self, element):
        super().__init__(element)
        self._items = None
        self._entity_ids: list[int] = []
        self._distance: float = self.DEFAULT_DISTANCE
        self._preview_polylines: list | None = None

    def is_available(
        self,
        target: Union[Point, Entity, "Constraint"] | None,
        target_type: str | None,
    ) -> bool:
        return bool(self.element.selection.entity_ids)

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def on_activate(self):
        sketch = self.element.sketch
        self._entity_ids = list(self.element.selection.entity_ids)
        self._items = build_offset_items(sketch, self._entity_ids)
        if not self._items:
            logger.warning(
                "Selection cannot be offset; select simple connected contours."
            )
            self.element.set_tool("select")
            return
        self._update_preview(self._distance)
        self._show_dialog()

    def on_deactivate(self):
        self._items = None
        self._preview_polylines = None

    def draw_overlay(self, ctx: cairo.Context):
        """Draws the offset preview in screen space."""
        if not self._preview_polylines:
            return
        transform = self._get_model_to_screen_transform()
        if transform is None:
            return
        ctx.save()
        ctx.set_source_rgba(0.2, 0.6, 1.0, 0.9)
        ctx.set_dash([4, 2])
        ctx.set_line_width(1.5)
        for polyline in self._preview_polylines:
            for i, (x, y) in enumerate(polyline):
                sx, sy = transform.transform_point(x, y)
                if i == 0:
                    ctx.move_to(sx, sy)
                else:
                    ctx.line_to(sx, sy)
            ctx.stroke()
        ctx.restore()

    def _get_model_to_screen_transform(self):
        canvas = self.element.canvas
        if not canvas:
            return None
        return (
            canvas.view_transform
            @ self.element.get_world_transform()
            @ self.element.content_transform
        )

    def _update_preview(self, distance: float):
        if self._items is None:
            return
        self._preview_polylines = OffsetCommand.preview_polylines(
            self._items, self.element.sketch.registry, distance
        )
        self.element.mark_dirty()

    def _show_dialog(self):
        editor = self.element.editor
        if not editor or not editor.parent_window:
            self.element.set_tool("select")
            return

        parent_window = editor.parent_window
        dialog = Adw.MessageDialog(
            transient_for=parent_window,
            modal=True,
            destroy_with_parent=True,
            heading=_("Offset Contour"),
            body=_(
                "Replaces the selected contours with their offset. "
                "Positive values grow closed contours; open contours "
                "become slots."
            ),
        )

        distance_row = SpinRow(
            _("Distance"),
            lower=-self.MAX_DISTANCE,
            upper=self.MAX_DISTANCE,
            step_increment=0.5,
            digits=2,
            value=self._distance,
            numeric=True,
        )
        distance_row.value_changed.connect(self._on_distance_changed)

        list_box = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        list_box.add_css_class("boxed-list")
        list_box.append(distance_row)
        dialog.set_extra_child(list_box)

        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("apply", _("Apply"))
        dialog.set_response_appearance(
            "apply", Adw.ResponseAppearance.SUGGESTED
        )
        dialog.set_default_response("apply")
        dialog.set_close_response("cancel")

        def on_response(source, response_id):
            if response_id == "apply":
                self._apply()
            else:
                self.element.set_tool("select")
            dialog.close()

        dialog.connect("response", on_response)
        dialog.present()

    def _on_distance_changed(self, row):
        self._distance = row.get_value()
        self._update_preview(self._distance)

    def _apply(self):
        cmd = OffsetCommand(
            self.element.sketch,
            self._entity_ids,
            self._distance,
        )
        self.element.execute_command(cmd)
        self.element.set_tool("select")
