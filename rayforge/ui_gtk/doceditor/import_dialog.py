import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import cairo
from blinker import Signal
from gi.repository import Adw, Gdk, GdkPixbuf, Gtk

from ...core.layer import Layer
from ...core.vectorization_spec import (
    PassthroughSpec,
    TraceSpec,
    VectorizationSpec,
)
from ...core.workpiece import WorkPiece
from ...doceditor.file_cmd import PreviewResult
from ..shared.patched_dialog_window import PatchedDialogWindow

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)

# A fixed, reasonable resolution for generating preview bitmaps.
PREVIEW_RENDER_SIZE_PX = 1024


class ImportDialog(PatchedDialogWindow):
    """
    A dialog for importing images with live preview of vectorization.
    """

    def __init__(
        self,
        parent: Gtk.Window,
        editor: "DocEditor",
        file_path: Path,
        mime_type: str,
    ):
        super().__init__(transient_for=parent, modal=True)
        self.editor = editor
        self.file_path = file_path
        self.mime_type = mime_type
        self.is_svg = self.mime_type == "image/svg+xml"
        self.response = Signal()

        # Internal state
        self._file_bytes: Optional[bytes] = None
        self._preview_result: Optional[PreviewResult] = None
        self._background_pixbuf: Optional[GdkPixbuf.Pixbuf] = None
        self._in_update = False  # Prevent signal recursion
        self._layer_widgets: List[Gtk.Switch] = []

        self.create_new_layers_switch = Adw.SwitchRow(
            title=_("Create New Layers"),
            subtitle=_("Create a new layer for each imported layer"),
            active=False,
        )

        self.set_title(_("Import Image"))
        self.set_default_size(1100, 800)

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        header_bar = Adw.HeaderBar()
        main_box.append(header_bar)

        # Banner for warnings (e.g. SVG empty content)
        self.warning_banner = Adw.Banner(
            title=_(
                "SVG produced no output in direct vector mode. "
                "SVGs containing text or other non-path elements "
                "should be converted to paths before importing "
                "(e.g., in Inkscape: Path > Object to Path)."
            ),
            button_label=_("Switch to Trace Mode"),
        )
        self.warning_banner.connect(
            "button-clicked", self._on_switch_to_trace_clicked
        )
        self.warning_banner.set_revealed(False)
        main_box.append(self.warning_banner)

        content_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, vexpand=True, hexpand=True
        )
        main_box.append(content_box)

        # Header Bar Buttons
        self.import_button = Gtk.Button(
            label=_("Import"), css_classes=["suggested-action"]
        )
        self.import_button.connect("clicked", self._on_import_clicked)
        header_bar.pack_end(self.import_button)

        cancel_button = Gtk.Button(label=_("Cancel"))
        cancel_button.connect("clicked", lambda btn: self.close())
        header_bar.pack_start(cancel_button)

        self.status_spinner = Gtk.Spinner(spinning=True)
        header_bar.pack_start(self.status_spinner)

        # Sidebar for Controls
        sidebar = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            width_request=500,
            hexpand=False,
            margin_top=12,
            margin_bottom=12,
            margin_start=12,
            margin_end=6,
        )
        content_box.append(sidebar)

        preferences_page = Adw.PreferencesPage()
        sidebar.append(preferences_page)

        # Import Mode Group (for SVG)
        mode_group = Adw.PreferencesGroup(title=_("Import Mode"))
        preferences_page.add(mode_group)

        self.use_vectors_switch = Adw.SwitchRow(
            title=_("Use Original Vectors"),
            subtitle=_("Import vector data directly (SVG only)"),
            active=True,
        )
        self.use_vectors_switch.connect(
            "notify::active", self._on_import_mode_toggled
        )
        mode_group.add(self.use_vectors_switch)
        mode_group.set_visible(self.is_svg)

        # Layers Group (Dynamic)
        self.layers_group = Adw.PreferencesGroup(title=_("Layers"))
        self.layers_group.set_visible(False)
        preferences_page.add(self.layers_group)

        # Trace Settings Group
        self.trace_group = Adw.PreferencesGroup(title=_("Trace Settings"))
        preferences_page.add(self.trace_group)

        # Import Whole Image
        self.import_whole_image_switch = Adw.SwitchRow(
            title=_("Import Whole Image"),
            subtitle=_("Import the entire image without tracing"),
            active=True,
        )
        self.import_whole_image_switch.connect(
            "notify::active", self._on_import_whole_image_toggled
        )
        self.trace_group.add(self.import_whole_image_switch)

        # Auto Threshold
        self.auto_threshold_switch = Adw.SwitchRow(
            title=_("Auto Threshold"),
            subtitle=_("Automatically determine the trace threshold"),
            active=True,
        )
        self.auto_threshold_switch.connect(
            "notify::active", self._on_auto_threshold_toggled
        )
        self.trace_group.add(self.auto_threshold_switch)

        # Manual Threshold Slider
        self.threshold_adjustment = Gtk.Adjustment.new(
            0.5, 0.0, 1.0, 0.01, 0.1, 0
        )
        self.threshold_scale = Gtk.Scale.new(
            Gtk.Orientation.HORIZONTAL, self.threshold_adjustment
        )
        self.threshold_scale.set_size_request(200, -1)
        self.threshold_scale.set_digits(2)
        self.threshold_scale.set_value_pos(Gtk.PositionType.RIGHT)
        self.threshold_scale.connect(
            "value-changed", self._schedule_preview_update
        )
        self.threshold_row = Adw.ActionRow(
            title=_("Threshold"),
            subtitle=_("Trace objects darker than this value"),
        )
        self.threshold_row.add_suffix(self.threshold_scale)
        self.threshold_row.set_sensitive(False)
        self.trace_group.add(self.threshold_row)

        # Invert
        self.invert_switch = Adw.SwitchRow(
            title=_("Invert"),
            subtitle=_("Trace light objects on a dark background"),
        )
        self.invert_switch.connect(
            "notify::active", self._schedule_preview_update
        )
        self.trace_group.add(self.invert_switch)

        # Preview Area
        preview_frame = Gtk.Frame(
            vexpand=True,
            hexpand=True,
            margin_top=12,
            margin_bottom=12,
            margin_start=6,
            margin_end=12,
        )
        preview_frame.add_css_class("card")
        content_box.append(preview_frame)

        self.preview_area = Gtk.DrawingArea(
            vexpand=True,
            hexpand=True,
            css_classes=["view"],
        )
        self.preview_area.set_draw_func(self._on_draw_preview)
        preview_frame.set_child(self.preview_area)

        # Initial Load & State
        self._load_initial_data()
        self._on_import_mode_toggled(self.use_vectors_switch)
        self._on_import_whole_image_toggled(
            self.import_whole_image_switch, None
        )

    def _on_import_mode_toggled(self, switch, *args):
        is_direct_import = self.is_svg and switch.get_active()
        self.trace_group.set_sensitive(not is_direct_import)
        self.layers_group.set_sensitive(is_direct_import)
        self.warning_banner.set_revealed(False)
        self._schedule_preview_update()

    def _on_switch_to_trace_clicked(self, banner):
        self.use_vectors_switch.set_active(False)

    def _on_auto_threshold_toggled(self, switch, _pspec):
        is_auto = switch.get_active()
        self.threshold_row.set_sensitive(not is_auto)
        self._schedule_preview_update()

    def _on_import_whole_image_toggled(self, switch, _pspec):
        is_whole_image = switch.get_active()
        self.auto_threshold_switch.set_sensitive(not is_whole_image)
        self.threshold_row.set_sensitive(
            not is_whole_image and not self.auto_threshold_switch.get_active()
        )
        self.invert_switch.set_sensitive(not is_whole_image)
        self._schedule_preview_update()

    def _load_initial_data(self):
        try:
            self._file_bytes = self.file_path.read_bytes()
            if self.is_svg:
                self._populate_layers_ui()
        except Exception:
            logger.error(
                f"Failed to read import file {self.file_path}", exc_info=True
            )
            self.close()

    def _populate_layers_ui(self):
        if not self._file_bytes:
            return

        scan_result = self.editor.file.scan_import_file(
            self._file_bytes, self.mime_type
        )
        layers = scan_result.get("layers", [])

        if not layers:
            return

        self.layers_group.set_visible(True)
        expander = Adw.ExpanderRow(title=_("Select Layers"), expanded=True)
        self.layers_group.add(expander)
        self._layer_widgets.clear()

        for layer in layers:
            row = Adw.ActionRow(title=layer["name"])
            switch = Gtk.Switch(active=True, valign=Gtk.Align.CENTER)
            switch._layer_id = layer["id"]  # type: ignore
            switch.connect("notify::active", self._schedule_preview_update)

            row.add_suffix(switch)
            row.set_activatable_widget(switch)
            expander.add_row(row)

            self._layer_widgets.append(switch)

        expander.add_row(self.create_new_layers_switch)

    def _get_active_layer_ids(self) -> Optional[List[str]]:
        if not self._layer_widgets:
            return None
        return [
            w._layer_id  # type: ignore
            for w in self._layer_widgets
            if w.get_active()
        ]

    def _get_current_spec(self) -> VectorizationSpec:
        """
        Constructs a VectorizationSpec from the current UI control values.
        """
        if self.is_svg and self.use_vectors_switch.get_active():
            return PassthroughSpec(
                active_layer_ids=self._get_active_layer_ids(),
                create_new_layers=self.create_new_layers_switch.get_active(),
            )
        else:
            if self.import_whole_image_switch.get_active():
                return TraceSpec(
                    threshold=1.0,
                    auto_threshold=False,
                    invert=False,
                )
            return TraceSpec(
                threshold=self.threshold_adjustment.get_value(),
                auto_threshold=self.auto_threshold_switch.get_active(),
                invert=self.invert_switch.get_active(),
            )

    def _schedule_preview_update(self, *args):
        if self._in_update:
            return
        logger.debug("Scheduling preview update")
        self.status_spinner.start()
        self.import_button.set_sensitive(False)

        # Dispatch task to TaskManager using FileCmd
        self.editor.task_manager.add_coroutine(
            self._update_preview_task, key="import-preview"
        )

    async def _update_preview_task(self, ctx):
        """
        Async task that calls the backend to generate the preview.
        """
        if not self._file_bytes:
            return

        spec = self._get_current_spec()
        ctx.set_message(_("Generating preview..."))

        result = await self.editor.file.generate_preview(
            self._file_bytes,
            self.file_path.name,
            self.mime_type,
            spec,
            PREVIEW_RENDER_SIZE_PX,
        )

        self.editor.task_manager.schedule_on_main_thread(
            self._update_ui_with_preview, result
        )

    def _update_ui_with_preview(self, result: Optional[PreviewResult]):
        """Updates the UI with the result of the preview task."""
        self._preview_result = result
        self._background_pixbuf = None

        if result and result.image_bytes:
            try:
                loader = GdkPixbuf.PixbufLoader.new()
                loader.write(result.image_bytes)
                loader.close()
                self._background_pixbuf = loader.get_pixbuf()
            except Exception:
                logger.error("Failed to create pixbuf from preview bytes.")

        self.preview_area.queue_draw()
        self.status_spinner.stop()
        self.import_button.set_sensitive(self._preview_result is not None)

        # Handle warnings/errors
        is_direct_vector = self.is_svg and self.use_vectors_switch.get_active()
        failed_generation = self._preview_result is None
        self.warning_banner.set_revealed(
            is_direct_vector and failed_generation
        )

    def _draw_checkerboard_background(
        self, ctx: cairo.Context, width: int, height: int
    ):
        """Fills the given context with a light gray checkerboard pattern."""
        CHECKER_SIZE = 10
        # Create a small surface to hold one tile of the pattern (2x2 checkers)
        tile_surface = cairo.ImageSurface(
            cairo.FORMAT_RGB24, CHECKER_SIZE * 2, CHECKER_SIZE * 2
        )
        tile_ctx = cairo.Context(tile_surface)

        # Color 1 (e.g., light gray)
        tile_ctx.set_source_rgb(0.85, 0.85, 0.85)
        tile_ctx.rectangle(0, 0, CHECKER_SIZE, CHECKER_SIZE)
        tile_ctx.fill()
        tile_ctx.rectangle(
            CHECKER_SIZE, CHECKER_SIZE, CHECKER_SIZE, CHECKER_SIZE
        )
        tile_ctx.fill()

        # Color 2 (e.g., slightly darker gray)
        tile_ctx.set_source_rgb(0.78, 0.78, 0.78)
        tile_ctx.rectangle(CHECKER_SIZE, 0, CHECKER_SIZE, CHECKER_SIZE)
        tile_ctx.fill()
        tile_ctx.rectangle(0, CHECKER_SIZE, CHECKER_SIZE, CHECKER_SIZE)
        tile_ctx.fill()

        pattern = cairo.SurfacePattern(tile_surface)
        pattern.set_extend(cairo.EXTEND_REPEAT)
        ctx.set_source(pattern)
        ctx.paint()

    def _on_draw_preview(
        self, area: Gtk.DrawingArea, ctx: cairo.Context, w, h
    ):
        """Draws the background image and vectors onto the preview area."""
        self._draw_checkerboard_background(ctx, w, h)

        if not self._preview_result or not self._background_pixbuf:
            return

        aspect_w = self._background_pixbuf.get_width()
        aspect_h = self._background_pixbuf.get_height()

        if aspect_w <= 0 or aspect_h <= 0:
            return

        # Calculate drawing area
        margin = 20
        view_w, view_h = w - 2 * margin, h - 2 * margin
        if view_w <= 0 or view_h <= 0:
            return

        scale = min(view_w / aspect_w, view_h / aspect_h)
        draw_w = aspect_w * scale
        draw_h = aspect_h * scale
        draw_x = (w - draw_w) / 2
        draw_y = (h - draw_h) / 2

        # 1. Draw Background Image
        ctx.save()
        ctx.translate(draw_x, draw_y)
        ctx.scale(draw_w / aspect_w, draw_h / aspect_h)
        Gdk.cairo_set_source_pixbuf(ctx, self._background_pixbuf, 0, 0)
        ctx.paint()
        ctx.restore()

        # 2. Draw Vectors
        # The backend returns a WorkPiece with normalized geometry (0-1).
        # We draw it over the full drawing area.
        payload = self._preview_result.payload
        if not payload or not payload.items:
            return

        ctx.save()
        ctx.translate(draw_x, draw_y)

        # Scale to the drawing area. Y is flipped to match standard coord
        # system. Geometry is Y-up, Cairo is Y-down.
        ctx.scale(draw_w, draw_h)
        ctx.translate(0, 1)
        ctx.scale(1, -1)

        max_dim = max(draw_w, draw_h)
        if max_dim > 0:
            ctx.set_line_width(2.0 / max_dim)

        def draw_item(item):
            if isinstance(item, WorkPiece) and item.boundaries:
                ctx.set_source_rgb(0.1, 0.5, 1.0)
                ctx.new_path()
                item.boundaries.to_cairo(ctx)
                ctx.stroke()
            elif isinstance(item, Layer):
                for child in item.children:
                    draw_item(child)

        for item in payload.items:
            draw_item(item)

        ctx.restore()

    def _on_import_clicked(self, button):
        final_spec = self._get_current_spec()
        logger.debug(f"_on_import_clicked: {final_spec}")
        self.response.send(self, response_id="import", spec=final_spec)
        self.close()
