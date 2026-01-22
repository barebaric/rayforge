import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set

import cairo
from blinker import Signal
from gi.repository import Adw, Gdk, GdkPixbuf, Gtk

from ...core.item import DocItem
from ...core.layer import Layer
from ...core.vectorization_spec import (
    PassthroughSpec,
    TraceSpec,
    VectorizationSpec,
)
from ...core.workpiece import WorkPiece
from ...doceditor.file_cmd import PreviewResult
from ...image.base_importer import ImporterFeature
from ...image.structures import ImportManifest
from ..shared.patched_dialog_window import PatchedDialogWindow
from ...core.matrix import Matrix

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
        features: Set[ImporterFeature],
    ):
        super().__init__(transient_for=parent, modal=True)
        self.editor = editor
        self.file_path = file_path
        self.mime_type = mime_type
        self.features = features
        self.response = Signal()

        # Internal state
        self._file_bytes: Optional[bytes] = None
        self._manifest: Optional[ImportManifest] = None
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
                "The file produced no output in direct vector mode. "
                "Files containing text or other non-path elements "
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

        # Import Mode Group (for files with multiple import options)
        mode_group = Adw.PreferencesGroup(title=_("Import Mode"))
        preferences_page.add(mode_group)

        self.use_vectors_switch = Adw.SwitchRow(
            title=_("Use Original Vectors"),
            subtitle=_("Import vector data directly"),
            active=True,
        )
        self.use_vectors_switch.connect(
            "notify::active", self._on_import_mode_toggled
        )
        mode_group.add(self.use_vectors_switch)
        # Show this group only if the importer supports both vector and trace
        can_trace = ImporterFeature.BITMAP_TRACING in self.features
        can_vector = ImporterFeature.DIRECT_VECTOR in self.features
        mode_group.set_visible(can_trace and can_vector)

        # Layers Group (Dynamic)
        self.layers_group = Adw.PreferencesGroup(title=_("Layers"))
        self.layers_group.set_visible(False)
        preferences_page.add(self.layers_group)

        # Trace Settings Group
        self.trace_group = Adw.PreferencesGroup(title=_("Trace Settings"))
        preferences_page.add(self.trace_group)
        self.trace_group.set_visible(can_trace)

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
        is_direct_import = (
            ImporterFeature.DIRECT_VECTOR in self.features
            and switch.get_active()
        )
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
            # Use the new generic scan method
            self._manifest = self.editor.file.scan_import_file(
                self._file_bytes, self.file_path, self.mime_type
            )
            # Log any warnings from the scan
            if self._manifest and self._manifest.warnings:
                for warning in self._manifest.warnings:
                    logger.warning(
                        f"Scan warning for {self.file_path.name}: {warning}"
                    )
            self._populate_layers_ui()
        except Exception:
            logger.error(
                f"Failed to read import file {self.file_path}", exc_info=True
            )
            self.close()

    def _populate_layers_ui(self):
        if not self._manifest or not self._manifest.layers:
            return

        self.layers_group.set_visible(True)
        expander = Adw.ExpanderRow(title=_("Select Layers"), expanded=True)
        self.layers_group.add(expander)
        self._layer_widgets.clear()

        for layer_info in self._manifest.layers:
            row = Adw.ActionRow(title=layer_info.name)

            count = layer_info.feature_count
            is_empty = count is not None and count == 0

            # Configure row subtitle based on content
            if is_empty:
                row.set_subtitle(_("Layer is empty"))
                row.set_sensitive(False)
            elif count is not None:
                row.set_subtitle(_("Layer with {n} vectors").format(n=count))

            switch = Gtk.Switch(
                active=not is_empty,
                valign=Gtk.Align.CENTER,
            )
            switch.set_sensitive(not is_empty)
            switch._layer_id = layer_info.id  # type: ignore
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
        if (
            ImporterFeature.DIRECT_VECTOR in self.features
            and self.use_vectors_switch.get_active()
        ):
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
        is_direct_vector = (
            ImporterFeature.DIRECT_VECTOR in self.features
            and self.use_vectors_switch.get_active()
        )
        failed_generation = (
            self._preview_result is None
            or self._preview_result.payload is None
            or not self._preview_result.payload.items
        )
        can_trace = ImporterFeature.BITMAP_TRACING in self.features
        # Only show warning if switching to trace mode is possible
        self.warning_banner.set_revealed(
            is_direct_vector and failed_generation and can_trace
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
        self, area: Gtk.DrawingArea, ctx: cairo.Context, w: int, h: int
    ):
        """
        Draws the preview using the authoritative frame of reference and
        pre-calculated transforms provided by the backend. This method
        contains no format-specific logic.
        """
        self._draw_checkerboard_background(ctx, w, h)

        if (
            not self._preview_result
            or not self._background_pixbuf
            or not self._preview_result.parse_result
            or not self._preview_result.payload
        ):
            return

        parse_result = self._preview_result.parse_result
        payload = self._preview_result.payload

        # The backend MUST provide the frame and background transform.
        assert parse_result.world_frame_of_reference is not None
        assert parse_result.background_world_transform is not None

        # --- 1. Establish the World-to-Canvas Transform ---
        frame_x, frame_y, frame_w, frame_h = (
            parse_result.world_frame_of_reference
        )
        if frame_w <= 1e-9 or frame_h <= 1e-9:
            return

        margin = 20
        view_w, view_h = w - 2 * margin, h - 2 * margin
        if view_w <= 0 or view_h <= 0:
            return

        scale = min(view_w / frame_w, view_h / frame_h)

        # This matrix maps the Y-Up world space to the Y-Down canvas space,
        # centering the world frame in the drawing area.
        world_to_canvas = (
            Matrix.translation(w / 2, h / 2)
            @ Matrix.scale(scale, -scale)
            @ Matrix.translation(
                -(frame_x + frame_w / 2), -(frame_y + frame_h / 2)
            )
        )

        # --- 2. Draw the Background Image ---
        ctx.save()

        # The background's transform maps a 1x1 unit square to its place
        # in the Y-Up world. We compose it with the master transform to get
        # its final position on the Y-Down canvas.
        final_bg_transform = (
            world_to_canvas @ parse_result.background_world_transform
        )

        # The transform positions a 1x1 unit square. We need to find the
        # top-left corner and the size on the canvas.
        top_left = final_bg_transform.transform_point((0, 1))
        top_right = final_bg_transform.transform_point((1, 1))
        bottom_left = final_bg_transform.transform_point((0, 0))

        canvas_w = top_right[0] - top_left[0]
        canvas_h = bottom_left[1] - top_left[1]

        # Draw the pixbuf directly into its calculated canvas rectangle.
        ctx.translate(top_left[0], top_left[1])
        ctx.scale(
            canvas_w / self._background_pixbuf.get_width(),
            canvas_h / self._background_pixbuf.get_height(),
        )
        Gdk.cairo_set_source_pixbuf(ctx, self._background_pixbuf, 0, 0)
        ctx.paint()
        ctx.restore()

        # --- 3. Draw Vector Overlays ---
        ctx.save()
        # Set the master transform for all vector drawing.
        ctx.transform(cairo.Matrix(*world_to_canvas.for_cairo()))

        def draw_item(item: DocItem):
            if isinstance(item, WorkPiece) and item.boundaries:
                ctx.save()
                # Apply the item's personal world matrix.
                ctx.transform(
                    cairo.Matrix(*item.get_world_transform().for_cairo())
                )

                # Set line width to a consistent 1.5px in device space.
                px, py = ctx.device_to_user_distance(1.5, 1.5)
                ctx.set_line_width(max(abs(px), abs(py)))
                ctx.set_source_rgb(0.1, 0.5, 1.0)  # Blue for vectors
                ctx.new_path()
                item.boundaries.to_cairo(ctx)
                ctx.stroke()
                ctx.restore()
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
