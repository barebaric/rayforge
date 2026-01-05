import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, List
import cairo
from gi.repository import Adw, Gdk, GdkPixbuf, Gtk
from blinker import Signal
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.vectorization_spec import (
    TraceSpec,
    PassthroughSpec,
    VectorizationSpec,
)
from ...core.workpiece import WorkPiece
from ...core.layer import Layer
from ...image import ImportPayload, import_file_from_bytes
from ...image.svg.svgutil import extract_layer_manifest

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)

# A fixed, reasonable resolution for generating preview bitmaps.
PREVIEW_RENDER_SIZE_PX = 1024


class ImportDialog(Adw.Window):
    """
    A dialog for importing images with live preview of vectorization.
    Also handles SVG import options (direct vs. trace).
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
        self._preview_payload: Optional[ImportPayload] = None
        self._background_pixbuf: Optional[GdkPixbuf.Pixbuf] = None
        self._in_update = False  # Prevent signal recursion
        self._layer_widgets: List[Gtk.Switch] = []

        self.set_title(_("Import Image"))
        self.set_default_size(1100, 800)

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        header_bar = Adw.HeaderBar()
        main_box.append(header_bar)

        # Banner for SVG direct vector mode issues
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

        # Header Bar
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
        self.threshold_row.set_sensitive(False)  # Disabled by default
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
        self._on_import_mode_toggled(
            self.use_vectors_switch
        )  # Sets initial sensitivity and schedules first update

    def _on_import_mode_toggled(self, switch, *args):
        is_direct_import = self.is_svg and switch.get_active()
        self.trace_group.set_sensitive(not is_direct_import)
        self.layers_group.set_sensitive(
            is_direct_import
        )  # Only vector import supports specific layer select
        self.warning_banner.set_revealed(False)
        self._schedule_preview_update()

    def _on_switch_to_trace_clicked(self, banner):
        self.use_vectors_switch.set_active(False)

    def _on_auto_threshold_toggled(self, switch, _pspec):
        is_auto = switch.get_active()
        self.threshold_row.set_sensitive(not is_auto)
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

        layers = extract_layer_manifest(self._file_bytes)
        if not layers:
            return

        self.layers_group.set_visible(True)
        expander = Adw.ExpanderRow(title=_("Select Layers"), expanded=True)
        self.layers_group.add(expander)
        self._layer_widgets.clear()

        for layer in layers:
            row = Adw.ActionRow(title=layer["name"])
            switch = Gtk.Switch(active=True, valign=Gtk.Align.CENTER)
            # Use direct attribute assignment for the layer ID
            switch._layer_id = layer["id"]  # type: ignore
            switch.connect("notify::active", self._schedule_preview_update)

            row.add_suffix(switch)
            # Make the whole row click toggle the switch
            row.set_activatable_widget(switch)
            expander.add_row(row)

            self._layer_widgets.append(switch)

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
                active_layer_ids=self._get_active_layer_ids()
            )
        else:
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

        # Use the TaskManager to run the blocking import off the main thread
        self.editor.task_manager.add_coroutine(
            self._update_preview_task, key="raster-import-preview"
        )

    def _extract_workpiece(self, item) -> Optional[WorkPiece]:
        """
        Recursively extract the first WorkPiece from a potentially nested item.
        """
        if isinstance(item, WorkPiece):
            return item
        if isinstance(item, Layer):
            # Check direct children
            for child in item.children:
                wp = self._extract_workpiece(child)
                if wp:
                    return wp
        return None

    async def _update_preview_task(self, ctx):
        """
        Async task to generate a vector preview. This task handles both
        direct vector import and bitmap tracing modes.
        """
        if not self._file_bytes:
            return

        spec = self._get_current_spec()
        ctx.set_message(_("Generating preview..."))

        try:
            # 1. Get the vector geometry and metadata from the importer
            payload = await asyncio.to_thread(
                import_file_from_bytes,
                self._file_bytes,
                self.file_path.name,
                self.mime_type,
                spec,
            )

            if not payload or not payload.items:
                return self.editor.task_manager.schedule_on_main_thread(
                    self._update_ui_with_preview, (None, b"")
                )

            reference_wp = self._extract_workpiece(payload.items[0])
            if not reference_wp:
                raise ValueError("Imported items contain no WorkPieces")

            # 2. Generate a high-resolution base image for the preview.
            vips_image = None
            if isinstance(spec, TraceSpec):
                # For tracing, prioritize pre-rendered data from the importer
                # (e.g., SVG-trace). Fall back to the original file bytes for
                # standard rasters (PNG, PDF, etc.).
                image_bytes = (
                    payload.source.base_render_data or self._file_bytes
                )
                if not image_bytes:
                    raise ValueError(
                        "No image data available for tracing preview."
                    )

                full_image = pyvips.Image.new_from_buffer(image_bytes, "")

                # Apply cropping if the importer specified a crop window
                # (e.g., PDF auto-crop)
                if (
                    reference_wp.source_segment
                    and reference_wp.source_segment.crop_window_px
                ):
                    x, y, w, h = map(
                        int, reference_wp.source_segment.crop_window_px
                    )
                    vips_image = full_image.crop(x, y, w, h)
                else:
                    vips_image = full_image

            elif isinstance(spec, PassthroughSpec):
                # For direct SVG import, render the vector data at a high
                # resolution.
                if not payload.source.base_render_data:
                    raise ValueError(
                        "Importer did not provide SVG data for direct import "
                        "preview."
                    )
                # The `scale` parameter is key to getting a sharp render from
                # SVG data.
                vips_image = pyvips.Image.new_from_buffer(
                    payload.source.base_render_data, "", scale=4.0
                )

            if not vips_image:
                raise ValueError(
                    "Could not generate a base image for the preview."
                )

            # 3. Thumbnail the high-resolution base image for display
            preview_vips = vips_image.thumbnail_image(
                PREVIEW_RENDER_SIZE_PX,
                height=PREVIEW_RENDER_SIZE_PX,
                size="both",
            )

            if isinstance(spec, TraceSpec) and spec.invert:
                preview_vips = preview_vips.flatten(
                    background=[255, 255, 255]
                ).invert()

            png_bytes = preview_vips.pngsave_buffer()

            self.editor.task_manager.schedule_on_main_thread(
                self._update_ui_with_preview, (payload, png_bytes)
            )

        except Exception as e:
            logger.error(
                f"Failed to generate import preview: {e}", exc_info=True
            )
            self.editor.task_manager.schedule_on_main_thread(
                self._update_ui_with_preview, None
            )

    def _update_ui_with_preview(
        self, result: Optional[Tuple[Optional[ImportPayload], bytes]]
    ):
        """Updates the UI with the result of the preview task."""
        if result is None:
            self._preview_payload = None
            self._background_pixbuf = None
            logger.warning("Preview generation failed.")
        else:
            payload, image_bytes = result
            self._preview_payload = payload
            try:
                if image_bytes:
                    loader = GdkPixbuf.PixbufLoader.new()
                    loader.write(image_bytes)
                    loader.close()
                    self._background_pixbuf = loader.get_pixbuf()
                else:
                    self._background_pixbuf = None
            except Exception:
                self._background_pixbuf = None
                logger.error("Failed to create pixbuf from rendered PNG data.")

        self.preview_area.queue_draw()
        self.status_spinner.stop()
        self.import_button.set_sensitive(self._preview_payload is not None)

        # Show warning banner for SVG direct vector mode failures
        is_direct_vector = self.is_svg and self.use_vectors_switch.get_active()
        self.warning_banner.set_revealed(
            is_direct_vector and self._preview_payload is None
        )

        if self._preview_payload is None:
            logger.warning("Preview generation resulted in no payload.")

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

        # Create a pattern from the tile and set it to repeat
        pattern = cairo.SurfacePattern(tile_surface)
        pattern.set_extend(cairo.EXTEND_REPEAT)

        # Use the pattern as the source for the main context and paint
        ctx.set_source(pattern)
        ctx.paint()

    def _on_draw_preview(
        self, area: Gtk.DrawingArea, ctx: cairo.Context, w, h
    ):
        """Draws the background image and vectors onto the preview area."""
        # Use the helper to draw a checkerboard over the entire area
        self._draw_checkerboard_background(ctx, w, h)

        if not self._preview_payload or not self._preview_payload.items:
            return

        items = self._preview_payload.items
        if not items:
            return

        if not self._background_pixbuf:
            return

        aspect_w = self._background_pixbuf.get_width()
        aspect_h = self._background_pixbuf.get_height()

        if aspect_w <= 0 or aspect_h <= 0:
            return

        # Calculate drawing area based on the background image's aspect ratio
        margin = 20
        view_w, view_h = w - 2 * margin, h - 2 * margin
        if view_w <= 0 or view_h <= 0:
            return

        scale = min(view_w / aspect_w, view_h / aspect_h)
        draw_w = aspect_w * scale
        draw_h = aspect_h * scale
        draw_x = (w - draw_w) / 2
        draw_y = (h - draw_h) / 2

        ctx.save()
        ctx.translate(draw_x, draw_y)
        ctx.scale(draw_w / aspect_w, draw_h / aspect_h)

        # Step 1: Draw the background image.
        Gdk.cairo_set_source_pixbuf(ctx, self._background_pixbuf, 0, 0)
        ctx.paint()
        ctx.restore()

        # --- Draw Vector Overlay ---
        ctx.save()
        ctx.translate(draw_x, draw_y)

        # Set up the context to draw the Y-up normalized geometry
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

        for item in items:
            draw_item(item)

        ctx.restore()

    def _on_import_clicked(self, button):
        final_spec = self._get_current_spec()
        logger.debug(f"_on_import_clicked: {final_spec}")
        self.response.send(self, response_id="import", spec=final_spec)
        self.close()
