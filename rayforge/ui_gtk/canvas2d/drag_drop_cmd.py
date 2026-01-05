"""
Command module for handling drag-and-drop and clipboard paste operations.

This module encapsulates all drag-and-drop import functionality and clipboard
paste operations, keeping them separate from the core UI components.
"""

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple
from gi.repository import GObject, Gdk, Gtk, Gio, GLib, Adw
from ...context import get_context
from ...core.sketcher import Sketch
from ...image import bitmap_mime_types

if TYPE_CHECKING:
    from ...ui_gtk.mainwindow import MainWindow
    from .surface import WorkSurface

logger = logging.getLogger(__name__)


class DragDropCmd:
    """Handles drag-and-drop file imports and clipboard paste operations."""

    def __init__(self, main_window: "MainWindow", surface: "WorkSurface"):
        """
        Initialize the drag-drop command handler.

        Args:
            main_window: The main application window
            surface: The WorkSurface canvas widget
        """
        self.main_window = main_window
        self.surface = surface
        self._drop_overlay_label: Optional[Gtk.Label] = None

        # Keep references to controllers
        self._sketch_target: Optional[Gtk.DropTarget] = None
        self._file_target: Optional[Gtk.DropTarget] = None

        self._apply_drop_overlay_css()

    def _apply_drop_overlay_css(self):
        """Apply CSS styling for the drop overlay."""
        display = Gdk.Display.get_default()

        # CSS for drop overlay
        drop_overlay_css = """
        .drop-overlay {
            font-size: 24px;
            font-weight: bold;
            color: white;
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 12px;
            padding: 24px 48px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        """

        if display:
            provider = Gtk.CssProvider()
            provider.load_from_string(drop_overlay_css)
            Gtk.StyleContext.add_provider_for_display(
                display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )

    def setup_drop_targets(self):
        """
        Configure the canvas to accept file drops for importing.
        Supports local files and file lists, as well as internal Sketch
        objects.
        """
        # We use separate drop targets for Sketches (Strings) and Files.
        # This avoids issues with mixed types in a single controller.

        # --- 1. Sketch Target (Strings) ---
        self._sketch_target = Gtk.DropTarget.new(
            GObject.TYPE_STRING, Gdk.DragAction.COPY
        )
        self._sketch_target.connect("drop", self._on_sketch_drop)
        self._sketch_target.connect("enter", self._on_sketch_drag_enter)
        self.surface.add_controller(self._sketch_target)

        # --- 2. File Target (Files & FileLists) ---
        # Initialize with a valid type (Gio.File) then extend to FileList
        self._file_target = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
        self._file_target.set_gtypes([Gio.File, Gdk.FileList])
        self._file_target.connect("drop", self._on_file_drop)
        self._file_target.connect("enter", self._on_file_drag_enter)
        self._file_target.connect("leave", self._on_drag_leave)
        self.surface.add_controller(self._file_target)

        logger.debug(
            "Split drop targets (Sketch/File) configured for WorkSurface"
        )

    # --- Sketch Handlers ---

    def _on_sketch_drag_enter(self, drop_target, x, y):
        # We accept copy for sketches. No overlay needed.
        logger.debug("Sketch drag entered surface")
        return Gdk.DragAction.COPY

    def _on_sketch_drop(self, drop_target, value, x, y):
        logger.debug(f"Sketch drop event: value={value}")
        if isinstance(value, str):
            # Convert widget coordinates to world coordinates (mm)
            world_x_mm, world_y_mm = self.surface._get_world_coords(x, y)
            return self._handle_sketch_drop(value, (world_x_mm, world_y_mm))
        return False

    # --- File Handlers ---

    def _on_file_drag_enter(self, drop_target, x, y):
        # Show overlay for files
        self._show_drop_overlay()
        return Gdk.DragAction.COPY

    def _on_drag_leave(self, drop_target):
        # Hide overlay
        if self._drop_overlay_label:
            logger.debug("Drag leave signal received, scheduling delayed hide")
            GLib.timeout_add(100, self._delayed_hide_overlay)

    def _on_file_drop(self, drop_target, value, x, y):
        self._hide_drop_overlay()

        logger.debug(f"File drop event: type={type(value)}")

        # Convert widget coordinates to world coordinates (mm)
        world_x_mm, world_y_mm = self.surface._get_world_coords(x, y)

        files = self._extract_files_from_drop_value(value)
        if files:
            logger.info(
                f"Processing file drop at world coords "
                f"({world_x_mm:.2f}, {world_y_mm:.2f}) mm"
            )
            file_infos = self._get_file_infos(files)
            self._import_dropped_files(file_infos, (world_x_mm, world_y_mm))
            return True

        return False

    # --- Overlay & Helper Methods ---

    def _show_drop_overlay(self):
        """Display 'Drop files to import' overlay on canvas."""
        if self._drop_overlay_label:
            return  # Already showing

        # Create overlay label with styling
        self._drop_overlay_label = Gtk.Label(label=_("Drop files to import"))
        self._drop_overlay_label.add_css_class("drop-overlay")
        self._drop_overlay_label.set_halign(Gtk.Align.CENTER)
        self._drop_overlay_label.set_valign(Gtk.Align.CENTER)

        # Make it semi-transparent
        self._drop_overlay_label.set_opacity(0.9)

        # Find the parent overlay (surface_overlay from MainWindow)
        overlay_parent = self._find_parent_overlay()
        if overlay_parent:
            overlay_parent.add_overlay(self._drop_overlay_label)
            logger.debug("Drop overlay added to parent Gtk.Overlay")
        else:
            logger.warning("Could not find parent overlay for drop message")

    def _delayed_hide_overlay(self) -> bool:
        """
        Hide overlay after a delay. Returns False to not repeat the timeout.
        """
        self._hide_drop_overlay()
        logger.debug("Delayed hide executed, overlay removed")
        return False  # Don't repeat

    def _hide_drop_overlay(self):
        """Remove the drop overlay from canvas. Safe to call multiple times."""
        if not self._drop_overlay_label:
            return  # Already removed or never created

        try:
            overlay_parent = self._find_parent_overlay()
            if overlay_parent:
                overlay_parent.remove_overlay(self._drop_overlay_label)
            self._drop_overlay_label = None
            logger.debug("Drop overlay removed")
        except Exception as e:
            logger.warning(f"Error removing drop overlay: {e}")
            self._drop_overlay_label = None  # Clear reference anyway

    def _find_parent_overlay(self):
        """Find the Gtk.Overlay parent that contains this canvas."""
        widget = self.surface.get_parent()
        while widget:
            if isinstance(widget, Gtk.Overlay):
                return widget
            widget = widget.get_parent()
        return None

    def _extract_files_from_drop_value(self, value):
        """Extract file list from drop value."""
        files = []
        if isinstance(value, Gdk.FileList):
            files = value.get_files()
        elif isinstance(value, Gio.File):
            files = [value]
        else:
            logger.warning(f"Unexpected drop value type: {type(value)}")
            return []

        if not files:
            logger.warning("No files in drop")
            return []

        return files

    def _get_file_infos(self, files):
        """Get file path and MIME type information for dropped files."""
        from ...image import importer_by_mime_type

        file_infos = []
        for gfile in files:
            path_str = gfile.get_path()
            if not path_str:
                logger.warning("File has no path, skipping")
                continue

            file_path = Path(path_str)
            try:
                file_info = gfile.query_info(
                    Gio.FILE_ATTRIBUTE_STANDARD_CONTENT_TYPE,
                    Gio.FileQueryInfoFlags.NONE,
                    None,
                )
                mime_type = file_info.get_content_type()
            except Exception as e:
                logger.warning(
                    f"Could not query file info for {file_path}: {e}"
                )
                continue

            # Check if we support this MIME type
            if mime_type not in importer_by_mime_type:
                logger.warning(
                    f"Unsupported file type: {mime_type} for {file_path}"
                )
                continue

            file_infos.append((file_path, mime_type))

        return file_infos

    def _import_dropped_files(self, file_infos, position_mm):
        """
        Import dropped files, separating SVGs from rasters for
        appropriate handling.

        Args:
            file_infos: List of (file_path, mime_type) tuples
            position_mm: (x, y) tuple in world coordinates
        """
        from ..doceditor import import_handler

        # Separate SVGs from raster files
        svg_files = []
        raster_files = []

        for file_path, mime_type in file_infos:
            if mime_type == "image/svg+xml":
                svg_files.append((file_path, mime_type))
            else:
                # All other supported types go to raster
                raster_files.append((file_path, mime_type))

        # Handle SVG files individually (each gets its own dialog)
        for file_path, mime_type in svg_files:
            world_x_mm, world_y_mm = position_mm
            logger.info(
                f"Importing SVG: {file_path} at "
                f"({world_x_mm:.2f}, {world_y_mm:.2f}) mm"
            )
            import_handler.import_file_at_position(
                self.main_window,
                self.surface.editor,
                file_path,
                mime_type,
                position_mm,
            )

        # Handle raster files with batch configuration
        if raster_files:
            if len(raster_files) == 1:
                # Single raster file - use standard import
                file_path, mime_type = raster_files[0]
                world_x_mm, world_y_mm = position_mm
                logger.info(
                    f"Importing raster: {file_path} at "
                    f"({world_x_mm:.2f}, {world_y_mm:.2f}) mm"
                )
                import_handler.import_file_at_position(
                    self.main_window,
                    self.surface.editor,
                    file_path,
                    mime_type,
                    position_mm,
                )
            else:
                # Multiple files - use batch import
                world_x_mm, world_y_mm = position_mm
                logger.info(
                    f"Batch importing {len(raster_files)} files at "
                    f"({world_x_mm:.2f}, {world_y_mm:.2f}) mm"
                )
                import_handler.import_multiple_files_at_position(
                    self.main_window,
                    self.surface.editor,
                    raster_files,
                    position_mm,
                )

    def handle_clipboard_paste(self):
        """
        Handle paste operation, checking clipboard for image data first,
        then falling back to workpiece paste.
        """
        # Priority 1: Check if system clipboard contains image data
        clipboard = self.main_window.get_clipboard()
        formats = clipboard.get_formats()

        # Check for any supported bitmap image formats
        has_image = any(
            formats.contain_mime_type(mime_type)
            for mime_type in bitmap_mime_types
        )

        if has_image:
            # Import image from clipboard asynchronously
            self._import_image_from_clipboard()
            return True

        return False  # Let caller handle workpiece paste

    def _import_image_from_clipboard(self):
        """
        Asynchronously read an image from the clipboard and import it.
        This entire process is thread-safe.
        """
        clipboard = self.main_window.get_clipboard()

        # This callback is guaranteed to run on the main GTK thread.
        def on_texture_ready(source_obj, result):
            try:
                texture = source_obj.read_texture_finish(result)
                if not texture:
                    logger.warning("Failed to read texture from clipboard")
                    self._show_clipboard_error()
                    return

                # Safely save the texture to a file from the main thread.
                temp_path = self._save_texture_to_temp_file(texture)
                if not temp_path:
                    self._show_clipboard_error()
                    return

                logger.info(f"Saved clipboard image to {temp_path}")

                # Import the file and schedule it for future cleanup.
                self._import_temp_file_and_cleanup(temp_path)

                # Now that the data has been successfully read and saved,
                # we can safely clear the clipboard content.
                source_obj.set_content(None)

            except GLib.Error as e:
                # This can happen if clipboard content changes during read.
                logger.warning(f"GLib error reading clipboard texture: {e}")
                self._show_clipboard_error()
            except Exception as e:
                logger.exception(f"Failed to process clipboard texture: {e}")
                self._show_clipboard_error()

        # Start the asynchronous clipboard read.
        clipboard.read_texture_async(None, on_texture_ready)

    def _save_texture_to_temp_file(self, texture) -> Optional[Path]:
        """
        Save GdkTexture to a temporary PNG file.
        MUST be called from the main GTK thread.

        Args:
            texture: GdkTexture to save

        Returns:
            Path to temporary file, or None on failure
        """
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".png"
            ) as tmp_file:
                temp_path = Path(tmp_file.name)

            # Get pixbuf from texture and save as PNG
            pixbuf = Gdk.pixbuf_get_from_texture(texture)
            if not pixbuf:
                logger.warning("Failed to convert texture to pixbuf")
                return None

            pixbuf.savev(str(temp_path), "png", [], [])
            return temp_path

        except Exception as e:
            logger.error(f"Failed to save texture: {e}")
            return None

    def _import_temp_file_and_cleanup(self, temp_path: Path) -> bool:
        """
        Import temporary file and schedule cleanup.
        Runs on main thread.

        Args:
            temp_path: Path to temporary file

        Returns:
            False (to not repeat GLib.idle_add)
        """
        try:
            machine = get_context().machine
            if machine:
                center_x = machine.dimensions[0] / 2
                center_y = machine.dimensions[1] / 2
            else:
                center_x, center_y = 50.0, 50.0  # Fallback

            # Import the temporary file
            from ..doceditor import import_handler

            import_handler.import_file_at_position(
                self.main_window,
                self.main_window.doc_editor,
                temp_path,
                "image/png",
                (center_x, center_y),
            )

            # Schedule cleanup after delay
            GLib.timeout_add_seconds(5, self._cleanup_temp_file, temp_path)

            # Show success notification
            self.main_window.toast_overlay.add_toast(
                Adw.Toast.new(_("Image imported from clipboard"))
            )

        except Exception as e:
            logger.exception(f"Failed to import from clipboard: {e}")
            self._show_clipboard_error()

        return False  # Don't repeat

    def _cleanup_temp_file(self, temp_path: Path) -> bool:
        """
        Clean up temporary file.

        Args:
            temp_path: Path to file to delete

        Returns:
            False (to not repeat GLib.timeout_add_seconds)
        """
        try:
            temp_path.unlink()
            logger.debug(f"Cleaned up clipboard temp file: {temp_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp file: {e}")

        return False  # Don't repeat

    def _show_clipboard_error(self) -> bool:
        """
        Show error notification for clipboard import failure.

        Returns:
            False (to not repeat GLib.idle_add)
        """
        self.main_window.toast_overlay.add_toast(
            Adw.Toast.new(_("Failed to import image from clipboard"))
        )
        return False

    def _handle_sketch_drop(
        self, sketch_uid: str, position_mm: Tuple[float, float]
    ) -> bool:
        """
        Handle a sketch UID dropped onto the canvas.

        Args:
            sketch_uid: The UID of the sketch being dropped
            position_mm: The (x, y) position in mm where to place the instance

        Returns:
            True if the drop was handled successfully
        """
        try:
            # Retrieve the sketch asset from the document
            doc = self.main_window.doc_editor.doc
            sketch = doc.get_asset_by_uid(sketch_uid)

            if not isinstance(sketch, Sketch):
                logger.warning(
                    f"Dropped sketch UID {sketch_uid} not found in document"
                )
                return False

            # Check if the sketch is empty.
            # A fresh sketch always has 1 point (the origin), so we check for
            # entities (lines, arcs, circles) to determine if it has content.
            if sketch.is_empty:
                dialog = Adw.MessageDialog(
                    transient_for=self.main_window,
                    heading=_("Empty Sketch"),
                    body=_(
                        "The selected sketch contains no geometry. Please "
                        "edit the sketch to add lines or shapes before "
                        "placing on the canvas."
                    ),
                )
                dialog.add_response("close", _("Close"))
                dialog.present()
                return False

            edit_cmd = self.main_window.doc_editor.edit

            # Create the sketch instance at the drop position
            new_workpiece = edit_cmd.add_sketch_instance(
                sketch_uid, position_mm
            )

            if new_workpiece:
                logger.info(
                    f"Created sketch instance {new_workpiece.uid[:8]} "
                    f"from sketch {sketch_uid[:8]} at {position_mm}"
                )
                return True
            else:
                logger.warning(
                    f"Failed to create sketch instance from {sketch_uid}"
                )
                return False

        except Exception as e:
            logger.exception(f"Error handling sketch drop: {e}")
            return False
