"""
Command module for handling drag-and-drop and clipboard paste operations.

This module encapsulates all drag-and-drop import functionality and clipboard
paste operations, keeping them separate from the core UI components.
"""
import logging
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from gi.repository import Gdk, Gtk, Gio, GLib, Adw

if TYPE_CHECKING:
    from ..mainwindow import MainWindow
    from ..workbench.surface import WorkSurface

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
        self._drag_active = False  # Track if drag is currently in progress
        self._drag_source = None  # Track the drag source for cleanup
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

    def setup_file_drop_target(self):
        """
        Configure the canvas to accept file drops for importing.
        Supports local files and file lists.
        """
        # Create drop target that accepts files
        drop_target = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
        drop_target.set_gtypes([Gio.File, Gdk.FileList])

        # Connect signals
        drop_target.connect("drop", self._on_file_dropped)
        drop_target.connect("enter", self._on_drag_enter)
        drop_target.connect("motion", self._on_drag_motion)
        drop_target.connect("leave", self._on_drag_leave)

        # Store reference for cleanup
        self._drag_source = drop_target

        # Attach to canvas widget
        self.surface.add_controller(drop_target)
        logger.debug("File drop target configured for WorkSurface")

    def _on_drag_enter(
        self, drop_target, x: float, y: float
    ) -> Gdk.DragAction:
        """
        Called when files are dragged over the canvas.
        Shows overlay message to provide visual feedback.
        """
        self._drag_active = True
        self._show_drop_overlay()
        logger.debug("Drag entered, overlay shown, drag_active=True")
        return Gdk.DragAction.COPY

    def _on_drag_motion(
        self, drop_target, x: float, y: float
    ) -> Gdk.DragAction:
        """
        Called when drag moves over the canvas.
        Ensures overlay stays visible and drag state is tracked.
        """
        # Keep drag active and ensure overlay is shown
        if not self._drag_active:
            logger.debug("Motion detected but drag not active, reactivating")
            self._drag_active = True
            self._show_drop_overlay()
        return Gdk.DragAction.COPY

    def _on_drag_leave(self, drop_target):
        """
        Called when drag leaves the canvas.
        Only removes overlay if drag is truly ending (defensive check).
        """
        # Don't immediately hide - GTK4 sometimes fires spurious leave events
        # We'll rely on drop or a delayed cleanup instead
        logger.debug("Drag leave signal received (may be spurious)")
        # Schedule a delayed check to hide overlay only if drag truly ended
        GLib.timeout_add(100, self._check_and_hide_overlay)

    def _show_drop_overlay(self):
        """Display 'Drop files to import' overlay on canvas."""
        if self._drop_overlay_label:
            return  # Already showing

        # Create overlay label with styling
        self._drop_overlay_label = Gtk.Label(label=_("Drop files to import"))
        self._drop_overlay_label.add_css_class("drop-overlay")
        self._drop_overlay_label.set_halign(Gtk.Align.CENTER)
        self._drop_overlay_label.set_valign(Gtk.Align.CENTER)

        # Make it semi-transparent and styled
        self._drop_overlay_label.set_opacity(0.9)

        # Find the parent overlay (surface_overlay from MainWindow)
        overlay_parent = self._find_parent_overlay()
        if overlay_parent:
            overlay_parent.add_overlay(self._drop_overlay_label)
            logger.debug("Drop overlay added to parent Gtk.Overlay")
        else:
            logger.warning("Could not find parent overlay for drop message")

    def _check_and_hide_overlay(self) -> bool:
        """
        Check if drag is still active before hiding overlay.
        Returns False to not repeat the timeout.
        """
        if not self._drag_active:
            self._hide_drop_overlay()
            logger.debug("Delayed hide: drag no longer active, removing overlay")
        else:
            logger.debug("Delayed hide: drag still active, keeping overlay")
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

    def _on_file_dropped(
        self, drop_target, value, x: float, y: float
    ) -> bool:
        """
        Handle files dropped onto the canvas.

        Args:
            drop_target: The Gtk.DropTarget that received the drop
            value: Either a Gio.File or Gdk.FileList
            x, y: Widget coordinates (pixels) where drop occurred

        Returns:
            True if drop was handled successfully, False otherwise
        """
        # Mark drag as no longer active
        self._drag_active = False
        logger.debug("Drop received, drag_active=False")

        # Hide overlay immediately
        self._hide_drop_overlay()

        try:
            # Convert widget coordinates to world coordinates (mm)
            world_x_mm, world_y_mm = self.surface._get_world_coords(x, y)
            logger.info(
                f"File(s) dropped at widget coords ({x:.1f}, {y:.1f}) "
                f"→ world coords ({world_x_mm:.2f}, {world_y_mm:.2f}) mm"
            )

            # Extract file list from value
            files = self._extract_files_from_drop_value(value)
            if not files:
                return False

            # Get file info for all dropped files
            file_infos = self._get_file_infos(files)

            # Delay file import slightly to ensure drag operation fully completes
            # This prevents modal dialogs from interfering with drag cleanup
            GLib.timeout_add(150, self._import_dropped_files_delayed,
                           file_infos, (world_x_mm, world_y_mm))

            return True

        except Exception as e:
            logger.exception(f"Error handling dropped file: {e}")
            return False

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
        from ..image import importer_by_mime_type

        file_infos = []
        for gfile in files:
            path_str = gfile.get_path()
            if not path_str:
                logger.warning("File has no path, skipping")
                continue

            file_path = Path(path_str)
            file_info = gfile.query_info(
                Gio.FILE_ATTRIBUTE_STANDARD_CONTENT_TYPE,
                Gio.FileQueryInfoFlags.NONE,
                None,
            )
            mime_type = file_info.get_content_type()

            # Check if we support this MIME type
            if mime_type not in importer_by_mime_type:
                logger.warning(
                    f"Unsupported file type: {mime_type} for {file_path}"
                )
                continue

            file_infos.append((file_path, mime_type))

        return file_infos

    def _import_dropped_files_delayed(self, file_infos, position_mm) -> bool:
        """
        Delayed wrapper for importing files after drag completes.
        Returns False to not repeat the timeout.
        """
        self._import_dropped_files(file_infos, position_mm)
        return False  # Don't repeat

    def _import_dropped_files(self, file_infos, position_mm):
        """
        Import dropped files, separating SVGs from rasters for
        appropriate handling.

        Args:
            file_infos: List of (file_path, mime_type) tuples
            position_mm: (x, y) tuple in world coordinates
        """
        from ..doceditor.ui import import_handler

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
                # Multiple raster files - use batch import
                world_x_mm, world_y_mm = position_mm
                logger.info(
                    f"Batch importing {len(raster_files)} raster files at "
                    f"({world_x_mm:.2f}, {world_y_mm:.2f}) mm"
                )
                import_handler.import_multiple_rasters_at_position(
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

        # Check for any supported image formats
        has_image = (
            formats.contain_mime_type("image/png") or
            formats.contain_mime_type("image/jpeg") or
            formats.contain_mime_type("image/bmp")
        )

        if has_image:
            # Import image from clipboard
            self._import_image_from_clipboard()
            # Clear the clipboard so next paste uses workpiece data
            clipboard.set_content(None)
            return True

        return False  # Let caller handle workpiece paste

    def _import_image_from_clipboard(self):
        """Import image data from the clipboard."""
        clipboard = self.main_window.get_clipboard()

        # Callback for when texture is read from clipboard
        def on_texture_ready(clipboard, result):
            try:
                texture = clipboard.read_texture_finish(result)
                if not texture:
                    logger.warning("Failed to read texture from clipboard")
                    return

                # Process image in background thread
                thread = threading.Thread(
                    target=self._save_and_import_texture,
                    args=(texture,),
                    daemon=True
                )
                thread.start()

            except Exception as e:
                logger.exception(f"Failed to read clipboard texture: {e}")
                self._show_clipboard_error()

        # Start async clipboard read with callback
        clipboard.read_texture_async(None, on_texture_ready)

    def _save_and_import_texture(self, texture):
        """
        Save texture to temporary file and schedule import on main thread.

        Args:
            texture: GdkTexture from clipboard
        """
        try:
            # Save texture to temporary file
            temp_path = self._save_texture_to_temp_file(texture)
            if not temp_path:
                return

            logger.info(f"Saved clipboard image to {temp_path}")

            # Schedule import on main thread
            GLib.idle_add(
                self._import_temp_file_and_cleanup,
                temp_path
            )

        except Exception as e:
            logger.exception(f"Failed to save clipboard image: {e}")
            GLib.idle_add(self._show_clipboard_error)

    def _save_texture_to_temp_file(self, texture) -> Optional[Path]:
        """
        Save GdkTexture to a temporary PNG file.

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
            # Calculate center position of canvas
            from ..config import config

            machine = config.machine
            if machine:
                center_x = machine.dimensions[0] / 2
                center_y = machine.dimensions[1] / 2
            else:
                center_x, center_y = 50.0, 50.0  # Fallback

            # Import the temporary file
            from ..doceditor.ui import import_handler
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
