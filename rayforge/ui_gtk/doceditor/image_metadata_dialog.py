import logging
from typing import Any, Optional, List, Tuple
from gi.repository import Gtk, Adw, Pango, Gdk
from ..icons import get_icon

logger = logging.getLogger(__name__)


class ImageMetadataDialog(Adw.Window):
    """
    A dialog that displays image metadata in a clean, organized format.
    """

    def __init__(self, parent: Optional[Gtk.Window] = None):
        super().__init__()
        self.set_title(_("Image Metadata"))
        self.set_transient_for(parent)
        self.set_modal(False)

        # Set a reasonable default size
        self.set_default_size(600, 500)

        # Create a vertical box to hold the header bar and the content
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        # Add a header bar for title and window controls
        self.header_bar = Adw.HeaderBar()
        main_box.append(self.header_bar)

        # Add copy button to header bar
        self.copy_button = Gtk.Button(child=get_icon("copy-symbolic"))
        self.copy_button.set_tooltip_text(_("Copy Metadata"))
        self.copy_button.connect("clicked", self._on_copy_clicked)
        self.header_bar.pack_end(self.copy_button)

        # The main content area should be scrollable
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        scrolled_window.set_vexpand(True)
        main_box.append(scrolled_window)

        # Create a preferences page and add it to the scrollable area
        self.scrolled_window = scrolled_window
        self.page = Adw.PreferencesPage()
        scrolled_window.set_child(self.page)

        # Status label for when no metadata is available
        self.status_label = Gtk.Label(label=_("No metadata available"))
        self.status_label.add_css_class("dim-label")
        self.status_label.set_halign(Gtk.Align.CENTER)
        self.status_label.set_valign(Gtk.Align.CENTER)
        self.status_label.set_visible(False)

        # Add a key controller to close the dialog on Escape press
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

    def set_metadata(self, import_source):
        """
        Sets the metadata to display in the dialog.

        Args:
            import_source: ImportSource object with explicitly modeled
                attributes
        """
        # Store import_source for clipboard operations
        self.import_source = import_source
        metadata = import_source.metadata
        filename = import_source.source_file.name

        # Update window title
        self.set_title(f"{filename} - Image Metadata")

        # Clear existing content by creating a new page
        self.page = Adw.PreferencesPage()
        self.scrolled_window.set_child(self.page)

        self.status_label.set_visible(False)

        # Group metadata by category
        metadata_info = []

        logger.debug(f"Processing metadata with {len(metadata)} items")
        for key, value in metadata.items():
            # All metadata attributes (except basic) go to metadata_info
            metadata_info.append((key, value))

        # Create sections for each category
        self._create_basic_section(import_source)

        if metadata_info:
            self._create_metadata_section(metadata_info)

    def _create_basic_section(self, import_source):
        """
        Creates the Basic Information section directly from import_source.
        """
        # Create preferences group for basic information
        group = Adw.PreferencesGroup()
        group.set_title(_("Basic Information"))
        group.set_description(
            _("Basic image properties like dimensions and format.")
        )

        # Add Source File row
        row = Adw.ActionRow()
        row.set_title("Source File")
        value_label = Gtk.Label(label=str(import_source.source_file))
        row.add_suffix(value_label)
        group.add(row)

        # Add UID row
        row = Adw.ActionRow()
        row.set_title("UID")
        value_label = Gtk.Label(label=import_source.uid)
        row.add_suffix(value_label)
        group.add(row)

        # Add Renderer row
        row = Adw.ActionRow()
        row.set_title("Renderer")
        value_label = Gtk.Label(
            label=import_source.renderer.__class__.__name__
        )
        row.add_suffix(value_label)
        group.add(row)

        # Add Vector Config row if applicable
        if import_source.vector_config:
            row = Adw.ActionRow()
            row.set_title("Vector Config")
            value_label = Gtk.Label(label="Configured")
            row.add_suffix(value_label)
            group.add(row)

        self.page.add(group)

    def _create_metadata_section(self, items: List[Tuple[str, Any]]):
        """
        Creates the Metadata section containing all metadata attributes.

        Args:
            items: List of (key, value) tuples for metadata
        """
        # Create preferences group for metadata
        group = Adw.PreferencesGroup()
        group.set_title(_("Metadata"))
        group.set_description(_("All metadata extracted from the image."))

        # Add key-value pairs as action rows
        for key, value in items:
            row = Adw.ActionRow()
            row.set_title(key)

            # Format value for display
            value_str = self._format_value(value)

            # Create value label
            value_label = Gtk.Label(label=value_str)
            value_label.set_ellipsize(Pango.EllipsizeMode.END)
            value_label.set_xalign(0.0)
            value_label.add_css_class("dim-label")

            # Add value label to row
            row.add_suffix(value_label)
            group.add(row)

        self.page.add(group)

    def _format_value(self, value: Any) -> str:
        """Formats a metadata value for display."""
        if value is None:
            return "N/A"
        elif isinstance(value, bool):
            return "Yes" if value else "No"
        elif isinstance(value, bytes):
            return f"Binary data ({len(value)} bytes)"
        elif isinstance(value, str) and value.startswith("<binary data"):
            # Already formatted binary data from image_util
            return value
        elif isinstance(value, str) and len(value) > 200:
            return f"Text data ({len(value)} characters)"
        elif isinstance(value, (list, tuple)):
            if len(value) > 10:
                return f"[{len(value)} items]"
            return ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            return f"[Dictionary with {len(value)} keys]"
        else:
            return str(value)

    def _on_copy_clicked(self, button):
        """Copy all metadata to clipboard."""
        display = Gdk.Display.get_default()
        if display:
            clipboard = display.get_clipboard()
        else:
            return

        # Get metadata and filename from import_source
        metadata = self.import_source.metadata
        filename = self.import_source.source_file.name

        # Format metadata as text
        text_parts = []
        if filename:
            text_parts.append(f"File: {filename}")
            text_parts.append("")

        # Group metadata by category
        metadata_info = []

        for key, value in metadata.items():
            # All metadata attributes (except basic) go to metadata_info
            metadata_info.append((key, value))

        # Add sections to text
        text_parts.append("Basic Information")
        text_parts.append("=" * 20)
        text_parts.append(
            f"Source File: {str(self.import_source.source_file)}"
        )
        text_parts.append(f"UID: {self.import_source.uid}")
        text_parts.append(
            f"Renderer: {self.import_source.renderer.__class__.__name__}"
        )
        if self.import_source.vector_config:
            text_parts.append("Vector Config: Configured")
        text_parts.append("")

        if metadata_info:
            text_parts.append("Metadata")
            text_parts.append("=" * 20)
            for key, value in metadata_info:
                text_parts.append(f"{key}: {self._format_value(value)}")

        # Copy to clipboard
        text = "\n".join(text_parts)
        clipboard.set(text)

        # Show a brief notification
        self._show_copy_notification()

    def _show_copy_notification(self):
        """Show a brief notification that metadata was copied."""
        # Create a simple notification by changing the window title briefly
        original_title = self.get_title()
        self.set_title(_("Metadata copied to clipboard"))

        # Restore original title after 2 seconds
        def restore_title():
            self.set_title(original_title)

        # Use GLib.timeout_add to restore the title
        import gi

        gi.require_version("GLib", "2.0")
        from gi.repository import GLib

        GLib.timeout_add(2000, restore_title)

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Handle key press events, closing the dialog on Escape."""
        if keyval == Gdk.KEY_Escape:
            self.close()
            return True
        return False
