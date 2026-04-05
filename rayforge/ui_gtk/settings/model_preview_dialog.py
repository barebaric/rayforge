"""Model preview dialog with 3D rendering for Rayforge settings."""

import logging
from pathlib import Path
from typing import Optional
from gettext import gettext as _
from gi.repository import Gio, Gtk, Adw

from ...core.model import Model
from ..icons import get_icon
from ..sim3d.canvas3d import initialized as canvas3d_initialized

logger = logging.getLogger(__name__)


class ModelPreviewDialog(Adw.MessageDialog):
    """Dialog showing a 3D model preview with metadata."""

    def __init__(
        self,
        model: Model,
        is_user: bool,
        resolved_path: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.is_user = is_user
        self._resolved_path = resolved_path
        self._setup_ui()

    def _setup_ui(self):
        self.set_heading(self.model.name)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)

        if canvas3d_initialized and self._resolved_path:
            from .model_preview_widget import ModelPreviewWidget

            preview = ModelPreviewWidget()
            preview.load_model(self._resolved_path)
            preview.set_vexpand(True)
            content_box.append(preview)

        group = Adw.PreferencesGroup()

        path_row = Adw.ActionRow(
            title=_("File"),
            subtitle=str(self.model.path),
        )
        group.add(path_row)

        if self.model.file_path and self.model.file_path.is_file():
            size_bytes = self.model.file_path.stat().st_size
            size_row = Adw.ActionRow(
                title=_("Size"),
                subtitle=_("{size:.1f} KB").format(size=size_bytes / 1024),
            )
            group.add(size_row)

        source_label = _("User") if self.is_user else _("Bundled")
        source_row = Adw.ActionRow(
            title=_("Source"),
            subtitle=source_label,
        )
        group.add(source_row)

        if self.is_user and self.model.file_path:
            open_row = Adw.ActionRow(title=_("Open in File Manager"))
            open_row.set_activatable(True)
            open_row.add_suffix(get_icon("open-in-new-symbolic"))
            open_row.connect("activated", self._on_open_in_file_manager)
            group.add(open_row)

        content_box.append(group)
        self.set_extra_child(content_box)

        self.add_response("close", _("Close"))
        if self.is_user:
            self.add_response("delete", _("Delete"))
            self.set_response_appearance(
                "delete", Adw.ResponseAppearance.DESTRUCTIVE
            )
        self.set_default_response("close")

    def _on_open_in_file_manager(self, row):
        if not self.model.file_path or not self.model.file_path.is_file():
            return
        parent = self.model.file_path.parent
        try:
            launcher = Gtk.FileLauncher(
                file=Gio.File.new_for_path(str(parent))
            )
            launcher.open_containing_folder()
        except Exception as e:
            logger.error(f"Failed to open file manager: {e}")
