"""Reusable model selection dialog for picking 3D models."""

import logging
from gettext import gettext as _
from pathlib import Path
from typing import Optional, List, Dict

from gi.repository import Gtk, Adw

from ...context import get_context
from ...core.model import Model
from ..icons import get_icon
from ..sim3d.canvas3d import initialized as canvas3d_initialized

logger = logging.getLogger(__name__)


class ModelSelectionDialog(Adw.MessageDialog):
    """A picker dialog that lists models from ModelManager libraries.

    Shows a 3D preview via ModelPreviewWidget when a model is selected.
    Returns the selected model_path string (or None on cancel).
    """

    def __init__(
        self,
        current_model_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._current_model_path = current_model_path
        self._selected_model_path: Optional[str] = None
        self._row_paths: Dict[Adw.ActionRow, str] = {}
        self._setup_ui()

    def _setup_ui(self):
        self.set_heading(_("Select Model"))

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)

        self._preview_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=6
        )
        self._preview_box.set_size_request(512, 288)

        model_mgr = get_context().model_mgr
        models: List[Model] = model_mgr.get_all_models()

        scrolled = Gtk.ScrolledWindow(
            min_content_height=200,
            max_content_height=300,
        )
        self._list_box = Gtk.ListBox(
            selection_mode=Gtk.SelectionMode.SINGLE,
            show_separators=True,
            css_classes=["boxed-list"],
        )
        self._list_box.connect("row-selected", self._on_row_selected)

        none_row = Adw.ActionRow(title=_("None"), activatable=True)
        none_row.add_prefix(get_icon("edit-clear-symbolic"))
        self._list_box.append(none_row)

        for model in models:
            row = Adw.ActionRow(title=model.name, activatable=True)
            self._row_paths[row] = str(model.path)
            suffix = model.path.suffix.upper().lstrip(".")
            row.set_subtitle(suffix)
            row.add_prefix(get_icon("image-x-generic-symbolic"))
            self._list_box.append(row)

        if self._current_model_path:
            for row, path in self._row_paths.items():
                if path == self._current_model_path:
                    self._list_box.select_row(row)
                    break
        else:
            self._list_box.select_row(none_row)

        scrolled.set_child(self._list_box)
        content_box.append(scrolled)
        content_box.append(self._preview_box)

        self.set_extra_child(content_box)

        self.add_response("cancel", _("Cancel"))
        self.add_response("select", _("Select"))
        self.set_default_response("select")
        self.set_response_appearance(
            "select", Adw.ResponseAppearance.SUGGESTED
        )

    def _on_row_selected(self, listbox, row):
        while child := self._preview_box.get_first_child():
            self._preview_box.remove(child)

        if row is None:
            self._selected_model_path = None
            return

        model_path = self._row_paths.get(row)
        if model_path is None:
            self._selected_model_path = None
            return

        self._selected_model_path = model_path

        if not canvas3d_initialized:
            return

        model_mgr = get_context().model_mgr
        model = Model(name="", path=Path(model_path))
        resolved = model_mgr.resolve(model)
        if resolved is None:
            return

        from ..settings.model_preview_widget import ModelPreviewWidget

        preview = ModelPreviewWidget()
        preview.load_model(resolved)
        preview.set_vexpand(True)
        preview.set_hexpand(True)
        self._preview_box.append(preview)

    def get_selected_model_path(self) -> Optional[str]:
        return self._selected_model_path
