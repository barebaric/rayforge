"""Reusable model selection dialog for picking 3D models."""

import logging
from gettext import gettext as _
from pathlib import Path
from typing import Optional, List

from gi.repository import Gtk, Adw

from ...context import get_context
from ...core.model import Model, ModelCategory
from ..canvas3d import initialized as canvas3d_initialized
from ..icons import get_icon

logger = logging.getLogger(__name__)


class ModelSelectionDialog(Adw.MessageDialog):
    """A picker dialog that lists models from ModelManager libraries.

    Shows a 3D preview via ModelPreviewWidget when a model is selected.
    Returns the selected model_id string (or None on cancel).
    """

    def __init__(
        self,
        category: Optional[ModelCategory] = None,
        current_model_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._category = category
        self._current_model_id = current_model_id
        self._selected_model_id: Optional[str] = None
        self._row_model_paths: dict[int, str] = {}
        self._setup_ui()

    def _setup_ui(self):
        self.set_heading(_("Select Model"))

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)

        model_mgr = get_context().model_mgr
        category = self._category
        if category is None:
            categories = model_mgr.get_categories()
            category = categories[0] if categories else None

        models: List[Model] = []
        if category:
            models = model_mgr.get_all_models(category)

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
            self._row_model_paths[id(row)] = str(model.path)
            suffix = model.path.suffix.upper().lstrip(".")
            row.set_subtitle(suffix)
            row.add_prefix(get_icon("image-x-generic-symbolic"))
            self._list_box.append(row)

        if self._current_model_id:
            i = 0
            while lb_row := self._list_box.get_row_at_index(i):
                if (
                    id(lb_row) in self._row_model_paths
                    and self._row_model_paths[id(lb_row)]
                    == self._current_model_id
                ):
                    self._list_box.select_row(lb_row)
                    break
                i += 1
        else:
            self._list_box.select_row(none_row)

        scrolled.set_child(self._list_box)
        content_box.append(scrolled)

        self._preview_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=6
        )
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
            self._selected_model_id = None
            return

        if row is None or id(row) not in self._row_model_paths:
            self._selected_model_id = None
            return

        self._selected_model_id = self._row_model_paths[id(row)]

        if not canvas3d_initialized:
            return

        model_mgr = get_context().model_mgr
        model = Model(name="", path=Path(self._selected_model_id))
        resolved = model_mgr.resolve(model)
        if resolved is None:
            return

        from ..settings.model_preview_widget import ModelPreviewWidget

        preview = ModelPreviewWidget()
        preview.load_model(resolved)
        preview.set_size_request(320, 200)
        self._preview_box.append(preview)

    def get_selected_model_id(self) -> Optional[str]:
        return self._selected_model_id
