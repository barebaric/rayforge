"""Model manager settings page for Rayforge."""

import logging
from pathlib import Path
from typing import Optional, cast, List
from gettext import gettext as _

from gi.repository import Gio, Gtk, Adw
from blinker import Signal

from ...context import get_context
from ...core.model import Model, ModelCategory, ModelLibrary
from ..icons import get_icon
from ..shared.gtk import apply_css
from ..shared.preferences_group import PreferencesGroupWithButton
from ..shared.preferences_page import TrackedPreferencesPage
from .model_preview_dialog import ModelPreviewDialog

logger = logging.getLogger(__name__)

_STANDALONE_LIST_CSS = """
.group-with-button-container > .standalone-list {
    border-bottom-left-radius: 12px;
    border-bottom-right-radius: 12px;
}
"""


class _LibraryRow(Adw.ActionRow):
    """A row representing a model library with a chevron indicator."""

    def __init__(self, library: ModelLibrary):
        super().__init__(
            title=library.display_name,
            activatable=True,
        )
        self.library = library
        try:
            self.add_prefix(get_icon(library.icon_name))
        except Exception:
            pass
        self.add_suffix(get_icon("go-next-symbolic"))


class _CategoryRow(Gtk.Box):
    """A row representing a model category."""

    def __init__(self, category: ModelCategory, count: int):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.category = category
        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        labels_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=0, hexpand=True
        )
        self.append(labels_box)

        title_label = Gtk.Label(
            label=category.display_name,
            halign=Gtk.Align.START,
            xalign=0,
        )
        labels_box.append(title_label)

        if count == 1:
            count_text = _("1 model")
        else:
            count_text = _("{count} models").format(count=count)
        subtitle_label = Gtk.Label(
            label=count_text,
            halign=Gtk.Align.START,
            xalign=0,
            css_classes=["dim-label"],
        )
        labels_box.append(subtitle_label)


class _CategoryListWidget(PreferencesGroupWithButton):
    """Selectable list of model categories for a given library."""

    def __init__(self, **kwargs):
        super().__init__(
            button_label="",
            selection_mode=Gtk.SelectionMode.SINGLE,
            empty_placeholder=_("No categories available."),
            **kwargs,
        )
        apply_css(_STANDALONE_LIST_CSS)
        self.list_box.add_css_class("standalone-list")
        self.category_selected = Signal()
        self._row_widgets: List[_CategoryRow] = []
        self._current_library: Optional[ModelLibrary] = None
        self.list_box.connect("row-selected", self._on_row_selected)

    def _create_add_button(self, button_label):
        box = Gtk.Box()
        box.set_visible(False)
        return box

    def _on_add_clicked(self, button: Gtk.Button):
        pass

    def set_library(self, library: Optional[ModelLibrary]):
        self._current_library = library
        self.populate_and_select()

    def populate_and_select(
        self, select_category: Optional[ModelCategory] = None
    ):
        if self._current_library is None:
            self._row_widgets.clear()
            self.set_items([])
            return

        model_mgr = get_context().model_mgr
        self._row_widgets.clear()
        categories = model_mgr.get_categories()
        rows = []
        for cat in categories:
            count = len(model_mgr.get_models(self._current_library, cat))
            row = _CategoryRow(cat, count)
            rows.append((cat, row))

        self.set_items([r for _, r in rows])

        row_to_select = None
        if select_category is not None:
            i = 0
            while row := self.list_box.get_row_at_index(i):
                child = row.get_child()
                if (
                    isinstance(child, _CategoryRow)
                    and child.category.id == select_category.id
                ):
                    row_to_select = row
                    break
                i += 1
        elif rows:
            row_to_select = self.list_box.get_row_at_index(0)

        if row_to_select:
            self.list_box.select_row(row_to_select)

    def create_row_widget(self, item: _CategoryRow) -> Gtk.Widget:
        self._row_widgets.append(item)
        return item

    def _on_row_selected(self, listbox, row):
        category = None
        selected_row = listbox.get_selected_row()
        if selected_row:
            child = selected_row.get_child()
            if isinstance(child, _CategoryRow):
                category = child.category
        self.category_selected.send(self, category=category)

    def refresh_counts(self):
        self.populate_and_select()


class _ModelRow(Gtk.Box):
    """A row representing a single 3D model."""

    def __init__(
        self,
        model: Model,
        is_user: bool,
        on_delete_callback,
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.model = model
        self.is_user = is_user
        self.on_delete_callback = on_delete_callback
        self._setup_ui()

    def _setup_ui(self):
        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        self.append(get_icon("image-x-generic-symbolic"))

        labels_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=0, hexpand=True
        )
        self.append(labels_box)

        title_label = Gtk.Label(
            label=self.model.name,
            halign=Gtk.Align.START,
            xalign=0,
        )
        labels_box.append(title_label)

        suffix = self.model.path.suffix.upper().lstrip(".")
        if not suffix:
            suffix = str(self.model.path)
        subtitle_parts = [suffix]
        if not self.is_user:
            subtitle_parts.append(_("Bundled"))
        subtitle_label = Gtk.Label(
            label=" \u00b7 ".join(subtitle_parts),
            halign=Gtk.Align.START,
            xalign=0,
            css_classes=["dim-label"],
        )
        labels_box.append(subtitle_label)

        if self.is_user:
            suffix_box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
            self.append(suffix_box)
            delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
            delete_button.add_css_class("flat")
            delete_button.connect("clicked", self._on_delete_clicked)
            suffix_box.append(delete_button)

    def _on_delete_clicked(self, button):
        self.on_delete_callback(self.model)


class _ModelListWidget(PreferencesGroupWithButton):
    """List of models for the selected category/library."""

    def __init__(self, **kwargs):
        super().__init__(
            button_label=_("Add Model"),
            empty_placeholder=_("No models in this category."),
            **kwargs,
        )
        self._current_library: Optional[ModelLibrary] = None
        self._current_category: Optional[ModelCategory] = None
        self._row_widgets: List[_ModelRow] = []
        self.list_box.set_show_separators(True)
        self.list_box.connect("row-activated", self._on_row_activated)

    def set_library(self, library: Optional[ModelLibrary]):
        self._current_library = library
        self.add_button.set_sensitive(
            library is not None and not library.read_only
        )

    def set_category(self, category: Optional[ModelCategory]):
        self._current_category = category
        self._populate()

    def _populate(self):
        if self._current_library is None or self._current_category is None:
            self.set_items([])
            return
        model_mgr = get_context().model_mgr
        models = model_mgr.get_models(
            self._current_library, self._current_category
        )
        self._row_widgets.clear()
        self.set_items(models)

    def refresh(self):
        self._populate()

    def create_row_widget(self, item: Model) -> Gtk.Widget:
        model_mgr = get_context().model_mgr
        is_user = model_mgr.is_user_model(item)
        row = _ModelRow(item, is_user, self._on_delete_model)
        self._row_widgets.append(row)
        return row

    def _on_row_activated(self, listbox, row):
        child = row.get_child()
        if not isinstance(child, _ModelRow):
            return
        model = child.model
        root = self.get_root()
        model_mgr = get_context().model_mgr
        resolved = model_mgr.resolve(model)
        dialog = ModelPreviewDialog(
            model=model,
            is_user=child.is_user,
            resolved_path=resolved,
            transient_for=cast(Gtk.Window, root) if root else None,
        )

        def on_response(d, response_id):
            if response_id == "delete":
                if model_mgr.remove_model(model):
                    self._populate()
            d.destroy()

        dialog.connect("response", on_response)
        dialog.present()

    def _on_delete_model(self, model: Model):
        root = self.get_root()
        dialog = Adw.MessageDialog(
            transient_for=cast(Gtk.Window, root) if root else None,
            heading=_("Delete '{name}'?").format(name=model.name),
            body=_(
                "The model file will be permanently removed. "
                "This action cannot be undone."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("delete", _("Delete"))
        dialog.set_response_appearance(
            "delete", Adw.ResponseAppearance.DESTRUCTIVE
        )
        dialog.set_default_response("cancel")

        def on_response(d, response_id):
            if response_id == "delete":
                model_mgr = get_context().model_mgr
                if model_mgr.remove_model(model):
                    self._populate()
            d.destroy()

        dialog.connect("response", on_response)
        dialog.present()

    def _on_add_clicked(self, button: Gtk.Button):
        if (
            self._current_library is None
            or self._current_category is None
            or self._current_library.read_only
        ):
            return

        root = self.get_root()
        file_dialog = Gtk.FileDialog()
        file_filter = Gtk.FileFilter()
        file_filter.set_name(_("3D Models"))
        file_filter.add_suffix("glb")
        file_filter.add_suffix("gltf")
        file_filter.add_mime_type("model/gltf-binary")
        file_filter.add_mime_type("model/gltf+json")
        filter_list = Gio.ListStore.new(Gtk.FileFilter)
        filter_list.append(file_filter)
        file_dialog.set_filters(filter_list)
        category = self._current_category

        def on_open(d, result):
            try:
                file = d.open_finish(result)
            except Exception:
                return
            if file is None:
                return
            src = Path(file.get_path())
            dest_relative = Path(category.id) / src.name
            model = Model(name=src.stem, path=dest_relative)
            model_mgr = get_context().model_mgr
            try:
                model_mgr.add_model(src, model)
                self._populate()
            except Exception as e:
                logger.error(f"Failed to add model: {e}")

        file_dialog.open(cast(Gtk.Window, root), None, on_open)


class ModelManagerPage(TrackedPreferencesPage):
    """
    Settings page for managing 3D model assets.

    Page 1 (default): Library list with chevrons.
    Page 2: Category selector at top, model list below.
    """

    key = "models"

    def __init__(self):
        super().__init__(
            title=_("Models"),
            icon_name="image-x-generic-symbolic",
        )

        self._current_library: Optional[ModelLibrary] = None

        self._stack = Gtk.Stack()
        self._stack.set_transition_type(
            Gtk.StackTransitionType.SLIDE_LEFT_RIGHT
        )
        self._stack.add_named(self._build_library_list_page(), "libraries")
        self._stack.add_named(self._build_library_detail_page(), "detail")
        wrapper = Adw.PreferencesGroup()
        wrapper.add(self._stack)
        self.add(wrapper)

        get_context().model_mgr.changed.connect(self._on_models_changed)

        self._populate_libraries()

    def _build_library_list_page(self) -> Adw.PreferencesGroup:
        group = Adw.PreferencesGroup(
            title=_("Model Libraries"),
            description=_("Select a library to browse its 3D models."),
        )
        self._library_list = Gtk.ListBox(
            selection_mode=Gtk.SelectionMode.NONE,
            show_separators=True,
            css_classes=["boxed-list"],
        )
        group.add(self._library_list)
        return group

    def _build_library_detail_page(self) -> Adw.PreferencesGroup:
        group = Adw.PreferencesGroup()
        group.set_margin_top(12)

        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        back_button = Gtk.Button(child=get_icon("go-previous-symbolic"))
        back_button.add_css_class("flat")
        back_button.connect(
            "clicked",
            lambda w: self._stack.set_visible_child_name("libraries"),
        )
        header_box.append(back_button)
        self._detail_title = Gtk.Label(
            label="",
            css_classes=["title-4"],
            halign=Gtk.Align.START,
        )
        header_box.append(self._detail_title)
        group.add(header_box)

        self._category_list = _CategoryListWidget(
            title=_("Categories"),
            description=_("Select a category to view its models."),
        )
        self._category_list.set_margin_top(12)
        group.add(self._category_list)

        self._model_list = _ModelListWidget(
            title=_("Models"),
            description=_("3D models in the selected category."),
        )
        self._model_list.set_margin_top(12)
        group.add(self._model_list)

        self._category_list.category_selected.connect(
            self._on_category_selected
        )

        return group

    def _populate_libraries(self):
        while child := self._library_list.get_row_at_index(0):
            self._library_list.remove(child)

        model_mgr = get_context().model_mgr
        for library in model_mgr.get_libraries():
            row = _LibraryRow(library)
            row.connect("activated", self._on_library_activated)
            self._library_list.append(row)

    def _on_library_activated(self, row):
        if not isinstance(row, _LibraryRow):
            return
        self._current_library = row.library
        self._detail_title.set_label(row.library.display_name)
        self._category_list.set_library(row.library)
        self._model_list.set_library(row.library)
        categories = get_context().model_mgr.get_categories()
        if categories:
            self._model_list.set_category(categories[0])
        self._stack.set_visible_child_name("detail")

    def _on_category_selected(self, sender, category=None):
        self._model_list.set_category(category)

    def _on_models_changed(self, sender):
        if self._stack.get_visible_child_name() == "detail":
            self._model_list.refresh()
            self._category_list.refresh_counts()
        self._populate_libraries()
