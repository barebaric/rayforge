from gettext import gettext as _

from gi.repository import Adw, Gtk

from ..icons import get_icon
from ..shared.patched_dialog_window import PatchedDialogWindow
from .addon_manager_page import AddonManagerPage
from .ai_settings_page import AISettingsPage
from .color_presets_page import ColorPresetPage
from .general_preferences_page import GeneralPreferencesPage
from .license_settings_page import LicenseSettingsPage
from .machine_settings_page import MachineSettingsPage
from .material_manager_page import MaterialManagerPage
from .recipe_manager_page import RecipeManagerPage
from .registry import settings_page_registry


class SettingsWindow(PatchedDialogWindow):
    """
    The main, non-modal settings window for the application.

    Addon-contributed settings pages are added and removed live when
    addons are enabled or disabled while the window is open.
    """

    # Mapping of built-in page names to indices
    PAGE_INDICES = {
        "general": 0,
        "machines": 1,
        "materials": 2,
        "recipes": 3,
        "color_presets": 4,
        "ai": 5,
        "addons": 6,
        "licenses": 7,
    }

    # Number of built-in (non-addon) pages, kept in sync with the
    # _add_page calls in __init__.
    _BUILTIN_PAGE_COUNT = 8

    def __init__(self, initial_page: str = "general", **kwargs):
        super().__init__(skip_usage_tracking=True, **kwargs)

        self._initial_page = initial_page
        self.set_title(_("Settings"))
        self.set_default_size(800, 800)
        self.set_size_request(-1, -1)

        # Main layout container
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        # Header bar
        header_bar = Adw.HeaderBar()
        main_box.append(header_bar)

        # Navigation Split View
        split_view = Adw.NavigationSplitView(vexpand=True)
        main_box.append(split_view)

        # Sidebar
        self.sidebar_list = Gtk.ListBox(
            selection_mode=Gtk.SelectionMode.SINGLE,
            css_classes=["navigation-sidebar"],
        )
        sidebar_page = Adw.NavigationPage.new(
            self.sidebar_list, _("Categories")
        )
        split_view.set_sidebar(sidebar_page)

        # Content
        self.content_stack = Gtk.Stack()

        # Tracks addon page classes currently added to the stack so
        # we can diff against the registry on live updates.
        self._addon_page_classes: list = []

        # Populate sidebar and content
        self._add_page(GeneralPreferencesPage)
        self._add_page(MachineSettingsPage)
        self._add_page(MaterialManagerPage)
        self._add_page(RecipeManagerPage)
        self._add_page(ColorPresetPage)
        self._add_page(AISettingsPage)
        self._add_page(AddonManagerPage)
        self._add_page(LicenseSettingsPage)

        # Addon-contributed pages (registered via the
        # register_settings_pages hook).
        for page_class in settings_page_registry.get_pages():
            self._add_addon_page(page_class)

        # Create the content's NavigationPage wrapper
        pages = self.content_stack.get_pages()
        first_stack_page = pages.get_item(0)  # type: ignore
        initial_title = first_stack_page.get_title()
        self.content_page = Adw.NavigationPage.new(
            self.content_stack, initial_title
        )
        split_view.set_content(self.content_page)

        # Populate
        self.sidebar_list.connect("row-selected", self._on_row_selected)
        # Select the initial page
        initial_index = self.PAGE_INDICES.get(self._initial_page, 0)
        initial_row = self.sidebar_list.get_row_at_index(initial_index)
        self.sidebar_list.select_row(initial_row)

        # Live-update when addon pages are registered/unregistered.
        self._registry_handler = settings_page_registry.changed.connect(
            self._on_registry_changed
        )

    def _add_page(self, page_class):
        page = page_class()
        page_name = page.get_title()
        self.content_stack.add_titled(page, page_name, page_name)

        row = Gtk.ListBoxRow()
        box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            margin_start=12,
            margin_end=12,
            margin_top=6,
            margin_bottom=6,
        )
        icon = get_icon(page.get_icon_name())
        label = Gtk.Label(label=page_name, xalign=0)
        box.append(icon)
        box.append(label)
        row.set_child(box)
        self.sidebar_list.append(row)

    def _add_addon_page(self, page_class):
        """Add an addon-contributed page and track it for live removal."""
        self._add_page(page_class)
        self._addon_page_classes.append(page_class)

    def _remove_addon_page(self, page_class):
        """Remove an addon-contributed page by its class identity."""
        idx = None
        for i, cls in enumerate(self._addon_page_classes):
            if cls is page_class:
                idx = i
                break
        if idx is None:
            return

        stack_index = self._BUILTIN_PAGE_COUNT + idx
        pages = self.content_stack.get_pages()
        stack_page = pages.get_item(stack_index)  # type: ignore
        child = stack_page.get_child()

        # Preserve current selection if it's not the page being removed.
        selected_row = self.sidebar_list.get_selected_row()
        selected_index = selected_row.get_index() if selected_row else None

        self.content_stack.remove(child)
        row = self.sidebar_list.get_row_at_index(stack_index)
        if row is not None:
            self.sidebar_list.remove(row)
        del self._addon_page_classes[idx]

        # Restore selection, falling back to the first page.
        if selected_index is not None and selected_index != stack_index:
            new_row = self.sidebar_list.get_row_at_index(selected_index)
            if new_row is not None:
                self.sidebar_list.select_row(new_row)
        else:
            first_row = self.sidebar_list.get_row_at_index(0)
            if first_row is not None:
                self.sidebar_list.select_row(first_row)

    def _on_registry_changed(self, registry):
        """Reconcile addon pages with the current registry contents."""
        current = registry.get_pages()
        current_set = {id(cls) for cls in current}

        # Remove pages that are no longer registered.
        for cls in list(self._addon_page_classes):
            if id(cls) not in current_set:
                self._remove_addon_page(cls)

        # Add pages that are newly registered, preserving registry order.
        existing_ids = {id(cls) for cls in self._addon_page_classes}
        for cls in current:
            if id(cls) not in existing_ids:
                self._add_addon_page(cls)

    def _on_row_selected(self, listbox, row):
        if row:
            index = row.get_index()
            pages = self.content_stack.get_pages()
            stack_page = pages.get_item(index)  # type: ignore
            widget_to_show = stack_page.get_child()
            self.content_stack.set_visible_child(widget_to_show)
            page_title = stack_page.get_title()
            self.content_page.set_title(page_title)

    def do_close_request(self, *args) -> bool:
        settings_page_registry.changed.disconnect(self._registry_handler)
        return super().do_close_request(*args)
