from gi.repository import Adw, Gdk, Gtk

from ..icons import get_icon
from ..shared.patched_dialog_window import PatchedDialogWindow
from .general_preferences_page import GeneralPreferencesPage
from .machine_settings_page import MachineSettingsPage
from .material_manager_page import MaterialManagerPage
from .package_manager_page import PackageManagerPage
from .recipe_manager_page import RecipeManagerPage


class SettingsWindow(PatchedDialogWindow):
    """
    The main, non-modal settings window for the application.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        # Populate sidebar and content
        self._add_page(GeneralPreferencesPage)
        self._add_page(MachineSettingsPage)
        self._add_page(MaterialManagerPage)
        self._add_page(RecipeManagerPage)
        self._add_page(PackageManagerPage)

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
        self.sidebar_list.select_row(self.sidebar_list.get_row_at_index(0))

        # Key controller
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

    def _add_page(self, page_class):
        # ... (Same as existing code)
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

    def _on_row_selected(self, listbox, row):
        # ... (Same as existing code)
        if row:
            index = row.get_index()
            pages = self.content_stack.get_pages()
            stack_page = pages.get_item(index)  # type: ignore
            widget_to_show = stack_page.get_child()
            self.content_stack.set_visible_child(widget_to_show)
            page_title = stack_page.get_title()
            self.content_page.set_title(page_title)

    def _on_key_pressed(self, controller, keyval, keycode, state):
        # ... (Same as existing code)
        if keyval == Gdk.KEY_Escape:
            self.close()
            return True
        return False
