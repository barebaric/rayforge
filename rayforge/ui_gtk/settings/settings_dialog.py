from gi.repository import Adw, Gdk, Gtk
from .general_preferences_page import GeneralPreferencesPage
from ..machine.settings_page import MachineSettingsPage
from ..doceditor.material_manager import MaterialManager
from ..doceditor.recipe_manager import RecipeManager
from ..icons import get_icon


class SettingsWindow(Adw.Window):
    """
    The main, non-modal settings window for the application.
    It is built using the modern Adwaita composition pattern and contains pages
    for general application settings and machine management.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_title(_("Settings"))
        self.set_default_size(800, 800)
        self.set_size_request(-1, -1)

        # Main layout container
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        # Header bar - Adw.NavigationSplitView will manage its title widget
        header_bar = Adw.HeaderBar()
        main_box.append(header_bar)

        # Navigation Split View for sidebar and content
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

        # Populate both the sidebar and the content stack
        self._add_page(GeneralPreferencesPage)
        self._add_page(MachineSettingsPage)
        self._add_page(MaterialManager)
        self._add_page(RecipeManager)

        # Create the content's NavigationPage wrapper using the first page's
        # title
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

        # Add a key controller to close the window on Escape press
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

    def _add_page(self, page_class):
        """
        Instantiates a page and adds it to the stack and sidebar.
        """
        # Note: These page classes don't require arguments
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
        """
        Handler for when a row is selected in the sidebar.
        Switches the visible page in the stack and updates the header bar
        title.
        """
        if row:
            index = row.get_index()
            # Get the Gtk.StackPage wrapper object
            pages = self.content_stack.get_pages()
            stack_page = pages.get_item(index)  # type: ignore
            # Get the actual child widget from the wrapper
            widget_to_show = stack_page.get_child()

            # Set the visible child using the actual widget
            self.content_stack.set_visible_child(widget_to_show)

            # Update the title using the title from the wrapper
            page_title = stack_page.get_title()
            self.content_page.set_title(page_title)

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """
        Handles key press events for the window. Closes the window when the
        Escape key is pressed.
        """
        if keyval == Gdk.KEY_Escape:
            self.close()
            return True  # Event handled, do not propagate further
        return False  # Event not handled
