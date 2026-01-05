from typing import List, Optional
from gi.repository import Gtk, Adw, Gdk, GLib
from ...machine.models.macro import Macro
from ...pipeline.encoder.context import GcodeContext
from ..icons import get_icon

# Define characters that are not allowed in macro names
FORBIDDEN_NAME_CHARS = "();[]{}<>"


class GcodeEditorDialog(Adw.Window):
    """A generic modal dialog for editing a G-code macro."""

    def __init__(
        self,
        parent: Gtk.Window,
        macro: Macro,
        *,
        allow_name_edit: bool = False,
        existing_macros: Optional[List[Macro]] = None,
        variable_context_level: str = "job",
    ):
        """
        Initializes the macro editor dialog.

        Args:
            parent: The parent window.
            macro: The macro to be edited.
            allow_name_edit: If True, shows an entry row to edit the macro
              name.
            existing_macros: A list of other macros to check for name
              uniqueness.
            variable_context_level: The context level for variable
              documentation.
        """
        super().__init__(modal=True, transient_for=parent)
        self.macro = macro
        self.saved = False
        self._allow_name_edit = allow_name_edit
        self.existing_macros = existing_macros or []
        self.variable_context_level = variable_context_level
        self.set_title(_("Edit Macro"))
        self.set_size_request(750, 700)

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        header = Adw.HeaderBar()
        main_box.append(header)

        cancel_button = Gtk.Button(label=_("Cancel"))
        cancel_button.connect("clicked", lambda w: self.close())
        header.pack_start(cancel_button)

        # Button for Variables
        variables_button = Gtk.MenuButton(
            child=get_icon("variable-symbolic"),
            tooltip_text=_("Insert Variable"),
        )
        header.pack_start(variables_button)
        self._build_variables_popover(variables_button)

        # Button for Macros (now always shown)
        macros_button = Gtk.MenuButton(
            child=get_icon("code-symbolic"),
            tooltip_text=_("Include Macro"),
        )
        header.pack_start(macros_button)
        self._build_macros_popover(macros_button)

        self.save_button = Gtk.Button(label=_("Save"))
        self.save_button.add_css_class("suggested-action")
        self.save_button.connect("clicked", self._on_save_clicked)
        header.pack_end(self.save_button)

        self.name_row = Adw.EntryRow(title=_("Name"))
        self.name_row.set_text(self.macro.name)
        self.name_row.set_margin_top(6)

        self.error_label = Gtk.Label(halign=Gtk.Align.START, margin_start=12)
        self.error_label.add_css_class("error")

        if self._allow_name_edit:
            main_box.append(self.name_row)
            main_box.append(self.error_label)
            self.name_row.connect("notify::text", self._validate_name)
        else:
            self.set_title(
                _("Edit Macro for {name}").format(name=self.macro.name)
            )

        scrolled_window = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vexpand=True,
            margin_top=6,
            margin_bottom=6,
            margin_start=6,
            margin_end=6,
        )
        main_box.append(scrolled_window)

        self.text_view = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR,
            pixels_above_lines=2,
            pixels_below_lines=2,
            left_margin=6,
            right_margin=6,
        )
        self.text_view.add_css_class("monospace")
        buffer = self.text_view.get_buffer()
        buffer.set_text("\n".join(self.macro.code), -1)
        scrolled_window.set_child(self.text_view)

        # Add a key controller to listen for the Escape key
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

        # Run initial validation
        self._validate_name()

    def _on_popover_closed(self, popover: Gtk.Popover):
        """Ensure the text view regains focus when a popover is closed."""
        self.text_view.grab_focus()

    def _build_variables_popover(self, parent_button: Gtk.MenuButton):
        """Creates and populates the popover with variable documentation."""
        self.variables_popover = Gtk.Popover()
        parent_button.set_popover(self.variables_popover)
        self.variables_popover.connect("closed", self._on_popover_closed)

        clamp = Adw.Clamp(maximum_size=350)
        self.variables_popover.set_child(clamp)

        popover_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=6,
            margin_top=6,
            margin_bottom=6,
            margin_start=6,
            margin_end=6,
        )
        clamp.set_child(popover_box)

        scrolled_window = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER, min_content_height=250
        )
        popover_box.append(scrolled_window)

        list_box = Gtk.ListBox()
        list_box.add_css_class("boxed-list")
        scrolled_window.set_child(list_box)

        # Variables section
        var_title = Gtk.Label(xalign=0, margin_bottom=6, margin_top=6)
        var_title.add_css_class("title-4")
        var_title.set_text(_("Available Variables"))
        var_header_row = Gtk.ListBoxRow(child=var_title, selectable=False)
        list_box.append(var_header_row)

        variables = GcodeContext.get_docs(self.variable_context_level)
        for var, desc in variables:
            row = Adw.ActionRow(subtitle=desc, activatable=True)
            escaped_var = GLib.markup_escape_text(f"{{{var}}}")
            row.set_title(
                f'<span font_family="monospace">{escaped_var}</span>'
            )
            row.set_use_markup(True)
            row.connect("activated", self._on_variable_activated, var)
            list_box.append(row)

    def _build_macros_popover(self, parent_button: Gtk.MenuButton):
        """Creates and populates the popover for including other macros."""
        self.macros_popover = Gtk.Popover()
        parent_button.set_popover(self.macros_popover)
        self.macros_popover.connect("closed", self._on_popover_closed)

        clamp = Adw.Clamp(maximum_size=350)
        self.macros_popover.set_child(clamp)

        popover_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=6,
            margin_top=6,
            margin_bottom=6,
            margin_start=6,
            margin_end=6,
        )
        clamp.set_child(popover_box)

        scrolled_window = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER, min_content_height=150
        )
        popover_box.append(scrolled_window)

        list_box = Gtk.ListBox()
        list_box.add_css_class("boxed-list")
        scrolled_window.set_child(list_box)

        macros_to_include = [
            m for m in self.existing_macros if m.uid != self.macro.uid
        ]

        if macros_to_include:
            for macro in sorted(macros_to_include, key=lambda s: s.name):
                row = Adw.ActionRow(title=macro.name, activatable=True)
                row.connect("activated", self._on_macro_activated, macro.name)
                list_box.append(row)
        else:
            placeholder = Gtk.Label(label=_("No other macros to include."))
            placeholder.add_css_class("dim-label")
            placeholder.set_margin_top(12)
            placeholder.set_margin_bottom(12)
            list_box.append(
                Gtk.ListBoxRow(child=placeholder, selectable=False)
            )

    def _insert_text_at_cursor(self, text: str):
        """Helper to insert text at the current cursor position."""
        buffer = self.text_view.get_buffer()
        insert_mark = buffer.get_insert()
        iterator = buffer.get_iter_at_mark(insert_mark)
        buffer.insert(iterator, text, -1)

    def _on_variable_activated(self, row: Adw.ActionRow, variable_name: str):
        """Called when a variable row is clicked."""
        self._insert_text_at_cursor(f"{{{variable_name}}}")
        self.variables_popover.popdown()

    def _on_macro_activated(self, row: Adw.ActionRow, macro_name: str):
        """Called when a macro include row is clicked."""
        self._insert_text_at_cursor(f"@include({macro_name})")
        self.macros_popover.popdown()

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Handler for key press events on the window."""
        if keyval == Gdk.KEY_Escape:
            self.close()
            return True  # Event handled, stop propagation
        return False

    def _validate_name(self, *args):
        """Checks the validity of the macro name and updates UI feedback."""
        if not self._allow_name_edit:
            self.save_button.set_sensitive(True)
            return

        name = self.name_row.get_text()
        error_message = ""

        if not name.strip():
            error_message = _("Name cannot be empty.")
        elif any(char in name for char in FORBIDDEN_NAME_CHARS):
            error_message = _(
                "Name contains invalid characters: {chars}"
            ).format(chars=FORBIDDEN_NAME_CHARS)
        else:
            for other_macro in self.existing_macros:
                # Check for name collision, ignoring the macro we are editing
                if (
                    other_macro.name == name
                    and other_macro.uid != self.macro.uid
                ):
                    error_message = _(
                        "This name is already used by another macro."
                    )
                    break

        if error_message:
            self.error_label.set_label(error_message)
            self.error_label.set_visible(True)
            self.save_button.set_sensitive(False)
        else:
            self.error_label.set_visible(False)
            self.save_button.set_sensitive(True)

    def _on_save_clicked(self, button: Gtk.Button):
        """Stores the UI content into the macro object and closes."""
        buffer = self.text_view.get_buffer()
        start, end = buffer.get_start_iter(), buffer.get_end_iter()
        text = buffer.get_text(start, end, include_hidden_chars=True)

        if self._allow_name_edit:
            self.macro.name = self.name_row.get_text()
        self.macro.code = text.splitlines()

        self.saved = True
        self.close()
