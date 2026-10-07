from gettext import gettext as _

from gi.repository import Adw, Gdk, Gtk

from ...machine.models.macro import Macro
from ..icons import get_icon
from ..shared.code_editor import CodeEditor
from ..shared.gtk import apply_css
from ..shared.patched_dialog_window import PatchedDialogWindow
from .validation import find_number_only_line, format_number_only_warning

# Define characters that are not allowed in macro names
FORBIDDEN_NAME_CHARS = "();[]{}<>"
apply_css(
    """
.macro-editor-card {
    border: 1px solid @borders;
}
"""
)


class GcodeEditorDialog(PatchedDialogWindow):
    """A generic modal dialog for editing a G-code macro."""

    def __init__(
        self,
        parent: Gtk.Window,
        macro: Macro,
        *,
        allow_name_edit: bool = False,
        existing_macros: list[Macro] | None = None,
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
        self.set_title(_("Edit Macro"))
        self.set_size_request(750, 700)

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        header = Adw.HeaderBar()
        main_box.append(header)

        self.warning_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            visible=False,
        )
        self.warning_box.set_margin_top(12)
        self.warning_box.set_margin_bottom(6)
        self.warning_box.set_margin_start(16)
        self.warning_box.set_margin_end(16)

        self.warning_icon = get_icon("warning-symbolic")
        self.warning_icon.add_css_class("warning")
        self.warning_icon.set_valign(Gtk.Align.CENTER)
        self.warning_box.append(self.warning_icon)

        self.warning_label = Gtk.Label(xalign=0, wrap=True, hexpand=True)
        self.warning_label.add_css_class("warning-label")
        self.warning_box.append(self.warning_label)

        main_box.append(self.warning_box)

        cancel_button = Gtk.Button(label=_("Cancel"))
        cancel_button.connect("clicked", lambda w: self.close())
        header.pack_start(cancel_button)

        self.save_button = Gtk.Button(label=_("Save"))
        self.save_button.add_css_class("suggested-action")
        self.save_button.connect("clicked", self._on_save_clicked)
        header.pack_end(self.save_button)

        self.name_row = Adw.EntryRow(title=_("Name"))
        self.name_row.set_text(self.macro.name)

        self.error_label = Gtk.Label(halign=Gtk.Align.START, margin_start=12)
        self.error_label.add_css_class("error")

        if self._allow_name_edit:
            # AdwEntryRow must live in a GtkListBox.
            name_list = Gtk.ListBox()
            name_list.set_selection_mode(Gtk.SelectionMode.NONE)
            name_list.add_css_class("boxed-list")
            name_list.set_margin_top(6)
            name_list.set_margin_start(6)
            name_list.set_margin_end(6)
            name_list.append(self.name_row)
            main_box.append(name_list)
            main_box.append(self.error_label)
            self.name_row.connect("notify::text", self._validate_name)
        else:
            self.set_title(
                _("Edit Macro for {name}").format(name=self.macro.name)
            )

        self.code_editor = CodeEditor(
            text="\n".join(self.macro.code),
            variable_context_level=variable_context_level,
            macros=self.existing_macros,
            exclude_macro_uid=self.macro.uid,
            min_content_height=400,
        )
        self.text_view = self.code_editor.text_view
        self.code_editor.set_vexpand(True)
        self.code_editor.set_hexpand(True)
        self.code_editor.set_margin_top(2)
        self.code_editor.set_margin_bottom(2)
        self.code_editor.set_margin_start(2)
        self.code_editor.set_margin_end(2)
        editor_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        editor_card.add_css_class("card")
        editor_card.add_css_class("macro-editor-card")
        editor_card.append(self.code_editor)
        editor_card.set_vexpand(True)
        editor_card.set_hexpand(True)
        editor_card.set_margin_top(6)
        editor_card.set_margin_bottom(6)
        editor_card.set_margin_start(6)
        editor_card.set_margin_end(6)
        main_box.append(editor_card)
        self.code_editor.buffer.connect("changed", self._validate_content)

        # Add a key controller to listen for the Escape key
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

        # Run initial validation
        self._validate_name()
        self._validate_content()

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Handler for key press events on the window."""
        if keyval == Gdk.KEY_Escape:
            self.close()
            return True  # Event handled, stop propagation
        return False

    def _validate_content(self, *args):
        """Warns (non-blocking) about lines that contain only a number."""
        buffer = self.text_view.get_buffer()
        start, end = buffer.get_start_iter(), buffer.get_end_iter()
        text = buffer.get_text(start, end, True)
        match = find_number_only_line(text)
        if match:
            self.warning_label.set_label(
                format_number_only_warning(match[1], lineno=match[0])
            )
            self.warning_box.set_visible(True)
        else:
            self.warning_box.set_visible(False)

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
        text = self.code_editor.get_text()

        if self._allow_name_edit:
            self.macro.name = self.name_row.get_text()
        self.macro.code = text.splitlines()

        self.saved = True
        self.close()
