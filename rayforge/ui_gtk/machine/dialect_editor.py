import copy
import re
from typing import List, Set, cast, Optional
from gi.repository import Adw, Gtk
from ...machine.models.dialect import GcodeDialect
from ...pipeline.encoder.context import GcodeContext
from ..icons import get_icon
from ..shared.patched_dialog_window import PatchedDialogWindow
from ..varset.varsetwidget import VarSetWidget


def _text_to_list(text: str) -> List[str]:
    """
    Converts a single string with newlines to a list of non-empty strings.
    """
    return [line for line in text.strip().split("\n") if line.strip()]


def _get_template_validation_error(
    template: str, allowed_vars: Set[str]
) -> Optional[str]:
    """
    Validates a template's syntax and variable names, returning an error
    string if invalid, or None if valid.
    """
    # 1. Check for basic syntax errors first.
    if "{{" in template or "}}" in template:
        return _("Escaped braces {{ or }} are not supported.")

    depth = 0
    in_brace = False
    for char in template:
        if char == "{":
            if in_brace:
                return _("Nested braces are not allowed.")
            depth += 1
            in_brace = True
        elif char == "}":
            if not in_brace:
                return _("Unmatched closing brace '}' found.")
            depth -= 1
            in_brace = False

    if depth != 0:
        return _("Unmatched opening brace '{' found.")

    # 2. Syntax is valid, now check the variables themselves.
    found_vars = re.findall(r"\{([^}]+)\}", template)
    invalid_vars = []
    for var in found_vars:
        if not var:
            return _("Empty braces '{}' are not allowed.")
        # Strip format specifier (e.g., from 'power:.0f' to 'power')
        base_var = var.split(":")[0]
        if base_var not in allowed_vars:
            invalid_vars.append(var)

    if invalid_vars:
        return _("Unsupported variable(s): {vars}").format(
            vars=", ".join(f"{{{v}}}" for v in invalid_vars)
        )

    return None  # All checks passed


class DialectEditorDialog(PatchedDialogWindow):
    """
    A dialog window for creating or editing a G-code dialect.
    This dialog is driven by VarSets provided by the GcodeDialect model itself.
    """

    def __init__(
        self,
        parent: Gtk.Window,
        dialect: GcodeDialect,
    ):
        super().__init__(transient_for=parent)
        self.set_default_size(600, 500)

        self.dialect = copy.deepcopy(dialect)
        self.saved = False
        self.supported_template_vars = (
            GcodeContext.get_template_variable_docs()
        )
        script_vars_docs = GcodeContext.get_docs("job")
        self.supported_script_vars = {var[0] for var in script_vars_docs}

        title = (
            _("Edit Dialect: {label}").format(label=self.dialect.label)
            if self.dialect.is_custom
            else _("New Dialect")
        )
        self.set_title(title)
        self.set_default_size(800, 800)

        header = Adw.HeaderBar()
        cancel_button = Gtk.Button(label=_("Cancel"))
        cancel_button.connect("clicked", lambda w: self.close())
        header.pack_start(cancel_button)

        self.save_button = Gtk.Button(label=_("Save"))
        self.save_button.get_style_context().add_class("suggested-action")
        self.save_button.connect("clicked", self._on_save_clicked)
        header.pack_end(self.save_button)

        # Get the editor definition from the model
        varsets = self.dialect.get_editor_varsets()

        self.info_widget = VarSetWidget()
        self.templates_widget = VarSetWidget()
        self.scripts_widget = VarSetWidget()

        self.info_widget.populate(varsets["info"])
        self.templates_widget.populate(varsets["templates"])
        self.scripts_widget.populate(varsets["scripts"])

        form_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        form_box.set_margin_top(20)
        form_box.set_margin_start(50)
        form_box.set_margin_end(50)
        form_box.set_margin_bottom(50)
        form_box.append(self.info_widget)
        form_box.append(self.templates_widget)
        form_box.append(self.scripts_widget)

        scrolled_content = Gtk.ScrolledWindow(child=form_box)
        scrolled_content.set_vexpand(True)

        main_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        main_vbox.append(header)
        main_vbox.append(scrolled_content)
        self.set_content(main_vbox)

        self._connect_validation_signals()
        self._validate_all_rows()  # Set initial state

    def _connect_validation_signals(self):
        """Connects `changed` signals for all relevant input widgets."""
        # Info section (Label)
        label_row = self.info_widget.widget_map.get("label", (None,))[0]
        if isinstance(label_row, Adw.EntryRow):
            label_row.connect("changed", lambda r: self._validate_all_rows())

        # Templates section
        for key, (row, var) in self.templates_widget.widget_map.items():
            if isinstance(row, Adw.EntryRow):
                row.connect("changed", self._on_row_changed, row, key, False)

        # Scripts section
        for key, (row, var) in self.scripts_widget.widget_map.items():
            text_view = getattr(row, "core_widget", None)
            if isinstance(text_view, Gtk.TextView):
                buffer = text_view.get_buffer()
                buffer.connect("changed", self._on_row_changed, row, key, True)

    def _set_row_error(
        self, row: Adw.PreferencesRow, error_msg: Optional[str]
    ):
        """Applies or removes an error state from a row."""
        # Ensure the error icon widget exists as a suffix.
        if not hasattr(row, "_error_icon_widget"):
            icon = get_icon("dialog-error-symbolic")
            # Adw.ActionRow and Adw.ExpanderRow support add_suffix
            if isinstance(row, (Adw.ActionRow, Adw.ExpanderRow, Adw.EntryRow)):
                row.add_suffix(icon)
            setattr(row, "_error_icon_widget", icon)

        error_widget = cast(
            Gtk.Image, getattr(row, "_error_icon_widget", None)
        )
        if not error_widget:
            return

        if error_msg:
            row.add_css_class("error")
            error_widget.set_tooltip_text(error_msg)
            error_widget.set_visible(True)
        else:
            row.remove_css_class("error")
            error_widget.set_visible(False)

    def _on_row_changed(
        self, widget, row: Adw.PreferencesRow, key: str, is_script: bool
    ):
        """Callback for when a template or script field changes."""
        error_msg = None
        if is_script:
            text_view = getattr(row, "core_widget", None)
            if isinstance(text_view, Gtk.TextView):
                buffer = text_view.get_buffer()
                start, end = buffer.get_start_iter(), buffer.get_end_iter()
                content = buffer.get_text(start, end, True)
                # Find the first error in any line of the script
                for line in content.splitlines():
                    error_msg = _get_template_validation_error(
                        line, self.supported_script_vars
                    )
                    if error_msg:
                        break
        elif isinstance(row, Adw.EntryRow):
            content = row.get_text()
            allowed = self.supported_template_vars.get(key)
            if allowed is not None:
                error_msg = _get_template_validation_error(content, allowed)

        self._set_row_error(row, error_msg)
        self._validate_all_rows()

    def _validate_all_rows(self):
        """Checks all rows for errors and updates Save button sensitivity."""
        is_valid = True
        # Check label
        label_row = cast(
            Adw.EntryRow, self.info_widget.widget_map.get("label", (None,))[0]
        )
        if not label_row or not label_row.get_text().strip():
            is_valid = False
            self._set_row_error(label_row, _("Label cannot be empty."))
        else:
            self._set_row_error(label_row, None)

        # Check all other rows for the 'error' class
        for group in (self.templates_widget, self.scripts_widget):
            for row, _var in group.widget_map.values():
                if row.has_css_class("error"):
                    is_valid = False
                    break
            if not is_valid:
                break

        self.save_button.set_sensitive(is_valid)

    def _update_dialect_from_ui(self):
        """Updates the dialect object from the values in the VarSetWidgets."""
        all_values = {}
        all_values.update(self.info_widget.get_values())
        all_values.update(self.templates_widget.get_values())
        all_values.update(self.scripts_widget.get_values())

        for key, value in all_values.items():
            if key in ("preamble", "postscript"):
                # Convert multi-line text back to list of strings
                setattr(self.dialect, key, _text_to_list(value))
            elif hasattr(self.dialect, key):
                setattr(self.dialect, key, value)

    def _on_save_clicked(self, button: Gtk.Button):
        # Validation is now continuous, so we can just save.
        self._update_dialect_from_ui()
        self.saved = True
        self.close()
