import copy
import re
from gettext import gettext as _
from typing import cast

from gi.repository import Adw, Gtk

from ...machine.models.dialect import GcodeDialect
from ...pipeline.encoder.context import GcodeContext
from ..icons import get_icon
from ..shared.patched_dialog_window import PatchedDialogWindow
from ..varset.varsetwidget import VarSetWidget
from .template_selector import DialectTemplateSelectorDialog
from .validation import (
    find_number_only_line,
    format_number_only_warning,
    is_number_only,
)


def _text_to_list(text: str) -> list[str]:
    """
    Converts a single string with newlines to a list of non-empty strings.
    """
    return [line for line in text.strip().split("\n") if line.strip()]


def _get_template_validation_error(
    template: str, allowed_vars: set[str]
) -> str | None:
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

        self.update_from_template_button = Gtk.Button(
            label=_("Update from Template")
        )
        self.update_from_template_button.connect(
            "clicked", self._on_update_from_template_clicked
        )
        header.pack_end(self.update_from_template_button)

        # Get the editor definition from the model
        varsets = self.dialect.get_editor_varsets()

        self.info_widget = VarSetWidget()
        self.settings_widget = VarSetWidget()
        self.templates_widget = VarSetWidget()
        self.scripts_widget = VarSetWidget()

        self.info_widget.populate(varsets["info"])
        self.settings_widget.populate(varsets["settings"])
        self.templates_widget.populate(varsets["templates"])
        self.scripts_widget.populate(varsets["scripts"])

        form_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        form_box.set_margin_top(20)
        form_box.set_margin_start(50)
        form_box.set_margin_end(50)
        form_box.set_margin_bottom(50)
        form_box.append(self.info_widget)
        form_box.append(self.settings_widget)
        form_box.append(self.templates_widget)
        form_box.append(self.scripts_widget)

        scrolled_content = Gtk.ScrolledWindow(child=form_box)
        scrolled_content.set_vexpand(True)

        main_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        main_vbox.append(header)

        self.warning_banner = Adw.Banner()
        self.warning_banner.set_revealed(False)
        main_vbox.append(self.warning_banner)

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

    def _set_row_error(self, row: Adw.PreferencesRow, error_msg: str | None):
        """Applies or removes an error state from a row."""
        error_widget = getattr(row, "_error_icon_widget", None)

        if error_msg:
            if not error_widget:
                error_widget = get_icon("error-symbolic")
                if isinstance(
                    row, (Adw.ActionRow, Adw.ExpanderRow, Adw.EntryRow)
                ):
                    row.add_suffix(error_widget)
                row._error_icon_widget = (  # type: ignore[attr-defined]
                    error_widget
                )
            row.add_css_class("error")
            error_widget.set_tooltip_text(error_msg)
            error_widget.set_visible(True)
        else:
            row.remove_css_class("error")
            if error_widget:
                error_widget.set_visible(False)

    def _set_row_warning(
        self, row: Adw.PreferencesRow, warning_msg: str | None
    ):
        """Applies or removes a non-blocking warning state from a row."""
        warning_widget = getattr(row, "_warning_icon_widget", None)

        if warning_msg:
            if not warning_widget:
                warning_widget = get_icon("warning-symbolic")
                if isinstance(
                    row, (Adw.ActionRow, Adw.ExpanderRow, Adw.EntryRow)
                ):
                    row.add_suffix(warning_widget)
                row._warning_icon_widget = (  # type: ignore[attr-defined]
                    warning_widget
                )
            row.add_css_class("warning")
            warning_widget.set_tooltip_text(warning_msg)
            warning_widget.set_visible(True)
        else:
            row.remove_css_class("warning")
            if warning_widget:
                warning_widget.set_visible(False)

    def _get_row_content(
        self, row: Adw.PreferencesRow, is_script: bool
    ) -> str | None:
        """Returns the editable content of a row, if it has any."""
        if is_script:
            text_view = getattr(row, "core_widget", None)
            if isinstance(text_view, Gtk.TextView):
                buffer = text_view.get_buffer()
                start, end = buffer.get_start_iter(), buffer.get_end_iter()
                return buffer.get_text(start, end, True)
            return None
        if isinstance(row, Adw.EntryRow):
            return row.get_text()
        return None

    def _get_row_state(
        self, row: Adw.PreferencesRow, key: str, is_script: bool
    ) -> tuple[str | None, str | None]:
        """Returns the (error, warning) messages for a row."""
        content = self._get_row_content(row, is_script)
        if content is None:
            return None, None

        error_msg = None
        if is_script:
            # Find the first error in any line of the script
            for line in content.splitlines():
                error_msg = _get_template_validation_error(
                    line, self.supported_script_vars
                )
                if error_msg:
                    break
        else:
            allowed = self.supported_template_vars.get(key)
            if allowed is not None:
                error_msg = _get_template_validation_error(content, allowed)

        warning_msg = None
        if is_script:
            match = find_number_only_line(content)
            if match:
                warning_msg = format_number_only_warning(
                    match[1], lineno=match[0]
                )
        elif is_number_only(content):
            warning_msg = format_number_only_warning(content.strip())

        return error_msg, warning_msg

    def _on_row_changed(
        self, widget, row: Adw.PreferencesRow, key: str, is_script: bool
    ):
        """Callback for when a template or script field changes."""
        self._validate_all_rows()

    def _validate_all_rows(self):
        """Checks all rows for errors and warnings and updates the UI."""
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

        # Check all template and script rows
        first_warning = None
        for group, is_script in (
            (self.templates_widget, False),
            (self.scripts_widget, True),
        ):
            for key, (row, _var) in group.widget_map.items():
                error_msg, warning_msg = self._get_row_state(
                    row, key, is_script
                )
                self._set_row_error(row, error_msg)
                self._set_row_warning(row, warning_msg)
                if error_msg:
                    is_valid = False
                if warning_msg and not first_warning:
                    first_warning = warning_msg

        self._update_warning_banner(first_warning)
        self.save_button.set_sensitive(is_valid)

    def _update_warning_banner(self, warning: str | None):
        """Reveals the banner for a non-blocking G-code warning."""
        if warning:
            if self.warning_banner.get_title() != warning:
                self.warning_banner.set_title(warning)
                self.warning_banner.set_revealed(True)
        else:
            self.warning_banner.set_revealed(False)

    def _update_dialect_from_ui(self):
        """Updates the dialect object from the values in the VarSetWidgets."""
        all_values = {}
        all_values.update(self.info_widget.get_values())
        all_values.update(self.settings_widget.get_values())
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

    def _on_update_from_template_clicked(self, button: Gtk.Button):
        """Opens template selector to update dialect from a template."""
        parent = cast(Gtk.Window, self.get_transient_for())
        dialog = DialectTemplateSelectorDialog(
            transient_for=parent,
            title=_("Update from Template"),
            body=_(
                "Select a template to copy its settings. "
                "Your label and description will be preserved."
            ),
            on_selected=self._on_template_selected,
        )
        dialog.present()

    def _on_template_selected(self, template: GcodeDialect):
        """Updates the dialect from the selected template."""
        current_label = self.dialect.label
        current_description = self.dialect.description

        preserved_fields = {"uid", "is_custom", "label", "description"}
        for field in template.__dataclass_fields__:
            if field not in preserved_fields:
                value = getattr(template, field)
                setattr(self.dialect, field, copy.deepcopy(value))

        self.dialect.label = current_label
        self.dialect.description = current_description

        self._refresh_ui_from_dialect()
        self.present()

    def _refresh_ui_from_dialect(self):
        """Refreshes the UI widgets from the dialect object."""
        self.info_widget.clear_dynamic_rows()
        self.settings_widget.clear_dynamic_rows()
        self.templates_widget.clear_dynamic_rows()
        self.scripts_widget.clear_dynamic_rows()

        varsets = self.dialect.get_editor_varsets()
        self.info_widget.populate(varsets["info"])
        self.settings_widget.populate(varsets["settings"])
        self.templates_widget.populate(varsets["templates"])
        self.scripts_widget.populate(varsets["scripts"])
        self._connect_validation_signals()
        self._validate_all_rows()
