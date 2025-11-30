import copy
from typing import List
from gi.repository import Adw, Gtk
from ...shared.ui.varsetwidget import VarSetWidget
from ..models.dialect import GcodeDialect


def _text_to_list(text: str) -> List[str]:
    """Converts a single string with newlines to a list of strings."""
    return text.strip().split("\n")


class DialectEditorDialog(Adw.Window):
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

        save_button = Gtk.Button(label=_("Save"))
        save_button.get_style_context().add_class("suggested-action")
        save_button.connect("clicked", self._on_save_clicked)
        header.pack_end(save_button)

        # Get the editor definition from the model
        varsets = self.dialect.get_editor_varsets()

        # Create and populate a VarSetWidget for each group defined by
        # the model
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

    def _validate(self) -> bool:
        """Validates that essential fields are filled."""
        values = self.info_widget.get_values()
        label = values.get("label", "").strip()
        return bool(label)

    def _on_save_clicked(self, button: Gtk.Button):
        if self._validate():
            self._update_dialect_from_ui()
            self.saved = True
            self.close()
