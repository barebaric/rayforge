import copy
from typing import List, cast
from gi.repository import Adw, Gtk
from ...machine.models.dialect import GcodeDialect


def _list_to_text(lines: List[str]) -> str:
    """Converts a list of strings to a single string with newlines."""
    return "\n".join(lines)


def _text_to_list(text: str) -> List[str]:
    """Converts a single string with newlines to a list of strings."""
    return text.strip().split("\n")


class DialectEditorDialog(Adw.Window):
    """A dialog window for creating or editing a G-code dialect."""

    def __init__(
        self,
        parent: Gtk.Window,
        dialect: GcodeDialect,
    ):
        super().__init__(transient_for=parent)
        self.set_default_size(600, 500)

        self.dialect = copy.deepcopy(dialect)  # Work on a copy
        self.saved = False

        self._widgets: dict[str, Gtk.Widget] = {}

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

        # General Info
        info_group = Adw.PreferencesGroup(title=_("General Information"))
        self._add_entry_row(
            info_group, "label", _("Label"), _("User-facing name")
        )
        self._add_entry_row(
            info_group, "description", _("Description"), _("Short description")
        )

        # Templates
        templates_group = Adw.PreferencesGroup(title=_("Command Templates"))
        self._add_entry_row(templates_group, "laser_on", _("Laser On"))
        self._add_entry_row(templates_group, "laser_off", _("Laser Off"))
        self._add_entry_row(templates_group, "travel_move", _("Travel Move"))
        self._add_entry_row(templates_group, "linear_move", _("Linear Move"))
        self._add_entry_row(templates_group, "arc_cw", _("Arc (CW)"))
        self._add_entry_row(templates_group, "arc_ccw", _("Arc (CCW)"))
        self._add_entry_row(templates_group, "tool_change", _("Tool Change"))
        self._add_entry_row(templates_group, "set_speed", _("Set Speed"))
        self._add_entry_row(templates_group, "air_assist_on", _("Air On"))
        self._add_entry_row(templates_group, "air_assist_off", _("Air Off"))

        # Preamble & Postscript
        pre_post_group = Adw.PreferencesGroup(title=_("Scripts"))
        self._add_text_view_row(pre_post_group, "preamble", _("Preamble"))
        self._add_text_view_row(pre_post_group, "postscript", _("Postscript"))

        form_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        form_box.set_margin_top(20)
        form_box.set_margin_start(50)
        form_box.set_margin_end(50)
        form_box.set_margin_bottom(50)
        form_box.append(info_group)
        form_box.append(templates_group)
        form_box.append(pre_post_group)

        scrolled_content = Gtk.ScrolledWindow(child=form_box)
        scrolled_content.set_vexpand(True)

        main_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        main_vbox.append(header)
        main_vbox.append(scrolled_content)
        self.set_content(main_vbox)
        self._populate_from_dialect()

    def _add_entry_row(
        self, group: Adw.PreferencesGroup, key: str, title: str, tooltip=""
    ):
        row = Adw.EntryRow(title=title)
        if tooltip:
            row.set_tooltip_text(tooltip)
        self._widgets[key] = row
        group.add(row)

    def _add_text_view_row(
        self, group: Adw.PreferencesGroup, key: str, title: str
    ):
        row = Adw.ExpanderRow(title=title)
        text_view = Gtk.TextView(
            monospace=True, wrap_mode=Gtk.WrapMode.WORD_CHAR
        )
        text_view.get_style_context().add_class("gcode-editor")
        scroller = Gtk.ScrolledWindow(
            child=text_view,
            min_content_height=100,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
        )
        row.add_row(scroller)
        self._widgets[key] = text_view
        group.add(row)

    def _populate_from_dialect(self):
        for key, widget in self._widgets.items():
            value = getattr(self.dialect, key, None)
            if value is None:
                continue
            if isinstance(widget, Adw.EntryRow):
                widget.set_text(str(value))
            elif isinstance(widget, Gtk.TextView):
                widget.get_buffer().set_text(_list_to_text(value))

    def _update_dialect_from_ui(self):
        for key, widget in self._widgets.items():
            if isinstance(widget, Adw.EntryRow):
                setattr(self.dialect, key, widget.get_text())
            elif isinstance(widget, Gtk.TextView):
                buffer = widget.get_buffer()
                start = buffer.get_start_iter()
                end = buffer.get_end_iter()
                text = buffer.get_text(start, end, True)
                setattr(self.dialect, key, _text_to_list(text))

    def _validate(self) -> bool:
        """Validates that essential fields are filled."""
        label_widget = cast(Adw.EntryRow, self._widgets["label"])
        label = label_widget.get_text().strip()
        if not label:
            return False
        return True

    def _on_save_clicked(self, button: Gtk.Button):
        if self._validate():
            self._update_dialect_from_ui()
            self.saved = True
            self.close()
