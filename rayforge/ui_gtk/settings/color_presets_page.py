"""Settings page for managing color presets (color rules)."""

import logging
import uuid
from gettext import gettext as _
from typing import Any, Optional, cast

from blinker import Signal
from gi.repository import Adw, Gdk, Gtk

from ...context import get_context
from ...core.color import normalize_color
from ...core.color_preset import ColorPreset
from ...core.step_registry import step_registry
from ..icons import get_icon
from ..shared.preferences_group import PreferencesGroupWithButton
from ..shared.preferences_page import TrackedPreferencesPage

logger = logging.getLogger(__name__)


def _rgba_to_hex(rgba: Gdk.RGBA) -> str:
    """Convert a Gdk.RGBA to a lowercase hex string."""
    r = round(rgba.red * 255)
    g = round(rgba.green * 255)
    b = round(rgba.blue * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgba(color: str) -> Gdk.RGBA:
    """Parse a hex color string into a Gdk.RGBA, defaulting to magenta."""
    rgba = Gdk.RGBA()
    if not rgba.parse(color):
        rgba.parse("#ff00ff")
    return rgba


def _available_step_types() -> list[tuple[str, str]]:
    """
    Returns (class name, typelabel) pairs for all non-hidden steps.

    The list is derived from ``step_registry`` so any registered step
    type (including addon-provided ones) is selectable.
    """
    entries = []
    for name, cls in step_registry.all_steps().items():
        if getattr(cls, "HIDDEN", False):
            continue
        entries.append((name, getattr(cls, "TYPELABEL", name)))
    entries.sort(key=lambda e: e[1].lower())
    return entries


class ColorPresetDialog(Adw.MessageDialog):
    """A dialog for creating or editing a ColorPreset."""

    def __init__(
        self,
        parent: Optional[Gtk.Window],
        preset: Optional[ColorPreset] = None,
        **kwargs,
    ):
        super().__init__(transient_for=parent, modal=True, **kwargs)
        self.preset = preset
        is_editing = preset is not None

        self.set_default_size(520, -1)

        if is_editing:
            self.set_heading(_("Edit Color Rule"))
            self.set_body(_("Update the color rule details:"))
            self.add_response("cancel", _("Cancel"))
            self.add_response("save", _("Save"))
            self.set_response_appearance(
                "save", Adw.ResponseAppearance.SUGGESTED
            )
            self.set_default_response("save")
        else:
            self.set_heading(_("Add Color Rule"))
            self.set_body(_("Map a color to a step type for SVG imports."))
            self.add_response("cancel", _("Cancel"))
            self.add_response("add", _("Add"))
            self.set_response_appearance(
                "add", Adw.ResponseAppearance.SUGGESTED
            )
            self.set_default_response("add")

        self._step_types = _available_step_types()
        current_step_type = preset.step_type if preset else None
        if current_step_type and current_step_type not in [
            name for name, _ in self._step_types
        ]:
            # Keep unavailable step types selectable so the preset is
            # preserved (e.g. the providing addon was uninstalled).
            self._step_types.append(
                (current_step_type, f"{current_step_type} (unavailable)")
            )

        # --- Color picker ---
        color_dialog = Gtk.ColorDialog()
        color_dialog.set_with_alpha(False)
        self.color_button = Gtk.ColorDialogButton(dialog=color_dialog)
        self.color_button.set_size_request(48, 32)
        self.color_button.set_rgba(
            _hex_to_rgba(preset.color) if preset else _hex_to_rgba("#ff0000")
        )
        self.color_row = Adw.ActionRow(
            title=_("Color"),
            subtitle=_("SVG color that triggers this rule"),
        )
        self.color_row.add_suffix(self.color_button)
        self.color_row.set_activatable_widget(self.color_button)

        # --- Label ---
        self.label_row = Adw.EntryRow(title=_("Label (optional)"))

        # --- Step type ---
        self.step_type_row = Adw.ComboRow(
            title=_("Step Type"),
            subtitle=_("Step type created when this color is imported"),
        )
        self._step_type_model = Gtk.StringList()
        for _name, typelabel in self._step_types:
            self._step_type_model.append(typelabel)
        self.step_type_row.set_model(self._step_type_model)
        if current_step_type:
            self.step_type_row.set_selected(
                self._step_type_index(current_step_type)
            )

        group = Adw.PreferencesGroup()
        group.add(self.label_row)
        group.add(self.color_row)
        group.add(self.step_type_row)

        self.set_extra_child(group)

        if preset:
            self.label_row.set_text(preset.label)

    def _step_type_index(self, class_name: str) -> int:
        for index, (name, _label) in enumerate(self._step_types):
            if name == class_name:
                return index
        return 0

    def _selected_step_type(self) -> str:
        index = self.step_type_row.get_selected()
        return self._step_types[index][0]

    def get_preset_data(self) -> dict[str, Any]:
        """Returns the entered data as a dict suitable for ColorPreset."""
        return {
            "color": _rgba_to_hex(self.color_button.get_rgba()),
            "step_type": self._selected_step_type(),
            "label": self.label_row.get_text().strip(),
        }


class ColorPresetRow(Gtk.Box):
    """A widget representing a single ColorPreset in a ListBox."""

    def __init__(self, preset: ColorPreset, on_edit, on_delete):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.preset = preset

        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        self.append(self._create_swatch(preset.color))

        labels_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, hexpand=True
        )
        self.append(labels_box)

        title_text = preset.label or _("Color {color}").format(
            color=preset.color
        )
        title = Gtk.Label(label=title_text, halign=Gtk.Align.START, xalign=0)
        labels_box.append(title)

        subtitle = Gtk.Label(
            label=self._get_subtitle(),
            halign=Gtk.Align.START,
            xalign=0,
        )
        subtitle.add_css_class("dim-label")
        labels_box.append(subtitle)

        suffix_box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        self.append(suffix_box)

        if self._step_type_unavailable():
            warning = Gtk.Image.new_from_icon_name("dialog-warning-symbolic")
            warning.set_tooltip_text(
                _("This step type is not currently available.")
            )
            suffix_box.append(warning)

        edit_button = Gtk.Button(child=get_icon("edit-symbolic"))
        edit_button.add_css_class("flat")
        edit_button.connect("clicked", lambda w: on_edit(preset))
        suffix_box.append(edit_button)

        delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
        delete_button.add_css_class("flat")
        delete_button.connect("clicked", lambda w: on_delete(preset))
        suffix_box.append(delete_button)

    def _step_type_unavailable(self) -> bool:
        return step_registry.get(self.preset.step_type) is None

    def _get_subtitle(self) -> str:
        cls = step_registry.get(self.preset.step_type)
        if cls is None:
            return _("{step_type} (unavailable)").format(
                step_type=self.preset.step_type
            )
        typelabel = getattr(cls, "TYPELABEL", self.preset.step_type)
        return f"{self.preset.step_type} · {typelabel}"

    @staticmethod
    def _create_swatch(color: str) -> Gtk.Widget:
        rgba = _hex_to_rgba(color)
        swatch = Gtk.DrawingArea(width_request=24, height_request=24)
        swatch.set_valign(Gtk.Align.CENTER)

        def on_draw(widget, cr, width, height):
            cr.set_source_rgb(rgba.red, rgba.green, rgba.blue)
            cr.paint()

        swatch.set_draw_func(on_draw)
        return swatch


class ColorPresetListWidget(PreferencesGroupWithButton):
    """Displays a list of color presets and allows adding/editing/deleting."""

    def __init__(self, **kwargs):
        super().__init__(button_label=_("Add Color Rule"), **kwargs)
        self.color_presets_changed = Signal()

        placeholder = Gtk.Label(
            label=_("No color rules found."),
            halign=Gtk.Align.CENTER,
            margin_top=12,
            margin_bottom=12,
        )
        placeholder.add_css_class("dim-label")
        self.list_box.set_placeholder(placeholder)
        self.list_box.set_show_separators(True)

        self.populate_presets()

    def populate_presets(self):
        preset_mgr = get_context().color_preset_mgr
        presets = sorted(
            preset_mgr.all_presets(), key=lambda p: p.color.lower()
        )
        self.set_items(presets)

    def create_row_widget(self, item: ColorPreset) -> Gtk.Widget:
        return ColorPresetRow(
            item, self._on_edit_preset, self._on_delete_preset
        )

    def _on_add_clicked(self, button):
        root = self.get_root()
        parent_window = (
            cast(Gtk.Window, root) if isinstance(root, Gtk.Window) else None
        )
        dialog = ColorPresetDialog(parent=parent_window)

        def on_response(d, response_id):
            if response_id == "add":
                self._save_from_dialog(d)
            d.close()

        dialog.connect("response", on_response)
        dialog.present()

    def _on_edit_preset(self, preset: ColorPreset):
        root = self.get_root()
        parent_window = (
            cast(Gtk.Window, root) if isinstance(root, Gtk.Window) else None
        )
        dialog = ColorPresetDialog(parent=parent_window, preset=preset)

        def on_response(d, response_id):
            if response_id == "save":
                self._save_from_dialog(d, existing=preset)
            d.close()

        dialog.connect("response", on_response)
        dialog.present()

    def _save_from_dialog(
        self, dialog: ColorPresetDialog, existing: Optional[ColorPreset] = None
    ):
        data = dialog.get_preset_data()
        color = normalize_color(data["color"])
        if not color:
            return
        if existing and existing.color != color:
            # Replacing the color changes the key; drop the old entry.
            get_context().color_preset_mgr.delete_preset(existing.color)
        preset = ColorPreset(
            color=color,
            step_type=data["step_type"],
            label=data["label"],
            uid=existing.uid if existing else str(uuid.uuid4()),
        )
        get_context().color_preset_mgr.add_preset(preset)
        self.populate_presets()
        self.color_presets_changed.send(self)

    def _on_delete_preset(self, preset: ColorPreset):
        root = self.get_root()
        dialog = Adw.MessageDialog(
            transient_for=(
                cast(Gtk.Window, root)
                if isinstance(root, Gtk.Window)
                else None
            ),
            heading=_("Delete color rule '{color}'?").format(
                color=preset.color
            ),
            body=_(
                "The color rule will be permanently removed. "
                "This action cannot be undone."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("delete", _("Delete"))
        dialog.set_response_appearance(
            "delete", Adw.ResponseAppearance.DESTRUCTIVE
        )

        def on_response(d, response_id):
            if response_id == "delete":
                get_context().color_preset_mgr.delete_preset(preset.color)
                self.populate_presets()
                self.color_presets_changed.send(self)
            d.destroy()

        dialog.connect("response", on_response)
        dialog.present()


class ColorPresetPage(TrackedPreferencesPage):
    """Widget for managing color rules."""

    key = "color_presets"

    def __init__(self):
        super().__init__(title=_("Color Rules"), icon_name="palette-symbolic")

        self.color_preset_list_editor = ColorPresetListWidget(
            title=_("Color Rules"),
            description=_(
                "Map SVG colors to step types so they are applied "
                "automatically when importing."
            ),
        )
        self.add(self.color_preset_list_editor)
