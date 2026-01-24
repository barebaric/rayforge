import logging
from typing import Optional, TYPE_CHECKING
from gi.repository import Adw, Gtk, Pango
from ...core.geo.font_config import FontConfig
from ...core.sketcher.commands.text_property import ModifyTextPropertyCommand
from ...core.sketcher.entities.text_box import TextBoxEntity
from ..shared.adwfix import get_spinrow_float
from ..icons import get_icon

if TYPE_CHECKING:
    from .editor import SketchEditor

logger = logging.getLogger(__name__)


class FontPropertiesWidget(Adw.PreferencesGroup):
    """
    A widget that displays font properties for a selected TextBoxEntity.
    Shows font family, size, bold, and italic options using Adw widgets.
    """

    def __init__(self, editor: "SketchEditor"):
        super().__init__()
        self.editor = editor
        self._text_entity_id: Optional[int] = None
        self._in_update = False
        self._current_font_family = "sans-serif"

        self.set_title(_("Font Properties"))
        self.set_visible(False)

        self._build_ui()

    def _build_ui(self):
        """Builds the UI for font properties."""
        self.font_family_row = Adw.ActionRow()
        self.font_family_row.set_title(_("Font Family"))
        self.font_family_row.set_subtitle(self._current_font_family)
        self.font_family_row.set_activatable(True)
        self.font_family_row.connect(
            "activated", self._on_font_family_row_activated
        )
        self.font_family_row.add_suffix(get_icon("go-next-symbolic"))
        self.add(self.font_family_row)

        adj = Gtk.Adjustment(
            value=10.0,
            lower=1.0,
            upper=500.0,
            step_increment=0.1,
        )
        self.font_size_row = Adw.SpinRow(adjustment=adj)
        self.font_size_row.set_title(_("Font Size"))
        self.font_size_row.set_digits(1)
        self.font_size_row.connect("notify::value", self._on_font_size_changed)
        self.add(self.font_size_row)

        self.bold_row = Adw.ActionRow()
        self.bold_row.set_title(_("Bold"))

        bold_switch = Gtk.Switch()
        bold_switch.set_valign(Gtk.Align.CENTER)
        bold_switch.connect("state-set", self._on_bold_changed)
        self.bold_row.add_suffix(bold_switch)
        self.bold_row.set_activatable_widget(bold_switch)
        self.bold_switch = bold_switch
        self.add(self.bold_row)

        self.italic_row = Adw.ActionRow()
        self.italic_row.set_title(_("Italic"))

        italic_switch = Gtk.Switch()
        italic_switch.set_valign(Gtk.Align.CENTER)
        italic_switch.connect("state-set", self._on_italic_changed)
        self.italic_row.add_suffix(italic_switch)
        self.italic_row.set_activatable_widget(italic_switch)
        self.italic_switch = italic_switch
        self.add(self.italic_row)

    def set_text_entity(self, entity_id: Optional[int]):
        """
        Sets the text entity to display font properties for.
        Hides the widget if entity_id is None.
        """
        self._text_entity_id = entity_id

        if entity_id is None:
            self.set_visible(False)
            return

        sketch_element = self.editor.sketch_element
        if not sketch_element:
            self.set_visible(False)
            return

        entity = sketch_element.sketch.registry.get_entity(entity_id)
        if not isinstance(entity, TextBoxEntity):
            self.set_visible(False)
            return

        self.set_visible(True)
        self._update_ui_from_model(entity.font_config)

    def _update_ui_from_model(self, font_config: FontConfig):
        """Updates the UI widgets from the font configuration."""
        self._in_update = True
        try:
            self.font_size_row.set_value(font_config.font_size)
            self.bold_switch.set_active(font_config.bold)
            self.italic_switch.set_active(font_config.italic)
            self._current_font_family = font_config.font_family
            self.font_family_row.set_subtitle(self._current_font_family)
        finally:
            self._in_update = False

    def _get_font_config_from_ui(self) -> FontConfig:
        """Creates a FontConfig from the current UI values."""
        return FontConfig(
            font_family=self._current_font_family,
            font_size=get_spinrow_float(self.font_size_row),
            bold=self.bold_switch.get_active(),
            italic=self.italic_switch.get_active(),
        )

    def _on_font_family_row_activated(self, row, *args):
        """Handles font family row activation to open font chooser."""
        if self._in_update or self._text_entity_id is None:
            return
        self._open_font_chooser_dialog()

    def _on_font_size_changed(self, row, *args):
        """Handles font size change."""
        if self._in_update or self._text_entity_id is None:
            return
        self._apply_font_config()

    def _on_bold_changed(self, switch, state):
        """Handles bold toggle change."""
        if self._in_update or self._text_entity_id is None:
            return
        self._apply_font_config()

    def _on_italic_changed(self, switch, state):
        """Handles italic toggle change."""
        if self._in_update or self._text_entity_id is None:
            return
        self._apply_font_config()

    def _open_font_chooser_dialog(self):
        """Opens a Gtk font chooser dialog for font selection."""
        dialog = Gtk.FontChooserDialog(
            title=_("Select Font"),
            transient_for=self.editor.parent_window
        )
        font_desc = self._get_font_description_from_ui()
        dialog.set_font_desc(font_desc)

        def on_response(dialog, response):
            logger.debug(f"Font chooser dialog response: {response}")
            if response == Gtk.ResponseType.OK:
                font_desc = dialog.get_font_desc()
                logger.debug(f"Selected font description: {font_desc}")
                if font_desc is not None:
                    self._update_ui_from_font_description(font_desc)
                    self._apply_font_config()
            dialog.destroy()

        dialog.connect("response", on_response)
        dialog.present()

    def _get_font_description_from_ui(self) -> Pango.FontDescription:
        """Creates a Pango.FontDescription from current UI values."""
        font_desc = Pango.FontDescription()
        font_desc.set_family(self._current_font_family)
        font_size_pt = get_spinrow_float(self.font_size_row)
        font_desc.set_size(int(font_size_pt * Pango.SCALE))
        style = (
            Pango.Style.ITALIC
            if self.italic_switch.get_active()
            else Pango.Style.NORMAL
        )
        font_desc.set_style(style)
        weight = (
            Pango.Weight.BOLD
            if self.bold_switch.get_active()
            else Pango.Weight.NORMAL
        )
        font_desc.set_weight(weight)
        return font_desc

    def _update_ui_from_font_description(
        self, font_desc: Pango.FontDescription
    ):
        """Updates UI widgets from a Pango.FontDescription."""
        self._in_update = True
        try:
            family = font_desc.get_family()
            if family:
                self._current_font_family = family
                self.font_family_row.set_subtitle(self._current_font_family)

            size = font_desc.get_size()
            if size > 0:
                font_size_pt = size / Pango.SCALE
                self.font_size_row.set_value(font_size_pt)

            style = font_desc.get_style()
            is_italic = style == Pango.Style.ITALIC
            self.italic_switch.set_active(is_italic)

            weight = font_desc.get_weight()
            is_bold = weight >= Pango.Weight.BOLD
            self.bold_switch.set_active(is_bold)
        finally:
            self._in_update = False

    def _apply_font_config(self):
        """Applies the current font configuration to the text entity."""
        if self._text_entity_id is None:
            return

        sketch_element = self.editor.sketch_element
        if not sketch_element:
            return

        entity = sketch_element.sketch.registry.get_entity(
            self._text_entity_id
        )
        if not isinstance(entity, TextBoxEntity):
            return

        new_font_config = self._get_font_config_from_ui()

        # Check if the text box is being edited and use the live buffer
        text_tool = sketch_element.tools.get("text_box")
        content = entity.content
        if text_tool and text_tool.editing_entity_id == self._text_entity_id:
            content = text_tool.text_buffer

        cmd = ModifyTextPropertyCommand(
            sketch=sketch_element.sketch,
            text_entity_id=self._text_entity_id,
            new_content=content,
            new_font_config=new_font_config,
        )
        self.editor.history_manager.execute(cmd)
