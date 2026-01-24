import logging
from typing import Optional, TYPE_CHECKING
from gi.repository import Adw, Gtk
from ...core.geo.font_config import FontConfig
from ...core.geo.text import get_available_font_families
from ...core.sketcher.commands.text_property import ModifyTextPropertyCommand
from ...core.sketcher.entities.text_box import TextBoxEntity
from ..shared.adwfix import get_spinrow_float

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

        self.set_title(_("Font Properties"))
        self.set_visible(False)

        self._build_ui()

    def _build_ui(self):
        """Builds the UI for font properties."""
        font_family_model = Gtk.StringList()
        for family in get_available_font_families():
            font_family_model.append(family)

        self.font_family_row = Adw.ComboRow()
        self.font_family_row.set_title(_("Font Family"))
        self.font_family_row.set_model(font_family_model)
        self.font_family_row.connect(
            "notify::selected-item", self._on_font_family_changed
        )
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

            model = self.font_family_row.get_model()
            if isinstance(model, Gtk.StringList):
                family = font_config.font_family
                for i in range(model.get_n_items()):
                    if model.get_string(i) == family:
                        self.font_family_row.set_selected(i)
                        break
                else:
                    self.font_family_row.set_selected(0)
        finally:
            self._in_update = False

    def _get_font_config_from_ui(self) -> FontConfig:
        """Creates a FontConfig from the current UI values."""
        model = self.font_family_row.get_model()
        font_family = "sans-serif"
        if isinstance(model, Gtk.StringList):
            selected = self.font_family_row.get_selected()
            if 0 <= selected < model.get_n_items():
                family = model.get_string(selected)
                if family is not None:
                    font_family = family

        return FontConfig(
            font_family=font_family,
            font_size=get_spinrow_float(self.font_size_row),
            bold=self.bold_switch.get_active(),
            italic=self.italic_switch.get_active(),
        )

    def _on_font_family_changed(self, row, *args):
        """Handles font family selection change."""
        if self._in_update or self._text_entity_id is None:
            return
        self._apply_font_config()

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
