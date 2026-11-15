"""
A card in the item properties panel that points to the settings of a
calibration test. It shows whenever the selection is (part of) one
calibration test. The settings themselves live in the settings of the
test's layer, like every other layer setting.
"""

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw, GObject, Graphene, Gtk

from rayforge.core.item import DocItem
from rayforge.core.layer import Layer
from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.doceditor.layer_settings_dialog import (
    LayerSettingsDialog,
)
from rayforge.ui_gtk.doceditor.property_providers.base import (
    PropertyProvider,
    property_provider_registry,
)
from rayforge.ui_gtk.mainwindow import MainWindow
from rayforge.ui_gtk.shared.expander import Expander

from ..calibration import GENERATORS, CalibrationGenerator
from ..calibration.lookup import record_for_items
from ..calibration.record import CalibrationRecord
from .calibration_settings import _show_text, _text_row

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class CalibrationPropertyProvider(PropertyProvider):
    """Points to the settings of a calibration test."""

    priority = 90
    separate_group = True
    group_title = _("Test Settings")

    def can_handle(self, items: list[DocItem]) -> bool:
        return record_for_items(items) is not None

    def create_widgets(self) -> list[Gtk.Widget]:
        self.help_row = _text_row("dim-label")
        self.open_button = Gtk.Button(label=_("Open Layer Settings"))
        self.open_button.connect("clicked", self._on_open_settings)
        self.button_row = Adw.ActionRow(activatable=False)
        self.button_row.add_suffix(self.open_button)
        self._record: CalibrationRecord | None = None
        self._layer: Layer | None = None
        self._scroll_handlers: list[tuple[GObject.Object, int]] = []
        return [self.help_row, self.button_row]

    @property
    def generator(self) -> CalibrationGenerator | None:
        if self._record is None:
            return None
        return GENERATORS.get(self._record.generator_id)

    def update_widgets(self, editor: "DocEditor", items: list[DocItem]):
        self.editor = editor
        self.items = items
        previous = self._record
        self._record = record_for_items(items)
        if self._record is not previous:
            self.stop_following()
        generator = self.generator
        if self._record is None or generator is None:
            return
        self.group_subtitle = self._record.name or generator.title
        _show_text(self.help_row, _settings_note(generator))
        layer = editor.doc.find_descendant_by_uid(self._record.layer_uid)
        self._layer = layer if isinstance(layer, Layer) else None

    def scroll_into_view(self):
        """
        Scrolls the panel so the Test Settings card starts at its top.
        The card comes after the workflows and the item properties, so
        it is often below the visible part of the panel. While the panel
        is still being laid out, the card is followed each time the
        panel's size changes, until another test is selected or the
        card is hidden.
        """
        scrolled = self._scrolled_window()
        card = self.help_row.get_ancestor(Expander)
        if scrolled is None or card is None:
            return
        self._follow_card(scrolled.get_vadjustment())
        if not self._scroll_handlers:
            adjustment = scrolled.get_vadjustment()
            self._scroll_handlers = [
                (adjustment, adjustment.connect("changed", self._follow_card)),
                (card, card.connect("notify::visible", self._on_card_shown)),
            ]

    def _follow_card(self, adjustment):
        scrolled = self._scrolled_window()
        offset = self._card_offset(scrolled) if scrolled else None
        if offset is not None:
            self._scroll_to(offset)

    def _on_card_shown(self, card: Gtk.Widget, _pspec):
        if not card.get_visible():
            self.stop_following()

    def stop_following(self):
        """Leaves the scroll position to the user."""
        for source, handler in self._scroll_handlers:
            source.disconnect(handler)
        self._scroll_handlers = []

    def _card_offset(self, scrolled: Gtk.ScrolledWindow) -> float | None:
        """Where the card starts in the panel, once it is laid out."""
        card = self.help_row.get_ancestor(Expander)
        content = scrolled.get_child()
        if isinstance(content, Gtk.Viewport):
            content = content.get_child()
        if card is None or content is None or card.get_height() == 0:
            return None
        ok, point = card.compute_point(content, Graphene.Point())
        return point.y if ok else None

    def _scrolled_window(self) -> Gtk.ScrolledWindow | None:
        """The scrolled right panel the settings are shown in."""
        scrolled = self.help_row.get_ancestor(Gtk.ScrolledWindow)
        return scrolled if isinstance(scrolled, Gtk.ScrolledWindow) else None

    def _scroll_to(self, offset: float):
        scrolled = self._scrolled_window()
        if scrolled is not None:
            adjustment = scrolled.get_vadjustment()
            top = adjustment.get_upper() - adjustment.get_page_size()
            adjustment.set_value(max(0.0, min(offset, top)))

    def _on_open_settings(self, button):
        if self._layer is None or self.editor is None:
            return
        root = self.help_row.get_root()
        if not isinstance(root, Gtk.Window):
            return
        open_layer_settings(root, self._layer, self.editor)


def open_layer_settings(
    window: Gtk.Window,
    layer: Layer,
    editor: "DocEditor",
):
    """Opens the settings dialog of the test's layer."""
    dialog = LayerSettingsDialog(layer, transient_for=window, editor=editor)
    dialog.present()


def show_new_test(window: MainWindow, items: list[WorkPiece]):
    """Selects a new test and opens the settings of its layer."""
    window.surface.select_items(items)
    for provider in window.item_props_widget.providers:
        if isinstance(provider, CalibrationPropertyProvider):
            provider.scroll_into_view()
    record = record_for_items(items)
    if record is None:
        return
    layer = window.doc_editor.doc.find_descendant_by_uid(record.layer_uid)
    if isinstance(layer, Layer):
        open_layer_settings(window, layer, window.doc_editor)


def register_test_settings(addon_name: str):
    """
    Registers the Test Settings, like the sketcher registers its own
    parameters. Called from the addon's init hook, not on import, so a
    module imported under two names does not register a second copy.
    """
    property_provider_registry.register(
        CalibrationPropertyProvider, addon_name
    )


def _settings_note(generator: CalibrationGenerator) -> str:
    return _(
        "This workpiece is part of the generated {test}. Its settings "
        "are in the settings of its layer."
    ).format(test=generator.title)
