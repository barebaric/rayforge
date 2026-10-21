"""
The settings of a calibration test in the item properties panel.

They show whenever the selection is (part of) one calibration test, so
they never cover the test on the canvas. Every change rebuilds the test
after a short pause, as one undo entry.
"""

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, GLib, GObject, Graphene, Gtk

from rayforge.core.item import DocItem
from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.doceditor.property_providers.base import (
    PropertyProvider,
)
from rayforge.ui_gtk.mainwindow import MainWindow
from rayforge.ui_gtk.shared.expander import Expander
from rayforge.ui_gtk.varset.varsetwidget import VarSetRowList

from ..calibration import GENERATORS, CalibrationGenerator
from ..calibration.lookup import record_for_items
from ..calibration.record import CalibrationRecord
from ..commands import CalibrationCmd

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

logger = logging.getLogger(__name__)

DEBOUNCE_DELAY_MS = 300


class CalibrationPropertyProvider(PropertyProvider):
    """Shows and applies the parameters of a calibration test."""

    priority = 90
    separate_group = True
    group_title = _("Test Settings")

    def can_handle(self, items: list[DocItem]) -> bool:
        return record_for_items(items) is not None

    def create_widgets(self) -> list[Gtk.Widget]:
        self.help_row = _text_row("dim-label")
        self.varset_widget = VarSetRowList()
        self.varset_widget.data_changed.connect(self._on_data_changed)
        self.warning_row = _text_row("warning")
        self.error_row = _text_row("error")
        self.note_row = _text_row("dim-label")

        self._record: CalibrationRecord | None = None
        self._generator_id: str | None = None
        self._pending_keys: set[str] = set()
        self._timer_id = 0
        self._scroll_handlers: list[tuple[GObject.Object, int]] = []
        return [
            self.help_row,
            self.warning_row,
            self.error_row,
            self.varset_widget,
            self.note_row,
        ]

    @property
    def generator(self) -> CalibrationGenerator | None:
        if self._record is None:
            return None
        return GENERATORS[self._record.generator_id]

    def update_widgets(self, editor: "DocEditor", items: list[DocItem]):
        self._in_update = True
        try:
            self.editor = editor
            self.items = items
            previous = self._record
            self._record = record_for_items(items)
            if self._record is not previous:
                self.stop_following()
            generator = self.generator
            if self._record is None or generator is None:
                return
            if generator.id != self._generator_id:
                self.varset_widget.clear_dynamic_rows()
                self._generator_id = generator.id
            params = CalibrationCmd(editor).params(self._record)
            self.varset_widget.populate(
                generator.settings_varset(params, editor.context)
            )
            self._show_params(params)
            self.error_row.set_visible(False)
            self.note_row.set_title(_rebuild_note(generator))
            self.group_subtitle = self._record.name or generator.title
        finally:
            self._in_update = False

    def scroll_into_view(self):
        """
        Scrolls the panel so the Test Settings card starts at its top.
        The card comes after the workflows and the item properties, so
        it is often below the visible part of the panel. While the panel
        is still being laid out, the card is followed each time the
        panel's size changes, until a setting is edited, another test is
        selected or the card is hidden.
        """
        scrolled = self._scrolled_window()
        card = self.varset_widget.get_ancestor(Expander)
        if scrolled is None or card is None:
            return
        self._follow_card(scrolled.get_vadjustment())
        if not self._scroll_handlers:
            adjustment = scrolled.get_vadjustment()
            self._scroll_handlers = [
                (adjustment, adjustment.connect("changed", self._follow_card)),
                (card, card.connect("notify::visible", self._on_card_shown)),
            ]

    def _follow_card(self, adjustment: Gtk.Adjustment):
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
        card = self.varset_widget.get_ancestor(Expander)
        content = scrolled.get_child()
        if isinstance(content, Gtk.Viewport):
            content = content.get_child()
        if card is None or content is None or card.get_height() == 0:
            return None
        ok, point = card.compute_point(content, Graphene.Point())
        return point.y if ok else None

    def _scrolled_window(self) -> Gtk.ScrolledWindow | None:
        """The scrolled right panel the settings are shown in."""
        scrolled = self.varset_widget.get_ancestor(Gtk.ScrolledWindow)
        return scrolled if isinstance(scrolled, Gtk.ScrolledWindow) else None

    def _scroll_offset(self) -> float:
        scrolled = self._scrolled_window()
        return scrolled.get_vadjustment().get_value() if scrolled else 0.0

    def _scroll_to(self, offset: float):
        scrolled = self._scrolled_window()
        if scrolled is not None:
            adjustment = scrolled.get_vadjustment()
            top = adjustment.get_upper() - adjustment.get_page_size()
            adjustment.set_value(max(0.0, min(offset, top)))

    def _show_params(self, params: dict[str, Any]):
        generator = self.generator
        assert generator is not None
        keys = set(self.varset_widget.keys())
        self.varset_widget.set_values(
            {k: v for k, v in params.items() if k in keys}
        )
        for key, (labels, subtitles) in generator.row_labels(params).items():
            adapter = self.varset_widget.adapter_for(key)
            if adapter is not None:
                adapter.set_item_labels(labels)
                adapter.set_item_subtitles(subtitles)
        _show_text(self.help_row, generator.help_text(params))
        warning = generator.warning(params, self.editor.context)
        _show_text(self.warning_row, warning)

    def _on_data_changed(self, sender: VarSetRowList, key: str):
        if self._in_update or self._record is None:
            return
        self._pending_keys.add(key)
        if self._timer_id:
            GLib.source_remove(self._timer_id)
        self._timer_id = GLib.timeout_add(DEBOUNCE_DELAY_MS, self._on_timer)

    def _on_timer(self) -> bool:
        self._timer_id = 0
        self.apply_now()
        return GLib.SOURCE_REMOVE

    def apply_now(self):
        """Rebuilds the test from the values in the rows."""
        if self._timer_id:
            GLib.source_remove(self._timer_id)
            self._timer_id = 0
        keys = set(self._pending_keys)
        self._pending_keys.clear()
        self.stop_following()
        record, generator = self._record, self.generator
        if not keys or record is None or generator is None:
            return
        cmd = CalibrationCmd(self.editor)
        values = self.varset_widget.get_values()
        params = cmd.params(record)
        params.update({key: values[key] for key in keys if key in values})
        params = generator.apply_change(params, keys, self.editor.context)
        self._in_update = True
        try:
            self._show_params(params)
        finally:
            self._in_update = False
        offset = self._scroll_offset()
        try:
            workpieces = cmd.update_params(record, params)
        except ValueError as e:
            _show_text(self.error_row, str(e))
            return
        self.error_row.set_visible(False)
        self._reselect(workpieces, offset)

    def _reselect(self, workpieces: list[WorkPiece], offset: float):
        """
        Selects the rebuilt test and puts the panel back where it was.
        The old workpieces are gone, so the selection was cleared; it is
        restored before the panel is drawn again.
        """
        root = self.varset_widget.get_root()
        if not isinstance(root, MainWindow):
            return
        root.surface.select_items(workpieces)
        self._scroll_to(offset)


def show_new_test(window: MainWindow, items: list[WorkPiece]):
    """Selects a new test and scrolls its settings into view."""
    window.surface.select_items(items)
    for provider in window.item_props_widget.providers:
        if isinstance(provider, CalibrationPropertyProvider):
            provider.scroll_into_view()


def _text_row(css_class: str) -> Adw.ActionRow:
    row = Adw.ActionRow(activatable=False)
    row.set_title_lines(0)
    row.add_css_class(css_class)
    row.set_visible(False)
    return row


def _show_text(row: Adw.ActionRow, text: str):
    row.set_title(GLib.markup_escape_text(text))
    row.set_visible(bool(text))


def _rebuild_note(generator: CalibrationGenerator) -> str:
    if generator.keeps_steps:
        return _(
            "Changing a setting rebuilds the test. The other settings of "
            "its operation stay."
        )
    return _(
        "Changing a setting rebuilds the test and replaces changes made "
        "to its operations."
    )
