"""
The settings of a calibration test in the item properties panel.

They show whenever the selection is (part of) one calibration test, so
they never cover the test on the canvas. Every change rebuilds the test
after a short pause, as one undo entry.
"""

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, GLib, Graphene, Gtk

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
#: Time the item properties need to slide in before they can be scrolled.
REVEAL_DELAY_MS = 400


class CalibrationPropertyProvider(PropertyProvider):
    """Shows and applies the parameters of a calibration test."""

    priority = 90
    separate_group = True
    group_title = _("Test Settings")

    def can_handle(self, items: list[DocItem]) -> bool:
        return record_for_items(items) is not None

    def create_widgets(self) -> list[Gtk.Widget]:
        self.help_row = Adw.ActionRow(activatable=False)
        self.help_row.set_title_lines(0)
        self.help_row.add_css_class("dim-label")
        self.varset_widget = VarSetRowList()
        self.varset_widget.data_changed.connect(self._on_data_changed)
        self.error_row = Adw.ActionRow(activatable=False)
        self.error_row.set_title_lines(0)
        self.error_row.add_css_class("error")
        self.error_row.set_visible(False)
        self.note_row = Adw.ActionRow(
            title=_(
                "Changing a setting rebuilds the test and replaces changes "
                "made to its operations."
            ),
            activatable=False,
        )
        self.note_row.set_title_lines(0)
        self.note_row.add_css_class("dim-label")

        self._record: CalibrationRecord | None = None
        self._generator_id: str | None = None
        self._pending_keys: set[str] = set()
        self._timer_id = 0
        return [
            self.help_row,
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
            self._record = record_for_items(items)
            generator = self.generator
            if self._record is None or generator is None:
                return
            if generator.id != self._generator_id:
                self.varset_widget.clear_dynamic_rows()
                self._generator_id = generator.id
            self.varset_widget.populate(generator.varset(editor.context))
            self._show_params(self._record.params)
            self.error_row.set_visible(False)
            self.group_subtitle = self._record.name or generator.title
        finally:
            self._in_update = False

    def scroll_into_view(self) -> bool:
        """
        Scrolls the panel so the Test Settings card starts at its top.
        The card comes after the workflows and the item properties, so
        it is often below the visible part of the panel.
        """
        card = self.varset_widget.get_ancestor(Expander)
        scrolled = self._scrolled_window()
        content = scrolled.get_child() if scrolled else None
        if card is None or content is None:
            return GLib.SOURCE_REMOVE
        ok, point = card.compute_point(content, Graphene.Point())
        if ok:
            self._scroll_to(point.y)
        return GLib.SOURCE_REMOVE

    def _scrolled_window(self) -> Gtk.ScrolledWindow | None:
        """The scrolled right panel the settings are shown in."""
        scrolled = self.varset_widget.get_ancestor(Gtk.ScrolledWindow)
        return scrolled if isinstance(scrolled, Gtk.ScrolledWindow) else None

    def _scroll_offset(self) -> float:
        scrolled = self._scrolled_window()
        return scrolled.get_vadjustment().get_value() if scrolled else 0.0

    def _scroll_to(self, offset: float) -> bool:
        scrolled = self._scrolled_window()
        if scrolled is not None:
            adjustment = scrolled.get_vadjustment()
            top = adjustment.get_upper() - adjustment.get_page_size()
            adjustment.set_value(max(0.0, min(offset, top)))
        return GLib.SOURCE_REMOVE

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
        help_text = generator.help_text(params)
        self.help_row.set_title(GLib.markup_escape_text(help_text))
        self.help_row.set_visible(bool(help_text))

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
        record, generator = self._record, self.generator
        if not keys or record is None or generator is None:
            return
        params = record.params
        params.update(self.varset_widget.get_values())
        params = generator.apply_change(params, keys, self.editor.context)
        self._in_update = True
        try:
            self._show_params(params)
        finally:
            self._in_update = False
        offset = self._scroll_offset()
        try:
            workpieces = CalibrationCmd(self.editor).update_params(
                record, params
            )
        except ValueError as e:
            self.error_row.set_title(GLib.markup_escape_text(str(e)))
            self.error_row.set_visible(True)
            return
        self.error_row.set_visible(False)
        self._reselect(workpieces, offset)

    def _reselect(self, workpieces: list[WorkPiece], offset: float):
        """
        Selects the rebuilt test and puts the panel back where it was:
        the old workpieces are gone, so the selection was cleared and the
        item properties slid out.
        """
        root = self.varset_widget.get_root()
        if not isinstance(root, MainWindow):
            return
        GLib.idle_add(select_items, root, workpieces)
        GLib.timeout_add(REVEAL_DELAY_MS, self._scroll_to, offset)


def select_items(window: MainWindow, items: list[WorkPiece]) -> bool:
    """
    Selects a rebuilt test, so its settings stay in the panel. Runs on
    idle, once the canvas has elements for the new workpieces.
    """
    window.surface.select_items(items)
    return GLib.SOURCE_REMOVE


def show_new_test(window: MainWindow, items: list[WorkPiece]) -> bool:
    """
    Selects a new test and scrolls its settings into view once the
    item properties have slid in.
    """
    select_items(window, items)
    for provider in window.item_props_widget.providers:
        if isinstance(provider, CalibrationPropertyProvider):
            GLib.timeout_add(REVEAL_DELAY_MS, provider.scroll_into_view)
    return GLib.SOURCE_REMOVE
