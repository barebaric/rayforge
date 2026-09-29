"""Composition of the raw editor, live preview, and formatting toolbar."""

from gettext import gettext as _

from gi.repository import GLib, Gtk

from ..even_split_paned import EvenSplitPaned
from .editor import MarkdownEditor
from .help import build_help_markdown
from .toolbar import MarkdownToolbar
from .view import MarkdownView


class MarkdownPreviewEditor(Gtk.Box):
    """Compose a raw editor and preview with debounced updates."""

    SIDE_BY_SIDE = "side-by-side"
    STACKED = "stacked"
    _NARROW_WIDTH = 700

    def __init__(self, text: str = "", **kwargs):
        super().__init__(
            orientation=Gtk.Orientation.VERTICAL, spacing=8, **kwargs
        )
        self.editor = MarkdownEditor(text)
        self.preview = MarkdownView(text)
        self.help_view = MarkdownView(build_help_markdown())
        self.toolbar = MarkdownToolbar(self.editor)
        self.toolbar.help_toggled.connect(self._on_help_toggled)
        self.layout_dropdown = Gtk.DropDown()
        self.layout_dropdown.set_model(
            Gtk.StringList.new([_("Side by side"), _("Stacked")])
        )
        self.layout_dropdown.set_selected(1)
        self.layout_dropdown.connect("notify::selected", self._layout_changed)
        self._updating_layout = False
        self._paned = EvenSplitPaned(orientation=Gtk.Orientation.VERTICAL)
        self.preview_scroller = Gtk.ScrolledWindow(child=self.preview)
        self.preview_scroller.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        self.preview_scroller.set_vexpand(True)
        self.preview_scroller.set_hexpand(True)
        self.help_scroller = Gtk.ScrolledWindow(child=self.help_view)
        self.help_scroller.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        self.help_scroller.set_vexpand(True)
        self.help_scroller.set_hexpand(True)
        self.preview_stack = Gtk.Stack()
        self.preview_stack.add_named(self.preview_scroller, "preview")
        self.preview_stack.add_named(self.help_scroller, "help")
        self.editor_pane, _editor_title = self._make_pane(
            _("Raw Markdown"), self.editor
        )
        self.preview_pane, self.preview_title = self._make_pane(
            _("Preview"), self.preview_stack
        )
        for pane in (self.editor_pane, self.preview_pane):
            pane.set_hexpand(True)
            pane.set_vexpand(True)
        self._paned.set_start_child(self.editor_pane)
        self._paned.set_end_child(self.preview_pane)
        self.action_bar = Gtk.Box(spacing=12)
        self.toolbar.set_hexpand(True)
        self.action_bar.append(self.toolbar)
        self.action_bar.append(self.layout_dropdown)
        self.append(self.action_bar)
        self.append(self._paned)
        self._paned.set_vexpand(True)
        self._user_selected_layout = False
        self._timeout_id = 0
        self.editor.buffer.connect("changed", self._on_source_changed)
        self._paned.connect("notify::max-position", self._on_paned_resized)
        self.connect("destroy", self._on_destroy)
        self._on_width_changed()

    def _on_destroy(self, *_args):
        self._cancel_pending_refresh()

    def _on_paned_resized(self, *_args):
        """Allocation changes may make the responsive layout stale."""
        self._on_width_changed()

    @staticmethod
    def _make_pane(title: str, child: Gtk.Widget) -> tuple[Gtk.Box, Gtk.Label]:
        pane = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        label = Gtk.Label(label=title, xalign=0)
        label.add_css_class("heading")
        pane.append(label)
        pane.append(child)
        return pane, label

    def _on_help_toggled(self, _toolbar, active: bool):
        self.preview_stack.set_visible_child_name(
            "help" if active else "preview"
        )
        self.preview_title.set_label(
            _("Formatting help") if active else _("Preview")
        )

    def show_help(self, active: bool) -> None:
        """Swap the preview pane for the formatting help."""
        self.toolbar.set_help_active(active)

    def _layout_changed(self, dropdown, _pspec):
        if self._updating_layout:
            return
        self._user_selected_layout = True
        self.set_layout(
            self.STACKED if dropdown.get_selected() == 1 else self.SIDE_BY_SIDE
        )

    def _on_width_changed(self, *_args):
        if self._user_selected_layout:
            return
        self.set_layout(
            self.STACKED
            if not self.get_width() or self.get_width() < self._NARROW_WIDTH
            else self.SIDE_BY_SIDE,
        )

    def set_layout(self, layout: str, update_dropdown: bool = True) -> None:
        if layout not in (self.SIDE_BY_SIDE, self.STACKED):
            raise ValueError(f"Unknown Markdown layout: {layout}")
        orientation = (
            Gtk.Orientation.VERTICAL
            if layout == self.STACKED
            else Gtk.Orientation.HORIZONTAL
        )
        if self._paned.get_orientation() != orientation:
            self._paned.set_orientation(orientation)
            self._paned.reset_split()
        self._update_pane_spacing()
        if update_dropdown:
            self._updating_layout = True
            try:
                self.layout_dropdown.set_selected(
                    1 if layout == self.STACKED else 0
                )
            finally:
                self._updating_layout = False

    def _update_pane_spacing(self) -> None:
        """Keep a gutter between the panes and the split handle."""
        spacing = 6
        stacked = self._paned.get_orientation() == Gtk.Orientation.VERTICAL
        self.editor_pane.set_margin_bottom(spacing if stacked else 0)
        self.editor_pane.set_margin_end(0 if stacked else spacing)
        self.preview_pane.set_margin_top(spacing if stacked else 0)
        self.preview_pane.set_margin_start(0 if stacked else spacing)

    def get_layout(self) -> str:
        return (
            self.STACKED
            if self._paned.get_orientation() == Gtk.Orientation.VERTICAL
            else self.SIDE_BY_SIDE
        )

    def _on_source_changed(self, *_args):
        self.show_help(False)
        self._cancel_pending_refresh()
        self._timeout_id = GLib.timeout_add(250, self._refresh_preview)

    def _refresh_preview(self):
        self.preview.set_text(self.editor.get_text())
        self._timeout_id = 0
        return GLib.SOURCE_REMOVE

    def _cancel_pending_refresh(self) -> None:
        if self._timeout_id:
            GLib.source_remove(self._timeout_id)
            self._timeout_id = 0

    def refresh_preview(self) -> None:
        self._cancel_pending_refresh()
        self._refresh_preview()
