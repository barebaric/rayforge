"""Dialog for editing raw Markdown with a live preview."""

from gettext import gettext as _

from blinker import Signal
from gi.repository import Adw, Gtk

from .preview_editor import MarkdownPreviewEditor

_SUPPORTED_HINT = _(
    "Supported: headings, paragraphs, bold, italic, lists, links, "
    "inline code, fenced code blocks, and expandable details sections."
)


class MarkdownEditorDialog(Adw.Window):
    """Non-modal window for editing raw Markdown and its live preview.

    Pass ``modal=True`` to raise it as a modal dialog instead.
    """

    def __init__(
        self,
        title: str,
        description: str | None = None,
        initial_text: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_title(title)
        self.set_default_size(1150, 750)
        self._result: str | None = None
        self._action_closed = False
        self.saved = Signal()
        self.cancelled = Signal()
        self.preview_editor = MarkdownPreviewEditor(initial_text)

        toolbar_view = Adw.ToolbarView()
        self.set_content(toolbar_view)

        header_bar = Adw.HeaderBar()
        header_bar.set_show_end_title_buttons(False)
        toolbar_view.add_top_bar(header_bar)

        self.cancel_button = Gtk.Button(label=_("Cancel"))
        self.cancel_button.connect("clicked", self._on_cancel)
        header_bar.pack_start(self.cancel_button)

        self.save_button = Gtk.Button(label=_("Save"))
        self.save_button.add_css_class("suggested-action")
        self.save_button.connect("clicked", self._on_save)
        header_bar.pack_end(self.save_button)

        root = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_top=18,
            margin_bottom=18,
            margin_start=18,
            margin_end=18,
        )
        if description:
            root.append(Gtk.Label(label=description, xalign=0, wrap=True))
        root.append(self.preview_editor)
        hint = Gtk.Label(label=_SUPPORTED_HINT, xalign=0, wrap=True)
        hint.add_css_class("dim-label")
        root.append(hint)
        toolbar_view.set_content(root)

    def get_text(self) -> str:
        return self.preview_editor.editor.get_text()

    def get_result(self) -> str | None:
        return self._result

    def _on_save(self, *_args):
        self._result = self.get_text()
        self._action_closed = True
        self.saved.send(self, text=self._result)
        self.close()

    def _on_cancel(self, *_args):
        self._result = None
        self._action_closed = True
        self.cancelled.send(self)
        self.close()

    def do_close_request(self):
        if not self._action_closed:
            self._action_closed = True
            self.cancelled.send(self)
        return False
