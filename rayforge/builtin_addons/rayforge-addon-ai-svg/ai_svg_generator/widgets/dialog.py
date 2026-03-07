import logging
from gettext import gettext as _
from pathlib import Path
from typing import TYPE_CHECKING

from gi.repository import Adw, GLib, Gdk, Gtk

from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.ui_gtk.shared.patched_dialog_window import PatchedDialogWindow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor
    from ..controller import AISvgGeneratorController

logger = logging.getLogger(__name__)


class AISvgGeneratorDialog(PatchedDialogWindow):
    """Dialog for generating workpieces using AI."""

    def __init__(
        self,
        editor: "DocEditor",
        controller: "AISvgGeneratorController",
        parent=None,
    ):
        super().__init__(transient_for=parent)
        self._editor = editor
        self._controller = controller
        self._generating = False
        self._pulse_source_id = None

        self.set_title(_("Generate a Workpiece"))
        self.set_default_size(800, 500)

        self._main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(self._main_box)

        toolbar_view = Adw.ToolbarView()
        toolbar_view.set_vexpand(True)
        self._main_box.append(toolbar_view)

        header_bar = Adw.HeaderBar()
        toolbar_view.add_top_bar(header_bar)

        self._cancel_btn = Gtk.Button(label=_("Cancel"))
        self._cancel_btn.connect("clicked", self._on_cancel_clicked)
        header_bar.pack_start(self._cancel_btn)

        self._generate_btn = Gtk.Button(label=_("Generate"))
        self._generate_btn.add_css_class("suggested-action")
        self._generate_btn.connect("clicked", self._on_generate_clicked)
        header_bar.pack_end(self._generate_btn)

        content_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_start=12,
            margin_end=12,
            margin_top=12,
            margin_bottom=12,
        )
        toolbar_view.set_content(content_box)

        description = Gtk.Label(
            label=_(
                "Describe the design you want to create. "
                "Example: 'A simple star shape with 5 points'"
            ),
            wrap=True,
            xalign=0,
            css_classes=["dim-label", "caption"],
            margin_bottom=6,
        )
        content_box.append(description)

        self._buffer = Gtk.TextBuffer()
        self._text_view = Gtk.TextView(
            buffer=self._buffer,
            wrap_mode=Gtk.WrapMode.WORD_CHAR,
            vexpand=True,
            hexpand=True,
            top_margin=6,
            bottom_margin=6,
        )
        self._text_view.add_css_class("card")

        frame = Gtk.Frame(child=self._text_view)
        frame.set_vexpand(True)
        frame.add_css_class("flat")
        content_box.append(frame)

        self._error_label = Gtk.Label(
            wrap=True,
            css_classes=["error", "caption"],
            margin_top=6,
            visible=False,
        )
        content_box.append(self._error_label)

        self._progress_bar = Gtk.ProgressBar(
            hexpand=True,
            valign=Gtk.Align.END,
            visible=False,
        )
        self._progress_bar.add_css_class("thin-progress-bar")
        self._apply_progress_bar_style()
        self._main_box.append(self._progress_bar)

        self._buffer.connect("changed", self._on_prompt_changed)
        self._on_prompt_changed(self._buffer)

        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self._text_view.add_controller(key_controller)

    def _apply_progress_bar_style(self) -> None:
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(
            b"""
            progressbar.thin-progress-bar {
                min-height: 5px;
            }
            """
        )
        Gtk.StyleContext.add_provider_for_display(
            self.get_display(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _start_pulse(self) -> None:
        self._progress_bar.set_visible(True)
        self._progress_bar.pulse()
        self._pulse_source_id = GLib.timeout_add(100, self._on_pulse_timeout)

    def _stop_pulse(self) -> None:
        if self._pulse_source_id:
            GLib.source_remove(self._pulse_source_id)
            self._pulse_source_id = None
        self._progress_bar.set_visible(False)

    def _on_pulse_timeout(self) -> bool:
        self._progress_bar.pulse()
        return True

    def _on_prompt_changed(self, buffer) -> None:
        text = buffer.get_text(
            buffer.get_start_iter(), buffer.get_end_iter(), False
        )
        self._generate_btn.set_sensitive(
            len(text.strip()) > 0 and not self._generating
        )

    def _on_key_pressed(self, controller, keyval, keycode, state) -> bool:
        if keyval == Gdk.KEY_Return and (
            state & Gdk.ModifierType.CONTROL_MASK
        ):
            if not self._generating and self.get_prompt():
                self._start_generation()
            return True
        return False

    def _on_cancel_clicked(self, button) -> None:
        self._controller.cancel()
        self.close()

    def _on_generate_clicked(self, button) -> None:
        if self._generating:
            return
        self._start_generation()

    def _start_generation(self) -> None:
        prompt = self.get_prompt()
        if not prompt:
            return

        self.set_generating(True)

        def on_success(temp_path: Path) -> None:
            self._stop_pulse()
            machine_dims = self._editor.machine_dimensions
            position_mm = None
            if machine_dims:
                ws_width, ws_height = machine_dims
                position_mm = (ws_width / 2, ws_height / 2)

            self._editor.file.load_file_from_path(
                filename=temp_path,
                mime_type="image/svg+xml",
                vectorization_spec=PassthroughSpec(trim_padding=0),
                position_mm=position_mm,
            )
            self.close()

        def on_error(message: str) -> None:
            self._stop_pulse()
            self.set_generating(False)
            user_message = message
            if len(message) > 100:
                user_message = message[:100] + "..."
            self.set_error(user_message)

        self._start_pulse()
        self._controller.generate(prompt, on_success, on_error)

    def get_prompt(self) -> str:
        return self._buffer.get_text(
            self._buffer.get_start_iter(),
            self._buffer.get_end_iter(),
            False,
        ).strip()

    def set_error(self, message: str) -> None:
        self._error_label.set_text(message)
        self._error_label.set_visible(bool(message))

    def set_generating(self, generating: bool) -> None:
        self._generating = generating
        self._generate_btn.set_sensitive(not generating)
        self._cancel_btn.set_label(
            _("Close") if not generating else _("Cancel")
        )
        self._text_view.set_sensitive(not generating)
        if not generating:
            self.set_error("")
