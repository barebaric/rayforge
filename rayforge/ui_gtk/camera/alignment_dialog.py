import logging
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...camera.controller import CameraController
from ..icons import get_icon
from ..shared.patched_dialog_window import PatchedDialogWindow
from .alignment_widget import CameraAlignment

logger = logging.getLogger(__name__)


class CameraAlignmentDialog(PatchedDialogWindow):
    def __init__(self, parent, controller: CameraController, **kwargs):
        super().__init__(
            transient_for=parent,
            modal=True,
            default_width=1280,
            default_height=960,
            **kwargs,
        )

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(content)

        header_bar = Adw.HeaderBar()
        header_title = _("{camera_name} – Image Alignment").format(
            camera_name=controller.config.name
        )
        header_bar.set_title_widget(
            Adw.WindowTitle(title=header_title, subtitle="")
        )
        content.append(header_bar)

        zoom_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        zoom_box.add_css_class("linked")

        btn_zoom_out = Gtk.Button(
            child=get_icon("zoom-out-symbolic"),
            tooltip_text=_("Zoom Out (Scroll Down)"),
        )
        btn_zoom_out.connect("clicked", lambda _: self._widget.zoom_out())
        btn_zoom_fit = Gtk.Button(
            child=get_icon("zoom-fit-best-symbolic"),
            tooltip_text=_("Fit to Window"),
        )
        btn_zoom_fit.connect("clicked", lambda _: self._widget.zoom_fit())
        btn_zoom_in = Gtk.Button(
            child=get_icon("zoom-in-symbolic"),
            tooltip_text=_("Zoom In (Scroll Up)"),
        )
        btn_zoom_in.connect("clicked", lambda _: self._widget.zoom_in())
        zoom_box.append(btn_zoom_out)
        zoom_box.append(btn_zoom_fit)
        zoom_box.append(btn_zoom_in)
        header_bar.pack_start(zoom_box)

        self._widget = CameraAlignment(controller)
        self._widget.applied.connect(lambda *_: self.close())
        self._widget.set_margin_start(12)
        self._widget.set_margin_end(12)
        self._widget.set_margin_top(12)
        content.append(self._widget)

        # The alignment surface owns its Reset/Clear/Apply buttons but
        # does not place them itself; the dialog hosts them in a
        # bottom button row.
        btn_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            halign=Gtk.Align.END,
            margin_top=12,
            margin_bottom=12,
            margin_start=12,
            margin_end=12,
        )
        for btn in self._widget.footer_buttons():
            btn_box.append(btn)
        content.append(btn_box)

    def do_close_request(self, *args) -> bool:
        logger.debug(
            f"CameraAlignmentDialog closing for {self._widget.camera.name}"
        )
        self._widget.stop()
        return False


__all__ = ["CameraAlignmentDialog"]
