import logging
from gi.repository import Gtk
from blinker import Signal
from .sketchcanvas import SketchCanvas
from rayforge.core.sketcher import Sketch

logger = logging.getLogger(__name__)


class SketchStudio(Gtk.Box):
    """
    The top-level container for the sketching environment.
    Manages the layout of the canvas and side panels and orchestrates the
    save/cancel lifecycle.
    """

    def __init__(self, parent_window: Gtk.Window, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)
        self.parent_window = parent_window

        # Signals
        self.finished = Signal()
        self.cancelled = Signal()

        self._build_ui()

    def _build_ui(self):
        # 1. Session Bar (Header)
        # We use a Gtk.Box styled as a toolbar to hold the session controls.
        self.session_bar = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6
        )
        self.session_bar.add_css_class("toolbar")
        self.session_bar.set_margin_top(6)
        self.session_bar.set_margin_bottom(6)
        self.session_bar.set_margin_start(12)
        self.session_bar.set_margin_end(12)
        self.append(self.session_bar)

        # Left: Cancel
        self.btn_cancel = Gtk.Button(icon_name="window-close-symbolic")
        self.btn_cancel.set_tooltip_text(_("Cancel Sketch"))
        self.btn_cancel.connect("clicked", self._on_cancel_clicked)
        self.session_bar.append(self.btn_cancel)

        # Center: Title (using spacers to center it roughly)
        spacer_l = Gtk.Box()
        spacer_l.set_hexpand(True)
        self.session_bar.append(spacer_l)

        self.lbl_title = Gtk.Label(label=_("Sketch Studio"))
        self.lbl_title.add_css_class("title-3")
        self.session_bar.append(self.lbl_title)

        spacer_r = Gtk.Box()
        spacer_r.set_hexpand(True)
        self.session_bar.append(spacer_r)

        # Right: Finish
        self.btn_finish = Gtk.Button(label=_("Finish"))
        self.btn_finish.add_css_class("suggested-action")
        self.btn_finish.connect("clicked", self._on_finish_clicked)
        self.session_bar.append(self.btn_finish)

        # 2. Main Content
        # Sidebar is currently disabled/hidden. Canvas takes full space.
        self.canvas = SketchCanvas(
            parent_window=self.parent_window, single_mode=True
        )
        self.canvas.set_vexpand(True)
        self.append(self.canvas)

    def set_sketch(self, sketch: Sketch):
        """Loads a sketch model into the studio."""
        logger.debug("Loading sketch into studio.")
        self.canvas.set_sketch(sketch)
        # Ensure the element bounds are updated for the new content
        self.canvas.sketch_element.update_bounds_from_sketch()
        # Reset view to center content
        self.canvas.reset_view()

    def _on_cancel_clicked(self, btn):
        logger.info("SketchStudio: Cancel clicked")
        self.cancelled.send(self)

    def _on_finish_clicked(self, btn):
        logger.info("SketchStudio: Finish clicked")
        # Retrieve the modified sketch from the canvas element
        if self.canvas.sketch_element:
            sketch = self.canvas.sketch_element.sketch
            self.finished.send(self, sketch=sketch)
