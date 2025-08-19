from gi.repository import Gtk, Adw, Gdk
from typing import Tuple
from .canvas3d import Canvas3D
from ...core.doc import Doc
from ...core.ops import Ops


class Canvas3DDialog(Adw.Window):
    """A dialog window to host the 3D canvas."""

    def __init__(
        self,
        doc: Doc,
        title: str,
        size: Tuple[float, float],
        y_down: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)

        width, height = self._get_initial_size(size)
        self.set_default_size(width, height)

        # Main content box
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.set_content(box)

        # Title bar
        header_bar = Adw.HeaderBar()
        header_bar.set_title_widget(
            Adw.WindowTitle(
                title=title,
                subtitle="",
            )
        )
        box.append(header_bar)

        # Instructions label with updated controls
        label_text = (
            "LMB Drag=Z-Rotate | MMB Drag=Orbit | Shift+MMB Drag=Pan | "
            "Scroll=Zoom | P=Projection | 1=Top | 7=Iso"
        )
        label = Gtk.Label(label=label_text)
        box.append(label)

        # The canvas itself
        width_mm, depth_mm = size
        self.canvas = Canvas3D(
            doc,
            width_mm=width_mm,
            depth_mm=depth_mm,
            y_down=y_down,
            vexpand=True,
        )
        box.append(self.canvas)

        # Add key controller for window-level shortcuts
        key_controller = Gtk.EventControllerKey.new()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Handles key press events for the dialog window."""
        # Check for Ctrl+W
        is_w = keyval in (Gdk.KEY_w, Gdk.KEY_W)
        is_ctrl = bool(state & Gdk.ModifierType.CONTROL_MASK)
        if is_ctrl and is_w:
            self.close()
            return True  # Event handled

        # Check for Escape key
        if keyval == Gdk.KEY_Escape:
            self.close()
            return True  # Event handled

        return False  # Event not handled, propagate further

    def set_ops(self, ops: Ops):
        """Passes the generated ops to the underlying canvas."""
        self.canvas.set_ops(ops)

    def _get_initial_size(
        self, machine_dims: Tuple[float, float]
    ) -> tuple[int, int]:
        """
        Calculate the initial window size to match the machine's aspect ratio,
        fitting within a maximum bounding box. The bounding box is 90% of the
        parent window's size, or a default size if no parent is available.
        """
        MAX_WIDTH, MAX_HEIGHT = 1024, 768  # Default max size

        parent = self.get_transient_for()
        if parent:
            parent_w = parent.get_width()
            parent_h = parent.get_height()
            # Ensure we have valid dimensions before overriding defaults
            if parent_w > 1 and parent_h > 1:
                # Use max(1, ...) to avoid sizes of 0, which would cause
                # a division-by-zero error.
                MAX_WIDTH = max(1, int(parent_w * 0.9))
                MAX_HEIGHT = max(1, int(parent_h * 0.9))

        DEFAULT_SIZE = MAX_WIDTH, MAX_HEIGHT

        machine_w, machine_h = machine_dims
        if not (machine_w and machine_w > 0 and machine_h and machine_h > 0):
            return DEFAULT_SIZE

        # We have valid dimensions, so we can calculate the aspect-correct size
        machine_aspect = float(machine_w) / float(machine_h)
        max_aspect = float(MAX_WIDTH) / float(MAX_HEIGHT)

        if machine_aspect > max_aspect:
            # Machine is wider than the max bounding box, so fit to width
            width = MAX_WIDTH
            height = int(width / machine_aspect)
        else:
            # Machine is taller or same aspect, so fit to height
            height = MAX_HEIGHT
            width = int(height * machine_aspect)

        return width, height
