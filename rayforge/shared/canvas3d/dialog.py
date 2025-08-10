from gi.repository import Gtk, Adw  # type: ignore
from .canvas3d import Canvas3D
from ...core.doc import Doc
from ...machine.models.machine import Machine


class Canvas3DDialog(Adw.Window):
    """A dialog window to host the 3D canvas."""

    def __init__(self, doc: Doc, machine: Machine, **kwargs):
        super().__init__(**kwargs)

        width, height = self._get_initial_size(machine)
        self.set_default_size(width, height)

        # Main content box
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.set_content(box)

        # Title bar
        header_bar = Adw.HeaderBar()
        header_bar.set_title_widget(
            Adw.WindowTitle(
                title=_(f"{machine.name} – Path Preview"),
                subtitle="",
            )
        )
        box.append(header_bar)

        # Instructions label, similar to the example runner in canvas3d.py
        label_text = (
            "MMB Drag=Arcball Orbit | Shift+MMB Drag=Pan | "
            "Scroll=Zoom | P=Toggle Projection"
        )
        label = Gtk.Label(label=label_text)
        box.append(label)

        # The canvas itself
        canvas = Canvas3D(doc, machine, vexpand=True)
        box.append(canvas)

    def _get_initial_size(self, machine: Machine) -> tuple[int, int]:
        """
        Calculate the initial window size to match the machine's aspect ratio,
        fitting within a maximum bounding box.
        """
        MAX_WIDTH, MAX_HEIGHT = 1024, 768
        DEFAULT_SIZE = MAX_WIDTH, MAX_HEIGHT

        machine_w, machine_h = machine.dimensions
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
