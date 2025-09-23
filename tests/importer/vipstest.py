# flake8: noqa: E402
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gio", "2.0")
from gi.repository import Gtk, Gio
import cairo
import pyvips
import numpy
from pathlib import Path
import sys
import traceback

# Assume the test base dir is the same dir as the current file
image_base_dir = Path(__file__).parent

image_paths = [
    image_base_dir / "png" / "8-bit-with-1-bit-color.png",
    image_base_dir / "png" / "color.png",
    image_base_dir / "png" / "grayscale.png",
]


def draw_checkerboard(cr, width, height):
    """Uses the provided cairo context to draw a checkerboard."""
    TILE_SIZE = 10
    cr.save()
    for y in range(0, int(height), TILE_SIZE):
        for x in range(0, int(width), TILE_SIZE):
            if (x // TILE_SIZE + y // TILE_SIZE) % 2 == 0:
                cr.set_source_rgb(0.8, 0.8, 0.8)  # Light gray
            else:
                cr.set_source_rgb(1.0, 1.0, 1.0)  # White
            cr.rectangle(x, y, TILE_SIZE, TILE_SIZE)
            cr.fill()
    cr.restore()


class VipsCairoWidget(Gtk.DrawingArea):
    """A custom GTK widget to display a pyvips image using Cairo."""

    def __init__(self, vips_image: pyvips.Image):
        super().__init__()

        # 1. Prepare image into a standard RGBA uchar format
        rgba_image = self.prepare_image_for_rgba(vips_image)
        # 2. Premultiply alpha. This promotes the image format to float.
        premultiplied_float_image = rgba_image.premultiply()

        # 3.
        # Cast the image back to uchar (8-bit) after premultiplication.
        # This gives us a buffer of the correct size for numpy.
        premultiplied_uchar_image = premultiplied_float_image.cast("uchar")

        # 4. Get the raw RGBA pixel data from the correctly formatted image
        rgba_memory = premultiplied_uchar_image.write_to_memory()

        # 5. Use numpy for robust channel shuffling from RGBA to BGRA
        rgba_array = numpy.frombuffer(rgba_memory, dtype=numpy.uint8).reshape(
            [
                premultiplied_uchar_image.height,
                premultiplied_uchar_image.width,
                4,
            ]
        )
        bgra_array = numpy.ascontiguousarray(rgba_array[..., [2, 1, 0, 3]])

        # 6. Create the Cairo surface from the correctly ordered BGRA
        # numpy array
        self.cairo_surface = cairo.ImageSurface.create_for_data(
            memoryview(bgra_array),
            cairo.FORMAT_ARGB32,
            premultiplied_uchar_image.width,
            premultiplied_uchar_image.height,
        )

        self.image_width = premultiplied_uchar_image.width
        self.image_height = premultiplied_uchar_image.height
        self.set_content_width(self.image_width)
        self.set_content_height(self.image_height)
        self.set_draw_func(self.on_draw)

    def on_draw(self, drawing_area, cr, width, height):
        draw_checkerboard(cr, width, height)
        cr.set_operator(cairo.OPERATOR_OVER)
        x_offset = (width - self.image_width) / 2
        y_offset = (height - self.image_height) / 2
        cr.set_source_surface(self.cairo_surface, x_offset, y_offset)
        cr.paint()

    def prepare_image_for_rgba(self, vips_image: pyvips.Image) -> pyvips.Image:
        """Ensures a pyvips image is in 8-bit RGBA sRGB format."""
        if vips_image.interpretation != "srgb":
            vips_image = vips_image.colourspace("srgb")
        if not vips_image.hasalpha():
            vips_image = vips_image.addalpha()
        if vips_image.format != "uchar":
            vips_image = vips_image.cast("uchar")
        if vips_image.bands != 4:
            raise ValueError(
                f"Image must have 4 bands (RGBA), but has {vips_image.bands}"
            )
        return vips_image


class ImageDisplayApp(Gtk.Application):
    def __init__(self):
        super().__init__(
            application_id="com.example.PyVipsCairoGtk4",
            flags=Gio.ApplicationFlags.DEFAULT_FLAGS,
        )
        self.connect("activate", self.on_activate)

    def on_activate(self, app):
        window = Gtk.ApplicationWindow(
            application=app, title="PyVips Cairo Display"
        )
        scrolled_window = Gtk.ScrolledWindow()
        window.set_child(scrolled_window)

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        hbox.set_homogeneous(False)
        hbox.set_margin_start(20)
        hbox.set_margin_end(20)
        hbox.set_margin_top(20)
        hbox.set_margin_bottom(20)
        scrolled_window.set_child(hbox)

        for filepath in image_paths:
            vips_image = pyvips.Image.new_from_file(filepath)
            print(f"Loaded {filepath.name}")

            vips_widget = VipsCairoWidget(vips_image)

            label = Gtk.Label(label=filepath.name)
            vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
            vbox.append(label)
            vbox.append(vips_widget)
            hbox.append(vbox)

        window.set_default_size(900, 500)
        window.present()


if __name__ == "__main__":
    app = ImageDisplayApp()
    app.run(sys.argv)