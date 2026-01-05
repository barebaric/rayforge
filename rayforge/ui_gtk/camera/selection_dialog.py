from gi.repository import Gtk, Adw, GdkPixbuf
from typing import Optional
from ..shared.gtk import apply_css
from ...camera.models.camera import Camera
from ...camera.controller import CameraController


class CameraSelectionDialog(Adw.MessageDialog):
    def __init__(self, parent, **kwargs):
        super().__init__(
            transient_for=parent,
            modal=True,
            heading=_("Select Camera"),
            body=_("Please select an available camera device"),
            close_response="cancel",
            **kwargs,
        )
        self.set_size_request(400, 300)  # Increased size
        self.selected_device_id: Optional[str] = None

        # Add CSS for hover effect
        apply_css("""
            .rounded-image {
                border-radius: 8px;
            }
        """)

        self.carousel = Adw.Carousel()
        self.carousel.set_vexpand(True)
        self.carousel.set_hexpand(True)
        self.carousel.set_allow_scroll_wheel(True)
        self.carousel.set_allow_long_swipes(True)
        self.carousel.set_interactive(True)
        self.carousel.set_margin_start(12)
        self.carousel.set_margin_end(12)
        self.carousel.set_margin_top(12)
        self.carousel.set_margin_bottom(12)

        self.set_extra_child(self.carousel)

        self.add_response("cancel", _("Cancel"))
        self.set_response_enabled("cancel", True)
        self.set_default_response("cancel")

        self.available_devices: list[str] = []
        self.list_available_cameras()

        self.carousel.connect("page-changed", self.on_page_changed)

    def list_available_cameras(self):
        self.available_devices = CameraController.list_available_devices()
        if not self.available_devices:
            label = Gtk.Label(label=_("No cameras found."))
            self.carousel.append(label)
            return

        for device_id in self.available_devices:
            # Create a temporary config and controller to capture a preview
            temp_config = Camera(
                name=_("Camera {device_id}").format(device_id=device_id),
                device_id=device_id,
            )
            temp_controller = CameraController(temp_config)
            temp_controller.capture_image()
            pixbuf = temp_controller.pixbuf

            if not pixbuf:
                label = Gtk.Label(
                    label=_(
                        "Failed to load image for Device ID: {device_id}"
                    ).format(device_id=device_id)
                )
                self.carousel.append(label)
                continue

            # Scale pixbuf to fit a reasonable size in the dialog.
            # For example, scale to a max height of 250px,
            # maintaining aspect ratio.
            max_height = 250
            width = pixbuf.get_width()
            height = pixbuf.get_height()
            if height > max_height:
                scale_factor = max_height / height
                width = int(width * scale_factor)
                height = max_height
                pixbuf = pixbuf.scale_simple(
                    width, height, GdkPixbuf.InterpType.BILINEAR
                )

            image_widget = Gtk.Picture.new_for_pixbuf(pixbuf)
            image_widget.set_halign(Gtk.Align.CENTER)
            image_widget.set_valign(Gtk.Align.CENTER)
            image_widget.set_size_request(200, 200)
            image_widget.add_css_class("rounded-image")
            image_widget.set_margin_start(10)
            image_widget.set_margin_end(10)
            image_widget.set_margin_top(10)
            image_widget.set_margin_bottom(5)

            # Add a label for the device ID
            label = Gtk.Label(
                label=_("Device ID: {device_id}").format(device_id=device_id)
            )
            label.set_halign(Gtk.Align.CENTER)
            label.set_valign(Gtk.Align.CENTER)
            label.set_margin_bottom(12)

            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            box.append(image_widget)
            box.append(label)
            box.set_halign(Gtk.Align.CENTER)
            box.set_valign(Gtk.Align.CENTER)

            # Add a gesture recognizer for click
            gesture = Gtk.GestureClick.new()
            gesture.connect(
                "released", self.on_carousel_item_clicked, device_id
            )
            box.add_controller(gesture)

            # Add event controller for hover effect
            motion_controller = Gtk.EventControllerMotion.new()
            motion_controller.connect(
                "enter", self.on_carousel_item_hover_enter, box
            )
            motion_controller.connect(
                "leave", self.on_carousel_item_hover_leave, box
            )
            box.add_controller(motion_controller)

            self.carousel.append(box)

        # Set initial selection if cameras are found
        if not self.available_devices:
            return

        # Scroll to to set the active page.
        first_child = self.carousel.get_nth_page(0)
        self.carousel.scroll_to(first_child, True)  # Second arg: animate
        self.selected_device_id = self.available_devices[0]

    def on_page_changed(self, carousel, page_number):
        if 0 <= page_number < len(self.available_devices):
            self.selected_device_id = self.available_devices[page_number]
        else:
            self.selected_device_id = None

    def on_carousel_item_clicked(self, gesture, n_press, x, y, device_id):
        self.selected_device_id = device_id
        self.response("select")
        self.close()

    def on_carousel_item_hover_enter(self, motion_controller, x, y, box):
        # Add a "card" style class for a subtle shadow effect
        box.add_css_class("card")

    def on_carousel_item_hover_leave(self, motion_controller, box):
        # Remove the "card" style class
        box.remove_css_class("card")
