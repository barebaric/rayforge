import logging
from gi.repository import Gtk, Adw, GdkPixbuf
from typing import Optional, Literal
from gettext import gettext as _
from ...camera.models.camera import Camera
from ...camera.controller import CameraController
from ...context import get_context
from ..shared.gtk import apply_css

logger = logging.getLogger(__name__)


class CameraSelectionDialog(Adw.MessageDialog):
    def __init__(
        self,
        parent,
        mode: Literal["available", "configured"] = "available",
        **kwargs,
    ):
        self._mode = mode
        body = (
            _("Please select an available camera device")
            if mode == "available"
            else _("Please select a configured camera")
        )
        super().__init__(
            transient_for=parent,
            modal=True,
            heading=_("Select Camera"),
            body=body,
            close_response="cancel",
            **kwargs,
        )
        self.set_size_request(450, 350)
        self.selected_device_id: Optional[str] = None

        apply_css("""
            .rounded-image {
                border-radius: 8px;
            }
            .nav-button {
                padding: 12px;
            }
        """)

        self.carousel = Adw.Carousel()
        self.carousel.set_vexpand(True)
        self.carousel.set_hexpand(True)
        self.carousel.set_allow_scroll_wheel(True)
        self.carousel.set_allow_long_swipes(True)
        self.carousel.set_interactive(True)

        self.prev_button = Gtk.Button(icon_name="go-previous-symbolic")
        self.prev_button.add_css_class("nav-button")
        self.prev_button.add_css_class("flat")
        self.prev_button.set_sensitive(False)
        self.prev_button.set_valign(Gtk.Align.CENTER)
        self.prev_button.connect("clicked", self.on_prev_clicked)

        self.next_button = Gtk.Button(icon_name="go-next-symbolic")
        self.next_button.add_css_class("nav-button")
        self.next_button.add_css_class("flat")
        self.next_button.set_sensitive(False)
        self.next_button.set_valign(Gtk.Align.CENTER)
        self.next_button.connect("clicked", self.on_next_clicked)

        carousel_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        carousel_box.append(self.prev_button)
        carousel_box.append(self.carousel)
        carousel_box.append(self.next_button)

        self.indicator = Adw.CarouselIndicatorDots()
        self.indicator.set_carousel(self.carousel)
        self.indicator.set_margin_bottom(6)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content_box.append(carousel_box)
        content_box.append(self.indicator)
        content_box.set_margin_start(12)
        content_box.set_margin_end(12)
        content_box.set_margin_top(12)
        content_box.set_margin_bottom(6)

        self.set_extra_child(content_box)

        self.add_response("cancel", _("Cancel"))
        self.set_response_enabled("cancel", True)
        self.set_default_response("cancel")

        self.available_devices: list[str] = []
        self._controllers: list[CameraController] = []
        if mode == "available":
            self.list_available_cameras()
        else:
            self.list_configured_cameras()

        self.carousel.connect("page-changed", self.on_page_changed)

        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self.on_key_pressed)
        self.add_controller(key_controller)

    def list_configured_cameras(self):
        camera_mgr = get_context().camera_mgr
        controllers = camera_mgr.controllers

        if not controllers:
            label = Gtk.Label(label=_("No cameras configured."))
            self.carousel.append(label)
            return

        for ctrl in controllers:
            device_id = ctrl.config.device_id
            self.available_devices.append(device_id)
            self._controllers.append(ctrl)
            self._add_camera_page(ctrl, device_id, ctrl.config.name)

        if self.available_devices:
            first_child = self.carousel.get_nth_page(0)
            self.carousel.scroll_to(first_child, True)
            self.selected_device_id = self.available_devices[0]
            self._update_nav_buttons()

    def _add_camera_page(
        self, controller: CameraController, device_id: str, name: str
    ):
        pixbuf = controller.pixbuf

        if not pixbuf:
            label = Gtk.Label(
                label=_(
                    "Failed to load image for Device ID: {device_id}"
                ).format(device_id=device_id)
            )
            self.carousel.append(label)
            return

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

        label = Gtk.Label(
            label=_("{name} (Device ID: {device_id})").format(
                name=name, device_id=device_id
            )
        )
        label.set_halign(Gtk.Align.CENTER)
        label.set_valign(Gtk.Align.CENTER)
        label.set_margin_bottom(12)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        box.append(image_widget)
        box.append(label)
        box.set_halign(Gtk.Align.CENTER)
        box.set_valign(Gtk.Align.CENTER)

        gesture = Gtk.GestureClick.new()
        gesture.connect("released", self.on_carousel_item_clicked, device_id)
        box.add_controller(gesture)

        motion_controller = Gtk.EventControllerMotion.new()
        motion_controller.connect(
            "enter", self.on_carousel_item_hover_enter, box
        )
        motion_controller.connect(
            "leave", self.on_carousel_item_hover_leave, box
        )
        box.add_controller(motion_controller)

        self.carousel.append(box)

    def list_available_cameras(self):
        self.available_devices = CameraController.list_available_devices()
        if not self.available_devices:
            label = Gtk.Label(label=_("No cameras found."))
            self.carousel.append(label)
            return

        for device_id in self.available_devices:
            temp_config = Camera(
                name=_("Camera {device_id}").format(device_id=device_id),
                device_id=device_id,
            )
            temp_controller = CameraController(temp_config)
            temp_controller.capture_image()
            self._add_camera_page(temp_controller, device_id, temp_config.name)

        if self.available_devices:
            first_child = self.carousel.get_nth_page(0)
            self.carousel.scroll_to(first_child, True)
            self.selected_device_id = self.available_devices[0]
            self._update_nav_buttons()

    def on_page_changed(self, carousel, page_number):
        if 0 <= page_number < len(self.available_devices):
            self.selected_device_id = self.available_devices[page_number]
        else:
            self.selected_device_id = None
        self._update_nav_buttons()

    def on_carousel_item_clicked(self, gesture, n_press, x, y, device_id):
        self.selected_device_id = device_id
        self.response("select")
        self.close()

    def on_carousel_item_hover_enter(self, motion_controller, x, y, box):
        # Add a "card" style class for a subtle shadow effect
        box.add_css_class("card")

    def on_carousel_item_hover_leave(self, motion_controller, box):
        box.remove_css_class("card")

    def on_prev_clicked(self, button):
        current = self.carousel.get_position()
        if current > 0:
            page = self.carousel.get_nth_page(int(current) - 1)
            self.carousel.scroll_to(page, True)

    def on_next_clicked(self, button):
        n_pages = self.carousel.get_n_pages()
        current = self.carousel.get_position()
        if current < n_pages - 1:
            page = self.carousel.get_nth_page(int(current) + 1)
            self.carousel.scroll_to(page, True)

    def on_key_pressed(self, controller, keyval, keycode, state):
        if keyval == 65361:
            self.on_prev_clicked(None)
            return True
        elif keyval == 65363:
            self.on_next_clicked(None)
            return True
        return False

    def _update_nav_buttons(self):
        n_pages = self.carousel.get_n_pages()
        current = int(self.carousel.get_position())
        self.prev_button.set_sensitive(current > 0)
        self.next_button.set_sensitive(current < n_pages - 1)
