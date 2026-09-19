import logging
import threading
import time
from gettext import gettext as _
from typing import Iterable, Literal, Optional

from gi.repository import Adw, GdkPixbuf, Gtk

from ...camera.controller import CameraController
from ...camera.models.camera import Camera
from ...camera.v4l import display_name
from ...context import get_context
from ..shared.gtk import apply_css
from ...shared.util.glib import idle_add

logger = logging.getLogger(__name__)


class CameraSelectionDialog(Adw.MessageDialog):
    # Result of the (expensive) device scan, cached briefly so reopening
    # the picker is instant instead of re-spawning the probe subprocess.
    _SCAN_CACHE_TTL = 5.0
    _scan_cache: tuple[float, list[str]] | None = None
    _scan_cache_lock = threading.Lock()

    @classmethod
    def _list_devices_cached(cls) -> list[str]:
        with cls._scan_cache_lock:
            now = time.monotonic()
            if (
                cls._scan_cache is not None
                and now - cls._scan_cache[0] < cls._SCAN_CACHE_TTL
            ):
                return list(cls._scan_cache[1])
            devices = CameraController.list_available_devices()
            cls._scan_cache = (now, devices)
            return list(devices)

    def __init__(
        self,
        parent,
        mode: Literal["available", "configured"] = "available",
        exclude_device_ids: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        self._mode = mode
        self._exclude_device_ids = set(exclude_device_ids or [])
        self._closed = False
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
        self.selected_device_id: str | None = None

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

        # The scan runs on a worker thread; show a spinner until it is done
        # so the dialog never blocks the GTK main thread (a synchronous
        # scan + per-device capture froze the window long enough for GTK to
        # flag the whole app as Not Responding).
        self._content_stack = Gtk.Stack()
        self._content_stack.set_vexpand(True)
        self._content_stack.add_named(self._build_spinner_page(), "loading")
        self._content_stack.add_named(content_box, "pages")
        self.set_extra_child(self._content_stack)

        self.add_response("cancel", _("Cancel"))
        self.set_response_enabled("cancel", True)
        self.set_default_response("cancel")
        self.connect("response", self._on_response_close)

        self.available_devices: list[str] = []
        self._controllers: list[CameraController] = []
        if mode == "available":
            self._content_stack.set_visible_child_name("loading")
            self._start_async_scan()
        else:
            self._content_stack.set_visible_child_name("pages")
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
            self._add_camera_page(ctrl.pixbuf, device_id, ctrl.config.name)

        if self.available_devices:
            first_child = self.carousel.get_nth_page(0)
            self.carousel.scroll_to(first_child, True)
            self.selected_device_id = self.available_devices[0]
            self._update_nav_buttons()

    def _add_camera_page(
        self, pixbuf: Optional[GdkPixbuf.Pixbuf], device_id: str, name: str
    ):
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

        label_text = name
        dev_name = display_name(device_id)
        if dev_name == device_id:
            label_text = _("Camera {device_id}").format(device_id=device_id)

        label = Gtk.Label(label=label_text)
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

    @staticmethod
    def _get_display_name(device_id: str) -> str:
        return display_name(device_id)

    def _build_spinner_page(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_vexpand(True)
        spinner = Gtk.Spinner()
        spinner.set_size_request(48, 48)
        spinner.set_halign(Gtk.Align.CENTER)
        spinner.set_valign(Gtk.Align.END)
        spinner.spin()
        label = Gtk.Label(label=_("Searching for cameras…"))
        label.set_halign(Gtk.Align.CENTER)
        label.set_valign(Gtk.Align.START)
        box.append(spinner)
        box.append(label)
        return box

    def _on_response_close(self, dialog, response_id):
        # Tells the worker thread and the idle callback that the dialog is
        # gone. Touching its widgets after destroy would warn or crash.
        self._closed = True

    def _start_async_scan(self):
        threading.Thread(
            target=self._scan_worker,
            name="CameraPickerScan",
            daemon=True,
        ).start()

    def _scan_worker(self):
        """Enumerate devices and grab previews off the main thread."""
        try:
            devices = self._list_devices_cached()
        except Exception:
            logger.exception("Camera device scan failed")
            devices = []

        camera_mgr = None
        try:
            camera_mgr = get_context().camera_mgr
        except Exception:
            logger.debug("Camera manager not available for live previews")

        pages: list[tuple[str, str, Optional[GdkPixbuf.Pixbuf]]] = []
        for device_id in devices:
            if device_id in self._exclude_device_ids:
                continue
            if self._closed:
                return
            name = display_name(device_id)
            pixbuf = self._grab_preview(camera_mgr, device_id)
            pages.append((device_id, name, pixbuf))

        idle_add(self._populate_from_scan, pages)

    def _grab_preview(self, camera_mgr, device_id: str):
        """Return a one-shot preview pixbuf for the given device.

        Prefer the last frame of an already-running controller: opening a
        device that a live capture thread holds is exactly the double-open
        that crashes OpenCV on Windows. Only for devices without a live
        controller fall back to a temporary synchronous capture (this runs
        on the worker thread, not the UI).
        """
        controller = (
            camera_mgr.get_controller(device_id) if camera_mgr else None
        )
        if controller is not None:
            return controller.pixbuf

        temp_config = Camera(name=device_id, device_id=device_id)
        temp_controller = CameraController(temp_config)
        try:
            temp_controller.capture_image()
            return temp_controller.pixbuf
        finally:
            temp_controller.dispose()

    def _populate_from_scan(
        self, pages: list[tuple[str, str, Optional[GdkPixbuf.Pixbuf]]]
    ):
        if self._closed:
            return False

        for device_id, name, pixbuf in pages:
            self.available_devices.append(device_id)
            self._add_camera_page(pixbuf, device_id, name)

        if not self.available_devices:
            self.carousel.append(Gtk.Label(label=_("No cameras found.")))
        else:
            first_child = self.carousel.get_nth_page(0)
            self.carousel.scroll_to(first_child, True)
            self.selected_device_id = self.available_devices[0]
            self._update_nav_buttons()

        self._content_stack.set_visible_child_name("pages")
        return False

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
