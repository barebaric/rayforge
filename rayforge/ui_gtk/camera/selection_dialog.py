import logging
import threading
from gettext import gettext as _

from gi.repository import Adw, Gdk, Gtk

from ...camera.controller import CameraController
from ...camera.models.camera import Camera, CameraSourceType
from ...camera.source import list_local_device_ids, validate_source_uri
from ...camera.v4l import display_name
from ...shared.util.glib import idle_add
from .capture_surface import numpy_to_pixbuf

logger = logging.getLogger(__name__)

_SCANNING_PLACEHOLDER = _("Scanning for devices\u2026")


class CameraSelectionDialog(Adw.MessageDialog):
    def __init__(
        self,
        parent,
        active_controllers: list[CameraController] | None = None,
        mode: str = "new",
        **kwargs,
    ):
        super().__init__(
            transient_for=parent,
            modal=True,
            heading=_("Add Camera"),
            body=_("Choose a camera source."),
            close_response="cancel",
            **kwargs,
        )
        self.set_size_request(720, 560)
        self.mode = mode
        if mode == "configured":
            self.set_heading(_("Select Camera"))
        self.camera_payload: dict | None = None
        self.selected_device_id: str | None = None
        # Controllers already active elsewhere in the app (e.g. a camera
        # already configured on the machine). Their frames are reused for
        # the preview instead of opening the device a second time, which
        # can crash the underlying V4L2/DirectShow driver.
        self._active_controllers: list[CameraController] = (
            list(active_controllers) if active_controllers else []
        )
        # Guards against async scan/preview callbacks touching widgets
        # after the dialog has been closed/destroyed.
        self._closed = False
        # Discards stale preview results from a superseded selection.
        self._preview_generation = 0
        self._available_devices: list[str] = []
        self._preview_pixbuf = None
        self._build_ui()
        self.add_response("cancel", _("Cancel"))
        self.add_response(
            "select", _("Select") if mode == "configured" else _("Add")
        )
        self.set_response_enabled("select", False)
        self.type_row.connect("notify::selected", self._on_type_changed)
        self.device_row.connect("notify::selected", self._on_form_changed)
        self.uri_entry.connect("changed", self._on_form_changed)
        self.connect("response", self._on_response)
        self.connect("destroy", self._on_destroy)
        # The device dropdown opens instantly with a placeholder; probing
        # hardware happens on a worker thread so showing this dialog never
        # blocks the UI, however many (or slow) devices are attached.
        if self.mode == "configured":
            self._load_configured_cameras()
        else:
            self._start_device_scan()

    def _build_ui(self) -> None:
        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_top=12,
            margin_bottom=12,
            margin_start=12,
            margin_end=12,
        )
        self.set_extra_child(box)
        box.set_size_request(-1, 500)

        source_group = Adw.PreferencesGroup()
        box.append(source_group)

        self.type_values = [
            None,
            CameraSourceType.LOCAL_DEVICE,
            CameraSourceType.HTTP_SNAPSHOT,
            CameraSourceType.HTTP_STREAM,
            CameraSourceType.RTSP,
        ]
        self.type_row = Adw.ComboRow(
            title=_("Source Type"),
            model=Gtk.StringList.new(
                [
                    _("Choose source type"),
                    _("Local camera"),
                    _("HTTP snapshot URL"),
                    _("HTTP stream URL"),
                    _("RTSP stream"),
                ]
            ),
        )
        source_group.add(self.type_row)

        self.device_row = Adw.ComboRow(
            title=_("Device"),
            subtitle=_("Choose an attached local camera"),
            # Populated once the background scan in _start_device_scan()
            # completes, so opening this dialog never blocks on hardware
            # probing.
            model=Gtk.StringList.new([_SCANNING_PLACEHOLDER]),
        )
        self.device_row.set_sensitive(False)
        source_group.add(self.device_row)

        preview_group = Adw.PreferencesGroup(
            title=_("Preview"),
            description=_("Capture one frame from the selected local camera"),
        )
        preview_frame = Gtk.Frame(hexpand=True)
        preview_frame.add_css_class("card")
        self.preview_image = Gtk.Picture(
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
        )
        self.preview_image.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.preview_image.set_size_request(360, 220)
        preview_frame.set_child(self.preview_image)
        preview_group.add(preview_frame)
        self.preview_group = preview_group
        box.append(preview_group)

        uri_row = Adw.ActionRow(
            title=_("URL"),
            subtitle=_("Enter the snapshot or stream URL"),
        )
        self.uri_entry = Gtk.Entry()
        self.uri_entry.set_valign(Gtk.Align.CENTER)
        self.uri_entry.set_width_chars(40)
        self.uri_entry.set_hexpand(True)
        uri_row.add_suffix(self.uri_entry)
        self.uri_row = uri_row
        source_group.add(uri_row)
        self._update_visibility()
        self.type_row.set_selected(1)
        self.device_row.set_visible(True)
        self.preview_group.set_visible(True)

    def _load_configured_cameras(self) -> None:
        from ...context import get_context

        controllers = get_context().camera_mgr.controllers
        self._active_controllers = list(controllers)
        self._available_devices = [
            controller.config.id for controller in controllers
        ]
        self.type_row.set_visible(False)
        self.device_row.set_model(
            Gtk.StringList.new(
                [controller.config.name for controller in controllers]
            )
        )
        self.device_row.set_sensitive(bool(controllers))
        self.device_row.set_selected(0 if controllers else -1)
        self.preview_group.set_visible(False)
        self.uri_row.set_visible(False)
        self._on_form_changed()

    def _selected_source_type(self):
        idx = self.type_row.get_selected()
        if 0 <= idx < len(self.type_values):
            return self.type_values[idx]
        return None

    def _update_visibility(self) -> None:
        source_type = self._selected_source_type()
        is_local = source_type is CameraSourceType.LOCAL_DEVICE
        self.device_row.set_visible(is_local)
        self.preview_group.set_visible(is_local)
        self.uri_row.set_visible(
            source_type
            in {
                CameraSourceType.HTTP_SNAPSHOT,
                CameraSourceType.HTTP_STREAM,
                CameraSourceType.RTSP,
            }
        )
        if not is_local:
            self.preview_image.set_paintable(None)
            self._preview_pixbuf = None

    def _build_payload(self) -> dict | None:
        source_type = self._selected_source_type()
        if source_type is None:
            return None
        if self.mode == "configured":
            idx = self.device_row.get_selected()
            if idx < 0 or idx >= len(self._available_devices):
                return None
            self.selected_device_id = self._available_devices[idx]
            return {"camera_id": self.selected_device_id}
        if source_type is CameraSourceType.LOCAL_DEVICE:
            idx = self.device_row.get_selected()
            if idx <= 0:
                return None
            device_id = self._available_devices[idx - 1]
            return {
                "name": display_name(device_id),
                "source_type": source_type.value,
                "source_config": {"device_id": device_id},
            }
        uri = self.uri_entry.get_text().strip()
        if not uri or validate_source_uri(source_type, uri):
            return None
        return {
            "name": uri,
            "source_type": source_type.value,
            "source_config": {"uri": uri},
        }

    def _on_type_changed(self, *args) -> None:
        self._update_visibility()
        self._on_form_changed()

    def _on_form_changed(self, *args) -> None:
        if self.mode == "configured":
            payload = self._build_payload()
            self.camera_payload = payload
            self.set_response_enabled("select", payload is not None)
            return
        source_type = self._selected_source_type()
        if source_type is not None and source_type is not (
            CameraSourceType.LOCAL_DEVICE
        ):
            error = validate_source_uri(source_type, self.uri_entry.get_text())
            self.uri_entry.set_css_classes(["error"] if error else [])
            self.uri_entry.set_tooltip_text(error)
        payload = self._build_payload()
        self.camera_payload = payload
        self.set_response_enabled("select", payload is not None)
        self._update_preview()

    def _update_preview(self) -> None:
        source_type = self._selected_source_type()
        if source_type is not CameraSourceType.LOCAL_DEVICE:
            return

        idx = self.device_row.get_selected()
        if idx <= 0 or idx - 1 >= len(self._available_devices):
            self.preview_image.set_paintable(None)
            self._preview_pixbuf = None
            return
        device_id = self._available_devices[idx - 1]

        # Any in-flight preview capture for a previously selected device
        # becomes stale as soon as the selection changes again; bump the
        # generation so its result is discarded when it eventually
        # arrives instead of clobbering the newer selection's preview.
        self._preview_generation += 1
        generation = self._preview_generation

        active_controller = self._find_active_local_controller(device_id)
        if active_controller is not None:
            # Reuse the already-open stream's latest frame instead of
            # opening the same device a second time, which can crash the
            # underlying V4L2/DirectShow driver.
            self._show_preview_frame(active_controller.raw_image_data)
            return

        self.preview_image.set_paintable(None)
        self._preview_pixbuf = None
        self._capture_preview_async(device_id, generation)

    def _find_active_local_controller(
        self, device_id: str
    ) -> CameraController | None:
        for controller in self._active_controllers:
            config = controller.config
            if (
                config.source_type is CameraSourceType.LOCAL_DEVICE
                and config.device_id == device_id
                and controller.has_active_source
            ):
                return controller
        return None

    def _capture_preview_async(self, device_id: str, generation: int) -> None:
        """Captures one preview frame off the UI thread.

        Opening a device (especially over USB/V4L2) can block for a
        noticeable amount of time; doing this synchronously would freeze
        the whole dialog (and app) while the user is just browsing
        devices in the dropdown.
        """

        def worker() -> None:
            image = self._capture_preview_frame(device_id)
            idle_add(self._on_preview_captured, generation, image)

        threading.Thread(
            target=worker, name="CameraPreviewCapture", daemon=True
        ).start()

    def _capture_preview_frame(self, device_id: str):
        camera = Camera(
            name=display_name(device_id),
            source_type=CameraSourceType.LOCAL_DEVICE,
            source_config={"device_id": device_id},
        )
        controller = CameraController(camera)
        try:
            controller.capture_image(apply_settings=False)
            return controller.raw_image_data
        finally:
            # This is a throwaway, one-shot controller: dispose it so it
            # is fully detached from its (equally throwaway) Camera model
            # instead of leaking a dangling signal connection.
            controller.dispose()

    def _on_preview_captured(self, generation: int, image) -> None:
        if self._closed or generation != self._preview_generation:
            # The dialog closed, or the user picked a different device
            # while this capture was in flight; drop the stale result.
            return
        self._show_preview_frame(image)

    def _show_preview_frame(self, image) -> None:
        if image is None:
            self.preview_image.set_paintable(None)
            self._preview_pixbuf = None
            return
        pixbuf = numpy_to_pixbuf(image)
        self._preview_pixbuf = pixbuf
        paintable = (
            Gdk.Texture.new_for_pixbuf(pixbuf) if pixbuf is not None else None
        )
        self.preview_image.set_paintable(paintable)

    def _start_device_scan(self) -> None:
        """Scans for local camera devices on a worker thread.

        Probing hardware (potentially several devices, each with several
        backend/retry attempts) can take long enough to be noticeable, so
        this must never run on the UI thread.
        """

        def worker() -> None:
            try:
                devices = list_local_device_ids()
            except Exception:
                logger.exception("Error scanning for local camera devices")
                devices = []
            idle_add(self._on_devices_scanned, devices)

        threading.Thread(
            target=worker, name="CameraDeviceScan", daemon=True
        ).start()

    def _on_devices_scanned(self, devices: list[str]) -> None:
        if self._closed:
            return
        self._available_devices = devices
        self.device_row.set_model(
            Gtk.StringList.new(
                [_("Choose device")]
                + [display_name(device_id) for device_id in devices]
            )
        )
        self.device_row.set_sensitive(True)
        if devices:
            self.type_row.set_selected(1)
            self.device_row.set_selected(1)

    def _on_response(self, dialog, response_id) -> None:
        self._closed = True
        self.preview_image.set_paintable(None)
        self._preview_pixbuf = None

    def _on_destroy(self, *args) -> None:
        self._closed = True
