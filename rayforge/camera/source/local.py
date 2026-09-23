import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..models.camera import Camera
from .base import (
    CameraSource,
    SourceConfig,
    SourceDescriptor,
    _capture_has_initial_frame,
    _get_linux_scan_targets,
    _open_local_devices,
    _open_local_devices_lock,
    _to_videocapture_arg,
)

logger = logging.getLogger(__name__)


@dataclass
class LocalDeviceConfig(SourceConfig):
    device_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"device_id": self.device_id, **self.extra}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocalDeviceConfig":
        return cls(
            device_id=str(data.get("device_id", "")),
            extra={
                key: value for key, value in data.items() if key != "device_id"
            },
        )


def get_backends_for_platform():
    from . import get_backends_for_platform as get_backends

    return get_backends()


class LocalDeviceSource(CameraSource):
    """Local USB or built-in camera opened through OpenCV backends."""

    supports_hardware_controls = True
    config_class = LocalDeviceConfig
    MAX_OPEN_RETRIES = 3
    RETRY_DELAY = 0.5
    READ_TIMEOUT_MSEC = 1000
    MAX_CONSECUTIVE_READ_FAILURES = 3
    MJPG_FOURCC = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore
    YUYV_FOURCC = cv2.VideoWriter_fourcc(*"YUYV")  # type: ignore

    def __init__(self, config: Camera):
        super().__init__(config)
        self.source_config = self.from_dict(config.source_config)
        assert isinstance(self.source_config, LocalDeviceConfig)
        # The device this source instance owns. Captured once at
        # construction time rather than read live from `config.device_id`
        # on every use, so a model mutation that happens while this
        # source is mid-open/mid-read cannot make it silently retarget to
        # a different device. Switching devices always goes through the
        # controller creating a *new* source (see CameraController's
        # source-change restart logic in `_on_config_changed`).
        self._device_id = self.source_config.device_id
        self.cap = None
        self._backend_used = None
        self._read_failures = 0
        self._disabled_backend_names: set[str] = set()
        self._suppress_open_settings = False
        self._registered_device_id: str | None = None

    @staticmethod
    def _scan_targets() -> list[str]:
        import sys

        if sys.platform.startswith("linux"):
            return _get_linux_scan_targets()
        return [str(i) for i in range(10)]

    @classmethod
    def list_available_sources(cls) -> list[SourceDescriptor]:
        """List available local camera device IDs.

        Targets that are already held open by a live capture (this
        camera's own stream or another camera's) are reported as
        available without re-probing them: opening a second
        cv2.VideoCapture on a device that another thread already has
        open can crash the underlying V4L2/DirectShow driver.
        """
        with _open_local_devices_lock:
            in_use = set(_open_local_devices)

        devices = []
        for target in cls._scan_targets():
            if target in in_use:
                devices.append(SourceDescriptor(label=target, value=target))
                continue
            for backend, name in get_backends_for_platform():
                try:
                    cap = cv2.VideoCapture(
                        _to_videocapture_arg(target), backend
                    )
                    if cap.isOpened() and _capture_has_initial_frame(
                        cap, attempts=1
                    ):
                        devices.append(
                            SourceDescriptor(label=target, value=target)
                        )
                        cap.release()
                        logger.debug(f"Found camera {target} via {name}")
                        break
                    if cap:
                        cap.release()
                except (cv2.error, OSError):
                    logger.debug("Error probing camera %s", target)
        return devices

    def _try_open(self, device_id: str, backend: int, backend_name: str):
        """Try to open camera with specific backend. Returns cap or None."""
        logger.debug(
            "Opening camera %s with %s backend", device_id, backend_name
        )
        cap = cv2.VideoCapture(_to_videocapture_arg(device_id), backend)
        if cap.isOpened():
            if _capture_has_initial_frame(
                cap, cancel_event=self._cancel_event
            ):
                logger.info(
                    "Camera %s opened with %s backend",
                    device_id,
                    backend_name,
                )
                return cap
            logger.warning(
                "Camera %s opened with %s backend but did not yield an "
                "initial frame (a common failure mode for Windows "
                "DirectShow/MediaFoundation backends); trying the next "
                "option",
                device_id,
                backend_name,
            )
        if cap:
            cap.release()
        return None

    def open(self) -> None:
        # Use the device this source instance owns (frozen at
        # construction), not a live re-read of `self.config.device_id`:
        # see the comment in __init__.
        device_id = self._device_id
        # Register the device as in-use before attempting to open it, not
        # just after success. Otherwise a concurrent scan (e.g. to
        # populate a device picker) can race with this open() call and
        # try to open the same device at the same time, which can crash
        # the underlying V4L2/DirectShow driver.
        self._register_open_device(device_id)
        last_error = None
        backends = [
            (backend, name)
            for backend, name in get_backends_for_platform()
            if name not in self._disabled_backend_names
        ]
        if not backends:
            self._disabled_backend_names.clear()
            backends = get_backends_for_platform()
        for backend, name in backends:
            if self._cancelled():
                self._unregister_open_device()
                raise OSError(
                    f"Open aborted by cancel event for camera {device_id}"
                )
            for attempt in range(self.MAX_OPEN_RETRIES):
                try:
                    cap = self._try_open(device_id, backend, name)
                    if cap:
                        if not self._suppress_open_settings:
                            try:
                                cap.set(
                                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                                    self.READ_TIMEOUT_MSEC,
                                )
                            except (cv2.error, AttributeError):
                                pass
                        self.cap = cap
                        self._backend_used = name
                        self._read_failures = 0
                        return
                except (cv2.error, OSError) as exc:
                    last_error = exc
                    logger.warning(
                        "Error opening camera %s with %s (attempt %s): %s",
                        device_id,
                        name,
                        attempt + 1,
                        exc,
                    )
                # Interruptible retry delay: abort promptly if the
                # controller requested a stop mid-open instead of only
                # noticing after every backend/attempt.
                if (
                    attempt < self.MAX_OPEN_RETRIES - 1
                    and self._wait_or_cancelled(self.RETRY_DELAY)
                ):
                    self._unregister_open_device()
                    raise OSError(
                        f"Open aborted by cancel event for camera {device_id}"
                    )
        self._unregister_open_device()
        raise OSError(
            f"Cannot open camera {device_id}. "
            f"Tried: {[name for _, name in backends]}. "
            f"Last error: {last_error}"
        )

    def close(self) -> None:
        self._unregister_open_device()
        if self.cap is None:
            return
        try:
            if self.cap.isOpened():
                self.cap.release()
        finally:
            self.cap = None

    def _register_open_device(self, device_id: str) -> None:
        with _open_local_devices_lock:
            _open_local_devices.add(device_id)
        self._registered_device_id = device_id

    def _unregister_open_device(self) -> None:
        if self._registered_device_id is None:
            return
        with _open_local_devices_lock:
            _open_local_devices.discard(self._registered_device_id)
        self._registered_device_id = None

    def read_frame(self) -> np.ndarray | None:
        if self.cap is None:
            raise OSError("Camera source is not open")
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._read_failures += 1
            if self._read_failures >= self.MAX_CONSECUTIVE_READ_FAILURES:
                if self._backend_used:
                    # Let the next reconnect try a different backend.
                    self._disabled_backend_names.add(self._backend_used)
                raise OSError(
                    "Local camera stopped returning frames; reconnecting"
                )
            return None
        self._read_failures = 0
        return frame

    def _apply_capture_settings(self, cap) -> None:
        preferred_fourcc = (
            self.YUYV_FOURCC if self.config.prefer_yuyv else self.MJPG_FOURCC
        )
        try:
            if cap.set(cv2.CAP_PROP_FOURCC, preferred_fourcc):
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except cv2.error:
                    pass
        except cv2.error:
            logger.debug("Could not set preferred pixel format")
        if self.config.resolution is not None:
            w, h = self.config.resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def _apply_image_settings(self, cap) -> None:
        if self.config.white_balance is None:
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        else:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.config.white_balance)
        cap.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness)

    def apply_preview_settings(self) -> None:
        """Keep local preview non-destructive by not writing device state."""
        return

    def open_for_preview(self) -> None:
        self._suppress_open_settings = True
        try:
            self.open()
        finally:
            self._suppress_open_settings = False

    def apply_settings(self) -> None:
        """Apply the current settings to the local capture device."""
        if self.cap is None:
            raise OSError("Camera source is not open")
        self._apply_capture_settings(self.cap)
        self._apply_image_settings(self.cap)

    def read_current_settings(self) -> dict[str, object]:
        if self.cap is None:
            raise OSError("Camera source is not open")
        cap = self.cap
        settings: dict[str, object] = {
            "contrast": float(cap.get(cv2.CAP_PROP_CONTRAST)),
            "brightness": float(cap.get(cv2.CAP_PROP_BRIGHTNESS)),
        }
        auto_wb = cap.get(cv2.CAP_PROP_AUTO_WB)
        if auto_wb >= 0.5:
            settings["white_balance"] = None
        else:
            white_balance = float(cap.get(cv2.CAP_PROP_WB_TEMPERATURE))
            if 2500 <= white_balance <= 10000:
                settings["white_balance"] = white_balance
        return settings
