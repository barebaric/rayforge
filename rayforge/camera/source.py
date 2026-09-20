import logging
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import urlsplit

import cv2
import numpy as np

from .models.camera import Camera, CameraSourceType

logger = logging.getLogger(__name__)

# Devices currently held open by a LocalDeviceSource in this process.
# Probing (e.g. to populate a device picker) must skip these targets:
# opening a second cv2.VideoCapture on a device that is already open by
# another thread can crash the underlying V4L2/DirectShow driver.
_open_local_devices_lock = threading.Lock()
_open_local_devices: set[str] = set()


def _to_videocapture_arg(target: str) -> int | str:
    """Convert a scan target for cv2.VideoCapture.

    OpenCV 5.0 treats string arguments as filenames, so numeric
    device IDs like "0" must be passed as integers. Device paths
    (e.g. /dev/v4l/by-id/...) are passed as strings.
    """
    if target.isdigit():
        return int(target)
    return target


@dataclass(frozen=True)
class SourceDescriptor:
    """User-facing label/value pair for a discoverable camera source."""

    label: str
    value: str


class CameraSource(ABC):
    """Abstract camera source interface used by CameraController."""

    supports_hardware_controls = False
    reconnect_delay_seconds = 2.0
    warning_log_interval_seconds: float | None = None

    def __init__(self, config: Camera):
        self.config = config
        # Set by the owning CameraController via bind_cancel_event() so
        # that open-retry loops and reconnect/frame-pacing waits inside
        # this source can abort promptly when the controller is asked to
        # stop, instead of only noticing on the next iteration.
        self._cancel_event: threading.Event | None = None

    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_frame(self) -> np.ndarray | None:
        raise NotImplementedError

    def apply_settings(self) -> None:
        return

    def apply_preview_settings(self) -> None:
        self.apply_settings()

    def open_for_preview(self) -> None:
        self.open()

    def read_current_settings(self) -> dict[str, object]:
        return {}

    def stop(self) -> None:
        self.close()

    def bind_cancel_event(self, event: threading.Event) -> None:
        """Attach the controller's stop event to this source instance."""
        self._cancel_event = event

    def _cancelled(self) -> bool:
        return self._cancel_event is not None and self._cancel_event.is_set()

    def _wait_or_cancelled(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds, or until cancelled.

        Returns True if the wait ended because of cancellation (so the
        caller should abort early), False if the full timeout elapsed.
        """
        if self._cancel_event is not None:
            return self._cancel_event.wait(timeout)
        time.sleep(timeout)
        return False

    @classmethod
    def list_available_sources(cls) -> list[SourceDescriptor]:
        return []


# Number of warm-up reads attempted right after a device reports
# isOpened() == True before its backend is trusted, and the delay
# between attempts. Some Windows backends (DirectShow/MediaFoundation
# in particular) report a successful open immediately but then fail
# every subsequent read() for a device that is not really usable (e.g.
# a resolution/format negotiation failure, or exclusive access denied).
# Validating a few reads up front avoids treating such a device as
# working only to have the capture loop immediately fail over anyway.
OPEN_VALIDATION_ATTEMPTS = 3
OPEN_VALIDATION_DELAY = 0.1


def _capture_has_initial_frame(
    cap: cv2.VideoCapture,
    attempts: int = OPEN_VALIDATION_ATTEMPTS,
    delay: float = OPEN_VALIDATION_DELAY,
    cancel_event: "threading.Event | None" = None,
) -> bool:
    """Check whether an opened capture device actually delivers frames."""
    for attempt in range(attempts):
        try:
            ret, frame = cap.read()
        except cv2.error:
            ret, frame = False, None

        if ret and frame is not None:
            return True

        if attempt < attempts - 1:
            if cancel_event is not None:
                if cancel_event.wait(delay):
                    return False
            else:
                time.sleep(delay)

    return False


def validate_source_uri(source_type: CameraSourceType, uri: str) -> str | None:
    """Return an error for an invalid network source URI, or None."""
    allowed_schemes = {
        CameraSourceType.HTTP_SNAPSHOT: ("http", "https"),
        CameraSourceType.HTTP_STREAM: ("http", "https"),
        CameraSourceType.RTSP: ("rtsp", "rtsps"),
    }.get(source_type)
    if allowed_schemes is None:
        return None

    try:
        parsed = urlsplit(uri.strip())
    except ValueError:
        return "URL is malformed"
    if parsed.scheme.lower() not in allowed_schemes:
        schemes = ", ".join(f"{scheme}://" for scheme in allowed_schemes)
        return f"URL must start with one of: {schemes}"
    if not parsed.netloc:
        return "URL must include a host"
    return None


def get_backends_for_platform():
    """Return list of (backend_constant, name) tuples for current platform."""
    import sys

    if sys.platform.startswith("linux"):
        return [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "default"),
        ]
    if sys.platform == "win32":
        return [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "MediaFoundation"),
            (cv2.CAP_ANY, "default"),
        ]
    return [(cv2.CAP_ANY, "default")]


def _get_linux_scan_targets() -> list[str]:
    """Get device identifiers to scan on Linux.

    Prefers persistent /dev/v4l/by-id/ paths. Falls back to
    numeric indices if by-id is not available.
    """
    from .v4l import get_sorted_by_id_paths

    by_id_paths = get_sorted_by_id_paths()
    if by_id_paths:
        return by_id_paths
    return [str(i) for i in range(10)]


class LocalDeviceSource(CameraSource):
    """Local USB or built-in camera opened through OpenCV backends."""

    supports_hardware_controls = True
    MAX_OPEN_RETRIES = 3
    RETRY_DELAY = 0.5
    READ_TIMEOUT_MSEC = 1000
    MAX_CONSECUTIVE_READ_FAILURES = 3
    MJPG_FOURCC = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore
    YUYV_FOURCC = cv2.VideoWriter_fourcc(*"YUYV")  # type: ignore

    def __init__(self, config: Camera):
        super().__init__(config)
        # The device this source instance owns. Captured once at
        # construction time rather than read live from `config.device_id`
        # on every use, so a model mutation that happens while this
        # source is mid-open/mid-read cannot make it silently retarget to
        # a different device. Switching devices always goes through the
        # controller creating a *new* source (see CameraController's
        # source-change restart logic in `_on_config_changed`).
        self._device_id: str = config.device_id
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


class OpenCvUrlSource(CameraSource):
    """Continuous URL-backed stream handled directly by OpenCV.

    This shared base is used for source types that OpenCV can open as a
    long-lived stream and then read frame-by-frame.
    """

    reconnect_delay_seconds = 10.0
    warning_log_interval_seconds = 30.0
    MAX_CONSECUTIVE_READ_FAILURES = 3

    def __init__(self, config: Camera):
        super().__init__(config)
        # Owned/frozen at construction time so a live model mutation
        # cannot retarget an already-open stream; see the analogous
        # comment on LocalDeviceSource.__init__.
        self._uri: str = config.source_uri
        self.cap = None
        self._read_failures = 0

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self._uri, cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.close()
            raise OSError(f"Cannot open stream {self._uri}")
        self._read_failures = 0

    def close(self) -> None:
        if self.cap is None:
            return
        try:
            if self.cap.isOpened():
                self.cap.release()
        finally:
            self.cap = None

    def read_frame(self) -> np.ndarray | None:
        if self.cap is None:
            raise OSError("Camera source is not open")
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._read_failures += 1
            if self._read_failures >= self.MAX_CONSECUTIVE_READ_FAILURES:
                raise OSError(
                    "Network camera stream stopped returning frames; "
                    "reconnecting"
                )
            return None
        self._read_failures = 0
        return frame


class RtspSource(OpenCvUrlSource):
    """RTSP camera stream source.

    This currently inherits the generic OpenCV URL-stream behavior unchanged,
    but keeps RTSP as a distinct source type for clearer configuration and
    future protocol-specific handling.
    """

    def open(self) -> None:
        error = validate_source_uri(CameraSourceType.RTSP, self._uri)
        if error:
            raise OSError(error)
        super().open()


class HttpStreamSource(OpenCvUrlSource):
    """HTTP/HTTPS stream source.

    This currently inherits the generic OpenCV URL-stream behavior unchanged.
    It is intended for continuous HTTP camera feeds such as MJPEG-over-HTTP.
    """

    def open(self) -> None:
        error = validate_source_uri(CameraSourceType.HTTP_STREAM, self._uri)
        if error:
            raise OSError(error)
        super().open()


class HttpSnapshotSource(CameraSource):
    """HTTP/HTTPS snapshot source that fetches one image per poll.

    We read at most one image every 2 seconds, and cache the last
    successfully decoded frame in memory for the current session.
    For example, the "Creality Falcon A1 Pro" stops returning valid
    images during a job run; in that case the cached frame is reused.
    """

    MIN_POLL_INTERVAL_SECONDS = 2.0
    OFFLINE_RETRY_INTERVAL_SECONDS = 10.0
    WARNING_LOG_INTERVAL_SECONDS = 30.0
    MAX_SNAPSHOT_BYTES = 20 * 1024 * 1024

    def __init__(self, config: Camera):
        super().__init__(config)
        # Owned/frozen at construction time; see LocalDeviceSource.__init__.
        self._uri: str = config.source_uri
        self._timeout = float(config.source_config.get("timeout_seconds", 5.0))
        self._last_success_frame: np.ndarray | None = None
        self._last_poll_time: float | None = None
        self._next_poll_interval = self.MIN_POLL_INTERVAL_SECONDS
        self._last_warning_log_time: float | None = None

    def open(self) -> None:
        error = validate_source_uri(CameraSourceType.HTTP_SNAPSHOT, self._uri)
        if error:
            raise OSError(error)

    def close(self) -> None:
        return

    def _cached_frame(self) -> np.ndarray | None:
        if self._last_success_frame is None:
            return None
        return self._last_success_frame.copy()

    def _should_log_warning(self, now: float) -> bool:
        if self._last_warning_log_time is None:
            self._last_warning_log_time = now
            return True
        if (
            now - self._last_warning_log_time
            >= self.WARNING_LOG_INTERVAL_SECONDS
        ):
            self._last_warning_log_time = now
            return True
        return False

    def _log_warning(self, now: float, message: str, *args) -> None:
        if self._should_log_warning(now):
            logger.warning(message, *args)

    def _read_snapshot_payload(self, response, now: float) -> bytes | None:
        payload = response.read(self.MAX_SNAPSHOT_BYTES + 1)
        if len(payload) <= self.MAX_SNAPSHOT_BYTES:
            return payload
        self._next_poll_interval = self.OFFLINE_RETRY_INTERVAL_SECONDS
        self._log_warning(
            now,
            "Snapshot URL returned an image larger than %s bytes, "
            "reusing last good frame if available",
            self.MAX_SNAPSHOT_BYTES,
        )
        return None

    def _wait_until_next_poll(self, now: float) -> float | None:
        if self._last_poll_time is None:
            return now
        elapsed = now - self._last_poll_time
        remaining = self._next_poll_interval - elapsed
        if remaining <= 0:
            return now
        if self._wait_or_cancelled(remaining):
            return None
        return time.monotonic()

    def read_frame(self) -> np.ndarray | None:
        now = time.monotonic()
        next_poll = self._wait_until_next_poll(now)
        if next_poll is None:
            return None
        now = next_poll

        request = urllib.request.Request(
            self._uri,
            headers={"User-Agent": "Rayforge Camera"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout
            ) as response:
                self._last_poll_time = now
                content_type = response.headers.get_content_type()
                if not content_type.startswith("image/"):
                    self._next_poll_interval = (
                        self.OFFLINE_RETRY_INTERVAL_SECONDS
                    )
                    self._log_warning(
                        now,
                        "Snapshot URL returned non-image content-type '%s'",
                        content_type,
                    )
                    return self._cached_frame()
                payload = self._read_snapshot_payload(response, now)
                if payload is None:
                    return self._cached_frame()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._last_poll_time = now
            self._next_poll_interval = self.OFFLINE_RETRY_INTERVAL_SECONDS
            cached_frame = self._cached_frame()
            if cached_frame is not None:
                self._log_warning(
                    now,
                    "Snapshot fetch failed, reusing last good frame: %s",
                    exc,
                )
                return cached_frame
            raise OSError(f"Failed to fetch snapshot: {exc}") from exc
        arr = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            self._next_poll_interval = self.OFFLINE_RETRY_INTERVAL_SECONDS
            self._log_warning(
                now,
                "Snapshot URL returned undecodable image data, "
                "reusing last good frame if available",
            )
            return self._cached_frame()
        self._next_poll_interval = self.MIN_POLL_INTERVAL_SECONDS
        self._last_success_frame = frame.copy()
        return frame


def create_camera_source(config: Camera) -> CameraSource:
    """Create the runtime source implementation for a camera config."""
    source_type = config.source_type
    if source_type is CameraSourceType.LOCAL_DEVICE:
        return LocalDeviceSource(config)
    if source_type is CameraSourceType.HTTP_SNAPSHOT:
        return HttpSnapshotSource(config)
    if source_type is CameraSourceType.HTTP_STREAM:
        return HttpStreamSource(config)
    if source_type is CameraSourceType.RTSP:
        return RtspSource(config)
    raise ValueError(f"Unsupported camera source type: {source_type}")


def list_local_device_ids() -> list[str]:
    """Return only the raw identifiers for discoverable local cameras."""
    return [
        descriptor.value
        for descriptor in LocalDeviceSource.list_available_sources()
    ]
