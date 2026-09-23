import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar
from urllib.parse import urlsplit

import cv2
import numpy as np

from ..models.camera import Camera, CameraSourceType

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


@dataclass
class SourceConfig:
    extra: dict[str, Any] = field(default_factory=dict)
    fields: ClassVar[frozenset[str]] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return {**self.extra}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceConfig":
        return cls(extra={key: value for key, value in data.items()})


class CameraSource(ABC):
    """Abstract camera source interface used by CameraController."""

    supports_hardware_controls = False
    reconnect_delay_seconds = 2.0
    warning_log_interval_seconds: float | None = None
    config_class: ClassVar[type[SourceConfig]] = SourceConfig

    def __init__(self, config: Camera):
        self.config = config
        # Set by the owning CameraController via bind_cancel_event() so
        # that open-retry loops and reconnect/frame-pacing waits inside
        # this source can abort promptly when the controller is asked to
        # stop, instead of only noticing on the next iteration.
        self._cancel_event: threading.Event | None = None
        self.source_config = SourceConfig.from_dict(config.source_config)

    def to_dict(self) -> dict[str, Any]:
        return self.source_config.to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceConfig:
        return cls.config_class.from_dict(data)

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
    from ..v4l import get_sorted_by_id_paths

    by_id_paths = get_sorted_by_id_paths()
    if by_id_paths:
        return by_id_paths
    return [str(i) for i in range(10)]
