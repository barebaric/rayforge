import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..models.camera import Camera, CameraSourceType
from .base import CameraSource, SourceConfig, validate_source_uri

logger = logging.getLogger(__name__)


@dataclass
class UrlSourceConfig(SourceConfig):
    uri: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"uri": self.uri, **self.extra}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UrlSourceConfig":
        return cls(
            uri=str(data.get("uri", "")),
            extra={key: value for key, value in data.items() if key != "uri"},
        )


@dataclass
class HttpSnapshotConfig(UrlSourceConfig):
    timeout_seconds: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "timeout_seconds": self.timeout_seconds,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HttpSnapshotConfig":
        return cls(
            uri=str(data.get("uri", "")),
            timeout_seconds=float(data.get("timeout_seconds", 5.0)),
            extra={
                key: value
                for key, value in data.items()
                if key not in {"uri", "timeout_seconds"}
            },
        )


class OpenCvUrlSource(CameraSource):
    """Continuous URL-backed stream handled directly by OpenCV.

    This shared base is used for source types that OpenCV can open as a
    long-lived stream and then read frame-by-frame.
    """

    reconnect_delay_seconds = 10.0
    warning_log_interval_seconds = 30.0
    MAX_CONSECUTIVE_READ_FAILURES = 3
    config_class = UrlSourceConfig

    def __init__(self, config: Camera):
        super().__init__(config)
        self.source_config = self.from_dict(config.source_config)
        assert isinstance(self.source_config, UrlSourceConfig)
        # Owned/frozen at construction time so a live model mutation
        # cannot retarget an already-open stream; see the analogous
        # comment on LocalDeviceSource.__init__.
        self._uri = self.source_config.uri
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
    config_class = HttpSnapshotConfig

    def __init__(self, config: Camera):
        super().__init__(config)
        self.source_config = self.from_dict(config.source_config)
        assert isinstance(self.source_config, HttpSnapshotConfig)
        self._uri = self.source_config.uri
        self._timeout = self.source_config.timeout_seconds
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
