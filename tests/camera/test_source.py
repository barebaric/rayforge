from unittest.mock import patch

import cv2
import pytest

from rayforge.camera.models.camera import Camera, CameraSourceType
from rayforge.camera.source import (
    LocalDeviceSource,
    validate_source_uri,
)
from rayforge.camera.source.local import (
    _capture_has_initial_frame,
    _open_local_devices,
    _open_local_devices_lock,
)


@pytest.mark.parametrize(
    ("source_type", "uri"),
    [
        (CameraSourceType.HTTP_SNAPSHOT, "https://camera.local/image.jpg"),
        (CameraSourceType.HTTP_STREAM, "http://camera.local/stream.mjpeg"),
        (CameraSourceType.RTSP, "rtsps://camera.local/stream"),
    ],
)
def test_validate_source_uri_accepts_supported_urls(source_type, uri):
    assert validate_source_uri(source_type, uri) is None


@pytest.mark.parametrize(
    ("source_type", "uri"),
    [
        (CameraSourceType.HTTP_SNAPSHOT, "rtsp://camera.local/image.jpg"),
        (CameraSourceType.HTTP_STREAM, "camera.local/stream.mjpeg"),
        (CameraSourceType.RTSP, "https://camera.local/stream"),
        (CameraSourceType.HTTP_SNAPSHOT, "https:///image.jpg"),
        (CameraSourceType.HTTP_STREAM, "http://["),
    ],
)
def test_validate_source_uri_rejects_invalid_urls(source_type, uri):
    assert validate_source_uri(source_type, uri) is not None


@pytest.fixture(autouse=True)
def _clear_open_local_devices():
    """Ensure the in-use device registry doesn't leak between tests."""
    with _open_local_devices_lock:
        _open_local_devices.clear()
    yield
    with _open_local_devices_lock:
        _open_local_devices.clear()


def test_list_available_sources_skips_devices_already_open():
    """Devices already open elsewhere must not be re-probed.

    Opening a second cv2.VideoCapture on a device that's already open
    (e.g. by an active capture thread) can crash the underlying V4L2/
    DirectShow driver, so already-open devices must be reported as
    available without touching cv2.VideoCapture again.
    """
    in_use_device = "/dev/v4l/by-id/in-use-camera"
    free_device = "/dev/v4l/by-id/free-camera"
    opened_targets = []

    class MockVideoCapture:
        def __init__(self, device, backend=None):
            opened_targets.append(device)
            self._opened = device == free_device

        def isOpened(self):
            return self._opened

        def read(self):
            import numpy as np

            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self):
            pass

    with _open_local_devices_lock:
        _open_local_devices.add(in_use_device)

    with (
        patch(
            "rayforge.camera.source.local.cv2.VideoCapture", MockVideoCapture
        ),
        patch(
            "rayforge.camera.source.local.get_backends_for_platform",
            return_value=[(cv2.CAP_V4L2, "V4L2")],
        ),
        patch.object(
            LocalDeviceSource,
            "_scan_targets",
            return_value=[in_use_device, free_device],
        ),
    ):
        devices = LocalDeviceSource.list_available_sources()

    values = [d.value for d in devices]
    assert in_use_device in values
    assert free_device in values
    # The in-use device must never have been probed via VideoCapture.
    assert in_use_device not in opened_targets
    assert free_device in opened_targets


def test_open_registers_device_before_probing_and_close_unregisters():
    """The device must be marked in-use before the open attempt itself.

    Registering only after a successful open leaves a race window where
    a concurrent scan could try to open the same device at the same
    time. Registering first closes that window.
    """
    device_id = "/dev/v4l/by-id/race-camera"
    registered_during_open = []

    class MockVideoCapture:
        def __init__(self, device, backend=None):
            with _open_local_devices_lock:
                registered_during_open.append(device in _open_local_devices)

        def isOpened(self):
            return True

        def read(self):
            import numpy as np

            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def set(self, *args, **kwargs):
            return True

        def release(self):
            pass

    camera = Camera("Test", device_id)
    source = LocalDeviceSource(camera)

    with (
        patch(
            "rayforge.camera.source.local.cv2.VideoCapture", MockVideoCapture
        ),
        patch(
            "rayforge.camera.source.local.get_backends_for_platform",
            return_value=[(cv2.CAP_V4L2, "V4L2")],
        ),
    ):
        source.open()

    assert registered_during_open == [True]
    with _open_local_devices_lock:
        assert device_id in _open_local_devices

    source.close()
    with _open_local_devices_lock:
        assert device_id not in _open_local_devices


def test_capture_has_initial_frame_rejects_never_reading_device():
    """Some backends (notably Windows DSHOW/MSMF) report isOpened() as
    True immediately even though the device never actually delivers a
    frame. Treating such a capture as usable would surface a camera
    that looks connected but is permanently frozen, so validation must
    reject it.
    """

    class NeverReadsCapture:
        def read(self):
            return False, None

    cap: cv2.VideoCapture = NeverReadsCapture()  # type: ignore[assignment]
    result = _capture_has_initial_frame(cap, attempts=3)
    assert result is False


def test_capture_has_initial_frame_accepts_device_that_eventually_reads():
    """A device that only starts delivering frames after a short warm-up
    (common for USB webcams) must still be accepted within the retry
    budget, not rejected on the first failed attempt.
    """
    import numpy as np

    class SlowStartCapture:
        def __init__(self):
            self.calls = 0

        def read(self):
            self.calls += 1
            if self.calls < 3:
                return False, None
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

    cap: cv2.VideoCapture = SlowStartCapture()  # type: ignore[assignment]
    result = _capture_has_initial_frame(cap, attempts=3, delay=0.0)
    assert result is True
    assert cap.calls == 3  # type: ignore[attr-defined]


def test_open_raises_when_device_never_delivers_initial_frame():
    """LocalDeviceSource.open() must reject a device that reports
    isOpened() == True but never successfully reads a frame, instead of
    silently exposing a permanently-frozen camera to the rest of the
    app (the Windows DSHOW/MSMF false-open scenario).
    """

    class OpensButNeverReadsCapture:
        def __init__(self, device, backend=None):
            pass

        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            pass

    camera = Camera("Test", "0")
    source = LocalDeviceSource(camera)

    with (
        patch(
            "rayforge.camera.source.local.cv2.VideoCapture",
            OpensButNeverReadsCapture,
        ),
        patch(
            "rayforge.camera.source.local.get_backends_for_platform",
            return_value=[(cv2.CAP_V4L2, "V4L2")],
        ),
        patch("rayforge.camera.source.base.time.sleep"),
        pytest.raises(OSError),
    ):
        source.open()


def test_open_aborts_promptly_when_cancel_event_is_set():
    """A stop request must interrupt the open-retry loop's inter-attempt
    delay immediately instead of waiting out the full retry budget, so
    controller shutdown stays responsive.
    """
    import threading

    class NeverOpensCapture:
        def __init__(self, device, backend=None):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    camera = Camera("Test", "0")
    source = LocalDeviceSource(camera)
    cancel_event = threading.Event()
    source.bind_cancel_event(cancel_event)
    cancel_event.set()

    with (
        patch(
            "rayforge.camera.source.local.cv2.VideoCapture", NeverOpensCapture
        ),
        patch(
            "rayforge.camera.source.local.get_backends_for_platform",
            return_value=[(cv2.CAP_V4L2, "V4L2")],
        ),
        pytest.raises(OSError),
    ):
        source.open()
