# flake8: noqa: E402
import gi

gi.require_version("GdkPixbuf", "2.0")
import threading
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from rayforge.camera.controller import CameraController
from rayforge.camera.models.camera import Camera, CameraSourceType
from rayforge.camera.outside_view import (
    LensModel,
    frame_coverage_extent,
    lens_remap_maps,
    radial_limit,
    to_opaque_bgra,
    warp_transparent,
    world_to_raw,
)
from rayforge.camera.source import (
    CameraSource,
    HttpSnapshotSource,
    HttpStreamSource,
)
from rayforge.camera.source.local import _to_videocapture_arg


def test_controller_initialization():
    camera_config = Camera("Test Camera", "device123")
    controller = CameraController(camera_config)
    assert controller.config.name == "Test Camera"
    assert controller.config is camera_config
    assert controller.image_data is None


@patch(
    "rayforge.camera.controller.idle_add",
    side_effect=lambda func, *args, **kwargs: func(*args, **kwargs),
)
def test_capture_image(mock_idle_add):
    original_videocapture = cv2.VideoCapture

    try:

        class MockVideoCapture:
            def __init__(self, device, backend=None):
                self.device = device
                self.opened = True

            def isOpened(self):
                return self.opened

            def read(self):
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                return True, dummy_frame

            def set(self, prop_id, value):
                pass

            def release(self):
                self.opened = False

        cv2.VideoCapture = MockVideoCapture

        camera_config = Camera("Mock Camera", "0")
        controller = CameraController(camera_config)
        controller.capture_image()

        assert controller.image_data is not None
        assert controller.image_data.shape == (480, 640, 3)

    finally:
        cv2.VideoCapture = original_videocapture


@patch(
    "rayforge.camera.controller.idle_add",
    side_effect=lambda func, *args, **kwargs: func(*args, **kwargs),
)
def test_capture_image_preview_avoids_device_setting_writes(
    mock_idle_add,
):
    original_videocapture = cv2.VideoCapture
    set_calls = []

    try:

        class MockVideoCapture:
            def __init__(self, device, backend=None):
                self.device = device
                self.opened = True

            def isOpened(self):
                return self.opened

            def read(self):
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                return True, dummy_frame

            def set(self, prop_id, value):
                set_calls.append((prop_id, value))
                return True

            def release(self):
                self.opened = False

        cv2.VideoCapture = MockVideoCapture

        camera_config = Camera("Mock Camera", "0")
        controller = CameraController(camera_config)
        controller.capture_image(apply_settings=False)

        assert controller.image_data is not None
    finally:
        cv2.VideoCapture = original_videocapture

    assert set_calls == []


def test_get_work_surface_image_basic():
    camera_config = Camera("Test Camera", "0")
    controller = CameraController(camera_config)

    controller._image_data = np.zeros((480, 640, 3), dtype=np.uint8)
    controller._image_data[100:400, 100:500] = [
        255,
        255,
        255,
    ]

    image_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
    world_points = [(0, 100), (100, 100), (100, 0), (0, 0)]
    camera_config.image_to_world = (image_points, world_points)

    output_size = (200, 200)
    physical_area = ((0, 0), (100, 100))

    aligned_image = controller.get_work_surface_image(
        output_size, physical_area
    )

    assert aligned_image is not None
    assert aligned_image.shape == (output_size[1], output_size[0], 3)

    center_x, center_y = output_size[0] // 2, output_size[1] // 2
    pixel_value = aligned_image[center_y, center_x]
    np.testing.assert_array_equal(pixel_value, [255, 255, 255])


def test_get_work_surface_image_no_corresponding_points():
    camera_config = Camera("Test Camera", "0")
    controller = CameraController(camera_config)
    controller._image_data = np.zeros((480, 640, 3), dtype=np.uint8)
    output_size = (200, 200)
    physical_area = ((0, 0), (100, 100))

    result = controller.get_work_surface_image(output_size, physical_area)
    assert result is not None
    assert result.shape == (output_size[1], output_size[0], 3)


def test_get_work_surface_image_no_image_data():
    camera_config = Camera("Test Camera", "0")
    image_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
    world_points = [(0, 100), (100, 100), (100, 0), (0, 0)]
    camera_config.image_to_world = (image_points, world_points)

    controller = CameraController(camera_config)

    output_size = (200, 200)
    physical_area = ((0, 0), (100, 100))

    aligned_image = controller.get_work_surface_image(
        output_size, physical_area
    )
    assert aligned_image is None


def test_to_videocapture_arg():
    assert _to_videocapture_arg("0") == 0
    assert _to_videocapture_arg("12") == 12
    assert (
        _to_videocapture_arg("/dev/v4l/by-id/usb-Foo_Webcam-video-index0")
        == "/dev/v4l/by-id/usb-Foo_Webcam-video-index0"
    )


def test_list_available_devices_uses_int_for_numeric_ids():
    """Numeric device IDs must be passed as ints to VideoCapture.

    OpenCV 5.0 treats string arguments as filenames, so probing with
    numeric strings would never find a camera (regression test for
    the "no device found" issue).
    """
    calls = []

    class MockVideoCapture:
        def __init__(self, device, backend=None):
            calls.append(device)

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self):
            pass

    targets = ["0", "1", "/dev/v4l/by-id/usb-Foo_Webcam-video-index0"]

    with (
        patch("rayforge.camera.source.cv2.VideoCapture", MockVideoCapture),
        patch(
            "rayforge.camera.source.get_backends_for_platform",
            return_value=[(cv2.CAP_V4L2, "V4L2")],
        ),
        patch(
            "rayforge.camera.source.LocalDeviceSource._scan_targets",
            return_value=targets,
        ),
    ):
        devices = CameraController.list_available_devices()

    assert devices == targets
    assert calls == [0, 1, "/dev/v4l/by-id/usb-Foo_Webcam-video-index0"]


@patch(
    "rayforge.camera.controller.idle_add",
    side_effect=lambda func, *args, **kwargs: func(*args, **kwargs),
)
def test_http_snapshot_capture_uses_source_factory(mock_idle_add):
    dummy_frame = np.zeros((10, 20, 3), dtype=np.uint8)

    class MockSource:
        def open(self):
            return

        def close(self):
            return

        def apply_settings(self):
            return

        def read_frame(self):
            return dummy_frame

    camera_config = Camera(
        "HTTP Camera",
        source_type=CameraSourceType.HTTP_SNAPSHOT,
        source_config={"uri": "https://example.com/cam.jpg"},
    )
    controller = CameraController(camera_config)

    with patch(
        "rayforge.camera.controller.create_camera_source",
        return_value=MockSource(),
    ):
        controller.capture_image()

    assert controller.image_data is not None
    assert controller.image_data.shape == (10, 20, 3)


def test_http_snapshot_failed_fetch_respects_offline_retry_interval():
    camera = Camera(
        "HTTP Camera",
        source_type=CameraSourceType.HTTP_SNAPSHOT,
        source_config={"uri": "https://example.com/cam.jpg"},
    )
    source = HttpSnapshotSource(camera)
    source._last_success_frame = np.zeros((4, 4, 3), dtype=np.uint8)

    time_values = iter([100.0, 100.5, 110.0, 111.0, 120.0])
    call_count = 0
    wait_times = []

    def fake_urlopen(request, timeout):
        nonlocal call_count
        call_count += 1
        raise ConnectionRefusedError("offline")

    def fake_wait(timeout):
        wait_times.append(timeout)
        return False

    with (
        patch(
            "rayforge.camera.source.time.monotonic", side_effect=time_values
        ),
        patch.object(source, "_wait_or_cancelled", side_effect=fake_wait),
        patch(
            "rayforge.camera.source.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ),
    ):
        assert source.read_frame() is not None
        assert source.read_frame() is not None
        assert source.read_frame() is not None

    assert call_count == 3
    assert wait_times == [9.5, 9.0]


def test_http_snapshot_waits_between_successful_polls():
    camera = Camera(
        "HTTP Camera",
        source_type=CameraSourceType.HTTP_SNAPSHOT,
        source_config={"uri": "https://example.com/cam.jpg"},
    )
    source = HttpSnapshotSource(camera)
    _, encoded = cv2.imencode(".jpg", np.zeros((4, 4, 3), dtype=np.uint8))
    payload = encoded.tobytes()
    time_values = iter([100.0, 100.5, 102.0])
    wait_times = []
    call_count = 0

    response = SimpleNamespace(
        headers=SimpleNamespace(get_content_type=lambda: "image/jpeg"),
        read=lambda size: payload,
    )

    class ResponseContext:
        def __enter__(self):
            return response

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout):
        nonlocal call_count
        call_count += 1
        return ResponseContext()

    def fake_wait(timeout):
        wait_times.append(timeout)
        return False

    with (
        patch(
            "rayforge.camera.source.time.monotonic", side_effect=time_values
        ),
        patch.object(source, "_wait_or_cancelled", side_effect=fake_wait),
        patch(
            "rayforge.camera.source.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ),
    ):
        assert source.read_frame() is not None
        assert source.read_frame() is not None

    assert call_count == 2
    assert wait_times == [1.5]


def test_http_snapshot_rejects_oversized_payloads():
    camera = Camera(
        "HTTP Camera",
        source_type=CameraSourceType.HTTP_SNAPSHOT,
        source_config={"uri": "https://example.com/cam.jpg"},
    )
    source = HttpSnapshotSource(camera)
    source._last_success_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    read_sizes = []
    logged_messages = []

    response = SimpleNamespace(
        headers=SimpleNamespace(get_content_type=lambda: "image/jpeg"),
        read=lambda size: read_sizes.append(size) or b"x" * size,
    )

    class ResponseContext:
        def __enter__(self):
            return response

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_warning(message, *args):
        logged_messages.append(message % args if args else message)

    with (
        patch("rayforge.camera.source.time.monotonic", return_value=100.0),
        patch(
            "rayforge.camera.source.urllib.request.urlopen",
            return_value=ResponseContext(),
        ),
        patch(
            "rayforge.camera.source.network.logger.warning",
            side_effect=fake_warning,
        ),
    ):
        frame = source.read_frame()

    assert frame is not None
    assert read_sizes == [HttpSnapshotSource.MAX_SNAPSHOT_BYTES + 1]
    assert source._next_poll_interval == source.OFFLINE_RETRY_INTERVAL_SECONDS
    assert "larger than" in logged_messages[0]


def test_http_snapshot_warning_logs_are_throttled():
    camera = Camera(
        "HTTP Camera",
        source_type=CameraSourceType.HTTP_SNAPSHOT,
        source_config={"uri": "https://example.com/cam.jpg"},
    )
    source = HttpSnapshotSource(camera)
    source._last_success_frame = np.zeros((4, 4, 3), dtype=np.uint8)

    time_values = iter([100.0, 131.0])
    response = SimpleNamespace(
        headers=SimpleNamespace(get_content_type=lambda: "application/json"),
        read=lambda: b'{"busy": true}',
    )
    logged_messages = []

    class ResponseContext:
        def __enter__(self):
            return response

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_warning(message, *args):
        logged_messages.append(message % args if args else message)

    with (
        patch(
            "rayforge.camera.source.time.monotonic", side_effect=time_values
        ),
        patch(
            "rayforge.camera.source.urllib.request.urlopen",
            return_value=ResponseContext(),
        ),
        patch(
            "rayforge.camera.source.network.logger.warning",
            side_effect=fake_warning,
        ),
    ):
        assert source.read_frame() is not None
        assert source.read_frame() is not None

    assert len(logged_messages) == 2


def test_network_streams_use_slower_reconnect_delay():
    camera = Camera(
        "HTTP Stream",
        source_type=CameraSourceType.HTTP_STREAM,
        source_config={"uri": "https://example.com/stream.mjpeg"},
    )
    controller = CameraController(camera)

    class FailingSource:
        reconnect_delay_seconds = 10.0
        warning_log_interval_seconds = 30.0

        def open(self):
            raise OSError("offline")

        def close(self):
            return

    wait_calls = []

    def fake_wait(delay):
        wait_calls.append(delay)
        controller._running = False
        return True

    with (
        patch(
            "rayforge.camera.controller.create_camera_source",
            return_value=FailingSource(),
        ),
        patch.object(controller._stop_event, "wait", side_effect=fake_wait),
    ):
        controller._running = True
        controller._capture_loop()

    assert wait_calls == [10.0]


def test_http_stream_source_reconnects_after_repeated_read_failures():
    camera = Camera(
        "HTTP Stream",
        source_type=CameraSourceType.HTTP_STREAM,
        source_config={"uri": "https://example.com/stream.mjpeg"},
    )

    class MockVideoCapture:
        def __init__(self, uri, backend=None):
            self.opened = True

        def isOpened(self):
            return self.opened

        def read(self):
            return False, None

        def release(self):
            self.opened = False

    with patch("rayforge.camera.source.cv2.VideoCapture", MockVideoCapture):
        source = HttpStreamSource(camera)
        source.open()

        assert source.read_frame() is None
        assert source.read_frame() is None
        with pytest.raises(OSError, match="stopped returning frames"):
            source.read_frame()
        source.close()


def test_http_stream_source_resets_read_failures_after_success():
    camera = Camera(
        "HTTP Stream",
        source_type=CameraSourceType.HTTP_STREAM,
        source_config={"uri": "https://example.com/stream.mjpeg"},
    )
    reads = iter(
        [
            (False, None),
            (False, None),
            (True, np.zeros((2, 2, 3), dtype=np.uint8)),
            (False, None),
            (False, None),
        ]
    )

    class MockVideoCapture:
        def __init__(self, uri, backend=None):
            self.opened = True

        def isOpened(self):
            return self.opened

        def read(self):
            return next(reads)

        def release(self):
            self.opened = False

    with patch("rayforge.camera.source.cv2.VideoCapture", MockVideoCapture):
        source = HttpStreamSource(camera)
        source.open()

        assert source.read_frame() is None
        assert source.read_frame() is None
        assert source.read_frame() is not None
        assert source.read_frame() is None
        assert source.read_frame() is None
        source.close()


def test_network_stream_frame_failure_logs_are_throttled():
    camera = Camera(
        "HTTP Stream",
        source_type=CameraSourceType.HTTP_STREAM,
        source_config={"uri": "https://example.com/stream.mjpeg"},
    )
    controller = CameraController(camera)
    controller._running = True

    class FailingSource:
        warning_log_interval_seconds = 30.0

        def apply_settings(self):
            return

        def read_frame(self):
            return None

    warning_messages = []
    time_values = iter([100.0, 100.0, 101.0, 102.0, 103.0])

    def fake_warning(message, *args):
        warning_messages.append(message % args if args else message)

    def fake_wait(delay):
        controller._running = False
        return True

    with (
        patch(
            "rayforge.camera.controller.time.monotonic",
            side_effect=time_values,
        ),
        patch(
            "rayforge.camera.controller.logger.warning",
            side_effect=fake_warning,
        ),
        patch.object(controller._stop_event, "wait", side_effect=fake_wait),
    ):
        controller._capture_frames_from_source(FailingSource())

    assert warning_messages == ["Frame failure 1/10 for HTTP Stream"]


def test_stop_capture_stream_stops_active_source_before_join():
    camera = Camera("Test Camera", "0")
    controller = CameraController(camera)
    stop_calls = []
    join_timeouts = []

    class ActiveSource(CameraSource):
        def open(self):
            return

        def close(self):
            return

        def read_frame(self):
            return None

        def stop(self):
            stop_calls.append("stopped")

    class MockThread(threading.Thread):
        def is_alive(self):
            return True

        def join(self, timeout=None):
            join_timeouts.append(timeout)

    controller._running = True
    controller._active_source = ActiveSource(camera)
    controller._capture_thread = MockThread()

    controller._stop_capture_stream()

    assert stop_calls == ["stopped"]
    # The mock thread never reports itself as dead, so `_stop_locked`
    # must force-close the source and join a second time before giving
    # up and marking the controller permanently stuck.
    assert join_timeouts == [2.0, 2.0]
    assert controller._thread_stuck is True


def test_device_id_change_restarts_capture_stream_while_running():
    """Switching the local camera device must close the old device and
    open the new one, instead of silently continuing to read from the
    already-open (stale) device or leaving it open forever.
    """
    camera = Camera("Test Camera", "old-device")
    camera.enabled = True
    controller = CameraController(camera)

    calls = []

    def fake_stop():
        calls.append("stop")

    def fake_start():
        calls.append("start")

    controller._active_subscribers = 1
    controller._running = True
    with (
        patch.object(controller, "_stop_capture_stream", fake_stop),
        patch.object(controller, "_start_capture_stream", fake_start),
    ):
        camera.device_id = "new-device"

    # Camera.device_id changes emit both `changed` and `settings_changed`,
    # so `_start_capture_stream` may be invoked more than once, but the
    # old device must be stopped exactly once, and that stop must happen
    # before the stream is (re)started.
    assert calls.count("stop") == 1
    assert calls[:2] == ["stop", "start"]


def test_unrelated_setting_change_does_not_restart_capture_stream():
    """Changing a setting unrelated to the physical source (e.g. name)
    must not tear down and recreate the running capture stream.
    """
    camera = Camera("Test Camera", "same-device")
    camera.enabled = True
    controller = CameraController(camera)

    calls = []

    def fake_stop():
        calls.append("stop")

    def fake_start():
        calls.append("start")

    controller._active_subscribers = 1
    controller._running = True
    with (
        patch.object(controller, "_stop_capture_stream", fake_stop),
        patch.object(controller, "_start_capture_stream", fake_start),
    ):
        camera.name = "Renamed Camera"

    assert calls == ["start"]


def test_local_device_source_sets_read_timeout_when_supported():
    camera = Camera("Local Camera", "0")
    set_calls = []

    class MockVideoCapture:
        def __init__(self, device, backend=None):
            self.opened = True

        def isOpened(self):
            return self.opened

        def read(self):
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def set(self, prop_id, value):
            set_calls.append((prop_id, value))
            return True

        def release(self):
            self.opened = False

    with patch("rayforge.camera.source.cv2.VideoCapture", MockVideoCapture):
        from rayforge.camera.source import LocalDeviceSource

        source = LocalDeviceSource(camera)
        source.open()
        source.close()

    assert (
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        LocalDeviceSource.READ_TIMEOUT_MSEC,
    ) in set_calls


def test_local_device_source_reconnects_after_repeated_read_failures():
    camera = Camera("Local Camera", "0")

    class MockVideoCapture:
        def __init__(self, device, backend=None):
            self.opened = True
            self._read_calls = 0

        def isOpened(self):
            return self.opened

        def set(self, prop_id, value):
            return True

        def release(self):
            self.opened = False

        def read(self):
            # The first read() is consumed by open()'s initial-frame
            # validation and must succeed; subsequent reads simulate the
            # device dying afterwards, which is what this test covers.
            self._read_calls += 1
            if self._read_calls == 1:
                return True, np.zeros((2, 2, 3), dtype=np.uint8)
            return False, None

    with patch("rayforge.camera.source.cv2.VideoCapture", MockVideoCapture):
        from rayforge.camera.source import LocalDeviceSource

        source = LocalDeviceSource(camera)
        source.open()

        assert source.read_frame() is None
        assert source.read_frame() is None
        with pytest.raises(OSError, match="stopped returning frames"):
            source.read_frame()
        source.close()


def test_local_device_source_skips_failing_backend_after_read_timeouts():
    camera = Camera("Local Camera", "0")
    opened_backends = []

    class MockVideoCapture:
        def __init__(self, device, backend=None):
            self.backend = backend
            self.opened = True
            self._read_calls = 0
            opened_backends.append(backend)

        def isOpened(self):
            return self.opened

        def set(self, prop_id, value):
            return True

        def release(self):
            self.opened = False

        def read(self):
            # The first read() is consumed by open()'s initial-frame
            # validation and must succeed; subsequent reads simulate the
            # device dying afterwards, which is what this test covers.
            self._read_calls += 1
            if self._read_calls == 1:
                return True, np.zeros((2, 2, 3), dtype=np.uint8)
            return False, None

    with (
        patch("rayforge.camera.source.cv2.VideoCapture", MockVideoCapture),
        patch(
            "rayforge.camera.source.get_backends_for_platform",
            return_value=[(11, "V4L2"), (22, "default")],
        ),
    ):
        from rayforge.camera.source import LocalDeviceSource

        source = LocalDeviceSource(camera)
        source.open()
        assert source._backend_used == "V4L2"
        assert source.read_frame() is None
        assert source.read_frame() is None
        with pytest.raises(OSError, match="stopped returning frames"):
            source.read_frame()
        source.close()
        source.open()
        source.close()

    assert opened_backends[0] == 11
    assert opened_backends[-1] == 22


def test_local_device_source_prefers_mjpg_when_yuyv_not_requested():
    camera = Camera("Local Camera", "0")
    camera.resolution = (640, 480)
    set_calls = []

    class MockVideoCapture:
        def __init__(self, device, backend=None):
            self.opened = True

        def isOpened(self):
            return self.opened

        def read(self):
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def set(self, prop_id, value):
            set_calls.append((prop_id, value))
            return True

        def release(self):
            self.opened = False

    with patch("rayforge.camera.source.cv2.VideoCapture", MockVideoCapture):
        from rayforge.camera.source import LocalDeviceSource

        source = LocalDeviceSource(camera)
        source.open()
        source.apply_settings()
        source.close()

    assert set_calls[1] == (
        cv2.CAP_PROP_FOURCC,
        LocalDeviceSource.MJPG_FOURCC,
    )
    assert set_calls[2] == (cv2.CAP_PROP_BUFFERSIZE, 1)
    assert set_calls[3] == (cv2.CAP_PROP_FRAME_WIDTH, 640)
    assert set_calls[4] == (cv2.CAP_PROP_FRAME_HEIGHT, 480)


def test_local_device_source_uses_yuyv_when_requested():
    camera = Camera("Local Camera", "0")
    camera.prefer_yuyv = True
    set_calls = []

    class MockVideoCapture:
        def __init__(self, device, backend=None):
            self.opened = True

        def isOpened(self):
            return self.opened

        def read(self):
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def set(self, prop_id, value):
            set_calls.append((prop_id, value))
            return True

        def release(self):
            self.opened = False

    with patch("rayforge.camera.source.cv2.VideoCapture", MockVideoCapture):
        from rayforge.camera.source import LocalDeviceSource

        source = LocalDeviceSource(camera)
        source.open()
        source.apply_settings()
        source.close()

    assert (
        cv2.CAP_PROP_FOURCC,
        LocalDeviceSource.YUYV_FOURCC,
    ) in set_calls


def test_local_device_source_reads_current_image_settings():
    camera = Camera("Local Camera", "0")

    class MockVideoCapture:
        def __init__(self, device, backend=None):
            self.opened = True

        def isOpened(self):
            return self.opened

        def read(self):
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def set(self, prop_id, value):
            return True

        def get(self, prop_id):
            values = {
                cv2.CAP_PROP_CONTRAST: 17.5,
                cv2.CAP_PROP_BRIGHTNESS: -4.0,
                cv2.CAP_PROP_AUTO_WB: 0.0,
                cv2.CAP_PROP_WB_TEMPERATURE: 5100.0,
            }
            return values[prop_id]

        def release(self):
            self.opened = False

    with patch("rayforge.camera.source.cv2.VideoCapture", MockVideoCapture):
        from rayforge.camera.source import LocalDeviceSource

        source = LocalDeviceSource(camera)
        source.open()
        settings = source.read_current_settings()
        source.close()

    assert settings == {
        "contrast": 17.5,
        "brightness": -4.0,
        "white_balance": 5100.0,
    }


def test_local_device_source_reads_auto_white_balance_as_none():
    camera = Camera("Local Camera", "0")

    class MockVideoCapture:
        def __init__(self, device, backend=None):
            self.opened = True

        def isOpened(self):
            return self.opened

        def read(self):
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def set(self, prop_id, value):
            return True

        def get(self, prop_id):
            values = {
                cv2.CAP_PROP_CONTRAST: 42.0,
                cv2.CAP_PROP_BRIGHTNESS: 3.0,
                cv2.CAP_PROP_AUTO_WB: 1.0,
                cv2.CAP_PROP_WB_TEMPERATURE: 5100.0,
            }
            return values[prop_id]

        def release(self):
            self.opened = False

    with patch("rayforge.camera.source.cv2.VideoCapture", MockVideoCapture):
        from rayforge.camera.source import LocalDeviceSource

        source = LocalDeviceSource(camera)
        source.open()
        settings = source.read_current_settings()
        source.close()

    assert settings == {
        "contrast": 42.0,
        "brightness": 3.0,
        "white_balance": None,
    }


def test_local_device_source_omits_unsupported_white_balance():
    camera = Camera("Local Camera", "0")

    class MockVideoCapture:
        def __init__(self, device, backend=None):
            self.opened = True

        def isOpened(self):
            return self.opened

        def read(self):
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def set(self, prop_id, value):
            return True

        def get(self, prop_id):
            values = {
                cv2.CAP_PROP_CONTRAST: 42.0,
                cv2.CAP_PROP_BRIGHTNESS: 3.0,
                cv2.CAP_PROP_AUTO_WB: 0.0,
                cv2.CAP_PROP_WB_TEMPERATURE: 0.0,
            }
            return values[prop_id]

        def release(self):
            self.opened = False

    with patch("rayforge.camera.source.cv2.VideoCapture", MockVideoCapture):
        from rayforge.camera.source import LocalDeviceSource

        source = LocalDeviceSource(camera)
        source.open()
        settings = source.read_current_settings()
        source.close()

    assert settings == {
        "contrast": 42.0,
        "brightness": 3.0,
    }


def test_start_capture_stream_refuses_when_thread_stuck():
    """Once a capture thread has been marked stuck, no new capture thread
    may be started, even if the caller retries. Starting a second thread
    while the old one might still be holding the device open risks
    opening the same hardware twice.
    """
    camera = Camera("Test Camera", "0")
    controller = CameraController(camera)
    controller._thread_stuck = True

    controller._start_capture_stream()

    assert controller._capture_thread is None
    assert controller._running is False


def test_stop_capture_stream_marks_stuck_thread_and_blocks_future_starts():
    """A capture thread that never terminates, even after a forced source
    close, must permanently disable further starts on that controller
    instead of silently allowing a second thread to open the device.
    """
    camera = Camera("Test Camera", "0")
    controller = CameraController(camera)

    class NeverDyingThread(threading.Thread):
        def is_alive(self):
            return True

        def join(self, timeout=None):
            return

    controller._running = True
    controller._active_source = None
    controller._capture_thread = NeverDyingThread()

    controller._stop_capture_stream()

    assert controller._thread_stuck is True

    controller._start_capture_stream()

    assert controller._running is False


def test_dispose_disconnects_config_signals_and_stops_stream():
    """dispose() must fully detach the controller from its camera config
    so a config mutation after teardown can never resurrect it, and must
    be safe to call more than once.
    """
    camera = Camera("Test Camera", "0")
    controller = CameraController(camera)

    stop_calls = []
    controller._stop_capture_stream = lambda: stop_calls.append("stop")

    controller.dispose()

    assert stop_calls == ["stop"]
    assert controller._disposed is True

    # A config change after dispose() must not be able to restart the
    # stream or otherwise reactivate the controller.
    start_calls = []
    controller._start_capture_stream = lambda: start_calls.append("start")
    camera.device_id = "1"

    assert start_calls == []

    # Calling dispose() again must be a no-op, not raise or double-stop.
    controller.dispose()
    assert stop_calls == ["stop"]


def test_subscribe_after_dispose_is_a_noop():
    camera = Camera("Test Camera", "0")
    controller = CameraController(camera)
    controller._stop_capture_stream = lambda: None
    controller.dispose()

    start_calls = []
    controller._start_capture_stream = lambda: start_calls.append("start")

    controller.subscribe()

    assert start_calls == []
    assert controller._active_subscribers == 0


def _aligned_gradient_controller() -> CameraController:
    """An aligned controller whose frame encodes pixel positions.

    The alignment maps 4 px/mm in x and 3 px/mm in y, so the 640x480
    frame covers world x in [-25, 135] and y in about [-33.3, 126.7].
    """
    camera_config = Camera("Test Camera", "0")
    controller = CameraController(camera_config)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, 0] = (np.arange(640) * 255 // 639)[np.newaxis, :]
    image[:, :, 1] = (np.arange(480) * 255 // 479)[:, np.newaxis]
    controller._image_data = image
    camera_config.image_to_world = (
        [(100, 100), (500, 100), (500, 400), (100, 400)],
        [(0, 100), (100, 100), (100, 0), (0, 0)],
    )
    return controller


WORKSPACE = ((0, 0), (100, 100))


def test_get_work_surface_images_workspace_matches_single_render():
    controller = _aligned_gradient_controller()

    workspace, outside = controller.get_work_surface_images(
        (200, 200), WORKSPACE, 20, 2048
    )

    expected = controller.get_work_surface_image((200, 200), WORKSPACE)
    assert workspace is not None and expected is not None
    np.testing.assert_array_equal(workspace, expected)
    assert outside is not None


def test_outside_view_area_size_and_world_coordinates():
    controller = _aligned_gradient_controller()

    _, outside = controller.get_work_surface_images(
        (200, 200), WORKSPACE, 20, 2048
    )

    assert outside is not None
    assert outside.margin_mm == 20
    assert outside.area == ((-20, -20), (120, 120))
    # Rendered at the camera's density of 4 px/mm, the sharper axis.
    assert outside.image.shape == (560, 560, 4)

    # Every covered pixel, including all edges and corners at negative
    # coordinates, matches the existing transform of the expanded area.
    reference = controller.get_work_surface_image((560, 560), outside.area)
    assert reference is not None
    alpha = outside.image[:, :, 3]
    covered = alpha == 255
    for row, col in [(0, 0), (0, -1), (-1, 0), (-1, -1), (280, 0)]:
        assert covered[row, col]
    np.testing.assert_array_equal(
        outside.image[:, :, :3][covered], reference[covered]
    )


def test_outside_view_margin_is_clamped_to_camera_coverage():
    controller = _aligned_gradient_controller()

    _, outside = controller.get_work_surface_images(
        (200, 200), WORKSPACE, 100, 2048
    )

    assert outside is not None
    # The frame reaches 35 mm beyond the right edge of the workspace.
    assert outside.margin_mm == pytest.approx(35.0)
    assert controller.config.outside_view_margin_mm == 30.0


def test_outside_view_missing_coverage_is_transparent():
    controller = _aligned_gradient_controller()
    controller._image_data = np.zeros((480, 640, 3), dtype=np.uint8)

    _, outside = controller.get_work_surface_images(
        (200, 200), WORKSPACE, 100, 2048
    )

    assert outside is not None
    image = outside.image
    # Left of x=-25 mm the camera sees nothing: fully transparent.
    assert image[:, :10, 3].max() == 0
    # Genuine black pixels inside the frame stay opaque.
    assert image[170, 170, 3] == 255
    np.testing.assert_array_equal(image[170, 170, :3], [0, 0, 0])


def test_outside_view_is_premultiplied_with_partial_edges():
    controller = _aligned_gradient_controller()
    controller._image_data = np.full((480, 640, 3), 255, dtype=np.uint8)

    # A non-integer density puts the frame edges between pixels.
    _, outside = controller.get_work_surface_images(
        (233, 233), WORKSPACE, 100, 2048
    )

    assert outside is not None
    alpha = outside.image[:, :, 3]
    assert ((alpha > 0) & (alpha < 255)).any()
    assert (outside.image[:, :, :3] <= alpha[:, :, np.newaxis]).all()


def test_outside_view_none_without_alignment_or_margin():
    controller = _aligned_gradient_controller()

    _, outside = controller.get_work_surface_images(
        (200, 200), WORKSPACE, 0, 2048
    )
    assert outside is None

    controller.config.image_to_world = None
    workspace, outside = controller.get_work_surface_images(
        (200, 200), WORKSPACE, 20, 2048
    )
    assert outside is None
    assert workspace is not None


def test_outside_view_none_without_coverage_beyond_workspace():
    controller = _aligned_gradient_controller()

    _, outside = controller.get_work_surface_images(
        (200, 200), ((-50, -50), (150, 150)), 20, 2048
    )

    assert outside is None


def test_outside_view_size_is_capped():
    controller = _aligned_gradient_controller()

    _, outside = controller.get_work_surface_images(
        (200, 200), WORKSPACE, 20, 400
    )

    assert outside is not None
    assert outside.image.shape[:2] == (400, 400)


def test_get_work_surface_images_uses_a_single_frame():
    controller = _aligned_gradient_controller()
    controller._image_data = np.full((480, 640, 3), 200, dtype=np.uint8)
    newer_frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    original = controller._work_surface_from_image

    def replace_frame_midway(*args):
        result = original(*args)
        with controller._frame_lock:
            controller._image_data = newer_frame
        return result

    with patch.object(
        controller, "_work_surface_from_image", replace_frame_midway
    ):
        workspace, outside = controller.get_work_surface_images(
            (200, 200), WORKSPACE, 20, 2048
        )

    assert workspace is not None and outside is not None
    assert (workspace == 200).all()
    covered = outside.image[:, :, 3] == 255
    assert (outside.image[:, :, :3][covered] == 200).all()


def _aligned_lens_controller(k1: float = -0.3) -> CameraController:
    """An aligned gradient controller with barrel lens correction.

    The raw frame is the gradient; the processed frame is its lens
    correction, which pushes the frame edges out of the image.
    """
    controller = _aligned_gradient_controller()
    controller.config.distortion_k1 = k1
    raw = controller._image_data
    assert raw is not None
    controller._raw_image_data = raw
    controller._image_data = controller._process_frame(raw)
    return controller


def test_outside_view_recovers_content_cropped_by_lens_correction():
    controller = _aligned_lens_controller()
    image = controller._image_data
    assert image is not None
    H = controller._compute_homography(image.shape[0])
    corrected_extent = frame_coverage_extent(H, image.shape, WORKSPACE)

    _, outside = controller.get_work_surface_images(
        (200, 200), WORKSPACE, 100, 2048
    )

    assert outside is not None
    assert outside.margin_mm > corrected_extent + 5


def test_outside_view_from_raw_matches_corrected_frame():
    controller = _aligned_lens_controller()
    image = controller._image_data
    assert image is not None

    _, outside = controller.get_work_surface_images(
        (200, 200), WORKSPACE, 100, 2048
    )

    assert outside is not None
    size = outside.image.shape[1::-1]
    H = controller._compute_homography(image.shape[0])
    reference = warp_transparent(to_opaque_bgra(image), H, size, outside.area)
    both = (reference[:, :, 3] == 255) & (outside.image[:, :, 3] == 255)
    assert both.mean() > 0.3
    difference = np.abs(
        reference[:, :, :3][both].astype(int)
        - outside.image[:, :, :3][both].astype(int)
    )
    assert difference.mean() < 2


def test_world_to_raw_inverts_lens_correction():
    controller = _aligned_lens_controller()
    image = controller._image_data
    assert image is not None
    lens = controller._lens_model(image)
    assert lens is not None
    corrected = np.array(
        [[320.0, 240.0], [50.0, 60.0], [600.0, 420.0], [10.0, 470.0]]
    )

    raw_x, raw_y, valid = world_to_raw(
        np.eye(3), lens, corrected[:, 0], corrected[:, 1]
    )

    assert valid.all()
    raw = np.stack([raw_x, raw_y], axis=1).reshape(-1, 1, 2)
    back = cv2.undistortPoints(
        raw, lens.camera_matrix, lens.distortion, P=lens.camera_matrix
    ).reshape(-1, 2)
    np.testing.assert_allclose(back, corrected, atol=0.05)


def test_radial_limit_excludes_folding_distortion():
    assert radial_limit(np.zeros(5)) == float("inf")

    distortion = np.array([-0.98, 1.75, 0.0, 0.0, -1.75])
    limit = radial_limit(distortion)

    assert 0.6 < limit < 0.8
    lens = LensModel(np.eye(3), distortion)
    _, _, valid = world_to_raw(
        np.eye(3),
        lens,
        np.array([0.5 * limit, 1.1 * limit]),
        np.zeros(2),
    )
    np.testing.assert_array_equal(valid, [True, False])


def test_outside_view_lens_maps_are_cached_across_frames():
    controller = _aligned_lens_controller()

    with patch(
        "rayforge.camera.controller.lens_remap_maps",
        wraps=lens_remap_maps,
    ) as build_maps:
        for value in (100, 150):
            raw = np.full((480, 640, 3), value, dtype=np.uint8)
            controller._raw_image_data = raw
            controller._image_data = controller._process_frame(raw)
            _, outside = controller.get_work_surface_images(
                (200, 200), WORKSPACE, 30, 2048
            )
            assert outside is not None

        controller.config.outside_view_margin_mm = 10
        controller.get_work_surface_images((200, 200), WORKSPACE, 10, 2048)

    assert build_maps.call_count == 2


def test_get_work_surface_images_uses_a_single_raw_frame():
    controller = _aligned_lens_controller()
    controller._raw_image_data = np.full((480, 640, 3), 200, np.uint8)
    controller._image_data = np.full((480, 640, 3), 200, np.uint8)
    newer_frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    original = controller._work_surface_from_image

    def replace_frames_midway(*args):
        result = original(*args)
        with controller._frame_lock:
            controller._image_data = newer_frame
            controller._raw_image_data = newer_frame
        return result

    with patch.object(
        controller, "_work_surface_from_image", replace_frames_midway
    ):
        workspace, outside = controller.get_work_surface_images(
            (200, 200), WORKSPACE, 20, 2048
        )

    assert workspace is not None and outside is not None
    assert (workspace == 200).all()
    covered = outside.image[:, :, 3] == 255
    assert covered.any()
    assert (outside.image[:, :, :3][covered] == 200).all()


def test_outside_view_does_not_depend_on_display_size():
    controller = _aligned_lens_controller()

    with patch(
        "rayforge.camera.controller.lens_remap_maps",
        wraps=lens_remap_maps,
    ) as build_maps:
        shapes = set()
        for output_size in [(200, 200), (800, 600), (1203, 777)]:
            _, outside = controller.get_work_surface_images(
                output_size, WORKSPACE, 20, 2048
            )
            assert outside is not None
            shapes.add(outside.image.shape)

    assert len(shapes) == 1
    assert build_maps.call_count == 1


def test_disabling_outside_view_frees_cached_maps():
    controller = _aligned_lens_controller()
    controller.config.outside_view_enabled = True
    controller.get_work_surface_images((200, 200), WORKSPACE, 20, 2048)
    assert controller._outside_cache

    controller.config.outside_view_enabled = False

    assert not controller._outside_cache
