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
from rayforge.camera.source import CameraSource, HttpSnapshotSource


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
    from rayforge.camera.controller import _to_videocapture_arg

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

    time_values = iter([100.0, 100.5, 111.0])
    call_count = 0

    def fake_urlopen(request, timeout):
        nonlocal call_count
        call_count += 1
        raise ConnectionRefusedError("offline")

    with (
        patch(
            "rayforge.camera.source.time.monotonic", side_effect=time_values
        ),
        patch(
            "rayforge.camera.source.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ),
    ):
        assert source.read_frame() is not None
        assert source.read_frame() is not None
        assert source.read_frame() is not None

    assert call_count == 2


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
            "rayforge.camera.source.logger.warning", side_effect=fake_warning
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
