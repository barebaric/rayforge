import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from rayforge.camera.v4l import (
    _is_video_capture_symlink,
    display_name,
    friendly_name_from_by_id,
    migrate_camera_data,
    resolve_device_id,
    scan_v4l_by_id,
)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
class TestIsVideoCaptureSymlink:
    def test_video_index_entry(self):
        assert _is_video_capture_symlink(
            "usb-046d_Logitech_C930e_1234-video-index0"
        )

    def test_non_video_entry(self):
        assert not _is_video_capture_symlink(
            "usb-046d_Logitech_C930e_1234-input0"
        )

    def test_alt_entry(self):
        assert not _is_video_capture_symlink(
            "usb-046d_Logitech_C930e_1234-alt0"
        )

    def test_plain_name(self):
        assert not _is_video_capture_symlink("some_random_name")


class TestFriendlyNameFromById:
    def test_usb_webcam_with_vendor_id(self):
        path = (
            "/dev/v4l/by-id/"
            "usb-046d_Logitech_Webcam_C930e_1234567890"
            "-video-index0"
        )
        assert friendly_name_from_by_id(path) == "Logitech Webcam C930e"

    def test_usb_webcam_with_inc(self):
        path = (
            "/dev/v4l/by-id/"
            "usb-1d6b_SunplusIT_Inc_Integrated_Camera_0001"
            "-video-index0"
        )
        result = friendly_name_from_by_id(path)
        assert result == "SunplusIT Inc Integrated Camera"

    def test_pci_device(self):
        path = "/dev/v4l/by-id/pci-Vendor_Product_123-video-index0"
        assert friendly_name_from_by_id(path) == "Product"

    def test_two_parts_vendor_and_serial(self):
        path = "/dev/v4l/by-id/usb-046d_C930e_1234-video-index0"
        assert friendly_name_from_by_id(path) == "C930e"

    def test_single_part(self):
        path = "/dev/v4l/by-id/usb-ProductName-video-index0"
        assert friendly_name_from_by_id(path) == "ProductName"


class TestScanV4lById:
    @pytest.mark.skipif(
        not sys.platform.startswith("linux"), reason="Linux only"
    )
    def test_nonexistent_dir(self):
        with mock.patch(
            "rayforge.camera.v4l.V4L_BY_ID_DIR",
            Path("/nonexistent_dir_for_testing"),
        ):
            assert scan_v4l_by_id() == {}

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"), reason="Linux only"
    )
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "rayforge.camera.v4l.V4L_BY_ID_DIR",
                Path(tmpdir),
            ):
                assert scan_v4l_by_id() == {}


class TestResolveDeviceId:
    def test_non_linux_returns_unchanged(self):
        with mock.patch("rayforge.camera.v4l.sys.platform", "win32"):
            assert resolve_device_id("0") == "0"

    def test_by_id_path_returns_unchanged(self):
        path = "/dev/v4l/by-id/usb-Vendor_Product-video-index0"
        with mock.patch("rayforge.camera.v4l.sys.platform", "linux"):
            assert resolve_device_id(path) == path

    def test_string_id_returns_unchanged(self):
        with mock.patch("rayforge.camera.v4l.sys.platform", "linux"):
            assert resolve_device_id("some_string") == "some_string"

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"), reason="Linux only"
    )
    def test_numeric_without_by_id_keeps_numeric(self):
        with mock.patch(
            "rayforge.camera.v4l.V4L_BY_ID_DIR",
            Path("/nonexistent_dir_for_testing"),
        ):
            assert resolve_device_id("5") == "5"


class TestDisplayName:
    def test_by_id_path_returns_friendly_name(self):
        path = (
            "/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_1234-video-index0"
        )
        result = display_name(path)
        assert result == "Logitech Webcam C930e"

    def test_numeric_id_returns_as_is(self):
        assert display_name("0") == "0"

    def test_string_id_returns_as_is(self):
        assert display_name("some_device") == "some_device"


class TestMigrateCameraData:
    def test_no_device_id_returns_unchanged(self):
        data = {"name": "cam"}
        assert migrate_camera_data(data) == {"name": "cam"}

    def test_non_linux_returns_unchanged(self):
        with mock.patch("rayforge.camera.v4l.sys.platform", "win32"):
            data = {"name": "cam", "device_id": "0"}
            result = migrate_camera_data(data)
            assert result["device_id"] == "0"

    def test_by_id_path_returns_unchanged(self):
        path = "/dev/v4l/by-id/usb-Vendor_Product-video-index0"
        data = {"name": "cam", "device_id": path}
        result = migrate_camera_data(data)
        assert result["device_id"] == path

    def test_does_not_mutate_input(self):
        data = {"name": "cam", "device_id": "0"}
        with mock.patch("rayforge.camera.v4l.sys.platform", "win32"):
            migrate_camera_data(data)
        assert data["device_id"] == "0"
