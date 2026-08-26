"""Tests for the wizard's hardware permission pre-flight checks."""

import os

import pytest

from rayforge.shared.util.permissions import (
    PermissionIssue,
    _device_group_name,
    check_camera_permissions,
    check_permissions,
    check_serial_permissions,
)

_SERIAL_LIST_TARGET = "rayforge.shared.util.permissions._list_usb_serial_ports"
_CAMERA_LIST_TARGET = "rayforge.shared.util.permissions._list_v4l_devices"
_ACCESS_TARGET = "rayforge.shared.util.permissions.os.access"
_GROUP_TARGET = "rayforge.shared.util.permissions._device_group_name"


@pytest.fixture
def linux(mocker):
    mocker.patch("rayforge.shared.util.permissions.sys.platform", "linux")
    mocker.patch("os.environ", {})


class TestCheckSerialPermissions:
    @pytest.mark.parametrize("platform", ["win32", "darwin"])
    def test_non_linux_never_reports(self, mocker, platform):
        mocker.patch("rayforge.shared.util.permissions.sys.platform", platform)
        assert check_serial_permissions() is None

    def test_no_ports_is_not_a_permission_problem(self, linux, mocker):
        mocker.patch(_SERIAL_LIST_TARGET, return_value=[])
        assert check_serial_permissions() is None

    def test_accessible_port_passes(self, linux, mocker):
        mocker.patch(_SERIAL_LIST_TARGET, return_value=["/dev/ttyUSB0"])
        mocker.patch(_ACCESS_TARGET, return_value=True)
        assert check_serial_permissions() is None

    def test_reports_derived_owner_group(self, linux, mocker):
        """The usermod command must name the group that actually owns
        the device node (uucp on Arch etc.), not a hardcoded distro."""
        mocker.patch(_SERIAL_LIST_TARGET, return_value=["/dev/ttyUSB0"])
        mocker.patch(_ACCESS_TARGET, side_effect=[False])
        mocker.patch(_GROUP_TARGET, return_value="uucp")
        issue = check_serial_permissions()
        assert isinstance(issue, PermissionIssue)
        assert issue.category == "serial"
        assert issue.commands[0] == "sudo usermod -a -G uucp $USER"
        assert issue.commands[1] == "groups | grep uucp"
        assert issue.note is not None

    def test_falls_back_to_dialout_without_group_lookup(self, linux, mocker):
        mocker.patch(_SERIAL_LIST_TARGET, return_value=["/dev/ttyUSB0"])
        mocker.patch(_ACCESS_TARGET, side_effect=[False])
        mocker.patch(_GROUP_TARGET, return_value=None)
        issue = check_serial_permissions()
        assert issue is not None
        assert "sudo usermod -a -G dialout $USER" in issue.commands

    def test_inaccessible_ports_snap_instructions(self, linux, mocker):
        mocker.patch(
            "os.environ", {"SNAP": "/snap/rayforge/x1", "SNAP_NAME": "rf"}
        )
        mocker.patch(_SERIAL_LIST_TARGET, return_value=["/dev/ttyUSB0"])
        mocker.patch(_ACCESS_TARGET, return_value=False)
        mocker.patch(_GROUP_TARGET, return_value="dialout")
        issue = check_serial_permissions()
        assert issue is not None
        joined = "\n".join(issue.commands)
        assert "hotplug" in joined
        assert "sudo snap connect rf:serial-port" in joined
        assert "snap connections rf | grep serial-port" in joined
        assert issue.note is not None
        assert "'dialout'" in issue.note


class TestCheckCameraPermissions:
    @pytest.mark.parametrize("platform", ["win32", "darwin"])
    def test_non_linux_never_reports(self, mocker, platform):
        mocker.patch("rayforge.shared.util.permissions.sys.platform", platform)
        assert check_camera_permissions() is None

    def test_no_cameras_is_not_a_permission_problem(self, linux, mocker):
        mocker.patch(_CAMERA_LIST_TARGET, return_value=[])
        assert check_camera_permissions() is None

    def test_accessible_camera_passes(self, linux, mocker):
        by_id = "/dev/v4l/by-id/usb-Foo-video-index0"
        mocker.patch(_CAMERA_LIST_TARGET, return_value=[by_id])
        mocker.patch(_ACCESS_TARGET, return_value=True)
        assert check_camera_permissions() is None

    def test_reports_derived_video_group(self, linux, mocker):
        by_id = "/dev/v4l/by-id/usb-Foo-video-index0"
        mocker.patch(_CAMERA_LIST_TARGET, return_value=[by_id])
        mocker.patch(_ACCESS_TARGET, return_value=False)
        mocker.patch(_GROUP_TARGET, return_value="video")
        issue = check_camera_permissions()
        assert isinstance(issue, PermissionIssue)
        assert issue.category == "camera"
        assert issue.commands[0] == "sudo usermod -a -G video $USER"
        assert issue.commands[1] == "groups | grep video"

    def test_inaccessible_cameras_snap_instructions(self, linux, mocker):
        mocker.patch(
            "os.environ", {"SNAP": "/snap/rayforge/x1", "SNAP_NAME": "rf"}
        )
        by_id = "/dev/v4l/by-id/usb-Foo-video-index0"
        mocker.patch(_CAMERA_LIST_TARGET, return_value=[by_id])
        mocker.patch(_ACCESS_TARGET, return_value=False)
        issue = check_camera_permissions()
        assert issue is not None
        joined = "\n".join(issue.commands)
        assert "sudo snap connect rf:camera" in joined
        assert "snap connections rf | grep camera" in joined


class TestDeviceGroupName:
    @pytest.mark.skipif(os.name != "posix", reason="POSIX only")
    def test_resolves_real_file_group(self, tmp_path):
        probe = tmp_path / "node"
        probe.write_text("")
        assert _device_group_name([str(probe)]) is not None

    @pytest.mark.skipif(os.name != "posix", reason="POSIX only")
    def test_missing_node_returns_none(self):
        assert _device_group_name(["/nonexistent-node"]) is None


class TestCheckPermissions:
    def test_empty_when_everything_ok(self, linux, mocker):
        mocker.patch(_SERIAL_LIST_TARGET, return_value=[])
        mocker.patch(_CAMERA_LIST_TARGET, return_value=[])
        assert check_permissions() == []

    def test_lists_serial_before_camera(self, linux, mocker):
        def _serial_fail():
            return ["/dev/ttyUSB0"]

        def _camera_fail():
            return ["/dev/v4l/by-id/usb-Foo-video-index0"]

        mocker.patch(_SERIAL_LIST_TARGET, side_effect=_serial_fail)
        mocker.patch(_CAMERA_LIST_TARGET, side_effect=_camera_fail)
        mocker.patch(_ACCESS_TARGET, return_value=False)
        mocker.patch(_GROUP_TARGET, return_value="dialout")

        issues = check_permissions()
        assert [i.category for i in issues] == ["serial", "camera"]
