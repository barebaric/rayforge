"""Tests for the anonymous usage tracker."""

import os
from unittest.mock import patch

from rayforge.machine.models.laser import LaserType
from rayforge.usage import UsageTracker, _get_user_agent


def test_user_agent_reports_windows():
    with patch("rayforge.usage.platform.system", return_value="Windows"):
        assert "Windows NT 10.0" in _get_user_agent()


def test_user_agent_reports_macos():
    with patch("rayforge.usage.platform.system", return_value="Darwin"):
        assert "Mac OS X" in _get_user_agent()


def test_user_agent_reports_linux_machine():
    with (
        patch("rayforge.usage.platform.system", return_value="Linux"),
        patch("rayforge.usage.platform.machine", return_value="aarch64"),
    ):
        assert "X11; Linux aarch64" in _get_user_agent()


def test_env_variable_blocks_events():
    tracker = UsageTracker()
    tracker.set_enabled(True)
    env = {"RAYFORGE_NO_USAGE_TRACKING": "1"}
    with (
        patch.dict(os.environ, env),
        patch.object(UsageTracker, "_send_event") as send_event,
    ):
        tracker.track_page_view("/view/2d", "2D View")
        send_event.assert_not_called()


class _FakeLaserHead:
    laser_type = LaserType.DIODE

    @property
    def effective_max_power_watts(self):
        return 20.0


class _FakeMachine:
    def __init__(self):
        self.placeholder = False
        self.driver_name = "GrblSerialDriver"
        self.rotary_modules = {}
        self.axis_extents = (430.0, 400.0)

    def get_default_laser_head(self):
        return _FakeLaserHead()


def _capture_payload(tracker, method, *args):
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(UsageTracker, "_send_event") as send_event,
    ):
        getattr(tracker, method)(*args)
    assert send_event.call_count >= 1
    return send_event.call_args.args[-1]


def test_page_view_reports_session_as_id():
    tracker = UsageTracker()
    tracker._screen = "1920x1080"
    tracker.set_enabled(True)
    payload = _capture_payload(tracker, "track_page_view", "/view/2d")
    assert payload["id"] == tracker._session_id
    assert "sessionId" not in payload
    assert payload["screen"] == "1920x1080"


def test_page_view_omits_screen_when_unknown():
    tracker = UsageTracker()
    tracker._screen = None
    tracker.set_enabled(True)
    payload = _capture_payload(tracker, "track_page_view", "/view/2d")
    assert "screen" not in payload


def test_track_machines_reports_machine_types():
    tracker = UsageTracker()
    tracker.set_enabled(True)
    payload = _capture_payload(tracker, "track_machines", [_FakeMachine()])
    assert payload["name"] == "machine"
    assert payload["data"] == {
        "driver": "GrblSerialDriver",
        "laser": "diode",
        "power_w": "20",
        "bed": "430x400",
        "rotary": "false",
    }


def test_track_machines_skips_placeholder_machines():
    tracker = UsageTracker()
    tracker.set_enabled(True)
    placeholder = _FakeMachine()
    placeholder.placeholder = True
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(UsageTracker, "_send_event") as send_event,
    ):
        tracker.track_machines([placeholder])
    send_event.assert_not_called()
