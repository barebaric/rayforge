"""Tests for the anonymous usage tracker."""

import os
from unittest.mock import patch

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
