"""Tests for daemon-advertised cameras."""

import pytest

from rayforge.camera import remote
from rayforge.camera.models.source_type import CameraSourceType


@pytest.mark.asyncio
async def test_fetch_remote_cameras_returns_network_sources(monkeypatch):
    async def fake_list_cameras(
        host, port, token=None, tls_ca=None, tls_hostname=None
    ):
        assert (host, port, token, tls_ca, tls_hostname) == (
            "pi.local",
            8765,
            "secret",
            "/tmp/ca.pem",
            "daemon.local",
        )
        return [
            {
                "name": "Enclosure",
                "uri": "http://pi.local:8080/stream",
                "type": "http_stream",
            },
            {
                "name": "Bed",
                "uri": "http://pi.local:8080/bed",
                "type": "http_snapshot",
            },
            {
                "name": "Local",
                "uri": "/dev/video0",
                "type": "local_device",
            },
            {
                "name": "Unknown",
                "uri": "http://pi.local/unknown",
                "type": "bogus",
            },
            {
                "name": "Duplicate",
                "uri": "http://pi.local:8080/stream",
                "type": "http_stream",
            },
        ]

    monkeypatch.setattr(remote, "daemon_list_cameras", fake_list_cameras)
    cameras = await remote.fetch_remote_cameras(
        {
            "remote_host": "pi.local",
            "remote_port": 8765,
            "token": "secret",
            "tls_ca": "/tmp/ca.pem",
            "tls_hostname": "daemon.local",
        }
    )
    assert cameras == [
        remote.RemoteCamera(
            name="Enclosure",
            uri="http://pi.local:8080/stream",
            source_type=CameraSourceType.HTTP_STREAM,
        ),
        remote.RemoteCamera(
            name="Bed",
            uri="http://pi.local:8080/bed",
            source_type=CameraSourceType.HTTP_SNAPSHOT,
        ),
    ]


@pytest.mark.asyncio
async def test_fetch_remote_cameras_ignores_local_args(monkeypatch):
    async def fail_list_cameras(*args, **kwargs):
        raise AssertionError("local machines must not query a daemon")

    monkeypatch.setattr(remote, "daemon_list_cameras", fail_list_cameras)
    assert await remote.fetch_remote_cameras({"port": "/dev/ttyUSB0"}) == []
    assert await remote.fetch_remote_cameras(None) == []


@pytest.mark.asyncio
async def test_fetch_remote_cameras_survives_unreachable_daemon(monkeypatch):
    async def fail_list_cameras(*args, **kwargs):
        raise ConnectionError("daemon is down")

    monkeypatch.setattr(remote, "daemon_list_cameras", fail_list_cameras)
    assert await remote.fetch_remote_cameras({"remote_host": "pi.local"}) == []
