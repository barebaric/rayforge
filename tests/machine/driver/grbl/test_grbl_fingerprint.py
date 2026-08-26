"""Tests for HTTP fingerprinting of mDNS candidates."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rayforge.machine.driver.grbl.grbl_fingerprint import fingerprint_grbl_http


def _response(status=200, text=""):
    response = MagicMock()
    response.status = status
    response.content.read = AsyncMock(return_value=text.encode("utf-8"))
    return response


def _patch_get(monkeypatch, response):
    """Patches aiohttp so ``async with ClientSession() as s:
    async with s.get(url) as r`` yields *response*."""
    request_ctx = MagicMock()
    request_ctx.__aenter__ = AsyncMock(return_value=response)
    session = MagicMock()
    session.get.return_value = request_ctx
    session_factory = MagicMock()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=session)
    monkeypatch.setattr(
        "rayforge.machine.driver.grbl.grbl_fingerprint.aiohttp.ClientSession",
        session_factory,
    )


@pytest.mark.asyncio
async def test_fluidnc_response_is_matched(monkeypatch):
    _patch_get(
        monkeypatch,
        _response(text='{"status":"FluidNC v3.7.8"}'),
    )
    service = await fingerprint_grbl_http("192.168.1.70", 80)
    assert service is not None
    assert service.host == "192.168.1.70"
    assert service.port == 80
    assert service.service_type == "_http._tcp"
    assert "fluidnc" in service.name.lower()


@pytest.mark.asyncio
async def test_esp3d_and_grblhal_responses_are_matched(monkeypatch):
    for fw in ("ESP3D 3.0", "grblHAL 1.1", "Grbl 1.1f"):
        _patch_get(monkeypatch, _response(text=fw))
        service = await fingerprint_grbl_http("h", 80)
        assert service is not None, fw


@pytest.mark.asyncio
async def test_unrelated_web_server_is_rejected(monkeypatch):
    _patch_get(monkeypatch, _response(text="<html>Router Admin</html>"))
    assert await fingerprint_grbl_http("h", 80) is None


@pytest.mark.asyncio
async def test_error_status_is_rejected(monkeypatch):
    _patch_get(monkeypatch, _response(status=404, text="FluidNC"))
    assert await fingerprint_grbl_http("h", 80) is None


@pytest.mark.asyncio
async def test_connection_failure_yields_none(monkeypatch):
    _patch_get(monkeypatch, _response())

    async def boom(*args, **kwargs):
        raise OSError("no route")

    monkeypatch.setattr(
        "rayforge.machine.driver.grbl.grbl_fingerprint._fetch", boom
    )
    assert await fingerprint_grbl_http("h", 80) is None


@pytest.mark.asyncio
async def test_slow_response_is_bounded(monkeypatch):
    _patch_get(monkeypatch, _response())

    async def slow(*args, **kwargs):
        import asyncio

        await asyncio.sleep(5)

    monkeypatch.setattr(
        "rayforge.machine.driver.grbl.grbl_fingerprint._fetch", slow
    )
    assert await fingerprint_grbl_http("h", 80, timeout=0.05) is None
