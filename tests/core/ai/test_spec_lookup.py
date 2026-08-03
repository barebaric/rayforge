"""Tests for the AI machine-spec lookup module."""

from typing import cast

import pytest

from rayforge.context import RayforgeContext
from rayforge.core.ai.provider import AIServiceError, ChatResponse
from rayforge.core.ai.spec_lookup import (
    _coerce_specs,
    _extract_json_object,
    is_ai_configured,
    lookup_machine_specs,
)


def test_extract_plain_json():
    assert _extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_fenced_json():
    content = 'Sure!\n```json\n{"a": 1, "b": [1, 2]}\n```\nHope this helps.'
    assert _extract_json_object(content) == {"a": 1, "b": [1, 2]}


def test_extract_json_wrapped_in_prose():
    content = (
        'Here you go: {"axis_extents": [400, 300], "origin": "top_left"}'
        " That's all."
    )
    assert _extract_json_object(content) == {
        "axis_extents": [400, 300],
        "origin": "top_left",
    }


def test_extract_json_nested_braces_and_strings():
    content = '{"a": {"b": 1}, "note": "has {brace} inside"}'
    assert _extract_json_object(content) == {
        "a": {"b": 1},
        "note": "has {brace} inside",
    }


def test_extract_invalid_returns_none():
    assert _extract_json_object("not json at all") is None


def test_extract_empty_returns_none():
    assert _extract_json_object("") is None
    assert _extract_json_object("   ") is None


def test_extract_non_dict_json_returns_none():
    assert _extract_json_object("[1, 2, 3]") is None


def test_coerce_drops_none():
    assert _coerce_specs({"a": None, "b": 1}) == {"b": 1}


def test_coerce_numeric_strings():
    assert _coerce_specs({"x": "400", "y": "300.5"}) == {"x": 400, "y": 300.5}


def test_coerce_keeps_plain_strings():
    assert _coerce_specs({"origin": "top_left"}) == {"origin": "top_left"}


def test_coerce_blank_strings_dropped():
    assert _coerce_specs({"a": "   ", "b": "x"}) == {"b": "x"}


class FakeAIService:
    def __init__(self, provider, response):
        self._provider = provider
        self._response = response

    def get_provider(self):
        return self._provider

    async def chat(self, messages):
        return self._response


class FakeContext:
    def __init__(self, service):
        self.ai_service = service


@pytest.mark.asyncio
async def test_lookup_returns_empty_without_provider():
    ctx = FakeContext(FakeAIService(None, None))
    assert (
        await lookup_machine_specs(
            "Sculpfun", "S30", cast(RayforgeContext, ctx)
        )
        == {}
    )


@pytest.mark.asyncio
async def test_lookup_returns_empty_on_empty_response():
    ctx = FakeContext(FakeAIService(object(), None))
    assert (
        await lookup_machine_specs(
            "Sculpfun", "S30", cast(RayforgeContext, ctx)
        )
        == {}
    )


@pytest.mark.asyncio
async def test_lookup_parses_response():
    response = ChatResponse(
        content='{"axis_extents": [400, 300], "max_power": 30000}',
        model="test-model",
    )
    ctx = FakeContext(FakeAIService(object(), response))
    result = await lookup_machine_specs(
        "Sculpfun", "S30", cast(RayforgeContext, ctx)
    )
    assert result == {"axis_extents": [400, 300], "max_power": 30000}


@pytest.mark.asyncio
async def test_lookup_returns_empty_on_service_error():
    async def chat(messages):
        raise AIServiceError("boom")

    service = FakeAIService(object(), None)
    service.chat = chat
    ctx = FakeContext(service)
    assert (
        await lookup_machine_specs(
            "Sculpfun", "S30", cast(RayforgeContext, ctx)
        )
        == {}
    )


@pytest.mark.asyncio
async def test_lookup_returns_empty_on_unexpected_error():
    async def chat(messages):
        raise RuntimeError("boom")

    service = FakeAIService(object(), None)
    service.chat = chat
    ctx = FakeContext(service)
    assert (
        await lookup_machine_specs(
            "Sculpfun", "S30", cast(RayforgeContext, ctx)
        )
        == {}
    )


def test_is_ai_configured_true():
    ctx = FakeContext(FakeAIService(object(), None))
    assert is_ai_configured(cast(RayforgeContext, ctx)) is True


def test_is_ai_configured_false():
    ctx = FakeContext(FakeAIService(None, None))
    assert is_ai_configured(cast(RayforgeContext, ctx)) is False
