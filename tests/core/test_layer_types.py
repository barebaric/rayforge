"""Unit tests for the layer type registry."""

import pytest

from rayforge.core.layer import Layer
from rayforge.core.layer_types import (
    LayerType,
    LayerTypeRegistry,
    layer_is_locked,
    layer_type_registry,
    locked_layer_notice,
)

TEST_ADDON = "layer_type_test_addon"


@pytest.fixture
def registry():
    return LayerTypeRegistry()


@pytest.fixture
def test_type():
    layer_type = LayerType(
        id="test_kind", title="Test Kind", icon_name="test", locked=True
    )
    layer_type_registry.register(layer_type, TEST_ADDON)
    yield layer_type
    layer_type_registry.unregister_all_from_addon(TEST_ADDON)


def test_register_and_get(registry):
    layer_type = LayerType(id="test_kind", title="Test Kind")
    registry.register(layer_type, "addon")
    assert registry.get("test_kind") is layer_type
    assert registry.get("other") is None
    assert registry.get("") is None


def test_register_twice_replaces(registry):
    first = LayerType(id="kind", title="First")
    second = LayerType(id="kind", title="Second")
    registry.register(first, "addon")
    registry.register(second, "addon")
    assert registry.get("kind") is second
    assert registry.all() == [second]


def test_unregister_all_from_addon(registry):
    registry.register(LayerType(id="test_kind", title="Test"), TEST_ADDON)
    registry.register(LayerType(id="other_kind", title="Other"), "other")
    removed = registry.unregister_all_from_addon(TEST_ADDON)
    assert removed == 1
    assert registry.get("test_kind") is None
    assert registry.get("other_kind") is not None


def test_unregister_all_from_addon_without_registrations(registry):
    assert registry.unregister_all_from_addon("missing") == 0


def test_layer_is_locked(test_type):
    layer = Layer("Layer")
    layer.layer_type = "test_kind"
    assert layer_is_locked(layer) is True


def test_unlocked_layer_type():
    layer_type_registry.register(
        LayerType(id="free_kind", title="Free"), TEST_ADDON
    )
    try:
        layer = Layer("Layer")
        layer.layer_type = "free_kind"
        assert layer_is_locked(layer) is False
    finally:
        layer_type_registry.unregister_all_from_addon(TEST_ADDON)


def test_unknown_layer_type_is_not_locked():
    layer = Layer("Layer")
    layer.layer_type = "vanished_kind"
    assert layer_is_locked(layer) is False


def test_empty_layer_type_is_not_locked(test_type):
    layer = Layer("Layer")
    assert layer_is_locked(layer) is False


def test_locked_layer_notice_names_the_type(test_type):
    layer = Layer("Layer")
    layer.layer_type = "test_kind"
    assert "Test Kind" in locked_layer_notice(layer)


def test_locked_layer_notice_for_unknown_type():
    layer = Layer("Layer")
    assert locked_layer_notice(layer)
