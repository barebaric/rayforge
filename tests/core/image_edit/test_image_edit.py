from collections.abc import Mapping
from typing import Any

import pytest
import pyvips

from rayforge.core.image_edit import (
    ImageEditOp,
    ImageEditRegistry,
    ImageEditStep,
    apply_image_edits,
    missing_image_edit_ops,
)


class _AddOp(ImageEditOp):
    NAME = "test_add"
    LABEL = "Add"

    @classmethod
    def apply(
        cls, image: pyvips.Image, params: Mapping[str, Any]
    ) -> pyvips.Image:
        return (image + params.get("amount", 1)).cast("uchar")


def _gray(value: int) -> pyvips.Image:
    return (pyvips.Image.black(2, 2) + value).cast("uchar")


@pytest.fixture
def registry() -> ImageEditRegistry:
    reg = ImageEditRegistry()
    reg.register(_AddOp, addon_name="test_addon")
    return reg


class TestImageEditStep:
    def test_round_trip(self):
        step = ImageEditStep("test_add", {"amount": 3}, enabled=False)
        assert ImageEditStep.from_dict(step.to_dict()) == step

    def test_defaults(self):
        step = ImageEditStep.from_dict({"op": "test_add"})
        assert step.params == {}
        assert step.enabled

    def test_unknown_keys_survive_a_round_trip(self):
        data = {"op": "future", "params": {"x": 1}, "version": 2}
        assert ImageEditStep.from_dict(data).to_dict() == {
            "op": "future",
            "params": {"x": 1},
            "enabled": True,
            "version": 2,
        }

    def test_extra_keys_do_not_affect_equality(self):
        a = ImageEditStep.from_dict({"op": "test_add", "note": "a"})
        assert a == ImageEditStep("test_add")


class TestImageEditRegistry:
    def test_get(self, registry):
        assert registry.get("test_add") is _AddOp
        assert registry.get("nope") is None

    def test_all_ops(self, registry):
        assert registry.all_ops() == {"test_add": _AddOp}

    def test_unregister_all_from_addon(self, registry):
        assert registry.unregister_all_from_addon("test_addon") == 1
        assert registry.get("test_add") is None
        assert registry.unregister_all_from_addon("test_addon") == 0

    def test_unregister(self, registry):
        assert registry.unregister("test_add")
        assert not registry.unregister("test_add")
        assert registry.unregister_all_from_addon("test_addon") == 0


class TestApplyImageEdits:
    def test_applies_ops_in_order(self, registry):
        edits = [
            ImageEditStep("test_add", {"amount": 2}),
            ImageEditStep("test_add", {"amount": 5}),
        ]
        result = apply_image_edits(_gray(10), edits, registry)
        assert result.avg() == 17

    def test_skips_disabled_steps(self, registry):
        edits = [ImageEditStep("test_add", {"amount": 2}, enabled=False)]
        assert apply_image_edits(_gray(10), edits, registry).avg() == 10

    def test_skips_unregistered_ops(self, registry):
        edits = [
            ImageEditStep("future_op"),
            ImageEditStep("test_add", {"amount": 1}),
        ]
        assert apply_image_edits(_gray(10), edits, registry).avg() == 11

    def test_no_edits_returns_the_same_image(self, registry):
        image = _gray(10)
        assert apply_image_edits(image, [], registry) is image


def test_missing_image_edit_ops(registry):
    edits = [
        ImageEditStep("test_add"),
        ImageEditStep("future_op"),
        ImageEditStep("other_op", enabled=False),
    ]
    assert missing_image_edit_ops(edits, registry) == {
        "future_op",
        "other_op",
    }
