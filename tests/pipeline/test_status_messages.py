"""
Tests for :mod:`rayforge.pipeline.status_messages`.
"""

import pytest

from rayforge.core.step import Step
from rayforge.core.step_registry import step_registry
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.status_messages import status_message_for_key
from rayforge.pipeline.transformer.base import OpsTransformer
from rayforge.pipeline.transformer.registry import transformer_registry

TEST_ADDON = "test_status_messages"


class _TestStep(Step):
    """Concrete ``Step`` for tests."""

    def __init__(self, name: str = "test"):
        super().__init__(typelabel="contour", name=name)

    def is_position_sensitive(self) -> bool:
        return False


class _RasterStep(Step):
    """Step whose ``ASSEMBLER_NAME`` matches a raygeo assembler."""

    TYPELABEL = "Raster"
    ASSEMBLER_NAME = "raster"

    def __init__(self, name: str = "raster"):
        super().__init__(typelabel=self.TYPELABEL, name=name)

    def is_position_sensitive(self) -> bool:
        return False


class _OverscanTransformer(OpsTransformer):
    """Transformer whose ``SPEC_NAME`` matches a raygeo spec."""

    SPEC_NAME = "overscan"

    @property
    def label(self) -> str:
        return "Overscan"

    @property
    def description(self) -> str:
        return ""

    def to_spec(self, workpiece, stock_geometries, settings):
        return None


class _MultiPassTransformer(OpsTransformer):
    """Transformer whose ``SPEC_NAME`` matches a raygeo spec."""

    SPEC_NAME = "multipass"

    @property
    def label(self) -> str:
        return "Multi-Pass"

    @property
    def description(self) -> str:
        return ""

    def to_spec(self, workpiece, stock_geometries, settings):
        return None


def _displace_colliding_steps() -> list[tuple[str, type]]:
    """Unregister steps whose assembler name matches a test step.

    The progress-label lookup returns the first registered step for an
    assembler name, so a real addon step (e.g. ``EngraveStep``) loaded
    by an earlier test would otherwise shadow the test's own step and
    make these tests order-dependent.  Returns the displaced steps so
    the caller can restore them.
    """
    test_assembler_names = {
        cls.ASSEMBLER_NAME
        for cls in (_TestStep, _RasterStep)
        if cls.ASSEMBLER_NAME
    }
    displaced = [
        (name, cls)
        for name, cls in step_registry.all_steps().items()
        if cls.ASSEMBLER_NAME in test_assembler_names
    ]
    for name, _cls in displaced:
        step_registry.unregister(name)
    return displaced


@pytest.fixture(autouse=True)
def register_progress_labels():
    """Register test steps/transformers the messages look up."""
    displaced = _displace_colliding_steps()
    step_registry.register(_RasterStep, addon_name=TEST_ADDON)
    transformer_registry.register(_OverscanTransformer, addon_name=TEST_ADDON)
    transformer_registry.register(_MultiPassTransformer, addon_name=TEST_ADDON)
    yield
    step_registry.unregister_all_from_addon(TEST_ADDON)
    transformer_registry.unregister_all_from_addon(TEST_ADDON)
    for name, step_class in displaced:
        step_registry.register(step_class)


def test_empty_key_returns_empty_string():
    assert status_message_for_key("", {}, {}) == ""


def test_job_key():
    assert status_message_for_key("job", {}, {}) == "Aggregating job"


def test_job_encode_key():
    assert (
        status_message_for_key("job:encode", {}, {})
        == "Generating machine code"
    )


def test_job_machinexform_key():
    assert (
        status_message_for_key("job:machinexform", {}, {})
        == "Applying machine transform"
    )


def test_workpiece_key_uses_names():
    wp = WorkPiece(name="front plate")
    step = _TestStep()
    workpieces = {wp.uid: wp}
    steps = {step.uid: step}
    key = f"workpiece:{wp.uid}:{step.uid}"
    assert (
        status_message_for_key(key, workpieces, steps)
        == "Processing 'front plate' — contour"
    )


def test_workpiece_key_with_unknown_uid():
    key = "workpiece:missing:also-missing"
    assert status_message_for_key(key, {}, {}) == "Processing"


def test_step_key_uses_typelabel():
    step = _TestStep()
    steps = {step.uid: step}
    key = f"step:{step.uid}"
    assert status_message_for_key(key, {}, steps) == "Assembling 'contour'"


def test_step_key_with_unknown_uid():
    assert status_message_for_key("step:missing", {}, {}) == "Assembling"


def test_unknown_key_falls_back():
    assert status_message_for_key("bogus", {}, {}) == "Processing"


def test_completion_marker_returns_empty():
    assert status_message_for_key("\tworkpiece:a:b", {}, {}) == ""


def test_workpiece_key_with_transformer():
    wp = WorkPiece(name="front plate")
    step = _TestStep()
    workpieces = {wp.uid: wp}
    steps = {step.uid: step}
    key = f"workpiece:{wp.uid}:{step.uid}\toverscan"
    assert (
        status_message_for_key(key, workpieces, steps)
        == "Processing 'front plate' — contour — Overscan"
    )


def test_step_key_with_transformer():
    step = _TestStep()
    steps = {step.uid: step}
    key = f"step:{step.uid}\tmultipass"
    assert (
        status_message_for_key(key, {}, steps)
        == "Assembling 'contour' — Multi-Pass"
    )


def test_workpiece_key_with_assembler():
    wp = WorkPiece(name="front plate")
    step = _TestStep()
    workpieces = {wp.uid: wp}
    steps = {step.uid: step}
    key = f"workpiece:{wp.uid}:{step.uid}\traster: assemble"
    assert (
        status_message_for_key(key, workpieces, steps)
        == "Processing 'front plate' — contour — Raster"
    )


def test_job_key_with_aggregate_detail():
    key = "job\taggregate: group 1/3"
    assert status_message_for_key(key, {}, {}) == "Aggregating job — Aggregate"


def test_unknown_detail_ignored():
    wp = WorkPiece(name="front plate")
    step = _TestStep()
    workpieces = {wp.uid: wp}
    steps = {step.uid: step}
    key = f"workpiece:{wp.uid}:{step.uid}\tcompute: done"
    assert (
        status_message_for_key(key, workpieces, steps)
        == "Processing 'front plate' — contour"
    )
