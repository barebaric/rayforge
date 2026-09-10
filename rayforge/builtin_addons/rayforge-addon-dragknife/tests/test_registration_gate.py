"""Tests for the raygeo availability gate in the addon worker."""

from dragknife import worker
from dragknife.transformers import (
    DragKnifeTransformer,
    TangentialKnifeTransformer,
)

from rayforge.core.step_registry import StepRegistry
from rayforge.pipeline.transformer.registry import TransformerRegistry


def test_steps_registered_when_raygeo_supported():
    registry = StepRegistry()
    worker.register_steps(registry)
    assert registry.get("DragKnifeCutStep") is not None
    assert registry.get("TangentialKnifeCutStep") is not None


def test_transformers_registered_when_raygeo_supported():
    registry = TransformerRegistry()
    worker.register_transformers(registry)
    assert registry.get("DragKnifeTransformer") is DragKnifeTransformer
    assert (
        registry.get("TangentialKnifeTransformer")
        is TangentialKnifeTransformer
    )


def test_registration_skipped_without_raygeo_support(monkeypatch):
    monkeypatch.setattr(worker, "KNIFE_TRANSFORMS_AVAILABLE", False)
    steps = StepRegistry()
    transformers = TransformerRegistry()
    worker.register_steps(steps)
    worker.register_transformers(transformers)
    assert steps.get("DragKnifeCutStep") is None
    assert steps.get("TangentialKnifeCutStep") is None
    assert transformers.get("DragKnifeTransformer") is None
    assert transformers.get("TangentialKnifeTransformer") is None
