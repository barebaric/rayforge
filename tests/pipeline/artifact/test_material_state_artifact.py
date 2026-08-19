"""Tests for :mod:`rayforge.pipeline.artifact.material_state`."""

from raygeo.ops.material.fold import fold_effects
from raygeo.ops.material.spec import (
    GridBudget,
    MaterialFoldSpec,
    PrismaticStock,
)

from rayforge.pipeline.artifact import (
    MaterialStateArtifact,
    MaterialStateArtifactHandle,
    create_handle_from_dict,
)
from rayforge.pipeline.artifact.store import ArtifactStore


def _make_empty_state():
    """Build a real (empty) raygeo ``MaterialState`` for tests."""
    stock = PrismaticStock(
        polygons=[[(0, 0), (100, 0), (100, 80), (0, 80)]],
        thickness=18.0,
    )
    spec = MaterialFoldSpec(stock=stock, entries=[], grid=GridBudget())
    return fold_effects(spec)


def test_artifact_type_property():
    artifact = MaterialStateArtifact(
        material_state=_make_empty_state(),
        stock_uid="stock-1",
        generation_id=1,
    )
    assert artifact.artifact_type == "MaterialStateArtifact"


def test_artifact_preserves_state():
    state = _make_empty_state()
    artifact = MaterialStateArtifact(
        material_state=state,
        stock_uid="stock-1",
        generation_id=7,
    )
    assert artifact.material_state is state
    assert artifact.stock_uid == "stock-1"
    assert artifact.generation_id == 7


def test_handle_roundtrip_via_store():
    """put → get → release preserves the wrapped MaterialState."""
    store = ArtifactStore()
    state = _make_empty_state()
    artifact = MaterialStateArtifact(
        material_state=state,
        stock_uid="stock-abc",
        generation_id=3,
    )
    handle = store.put(artifact, "material")
    assert isinstance(handle, MaterialStateArtifactHandle)
    assert handle.stock_uid == "stock-abc"
    assert handle.generation_id == 3

    retrieved = store.get(handle)
    assert isinstance(retrieved, MaterialStateArtifact)
    assert retrieved.stock_uid == "stock-abc"
    assert retrieved.generation_id == 3
    assert retrieved.material_state.profile == state.profile
    store.release(handle)


def test_handle_serialization():
    """Handle ``to_dict``/``create_handle_from_dict`` round-trips the
    stock uid."""
    original = MaterialStateArtifactHandle(
        key="test_material_123",
        handle_class_name="MaterialStateArtifactHandle",
        artifact_type_name="MaterialStateArtifact",
        generation_id=1,
        stock_uid="stock-xyz",
    )
    handle_dict = original.to_dict()
    reconstructed = create_handle_from_dict(handle_dict)
    assert isinstance(reconstructed, MaterialStateArtifactHandle)
    assert original == reconstructed
    assert reconstructed.stock_uid == "stock-xyz"
