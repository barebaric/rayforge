"""Tests for layer-level post processors in the intent builder.

Covers the merge helper, the machine-transform spec wiring (machine
defaults + per-layer overrides), and the cache-token sensitivity of
the new inputs.
"""

import pytest

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.pipeline.intent_builder import (
    IntentBuilder,
    _canonical,
    _hash_int,
    _merge_post_processor_dicts,
)

MESH_DICT = {"name": "MeshCorrectionTransformer", "enabled": True}
OTHER_DICT = {"name": "SomeOtherTransformer", "enabled": True}


@pytest.fixture(autouse=True)
def _register_mesh_transformer():
    """The post addon is not loaded in builder tests; register the
    transformer directly so the registry can resolve MESH_DICT."""
    import importlib.util
    from pathlib import Path

    from rayforge.pipeline.transformer.registry import (
        transformer_registry,
    )

    module_path = (
        Path(__file__).parent.parent.parent
        / "rayforge"
        / "builtin_addons"
        / "rayforge-addon-post"
        / "post_processors"
        / "transformers"
        / "mesh_correction_transformer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mesh_correction_transformer", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transformer_registry.register(
        module.MeshCorrectionTransformer, addon_name="post_processors"
    )
    yield


def _make_bed_mesh():
    from rayforge.machine.models.bed_mesh import BedMesh

    mesh = BedMesh()
    mesh.set_params(
        x0=0.0,
        y0=0.0,
        dx=50.0,
        dy=50.0,
        nx=2,
        ny=2,
        heights=[0.0, 1.0, 2.0, 3.0],
    )
    return mesh


# ----------------------------------------------------------------------
# _merge_post_processor_dicts
# ----------------------------------------------------------------------


def test_merge_empty_both():
    assert _merge_post_processor_dicts([], []) == []


def test_merge_defaults_pass_through():
    assert _merge_post_processor_dicts([MESH_DICT], []) == [MESH_DICT]


def test_merge_override_replaces_default():
    override = {"name": "MeshCorrectionTransformer", "enabled": False}
    merged = _merge_post_processor_dicts([MESH_DICT, OTHER_DICT], [override])
    assert merged == [override, OTHER_DICT]


def test_merge_override_adds_new_transformer():
    extra = {"name": "ExtraTransformer", "enabled": True}
    assert _merge_post_processor_dicts([], [extra]) == [extra]


def test_merge_preserves_default_order():
    a = {"name": "A", "enabled": True}
    b = {"name": "B", "enabled": True}
    override_b = {"name": "B", "z_offset": 2.0}
    assert _merge_post_processor_dicts([a, b], [override_b]) == [
        a,
        override_b,
    ]


# ----------------------------------------------------------------------
# Machine-transform stage wiring
# ----------------------------------------------------------------------


def _builder(machine):
    return IntentBuilder(machine=machine)


def test_no_post_processors_builds_empty_lists(isolated_machine):
    doc = Doc()
    spec = _builder(isolated_machine)._build_machine_transform_stage(doc)
    assert spec.default_transformers == []
    assert spec.layer_transformers == []


def test_machine_defaults_become_default_transformers(isolated_machine):
    isolated_machine.default_post_processors_dicts = [dict(MESH_DICT)]
    isolated_machine.set_bed_mesh(_make_bed_mesh())

    spec = _builder(isolated_machine)._build_machine_transform_stage(Doc())
    assert len(spec.default_transformers) == 1
    assert spec.layer_transformers == []


def test_mesh_correction_skipped_without_mesh(isolated_machine):
    """No probed mesh: the default is dropped instead of failing."""
    isolated_machine.default_post_processors_dicts = [dict(MESH_DICT)]
    spec = _builder(isolated_machine)._build_machine_transform_stage(Doc())
    assert spec.default_transformers == []


def test_layer_override_builds_layer_transformers(isolated_machine):
    isolated_machine.default_post_processors_dicts = [dict(MESH_DICT)]
    isolated_machine.set_bed_mesh(_make_bed_mesh())

    doc = Doc()
    doc.active_layer.post_processors_dicts = [
        {"name": "MeshCorrectionTransformer", "z_offset": 1.5}
    ]
    spec = _builder(isolated_machine)._build_machine_transform_stage(doc)

    assert spec.default_transformers  # machine default still global
    assert len(spec.layer_transformers) == 1
    layer_uid, specs = spec.layer_transformers[0]
    assert layer_uid == doc.active_layer.uid
    assert len(specs) == 1
    assert specs[0].z_offset == 1.5


def test_layer_disable_entry_yields_no_specs(isolated_machine):
    """A disabled layer entry overrides the default to 'off'."""
    isolated_machine.default_post_processors_dicts = [dict(MESH_DICT)]
    isolated_machine.set_bed_mesh(_make_bed_mesh())

    doc = Doc()
    doc.active_layer.post_processors_dicts = [
        {"name": "MeshCorrectionTransformer", "enabled": False}
    ]
    spec = _builder(isolated_machine)._build_machine_transform_stage(doc)
    assert spec.layer_transformers == []


def test_layer_entry_without_machine_default(isolated_machine):
    """A layer entry works even when the machine has no default."""
    isolated_machine.set_bed_mesh(_make_bed_mesh())

    doc = Doc()
    doc.active_layer.post_processors_dicts = [dict(MESH_DICT)]
    spec = _builder(isolated_machine)._build_machine_transform_stage(doc)
    assert spec.default_transformers == []
    assert len(spec.layer_transformers) == 1


def test_post_processor_layer_round_trip(isolated_machine):
    """Layer.post_processors_dicts survive to_dict/from_dict."""
    layer = Layer(name="l")
    layer.post_processors_dicts = [dict(MESH_DICT)]
    restored = Layer.from_dict(layer.to_dict())
    assert restored.post_processors_dicts == [dict(MESH_DICT)]


# ----------------------------------------------------------------------
# Cache-token sensitivity
# ----------------------------------------------------------------------


def test_machine_transform_token_changes_with_mesh(isolated_machine):
    doc = Doc()
    builder = _builder(isolated_machine)
    token_before = builder._machine_transform_token(doc, {})

    isolated_machine.set_bed_mesh(_make_bed_mesh())
    token_after = builder._machine_transform_token(doc, {})
    assert token_before != token_after


def test_machine_transform_token_changes_with_defaults(isolated_machine):
    doc = Doc()
    builder = _builder(isolated_machine)
    token_before = builder._machine_transform_token(doc, {})

    isolated_machine.default_post_processors_dicts = [dict(MESH_DICT)]
    token_after = builder._machine_transform_token(doc, {})
    assert token_before != token_after


def test_machine_transform_token_changes_with_layer_override(isolated_machine):
    doc = Doc()
    builder = _builder(isolated_machine)
    token_before = builder._machine_transform_token(doc, {})

    doc.active_layer.post_processors_dicts = [dict(MESH_DICT)]
    token_after = builder._machine_transform_token(doc, {})
    assert token_before != token_after


def test_canonical_stable_for_dicts():
    assert _canonical([MESH_DICT]) == _canonical(
        [{"enabled": True, "name": "MeshCorrectionTransformer"}]
    )
    assert _hash_int({"a": _canonical([MESH_DICT])}) == _hash_int(
        {
            "a": _canonical(
                [{"name": "MeshCorrectionTransformer", "enabled": True}]
            )
        }
    )
