"""Tests for the versioned document schema migration chain."""

import copy
import json
from pathlib import Path

import pytest
from raygeo.geo import Matrix

from rayforge.core.doc import Doc
from rayforge.core.doc_schema import (
    CURRENT_DOC_SCHEMA_VERSION,
    migrate,
)

V0_DOC_PATH = Path(__file__).parent.parent / "assets" / "doc_schema_v0.ryp"

IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _v0_doc() -> dict:
    """The v0 sample document, as saved by an older app version."""
    return json.loads(V0_DOC_PATH.read_text())


def _workflow(layer: dict) -> dict:
    return next(
        child for child in layer["children"] if child["type"] == "workflow"
    )


def _content(layer: dict, type_name: str) -> list[dict]:
    return [child for child in layer["children"] if child["type"] == type_name]


def test_migrate_v0_to_v1_shape():
    v1 = migrate(copy.deepcopy(_v0_doc()))

    assert v1["schema_version"] == CURRENT_DOC_SCHEMA_VERSION
    assert v1["active_layer_index"] == 1

    types = [child["type"] for child in v1["children"]]
    assert types == ["layer", "layer"]

    cut, engrave = v1["children"]
    assert cut["name"] == "Layer 1"
    assert cut["wcs"] == "G55"
    assert cut["visible"] is True
    assert "color" not in cut
    assert "post_processors_dicts" not in cut

    # The old layer became a workflow with its own workpieces.
    cut_workflow = _workflow(cut)
    assert cut_workflow["uid"] == "layer-1"
    assert cut_workflow["name"] == "Cut"
    assert cut_workflow["color"] == "#ff0000"
    assert cut_workflow["workpiece_uids"] == ["wp-1"]
    assert cut_workflow["post_processors_dicts"] == [
        {"name": "MeshCorrection"}
    ]

    # The workpiece stayed a content child of the new layer.
    assert [wp["uid"] for wp in _content(cut, "workpiece")] == ["wp-1"]

    # The second old layer kept its visibility and became a workflow.
    assert engrave["name"] == "Layer 2"
    assert engrave["visible"] is False
    engrave_workflow = _workflow(engrave)
    assert engrave_workflow["uid"] == "layer-2"
    assert engrave_workflow["name"] == "Engrave"
    assert engrave_workflow["color"] == "#00ff00"
    assert engrave_workflow["workpiece_uids"] == []


def test_migrate_groups_layers_by_setup():
    v0 = _v0_doc()
    v0["children"][1]["wcs"] = v0["children"][0]["wcs"]
    v0["children"][1]["visible"] = True

    v1 = migrate(v0)

    assert len(v1["children"]) == 1
    layer = v1["children"][0]
    workflows = [
        child for child in layer["children"] if child["type"] == "workflow"
    ]
    assert [workflow["name"] for workflow in workflows] == [
        "Cut",
        "Engrave",
    ]
    assert [workflow["uid"] for workflow in workflows] == [
        "layer-1",
        "layer-2",
    ]
    assert workflows[0]["workpiece_uids"] == ["wp-1"]
    assert workflows[1]["workpiece_uids"] == []
    assert v1["active_layer_index"] == 0


def test_migrate_bakes_layer_transform_into_children():
    v0 = _v0_doc()
    transform = Matrix.translation(10, 20)
    v0["children"][0]["matrix"] = transform.to_list()

    v1 = migrate(v0)
    layer = v1["children"][0]
    assert _workflow(layer)["matrix"] == transform.to_list()
    assert _content(layer, "workpiece")[0]["matrix"] == transform.to_list()


def test_migrate_renames_embedded_import_mode():
    v1 = migrate(copy.deepcopy(_v0_doc()))
    workpiece = _content(v1["children"][0], "workpiece")[0]
    spec = workpiece["source_segment"]["vectorization_spec"]
    assert spec["layer_import_mode"] == "new_workflows"


def test_migrate_folds_document_stock_into_first_layer():
    v1 = migrate(copy.deepcopy(_v0_doc()))
    first = v1["children"][0]
    stock = _content(first, "stockitem")
    assert len(stock) == 1
    assert stock[0]["uid"] == "stock-1"


def test_migrate_is_idempotent():
    once = migrate(copy.deepcopy(_v0_doc()))
    twice = migrate(copy.deepcopy(once))
    assert twice == once


def test_migrate_keeps_unknown_keys():
    v0 = _v0_doc()
    v0["future_doc_field"] = {"x": 1}
    v0["children"][0]["future_layer_field"] = "kept"
    v1 = migrate(copy.deepcopy(v0))
    assert v1["future_doc_field"] == {"x": 1}
    assert _workflow(v1["children"][0])["future_layer_field"] == "kept"


def test_migrate_does_not_mutate_input():
    v0 = _v0_doc()
    snapshot = copy.deepcopy(v0)
    migrate(v0)
    assert v0 == snapshot


def test_migrate_current_document_is_unchanged():
    data = {
        "uid": "doc-uid",
        "type": "doc",
        "schema_version": CURRENT_DOC_SCHEMA_VERSION,
        "active_layer_index": 0,
        "children": [
            {
                "uid": "layer-1",
                "type": "layer",
                "name": "Layer 1",
                "matrix": IDENTITY,
                "children": [],
            }
        ],
    }
    assert migrate(copy.deepcopy(data)) == data


def test_doc_round_trip_stamps_version():
    doc = Doc()
    data = doc.to_dict()
    assert data["schema_version"] == CURRENT_DOC_SCHEMA_VERSION

    restored = Doc.from_dict(data)
    assert len(restored.layers) == 1
    assert restored.layers[0].uid == doc.layers[0].uid
    assert (
        restored.layers[0].workflows[0].uid == doc.layers[0].workflows[0].uid
    )


def test_saving_a_v0_document_writes_v1():
    doc = Doc.from_dict(_v0_doc())
    data = doc.to_dict()
    assert data["schema_version"] == CURRENT_DOC_SCHEMA_VERSION
    assert [child["type"] for child in data["children"]] == [
        "layer",
        "layer",
    ]
    assert doc.active_layer.name == "Layer 2"
    assert [workflow.name for workflow in doc.workflows] == [
        "Cut",
        "Engrave",
    ]


def test_unknown_future_version_is_loaded_as_is(caplog):
    data = {
        "uid": "doc-uid",
        "type": "doc",
        "schema_version": CURRENT_DOC_SCHEMA_VERSION + 5,
        "children": [],
    }
    with caplog.at_level("WARNING"):
        result = migrate(copy.deepcopy(data))
    assert result == data
    assert any("newer" in record.message for record in caplog.records)


@pytest.mark.parametrize("version", [0, None])
def test_missing_or_zero_version_runs_chain(version):
    data = _v0_doc()
    if version is None:
        data.pop("active_layer_index", None)
    else:
        data["schema_version"] = version
    result = migrate(copy.deepcopy(data))
    assert result["schema_version"] == CURRENT_DOC_SCHEMA_VERSION
    assert result["children"][0]["type"] == "layer"


def test_newer_schema_sets_flag():
    newer = CURRENT_DOC_SCHEMA_VERSION + 3
    data = {
        "uid": "doc-uid",
        "type": "doc",
        "schema_version": newer,
        "children": [],
    }
    doc = Doc.from_dict(data)
    assert doc.newer_schema_version == newer


def test_current_schema_has_no_newer_flag():
    assert Doc().newer_schema_version is None
    assert Doc.from_dict(Doc().to_dict()).newer_schema_version is None


def test_migrate_uses_workflow_name_when_layer_matches_part():
    """A layer named after its only part keeps the workflow's name."""
    v0 = _v0_doc()
    v0["children"] = [v0["children"][0]]
    v0["children"][0]["children"][1]["name"] = "Cut"
    v0["children"][0]["name"] = "Cut"
    v0["children"][0]["children"][0]["name"] = "Cut Workflow"
    v0["children"][0]["visible"] = True

    v1 = migrate(v0)

    workflow = _workflow(v1["children"][0])
    assert workflow["name"] == "Cut Workflow"
