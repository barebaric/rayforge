"""Tests for the versioned document schema migration chain."""

import copy
import json
from pathlib import Path

import pytest

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


def test_migrate_v0_to_v1_shape():
    v1 = migrate(copy.deepcopy(_v0_doc()))

    assert v1["schema_version"] == CURRENT_DOC_SCHEMA_VERSION
    assert v1["active_layer_index"] == 1

    types = [child["type"] for child in v1["children"]]
    assert types == ["layer", "layer"]

    cut, engrave = v1["children"]
    assert cut["uid"] == "layer-1"
    assert cut["name"] == "Cut"
    assert cut["wcs"] == "G55"
    assert cut["visible"] is True
    assert "color" not in cut
    assert "post_processors_dicts" not in cut

    # The workflow inherited the layer's color and post processors.
    workflows = [
        child for child in cut["children"] if child["type"] == "workflow"
    ]
    assert workflows[0]["uid"] == "wf-1"
    assert workflows[0]["color"] == "#ff0000"
    assert workflows[0]["post_processors_dicts"] == [
        {"name": "MeshCorrection"}
    ]

    # The workpiece stayed a content child of the layer.
    kinds = {child["type"] for child in cut["children"]}
    assert kinds == {"workflow", "workpiece", "stockitem"}

    # The second layer keeps its visibility; the workflow took the
    # color.
    engrave_workflow = next(
        child for child in engrave["children"] if child["type"] == "workflow"
    )
    assert engrave_workflow["color"] == "#00ff00"


def test_migrate_renames_embedded_import_mode():
    v1 = migrate(copy.deepcopy(_v0_doc()))
    workpiece = next(
        child
        for child in v1["children"][0]["children"]
        if child["type"] == "workpiece"
    )
    spec = workpiece["source_segment"]["vectorization_spec"]
    assert spec["layer_import_mode"] == "new_workflows"


def test_migrate_folds_document_stock_into_first_layer():
    v1 = migrate(copy.deepcopy(_v0_doc()))
    first = v1["children"][0]
    stock = [
        child for child in first["children"] if child["type"] == "stockitem"
    ]
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
    assert v1["children"][0]["future_layer_field"] == "kept"


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
    assert doc.active_layer.name == "Engrave"


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
