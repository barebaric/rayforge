"""
Integration tests for SVG color rules (color -> step type).

These exercise the full import flow: importing an SVG by color source
produces layers whose color is matched against color rules when default
steps are added, yielding the configured step type.
"""

from pathlib import Path

import pytest

from rayforge import config
from rayforge.core.color_preset import (
    ColorPreset,
    get_color_preset_mgr,
    reset_color_preset_mgr,
)
from rayforge.core.vectorization_spec import (
    LayerImportMode,
    LayerSource,
    PassthroughSpec,
)
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.file_cmd import FileCmd
from rayforge.doceditor.step_cmd import StepCmd
from rayforge.image.svg.importer import SvgImporter
from rayforge.image.svg.svg_vector import SvgVectorImporter

SVG_COLORS = b"""
<svg width="100mm" height="100mm" viewBox="0 0 100 100"
     xmlns="http://www.w3.org/2000/svg">
    <rect x="0" y="0" width="10" height="10" fill="#ff0000"/>
    <rect x="50" y="50" width="10" height="10" fill="#00ff00"/>
</svg>
"""


@pytest.fixture
def color_rules(tmp_path, monkeypatch):
    """
    Points the color preset manager at a temporary directory so tests
    can add rules without touching the real user configuration.
    """
    reset_color_preset_mgr()
    monkeypatch.setattr(
        config, "USER_COLOR_PRESETS_DIR", tmp_path / "color_presets"
    )
    yield get_color_preset_mgr()
    reset_color_preset_mgr()


def _step_type_names(workflow) -> list:
    return [type(s).__name__ for s in workflow.steps]


def _import_color_workflows(editor) -> list:
    """Builds one attached workflow per imported source color."""
    importer = SvgVectorImporter(SVG_COLORS, Path("colors.svg"))
    spec = PassthroughSpec(
        layer_source=LayerSource.COLORS,
        layer_import_mode=LayerImportMode.NEW_WORKFLOWS,
    )
    import_result = importer.get_doc_items(spec)
    assert import_result is not None
    payload = import_result.payload
    assert payload is not None

    layer = editor.doc.active_layer
    workflows = []
    for info in payload.source_layers:
        workpieces = [
            item
            for item in payload.items
            if isinstance(item, WorkPiece)
            and item.source_segment is not None
            and item.source_segment.layer_id == info.layer_id
        ]
        workflow = editor.workflow.create_for_source_layer(
            layer, info, workpieces
        )
        assert workflow is not None
        layer.add_child(workflow)
        for wp in workpieces:
            layer.add_child(wp)
        workflows.append(workflow)
    return workflows


@pytest.mark.asyncio
async def test_color_rule_creates_configured_step(doc_editor, color_rules):
    """Red maps to EngraveStep; green is unmatched (default Contour)."""
    color_rules.add_preset(
        ColorPreset(color="#ff0000", step_type="EngraveStep")
    )

    workflows = _import_color_workflows(doc_editor)
    StepCmd(doc_editor).add_default_steps_for_workflows(workflows)

    by_color = {workflow.color: workflow for workflow in workflows}
    assert _step_type_names(by_color["#ff0000"]) == ["EngraveStep"]
    assert _step_type_names(by_color["#00ff00"]) == ["ContourStep"]


@pytest.mark.asyncio
async def test_no_rules_uses_default_behavior(doc_editor, color_rules):
    """Without any color rules, the default Contour step is added."""
    workflows = _import_color_workflows(doc_editor)
    StepCmd(doc_editor).add_default_steps_for_workflows(workflows)

    for workflow in workflows:
        assert _step_type_names(workflow) == ["ContourStep"]


@pytest.mark.asyncio
async def test_unavailable_step_type_falls_back(
    doc_editor, color_rules, caplog
):
    """A rule pointing at an unregistered step type falls back gracefully."""
    color_rules.add_preset(
        ColorPreset(color="#ff0000", step_type="DoesNotExistStep")
    )

    workflows = _import_color_workflows(doc_editor)
    with caplog.at_level("WARNING"):
        StepCmd(doc_editor).add_default_steps_for_workflows(workflows)

    red = next(
        workflow for workflow in workflows if workflow.color == "#ff0000"
    )
    assert _step_type_names(red) == ["ContourStep"]
    assert any("DoesNotExistStep" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_rule_not_applied_to_layer_source_import(
    doc_editor, color_rules
):
    """
    Color rules must not match layers imported by SVG layer name, even
    if the layer happens to carry the default palette color.
    """
    color_rules.add_preset(
        ColorPreset(color="#00ccff", step_type="EngraveStep")
    )

    svg = b"""
    <svg width="100mm" height="100mm" viewBox="0 0 100 100"
         xmlns="http://www.w3.org/2000/svg">
        <g id="g1"><rect x="0" y="0" width="10" height="10"/></g>
    </svg>
    """
    importer = SvgVectorImporter(svg, Path("layer.svg"))
    spec = PassthroughSpec(
        layer_source=LayerSource.SVG_LAYERS,
        layer_import_mode=LayerImportMode.NEW_WORKFLOWS,
    )
    import_result = importer.get_doc_items(spec)
    assert import_result is not None
    payload = import_result.payload
    assert payload is not None

    layer = doc_editor.doc.active_layer
    workflows = []
    for info in payload.source_layers:
        workpieces = [
            item
            for item in payload.items
            if isinstance(item, WorkPiece)
            and item.source_segment is not None
            and item.source_segment.layer_id == info.layer_id
        ]
        workflow = doc_editor.workflow.create_for_source_layer(
            layer, info, workpieces
        )
        assert workflow is not None
        layer.add_child(workflow)
        for wp in workpieces:
            layer.add_child(wp)
        workflows.append(workflow)

    StepCmd(doc_editor).add_default_steps_for_workflows(workflows)

    for workflow in workflows:
        # Default behavior, never the color rule's EngraveStep.
        assert "EngraveStep" not in _step_type_names(workflow)


@pytest.mark.asyncio
async def test_color_rule_via_full_commit_flow(
    doc_editor, color_rules, task_mgr
):
    """
    Reproduces the real import path: commit items to the document via
    FileCmd._finalize_import_on_main_thread, then verify the color rule
    was applied to the resulting layers.
    """

    color_rules.add_preset(
        ColorPreset(color="#ff0000", step_type="EngraveStep")
    )

    spec = PassthroughSpec(
        layer_source=LayerSource.COLORS,
        layer_import_mode=LayerImportMode.NEW_WORKFLOWS,
    )
    importer = SvgImporter(SVG_COLORS, Path("colors.svg"))
    import_result = importer.get_doc_items(spec)
    assert import_result is not None
    payload = import_result.payload
    assert payload is not None

    file_cmd = FileCmd(doc_editor, task_mgr)
    file_cmd._finalize_import_on_main_thread(
        payload, Path("colors.svg"), None, spec
    )

    workflows = [
        workflow
        for layer in doc_editor.doc.layers
        for workflow in layer.workflows
    ]
    by_color = {workflow.color: workflow for workflow in workflows}
    assert _step_type_names(by_color["#ff0000"]) == ["EngraveStep"]
    assert _step_type_names(by_color["#00ff00"]) == ["ContourStep"]
