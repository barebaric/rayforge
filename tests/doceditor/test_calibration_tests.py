"""
End-to-end tests for the calibration tests (Material Test Grid,
Interval Test, Focus Test): the generated layer, its steps, rebuilding,
undo, saving, and the G-code it produces on GRBL.
"""

import importlib
import re
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from rayforge.context import get_context
from rayforge.core.group import Group
from rayforge.core.undo import ChangePropertyCommand
from rayforge.core.vectorization_spec import ProceduralSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image.procedural import ProceduralImporter
from rayforge.machine.models.machine import Origin
from rayforge.pipeline.artifact import JobArtifact

DATA = Path(__file__).parent / "data" / "calibration_tests"

# The laser addon is importable through the sys.path setup of the root
# conftest; static analysis does not see that path.
_calibration = importlib.import_module("laser_essentials.calibration")
_command = importlib.import_module("laser_essentials.commands.calibration_cmd")
FocusMode = _calibration.FocusMode
CalibrationCmd = _command.CalibrationCmd
_sync = importlib.import_module("laser_essentials.commands.calibration_sync")
CalibrationSync = _sync.CalibrationSync
sync_grid_preview = _sync.sync_grid_preview


def _steps_module():
    return importlib.import_module("laser_essentials.steps")


@pytest.fixture
def machine(context_initializer):
    machine = get_context().machine
    assert machine is not None
    machine.set_dialect_uid("grbl")
    machine.set_origin(Origin.BOTTOM_LEFT)
    machine.set_axis_extents(400, 400)
    return machine


@pytest_asyncio.fixture(autouse=True)
async def settle_editor(doc_editor):
    yield
    await doc_editor.wait_until_settled()


async def _gcode(editor) -> str:
    await editor.wait_until_settled()
    pipeline = editor.pipeline
    handle = await pipeline.generate_job_artifact_async()
    assert handle is not None
    with pipeline.artifact_store.checkout_handle(handle) as artifact:
        assert isinstance(artifact, JobArtifact)
        assert artifact.machine_code is not None
        text = artifact.machine_code
    await editor.wait_until_settled()
    return text


def _assert_golden(gcode: str, name: str):
    expected = (DATA / name).read_text(encoding="utf-8")
    assert gcode.strip().splitlines() == expected.strip().splitlines()


def _steps(layer):
    assert layer.workflow is not None
    return list(layer.workflow.steps)


def _cut_moves(gcode: str) -> list[tuple[float, float]]:
    """End points of all G1 moves, tracking modal X/Y."""
    x = y = 0.0
    points = []
    for line in gcode.splitlines():
        code = line.split(";")[0]
        words = dict(re.findall(r"([GXY])([-+]?\d*\.?\d+)", code))
        if "X" in words:
            x = float(words["X"])
        if "Y" in words:
            y = float(words["Y"])
        if words.get("G") == "1":
            points.append((x, y))
    return points


def _layer(editor, record):
    layer = editor.doc.find_descendant_by_uid(record.layer_uid)
    assert layer is not None
    return layer


def _bounds(workpieces):
    rects = [
        wp.get_world_geometry().rect()  # type: ignore[union-attr]
        for wp in workpieces
    ]
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def _records(editor):
    return [
        a
        for a in editor.doc.get_all_assets()
        if a.asset_type_name == "calibration_test"
    ]


# ----------------------------------------------------------------------
# Shared behaviour of all three tests
# ----------------------------------------------------------------------


ALL_TESTS = ["material_test_grid", "interval_test", "focus_test"]


@pytest.mark.asyncio
@pytest.mark.parametrize("generator_id", ALL_TESTS)
async def test_each_test_gets_its_own_active_layer(
    doc_editor, machine, generator_id
):
    cmd = CalibrationCmd(doc_editor)
    before = list(doc_editor.doc.layers)
    record = cmd.create(generator_id)

    layer = _layer(doc_editor, record)
    assert layer not in before
    assert doc_editor.doc.active_layer is layer
    assert layer.name == cmd.generator(generator_id).title
    assert layer.stock_material_uid is not None
    assert layer.color not in {other.color for other in before}
    assert record in _records(doc_editor)
    assert record.hidden


@pytest.mark.asyncio
@pytest.mark.parametrize("generator_id", ALL_TESTS)
async def test_every_step_is_linked_to_its_own_workpiece(
    doc_editor, machine, generator_id
):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create(generator_id)
    layer = _layer(doc_editor, record)

    workpieces = {wp.uid for wp in layer.all_workpieces}
    assert set(record.state.workpiece_uids) == workpieces
    linked = [s for s in _steps(layer) if s.workpiece_uids]
    assert len(linked) == len(workpieces)
    assert {s.workpiece_uids[0] for s in linked} == workpieces
    assert [s.uid for s in _steps(layer)] == list(record.state.step_uids)


@pytest.mark.asyncio
@pytest.mark.parametrize("generator_id", ALL_TESTS)
async def test_create_is_centered_and_one_undo_step(
    doc_editor, machine, generator_id
):
    cmd = CalibrationCmd(doc_editor)
    layers_before = list(doc_editor.doc.layers)
    assets_before = set(doc_editor.doc.assets)
    active_before = doc_editor.doc.active_layer
    record = cmd.create(generator_id)
    layer = _layer(doc_editor, record)

    min_x, min_y, max_x, max_y = _bounds(cmd.workpieces(record))
    center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
    assert center == pytest.approx((200.0, 200.0), abs=0.01)

    doc_editor.history_manager.undo()
    assert list(doc_editor.doc.layers) == layers_before
    assert set(doc_editor.doc.assets) == assets_before
    assert doc_editor.doc.active_layer is active_before

    doc_editor.history_manager.redo()
    assert layer in doc_editor.doc.layers
    assert record in _records(doc_editor)


@pytest.mark.asyncio
async def test_layer_names_stay_unique(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    first = _layer(doc_editor, cmd.create("interval_test"))
    second = _layer(doc_editor, cmd.create("interval_test"))
    assert first.name == "Interval Test"
    assert second.name == "Interval Test 2"


@pytest.mark.asyncio
async def test_update_keeps_position_and_is_one_undo_step(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("interval_test", {"count": 3})
    for wp in cmd.workpieces(record):
        wp.pos = (wp.pos[0] + 30.0, wp.pos[1] - 20.0)
    corner = _bounds(cmd.workpieces(record))[:2]
    old_uids = set(record.state.workpiece_uids)
    undo_depth = len(doc_editor.history_manager.undo_stack)

    params = record.params
    params["count"] = 6
    new_wps = cmd.update_params(record, params)

    layer = _layer(doc_editor, record)
    assert len(doc_editor.history_manager.undo_stack) == undo_depth + 1
    assert record.params["count"] == 6
    assert len(new_wps) == 7
    assert set(record.state.workpiece_uids).isdisjoint(old_uids)
    assert {wp.uid for wp in layer.all_workpieces} == {
        wp.uid for wp in new_wps
    }
    assert _bounds(new_wps)[:2] == pytest.approx(corner, abs=0.01)

    doc_editor.history_manager.undo()
    assert record.params["count"] == 3
    assert {wp.uid for wp in layer.all_workpieces} == old_uids
    assert len(_steps(layer)) == 4


@pytest.mark.asyncio
async def test_update_stays_on_the_work_area(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("focus_test", {"mode": "ramp", "ramp_length": 50.0})
    for wp in cmd.workpieces(record):
        wp.pos = (wp.pos[0] + 120.0, wp.pos[1])
    params = record.params
    params["ramp_length"] = 300.0
    workpieces = cmd.update_params(record, params)
    min_x, _min_y, max_x, _max_y = _bounds(workpieces)
    assert max_x == pytest.approx(400.0, abs=0.01)
    assert min_x > 0


@pytest.mark.asyncio
async def test_create_needs_a_machine(doc_editor, machine, mocker):
    mocker.patch.object(
        type(doc_editor.context), "machine", new_callable=mocker.PropertyMock
    ).return_value = None
    layers = list(doc_editor.doc.layers)
    with pytest.raises(ValueError):
        CalibrationCmd(doc_editor).create("interval_test")
    assert list(doc_editor.doc.layers) == layers


@pytest.mark.asyncio
async def test_update_after_the_layer_was_deleted(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("interval_test")
    doc_editor.doc.remove_child(_layer(doc_editor, record))
    with pytest.raises(ValueError):
        cmd.update_params(record, {**record.params, "count": 3})


@pytest.mark.asyncio
async def test_update_after_all_cells_were_deleted(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("interval_test", {"count": 2})
    layer = _layer(doc_editor, record)
    for wp in cmd.workpieces(record):
        layer.remove_child(wp)
    workpieces = cmd.update_params(record, {**record.params, "count": 3})
    min_x, min_y, max_x, max_y = _bounds(workpieces)
    center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
    assert center == pytest.approx((200.0, 200.0), abs=0.01)
    assert len(layer.all_workpieces) == 4


@pytest.mark.asyncio
async def test_update_keeps_steps_the_user_added(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("interval_test", {"count": 2})
    layer = _layer(doc_editor, record)
    extra = _steps(layer)[0].__class__.create(doc_editor.context)
    extra.name = "Cut out"
    layer.workflow.add_step(extra)

    params = record.params
    params["count"] = 4
    cmd.update_params(record, params)

    steps = _steps(layer)
    assert steps[-1] is extra
    assert len(steps) == 1 + 4 + 1


@pytest.mark.asyncio
async def test_update_keeps_a_user_step_between_the_cells(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("interval_test", {"count": 2})
    layer = _layer(doc_editor, record)
    extra = _steps(layer)[0].__class__.create(doc_editor.context)
    layer.workflow.add_child(extra, index=2)

    cmd.update_params(record, {**record.params, "count": 3})

    steps = _steps(layer)
    assert steps.index(extra) == 4
    assert len(steps) == 5
    doc_editor.history_manager.undo()
    assert _steps(layer).index(extra) == 3


@pytest.mark.asyncio
async def test_update_and_undo_keep_cells_in_their_group(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("interval_test", {"count": 2})
    layer = _layer(doc_editor, record)
    cells = cmd.workpieces(record)
    group = Group("Cells")
    layer.add_child(group)
    for wp in cells[1:]:
        group.add_child(wp)

    new = cmd.update_params(record, {**record.params, "count": 3})
    assert all(wp.parent is layer for wp in new)
    assert group.children == []

    doc_editor.history_manager.undo()
    assert [wp.parent for wp in cells] == [layer, group, group]
    doc_editor.history_manager.redo()
    assert [wp.parent for wp in new] == [layer] * 4
    assert all(wp.parent is None for wp in cells)


@pytest.mark.asyncio
async def test_invalid_params_leave_the_test_alone(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("interval_test")
    state = record.state
    undo_depth = len(doc_editor.history_manager.undo_stack)

    params = record.params
    params["min_interval"] = params["max_interval"]
    with pytest.raises(ValueError):
        cmd.update_params(record, params)
    assert record.state == state
    assert len(doc_editor.history_manager.undo_stack) == undo_depth


@pytest.mark.asyncio
async def test_update_without_change_does_nothing(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("focus_test")
    undo_depth = len(doc_editor.history_manager.undo_stack)
    workpieces = cmd.update_params(record, record.params)
    assert len(doc_editor.history_manager.undo_stack) == undo_depth
    assert [wp.uid for wp in workpieces] == list(record.state.workpiece_uids)


@pytest.mark.asyncio
async def test_update_removes_old_assets_unless_still_used(
    doc_editor, machine
):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("interval_test", {"count": 2})
    layer = _layer(doc_editor, record)
    cells = cmd.workpieces(record)
    copy = cells[-1].duplicate()
    layer.add_child(copy)
    kept = copy.source_segment.source_asset_uid
    old_assets = set(record.state.asset_uids)

    params = record.params
    params["count"] = 3
    cmd.update_params(record, params)

    assert kept in doc_editor.doc.assets
    gone = old_assets - {kept}
    assert gone
    assert gone.isdisjoint(doc_editor.doc.assets)
    assert set(record.state.asset_uids) <= set(doc_editor.doc.assets)


@pytest.mark.asyncio
async def test_record_survives_save_and_load(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("focus_test", {"count": 4})
    data = doc_editor.doc.to_dict()

    loaded = type(doc_editor.doc).from_dict(data)
    records = [
        a
        for a in loaded.get_all_assets()
        if a.asset_type_name == "calibration_test"
    ]
    assert len(records) == 1
    copy = records[0]
    assert copy.uid == record.uid
    assert copy.generator_id == "focus_test"
    assert copy.params == record.params
    for name in ("params", "workpiece_uids", "step_uids", "asset_uids"):
        assert getattr(copy.state, name) == getattr(record.state, name)
    items = [
        loaded.find_descendant_by_uid(u) for u in copy.state.workpiece_uids
    ]
    assert CalibrationCmd.record_for_items(items) is copy


@pytest.mark.asyncio
async def test_deleting_the_layer_removes_the_record(doc_editor, machine):
    """The record and hidden assets of a test leave with its layer, are
    not saved, and come back when the deletion is undone."""
    sync = CalibrationSync(doc_editor)
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("interval_test", {"count": 2})
    owned = {record.uid, *record.state.asset_uids}
    assert owned <= set(doc_editor.doc.assets)

    doc_editor.layer.delete_layer(_layer(doc_editor, record))

    assert owned.isdisjoint(doc_editor.doc.assets)
    saved = doc_editor.doc.to_dict()["assets"]
    assert owned.isdisjoint(a["uid"] for a in saved)

    doc_editor.history_manager.undo()
    assert owned <= set(doc_editor.doc.assets)
    workpieces = cmd.workpieces(record)
    assert cmd.record_for_items(workpieces) is record
    doc_editor.history_manager.redo()
    assert owned.isdisjoint(doc_editor.doc.assets)
    assert sync is not None


@pytest.mark.asyncio
async def test_undoing_a_new_test_leaves_nothing(doc_editor, machine):
    sync = CalibrationSync(doc_editor)
    assets = set(doc_editor.doc.assets)
    record = CalibrationCmd(doc_editor).create("focus_test")
    doc_editor.history_manager.undo()
    assert set(doc_editor.doc.assets) == assets
    doc_editor.history_manager.redo()
    assert record.uid in doc_editor.doc.assets
    assert set(record.state.asset_uids) <= set(doc_editor.doc.assets)
    assert sync is not None


@pytest.mark.asyncio
async def test_sync_follows_a_new_document(doc_editor, machine):
    sync = CalibrationSync(doc_editor)
    old_doc = doc_editor.doc
    doc_editor.set_doc(type(old_doc)())
    record = CalibrationCmd(doc_editor).create("interval_test")
    doc_editor.layer.delete_layer(_layer(doc_editor, record))
    assert record.uid not in doc_editor.doc.assets
    assert sync is not None


@pytest.mark.asyncio
async def test_sync_ignores_other_changes(doc_editor, machine):
    """Removing a layer without a test, or a workpiece, leaves the assets
    alone; a grid step without a preview is not touched."""
    sync = CalibrationSync(doc_editor)
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("material_test_grid")
    assets = set(doc_editor.doc.assets)
    (wp,) = cmd.workpieces(record)
    other = doc_editor.doc.layers[0]

    doc_editor.layer.delete_layer(other)
    _layer(doc_editor, record).remove_child(wp)
    assert set(doc_editor.doc.assets) == assets

    (step,) = _steps(_layer(doc_editor, record))
    step.grid_dimensions = (3, 3)
    step.updated.send(step)
    step.workpiece_uids = []
    step.updated.send(step)
    assert sync is not None


@pytest.mark.asyncio
async def test_grid_preview_with_broken_settings_is_left_alone(
    doc_editor, machine
):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("material_test_grid")
    (wp,) = cmd.workpieces(record)
    (step,) = _steps(_layer(doc_editor, record))
    wp.source.original_data = b"not json"
    step.grid_dimensions = (3, 3)
    sync_grid_preview(doc_editor.doc, step)
    assert wp.size == pytest.approx((73.0, 73.0))


@pytest.mark.asyncio
async def test_steps_are_replaced_when_they_do_not_match(doc_editor, machine):
    reuse = _command._reuse_steps
    generator = CalibrationCmd.generator("material_test_grid")
    contour = importlib.import_module("laser_essentials.steps").ContourStep()
    built = SimpleNamespace(steps=[_steps_module().MaterialTestStep()])
    assert reuse(generator, [contour], built) == []
    assert reuse(generator, [], built) == []
    interval = CalibrationCmd.generator("interval_test")
    assert interval.step_settings(contour) == {}
    assert generator.params_from_steps({"a": 1}, []) == {"a": 1}


@pytest.mark.asyncio
async def test_record_for_items(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    interval = cmd.create("interval_test")
    focus = cmd.create("focus_test")
    interval_wps = cmd.workpieces(interval)
    focus_wps = cmd.workpieces(focus)

    assert cmd.record_for_items(interval_wps) is interval
    assert cmd.record_for_items(focus_wps[:1]) is focus
    assert cmd.record_for_items(interval_wps + focus_wps) is None
    assert cmd.record_for_items([]) is None
    assert cmd.record_for_items([_layer(doc_editor, focus)]) is None


# ----------------------------------------------------------------------
# Material Test Grid
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_material_test_grid_layer(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create(
        "material_test_grid",
        {"grid_dimensions": (3, 4), "shape_size": 5.0, "spacing": 1.0},
    )
    layer = _layer(doc_editor, record)
    (step,) = _steps(layer)
    (wp,) = layer.all_workpieces

    assert step.typelabel == "Material Test Grid"
    assert step.workpiece_uids == [wp.uid]
    assert step.grid_dimensions == (3, 4)
    assert step.line_interval_mm is None
    width = 3 * 5.0 + 2 * 1.0 + 7.5
    height = 4 * 5.0 + 3 * 1.0 + 7.5
    assert wp.size == pytest.approx((width, height))


@pytest.mark.asyncio
async def test_material_test_grid_update_resizes(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("material_test_grid")
    params = record.params
    params["grid_dimensions"] = (2, 2)
    params["include_labels"] = False
    (wp,) = cmd.update_params(record, params)
    (step,) = _steps(_layer(doc_editor, record))
    assert step.grid_dimensions == (2, 2)
    assert step.include_labels is False
    assert wp.size == pytest.approx((2 * 10.0 + 2.0, 2 * 10.0 + 2.0))


def _old_style_grid(editor, grid_params):
    """A grid built the way Tools → Create Material Test Grid did it
    before it got its own layer: on the active layer, centered."""
    steps = importlib.import_module("laser_essentials.steps")
    helpers = importlib.import_module("laser_essentials.material_test_helpers")
    step = steps.MaterialTestStep.create(editor.context)
    step.name = "Material Test Grid"
    for key, value in grid_params.items():
        setattr(step, key, value)
    draw = helpers.draw_material_test_preview
    size = helpers.get_material_test_proportional_size
    importer = ProceduralImporter(
        drawing_function_path=f"{draw.__module__}.{draw.__name__}",
        size_function_path=f"{size.__module__}.{size.__name__}",
        params={
            k: getattr(step, k)
            for k in importlib.import_module(
                "laser_essentials.calibration.material_test_grid"
            ).PARAM_KEYS
        },
        name=step.name,
    )
    result = importer.get_doc_items(ProceduralSpec())
    assert result is not None and result.payload is not None
    payload = result.payload
    editor.doc.add_asset(payload.source)
    wp = payload.items[0]
    step.workpiece_uids = [wp.uid]
    width, height = wp.size
    wp.pos = (200 - width / 2, 200 - height / 2)
    layer = editor.doc.active_layer
    layer.workflow.add_step(step)
    layer.add_child(wp)


@pytest.mark.asyncio
async def test_material_test_grid_gcode_matches_the_old_grid(
    doc_editor, machine
):
    """The grid on its own layer runs exactly like a grid made the old
    way on the active layer."""
    params = {"grid_dimensions": (3, 2), "test_type": "Engrave"}
    cmd = CalibrationCmd(doc_editor)
    cmd.create("material_test_grid", params)
    generated = await _gcode(doc_editor)
    doc_editor.history_manager.undo()

    _old_style_grid(doc_editor, params)
    old = await _gcode(doc_editor)
    assert generated.count("M4") > 6
    assert generated.splitlines() == old.splitlines()


def _apply_recipe_values(editor, step, values):
    """Change step settings the way the recipe control does."""
    with editor.history_manager.transaction("Apply Recipe") as t:
        for key, value in values.items():
            t.execute(
                ChangePropertyCommand(
                    target=step,
                    property_name=key,
                    new_value=value,
                    on_change_callback=lambda: step.updated.send(step),
                )
            )


@pytest.mark.asyncio
async def test_material_test_grid_follows_a_recipe(doc_editor, machine):
    """A recipe applied to the grid's step resizes the preview, shows in
    the Test Settings, and stays when another setting changes."""
    sync = CalibrationSync(doc_editor)
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("material_test_grid")
    (step,) = _steps(_layer(doc_editor, record))
    (wp,) = cmd.workpieces(record)
    assert wp.size == pytest.approx((73.0, 73.0))

    _apply_recipe_values(doc_editor, step, {"grid_dimensions": [6, 6]})

    (wp,) = cmd.workpieces(record)
    assert wp.size == pytest.approx((6 * 10 + 5 * 2 + 15,) * 2)
    params = cmd.params(record)
    assert tuple(params["grid_dimensions"]) == (6, 6)

    cmd.update_params(record, {**params, "spacing": 3.0})
    assert tuple(step.grid_dimensions) == (6, 6)
    (wp,) = cmd.workpieces(record)
    assert wp.size == pytest.approx((6 * 10 + 5 * 3 + 15,) * 2)

    doc_editor.history_manager.undo()
    doc_editor.history_manager.undo()
    (wp,) = cmd.workpieces(record)
    assert tuple(step.grid_dimensions) == (5, 5)
    assert wp.size == pytest.approx((73.0, 73.0))
    assert sync is not None


@pytest.mark.asyncio
async def test_material_test_grid_keeps_its_operation(doc_editor, machine):
    """Changing the grid keeps the step and what was set in its own
    settings, such as air assist."""
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("material_test_grid")
    layer = _layer(doc_editor, record)
    (step,) = _steps(layer)
    step.air_assist = True

    cmd.update_params(record, {**cmd.params(record), "shape_size": 8.0})

    assert _steps(layer) == [step]
    assert step.air_assist is True
    assert step.shape_size == 8.0
    (wp,) = cmd.workpieces(record)
    assert step.workpiece_uids == [wp.uid]
    doc_editor.history_manager.undo()
    assert step.shape_size == 10.0
    (old_wp,) = cmd.workpieces(record)
    assert step.workpiece_uids == [old_wp.uid]


@pytest.mark.asyncio
async def test_legacy_material_test_grid_gets_a_record(doc_editor, machine):
    """A grid from an older version (step and workpiece on an ordinary
    layer, no record) can be changed like a new one."""
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("material_test_grid")
    layer = _layer(doc_editor, record)
    doc_editor.doc.remove_asset(record)
    (wp,) = layer.all_workpieces
    (step,) = _steps(layer)
    step.grid_dimensions = (3, 3)

    legacy = cmd.record_for_items([wp])
    assert legacy is not None
    assert legacy not in _records(doc_editor)
    assert legacy.params["grid_dimensions"] == (3, 3)
    assert legacy.params["line_interval_mm"] == 0.0

    params = legacy.params
    params["grid_dimensions"] = (4, 2)
    (new_wp,) = cmd.update_params(legacy, params)
    assert legacy in _records(doc_editor)
    (new_step,) = _steps(layer)
    assert new_step.grid_dimensions == (4, 2)
    assert new_step.workpiece_uids == [new_wp.uid]
    assert cmd.record_for_items([new_wp]) is legacy

    doc_editor.history_manager.undo()
    assert legacy not in _records(doc_editor)
    assert _steps(layer) == [step]


# ----------------------------------------------------------------------
# Interval Test
# ----------------------------------------------------------------------


INTERVAL_PARAMS = {
    "min_interval": 0.5,
    "max_interval": 1.0,
    "count": 2,
    "cell_size": 4.0,
    "spacing": 2.0,
    "power_percent": 40.0,
    "speed": 2000,
    "include_labels": False,
}


@pytest.mark.asyncio
async def test_interval_test_layer(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("interval_test", {"count": 3})
    layer = _layer(doc_editor, record)

    steps = _steps(layer)
    # One label step, then one engrave step per cell.
    assert [s.typelabel for s in steps] == [
        "Contour",
        "Engrave",
        "Engrave",
        "Engrave",
    ]
    assert len(layer.all_workpieces) == 4
    intervals = [s.line_interval_mm for s in steps[1:]]
    assert intervals == pytest.approx([0.05, 0.15, 0.25])
    assert [s.power for s in steps[1:]] == pytest.approx([0.3] * 3)
    assert steps[0].power == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_interval_test_scan_line_pitch(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("interval_test", INTERVAL_PARAMS)
    cells = sorted(
        (
            wp
            for wp in _layer(doc_editor, record).all_workpieces
            if isinstance(wp, WorkPiece)
        ),
        key=lambda wp: wp.pos[0],
    )
    gcode = await _gcode(doc_editor)

    for cell, interval in zip(cells, (0.5, 1.0)):
        x0, y0, x1, y1 = cell.get_world_geometry().rect()  # type: ignore
        ys = sorted(
            {
                round(y, 3)
                for x, y in _cut_moves(gcode)
                if x0 - 0.01 <= x <= x1 + 0.01 and y0 <= y <= y1
            }
        )
        assert len(ys) >= 3
        pitches = [b - a for a, b in pairwise(ys)]
        assert pitches == pytest.approx([interval] * len(pitches), abs=0.01)
    # Every cell is filled at the same, constant power (40 % of S1000).
    powers = set(re.findall(r"^M4 S(\d+)$", gcode, re.MULTILINE))
    assert powers == {"400"}
    _assert_golden(gcode, "interval_test.gcode")


# ----------------------------------------------------------------------
# Focus Test
# ----------------------------------------------------------------------


def _focus(mode, **kwargs) -> dict:
    values = {
        "mode": mode.value,
        "start_offset": -0.5,
        "step": 0.5,
        "count": 3,
        "line_length": 5.0,
        "spacing": 3.0,
        "power_percent": 20.0,
        "speed": 1200,
        "include_labels": False,
    }
    values.update(kwargs)
    return values


def _modes(doc_editor):
    module = importlib.import_module("laser_essentials.calibration.focus_test")
    return module.available_modes(doc_editor.context)


@pytest.mark.asyncio
async def test_focus_modes_without_z_axis(doc_editor, machine, mocker):
    mocker.patch.object(
        type(machine), "has_z_axis", new_callable=mocker.PropertyMock
    ).return_value = False
    cmd = CalibrationCmd(doc_editor)
    modes = _modes(doc_editor)
    assert modes == [FocusMode.MANUAL, FocusMode.RAMP]
    record = cmd.create("focus_test")
    assert record.params["mode"] == FocusMode.MANUAL.value
    with pytest.raises(ValueError):
        cmd.create("focus_test", _focus(FocusMode.Z_AXIS))


@pytest.mark.asyncio
async def test_focus_modes_without_gcode_driver(doc_editor, machine, mocker):
    mocker.patch.object(machine, "driver_name", "RuidaRPAAdapter")
    assert _modes(doc_editor) == [FocusMode.RAMP]
    generator = CalibrationCmd.generator("focus_test")
    assert generator.defaults(doc_editor.context)["mode"] == "ramp"


@pytest.mark.asyncio
async def test_focus_manual_pauses_between_lines(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("focus_test", _focus(FocusMode.MANUAL))
    names = [s.name for s in _steps(_layer(doc_editor, record))]
    assert len(names) == 6
    assert names[0].startswith("Pause")
    assert names[1] == "Focus -0.5 mm"
    assert names[3] == "Focus 0.0 mm"
    assert names[5] == "Focus +0.5 mm"

    gcode = await _gcode(doc_editor)
    lines = [line.strip() for line in gcode.splitlines()]
    assert lines.count("M0") == 3
    # Every pause comes after the laser was switched off.
    for index, line in enumerate(lines):
        if line == "M0":
            previous = [p for p in lines[:index] if p.startswith("M")]
            assert not previous or previous[-1].startswith("M5")
    _assert_golden(gcode, "focus_manual.gcode")


@pytest.mark.asyncio
async def test_focus_z_axis_moves_and_returns(doc_editor, machine, mocker):
    mocker.patch.object(
        type(machine), "has_z_axis", new_callable=mocker.PropertyMock
    ).return_value = True
    cmd = CalibrationCmd(doc_editor)
    assert _modes(doc_editor)[0] == FocusMode.Z_AXIS
    record = cmd.create("focus_test")
    assert record.params["mode"] == FocusMode.Z_AXIS.value
    doc_editor.history_manager.undo()
    cmd.create("focus_test", _focus(FocusMode.Z_AXIS))

    gcode = await _gcode(doc_editor)
    z_moves = [
        float(m.group(1))
        for m in re.finditer(r"^G0 Z([-+]?\d*\.?\d+)$", gcode, re.MULTILINE)
    ]
    assert z_moves == pytest.approx([-0.5, 0.5, 0.5, -0.5])
    assert sum(z_moves) == pytest.approx(0.0)
    assert gcode.count("G91") == 4
    _assert_golden(gcode, "focus_z_axis.gcode")


@pytest.mark.asyncio
async def test_focus_ramp_has_no_commands(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create(
        "focus_test",
        _focus(FocusMode.RAMP, ramp_length=30.0, include_labels=True),
    )
    steps = _steps(_layer(doc_editor, record))
    assert [s.typelabel for s in steps] == ["Contour", "Contour"]
    gcode = await _gcode(doc_editor)
    assert "M0" not in gcode
    assert "G91" not in gcode


@pytest.mark.asyncio
async def test_focus_gcode_follows_a_rebuild(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("focus_test", _focus(FocusMode.MANUAL))
    assert (await _gcode(doc_editor)).count("M0") == 3

    cmd.update_params(record, {**record.params, "count": 5})
    assert (await _gcode(doc_editor)).count("M0") == 5

    doc_editor.history_manager.undo()
    assert (await _gcode(doc_editor)).count("M0") == 3


@pytest.mark.asyncio
async def test_focus_switch_mode_rebuilds_commands(doc_editor, machine):
    cmd = CalibrationCmd(doc_editor)
    record = cmd.create("focus_test", _focus(FocusMode.MANUAL))
    layer = _layer(doc_editor, record)
    assert len(_steps(layer)) == 6

    params = record.params
    params["mode"] = FocusMode.RAMP.value
    cmd.update_params(record, params)
    assert [s.name for s in _steps(layer)] == ["Focus ramp"]
