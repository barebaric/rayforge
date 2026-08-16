# flake8: noqa: E402
"""UI tests for the ScenePresenter."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from unittest.mock import MagicMock, patch

import pytest
from blinker import Signal
from raygeo.ops.axis import Axis

from rayforge.machine.models.laser import LaserHead
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.rotary_module import RotaryMode, RotaryModule
from rayforge.simulator.scene3d import (
    CompiledSceneArtifact,
    compile_stock_scene,
)
from rayforge.ui_gtk.sim3d.scene_presenter import ScenePresenter
from rayforge.ui_gtk.sim3d.viewport import ViewportConfig


def _make_presenter(**overrides):
    calls = {"rendered": [], "scene_dirty": [], "artifact_dirty": []}
    defaults = {
        "context": MagicMock(machine=None),
        "doc_editor": MagicMock(),
        "scene": MagicMock(),
        "theme_resolver": MagicMock(
            theme_is_dirty=False,
            color_set=MagicMock(),
        ),
        "get_viewport": lambda: MagicMock(),
        "get_gl_initialized": lambda: True,
        "get_show_travel_moves": lambda: False,
        "get_camera_available": lambda: True,
        "make_current": MagicMock(),
        "mark_scene_dirty": lambda: calls["scene_dirty"].append(True),
        "mark_artifact_dirty": lambda: calls["artifact_dirty"].append(True),
        "reset_view": MagicMock(),
        "request_render": lambda: calls["rendered"].append(True),
        "upload_complete": Signal(),
    }
    defaults.update(overrides)

    presenter = ScenePresenter(
        defaults["context"],
        defaults["doc_editor"],
        defaults["scene"],
        theme_resolver=defaults["theme_resolver"],
        get_viewport=defaults["get_viewport"],
        get_gl_initialized=defaults["get_gl_initialized"],
        get_show_travel_moves=defaults["get_show_travel_moves"],
        get_camera_available=defaults["get_camera_available"],
        make_current=defaults["make_current"],
        mark_scene_dirty=defaults["mark_scene_dirty"],
        mark_artifact_dirty=defaults["mark_artifact_dirty"],
        reset_view=defaults["reset_view"],
        request_render=defaults["request_render"],
        upload_complete=defaults["upload_complete"],
    )
    return presenter, defaults, calls


@pytest.mark.ui
def test_initial_state(ui_context_initializer):
    presenter, _, _ = _make_presenter()
    assert presenter.op_player is None
    assert presenter.compiled_artifact is None
    assert presenter.job_handle is None
    assert presenter.playback_overlay is None
    assert presenter.scene_preparation_task is None


@pytest.mark.ui
def test_update_scene_from_doc_skips_when_not_initialized(
    ui_context_initializer,
):
    presenter, _, calls = _make_presenter(get_gl_initialized=lambda: False)
    presenter.update_scene_from_doc()
    assert calls["scene_dirty"] == []
    assert calls["rendered"] == []


@pytest.mark.ui
def test_update_scene_from_doc_schedules_compilation(ui_context_initializer):
    doc_editor = MagicMock()
    doc_editor.doc.layers = []
    presenter, _, _ = _make_presenter(doc_editor=doc_editor)
    presenter._schedule_scene_preparation = MagicMock()

    presenter.update_scene_from_doc()

    presenter._schedule_scene_preparation.assert_called_once()
    config = presenter._schedule_scene_preparation.call_args.args[0]
    assert "world_to_visual" in config
    assert "world_to_cyl_local" in config


@pytest.mark.ui
def test_update_scene_from_doc_populates_laser_dot_widths(
    ui_context_initializer,
):
    """Laser head spot sizes reach the render config so the 3D raster
    preview can draw scanlines at the physical dot width."""
    head1 = LaserHead()
    head1.spot_size_mm = (0.1, 0.2)
    head2 = LaserHead()
    head2.spot_size_mm = (0.3, 0.4)

    machine = MagicMock()
    machine.heads = [head1, head2]
    machine.assembly = MagicMock()
    machine.assembly.has_rotary = False

    doc_editor = MagicMock()
    doc_editor.doc.layers = []
    presenter, _, _ = _make_presenter(
        context=MagicMock(machine=machine),
        doc_editor=doc_editor,
        get_viewport=lambda: ViewportConfig.default(),
    )
    presenter._schedule_scene_preparation = MagicMock()

    presenter.update_scene_from_doc()

    config = presenter._schedule_scene_preparation.call_args.args[0]
    assert config["laser_dot_widths_mm"] == {
        head1.uid: 0.1,
        head2.uid: 0.3,
    }


@pytest.mark.ui
def test_update_scene_from_doc_dot_widths_empty_without_machine(
    ui_context_initializer,
):
    doc_editor = MagicMock()
    doc_editor.doc.layers = []
    presenter, _, _ = _make_presenter(doc_editor=doc_editor)
    presenter._schedule_scene_preparation = MagicMock()

    presenter.update_scene_from_doc()

    config = presenter._schedule_scene_preparation.call_args.args[0]
    assert config.get("laser_dot_widths_mm") is None


@pytest.mark.ui
def test_on_scene_prepared_ignores_cancelled_task(ui_context_initializer):
    presenter, _, calls = _make_presenter()
    task = MagicMock()
    task.get_status.return_value = "canceled"
    task.is_cancelled.return_value = True

    presenter._on_scene_prepared(task)

    assert presenter.compiled_artifact is None
    assert calls["artifact_dirty"] == []


@pytest.mark.ui
def test_on_scene_prepared_failure_clears_state(ui_context_initializer):
    presenter, _, calls = _make_presenter()
    presenter._compiled_artifact = MagicMock()
    presenter._op_player = MagicMock()
    task = MagicMock()
    task.get_status.return_value = "failed"
    task.is_cancelled.return_value = False

    presenter._on_scene_prepared(task)

    assert presenter.compiled_artifact is None
    assert presenter.op_player is None
    assert calls["artifact_dirty"] == [True]


@pytest.mark.ui
def test_on_scene_prepared_success(ui_context_initializer):
    presenter, _, calls = _make_presenter()
    artifact = MagicMock(spec=CompiledSceneArtifact)
    task = MagicMock()
    task.get_status.return_value = "completed"
    task.result.return_value = artifact

    presenter._on_scene_prepared(task)

    assert presenter.compiled_artifact is artifact
    assert calls["artifact_dirty"] == [True]
    assert calls["rendered"] == [True]


@pytest.mark.ui
def test_on_scene_prepared_null_artifact_clears(ui_context_initializer):
    presenter, _, calls = _make_presenter()
    task = MagicMock()
    task.get_status.return_value = "completed"
    task.result.return_value = None

    presenter._on_scene_prepared(task)

    assert presenter.compiled_artifact is None
    assert calls["artifact_dirty"] == [True]


@pytest.mark.ui
def test_on_playback_layer_changed_builds_assembly(ui_context_initializer):
    machine = Machine(MagicMock())
    rm = RotaryModule()
    rm.set_mode(RotaryMode.TRUE_4TH_AXIS)
    rm.set_axis(Axis.A)
    machine.add_rotary_module(rm)

    doc = MagicMock()
    layer = MagicMock()
    layer.rotary_enabled = True
    layer.rotary_module_uid = rm.uid
    doc.layers = [layer]

    scene = MagicMock()
    presenter, _, calls = _make_presenter(
        context=MagicMock(machine=machine),
        scene=scene,
        doc_editor=MagicMock(doc=doc),
    )
    player = MagicMock()
    player.get_effective_layer.return_value = layer

    presenter._on_playback_layer_changed(player, layer_uid=layer.uid)

    assert presenter.playback_assembly is not None
    assert calls["rendered"] == [True]
    scene.set_cylinder_transform.assert_called_once()


@pytest.mark.ui
def test_on_playback_layer_changed_no_machine(ui_context_initializer):
    scene = MagicMock()
    presenter, _, calls = _make_presenter(scene=scene)
    player = MagicMock()

    presenter._on_playback_layer_changed(player)

    assert presenter.playback_assembly is None
    assert calls["rendered"] == []


@pytest.mark.ui
def test_build_op_player_async_empty_ops_clears_offsets(
    ui_context_initializer,
):
    scene = MagicMock()
    scene.ops_renderers = []
    scene.ring_renderers = []
    presenter, _, calls = _make_presenter(
        context=MagicMock(machine=MagicMock()),
        scene=scene,
    )
    ops = MagicMock()
    ops.is_empty.return_value = True
    presenter._get_ops_for_playback = MagicMock(return_value=ops)

    presenter._build_op_player_async()

    assert presenter.op_player is None
    assert calls["rendered"] == [True]


@pytest.mark.ui
def test_build_op_player_async_preserves_playhead(ui_context_initializer):
    presenter, _, _ = _make_presenter(
        context=MagicMock(machine=MagicMock()),
    )
    presenter._scene.ops_renderers = []
    presenter._scene.ring_renderers = []
    ops = MagicMock()
    ops.is_empty.return_value = False

    previous = MagicMock()
    previous.ops = ops
    previous.current_index = 7
    previous.snapshots = [("snap",)]
    presenter._op_player = previous

    presenter._get_ops_for_playback = MagicMock(return_value=ops)
    presenter._get_time_ops_for_playback = MagicMock(return_value=ops)

    with (
        patch(
            "rayforge.ui_gtk.sim3d.scene_presenter.OpPlayer"
        ) as mock_player_cls,
        patch(
            "rayforge.ui_gtk.sim3d.scene_presenter.task_mgr"
        ) as mock_task_mgr,
    ):
        mock_player = mock_player_cls.return_value
        presenter._build_op_player_async()

    mock_player_cls.assert_called_once_with(
        ops,
        presenter._context.machine,
        presenter.doc,
        build_snapshots=False,
        time_ops=ops,
    )
    mock_player.set_snapshots.assert_called_once_with(previous.snapshots)
    mock_player.seek.assert_called_once_with(7)
    assert presenter.op_player is mock_player
    mock_task_mgr.run_thread.assert_called_once()


@pytest.mark.ui
def test_on_pipeline_state_changed_clears_stale_job(ui_context_initializer):
    doc_editor = MagicMock()
    doc_editor.pipeline = MagicMock()
    doc_editor.pipeline.data_generation_id = 2
    presenter, _, calls = _make_presenter(doc_editor=doc_editor)
    presenter._current_job_handle = MagicMock()
    presenter._current_job_handle.generation_id = 1
    presenter._compiled_artifact = MagicMock()

    presenter._on_pipeline_state_changed(None, is_processing=False)

    assert presenter.job_handle is None
    assert presenter.compiled_artifact is None
    assert calls["artifact_dirty"] == [True]
    assert calls["rendered"] == [True]


@pytest.mark.ui
def test_on_upload_complete_binds_player(ui_context_initializer):
    scene = MagicMock()
    presenter, _, _ = _make_presenter(scene=scene)
    presenter._build_op_player_async = MagicMock()
    presenter._op_player = MagicMock()
    presenter._compiled_artifact = MagicMock()

    presenter._on_upload_complete(None)

    presenter._build_op_player_async.assert_called_once()
    scene.extract_playback_offsets.assert_called_once_with(
        presenter.compiled_artifact
    )


@pytest.mark.ui
def test_has_stale_job(ui_context_initializer):
    doc_editor = MagicMock()
    doc_editor.pipeline = MagicMock()
    doc_editor.pipeline.data_generation_id = 5
    presenter, _, _ = _make_presenter(doc_editor=doc_editor)
    assert presenter.has_stale_job() is True

    handle = MagicMock()
    handle.generation_id = 5
    presenter._current_job_handle = handle
    assert presenter.has_stale_job() is False

    doc_editor.pipeline.data_generation_id = 6
    assert presenter.has_stale_job() is True


@pytest.mark.ui
def test_connect_and_disconnect_subscribe_pipeline(ui_context_initializer):
    presenter, defaults, _ = _make_presenter()
    pipeline = MagicMock()
    defaults["doc_editor"].pipeline = pipeline

    presenter.connect()
    pipeline.processing_state_changed.connect.assert_called_once_with(
        presenter._on_pipeline_state_changed
    )
    pipeline.job_generation_finished.connect.assert_called_once_with(
        presenter._on_job_generation_finished
    )

    presenter.disconnect()
    pipeline.processing_state_changed.disconnect.assert_called_once_with(
        presenter._on_pipeline_state_changed
    )
    pipeline.job_generation_finished.disconnect.assert_called_once_with(
        presenter._on_job_generation_finished
    )


@pytest.mark.ui
def test_cancel_scene_preparation(ui_context_initializer):
    presenter, _, _ = _make_presenter()
    task = MagicMock()
    presenter._scene_preparation_task = task

    presenter.cancel_scene_preparation()

    task.cancel.assert_called_once()
    assert presenter.scene_preparation_task is None


@pytest.mark.ui
def test_schedule_stock_only_compilation_without_job(
    ui_context_initializer,
):
    """Stock is document content: it compiles even without a job."""
    presenter, _, _ = _make_presenter()
    assert presenter._current_job_handle is None
    config = {"world_to_visual": [], "stock_specs": [{"name": "oak"}]}

    with patch(
        "rayforge.ui_gtk.sim3d.scene_presenter.task_mgr"
    ) as mock_task_mgr:
        presenter._schedule_scene_preparation(config)

    mock_task_mgr.run_thread.assert_called_once()
    assert mock_task_mgr.run_thread.call_args.args[0] is compile_stock_scene
    assert mock_task_mgr.run_thread.call_args.args[1] is config


@pytest.mark.ui
def test_schedule_clears_artifact_without_job_or_stock(
    ui_context_initializer,
):
    presenter, _, calls = _make_presenter()
    presenter._current_job_handle = None
    presenter._compiled_artifact = MagicMock()

    with patch(
        "rayforge.ui_gtk.sim3d.scene_presenter.task_mgr"
    ) as mock_task_mgr:
        presenter._schedule_scene_preparation({"stock_specs": []})

    mock_task_mgr.run_thread.assert_not_called()
    assert presenter._compiled_artifact is None
    assert calls["artifact_dirty"] == [True]
    assert calls["rendered"] == [True]


@pytest.mark.ui
def test_update_workpiece_images_skips_when_not_initialized(
    ui_context_initializer,
):
    presenter, _, calls = _make_presenter(get_gl_initialized=lambda: False)
    presenter.update_workpiece_images_from_doc()
    assert calls["rendered"] == []


@pytest.mark.ui
def test_update_workpiece_images_clears_without_workpieces(
    ui_context_initializer,
):
    doc_editor = MagicMock()
    doc_editor.doc.get_descendants.return_value = []
    scene = MagicMock()
    scene.workpiece_image_renderer = MagicMock()
    scene.workpiece_image_renderer.instances = [MagicMock()]
    presenter, _, calls = _make_presenter(doc_editor=doc_editor, scene=scene)

    with patch(
        "rayforge.ui_gtk.sim3d.scene_presenter.task_mgr"
    ) as mock_task_mgr:
        presenter.update_workpiece_images_from_doc()

    mock_task_mgr.run_thread.assert_not_called()
    scene.workpiece_image_renderer.clear.assert_called_once()
    assert calls["rendered"] == [True]


@pytest.mark.ui
def test_update_workpiece_images_schedules_rendering(
    ui_context_initializer,
):
    from rayforge.core.workpiece import WorkPiece

    wp = WorkPiece(name="photo")
    wp._source_segment = MagicMock()
    wp.set_size(100, 50)

    doc_editor = MagicMock()
    doc_editor.doc.get_descendants.return_value = [wp]
    doc_editor.doc.get_asset_by_uid.return_value = None
    scene = MagicMock()
    scene.workpiece_image_renderer = MagicMock()
    scene.workpiece_image_renderer.instances = []
    presenter, _, _ = _make_presenter(
        doc_editor=doc_editor,
        scene=scene,
        context=MagicMock(machine=None),
    )

    with patch(
        "rayforge.ui_gtk.sim3d.scene_presenter.task_mgr"
    ) as mock_task_mgr:
        presenter.update_workpiece_images_from_doc()

    mock_task_mgr.run_thread.assert_called_once()
    _, workpieces, matrices = mock_task_mgr.run_thread.call_args.args
    assert workpieces == [wp]
    assert len(matrices) == 1
    assert matrices[0].shape == (4, 4)


@pytest.mark.ui
def test_update_workpiece_images_skips_hidden_provider(
    ui_context_initializer,
):
    from rayforge.core.workpiece import WorkPiece

    wp = WorkPiece(name="hidden")
    wp._source_segment = MagicMock()
    wp.geometry_provider_uid = "provider-1"

    doc_editor = MagicMock()
    doc_editor.doc.get_descendants.return_value = [wp]
    hidden_provider = MagicMock()
    hidden_provider.hidden = True
    doc_editor.doc.get_asset_by_uid.return_value = hidden_provider
    scene = MagicMock()
    scene.workpiece_image_renderer = MagicMock()
    scene.workpiece_image_renderer.instances = []
    presenter, _, _ = _make_presenter(doc_editor=doc_editor, scene=scene)

    with patch(
        "rayforge.ui_gtk.sim3d.scene_presenter.task_mgr"
    ) as mock_task_mgr:
        presenter.update_workpiece_images_from_doc()

    mock_task_mgr.run_thread.assert_not_called()


@pytest.mark.ui
def test_on_workpiece_images_ready_uploads(ui_context_initializer):
    scene = MagicMock()
    scene.workpiece_image_renderer = MagicMock()
    presenter, defaults, calls = _make_presenter(scene=scene)
    presenter._workpiece_image_generation = 3
    task = MagicMock()
    task.get_status.return_value = "completed"
    images = [{"pixels": MagicMock(), "model_matrix": MagicMock()}]
    task.result.return_value = images

    presenter._on_workpiece_images_ready(3, task)

    scene.workpiece_image_renderer.set_images.assert_called_once_with(images)
    defaults["make_current"].assert_called_once()
    assert calls["rendered"] == [True]


@pytest.mark.ui
def test_on_workpiece_images_ready_ignores_stale_generation(
    ui_context_initializer,
):
    scene = MagicMock()
    scene.workpiece_image_renderer = MagicMock()
    presenter, _, calls = _make_presenter(scene=scene)
    presenter._workpiece_image_generation = 4
    task = MagicMock()
    task.get_status.return_value = "completed"

    presenter._on_workpiece_images_ready(3, task)

    scene.workpiece_image_renderer.set_images.assert_not_called()
    assert calls["rendered"] == []
