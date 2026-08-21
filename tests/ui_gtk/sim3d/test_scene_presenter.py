# flake8: noqa: E402
"""UI tests for the ScenePresenter."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from blinker import Signal
from raygeo.compressed_array import CompressedArray
from raygeo.geo import Geometry, Matrix
from raygeo.ops.axis import Axis
from raygeo.ops.material import RasterEffect
from raygeo.ops.material.fold import fold_effects
from raygeo.ops.material.spec import (
    FoldEntry,
    MaterialFoldSpec,
    PrismaticStock,
)

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.material import Material
from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset
from rayforge.core.workpiece import WorkPiece
from rayforge.machine.models.laser import LaserHead
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.rotary_module import RotaryMode, RotaryModule
from rayforge.pipeline.artifact.material_state import MaterialStateArtifact
from rayforge.simulator.scene3d import (
    CompiledSceneArtifact,
    compile_stock_scene,
)
from rayforge.ui_gtk.sim3d.scene_presenter import (
    ScenePresenter,
    _render_workpiece_images,
)
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
    assert presenter.visibility.show_grid is True
    assert presenter.visibility.show_travel_moves is False


@pytest.mark.ui
def test_set_show_grid_rerenders(ui_context_initializer):
    presenter, _, calls = _make_presenter()
    presenter.set_show_grid(False)
    assert presenter.visibility.show_grid is False
    assert calls["rendered"] == [True]
    # Setting the same value again is a no-op.
    presenter.set_show_grid(False)
    assert calls["rendered"] == [True]


@pytest.mark.ui
def test_set_show_travel_moves_rebuilds_renderers(ui_context_initializer):
    presenter, defaults, calls = _make_presenter()
    presenter._compiled_artifact = MagicMock()
    presenter.set_show_travel_moves(True)
    assert presenter.visibility.show_travel_moves is True
    assert calls["rendered"] == [True]
    defaults["scene"].update_from_artifact.assert_called_once()


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
    machine.get_default_rotary_module.return_value = None

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
    pipeline.material_state_ready.connect.assert_called_once_with(
        presenter._on_material_state_ready
    )

    presenter.disconnect()
    pipeline.processing_state_changed.disconnect.assert_called_once_with(
        presenter._on_pipeline_state_changed
    )
    pipeline.job_generation_finished.disconnect.assert_called_once_with(
        presenter._on_job_generation_finished
    )
    pipeline.material_state_ready.disconnect.assert_called_once_with(
        presenter._on_material_state_ready
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
    _, workpieces, matrices, rotary_specs = (
        mock_task_mgr.run_thread.call_args.args
    )
    assert workpieces == [wp]
    assert len(matrices) == 1
    assert matrices[0].shape == (4, 4)
    assert rotary_specs == [None]


@pytest.mark.ui
def test_update_workpiece_images_rotary_layer_gets_cylinder_spec(
    ui_context_initializer,
):
    wp = WorkPiece(name="label")
    wp._source_segment = MagicMock()
    wp.set_size(60, 40)

    layer = Layer(name="rot")
    layer.set_rotary_enabled(True)
    layer.set_rotary_diameter(50.0)
    wp.parent = layer

    machine = MagicMock()
    rm = RotaryModule()
    rm.set_mode(RotaryMode.TRUE_4TH_AXIS)
    rm.set_axis(Axis.A)
    machine.get_rotary_module_for_layer.return_value = rm

    doc_editor = MagicMock()
    doc_editor.doc.get_descendants.return_value = [wp]
    doc_editor.doc.get_asset_by_uid.return_value = None
    scene = MagicMock()
    scene.workpiece_image_renderer = MagicMock()
    scene.workpiece_image_renderer.instances = []
    presenter, _, _ = _make_presenter(
        doc_editor=doc_editor,
        scene=scene,
        context=MagicMock(machine=machine),
    )

    with patch(
        "rayforge.ui_gtk.sim3d.scene_presenter.task_mgr"
    ) as mock_task_mgr:
        presenter.update_workpiece_images_from_doc()

    mock_task_mgr.run_thread.assert_called_once()
    _, workpieces, _matrices, rotary_specs = (
        mock_task_mgr.run_thread.call_args.args
    )
    assert workpieces == [wp]
    assert rotary_specs is not None
    assert len(rotary_specs) == 1
    world_matrix, diameter, reverse = rotary_specs[0]
    assert diameter == 50.0
    assert reverse is False
    assert world_matrix.shape == (4, 4)


@pytest.mark.ui
def test_update_workpiece_images_skips_hidden_provider(
    ui_context_initializer,
):
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
def test_render_workpiece_images_wraps_rotary_onto_cylinder(
    ui_context_initializer,
):
    """Rotary workpiece images get cylinder vertices on the surface."""
    wp = WorkPiece(name="label")
    wp._source_segment = MagicMock()
    wp.set_size(100, 50)

    world_matrix = np.eye(4)
    world_matrix[0, 0] = 100.0  # 100 mm along the cylinder axis
    world_matrix[1, 1] = 50.0  # 50 mm of surface -> 114.6 degrees
    world_matrix[0, 3] = 0.0
    world_matrix[1, 3] = -25.0

    with patch(
        "rayforge.ui_gtk.sim3d.scene_presenter._workpiece_image_pixels"
    ) as mock_pixels:
        mock_pixels.return_value = np.zeros((16, 32, 4), dtype=np.uint8)
        images = _render_workpiece_images(
            [wp],
            [np.eye(4, dtype=np.float32)],
            [(world_matrix, 50.0, False)],
        )

    assert len(images) == 1
    image = images[0]
    assert image.cylinder_vertices is not None
    verts = image.cylinder_vertices.reshape(-1, 5)
    assert verts.shape[0] > 0
    # Vertices must sit on the cylinder surface at radius = d/2.
    radii = np.sqrt(verts[:, 1] ** 2 + verts[:, 2] ** 2)
    np.testing.assert_allclose(radii, 25.0, atol=1e-3)
    # The X extent stays the image's world X footprint.
    np.testing.assert_allclose(verts[:, 0].min(), 0.0, atol=1e-3)
    np.testing.assert_allclose(verts[:, 0].max(), 100.0, atol=1e-3)
    assert image.rotary_diameter == 50.0


@pytest.mark.ui
def test_render_workpiece_images_flat_without_rotary_spec(
    ui_context_initializer,
):
    wp = WorkPiece(name="photo")
    wp._source_segment = MagicMock()
    wp.set_size(100, 50)

    with patch(
        "rayforge.ui_gtk.sim3d.scene_presenter._workpiece_image_pixels"
    ) as mock_pixels:
        mock_pixels.return_value = np.zeros((16, 32, 4), dtype=np.uint8)
        images = _render_workpiece_images(
            [wp],
            [np.eye(4, dtype=np.float32)],
            [None],
        )

    assert len(images) == 1
    assert images[0].cylinder_vertices is None
    assert images[0].model_matrix is not None


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


# ── LUT overlay toggle ──────────────────────────────────────────


# ── Folded material states ──────────────────────────────────────


def _burned_state():
    """A real MaterialState with a raster-burn surface map."""
    effect = RasterEffect(
        np.full((10, 10), 255, dtype=np.uint8),
        origin_mm=(0.0, 0.0),
        px_per_mm=(10.0, 10.0),
    )
    spec = MaterialFoldSpec(
        stock=PrismaticStock(
            polygons=[
                [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
            ],
            thickness=3.0,
        ),
        entries=[FoldEntry("w1", Matrix.identity(), [effect])],
    )
    return fold_effects(spec)


def _empty_state():
    spec = MaterialFoldSpec(
        stock=PrismaticStock(
            polygons=[
                [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
            ],
            thickness=3.0,
        ),
        entries=[],
    )
    return fold_effects(spec)


@pytest.mark.ui
def test_on_material_state_ready_stores_burn(ui_context_initializer):
    presenter, defaults, _ = _make_presenter()
    presenter.update_scene_from_doc = MagicMock()

    state = _burned_state()
    assert state.surface_map is not None
    assert state.grid is not None
    defaults[
        "context"
    ].artifact_store.get.return_value = MaterialStateArtifact(
        material_state=state, stock_uid="s1", generation_id=1
    )
    stock_item = MagicMock()
    stock_item.uid = "s1"

    presenter._on_material_state_ready(
        None, item=stock_item, handle=MagicMock(), generation_id=1
    )

    burn = presenter._material_states.get("s1")
    assert burn is not None
    np.testing.assert_array_equal(
        burn["surface_map"].to_numpy(), state.surface_map.to_numpy()
    )
    assert burn["origin_mm"] == tuple(state.grid.origin_mm)
    assert burn["px_per_mm"] == tuple(state.grid.px_per_mm)
    assert burn["size_px"] == tuple(state.grid.size_px)
    presenter.update_scene_from_doc.assert_called_once()


@pytest.mark.ui
def test_on_material_state_ready_drops_empty_state(ui_context_initializer):
    presenter, defaults, _ = _make_presenter()
    presenter.update_scene_from_doc = MagicMock()

    defaults[
        "context"
    ].artifact_store.get.return_value = MaterialStateArtifact(
        material_state=_empty_state(),
        stock_uid="s1",
        generation_id=1,
    )
    stock_item = MagicMock()
    stock_item.uid = "s1"

    presenter._on_material_state_ready(
        None, item=stock_item, handle=MagicMock(), generation_id=1
    )

    assert "s1" not in presenter._material_states


@pytest.mark.ui
def test_on_material_state_ready_ignores_other_artifacts(
    ui_context_initializer,
):
    presenter, defaults, _ = _make_presenter()
    presenter.update_scene_from_doc = MagicMock()

    defaults["context"].artifact_store.get.return_value = MagicMock()

    stock_item = MagicMock()
    stock_item.uid = "s1"

    presenter._on_material_state_ready(
        None, item=stock_item, handle=MagicMock(), generation_id=1
    )

    assert presenter._material_states == {}
    presenter.update_scene_from_doc.assert_not_called()


# ── Stock-top content lift ───────────────────────────────────────


def _machine_with(z_axis: bool):
    machine = MagicMock()
    machine.has_z_axis = z_axis
    machine.heads = []
    machine.assembly = MagicMock()
    machine.assembly.has_rotary = False
    machine.get_default_rotary_module.return_value = None
    return machine


def _doc_with_stock(thickness):
    doc = Doc()
    asset = StockAsset(name="sheet")
    asset.set_thickness(thickness)
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(100, 0)
    geo.line_to(100, 80)
    geo.line_to(0, 80)
    geo.close_path()
    asset.geometry = geo
    doc.add_asset(asset)
    doc.add_child(StockItem(stock_asset_uid=asset.uid, name="sheet"))
    return doc


def _content_z(render_config_dict) -> float:
    blob = render_config_dict["world_to_visual"]
    w2v = np.frombuffer(blob, dtype=np.float32).reshape(4, 4)
    return float(w2v[2, 3])


@pytest.mark.ui
def test_content_not_lifted_for_z_machine(ui_context_initializer):
    """Has-Z machines render content at its authored Z (plus WCS Z):
    no stock-top lift."""
    doc_editor = MagicMock()
    doc_editor.doc = _doc_with_stock(thickness=4.0)
    presenter, _, _ = _make_presenter(
        context=MagicMock(machine=_machine_with(z_axis=True)),
        doc_editor=doc_editor,
        get_viewport=lambda: ViewportConfig.default(),
    )
    presenter._schedule_scene_preparation = MagicMock()

    presenter.update_scene_from_doc()

    config = presenter._schedule_scene_preparation.call_args.args[0]
    assert _content_z(config) == pytest.approx(0.0)


@pytest.mark.ui
def test_content_lifted_to_stock_top_for_no_z_machine(ui_context_initializer):
    doc_editor = MagicMock()
    doc_editor.doc = _doc_with_stock(thickness=4.0)
    presenter, _, _ = _make_presenter(
        context=MagicMock(machine=_machine_with(z_axis=False)),
        doc_editor=doc_editor,
        get_viewport=lambda: ViewportConfig.default(),
    )
    presenter._schedule_scene_preparation = MagicMock()

    presenter.update_scene_from_doc()

    config = presenter._schedule_scene_preparation.call_args.args[0]
    assert _content_z(config) == pytest.approx(4.0)


@pytest.mark.ui
def test_content_not_lifted_without_stock(ui_context_initializer):
    doc_editor = MagicMock()
    doc_editor.doc = Doc()
    presenter, _, _ = _make_presenter(
        context=MagicMock(machine=_machine_with(z_axis=True)),
        doc_editor=doc_editor,
        get_viewport=lambda: ViewportConfig.default(),
    )
    presenter._schedule_scene_preparation = MagicMock()

    presenter.update_scene_from_doc()

    config = presenter._schedule_scene_preparation.call_args.args[0]
    assert _content_z(config) == pytest.approx(0.0)


# ── Rotary burn specs ────────────────────────────────────────────


def _rotary_doc_editor(layer):
    doc_editor = MagicMock()
    doc = MagicMock()
    doc.stock_items = []
    doc.layers = [layer]
    doc_editor.doc = doc
    return doc_editor


@pytest.mark.ui
def test_build_rotary_stock_specs_attaches_burn(ui_context_initializer):
    """A rotary layer whose uid has a stored material state gets a
    burn entry in its spec."""
    layer = Layer(name="rot")
    layer.set_rotary_enabled(True)
    layer.set_rotary_diameter(50.0)
    material = Material(uid="m1", name="cherry")
    doc_editor = _rotary_doc_editor(layer)
    doc_editor.doc.get_asset_by_uid.return_value = None

    machine = MagicMock()
    rm = RotaryModule()
    rm.max_workpiece_length = 300.0
    machine.get_default_rotary_module.return_value = rm

    with patch.object(Layer, "stock_material", material):
        presenter, _, _ = _make_presenter(doc_editor=doc_editor)
        viewport = MagicMock()
        viewport.width_mm = 400.0
        presenter._material_states[layer.uid] = {
            "handle_key": "h1",
            "surface_map": CompressedArray.from_uint8_2d(
                np.zeros((4, 4), dtype=np.uint8)
            ),
            "origin_mm": (0.0, 0.0),
            "px_per_mm": (10.0, 10.0),
            "size_px": (4, 4),
        }
        specs = presenter._build_rotary_stock_specs(viewport, machine)

    assert len(specs) == 1
    assert specs[0]["kind"] == "rotary"
    assert "burn" in specs[0]
    assert specs[0]["burn"]["size_px"] == (4, 4)


@pytest.mark.ui
def test_refresh_material_states_includes_rotary_layers(
    ui_context_initializer,
):
    """Handles keyed by a rotary layer's uid are picked up on
    refresh, not just flat stock item handles."""
    layer = Layer(name="rot")
    layer.set_rotary_enabled(True)
    layer.set_rotary_diameter(50.0)

    state = _burned_state()
    handle = MagicMock()
    pipeline = MagicMock()
    pipeline._material_state_handles = {layer.uid: handle}

    doc_editor = _rotary_doc_editor(layer)
    doc_editor.pipeline = pipeline

    material = Material(uid="m1", name="cherry")
    with patch.object(Layer, "stock_material", material):
        presenter, defaults, _ = _make_presenter(doc_editor=doc_editor)
        defaults[
            "context"
        ].artifact_store.get.return_value = MaterialStateArtifact(
            material_state=state,
            stock_uid=layer.uid,
            generation_id=1,
        )

        changed = presenter._refresh_material_states()
    assert changed is True
    assert layer.uid in presenter._material_states
