import numpy as np
from raygeo.ops import Ops

from rayforge.pipeline.artifact.job import JobArtifact
from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.pipeline.artifact.workpiece import WorkPieceArtifact
from rayforge.simulator.scene3d.render_config import (
    LayerRenderConfig,
    RenderConfig3D,
)
from rayforge.simulator.scene3d.scene_compiler_runner import (
    compile_scene_from_job,
)


def _make_ops():
    ops = Ops()
    ops.job_start()
    ops.layer_start("layer1")
    ops.move_to(0.0, 0.0, 0.0)
    ops.set_power(0.5)
    ops.line_to(5.0, 5.0, 0.0)
    ops.layer_end("layer1")
    ops.job_end()
    return ops


def _make_config_dict():
    return RenderConfig3D(
        world_to_visual=np.eye(4, dtype=np.float32),
        world_to_cyl_local=np.eye(4, dtype=np.float32),
        layer_configs={
            "layer1": LayerRenderConfig(
                rotary_enabled=False, rotary_diameter=0.0
            )
        },
    ).to_dict()


def _store_job(store, *, mapped_ops=False, ops=None, distance=0.0):
    artifact = JobArtifact(
        ops=_make_ops() if ops is None else ops,
        distance=distance,
        generation_id=1,
        mapped_ops=_make_ops() if mapped_ops else None,
    )
    handle = store.put(artifact, creator_tag="test")
    return handle.to_dict()


def test_compile_from_job_happy_path():
    store = ArtifactStore()
    handle_dict = _store_job(store)

    result = compile_scene_from_job(store, handle_dict, _make_config_dict())

    assert result is not None
    assert len(result.vertex_layers) == 1
    assert result.generation_id == 1


def test_compile_from_job_propagates_generation_id():
    store = ArtifactStore()
    artifact = JobArtifact(
        ops=_make_ops(),
        distance=0.0,
        generation_id=42,
    )
    handle = store.put(artifact, creator_tag="test")

    result = compile_scene_from_job(
        store, handle.to_dict(), _make_config_dict()
    )

    assert result is not None
    assert result.generation_id == 42


def test_compile_from_job_uses_mapped_ops():
    store = ArtifactStore()
    handle_dict = _store_job(store, mapped_ops=True)

    result = compile_scene_from_job(store, handle_dict, _make_config_dict())

    assert result is not None


def test_compile_from_job_empty_ops_returns_none():
    store = ArtifactStore()
    handle_dict = _store_job(store, ops=Ops())

    result = compile_scene_from_job(store, handle_dict, _make_config_dict())

    assert result is None


def test_compile_from_job_missing_handle_returns_none():
    store = ArtifactStore()
    handle_dict = _store_job(store)
    store.shutdown()

    result = compile_scene_from_job(store, handle_dict, _make_config_dict())

    assert result is None


def test_compile_from_job_wrong_artifact_type_returns_none():
    store = ArtifactStore()
    artifact = WorkPieceArtifact(
        ops=_make_ops(),
        is_scalable=False,
        generation_size=(100.0, 100.0),
        generation_id=1,
    )
    handle = store.put(artifact, creator_tag="test")

    result = compile_scene_from_job(
        store, handle.to_dict(), _make_config_dict()
    )

    assert result is None
