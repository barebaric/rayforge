import types
from typing import cast

from rayforge.core.ops import Ops
from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.pipeline.artifact.job import JobArtifact
from rayforge.shared.tasker.proxy import ExecutionContextProxy
from rayforge.simulator.scene3d.compiled_scene import CompiledSceneArtifact
from rayforge.simulator.scene3d.scene_compiler_runner import (
    compile_scene_in_subprocess,
)

from compile_scene_helper import (
    make_test_config,
    make_assembled_ops,
    make_flat_layer_config,
    make_rotary_layer_config,
)


def _make_proxy(track_events=None, ack=True, store=None):
    events = track_events if track_events is not None else []
    adopted_handles = []

    def send_event(name, data, logger=None, timeout=5.0):
        events.append((name, data))
        if ack and store is not None and "handle_dict" in data:
            adopted_handles.append(store.adopt_from_dict(data["handle_dict"]))
        return ack

    return (
        cast(
            ExecutionContextProxy,
            types.SimpleNamespace(
                is_cancelled=lambda: False,
                send_event_and_wait=send_event,
            ),
        ),
        events,
        adopted_handles,
    )


def _make_flat_config_dict():
    cfg = make_test_config(layer_configs={"layer1": make_flat_layer_config()})
    return cfg.to_dict()


def _make_rotary_config_dict(diameter=50.0):
    cfg = make_test_config(
        layer_configs={"layer1": make_rotary_layer_config(diameter=diameter)}
    )
    return cfg.to_dict()


def _store_job(store, step_ops_list):
    assembled = make_assembled_ops(step_ops_list)
    artifact = JobArtifact(ops=assembled, distance=0.0, generation_id=1)
    handle = store.put(artifact, creator_tag="job")
    return handle.to_dict()


class TestCompileSceneInSubprocess:
    def test_produces_compiled_artifact(self):
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_power(1.0)
        ops.line_to(5.0, 5.0, 0.0)

        store = ArtifactStore()
        handle_dict = _store_job(store, [("layer1", ops)])
        config_dict = _make_flat_config_dict()

        proxy, events, adopted = _make_proxy(store=store)
        result = compile_scene_in_subprocess(
            proxy, store, handle_dict, config_dict
        )

        assert result is None
        assert len(events) == 1
        event_name, event_data = events[0]
        assert event_name == "scene_compiled"

        handle = adopted[0]
        artifact = store.get(handle)
        assert isinstance(artifact, CompiledSceneArtifact)
        assert len(artifact.vertex_layers) == 1

        flat_vl = artifact.vertex_layers[0]
        assert not flat_vl.is_rotary
        pv = flat_vl.powered_verts.reshape(-1, 3)
        assert pv.shape[0] == 2

        store.release(handle)

    def test_marker_only_ops_produces_compiled_artifact(self):
        store = ArtifactStore()
        handle_dict = _store_job(store, [("layer1", Ops())])
        config_dict = _make_flat_config_dict()

        proxy, events, adopted = _make_proxy(store=store)
        result = compile_scene_in_subprocess(
            proxy, store, handle_dict, config_dict
        )

        assert result is None
        assert len(events) == 1

        handle = adopted[0]
        artifact = store.get(handle)
        assert isinstance(artifact, CompiledSceneArtifact)
        assert len(artifact.vertex_layers) == 0

        store.release(handle)

    def test_stale_handle_skips_gracefully(self):
        store = ArtifactStore()
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_power(1.0)
        ops.line_to(1.0, 1.0, 0.0)

        handle = store.put(
            JobArtifact(
                ops=make_assembled_ops([("layer1", ops)]),
                distance=0.0,
                generation_id=1,
            ),
            creator_tag="job",
        )
        handle_dict = handle.to_dict()
        store.release(handle)

        config_dict = _make_flat_config_dict()

        proxy, events, _ = _make_proxy()
        result = compile_scene_in_subprocess(
            proxy, store, handle_dict, config_dict
        )

        assert result is None

    def test_nack_releases_handle(self):
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_power(1.0)
        ops.line_to(5.0, 5.0, 0.0)

        store = ArtifactStore()
        handle_dict = _store_job(store, [("layer1", ops)])
        config_dict = _make_flat_config_dict()

        proxy, events, _ = _make_proxy(ack=False)
        result = compile_scene_in_subprocess(
            proxy, store, handle_dict, config_dict
        )

        assert result is None

    def test_rotary_step_produces_rotary_layer(self):
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_power(1.0)
        ops.line_to(5.0, 0.0, 0.0)

        store = ArtifactStore()
        handle_dict = _store_job(store, [("layer1", ops)])
        config_dict = _make_rotary_config_dict(diameter=50.0)

        proxy, events, adopted = _make_proxy(store=store)
        result = compile_scene_in_subprocess(
            proxy, store, handle_dict, config_dict
        )

        assert result is None
        assert len(events) == 1

        handle = adopted[0]
        artifact = store.get(handle)
        assert isinstance(artifact, CompiledSceneArtifact)
        assert len(artifact.vertex_layers) == 1

        rot_vl = artifact.vertex_layers[0]
        assert rot_vl.is_rotary
        pv = rot_vl.powered_verts.reshape(-1, 3)
        assert pv.shape[0] == 2
        assert abs(pv[0, 2] - 25.0) < 1e-3

        store.release(handle)
