import types
from typing import cast

from rayforge.core.ops import Ops
from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.pipeline.artifact.step_ops import StepOpsArtifact
from rayforge.shared.tasker.proxy import ExecutionContextProxy
from rayforge.ui_gtk.canvas3d.compiled_scene import CompiledSceneArtifact
from rayforge.ui_gtk.canvas3d.render_config import StepRenderConfig
from rayforge.ui_gtk.canvas3d.scene_compiler_runner import (
    compile_scene_in_subprocess,
)

from compile_scene_helper import make_test_config


def _make_proxy(track_events=None, ack=True):
    events = track_events if track_events is not None else []

    def send_event(name, data, logger=None, timeout=5.0):
        events.append((name, data))
        return ack

    return cast(
        ExecutionContextProxy,
        types.SimpleNamespace(
            is_cancelled=lambda: False,
            send_event_and_wait=send_event,
        ),
    ), events


def _make_config_dict(**step_overrides):
    cfg = make_test_config()
    if step_overrides:
        sc = StepRenderConfig(
            rotary_enabled=step_overrides.get("rotary_enabled", False),
            rotary_diameter=step_overrides.get("rotary_diameter", 25.0),
            laser_uid=step_overrides.get("laser_uid", ""),
        )
        cfg.step_configs["step1"] = sc
    return cfg.to_dict()


def _store_ops(store, ops, step_uid="step1"):
    artifact = StepOpsArtifact(ops=ops, generation_id=1)
    handle = store.put(artifact, creator_tag="step_ops")
    return step_uid, handle.to_dict()


class TestCompileSceneInSubprocess:
    def test_produces_compiled_artifact(self):
        store = ArtifactStore()
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_power(1.0)
        ops.line_to(5.0, 5.0, 0.0)

        step_ops = [_store_ops(store, ops)]
        config_dict = _make_config_dict()

        proxy, events = _make_proxy()
        result = compile_scene_in_subprocess(
            proxy, store, step_ops, config_dict
        )

        assert result is None
        assert len(events) == 1
        event_name, event_data = events[0]
        assert event_name == "scene_compiled"

        handle_dict = event_data["handle_dict"]
        handle = store.adopt_from_dict(handle_dict)
        artifact = store.get(handle)
        assert isinstance(artifact, CompiledSceneArtifact)
        assert len(artifact.vertex_layers) == 2

        flat_vl = artifact.vertex_layers[0]
        pv = flat_vl.powered_verts.reshape(-1, 3)
        assert pv.shape[0] == 2

        store.release(handle)

    def test_empty_ops_returns_none(self):
        store = ArtifactStore()
        ops = Ops()

        step_ops = [_store_ops(store, ops)]
        config_dict = _make_config_dict()

        proxy, events = _make_proxy()
        result = compile_scene_in_subprocess(
            proxy, store, step_ops, config_dict
        )

        assert result is None
        assert len(events) == 0

    def test_no_step_handles_returns_none(self):
        store = ArtifactStore()
        config_dict = _make_config_dict()

        proxy, events = _make_proxy()
        result = compile_scene_in_subprocess(proxy, store, [], config_dict)

        assert result is None
        assert len(events) == 0

    def test_missing_config_skips_step(self):
        store = ArtifactStore()
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_power(1.0)
        ops.line_to(1.0, 1.0, 0.0)

        step_ops = [
            (
                "unknown_step",
                store.put(
                    StepOpsArtifact(ops=ops, generation_id=1),
                    creator_tag="step_ops",
                ).to_dict(),
            )
        ]
        config_dict = _make_config_dict()

        proxy, events = _make_proxy()
        result = compile_scene_in_subprocess(
            proxy, store, step_ops, config_dict
        )

        assert result is None
        assert len(events) == 0

    def test_stale_handle_skips_gracefully(self):
        store = ArtifactStore()
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_power(1.0)
        ops.line_to(1.0, 1.0, 0.0)

        handle = store.put(
            StepOpsArtifact(ops=ops, generation_id=1),
            creator_tag="step_ops",
        )
        handle_dict = handle.to_dict()
        store.release(handle)

        step_ops = [("step1", handle_dict)]
        config_dict = _make_config_dict()

        proxy, events = _make_proxy()
        result = compile_scene_in_subprocess(
            proxy, store, step_ops, config_dict
        )

        assert result is None

    def test_nack_releases_handle(self):
        store = ArtifactStore()
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_power(1.0)
        ops.line_to(5.0, 5.0, 0.0)

        step_ops = [_store_ops(store, ops)]
        config_dict = _make_config_dict()

        proxy, events = _make_proxy(ack=False)
        result = compile_scene_in_subprocess(
            proxy, store, step_ops, config_dict
        )

        assert result is None

    def test_rotary_step_produces_rotary_layer(self):
        store = ArtifactStore()
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_power(1.0)
        ops.line_to(5.0, 0.0, 0.0)

        step_ops = [_store_ops(store, ops)]
        config_dict = _make_config_dict(
            rotary_enabled=True, rotary_diameter=50.0
        )

        proxy, events = _make_proxy()
        result = compile_scene_in_subprocess(
            proxy, store, step_ops, config_dict
        )

        assert result is None
        assert len(events) == 1

        handle_dict = events[0][1]["handle_dict"]
        handle = store.adopt_from_dict(handle_dict)
        artifact = store.get(handle)
        assert isinstance(artifact, CompiledSceneArtifact)

        rot_vl = artifact.vertex_layers[1]
        pv = rot_vl.powered_verts.reshape(-1, 3)
        assert pv.shape[0] == 2
        assert abs(pv[0, 2] - 25.0) < 1e-3

        store.release(handle)
