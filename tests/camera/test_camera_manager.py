from types import SimpleNamespace
from typing import cast

from rayforge.camera.controller import CameraController
from rayforge.camera.manager import CameraManager
from rayforge.camera.models.camera import Camera
from rayforge.context import RayforgeContext


def _make_manager() -> CameraManager:
    # CameraManager only needs `context.config` for the paths under test
    # here (initialize()/_reconcile_controllers are not exercised), so a
    # minimal stand-in context is sufficient and keeps the test focused.
    context = cast(RayforgeContext, SimpleNamespace(config=None))
    return CameraManager(context)


def test_destroy_controller_disposes_and_removes_it():
    """Destroying a controller must both unsubscribe it and dispose it,
    so it is fully detached from its camera model and can never be
    resurrected by a later, unrelated config change.
    """
    camera = Camera("Test Camera", "0")
    controller = CameraController(camera)
    manager = _make_manager()
    manager._controllers[camera.id] = controller

    removed_signals = []

    def _on_removed(sender, controller):
        removed_signals.append(controller)

    # blinker keeps only a weak reference by default; a bare lambda would
    # be garbage-collected before the signal fires.
    manager.controller_removed.connect(_on_removed, weak=False)

    manager._destroy_controller(camera.id)

    assert camera.id not in manager._controllers
    assert controller._disposed is True
    assert removed_signals == [controller]

    # A config mutation reaching the disposed controller after teardown
    # must not resurrect its capture stream.
    start_calls = []
    controller._start_capture_stream = lambda: start_calls.append("start")
    camera.enabled = True
    assert start_calls == []


def test_destroy_controller_is_idempotent_for_unknown_id():
    manager = _make_manager()
    # Must not raise when asked to destroy a controller that isn't tracked.
    manager._destroy_controller("does-not-exist")
