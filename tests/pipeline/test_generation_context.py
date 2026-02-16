from typing import cast
from unittest.mock import MagicMock
import uuid
import pytest
from rayforge.context import get_context
from rayforge.core.ops import Ops
from rayforge.pipeline import CoordinateSystem
from rayforge.pipeline.artifact.key import ArtifactKey
from rayforge.pipeline.artifact.handle import BaseArtifactHandle
from rayforge.pipeline.artifact.workpiece import (
    WorkPieceArtifact,
    WorkPieceArtifactHandle,
)
from rayforge.pipeline.context import (
    ContextState,
    GenerationContext,
)


def test_initialization():
    """Test that a GenerationContext is initialized correctly."""
    ctx = GenerationContext(generation_id=1)

    assert ctx.generation_id == 1
    assert ctx.active_tasks == set()
    assert ctx.resources == set()
    assert not ctx.has_active_tasks()


def test_add_task():
    """Test that tasks can be added to the context."""
    ctx = GenerationContext(generation_id=1)
    key = ArtifactKey.for_workpiece(str(uuid.uuid4()))

    ctx.add_task(key)

    assert key in ctx.active_tasks
    assert ctx.has_active_tasks()


def test_task_did_finish():
    """Test that tasks can be marked as finished."""
    ctx = GenerationContext(generation_id=1)
    key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
    ctx.add_task(key)

    ctx.task_did_finish(key)

    assert key not in ctx.active_tasks
    assert not ctx.has_active_tasks()


def test_task_did_finish_idempotent():
    """Test that task_did_finish can be called multiple times."""
    ctx = GenerationContext(generation_id=1)
    key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
    ctx.add_task(key)

    ctx.task_did_finish(key)
    ctx.task_did_finish(key)

    assert not ctx.has_active_tasks()


def test_add_multiple_tasks():
    """Test that multiple tasks can be tracked."""
    ctx = GenerationContext(generation_id=1)
    key1 = ArtifactKey.for_workpiece(str(uuid.uuid4()))
    key2 = ArtifactKey.for_step(str(uuid.uuid4()))
    key3 = ArtifactKey.for_view(str(uuid.uuid4()))

    ctx.add_task(key1)
    ctx.add_task(key2)
    ctx.add_task(key3)

    assert len(ctx.active_tasks) == 3
    assert key1 in ctx.active_tasks
    assert key2 in ctx.active_tasks
    assert key3 in ctx.active_tasks


def test_add_resource():
    """Test that resources can be added to the context."""
    ctx = GenerationContext(generation_id=1)
    handle = MagicMock(spec=BaseArtifactHandle)

    ctx.add_resource(handle)

    assert handle in ctx.resources


def test_add_multiple_resources():
    """Test that multiple resources can be tracked."""
    ctx = GenerationContext(generation_id=1)
    handle1 = MagicMock(spec=BaseArtifactHandle)
    handle2 = MagicMock(spec=BaseArtifactHandle)

    ctx.add_resource(handle1)
    ctx.add_resource(handle2)

    assert len(ctx.resources) == 2
    assert handle1 in ctx.resources
    assert handle2 in ctx.resources


def test_shutdown_calls_release_callback():
    """Test that shutdown releases all tracked resources."""
    release_callback = MagicMock()
    ctx = GenerationContext(generation_id=1, release_callback=release_callback)
    handle1 = MagicMock(spec=BaseArtifactHandle)
    handle2 = MagicMock(spec=BaseArtifactHandle)
    ctx.add_resource(handle1)
    ctx.add_resource(handle2)

    ctx.shutdown()

    assert release_callback.call_count == 2
    release_callback.assert_any_call(handle1)
    release_callback.assert_any_call(handle2)


def test_shutdown_clears_resources():
    """Test that shutdown clears the resources set."""
    release_callback = MagicMock()
    ctx = GenerationContext(generation_id=1, release_callback=release_callback)
    handle = MagicMock(spec=BaseArtifactHandle)
    ctx.add_resource(handle)

    ctx.shutdown()

    assert ctx.resources == set()


def test_shutdown_idempotent():
    """Test that shutdown can be called multiple times safely."""
    release_callback = MagicMock()
    ctx = GenerationContext(generation_id=1, release_callback=release_callback)
    handle = MagicMock(spec=BaseArtifactHandle)
    ctx.add_resource(handle)

    ctx.shutdown()
    ctx.shutdown()

    assert release_callback.call_count == 1


def test_shutdown_without_callback():
    """Test that shutdown works without a release callback."""
    ctx = GenerationContext(generation_id=1)
    handle = MagicMock(spec=BaseArtifactHandle)
    ctx.add_resource(handle)

    ctx.shutdown()

    assert ctx.resources == set()


def test_active_tasks_returns_copy():
    """Test that active_tasks returns a copy, not the original set."""
    ctx = GenerationContext(generation_id=1)
    key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
    ctx.add_task(key)

    tasks = ctx.active_tasks
    tasks.clear()

    assert ctx.has_active_tasks()


def test_resources_returns_copy():
    """Test that resources returns a copy, not the original set."""
    ctx = GenerationContext(generation_id=1)
    handle = MagicMock(spec=BaseArtifactHandle)
    ctx.add_resource(handle)

    resources = ctx.resources
    resources.clear()

    assert len(ctx.resources) == 1


@pytest.fixture
def handles_to_release():
    handles = []
    yield handles
    for handle in handles:
        get_context().artifact_store.release(handle)


def _create_sample_artifact() -> WorkPieceArtifact:
    """Helper to create a sample artifact for testing."""
    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.line_to(10, 0, 0)
    return WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(100, 100),
        generation_size=(50, 50),
    )


def test_put_tracks_resource_in_context(handles_to_release):
    """
    Test that ArtifactStore.put registers the handle with the
    GenerationContext when provided.
    """
    ctx = GenerationContext(generation_id=1)
    store = get_context().artifact_store
    artifact = _create_sample_artifact()

    assert len(ctx.resources) == 0

    handle = store.put(artifact, generation_context=ctx)
    handles_to_release.append(handle)

    assert len(ctx.resources) == 1
    assert handle in ctx.resources


def test_put_tracks_multiple_resources_in_context(handles_to_release):
    """
    Test that multiple ArtifactStore.put calls track all handles
    in the GenerationContext.
    """
    ctx = GenerationContext(generation_id=1)
    store = get_context().artifact_store

    artifact1 = _create_sample_artifact()
    artifact2 = _create_sample_artifact()

    handle1 = store.put(artifact1, generation_context=ctx)
    handle2 = store.put(artifact2, generation_context=ctx)
    handles_to_release.extend([handle1, handle2])

    assert len(ctx.resources) == 2
    assert handle1 in ctx.resources
    assert handle2 in ctx.resources


def test_put_without_context_does_not_track(handles_to_release):
    """
    Test that ArtifactStore.put without a context does not
    affect any GenerationContext.
    """
    ctx = GenerationContext(generation_id=1)
    store = get_context().artifact_store
    artifact = _create_sample_artifact()

    handle = store.put(artifact)
    handles_to_release.append(handle)

    assert len(ctx.resources) == 0


def test_context_resources_are_real_handles(handles_to_release):
    """
    Test that the resources tracked in the context are real handles
    that can be used to retrieve artifacts.
    """
    ctx = GenerationContext(generation_id=1)
    store = get_context().artifact_store
    artifact = _create_sample_artifact()

    handle = store.put(artifact, generation_context=ctx)
    handles_to_release.append(handle)

    resources = ctx.resources
    assert len(resources) == 1

    tracked_handle = list(resources)[0]
    assert isinstance(tracked_handle, WorkPieceArtifactHandle)

    retrieved = store.get(tracked_handle)
    assert isinstance(retrieved, WorkPieceArtifact)
    retrieved_wp = cast(WorkPieceArtifact, retrieved)
    assert retrieved_wp.generation_size == (50, 50)


def test_task_did_finish_triggers_shutdown_when_superseded():
    """Test that task_did_finish triggers shutdown when superseded."""
    release_callback = MagicMock()
    ctx = GenerationContext(
        generation_id=1,
        release_callback=release_callback,
    )
    key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
    handle = MagicMock(spec=BaseArtifactHandle)
    ctx.add_task(key)
    ctx.add_resource(handle)
    ctx.mark_superseded()

    ctx.task_did_finish(key)

    assert ctx.is_shutdown
    release_callback.assert_called_once_with(handle)


def test_task_did_finish_no_shutdown_when_active():
    """Test that task_did_finish does not shutdown when still active."""
    release_callback = MagicMock()
    ctx = GenerationContext(
        generation_id=1,
        release_callback=release_callback,
    )
    key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
    handle = MagicMock(spec=BaseArtifactHandle)
    ctx.add_task(key)
    ctx.add_resource(handle)

    ctx.task_did_finish(key)

    assert not ctx.is_shutdown
    release_callback.assert_not_called()


def test_task_did_finish_no_shutdown_with_remaining_tasks():
    """Test no shutdown when superseded but tasks remain."""
    release_callback = MagicMock()
    ctx = GenerationContext(
        generation_id=1,
        release_callback=release_callback,
    )
    key1 = ArtifactKey.for_workpiece(str(uuid.uuid4()))
    key2 = ArtifactKey.for_step(str(uuid.uuid4()))
    handle = MagicMock(spec=BaseArtifactHandle)
    ctx.add_task(key1)
    ctx.add_task(key2)
    ctx.add_resource(handle)
    ctx.mark_superseded()

    ctx.task_did_finish(key1)

    assert not ctx.is_shutdown
    release_callback.assert_not_called()


def test_last_task_finishes_triggers_shutdown():
    """Test that the last task finishing triggers shutdown."""
    release_callback = MagicMock()
    ctx = GenerationContext(
        generation_id=1,
        release_callback=release_callback,
    )
    key1 = ArtifactKey.for_workpiece(str(uuid.uuid4()))
    key2 = ArtifactKey.for_step(str(uuid.uuid4()))
    handle = MagicMock(spec=BaseArtifactHandle)
    ctx.add_task(key1)
    ctx.add_task(key2)
    ctx.add_resource(handle)
    ctx.mark_superseded()

    ctx.task_did_finish(key1)
    assert not ctx.is_shutdown

    ctx.task_did_finish(key2)
    assert ctx.is_shutdown
    release_callback.assert_called_once_with(handle)


def test_no_auto_shutdown_without_callbacks():
    """Test no auto-shutdown when context is not superseded."""
    ctx = GenerationContext(generation_id=1)
    key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
    handle = MagicMock(spec=BaseArtifactHandle)
    ctx.add_task(key)
    ctx.add_resource(handle)

    ctx.task_did_finish(key)

    assert not ctx.is_shutdown


def test_state_property_returns_initial_state():
    """Test that state property returns initial ACTIVE state."""
    ctx = GenerationContext(generation_id=1)
    assert ctx.state == ContextState.ACTIVE


def test_mark_superseded_changes_state():
    """Test that mark_superseded changes state to SUPERSEDED."""
    ctx = GenerationContext(generation_id=1)
    assert ctx.state == ContextState.ACTIVE

    ctx.mark_superseded()

    assert ctx.state == ContextState.SUPERSEDED
