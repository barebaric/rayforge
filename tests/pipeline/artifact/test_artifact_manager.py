"""
Tests for ArtifactManager as a pure cache.

The ArtifactManager stores handles and tracks state.
State tracking is the single source of truth in the ledger.
"""

import uuid
from unittest.mock import Mock
import pytest
from rayforge.pipeline.artifact import (
    ArtifactManager,
    ArtifactKey,
    WorkPieceArtifactHandle,
    StepRenderArtifactHandle,
    StepOpsArtifactHandle,
    JobArtifactHandle,
)
from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.pipeline.artifact.manager import make_composite_key
from rayforge.pipeline.dag.node import NodeState


STEP1_UID = str(uuid.uuid4())
STEP2_UID = str(uuid.uuid4())
WP1_UID = str(uuid.uuid4())
WP2_UID = str(uuid.uuid4())
WP3_UID = str(uuid.uuid4())
JOB_UID = str(uuid.uuid4())


def create_mock_handle(handle_class, name: str) -> Mock:
    """Creates a mock handle that behaves like a real handle for tests."""
    handle = Mock(spec=handle_class)
    handle.shm_name = f"shm_{name}"
    return handle


@pytest.fixture
def manager():
    mock_store = Mock(spec=ArtifactStore)
    mock_store._refcounts = {}
    return ArtifactManager(mock_store)


@pytest.fixture
def mock_store(manager):
    return manager._store


def test_put_and_get_step_handles(manager):
    """Tests storage and retrieval of step handle types."""
    render_handle = create_mock_handle(
        StepRenderArtifactHandle, "step1_render"
    )
    ops_handle = create_mock_handle(StepOpsArtifactHandle, "step1_ops")

    manager.put_step_render_handle(STEP1_UID, render_handle)
    manager.put_step_ops_handle(ArtifactKey.for_step(STEP1_UID), ops_handle, 0)

    retrieved_render = manager.get_step_render_handle(STEP1_UID)
    retrieved_ops = manager.get_step_ops_handle(
        ArtifactKey.for_step(STEP1_UID), 0
    )

    assert retrieved_render is render_handle
    assert retrieved_ops is ops_handle
    manager._store.release.assert_not_called()


def test_put_and_get_job(manager):
    """Tests basic storage and retrieval of a job handle."""
    handle = create_mock_handle(JobArtifactHandle, "job")
    job_key = ArtifactKey(id=JOB_UID, group="job")
    manager.cache_handle(job_key, handle, 0)

    retrieved = manager.get_job_handle(job_key, 0)
    assert retrieved is handle
    manager._store.release.assert_not_called()


def test_invalidate_workpiece_cascades_correctly(manager):
    """Tests that invalidating a workpiece removes only that workpiece."""
    wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
    ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")

    manager.cache_handle(ArtifactKey.for_workpiece(WP1_UID), wp_h, 0)
    manager.put_step_render_handle(WP1_UID, render_h)
    manager.put_step_ops_handle(ArtifactKey.for_step(WP1_UID), ops_h, 0)

    manager.invalidate_for_workpiece(ArtifactKey.for_workpiece(WP1_UID))

    assert (
        manager.get_workpiece_handle(ArtifactKey.for_workpiece(WP1_UID), 0)
        is None
    )
    assert (
        manager.get_step_ops_handle(ArtifactKey.for_step(WP1_UID), 0) is ops_h
    )
    assert manager.get_step_render_handle(WP1_UID) is render_h
    manager._store.release.assert_any_call(wp_h)


def test_invalidate_step_clears_step_artifacts(manager):
    """Tests that invalidating a step clears step artifacts only."""
    wp1_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    wp2_h = create_mock_handle(WorkPieceArtifactHandle, "wp2")
    render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
    ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")

    wp_key = ArtifactKey.for_workpiece(WP1_UID, STEP1_UID)
    manager.cache_handle(wp_key, wp1_h, 0)
    manager.cache_handle(wp_key, wp2_h, 1)
    manager.put_step_render_handle(STEP1_UID, render_h)
    manager.put_step_ops_handle(ArtifactKey.for_step(STEP1_UID), ops_h, 0)

    manager.invalidate_for_step(ArtifactKey.for_step(STEP1_UID))

    assert manager.get_workpiece_handle(wp_key, 0) is wp1_h
    assert (
        manager.get_step_ops_handle(ArtifactKey.for_step(STEP1_UID), 0) is None
    )
    assert manager.get_step_render_handle(STEP1_UID) is None
    manager._store.release.assert_any_call(render_h)
    manager._store.release.assert_any_call(ops_h)


def test_put_job_replaces_old_handle(manager):
    """Tests that putting a job handle releases old handle."""
    old_job_h = create_mock_handle(JobArtifactHandle, "job_old")
    new_job_h = create_mock_handle(JobArtifactHandle, "job_new")
    job_key = ArtifactKey(id=JOB_UID, group="job")

    manager.cache_handle(job_key, old_job_h, 0)
    manager.cache_handle(job_key, new_job_h, 0)

    assert manager.get_job_handle(job_key, 0) is new_job_h
    manager._store.release.assert_called_with(old_job_h)


def test_shutdown_releases_all_artifacts(manager):
    """Tests that shutdown releases all cached artifacts."""
    wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
    ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
    job_h = create_mock_handle(JobArtifactHandle, "job")

    manager.cache_handle(ArtifactKey.for_workpiece(WP1_UID), wp_h, 0)
    manager.put_step_render_handle(STEP1_UID, render_h)
    manager.put_step_ops_handle(ArtifactKey.for_step(STEP1_UID), ops_h, 0)
    manager.cache_handle(ArtifactKey(id=JOB_UID, group="job"), job_h, 0)

    manager.shutdown()

    manager._store.release.assert_any_call(wp_h)
    manager._store.release.assert_any_call(render_h)
    manager._store.release.assert_any_call(ops_h)
    manager._store.release.assert_any_call(job_h)
    assert len(manager._ledger) == 0
    assert len(manager._step_render_handles) == 0


def test_get_all_workpiece_keys(manager):
    """Tests getting all workpiece keys."""
    manager.cache_handle(
        ArtifactKey.for_workpiece(WP1_UID),
        create_mock_handle(WorkPieceArtifactHandle, "wp1"),
        0,
    )
    manager.cache_handle(
        ArtifactKey.for_workpiece(WP2_UID),
        create_mock_handle(WorkPieceArtifactHandle, "wp2"),
        0,
    )
    manager.cache_handle(
        ArtifactKey.for_workpiece(WP3_UID),
        create_mock_handle(WorkPieceArtifactHandle, "wp3"),
        0,
    )

    keys = manager.get_all_workpiece_keys()
    assert len(keys) == 3
    assert ArtifactKey.for_workpiece(WP1_UID) in keys
    assert ArtifactKey.for_workpiece(WP2_UID) in keys
    assert ArtifactKey.for_workpiece(WP3_UID) in keys


def test_get_all_step_render_uids(manager):
    """Tests getting all step render UIDs."""
    manager._step_render_handles[STEP1_UID] = Mock()
    manager._step_render_handles[STEP2_UID] = Mock()

    uids = manager.get_all_step_render_uids()
    assert len(uids) == 2
    assert STEP1_UID in uids
    assert STEP2_UID in uids


def test_has_step_render_handle(manager):
    """Tests checking if a step render handle exists."""
    manager._step_render_handles[STEP1_UID] = Mock()
    assert manager.has_step_render_handle(STEP1_UID)
    assert not manager.has_step_render_handle(STEP2_UID)


def test_pop_step_render_handle(manager):
    """Tests popping a step render handle."""
    render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
    manager._step_render_handles[STEP1_UID] = render_h

    popped = manager.pop_step_render_handle(STEP1_UID)
    assert popped is render_h
    assert manager.get_step_render_handle(STEP1_UID) is None


def test_checkout_step_render_handle(manager):
    """Tests checking out a step ops handle."""
    ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
    manager.put_step_ops_handle(ArtifactKey.for_step(STEP1_UID), ops_h, 0)

    with manager.checkout(ArtifactKey.for_step(STEP1_UID), 0) as handle:
        assert handle is ops_h
        assert manager._ref_counts[ArtifactKey.for_step(STEP1_UID)] == 1
        manager._store.retain.assert_called_once_with(ops_h)

    manager._store.release.assert_called_once_with(ops_h)
    assert ArtifactKey.for_step(STEP1_UID) not in manager._ref_counts


def test_checkout_job_handle(manager):
    """Tests checking out job handle."""
    job_h = create_mock_handle(JobArtifactHandle, "job")
    job_key = ArtifactKey(id=JOB_UID, group="job")
    manager.cache_handle(job_key, job_h, 0)

    with manager.checkout(job_key, 0) as handle:
        assert handle is job_h
        assert manager._ref_counts[job_key] == 1
        manager._store.retain.assert_called_with(job_h)

    manager._store.release.assert_called_with(job_h)
    assert job_key not in manager._ref_counts


def test_get_workpiece_handle_from_ledger(manager):
    """Tests getting workpiece handle from ledger."""
    wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    manager.cache_handle(ArtifactKey.for_workpiece(WP1_UID), wp_h, 0)

    retrieved = manager.get_workpiece_handle(
        ArtifactKey.for_workpiece(WP1_UID), 0
    )
    assert retrieved is wp_h


def test_get_workpiece_handle_returns_none_when_not_found(manager):
    """Tests getting workpiece handle returns None when not found."""
    retrieved = manager.get_workpiece_handle(
        ArtifactKey.for_workpiece(WP1_UID), 0
    )
    assert retrieved is None


def test_prune_removes_obsolete_data_generation(manager):
    """Tests pruning removes ledger entries from non-active data gens."""
    wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    manager.cache_handle(ArtifactKey.for_workpiece(WP1_UID), wp_h, 0)

    manager.prune(active_data_gen_ids={1})

    wp_composite = make_composite_key(ArtifactKey.for_workpiece(WP1_UID), 0)
    assert wp_composite not in manager._ledger


def test_prune_keeps_active_data_generation(manager):
    """Tests pruning keeps artifacts from active data generations."""
    wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    manager.cache_handle(ArtifactKey.for_workpiece(WP1_UID), wp_h, 0)

    manager.prune(active_data_gen_ids={0})

    wp_composite = make_composite_key(ArtifactKey.for_workpiece(WP1_UID), 0)
    assert wp_composite in manager._ledger
    manager._store.release.assert_not_called()


def test_prune_preserves_step_for_processing_data_gen(manager):
    """Tests pruning preserves step entries for processing data gens."""
    step_h0 = create_mock_handle(StepOpsArtifactHandle, "step0")
    step_h1 = create_mock_handle(StepOpsArtifactHandle, "step1")
    step_key0 = ArtifactKey.for_step(STEP1_UID)
    step_key1 = ArtifactKey.for_step(STEP2_UID)
    manager.cache_handle(step_key0, step_h0, 0)
    manager.cache_handle(step_key1, step_h1, 1)

    manager.prune(
        active_data_gen_ids={1},
        processing_data_gen_ids={1, 0},
    )

    step_composite_0 = make_composite_key(step_key0, 0)
    step_composite_1 = make_composite_key(step_key1, 1)
    assert step_composite_0 in manager._ledger
    assert step_composite_1 in manager._ledger
    manager._store.release.assert_not_called()


def test_is_generation_current_returns_true_for_matching(manager):
    """Test is_generation_current returns True for matching gen ID."""
    wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    wp_key = ArtifactKey.for_workpiece(WP1_UID)
    manager.cache_handle(wp_key, wp_h, 1)

    result = manager.is_generation_current(wp_key, 1)
    assert result


def test_is_generation_current_returns_false_for_mismatch(manager):
    """Test is_generation_current returns False for gen ID mismatch."""
    wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    wp_key = ArtifactKey.for_workpiece(WP1_UID)
    manager.cache_handle(wp_key, wp_h, 1)

    result = manager.is_generation_current(wp_key, 2)
    assert not result


def test_is_generation_current_returns_false_for_missing(manager):
    """Test is_generation_current returns False for missing entry."""
    wp_key = ArtifactKey.for_workpiece(WP1_UID)
    result = manager.is_generation_current(wp_key, 0)
    assert not result


def test_get_artifact_retrieves_from_store(manager):
    """Test get_artifact retrieves artifact from store."""
    handle = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    artifact = Mock()
    manager._store.get.return_value = artifact

    result = manager.get_artifact(handle)

    assert result is artifact
    manager._store.get.assert_called_once_with(handle)


def test_put_step_render_handle_replaces_old(manager):
    """
    Test put_step_render_handle replaces old handle and releases it.
    """
    old_h = create_mock_handle(StepRenderArtifactHandle, "old")
    new_h = create_mock_handle(StepRenderArtifactHandle, "new")
    manager._step_render_handles[STEP1_UID] = old_h

    manager.put_step_render_handle(STEP1_UID, new_h)

    assert manager._step_render_handles[STEP1_UID] is new_h
    manager._store.release.assert_called_once_with(old_h)


def test_put_step_render_handle_without_old(manager):
    """Test put_step_render_handle stores without old handle."""
    new_h = create_mock_handle(StepRenderArtifactHandle, "new")

    manager.put_step_render_handle(STEP1_UID, new_h)

    assert manager._step_render_handles[STEP1_UID] is new_h
    manager._store.release.assert_not_called()


def test_put_step_ops_handle_creates_entry(manager):
    """Test put_step_ops_handle creates entry."""
    handle = create_mock_handle(StepOpsArtifactHandle, "ops")
    step_key = ArtifactKey.for_step(STEP1_UID)

    manager.put_step_ops_handle(step_key, handle, 1)

    composite_key = make_composite_key(step_key, 1)
    entry = manager._ledger.get(composite_key)
    assert entry is not None
    assert entry.handle is handle
    assert entry.generation_id == 1


def test_retain_handle_calls_store_retain(manager):
    """Test retain_handle delegates to store.retain."""
    handle = create_mock_handle(WorkPieceArtifactHandle, "wp1")

    manager.retain_handle(handle)

    manager._store.retain.assert_called_once_with(handle)


def test_get_all_workpiece_keys_for_generation(manager):
    """Test getting workpiece keys for specific generation."""
    manager.cache_handle(
        ArtifactKey.for_workpiece(WP1_UID),
        create_mock_handle(WorkPieceArtifactHandle, "wp1"),
        0,
    )
    manager.cache_handle(
        ArtifactKey.for_workpiece(WP2_UID),
        create_mock_handle(WorkPieceArtifactHandle, "wp2"),
        1,
    )
    manager.cache_handle(
        ArtifactKey.for_workpiece(WP3_UID),
        create_mock_handle(WorkPieceArtifactHandle, "wp3"),
        0,
    )

    keys = manager.get_all_workpiece_keys_for_generation(0)

    assert len(keys) == 2
    assert ArtifactKey.for_workpiece(WP1_UID) in keys
    assert ArtifactKey.for_workpiece(WP3_UID) in keys
    assert ArtifactKey.for_workpiece(WP2_UID) not in keys


def test_invalidate_for_job_releases_and_sets_dirty(manager):
    """Test invalidate_for_job releases handle and sets state DIRTY."""
    job_h = create_mock_handle(JobArtifactHandle, "job")
    job_key = ArtifactKey(id=JOB_UID, group="job")
    manager.cache_handle(job_key, job_h, 0)

    manager.invalidate_for_job(job_key)

    job_composite = make_composite_key(job_key, 0)
    entry = manager._ledger.get(job_composite)
    assert entry is not None
    assert entry.handle is None
    assert entry.state == NodeState.DIRTY
    manager._store.release.assert_called_once_with(job_h)


def test_checkout_handle_with_none(manager):
    """Test checkout_handle yields None for None handle."""
    with manager.checkout_handle(None) as artifact:
        assert artifact is None


def test_checkout_handle_retains_and_releases(manager):
    """Test checkout_handle retains on entry and releases on exit."""
    handle = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    artifact = Mock()
    manager._store.get.return_value = artifact

    with manager.checkout_handle(handle):
        manager._store.retain.assert_called_once_with(handle)

    manager._store.release.assert_called_once_with(handle)


def test_checkout_handle_releases_on_exception(manager):
    """Test checkout_handle releases even if exception occurs."""
    handle = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    artifact = Mock()
    manager._store.get.return_value = artifact

    with pytest.raises(ValueError):
        with manager.checkout_handle(handle):
            manager._store.retain.assert_called_once_with(handle)
            raise ValueError("test")

    manager._store.release.assert_called_once_with(handle)


def test_mark_done_without_handle(manager):
    """Test mark_done works when entry has no handle."""
    key = ArtifactKey.for_workpiece(WP1_UID)
    manager.declare_generation({key}, 0)

    manager.mark_done(key, 0)

    composite_key = make_composite_key(key, 0)
    entry = manager._ledger.get(composite_key)
    assert entry is not None
    assert entry.handle is None
    manager._store.release.assert_not_called()


def test_mark_done_nonexistent_skips(manager):
    """Test mark_done skips when entry does not exist."""
    key = ArtifactKey.for_workpiece(WP1_UID)

    manager.mark_done(key, 0)

    assert len(manager._ledger) == 0


def test_complete_generation_marks_existing_entry_done(manager):
    """Test complete_generation marks existing entry as done."""
    wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    key = ArtifactKey.for_workpiece(WP1_UID)
    manager.cache_handle(key, wp_h, 0)

    manager.complete_generation(key, 0)

    composite_key = make_composite_key(key, 0)
    entry = manager._ledger.get(composite_key)
    assert entry is not None
    assert entry.handle is wp_h


def test_complete_generation_creates_new_entry_with_handle(manager):
    """Test complete_generation creates new entry with handle."""
    wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    key = ArtifactKey.for_workpiece(WP1_UID)

    manager.complete_generation(key, 0, wp_h)

    composite_key = make_composite_key(key, 0)
    entry = manager._ledger.get(composite_key)
    assert entry is not None
    assert entry.handle is wp_h


def test_complete_generation_replaces_handle(manager):
    """Test complete_generation replaces existing handle."""
    old_h = create_mock_handle(WorkPieceArtifactHandle, "old")
    new_h = create_mock_handle(WorkPieceArtifactHandle, "new")
    key = ArtifactKey.for_workpiece(WP1_UID)
    manager.cache_handle(key, old_h, 0)

    manager.complete_generation(key, 0, new_h)

    composite_key = make_composite_key(key, 0)
    entry = manager._ledger.get(composite_key)
    assert entry is not None
    assert entry.handle is new_h
    manager._store.release.assert_called_once_with(old_h)


def test_complete_generation_nonexistent_without_handle_creates_entry(manager):
    """Test complete_generation creates done entry when no handle."""
    key = ArtifactKey.for_workpiece(WP1_UID)

    manager.complete_generation(key, 0)

    composite_key = make_composite_key(key, 0)
    entry = manager._ledger.get(composite_key)
    assert entry is not None
    assert entry.handle is None


def test_declare_generation_creates_placeholder_entries(manager):
    """Test declare_generation creates placeholder entries for new keys."""
    wp1_key = ArtifactKey.for_workpiece(WP1_UID)
    wp2_key = ArtifactKey.for_workpiece(WP2_UID)

    manager.declare_generation({wp1_key, wp2_key}, 0)

    wp1_composite = make_composite_key(wp1_key, 0)
    wp2_composite = make_composite_key(wp2_key, 0)
    entry1 = manager._ledger.get(wp1_composite)
    entry2 = manager._ledger.get(wp2_composite)
    assert entry1 is not None
    assert entry2 is not None
    assert entry1.handle is None
    assert entry2.handle is None


def test_declare_generation_skips_existing_entries(manager):
    """Test declare_generation skips existing entries."""
    wp_key = ArtifactKey.for_workpiece(WP1_UID)
    wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    manager.cache_handle(wp_key, wp_h, 0)

    manager.declare_generation({wp_key}, 0)

    wp_composite = make_composite_key(wp_key, 0)
    entry = manager._ledger.get(wp_composite)
    assert entry is not None
    assert entry.handle is wp_h


def test_declare_generation_copies_workpiece_handle_from_previous(manager):
    """Test declare_generation copies workpiece handles from previous gen."""
    wp_key = ArtifactKey.for_workpiece(WP1_UID)
    wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    manager.cache_handle(wp_key, wp_h, 0)

    manager.declare_generation({wp_key}, 1)

    wp_composite_g0 = make_composite_key(wp_key, 0)
    wp_composite_g1 = make_composite_key(wp_key, 1)
    entry_g0 = manager._ledger.get(wp_composite_g0)
    entry_g1 = manager._ledger.get(wp_composite_g1)
    assert entry_g0 is not None
    assert entry_g1 is not None
    assert entry_g0.handle is wp_h
    assert entry_g1.handle is wp_h
    manager._store.retain.assert_called()


def test_declare_generation_copies_step_handle_from_previous(manager):
    """Test declare_generation copies step handles from previous gen."""
    step_key = ArtifactKey.for_step(STEP1_UID)
    step_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
    manager.cache_handle(step_key, step_h, 0)

    manager.declare_generation({step_key}, 1)

    step_composite_g0 = make_composite_key(step_key, 0)
    step_composite_g1 = make_composite_key(step_key, 1)
    entry_g0 = manager._ledger.get(step_composite_g0)
    entry_g1 = manager._ledger.get(step_composite_g1)
    assert entry_g0 is not None
    assert entry_g1 is not None
    assert entry_g0.handle is step_h
    assert entry_g1.handle is step_h
    manager._store.retain.assert_called()


def test_get_state_returns_dirty_for_missing_entry(manager):
    """Test get_state returns DIRTY for missing entry."""
    key = ArtifactKey.for_workpiece(WP1_UID)
    result = manager.get_state(key, 0)
    assert result == NodeState.DIRTY


def test_get_state_returns_entry_state(manager):
    """Test get_state returns the state from the entry."""
    wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    key = ArtifactKey.for_workpiece(WP1_UID)
    manager.cache_handle(key, wp_h, 0)
    manager.set_state(key, 0, NodeState.VALID)

    result = manager.get_state(key, 0)
    assert result == NodeState.VALID


def test_set_state_updates_entry(manager):
    """Test set_state updates the state of an entry."""
    wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    key = ArtifactKey.for_workpiece(WP1_UID)
    manager.cache_handle(key, wp_h, 0)

    manager.set_state(key, 0, NodeState.PROCESSING)

    composite_key = make_composite_key(key, 0)
    entry = manager._ledger.get(composite_key)
    assert entry is not None
    assert entry.state == NodeState.PROCESSING


def test_set_state_ignores_missing_entry(manager):
    """Test set_state does nothing for missing entry."""
    key = ArtifactKey.for_workpiece(WP1_UID)

    manager.set_state(key, 0, NodeState.VALID)

    composite_key = make_composite_key(key, 0)
    assert composite_key not in manager._ledger


def test_has_artifact_returns_true_for_existing_handle(manager):
    """Test has_artifact returns True when handle exists."""
    wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    key = ArtifactKey.for_workpiece(WP1_UID)
    manager.cache_handle(key, wp_h, 0)

    result = manager.has_artifact(key, 0)
    assert result


def test_has_artifact_returns_false_for_missing_entry(manager):
    """Test has_artifact returns False when entry missing."""
    key = ArtifactKey.for_workpiece(WP1_UID)
    result = manager.has_artifact(key, 0)
    assert not result


def test_has_artifact_returns_false_for_none_handle(manager):
    """Test has_artifact returns False when handle is None."""
    key = ArtifactKey.for_workpiece(WP1_UID)
    manager.declare_generation({key}, 0)

    result = manager.has_artifact(key, 0)
    assert not result


@pytest.fixture
def retain_manager():
    mock_store = Mock(spec=ArtifactStore)
    mock_store._refcounts = {}
    return ArtifactManager(mock_store)


def test_cache_handle_retains_handle(retain_manager):
    """Test that cache_handle stores handle (no retain when refcount=0)."""
    handle = create_mock_handle(JobArtifactHandle, "job")
    job_key = ArtifactKey(id=JOB_UID, group="job")

    retain_manager.cache_handle(job_key, handle, 0)

    assert retain_manager.get_job_handle(job_key, 0) is handle
    retain_manager._store.retain.assert_not_called()


def test_cache_handle_retains_new_handle_releases_old(retain_manager):
    """
    Test that replacing a handle releases old (no retain when refcount=0).
    """
    old_handle = create_mock_handle(JobArtifactHandle, "job_old")
    new_handle = create_mock_handle(JobArtifactHandle, "job_new")
    job_key = ArtifactKey(id=JOB_UID, group="job")

    retain_manager.cache_handle(job_key, old_handle, 0)

    retain_manager.cache_handle(job_key, new_handle, 0)

    retain_manager._store.retain.assert_not_called()
    retain_manager._store.release.assert_called_once_with(old_handle)


def test_cache_handle_retains_multiple_commits(retain_manager):
    """
    Test that each cache_handle call stores handle (no retain when refcount=0).
    """
    handle1 = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    handle2 = create_mock_handle(WorkPieceArtifactHandle, "wp2")

    wp1_key = ArtifactKey.for_workpiece(WP1_UID)
    wp2_key = ArtifactKey.for_workpiece(WP2_UID)

    retain_manager.cache_handle(wp1_key, handle1, 0)
    retain_manager.cache_handle(wp2_key, handle2, 0)

    assert retain_manager.get_workpiece_handle(wp1_key, 0) is handle1
    assert retain_manager.get_workpiece_handle(wp2_key, 0) is handle2
    retain_manager._store.retain.assert_not_called()


# Tests for report_completion context manager


def test_report_completion_marks_valid(manager):
    """Test report_completion marks entry as VALID."""
    key = ArtifactKey.for_workpiece(WP1_UID)
    manager.declare_generation({key}, 0)

    with manager.report_completion(key, 0) as handle:
        assert handle is None

    assert manager.get_state(key, 0) == NodeState.VALID


def test_report_completion_yields_handle(manager):
    """Test report_completion yields cached handle."""
    key = ArtifactKey.for_workpiece(WP1_UID)
    handle = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    manager.declare_generation({key}, 0)
    manager.cache_handle(key, handle, 0)

    with manager.report_completion(key, 0) as yielded:
        assert yielded is handle

    assert manager.get_state(key, 0) == NodeState.VALID
    manager._store.retain.assert_called_with(handle)
    manager._store.release.assert_called_with(handle)


def test_report_completion_missing_entry_yields_none(manager):
    """Test report_completion yields None for missing entry."""
    key = ArtifactKey.for_workpiece(WP1_UID)

    with manager.report_completion(key, 0) as handle:
        assert handle is None

    assert manager.get_state(key, 0) == NodeState.DIRTY


def test_report_completion_nonexistent_generation_yields_none(manager):
    """Test report_completion yields None for non-existent generation."""
    key = ArtifactKey.for_workpiece(WP1_UID)

    with manager.report_completion(key, 99) as handle:
        assert handle is None


# Tests for report_failure context manager


def test_report_failure_marks_error(manager):
    """Test report_failure marks entry as ERROR."""
    key = ArtifactKey.for_workpiece(WP1_UID)
    manager.declare_generation({key}, 0)

    with manager.report_failure(key, 0) as handle:
        assert handle is None

    assert manager.get_state(key, 0) == NodeState.ERROR


def test_report_failure_yields_and_clears_handle(manager):
    """Test report_failure yields handle and clears it from entry."""
    key = ArtifactKey.for_workpiece(WP1_UID)
    handle = create_mock_handle(WorkPieceArtifactHandle, "wp1")
    manager.declare_generation({key}, 0)
    manager.cache_handle(key, handle, 0)

    with manager.report_failure(key, 0) as yielded:
        assert yielded is handle
        assert manager.get_workpiece_handle(key, 0) is None

    assert manager.get_state(key, 0) == NodeState.ERROR
    manager._store.retain.assert_called_with(handle)
    manager._store.release.assert_called_with(handle)


def test_report_failure_missing_entry_yields_none(manager):
    """Test report_failure yields None for missing entry."""
    key = ArtifactKey.for_workpiece(WP1_UID)

    with manager.report_failure(key, 0) as handle:
        assert handle is None


def test_report_failure_nonexistent_generation_yields_none(manager):
    """Test report_failure yields None for non-existent generation."""
    key = ArtifactKey.for_workpiece(WP1_UID)

    with manager.report_failure(key, 99) as handle:
        assert handle is None


# Tests for report_cancellation context manager


def test_report_cancellation_with_handle_keeps_valid(manager):
    """Test report_cancellation keeps existing handle and marks VALID."""
    key = ArtifactKey.for_workpiece(WP1_UID)
    existing_handle = create_mock_handle(
        WorkPieceArtifactHandle, "wp_existing"
    )
    manager.declare_generation({key}, 0)
    manager.cache_handle(key, existing_handle, 0)

    with manager.report_cancellation(key, 0) as handle:
        assert handle is existing_handle

    assert manager.get_state(key, 0) == NodeState.VALID
    assert manager.get_workpiece_handle(key, 0) is existing_handle


def test_report_cancellation_without_handle_marks_dirty(manager):
    """Test report_cancellation marks DIRTY when no handle exists."""
    key = ArtifactKey.for_workpiece(WP1_UID)
    manager.declare_generation({key}, 0)

    with manager.report_cancellation(key, 0) as handle:
        assert handle is None

    assert manager.get_state(key, 0) == NodeState.DIRTY


def test_report_cancellation_missing_entry_yields_none(manager):
    """Test report_cancellation yields None for missing entry."""
    key = ArtifactKey.for_workpiece(WP1_UID)

    with manager.report_cancellation(key, 0) as handle:
        assert handle is None


def test_report_cancellation_nonexistent_generation_yields_none(manager):
    """Test report_cancellation yields None for non-existent generation."""
    key = ArtifactKey.for_workpiece(WP1_UID)

    with manager.report_cancellation(key, 99) as handle:
        assert handle is None
