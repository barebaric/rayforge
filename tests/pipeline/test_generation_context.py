import unittest
from typing import cast
from unittest.mock import MagicMock
import uuid
from rayforge.context import get_context
from rayforge.core.ops import Ops
from rayforge.pipeline import CoordinateSystem
from rayforge.pipeline.artifact.key import ArtifactKey
from rayforge.pipeline.artifact.handle import BaseArtifactHandle
from rayforge.pipeline.artifact.workpiece import (
    WorkPieceArtifact,
    WorkPieceArtifactHandle,
)
from rayforge.pipeline.context import GenerationContext


class TestGenerationContext(unittest.TestCase):
    """Test suite for the GenerationContext class."""

    def test_initialization(self):
        """Test that a GenerationContext is initialized correctly."""
        ctx = GenerationContext(generation_id=1)

        self.assertEqual(ctx.generation_id, 1)
        self.assertEqual(ctx.active_tasks, set())
        self.assertEqual(ctx.resources, set())
        self.assertFalse(ctx.has_active_tasks())

    def test_add_task(self):
        """Test that tasks can be added to the context."""
        ctx = GenerationContext(generation_id=1)
        key = ArtifactKey.for_workpiece(str(uuid.uuid4()))

        ctx.add_task(key)

        self.assertIn(key, ctx.active_tasks)
        self.assertTrue(ctx.has_active_tasks())

    def test_task_did_finish(self):
        """Test that tasks can be marked as finished."""
        ctx = GenerationContext(generation_id=1)
        key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
        ctx.add_task(key)

        ctx.task_did_finish(key)

        self.assertNotIn(key, ctx.active_tasks)
        self.assertFalse(ctx.has_active_tasks())

    def test_task_did_finish_idempotent(self):
        """Test that task_did_finish can be called multiple times."""
        ctx = GenerationContext(generation_id=1)
        key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
        ctx.add_task(key)

        ctx.task_did_finish(key)
        ctx.task_did_finish(key)

        self.assertFalse(ctx.has_active_tasks())

    def test_add_multiple_tasks(self):
        """Test that multiple tasks can be tracked."""
        ctx = GenerationContext(generation_id=1)
        key1 = ArtifactKey.for_workpiece(str(uuid.uuid4()))
        key2 = ArtifactKey.for_step(str(uuid.uuid4()))
        key3 = ArtifactKey.for_view(str(uuid.uuid4()))

        ctx.add_task(key1)
        ctx.add_task(key2)
        ctx.add_task(key3)

        self.assertEqual(len(ctx.active_tasks), 3)
        self.assertIn(key1, ctx.active_tasks)
        self.assertIn(key2, ctx.active_tasks)
        self.assertIn(key3, ctx.active_tasks)

    def test_add_resource(self):
        """Test that resources can be added to the context."""
        ctx = GenerationContext(generation_id=1)
        handle = MagicMock(spec=BaseArtifactHandle)

        ctx.add_resource(handle)

        self.assertIn(handle, ctx.resources)

    def test_add_multiple_resources(self):
        """Test that multiple resources can be tracked."""
        ctx = GenerationContext(generation_id=1)
        handle1 = MagicMock(spec=BaseArtifactHandle)
        handle2 = MagicMock(spec=BaseArtifactHandle)

        ctx.add_resource(handle1)
        ctx.add_resource(handle2)

        self.assertEqual(len(ctx.resources), 2)
        self.assertIn(handle1, ctx.resources)
        self.assertIn(handle2, ctx.resources)

    def test_shutdown_calls_release_callback(self):
        """Test that shutdown releases all tracked resources."""
        release_callback = MagicMock()
        ctx = GenerationContext(
            generation_id=1, release_callback=release_callback
        )
        handle1 = MagicMock(spec=BaseArtifactHandle)
        handle2 = MagicMock(spec=BaseArtifactHandle)
        ctx.add_resource(handle1)
        ctx.add_resource(handle2)

        ctx.shutdown()

        self.assertEqual(release_callback.call_count, 2)
        release_callback.assert_any_call(handle1)
        release_callback.assert_any_call(handle2)

    def test_shutdown_clears_resources(self):
        """Test that shutdown clears the resources set."""
        release_callback = MagicMock()
        ctx = GenerationContext(
            generation_id=1, release_callback=release_callback
        )
        handle = MagicMock(spec=BaseArtifactHandle)
        ctx.add_resource(handle)

        ctx.shutdown()

        self.assertEqual(ctx.resources, set())

    def test_shutdown_idempotent(self):
        """Test that shutdown can be called multiple times safely."""
        release_callback = MagicMock()
        ctx = GenerationContext(
            generation_id=1, release_callback=release_callback
        )
        handle = MagicMock(spec=BaseArtifactHandle)
        ctx.add_resource(handle)

        ctx.shutdown()
        ctx.shutdown()

        self.assertEqual(release_callback.call_count, 1)

    def test_shutdown_without_callback(self):
        """Test that shutdown works without a release callback."""
        ctx = GenerationContext(generation_id=1)
        handle = MagicMock(spec=BaseArtifactHandle)
        ctx.add_resource(handle)

        ctx.shutdown()

        self.assertEqual(ctx.resources, set())

    def test_active_tasks_returns_copy(self):
        """Test that active_tasks returns a copy, not the original set."""
        ctx = GenerationContext(generation_id=1)
        key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
        ctx.add_task(key)

        tasks = ctx.active_tasks
        tasks.clear()

        self.assertTrue(ctx.has_active_tasks())

    def test_resources_returns_copy(self):
        """Test that resources returns a copy, not the original set."""
        ctx = GenerationContext(generation_id=1)
        handle = MagicMock(spec=BaseArtifactHandle)
        ctx.add_resource(handle)

        resources = ctx.resources
        resources.clear()

        self.assertEqual(len(ctx.resources), 1)


class TestArtifactStoreContextIntegration(unittest.TestCase):
    """Test suite for Step 3: ArtifactStore.put with GenerationContext."""

    def setUp(self):
        self.handles_to_release = []

    def tearDown(self):
        for handle in self.handles_to_release:
            get_context().artifact_store.release(handle)

    def _create_sample_artifact(self) -> WorkPieceArtifact:
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

    def test_put_tracks_resource_in_context(self):
        """
        Test that ArtifactStore.put registers the handle with the
        GenerationContext when provided.
        """
        ctx = GenerationContext(generation_id=1)
        store = get_context().artifact_store
        artifact = self._create_sample_artifact()

        self.assertEqual(len(ctx.resources), 0)

        handle = store.put(artifact, generation_context=ctx)
        self.handles_to_release.append(handle)

        self.assertEqual(len(ctx.resources), 1)
        self.assertIn(handle, ctx.resources)

    def test_put_tracks_multiple_resources_in_context(self):
        """
        Test that multiple ArtifactStore.put calls track all handles
        in the GenerationContext.
        """
        ctx = GenerationContext(generation_id=1)
        store = get_context().artifact_store

        artifact1 = self._create_sample_artifact()
        artifact2 = self._create_sample_artifact()

        handle1 = store.put(artifact1, generation_context=ctx)
        handle2 = store.put(artifact2, generation_context=ctx)
        self.handles_to_release.extend([handle1, handle2])

        self.assertEqual(len(ctx.resources), 2)
        self.assertIn(handle1, ctx.resources)
        self.assertIn(handle2, ctx.resources)

    def test_put_without_context_does_not_track(self):
        """
        Test that ArtifactStore.put without a context does not
        affect any GenerationContext.
        """
        ctx = GenerationContext(generation_id=1)
        store = get_context().artifact_store
        artifact = self._create_sample_artifact()

        handle = store.put(artifact)
        self.handles_to_release.append(handle)

        self.assertEqual(len(ctx.resources), 0)

    def test_context_resources_are_real_handles(self):
        """
        Test that the resources tracked in the context are real handles
        that can be used to retrieve artifacts.
        """
        ctx = GenerationContext(generation_id=1)
        store = get_context().artifact_store
        artifact = self._create_sample_artifact()

        handle = store.put(artifact, generation_context=ctx)
        self.handles_to_release.append(handle)

        resources = ctx.resources
        self.assertEqual(len(resources), 1)

        tracked_handle = list(resources)[0]
        self.assertIsInstance(tracked_handle, WorkPieceArtifactHandle)

        retrieved = store.get(tracked_handle)
        self.assertIsInstance(retrieved, WorkPieceArtifact)
        retrieved_wp = cast(WorkPieceArtifact, retrieved)
        self.assertEqual(retrieved_wp.generation_size, (50, 50))


class TestGenerationContextAutoShutdown(unittest.TestCase):
    """Test suite for context-driven cleanup."""

    def test_task_did_finish_triggers_shutdown_when_superseded(self):
        """Test that task_did_finish triggers shutdown when superseded."""
        release_callback = MagicMock()
        is_superseded = MagicMock(return_value=True)
        ctx = GenerationContext(
            generation_id=1,
            release_callback=release_callback,
            is_superseded_callback=is_superseded,
        )
        key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
        handle = MagicMock(spec=BaseArtifactHandle)
        ctx.add_task(key)
        ctx.add_resource(handle)

        ctx.task_did_finish(key)

        self.assertTrue(ctx.is_shutdown)
        release_callback.assert_called_once_with(handle)

    def test_task_did_finish_no_shutdown_when_active(self):
        """Test that task_did_finish does not shutdown when still active."""
        release_callback = MagicMock()
        is_superseded = MagicMock(return_value=False)
        ctx = GenerationContext(
            generation_id=1,
            release_callback=release_callback,
            is_superseded_callback=is_superseded,
        )
        key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
        handle = MagicMock(spec=BaseArtifactHandle)
        ctx.add_task(key)
        ctx.add_resource(handle)

        ctx.task_did_finish(key)

        self.assertFalse(ctx.is_shutdown)
        release_callback.assert_not_called()

    def test_task_did_finish_no_shutdown_with_remaining_tasks(self):
        """Test no shutdown when superseded but tasks remain."""
        release_callback = MagicMock()
        is_superseded = MagicMock(return_value=True)
        ctx = GenerationContext(
            generation_id=1,
            release_callback=release_callback,
            is_superseded_callback=is_superseded,
        )
        key1 = ArtifactKey.for_workpiece(str(uuid.uuid4()))
        key2 = ArtifactKey.for_step(str(uuid.uuid4()))
        handle = MagicMock(spec=BaseArtifactHandle)
        ctx.add_task(key1)
        ctx.add_task(key2)
        ctx.add_resource(handle)

        ctx.task_did_finish(key1)

        self.assertFalse(ctx.is_shutdown)
        release_callback.assert_not_called()

    def test_last_task_finishes_triggers_shutdown(self):
        """Test that the last task finishing triggers shutdown."""
        release_callback = MagicMock()
        is_superseded = MagicMock(return_value=True)
        ctx = GenerationContext(
            generation_id=1,
            release_callback=release_callback,
            is_superseded_callback=is_superseded,
        )
        key1 = ArtifactKey.for_workpiece(str(uuid.uuid4()))
        key2 = ArtifactKey.for_step(str(uuid.uuid4()))
        handle = MagicMock(spec=BaseArtifactHandle)
        ctx.add_task(key1)
        ctx.add_task(key2)
        ctx.add_resource(handle)

        ctx.task_did_finish(key1)
        self.assertFalse(ctx.is_shutdown)

        ctx.task_did_finish(key2)
        self.assertTrue(ctx.is_shutdown)
        release_callback.assert_called_once_with(handle)

    def test_no_auto_shutdown_without_callbacks(self):
        """Test no auto-shutdown when no callbacks are provided."""
        ctx = GenerationContext(generation_id=1)
        key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
        handle = MagicMock(spec=BaseArtifactHandle)
        ctx.add_task(key)
        ctx.add_resource(handle)

        ctx.task_did_finish(key)

        self.assertFalse(ctx.is_shutdown)

    def test_is_superseded_checked_at_task_finish(self):
        """Test that is_superseded is checked when task finishes."""
        is_superseded = MagicMock(return_value=False)
        ctx = GenerationContext(
            generation_id=1,
            is_superseded_callback=is_superseded,
        )
        key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
        ctx.add_task(key)

        ctx.task_did_finish(key)

        is_superseded.assert_called_once()


if __name__ == "__main__":
    unittest.main()
