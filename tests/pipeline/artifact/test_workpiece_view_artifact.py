import unittest
import numpy as np
from typing import cast
from multiprocessing import shared_memory
from rayforge.context import get_context
from rayforge.pipeline.artifact.workpiece_view import (
    RenderContext,
    WorkPieceViewArtifact,
    WorkPieceViewArtifactHandle,
)
from rayforge.pipeline.artifact import create_handle_from_dict


class TestWorkPieceViewArtifact(unittest.TestCase):
    """Test suite for the new WorkPieceViewArtifact and its components."""

    def setUp(self):
        """Initializes a list to track created handles for cleanup."""
        self.handles_to_release = []

    def tearDown(self):
        """Ensures all SHM blocks are released after tests."""
        for handle in self.handles_to_release:
            get_context().artifact_store.release(handle)

    def test_render_context_serialization(self):
        """Tests that RenderContext can be serialized and deserialized."""
        original_context = RenderContext(
            pixels_per_mm=(10.5, 10.5),
            show_travel_moves=True,
            margin_px=5,
            color_set_dict={"cut": ["#ff00ff", 1.0]},
        )
        context_dict = original_context.to_dict()
        reconstructed_context = RenderContext.from_dict(context_dict)
        self.assertEqual(original_context, reconstructed_context)

    def test_handle_serialization(self):
        """
        Tests that WorkPieceViewArtifactHandle can be serialized and
        deserialized.
        """
        original_handle = WorkPieceViewArtifactHandle(
            shm_name="test_shm_123",
            handle_class_name="WorkPieceViewArtifactHandle",
            artifact_type_name="WorkPieceViewArtifact",
            bbox_mm=(10.0, 20.0, 100.0, 50.0),
            array_metadata={
                "bitmap_data": {
                    "dtype": "uint8",
                    "shape": (100, 200, 4),
                    "offset": 0,
                }
            },
        )
        handle_dict = original_handle.to_dict()
        # Use the global factory function for polymorphic deserialization
        reconstructed_handle = create_handle_from_dict(handle_dict)

        self.assertIsInstance(
            reconstructed_handle, WorkPieceViewArtifactHandle
        )
        self.assertEqual(original_handle, reconstructed_handle)

    def test_artifact_store_lifecycle(self):
        """
        Tests the full put -> get -> release lifecycle with a
        WorkPieceViewArtifact.
        """
        # 1. Create a sample artifact
        bitmap_data = np.arange(10 * 20 * 4, dtype=np.uint8).reshape(
            (20, 10, 4)
        )
        bbox_mm = (5.0, 5.0, 50.0, 100.0)
        original_artifact = WorkPieceViewArtifact(
            bitmap_data=bitmap_data, bbox_mm=bbox_mm
        )

        # 2. Put the artifact into shared memory
        handle_base = get_context().artifact_store.put(original_artifact)
        self.handles_to_release.append(handle_base)

        # Cast for type checker
        handle = cast(WorkPieceViewArtifactHandle, handle_base)

        # 3. Verify the handle type and metadata
        self.assertIsInstance(handle, WorkPieceViewArtifactHandle)
        self.assertEqual(handle.bbox_mm, bbox_mm)

        # 4. Get the artifact back
        retrieved_base = get_context().artifact_store.get(handle)

        # Cast for type checker
        retrieved_artifact = cast(WorkPieceViewArtifact, retrieved_base)

        # 5. Verify the retrieved artifact's data
        self.assertIsInstance(retrieved_artifact, WorkPieceViewArtifact)
        self.assertEqual(retrieved_artifact.bbox_mm, bbox_mm)
        np.testing.assert_array_equal(
            retrieved_artifact.bitmap_data, bitmap_data
        )

        # 6. Release the memory
        get_context().artifact_store.release(handle)

        # 7. Verify that the memory is no longer accessible
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.shm_name)


if __name__ == "__main__":
    unittest.main()
