import unittest
from typing import cast
import numpy as np
from multiprocessing import shared_memory
from rayforge.core.ops import Ops
from rayforge.pipeline import CoordinateSystem
from rayforge.pipeline.artifact import (
    ArtifactStore,
    VectorArtifact,
    HybridRasterArtifact,
)


class TestArtifactStore(unittest.TestCase):
    """Test suite for the ArtifactStore shared memory manager."""

    def setUp(self):
        """Initializes a list to track created handles for cleanup."""
        self.handles_to_release = []

    def tearDown(self):
        """
        Ensures all shared memory blocks created during tests are released.
        """
        for handle in self.handles_to_release:
            ArtifactStore.release(handle)

    def _create_sample_vector_artifact(self) -> VectorArtifact:
        """Helper to generate a consistent vector artifact for tests."""
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.line_to(10, 0, 0)
        # Arc from (10,0) to (0,10) centered at (0,0).
        # Center offset from start point (10,0) is (-10, 0).
        ops.arc_to(0, 10, i=-10, j=0, clockwise=False, z=0)
        return VectorArtifact(
            ops=ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=(100, 100),
            generation_size=(50, 50),
        )

    def _create_sample_hybrid_artifact(self) -> HybridRasterArtifact:
        """Helper to generate a consistent hybrid artifact for tests."""
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.scan_to(10, 0, 0, power_values=bytearray(range(256)))
        texture = np.arange(10000, dtype=np.uint8).reshape((100, 100))
        return HybridRasterArtifact(
            ops=ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            power_texture_data=texture,
            dimensions_mm=(50.0, 50.0),
            position_mm=(5.0, 10.0),
            source_dimensions=(200, 200),
            generation_size=(50, 50),
        )

    def test_internal_conversion_round_trip(self):
        """
        Tests the private conversion methods in isolation without shared
        memory.
        This validates the core serialization logic.
        """
        for artifact_factory in [
            self._create_sample_vector_artifact,
            self._create_sample_hybrid_artifact,
        ]:
            with self.subTest(
                artifact_type=artifact_factory().__class__.__name__
            ):
                original_artifact = artifact_factory()

                # Simulate the `put` process
                arrays, _ = ArtifactStore._convert_artifact_to_arrays(
                    original_artifact
                )

                # Create a dummy handle with the necessary metadata
                handle = ArtifactStore.put(original_artifact)
                self.handles_to_release.append(handle)  # Ensure cleanup

                # Simulate the `get` process
                reconstructed_artifact = (
                    ArtifactStore._reconstruct_artifact_from_arrays(
                        handle, arrays
                    )
                )

                # Verify equality
                self.assertEqual(
                    original_artifact.to_dict(),
                    reconstructed_artifact.to_dict(),
                )

    def test_put_get_release_vector_artifact(self):
        """
        Tests the full put -> get -> release lifecycle with a
        VectorArtifact.
        """
        original_artifact = self._create_sample_vector_artifact()

        # 1. Put the artifact into shared memory
        handle = ArtifactStore.put(original_artifact)
        self.handles_to_release.append(handle)

        # 2. Get the artifact back
        retrieved_artifact = ArtifactStore.get(handle)

        # 3. Verify the retrieved data
        self.assertIsInstance(retrieved_artifact, VectorArtifact)
        self.assertEqual(
            len(original_artifact.ops.commands),
            len(retrieved_artifact.ops.commands),
        )
        self.assertEqual(
            original_artifact.generation_size,
            retrieved_artifact.generation_size,
        )

        # 4. Release the memory
        ArtifactStore.release(handle)

        # 5. Verify that the memory is no longer accessible
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.shm_name)

    def test_put_get_release_hybrid_artifact(self):
        """
        Tests the full put -> get -> release lifecycle with a
        HybridRasterArtifact.
        """
        original_artifact = self._create_sample_hybrid_artifact()

        # 1. Put
        handle = ArtifactStore.put(original_artifact)
        self.handles_to_release.append(handle)

        # 2. Get and assert type to help type checker
        retrieved_artifact = ArtifactStore.get(handle)
        self.assertIsInstance(retrieved_artifact, HybridRasterArtifact)
        retrieved_hybrid = cast(HybridRasterArtifact, retrieved_artifact)

        # 3. Verify hybrid-specific attributes
        self.assertEqual(
            original_artifact.dimensions_mm, retrieved_hybrid.dimensions_mm
        )
        np.testing.assert_array_equal(
            original_artifact.power_texture_data,
            retrieved_hybrid.power_texture_data,
        )
        self.assertEqual(
            len(original_artifact.ops.commands),
            len(retrieved_hybrid.ops.commands),
        )

        # 4. Release
        ArtifactStore.release(handle)

        # 5. Verify release
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.shm_name)


if __name__ == "__main__":
    unittest.main()
