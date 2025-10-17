import unittest
from rayforge.pipeline.artifact import (
    WorkPieceArtifactHandle,
    StepOpsArtifactHandle,
    StepRenderArtifactHandle,
    create_handle_from_dict,
)


class TestArtifactHandle(unittest.TestCase):
    """Test suite for the ArtifactHandle data class."""

    def test_workpiece_handle_serialization_round_trip(self):
        """
        Tests that a WorkPieceArtifactHandle can be converted to a dict and
        back.
        """
        handle = WorkPieceArtifactHandle(
            shm_name="test_shm_123",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=False,
            source_coordinate_system_name="PIXEL_SPACE",
            source_dimensions=(1024, 768),
            generation_size=(100.0, 75.0),
            time_estimate=123.4,
            dimensions_mm=(100.0, 75.0),
            position_mm=(10.0, 20.0),
            array_metadata={
                "ops_types": {"dtype": "int32", "shape": (10,), "offset": 0},
                "power_texture": {
                    "dtype": "uint8",
                    "shape": (768, 1024),
                    "offset": 40,
                },
            },
        )

        handle_dict = handle.to_dict()
        reconstructed_handle = create_handle_from_dict(handle_dict)

        self.assertEqual(handle, reconstructed_handle)
        self.assertIsInstance(reconstructed_handle, WorkPieceArtifactHandle)
        self.assertEqual(
            reconstructed_handle.array_metadata["power_texture"]["shape"],
            (768, 1024),
        )

    def test_step_ops_handle_serialization_round_trip(self):
        """
        Tests that a StepOpsArtifactHandle can be converted to a dict and back.
        """
        handle = StepOpsArtifactHandle(
            shm_name="test_shm_ops_456",
            handle_class_name="StepOpsArtifactHandle",
            artifact_type_name="StepOpsArtifact",
            is_scalable=False,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=None,
            time_estimate=99.9,
            array_metadata={
                "ops_types": {"dtype": "int32", "shape": (50,), "offset": 0},
            },
        )
        handle_dict = handle.to_dict()
        reconstructed_handle = create_handle_from_dict(handle_dict)

        self.assertEqual(handle, reconstructed_handle)
        self.assertIsInstance(reconstructed_handle, StepOpsArtifactHandle)
        self.assertEqual(reconstructed_handle.time_estimate, 99.9)

    def test_step_render_handle_serialization_round_trip(self):
        """
        Tests that a StepRenderArtifactHandle can be converted to a dict and
        back.
        """
        handle = StepRenderArtifactHandle(
            shm_name="test_shm_render_789",
            handle_class_name="StepRenderArtifactHandle",
            artifact_type_name="StepRenderArtifact",
            is_scalable=False,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=None,
            time_estimate=None,
            array_metadata={
                "powered_vertices": {
                    "dtype": "float32",
                    "shape": (100, 3),
                    "offset": 0,
                },
                "texture_data_0": {
                    "dtype": "uint8",
                    "shape": (50, 50),
                    "offset": 1200,
                },
            },
        )
        handle_dict = handle.to_dict()
        reconstructed_handle = create_handle_from_dict(handle_dict)

        self.assertEqual(handle, reconstructed_handle)
        self.assertIsInstance(reconstructed_handle, StepRenderArtifactHandle)
        self.assertIsNone(reconstructed_handle.time_estimate)


if __name__ == "__main__":
    unittest.main()
