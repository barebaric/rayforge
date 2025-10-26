import unittest
from unittest.mock import MagicMock, ANY
import numpy as np
from typing import cast

from rayforge.context import get_context
from rayforge.pipeline.stage.workpiece_view import (
    WorkPieceViewGeneratorStage,
)
from rayforge.pipeline.artifact import (
    RenderContext,
    WorkPieceArtifact,
    WorkPieceArtifactHandle,
    WorkPieceViewArtifact,
    WorkPieceViewArtifactHandle,
)
from rayforge.core.ops import Ops
from rayforge.pipeline import CoordinateSystem
from rayforge.pipeline.artifact.base import VertexData, TextureData
from rayforge.shared.util.colors import ColorSet
from rayforge.pipeline.stage.workpiece_view_runner import (
    make_workpiece_view_artifact_in_subprocess,
)


def create_test_color_set(spec: dict) -> ColorSet:
    """Creates a mock resolved ColorSet for testing without GTK."""
    resolved_data = {}
    for key, colors in spec.items():
        # This simplified resolver just creates a basic LUT for testing
        lut = np.zeros((256, 4), dtype=np.float32)
        if key == "cut":
            # Black to Red gradient for 'cut'
            lut[:, 0] = np.linspace(0, 1, 256)
            lut[:, 3] = 1.0
        elif key == "engrave":
            # Black to White gradient for 'engrave'
            lut[:, 0] = np.linspace(0, 1, 256)
            lut[:, 1] = np.linspace(0, 1, 256)
            lut[:, 2] = np.linspace(0, 1, 256)
            lut[:, 3] = 1.0
        resolved_data[key] = lut
    return ColorSet(_data=resolved_data)


class TestWorkPieceViewStage(unittest.TestCase):
    """Test suite for the WorkPieceViewGeneratorStage."""

    def setUp(self):
        self.mock_artifact_cache = MagicMock()
        self.mock_task_manager = MagicMock()
        self.stage = WorkPieceViewGeneratorStage(
            self.mock_task_manager, self.mock_artifact_cache
        )
        self.handles_to_release: list = []

    def tearDown(self):
        for handle in self.handles_to_release:
            get_context().artifact_store.release(handle)

    def _store_artifact(self, artifact) -> WorkPieceArtifactHandle:
        """Helper to store an artifact and track its handle for cleanup."""
        handle = get_context().artifact_store.put(artifact)
        self.handles_to_release.append(handle)
        return cast(WorkPieceArtifactHandle, handle)

    def _create_vector_artifact(self) -> WorkPieceArtifactHandle:
        """Creates and stores a simple vector-based WorkPieceArtifact."""
        # A 10x10mm red square at (5,5)
        verts = np.array(
            [
                [5, 5, 0],
                [15, 5, 0],
                [15, 5, 0],
                [15, 15, 0],
                [15, 15, 0],
                [5, 15, 0],
                [5, 15, 0],
                [5, 5, 0],
            ],
            dtype=np.float32,
        )
        # Full power (red)
        colors = np.full((verts.shape[0], 4), [1, 0, 0, 1], np.float32)
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            vertex_data=VertexData(
                powered_vertices=verts, powered_colors=colors
            ),
        )
        return self._store_artifact(artifact)

    def _create_texture_artifact(self) -> WorkPieceArtifactHandle:
        """
        Creates and stores a WorkPieceArtifact with a texture that
        simulates the result of assembling two chunks.
        """
        # Y=0 in numpy is the top of the workpiece.
        # Top 30 rows are gray (power 128)
        top_chunk_data = np.full((30, 50), 128, dtype=np.uint8)
        # Bottom 20 rows are white (power 255)
        bottom_chunk_data = np.full((20, 50), 255, dtype=np.uint8)

        full_texture_data = np.zeros((50, 50), dtype=np.uint8)
        full_texture_data[0:30, :] = top_chunk_data
        full_texture_data[30:50, :] = bottom_chunk_data

        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            texture_data=TextureData(
                power_texture_data=full_texture_data,
                dimensions_mm=(50.0, 50.0),
                position_mm=(0.0, 0.0),
            ),
        )
        return self._store_artifact(artifact)

    def test_stage_requests_vector_render(self):
        """
        Tests that the stage correctly calls the task manager for a
        vector artifact.
        """
        step_uid, wp_uid = "s1", "w1"
        source_handle = self._create_vector_artifact()
        self.mock_artifact_cache.get_workpiece_handle.return_value = (
            source_handle
        )
        context = RenderContext(
            pixels_per_mm=(10.0, 10.0),
            show_travel_moves=False,
            margin_px=0,
            color_set_dict={},
        )

        self.stage.request_view_render(step_uid, wp_uid, context)

        self.mock_task_manager.run_process.assert_called_once()
        call_args = self.mock_task_manager.run_process.call_args
        self.assertEqual(
            call_args.kwargs["workpiece_artifact_handle_dict"],
            source_handle.to_dict(),
        )
        self.assertEqual(
            call_args.kwargs["render_context_dict"], context.to_dict()
        )

    def test_stage_handles_events_and_completion(self):
        """
        Tests that the stage correctly handles events from the runner and
        emits its own signals.
        """
        step_uid, wp_uid = "s1", "w1"
        key = (step_uid, wp_uid)
        source_handle = self._create_vector_artifact()
        self.mock_artifact_cache.get_workpiece_handle.return_value = (
            source_handle
        )
        context = RenderContext((10.0, 10.0), False, 0, {})

        self.stage.request_view_render(step_uid, wp_uid, context)

        # Get the callbacks from the task manager call
        self.mock_task_manager.run_process.assert_called_once()
        call_kwargs = self.mock_task_manager.run_process.call_args.kwargs
        when_event_cb = call_kwargs["when_event"]
        when_done_cb = call_kwargs["when_done"]

        mock_task = MagicMock()
        mock_task.key = key
        mock_task.get_status.return_value = "completed"

        # Mock signal handlers
        created_handler = MagicMock()
        updated_handler = MagicMock()
        ready_handler = MagicMock()
        finished_handler = MagicMock()
        self.stage.view_artifact_created.connect(created_handler)
        self.stage.view_artifact_updated.connect(updated_handler)
        self.stage.view_artifact_ready.connect(ready_handler)
        self.stage.generation_finished.connect(finished_handler)

        # Act 1: Simulate "created" event
        handle_dict = {
            "shm_name": "test",
            "bbox_mm": (0, 0, 1, 1),
            "handle_class_name": "WorkPieceViewArtifactHandle",
            "artifact_type_name": "WorkPieceViewArtifact",
        }
        when_event_cb(
            mock_task, "view_artifact_created", {"handle_dict": handle_dict}
        )

        # Assert 1: Both new and old signals fire
        created_handler.assert_called_once()
        ready_handler.assert_called_once()
        self.assertIsInstance(
            created_handler.call_args.kwargs["handle"],
            WorkPieceViewArtifactHandle,
        )
        self.assertIsInstance(
            ready_handler.call_args.kwargs["handle"],
            WorkPieceViewArtifactHandle,
        )

        # Act 2: Simulate "updated" event
        when_event_cb(mock_task, "view_artifact_updated", {})

        # Assert 2
        updated_handler.assert_called_once()
        self.assertEqual(
            updated_handler.call_args.kwargs["step_uid"], step_uid
        )

        # Act 3: Simulate task completion
        when_done_cb(mock_task)

        # Assert 3
        finished_handler.assert_called_once()
        self.assertEqual(finished_handler.call_args.kwargs["key"], key)
        # `ready` handler should NOT be called again on completion
        ready_handler.assert_called_once()

    def test_pixel_perfect_vector_render(self):
        """
        Validates the runner function produces a pixel-perfect render for
        vector data.
        """
        source_handle = self._create_vector_artifact()
        color_set = create_test_color_set({"cut": ("#000", "#F00")})
        context = RenderContext(
            pixels_per_mm=(1.0, 1.0),  # 1px per mm
            show_travel_moves=False,
            margin_px=1,  # Use a margin
            color_set_dict=color_set.to_dict(),
        )
        mock_proxy = MagicMock()

        result = make_workpiece_view_artifact_in_subprocess(
            mock_proxy,
            source_handle.to_dict(),
            context.to_dict(),
        )
        self.assertIsNone(result)

        # Extract handle from the "created" event
        mock_proxy.send_event.assert_any_call("view_artifact_created", ANY)
        created_call = next(
            c
            for c in mock_proxy.send_event.call_args_list
            if c[0][0] == "view_artifact_created"
        )
        handle_dict = created_call[0][1]["handle_dict"]

        self.assertIsNotNone(handle_dict)
        handle = WorkPieceViewArtifactHandle.from_dict(handle_dict)
        self.handles_to_release.append(handle)

        artifact = cast(
            WorkPieceViewArtifact, get_context().artifact_store.get(handle)
        )
        self.assertEqual(artifact.bbox_mm, (5.0, 5.0, 10.0, 10.0))
        self.assertEqual(artifact.bitmap_data.shape, (12, 12, 4))

        pixel_color = artifact.bitmap_data[1, 5]
        # Assert Red is dominant and Alpha is significant
        self.assertGreater(pixel_color[2], 100)  # Red
        self.assertLess(pixel_color[1], 50)  # Green
        self.assertLess(pixel_color[0], 50)  # Blue
        self.assertGreater(pixel_color[3], 100)  # Alpha

    def test_pixel_perfect_texture_chunk_alignment(self):
        """
        Validates the runner correctly renders an artifact that was
        assembled from chunks, ensuring perfect alignment.
        """
        source_handle = self._create_texture_artifact()
        color_set = create_test_color_set({"engrave": ("#000", "#FFF")})
        context = RenderContext(
            pixels_per_mm=(1.0, 1.0),
            show_travel_moves=False,
            margin_px=0,
            color_set_dict=color_set.to_dict(),
        )
        mock_proxy = MagicMock()

        result = make_workpiece_view_artifact_in_subprocess(
            mock_proxy,
            source_handle.to_dict(),
            context.to_dict(),
        )
        self.assertIsNone(result)

        # Extract handle from the "created" event
        mock_proxy.send_event.assert_any_call("view_artifact_created", ANY)
        created_call = next(
            c
            for c in mock_proxy.send_event.call_args_list
            if c[0][0] == "view_artifact_created"
        )
        handle_dict = created_call[0][1]["handle_dict"]

        self.assertIsNotNone(handle_dict)
        handle = WorkPieceViewArtifactHandle.from_dict(handle_dict)
        self.handles_to_release.append(handle)
        artifact = cast(
            WorkPieceViewArtifact, get_context().artifact_store.get(handle)
        )

        self.assertEqual(artifact.bbox_mm, (0.0, 0.0, 50.0, 50.0))
        self.assertEqual(artifact.bitmap_data.shape, (50, 50, 4))

        # Check pixel from the top chunk (power 128 -> gray)
        gray_pixel = artifact.bitmap_data[10, 25]
        self.assertAlmostEqual(gray_pixel[0], 128, delta=1)
        self.assertAlmostEqual(gray_pixel[1], 128, delta=1)
        self.assertAlmostEqual(gray_pixel[2], 128, delta=1)
        self.assertAlmostEqual(gray_pixel[3], 255, delta=1)

        # Check pixel from the bottom chunk (power 255 -> white)
        white_pixel = artifact.bitmap_data[40, 25]
        np.testing.assert_array_almost_equal(
            white_pixel, [255, 255, 255, 255], decimal=0
        )


if __name__ == "__main__":
    unittest.main()
