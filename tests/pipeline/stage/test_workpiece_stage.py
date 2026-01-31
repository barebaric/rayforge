import pytest
from unittest.mock import MagicMock

from rayforge.core.doc import Doc
from rayforge.pipeline.stage.base import PipelineStage
from rayforge.pipeline.stage.workpiece_stage import WorkPiecePipelineStage
from rayforge.context import get_context
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact import WorkPieceArtifact, VertexData
from rayforge.pipeline.coord import CoordinateSystem
import numpy as np


@pytest.fixture
def mock_task_mgr():
    """Provides a mock TaskManager."""
    return MagicMock()


@pytest.fixture
def mock_artifact_cache():
    """Provides a mock ArtifactCache."""
    return MagicMock()


@pytest.fixture
def mock_doc():
    """Provides a mock Doc object."""
    return MagicMock(spec=Doc)


class TestWorkPiecePipelineStage:
    def test_instantiation(self, mock_task_mgr, mock_artifact_cache):
        """Test that WorkPiecePipelineStage can be created."""
        stage = WorkPiecePipelineStage(mock_task_mgr, mock_artifact_cache)
        assert isinstance(stage, PipelineStage)
        assert stage._task_manager is mock_task_mgr
        assert stage._artifact_cache is mock_artifact_cache

    def test_interface_compliance(
        self, mock_task_mgr, mock_artifact_cache, mock_doc
    ):
        """Test that the stage implements all required abstract methods."""
        stage = WorkPiecePipelineStage(mock_task_mgr, mock_artifact_cache)
        # These should not raise NotImplementedError
        stage.reconcile(mock_doc)
        stage.shutdown()

    def test_visual_chunk_available_signal_emitted_on_chunk_ready(
        self, mock_task_mgr, mock_artifact_cache
    ):
        """Test that visual_chunk_available signal is emitted correctly."""
        stage = WorkPiecePipelineStage(mock_task_mgr, mock_artifact_cache)

        # Track signal emissions
        received_signals = []

        def on_chunk_available(sender, key, chunk_handle, generation_id):
            received_signals.append(
                {
                    "key": key,
                    "chunk_handle": chunk_handle,
                    "generation_id": generation_id,
                }
            )

        stage.visual_chunk_available.connect(on_chunk_available)

        # Create a mock chunk artifact
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.line_to(1, 1, 0)

        vertex_data = VertexData(
            powered_vertices=np.array([[0, 0, 0], [1, 1, 0]]),
            powered_colors=np.array([[1, 1, 1, 1], [1, 1, 1, 1]]),
        )

        chunk_artifact = WorkPieceArtifact(
            ops=ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=(10.0, 10.0),
            generation_size=(10.0, 10.0),
            vertex_data=vertex_data,
        )

        handle = get_context().artifact_store.put(
            chunk_artifact, creator_tag="test_chunk"
        )

        try:
            # Create a mock task
            mock_task = MagicMock()
            key = ("step_123", "workpiece_456")
            mock_task.key = key

            # Set up the generation ID map to make the event valid
            stage._generation_id_map[key] = 1

            # Simulate receiving a visual_chunk_ready event
            event_data = {
                "handle_dict": handle.to_dict(),
                "generation_id": 1,
            }

            stage._on_task_event_received(
                mock_task, "visual_chunk_ready", event_data
            )

            # Verify the signal was emitted
            assert len(received_signals) == 1
            signal_data = received_signals[0]
            assert signal_data["key"] == ("step_123", "workpiece_456")
            assert signal_data["generation_id"] == 1
            assert signal_data["chunk_handle"].to_dict() == handle.to_dict()
        finally:
            get_context().artifact_store.release(handle)
