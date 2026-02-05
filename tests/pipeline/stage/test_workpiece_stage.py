import pytest
from unittest.mock import MagicMock

from rayforge.core.doc import Doc
from rayforge.pipeline.stage.base import PipelineStage
from rayforge.pipeline.stage.workpiece_stage import WorkPiecePipelineStage
from rayforge.context import get_context
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.coord import CoordinateSystem


@pytest.fixture
def mock_task_mgr():
    """Provides a mock TaskManager."""
    return MagicMock()


@pytest.fixture
def mock_artifact_manager():
    """Provides a mock ArtifactManager."""
    return MagicMock()


@pytest.fixture
def mock_machine():
    """Provides a mock Machine."""
    return MagicMock()


@pytest.fixture
def mock_doc():
    """Provides a mock Doc object."""
    return MagicMock(spec=Doc)


class TestWorkPiecePipelineStage:
    def test_instantiation(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test that WorkPiecePipelineStage can be created."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        assert isinstance(stage, PipelineStage)
        assert stage._task_manager is mock_task_mgr
        assert stage._artifact_manager is mock_artifact_manager
        assert stage._machine is mock_machine

    def test_interface_compliance(
        self, mock_task_mgr, mock_artifact_manager, mock_machine, mock_doc
    ):
        """Test that the stage implements all required abstract methods."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        # These should not raise NotImplementedError
        stage.reconcile(mock_doc)
        stage.shutdown()

    def test_visual_chunk_available_signal_emitted_on_chunk_ready(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test that visual_chunk_available signal is emitted correctly."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )

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

        chunk_artifact = WorkPieceArtifact(
            ops=ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=(10.0, 10.0),
            generation_size=(10.0, 10.0),
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

            # Configure mock to return the actual handle
            mock_artifact_manager.adopt_artifact.return_value = handle

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

    def test_validate_workpiece_for_launch_with_invalid_size(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test validation rejects workpiece with invalid size."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        mock_workpiece = MagicMock()
        mock_workpiece.size = (0.0, 10.0)
        key = ("step_1", "workpiece_1")

        result = stage._validate_workpiece_for_launch(key, mock_workpiece)

        assert result is False
        mock_artifact_manager.invalidate_for_workpiece.assert_called_once()

    def test_validate_workpiece_for_launch_with_active_task(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test validation cancels existing active task."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        mock_workpiece = MagicMock()
        mock_workpiece.size = (10.0, 10.0)
        key = ("step_1", "workpiece_1")
        mock_task = MagicMock()
        stage._active_tasks[key] = mock_task

        result = stage._validate_workpiece_for_launch(key, mock_workpiece)

        assert result is True
        mock_task_mgr.cancel_task.assert_called_once()

    def test_validate_workpiece_for_launch_with_valid_workpiece(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test validation passes for valid workpiece."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        mock_workpiece = MagicMock()
        mock_workpiece.size = (10.0, 10.0)
        key = ("step_1", "workpiece_1")

        result = stage._validate_workpiece_for_launch(key, mock_workpiece)

        assert result is True

    def test_prepare_generation_id(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test generation ID preparation and signal sending."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        mock_step = MagicMock()
        mock_step.uid = "step_1"
        mock_workpiece = MagicMock()
        mock_workpiece.uid = "workpiece_1"
        key = ("step_1", "workpiece_1")

        gen_id = stage._prepare_generation_id(key, mock_step, mock_workpiece)

        assert gen_id == 1
        assert stage._generation_id_map[key] == 1

    def test_prepare_generation_id_increments(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test generation ID increments on subsequent calls."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        mock_step = MagicMock()
        mock_step.uid = "step_1"
        mock_workpiece = MagicMock()
        mock_workpiece.uid = "workpiece_1"
        key = ("step_1", "workpiece_1")
        stage._generation_id_map[key] = 1

        gen_id = stage._prepare_generation_id(key, mock_step, mock_workpiece)

        assert gen_id == 2
        assert stage._generation_id_map[key] == 2

    def test_prepare_task_settings_without_machine(
        self, mock_task_mgr, mock_artifact_manager
    ):
        """Test settings preparation fails without machine."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, None
        )
        mock_step = MagicMock()

        result = stage._prepare_task_settings(mock_step)

        assert result is None

    def test_prepare_task_settings_with_laser_error(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test settings preparation fails when laser selection fails."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        mock_step = MagicMock()
        mock_step.get_selected_laser.side_effect = ValueError("No laser")

        result = stage._prepare_task_settings(mock_step)

        assert result is None

    def test_prepare_task_settings_success(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test settings preparation succeeds with valid inputs."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        mock_step = MagicMock()
        mock_step.get_settings.return_value = {"speed": 100}
        mock_laser = MagicMock()
        mock_laser.to_dict.return_value = {"power": 50}
        mock_step.get_selected_laser.return_value = mock_laser
        mock_machine.supports_arcs = True
        mock_machine.arc_tolerance = 0.1

        result = stage._prepare_task_settings(mock_step)

        assert result is not None
        settings, laser_dict = result
        assert settings["machine_supports_arcs"] is True
        assert settings["arc_tolerance"] == 0.1
        assert laser_dict == {"power": 50}

    def test_validate_task_event_with_none_generation_id(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test validation fails with None generation ID."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ("step_1", "workpiece_1")

        is_valid, is_stale = stage._validate_task_event(key, None)

        assert is_valid is False
        assert is_stale is False

    def test_validate_task_event_with_stale_id(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test validation detects stale generation ID."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ("step_1", "workpiece_1")
        stage._generation_id_map[key] = 2

        is_valid, is_stale = stage._validate_task_event(key, 1)

        assert is_valid is False
        assert is_stale is True

    def test_validate_task_event_with_valid_id(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test validation passes with matching generation ID."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ("step_1", "workpiece_1")
        stage._generation_id_map[key] = 1

        is_valid, is_stale = stage._validate_task_event(key, 1)

        assert is_valid is True
        assert is_stale is False

    def test_set_adoption_event(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test adoption event is set correctly."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ("step_1", "workpiece_1")
        mock_event = MagicMock()
        stage._adoption_events[key] = mock_event

        stage._set_adoption_event(key)

        mock_event.set.assert_called_once()

    def test_set_adoption_event_with_none_event(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test adoption event handles None gracefully."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ("step_1", "workpiece_1")

        stage._set_adoption_event(key)

    def test_handle_canceled_task(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test canceled task handler sends finished signal."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ("step_1", "workpiece_1")
        mock_step = MagicMock()
        mock_workpiece = MagicMock()

        received_signals = []

        def on_finished(sender, step, workpiece, generation_id):
            received_signals.append(
                {
                    "step": step,
                    "workpiece": workpiece,
                    "generation_id": generation_id,
                }
            )

        stage.generation_finished.connect(on_finished)
        stage._handle_canceled_task(key, mock_step, mock_workpiece, 1)

        assert len(received_signals) == 1
        assert received_signals[0]["step"] == mock_step
        assert received_signals[0]["workpiece"] == mock_workpiece
        assert received_signals[0]["generation_id"] == 1

    def test_check_result_stale_due_to_size_with_scalable(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test stale check returns False for scalable artifacts."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ("step_1", "workpiece_1")
        mock_workpiece = MagicMock()
        mock_workpiece.size = (20.0, 20.0)
        mock_handle = MagicMock()
        mock_handle.is_scalable = True
        mock_artifact_manager.get_workpiece_handle.return_value = mock_handle

        result = stage._check_result_stale_due_to_size(key, mock_workpiece)

        assert result is False

    def test_check_result_stale_due_to_size_with_size_change(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test stale check detects size change for non-scalable."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ("step_1", "workpiece_1")
        mock_workpiece = MagicMock()
        mock_workpiece.size = (20.0, 20.0)
        mock_handle = MagicMock()
        mock_handle.is_scalable = False
        mock_handle.generation_size = (10.0, 10.0)
        mock_artifact_manager.get_workpiece_handle.return_value = mock_handle

        result = stage._check_result_stale_due_to_size(key, mock_workpiece)

        assert result is True

    def test_check_result_stale_due_to_size_no_change(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test stale check returns False when size matches."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ("step_1", "workpiece_1")
        mock_workpiece = MagicMock()
        mock_workpiece.size = (10.0, 10.0)
        mock_handle = MagicMock()
        mock_handle.is_scalable = False
        mock_handle.generation_size = (10.0, 10.0)
        mock_artifact_manager.get_workpiece_handle.return_value = mock_handle

        result = stage._check_result_stale_due_to_size(key, mock_workpiece)

        assert result is False

    def test_handle_failed_task(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test failed task handler cleans up entry."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ("step_1", "workpiece_1")
        mock_step = MagicMock()
        mock_step.name = "test_step"
        mock_workpiece = MagicMock()
        mock_workpiece.name = "test_workpiece"

        stage._handle_failed_task(key, mock_step, mock_workpiece)

        mock_artifact_manager.invalidate_for_workpiece.assert_called_once()

    def test_send_generation_finished_signal(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test generation finished signal is sent."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ("step_1", "workpiece_1")
        mock_step = MagicMock()
        mock_workpiece = MagicMock()

        received_signals = []

        def on_finished(sender, step, workpiece, generation_id):
            received_signals.append(
                {
                    "step": step,
                    "workpiece": workpiece,
                    "generation_id": generation_id,
                }
            )

        stage.generation_finished.connect(on_finished)
        stage._send_generation_finished_signal(
            key, mock_step, mock_workpiece, 1
        )

        assert len(received_signals) == 1
        assert received_signals[0]["step"] == mock_step
        assert received_signals[0]["workpiece"] == mock_workpiece
        assert received_signals[0]["generation_id"] == 1
