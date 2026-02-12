import uuid
import pytest
from unittest.mock import MagicMock

from rayforge.core.doc import Doc
from rayforge.pipeline.artifact.key import ArtifactKey
from rayforge.pipeline.stage.base import PipelineStage
from rayforge.pipeline.stage.workpiece_stage import WorkPiecePipelineStage


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


@pytest.fixture
def workpiece_uuid():
    """Provides a valid UUID for workpiece tests."""
    return str(uuid.uuid4())


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
        """Test that stage implements all required abstract methods."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        # These should not raise NotImplementedError
        stage.reconcile(mock_doc, 1)
        stage.shutdown()

    def test_validate_workpiece_for_launch_with_invalid_size(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        workpiece_uuid,
    ):
        """Test validation rejects workpiece with invalid size."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        mock_workpiece = MagicMock()
        mock_workpiece.size = (0.0, 10.0)
        key = ArtifactKey.for_workpiece(workpiece_uuid)

        result = stage._validate_workpiece_for_launch(key, mock_workpiece)

        assert result is False
        mock_artifact_manager.invalidate_for_workpiece.assert_called_once_with(
            key
        )

    def test_validate_workpiece_for_launch_with_valid_workpiece(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        workpiece_uuid,
    ):
        """Test validation passes for valid workpiece."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        mock_workpiece = MagicMock()
        mock_workpiece.size = (10.0, 10.0)
        key = ArtifactKey.for_workpiece(workpiece_uuid)

        result = stage._validate_workpiece_for_launch(key, mock_workpiece)

        assert result is True

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

    def test_handle_canceled_task(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        workpiece_uuid,
    ):
        """Test canceled task handler sends finished signal."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ArtifactKey.for_workpiece(workpiece_uuid)
        mock_step = MagicMock()
        mock_workpiece = MagicMock()

        received_signals = []

        def on_finished(sender, step, workpiece, generation_id, task_status):
            received_signals.append(
                {
                    "step": step,
                    "workpiece": workpiece,
                    "generation_id": generation_id,
                    "task_status": task_status,
                }
            )

        stage.generation_finished.connect(on_finished)
        ledger_key = ArtifactKey.for_workpiece(workpiece_uuid)
        stage._handle_canceled_task(
            key, ledger_key, mock_step, mock_workpiece, 1
        )

        assert len(received_signals) == 1
        assert received_signals[0]["step"] == mock_step
        assert received_signals[0]["workpiece"] == mock_workpiece
        assert received_signals[0]["generation_id"] == 1
        assert received_signals[0]["task_status"] == "canceled"

    def test_check_result_stale_due_to_size_with_scalable(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        workpiece_uuid,
    ):
        """Test stale check returns False for scalable artifacts."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ArtifactKey.for_workpiece(workpiece_uuid)
        mock_workpiece = MagicMock()
        mock_workpiece.size = (20.0, 20.0)
        mock_handle = MagicMock()
        mock_handle.is_scalable = True
        mock_artifact_manager.get_workpiece_handle.return_value = mock_handle

        result = stage._check_result_stale_due_to_size(key, mock_workpiece, 1)

        assert result is False

    def test_check_result_stale_due_to_size_with_size_change(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        workpiece_uuid,
    ):
        """Test stale check detects size change for non-scalable."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ArtifactKey.for_workpiece(workpiece_uuid)
        mock_workpiece = MagicMock()
        mock_workpiece.size = (20.0, 20.0)
        mock_handle = MagicMock()
        mock_handle.is_scalable = False
        mock_handle.generation_size = (10.0, 10.0)
        mock_artifact_manager.get_workpiece_handle.return_value = mock_handle

        result = stage._check_result_stale_due_to_size(key, mock_workpiece, 1)

        assert result is True

    def test_check_result_stale_due_to_size_no_change(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        workpiece_uuid,
    ):
        """Test stale check returns False when size matches."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ArtifactKey.for_workpiece(workpiece_uuid)
        mock_workpiece = MagicMock()
        mock_workpiece.size = (10.0, 10.0)
        mock_handle = MagicMock()
        mock_handle.is_scalable = False
        mock_handle.generation_size = (10.0, 10.0)
        mock_artifact_manager.get_workpiece_handle.return_value = mock_handle

        result = stage._check_result_stale_due_to_size(key, mock_workpiece, 1)

        assert result is False

    def test_handle_failed_task(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        workpiece_uuid,
    ):
        """Test failed task handler marks entry as error."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        ledger_key = ArtifactKey.for_workpiece(workpiece_uuid)
        mock_step = MagicMock()
        mock_step.name = "test_step"
        mock_workpiece = MagicMock()
        mock_workpiece.name = "test_workpiece"

        stage._handle_failed_task(ledger_key, mock_step, mock_workpiece, 1)

        mock_artifact_manager.fail_generation.assert_called_once()

    def test_send_generation_finished_signal(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        workpiece_uuid,
    ):
        """Test generation finished signal is sent."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        key = ArtifactKey.for_workpiece(workpiece_uuid)
        mock_step = MagicMock()
        mock_workpiece = MagicMock()

        received_signals = []

        def on_finished(sender, step, workpiece, generation_id, task_status):
            received_signals.append(
                {
                    "step": step,
                    "workpiece": workpiece,
                    "generation_id": generation_id,
                    "task_status": task_status,
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
        assert received_signals[0]["task_status"] == "completed"
