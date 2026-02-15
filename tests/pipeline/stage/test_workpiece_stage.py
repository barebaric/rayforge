import uuid
import pytest
from unittest.mock import MagicMock

from rayforge.core.doc import Doc
from rayforge.pipeline.artifact.key import ArtifactKey
from rayforge.pipeline.artifact.workpiece import WorkPieceArtifact
from rayforge.pipeline.stage.base import PipelineStage
from rayforge.pipeline.stage.workpiece_stage import WorkPiecePipelineStage
from rayforge.shared.util.size import sizes_are_close


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
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
    ):
        """Test that WorkPiecePipelineStage can be created."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr,
            mock_artifact_manager,
            mock_machine,
        )
        assert isinstance(stage, PipelineStage)
        assert stage._task_manager is mock_task_mgr
        assert stage._artifact_manager is mock_artifact_manager
        assert stage._machine is mock_machine

    def test_interface_compliance(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc,
    ):
        """Test that stage implements all required methods."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr,
            mock_artifact_manager,
            mock_machine,
        )
        stage.reconcile(mock_doc, 1)
        stage.shutdown()

    def test_sizes_are_close_match(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
    ):
        """Test size comparison returns True for matching sizes."""
        assert sizes_are_close((10.0, 20.0), (10.0, 20.0)) is True
        assert sizes_are_close((10.000001, 20.0), (10.0, 20.0)) is True

    def test_sizes_are_close_mismatch(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
    ):
        """Test size comparison returns False for mismatched sizes."""
        assert sizes_are_close((10.0, 20.0), (20.0, 10.0)) is False
        assert sizes_are_close((10.0, 20.0), (10.0, 30.0)) is False

    def test_sizes_are_close_none(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
    ):
        """Test size comparison returns False for None values."""
        assert sizes_are_close(None, (10.0, 20.0)) is False
        assert sizes_are_close((10.0, 20.0), None) is False
        assert sizes_are_close(None, None) is False

    def test_invalidate_for_step(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        workpiece_uuid,
    ):
        """Test invalidating workpieces for a step."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr,
            mock_artifact_manager,
            mock_machine,
        )
        step_uid = str(uuid.uuid4())
        wp_key = ArtifactKey.for_workpiece(workpiece_uuid)
        mock_artifact_manager.get_dependents.return_value = [wp_key]

        signal_handler = MagicMock()
        stage.node_state_changed.connect(signal_handler)

        stage.invalidate_for_step(step_uid)

        mock_artifact_manager.get_dependents.assert_called_once()
        mock_artifact_manager.invalidate_for_step.assert_called()
        signal_handler.assert_called_once()

    def test_invalidate_for_workpiece(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        workpiece_uuid,
    ):
        """Test invalidating a specific workpiece."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr,
            mock_artifact_manager,
            mock_machine,
        )
        wp_key = ArtifactKey.for_workpiece(workpiece_uuid)
        mock_artifact_manager.get_all_workpiece_keys.return_value = [wp_key]

        stage.invalidate_for_workpiece(workpiece_uuid)

        mock_artifact_manager.get_all_workpiece_keys.assert_called_once()
        mock_artifact_manager.invalidate_for_workpiece.assert_called()

    def test_get_artifact_missing_handle(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        workpiece_uuid,
    ):
        """Test get_artifact returns None when handle is missing."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr,
            mock_artifact_manager,
            mock_machine,
        )
        mock_artifact_manager.get_workpiece_handle.return_value = None

        result = stage.get_artifact("step_uid", workpiece_uuid, (10, 10), 1)

        assert result is None

    def test_get_artifact_success(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        workpiece_uuid,
    ):
        """Test get_artifact returns artifact when handle is valid."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr,
            mock_artifact_manager,
            mock_machine,
        )
        mock_handle = MagicMock()
        mock_handle.is_scalable = True
        mock_handle.generation_size = (10.0, 10.0)
        mock_artifact = MagicMock(spec=WorkPieceArtifact)
        mock_artifact_manager.get_workpiece_handle.return_value = mock_handle
        mock_artifact_manager.get_artifact.return_value = mock_artifact

        result = stage.get_artifact("step_uid", workpiece_uuid, (10, 10), 1)

        assert result is mock_artifact

    def test_cleanup_entry_running_task(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        workpiece_uuid,
    ):
        """Test cleanup cancels running task."""
        stage = WorkPiecePipelineStage(
            mock_task_mgr,
            mock_artifact_manager,
            mock_machine,
        )
        key = ArtifactKey.for_workpiece(workpiece_uuid)
        mock_task = MagicMock()
        mock_task.is_running.return_value = True
        mock_task_mgr.get_task.return_value = mock_task

        stage._cleanup_entry(key)

        mock_task_mgr.cancel_task.assert_called_once_with(key)
        mock_artifact_manager.invalidate_for_workpiece.assert_called_once_with(
            key
        )
