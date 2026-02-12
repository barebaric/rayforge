import pytest
from unittest.mock import MagicMock

from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.pipeline.artifact import ArtifactManager
from rayforge.pipeline.stage.base import PipelineStage
from rayforge.pipeline.stage.job_stage import JobPipelineStage


@pytest.fixture
def mock_task_mgr():
    """Provides a mock TaskManager."""
    mock_mgr = MagicMock()
    return mock_mgr


@pytest.fixture
def artifact_manager():
    """Provides a real ArtifactManager instance for testing."""
    mock_store = MagicMock(spec=ArtifactStore)
    manager = ArtifactManager(mock_store)
    yield manager
    manager.shutdown()


@pytest.mark.usefixtures("context_initializer")
class TestJobPipelineStage:
    def test_instantiation(
        self, mock_task_mgr, artifact_manager, test_machine_and_config
    ):
        """Test that JobPipelineStage can be created."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)
        assert isinstance(stage, PipelineStage)
        assert stage._task_manager is mock_task_mgr
        assert stage._artifact_manager is artifact_manager
        assert stage._machine is machine

    def test_shutdown(
        self, mock_task_mgr, artifact_manager, test_machine_and_config
    ):
        """Test that shutdown can be called."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)
        stage.shutdown()
