import pytest
from unittest.mock import MagicMock

from rayforge.core.doc import Doc
from rayforge.pipeline.stage.base import PipelineStage
from rayforge.pipeline.stage.job import JobGeneratorStage


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


class TestJobGeneratorStage:
    def test_instantiation(self, mock_task_mgr, mock_artifact_cache):
        """Test that JobGeneratorStage can be created."""
        stage = JobGeneratorStage(mock_task_mgr, mock_artifact_cache)
        assert isinstance(stage, PipelineStage)
        assert stage._task_manager is mock_task_mgr
        assert stage._artifact_cache is mock_artifact_cache

    def test_interface_compliance(
        self, mock_task_mgr, mock_artifact_cache, mock_doc
    ):
        """Test that the stage implements all required abstract methods."""
        stage = JobGeneratorStage(mock_task_mgr, mock_artifact_cache)
        # These should not raise NotImplementedError
        stage.reconcile(mock_doc)
        stage.shutdown()
