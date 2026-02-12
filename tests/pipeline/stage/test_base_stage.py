from unittest.mock import MagicMock

from rayforge.pipeline.stage.base import PipelineStage


def test_pipeline_stage_can_be_instantiated():
    """Verify that PipelineStage can be instantiated directly."""
    mock_task_mgr = MagicMock()
    mock_artifact_manager = MagicMock()
    stage = PipelineStage(mock_task_mgr, mock_artifact_manager)
    assert stage._task_manager is mock_task_mgr
    assert stage._artifact_manager is mock_artifact_manager


def test_pipeline_stage_default_reconcile():
    """Verify that default reconcile does nothing."""
    mock_task_mgr = MagicMock()
    mock_artifact_manager = MagicMock()
    mock_doc = MagicMock()
    stage = PipelineStage(mock_task_mgr, mock_artifact_manager)

    stage.reconcile(mock_doc, 1)


def test_pipeline_stage_default_shutdown():
    """Verify that default shutdown does nothing."""
    mock_task_mgr = MagicMock()
    mock_artifact_manager = MagicMock()
    stage = PipelineStage(mock_task_mgr, mock_artifact_manager)

    stage.shutdown()


def test_pipeline_stage_default_is_busy():
    """Verify that default is_busy returns False."""
    mock_task_mgr = MagicMock()
    mock_artifact_manager = MagicMock()
    stage = PipelineStage(mock_task_mgr, mock_artifact_manager)

    assert stage.is_busy is False
