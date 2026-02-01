import pytest
from unittest.mock import MagicMock

from rayforge.pipeline.stage.base import PipelineStage


def test_pipeline_stage_is_abstract():
    """Verify that PipelineStage ABC cannot be instantiated directly."""
    mock_task_mgr = MagicMock()
    mock_artifact_manager = MagicMock()
    # The error message from `abc` is "Can't instantiate abstract class ...
    # without an implementation for abstract method 'reconcile'". The regex
    # below is updated to match this specific message format.
    match_str = "without an implementation for abstract method 'reconcile'"
    with pytest.raises(TypeError, match=match_str):
        # The following line is expected to fail at runtime, which is what
        # test asserts. We ignore the static analysis error because this
        # is intended behavior for this test.
        PipelineStage(mock_task_mgr, mock_artifact_manager)  # type: ignore
