import pytest
from unittest.mock import MagicMock
from rayforge.core.doc import Doc
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.machine.models.machine import Laser, Machine
from rayforge.pipeline.stage.base import PipelineStage
from rayforge.pipeline.stage.time import TimeEstimatorStage


@pytest.fixture(autouse=True)
def setup_real_config(mocker):
    """Provides a mock machine config for all tests in this file."""
    test_laser = Laser()
    test_laser.max_power = 1000
    test_machine = Machine()
    test_machine.dimensions = (200, 150)
    test_machine.max_cut_speed = 5000
    test_machine.max_travel_speed = 10000
    test_machine.acceleration = 1000
    test_machine.heads.clear()
    test_machine.add_head(test_laser)

    class TestConfig:
        machine = test_machine

    test_config = TestConfig()
    mocker.patch("rayforge.pipeline.stage.time.config.config", test_config)
    return test_config


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


@pytest.fixture
def mock_step():
    """Provides a mock Step object with a UID."""
    step = MagicMock(spec=Step)
    step.uid = "step1"
    step.per_step_transformers_dicts = []
    return step


@pytest.fixture
def mock_workpiece():
    """Provides a mock WorkPiece object with a UID and size."""
    wp = MagicMock(spec=WorkPiece)
    wp.uid = "wp1"
    wp.size = (100.0, 50.0)
    return wp


@pytest.fixture
def stage(mock_task_mgr, mock_artifact_cache):
    """Provides a TimeEstimatorStage instance."""
    return TimeEstimatorStage(mock_task_mgr, mock_artifact_cache)


class TestTimeEstimatorStage:
    def test_instantiation(self, stage, mock_task_mgr, mock_artifact_cache):
        """Test that TimeEstimatorStage can be created."""
        assert isinstance(stage, PipelineStage)
        assert stage._task_manager is mock_task_mgr
        assert stage._artifact_cache is mock_artifact_cache
        assert not stage.is_busy

    def test_is_busy_property(self, stage):
        """Test the is_busy property reflects the active tasks dict."""
        assert not stage.is_busy
        stage._active_tasks["some_key"] = MagicMock()
        assert stage.is_busy
        stage._active_tasks.clear()
        assert not stage.is_busy

    def test_get_estimate_returns_cached_value(
        self, stage, mock_step, mock_workpiece
    ):
        """Test get_estimate retrieves a value from the cache."""
        key = ("step1", "wp1", 100.0, 50.0)
        stage._time_cache[key] = 123.45
        assert stage.get_estimate(mock_step, mock_workpiece) == 123.45

    def test_generate_estimate_skips_if_cached(
        self, stage, mock_step, mock_workpiece
    ):
        """Test that estimate generation is skipped if a value exists."""
        key = ("step1", "wp1", 100.0, 50.0)
        stage._time_cache[key] = 123.45
        stage.generate_estimate(mock_step, mock_workpiece)
        stage._task_manager.run_process.assert_not_called()

    def test_generate_estimate_skips_if_no_handle(
        self, stage, mock_step, mock_workpiece
    ):
        """Test that estimate generation is skipped if no artifact exists."""
        stage._artifact_cache.get_workpiece_handle.return_value = None
        stage.generate_estimate(mock_step, mock_workpiece)
        stage._artifact_cache.get_workpiece_handle.assert_called_with(
            "step1", "wp1"
        )
        stage._task_manager.run_process.assert_not_called()

    def test_generate_estimate_starts_task(
        self, stage, mock_step, mock_workpiece
    ):
        """Test that a new estimation task is started correctly."""
        mock_handle = MagicMock()
        mock_handle.to_dict.return_value = {"shm_name": "test"}
        stage._artifact_cache.get_workpiece_handle.return_value = mock_handle

        stage.generate_estimate(mock_step, mock_workpiece)

        stage._task_manager.run_process.assert_called_once()
        args = stage._task_manager.run_process.call_args[0]
        kwargs = stage._task_manager.run_process.call_args[1]

        assert args[1] == {"shm_name": "test"}  # handle dict
        assert args[2] == (100.0, 50.0)  # size
        assert kwargs["key"] == ("step1", "wp1", 100.0, 50.0)

    def test_on_estimation_complete_success(self, stage):
        """Test the callback logic for a successfully completed task."""
        key = ("step1", "wp1", 100.0, 50.0)
        stage._generation_id_map[key] = 1
        mock_task = MagicMock()
        mock_task.get_status.return_value = "completed"
        mock_task.result.return_value = (99.0, 1)  # time, gen_id
        signal_handler = MagicMock()
        stage.estimation_updated.connect(signal_handler)

        stage._on_estimation_complete(mock_task, key, 1)

        assert stage._time_cache[key] == 99.0
        signal_handler.assert_called_once_with(stage)

    def test_on_estimation_complete_stale(self, stage):
        """Test that a stale result is ignored."""
        key = ("step1", "wp1", 100.0, 50.0)
        stage._generation_id_map[key] = 2  # Current is 2
        mock_task = MagicMock()
        mock_task.get_status.return_value = "completed"
        mock_task.result.return_value = (99.0, 1)  # Result is for 1

        stage._on_estimation_complete(mock_task, key, 1)

        assert key not in stage._time_cache
        # The callback itself is stale, so it returns early
        mock_task.result.assert_not_called()

    def test_invalidate_for_step(self, stage):
        """Test invalidation by step UID."""
        key1 = ("step1", "wp1", 100.0, 50.0)
        key2 = ("step1", "wp2", 100.0, 50.0)
        key3 = ("step2", "wp1", 100.0, 50.0)
        stage._time_cache = {key1: 10, key2: 20, key3: 30}
        mock_task = MagicMock()
        stage._active_tasks[key1] = mock_task

        stage.invalidate_for_step("step1")

        assert key1 not in stage._time_cache
        assert key2 not in stage._time_cache
        assert key3 in stage._time_cache
        stage._task_manager.cancel_task.assert_called_once_with(mock_task.key)

    def test_shutdown(self, stage):
        """Test that shutdown cancels all active tasks."""
        key1 = ("s1", "w1", 1, 1)
        key2 = ("s2", "w2", 2, 2)
        stage._active_tasks[key1] = MagicMock()
        stage._active_tasks[key2] = MagicMock()

        stage.shutdown()

        assert stage._task_manager.cancel_task.call_count == 2
        assert not stage.is_busy
