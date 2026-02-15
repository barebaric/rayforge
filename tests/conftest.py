import sys
import multiprocessing

import pytest
import gettext
import asyncio
import logging
import gc
from functools import partial
import pytest_asyncio
from typing import TYPE_CHECKING, AsyncGenerator
from unittest.mock import patch, MagicMock
from gi.repository import GLib
from rayforge.worker_init import initialize_worker
from rayforge import context as rayforge_context
from rayforge.shared.tasker.progress import ProgressContext


class PyvipsLogFilter(logging.Filter):
    """Filter to suppress spammy pyvips debug messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith("pyvips"):
            return record.levelno >= logging.WARNING
        if record.name == "VipsObject":
            return record.levelno >= logging.WARNING
        return True


def pytest_configure(config):
    """
    Configure test-only components.
    This hook is called early in the pytest process after initial imports.
    """
    if sys.platform.startswith("linux"):
        multiprocessing.set_start_method("spawn", force=True)

    logging.getLogger("pyvips").setLevel(logging.WARNING)
    logging.getLogger("pyvips.vobject").setLevel(logging.WARNING)

    logging.getLogger().addFilter(PyvipsLogFilter())


if TYPE_CHECKING:
    from rayforge.machine.models.machine import Machine


# Set up gettext immediately at the module level.
# This ensures the '_' function is available in builtins before any
# application modules that use it are imported by pytest.
gettext.install("rayforge")

logger = logging.getLogger(__name__)


def _test_worker_initializer():
    """
    A top-level, picklable worker initializer for the test environment.
    This patches GLib.idle_add inside the worker process to prevent deadlocks.
    """
    fail_msg = (
        "A call to GLib.idle_add() was detected within a worker during tests. "
        "This is forbidden as it can cause deadlocks. All main-thread "
        "callbacks should be routed through the test-aware TaskManager "
        "scheduler."
    )

    # This patch runs *inside the new worker process*
    # after it starts, but before the application's real initializer runs.
    with patch(
        "gi.repository.GLib.idle_add",
        side_effect=lambda *args, **kwargs: pytest.fail(fail_msg),
    ):
        with patch(
            "rayforge.shared.util.glib.idle_add",
            side_effect=lambda *args, **kwargs: pytest.fail(
                "GLib.idle_add called from within a worker process."
            ),
        ):
            # Now call the application's real initializer.
            initialize_worker()


@pytest.fixture(scope="function", autouse=True)
def block_glib_event_loop(request):
    """
    Session-wide, autouse fixture to patch GLib event loop functions
    BEFORE any application modules are imported by tests. This is critical
    to prevent tests from deadlocking or interacting with a non-existent
    GTK main loop.

    This uses unittest.mock.patch directly because the standard `mocker`
    fixture is function-scoped and cannot be used in a session-scoped fixture.

    This patch is SKIPPED for tests marked with 'ui'.
    """
    # Conditionally apply the patch. If the 'ui' marker is present on the
    # test, we do not apply the patch, as UI tests require GLib.idle_add.
    if "ui" in request.keywords:
        yield
        return

    fail_msg = (
        "A call to GLib.idle_add() was detected during backend tests. "
        "This is forbidden as it can cause deadlocks. All main-thread "
        "callbacks should be routed through the test-aware TaskManager "
        "scheduler."
    )

    # Using `patch` as a context manager. It will be active during the `yield`.
    with patch(
        "gi.repository.GLib.idle_add",
        side_effect=lambda *args, **kwargs: pytest.fail(fail_msg),
    ):
        with patch(
            "rayforge.shared.util.glib.idle_add",
            side_effect=lambda *args, **kwargs: pytest.fail(fail_msg),
        ):
            # The test session runs here, with the patches active.
            yield


@pytest.fixture(autouse=True)
def clean_context_singleton():
    """
    Ensures the RayforgeContext singleton is completely shut down and reset
    after each test.

    This fixture is synchronous to be compatible with both sync and async
    tests. It intelligently runs the async shutdown() method using the
    appropriate mechanism.
    """
    # Yield control to the test to execute
    yield

    # Teardown: This code runs after the test has finished
    instance = rayforge_context._context_instance
    if not instance:
        # If no context was created during the test, there's nothing to do.
        return

    # Now, we need to run an async function (instance.shutdown) from a
    # synchronous context (the fixture teardown). We must handle two cases:
    # 1. The test was async: An event loop is already running.
    # 2. The test was sync: No event loop is running.

    try:
        loop = asyncio.get_running_loop()
        # Case 1: An event loop is running (the test was async).
        # We run our async shutdown within this existing loop.
        if not loop.is_closed():
            loop.run_until_complete(instance.shutdown())
        else:
            # The loop was closed by the test, so we must start a new one.
            asyncio.run(instance.shutdown())
    except RuntimeError:
        # Case 2: No event loop is running (the test was sync).
        # We can safely use asyncio.run() to create a new, temporary
        # event loop just for our shutdown task.
        asyncio.run(instance.shutdown())

    # Finally, reset the global singleton variable so the next test
    # starts with a completely fresh context.
    rayforge_context._context_instance = None
    gc.collect()


@pytest_asyncio.fixture(scope="function")
async def task_mgr():
    """
    Provides a test-isolated TaskManager for ASYNC tests, configured to
    bridge its callbacks to the asyncio event loop.
    """
    from rayforge.shared.tasker.manager import TaskManager

    main_loop = asyncio.get_running_loop()

    def asyncio_scheduler(callback, *args, **kwargs):
        if not main_loop.is_closed():
            main_loop.call_soon_threadsafe(partial(callback, *args, **kwargs))

    tm = TaskManager(
        main_thread_scheduler=asyncio_scheduler,
        worker_initializer=_test_worker_initializer,
    )

    yield tm
    if tm.has_tasks():
        pytest.fail("Task manager still has tasks at end of test.")
    tm.shutdown()


@pytest_asyncio.fixture(scope="function")
async def context_initializer(tmp_path, task_mgr, monkeypatch):
    """
    A fixture that initializes the application context.
    """
    from rayforge import config
    from rayforge.context import get_context
    from rayforge import context as context_module
    from rayforge.shared import tasker

    # 1. Isolate test configuration files
    temp_config_dir = tmp_path / "config"
    temp_dialect_dir = temp_config_dir / "dialects"
    temp_machine_dir = temp_config_dir / "machines"
    monkeypatch.setattr(config, "CONFIG_DIR", temp_config_dir)
    monkeypatch.setattr(config, "DIALECT_DIR", temp_dialect_dir)
    monkeypatch.setattr(config, "MACHINE_DIR", temp_machine_dir)

    # 2. Patch the global task_mgr proxy to use our test-isolated instance.
    monkeypatch.setattr(tasker.task_mgr, "_instance", task_mgr)

    # 3. Get the context and run the full initialization
    context = get_context()
    context.initialize_full_context()
    yield context

    # 4. Teardown: shutdown and fully reset the context singleton and globals
    # This is crucial for test isolation.
    if task_mgr.has_tasks():
        pytest.fail("Task manager still has tasks at end of test.")

    await context.shutdown()
    context_module._context_instance = None

    # 5. Clean up config file to prevent errors in subsequent tests
    config_file = temp_config_dir / "config.yaml"
    if config_file.exists():
        config_file.unlink()


@pytest_asyncio.fixture
async def machine(context_initializer) -> AsyncGenerator["Machine", None]:
    """
    Provides a fresh, test-isolated Machine instance with automatic async
    teardown.
    """
    from rayforge.machine.models.machine import Machine

    m = Machine(context_initializer)
    context_initializer.machine_mgr.add_machine(m)
    yield m
    # Proper async teardown is handled here by the fixture runner
    await m.shutdown()


@pytest.fixture
def test_machine_and_config(context_initializer):
    """
    Sets up a well-defined test machine and sets it as the active config
    using the real application mechanisms. This replaces manual mocking.
    """
    from rayforge.machine.models.machine import Laser, Machine

    context = context_initializer
    test_laser = Laser()
    test_laser.max_power = 1000
    test_machine = Machine(context_initializer)
    test_machine.dimensions = (200, 150)
    test_machine.max_cut_speed = 5000
    test_machine.max_travel_speed = 10000
    test_machine.heads.clear()
    test_machine.add_head(test_laser)

    # Use the real managers from the context to set up the state
    context.machine_mgr.machines.clear()
    context.machine_mgr.add_machine(test_machine)
    context.config.set_machine(test_machine)

    yield test_machine, context.config


@pytest.fixture(scope="function")
def ui_task_mgr():
    """
    Provides a test-isolated TaskManager for SYNC UI tests. It uses
    GLib.idle_add to safely communicate with the main GTK thread.
    """
    from rayforge.shared.tasker.manager import TaskManager

    tm = TaskManager(main_thread_scheduler=GLib.idle_add)
    yield tm
    if tm.has_tasks():
        logger.warning(
            "Task manager still has tasks at end of test. Shutting down."
        )
    tm.shutdown()


@pytest.fixture(scope="function")
def ui_context_initializer(tmp_path, monkeypatch, ui_task_mgr):
    """
    A SYNCHRONOUS context initializer for UI tests. It uses the GLib-based
    `ui_task_mgr`.
    """
    from rayforge import config
    from rayforge import context as context_module
    from rayforge.context import get_context
    from rayforge.shared import tasker

    temp_config_dir = tmp_path / "config"
    temp_dialect_dir = temp_config_dir / "dialects"
    temp_machine_dir = temp_config_dir / "machines"
    monkeypatch.setattr(config, "CONFIG_DIR", temp_config_dir)
    monkeypatch.setattr(config, "DIALECT_DIR", temp_dialect_dir)
    monkeypatch.setattr(config, "MACHINE_DIR", temp_machine_dir)
    monkeypatch.setattr(tasker, "task_mgr", ui_task_mgr)

    context = get_context()
    context.initialize_full_context()
    yield context

    asyncio.run(context.shutdown())
    context_module._context_instance = None


@pytest.fixture
def mock_machine():
    """
    Provides a mock Machine instance for tests that don't need a full context.
    """
    from rayforge.machine.models.machine import Machine

    mock_machine = MagicMock(spec=Machine)
    mock_machine.dimensions = (200, 150)
    mock_machine.max_cut_speed = 5000
    mock_machine.max_travel_speed = 10000
    return mock_machine


@pytest.fixture
def mock_artifact_store():
    """
    Provides a mock ArtifactStore instance for tests.
    """
    from rayforge.pipeline.artifact.store import ArtifactStore

    mock_store = MagicMock(spec=ArtifactStore)
    return mock_store


@pytest_asyncio.fixture
async def doc_editor(task_mgr, context_initializer):
    """
    Provides a DocEditor instance with proper cleanup.
    """
    from rayforge.doceditor.editor import DocEditor

    editor = DocEditor(task_manager=task_mgr, context=context_initializer)
    yield editor
    editor.cleanup()


@pytest.fixture
def mock_progress_context():
    """
    Provides a mock ProgressContext for testing compute functions.

    This mock tracks all progress and message calls for verification.
    Allows testing cancellation by setting is_cancelled = True.
    """

    class _SimpleMockProgressContext:
        def __init__(self):
            self.progress_calls: list[tuple[float, str]] = []
            self.message_calls: list[str] = []
            self._is_cancelled = False
            self._total = 1.0
            self._sub_contexts: list["_SimpleMockProgressContext"] = []

        def is_cancelled(self) -> bool:
            return self._is_cancelled

        def set_progress(self, progress: float) -> None:
            normalized = (
                progress / self._total if self._total > 0 else progress
            )
            self.progress_calls.append((normalized, ""))

        def set_message(self, message: str) -> None:
            self.message_calls.append(message)

        def set_total(self, total: float) -> None:
            if total <= 0:
                self._total = 1.0
            else:
                self._total = float(total)

        def sub_context(
            self,
            base_progress: float,
            progress_range: float,
            total: float = 1.0,
        ) -> "_SimpleMockProgressContext":
            sub_ctx = _SimpleMockProgressContext()
            sub_ctx._total = total
            self._sub_contexts.append(sub_ctx)
            return sub_ctx

        def flush(self) -> None:
            pass

    return _SimpleMockProgressContext()


class MockProgressContext(ProgressContext):
    """Unified mock ProgressContext for testing.

    This class provides a comprehensive mock implementation that tracks all
    progress-related calls for verification in tests. It extends
    ProgressContext and provides cancellation control.

    Attributes:
        progress_calls: List of normalized progress values reported.
        message_calls: List of messages set via set_message().
        total_calls: List of total values set via set_total().
        flush_calls: Number of times flush() was called.
        sub_contexts: List of sub-contexts created via sub_context().
        _cancelled: Internal cancellation state.
    """

    def __init__(self, cancelled: bool = False):
        """Initialize the mock progress context.

        Args:
            cancelled: Initial cancellation state.
        """
        self._inner = _MockProgressContextImpl(cancelled)
        super().__init__(base_progress=0.0, progress_range=1.0, total=1.0)
        # Clear initialization call from parent constructor
        self._inner.total_calls.clear()

    def is_cancelled(self) -> bool:
        """Check if the operation has been cancelled."""
        return self._inner.is_cancelled()

    def set_progress(self, progress: float) -> None:
        """Set progress as an absolute value."""
        self._inner.set_progress(progress)

    def set_message(self, message: str) -> None:
        """Set a descriptive status message."""
        self._inner.set_message(message)

    def set_total(self, total: float) -> None:
        """Set the total value for progress normalization."""
        self._inner.set_total(total)

    def sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float,
    ) -> "MockProgressContext":
        """Create a sub-context for hierarchical progress reporting."""
        inner_sub = self._inner.sub_context(
            base_progress, progress_range, total
        )
        wrapper = MockProgressContext(
            cancelled=inner_sub._progress_context._cancelled
        )
        wrapper._inner = inner_sub
        return wrapper

    def flush(self) -> None:
        """Immediately send any pending updates."""
        self._inner.flush()

    def set_cancelled(self, cancelled: bool) -> None:
        """Helper to change cancellation state during tests."""
        self._inner.set_cancelled(cancelled)

    def _report_normalized_progress(self, progress: float) -> None:
        """Report a normalized progress value."""
        pass

    def _create_sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float,
    ) -> "ProgressContext":
        """Factory method for creating sub-contexts."""
        inner_sub = self._inner.sub_context(
            base_progress, progress_range, total
        )
        wrapper = MockProgressContext(
            cancelled=inner_sub._progress_context._cancelled
        )
        wrapper._inner = inner_sub
        return wrapper

    @property
    def progress_calls(self) -> list[float]:
        """Get list of normalized progress values."""
        return self._inner.progress_calls

    @property
    def message_calls(self) -> list[str]:
        """Get list of messages."""
        return self._inner.message_calls

    @property
    def total_calls(self) -> list[float]:
        """Get list of total values."""
        return self._inner.total_calls

    @property
    def flush_calls(self) -> int:
        """Get number of flush calls."""
        return self._inner.flush_calls

    @property
    def sub_contexts(self) -> list["_MockProgressContextImpl"]:
        """Get list of sub-contexts."""
        return self._inner.sub_contexts


class _MockProgressContextImpl:
    """Internal implementation of MockProgressContext.

    This class extends ProgressContext and provides the actual tracking
    functionality. The outer MockProgressContext wraps this to provide
    a cleaner interface and return wrapped sub-contexts.
    """

    def __init__(self, cancelled: bool = False):
        """Initialize the mock progress context implementation.

        Args:
            cancelled: Initial cancellation state.
        """
        self._progress_context = _InnerMockProgressContext(cancelled=cancelled)
        self.progress_calls: list[float] = []
        self.message_calls: list[str] = []
        self.total_calls: list[float] = []
        self.flush_calls: int = 0
        self.sub_contexts: list["_MockProgressContextImpl"] = []

    def is_cancelled(self) -> bool:
        """Check if the operation has been cancelled."""
        return self._progress_context.is_cancelled()

    def set_progress(self, progress: float) -> None:
        """Set progress as an absolute value."""
        self._progress_context.set_progress(progress)

    def set_message(self, message: str) -> None:
        """Set a descriptive status message."""
        self.message_calls.append(message)
        self._progress_context.set_message(message)

    def set_total(self, total: float) -> None:
        """Set the total value for progress normalization."""
        self.total_calls.append(total)
        self._progress_context.set_total(total)

    def sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float,
    ) -> "_MockProgressContextImpl":
        """Create a sub-context for hierarchical progress reporting."""
        inner_sub = self._progress_context.sub_context(
            base_progress, progress_range, total
        )
        wrapper = _MockProgressContextImpl(cancelled=inner_sub._cancelled)
        wrapper._progress_context = inner_sub
        self.sub_contexts.append(wrapper)
        return wrapper

    def flush(self) -> None:
        """Immediately send any pending updates."""
        self.flush_calls += 1
        self._progress_context.flush()

    def set_cancelled(self, cancelled: bool) -> None:
        """Helper to change cancellation state during tests."""
        self._progress_context.set_cancelled(cancelled)


class _InnerMockProgressContext:
    """Inner class that extends ProgressContext.

    This class implements the ProgressContext abstract methods and
    delegates to the wrapper for tracking.
    """

    def __init__(self, cancelled: bool = False):
        """Initialize the inner mock progress context.

        Args:
            cancelled: Initial cancellation state.
        """
        self._cancelled = cancelled
        self._wrapper: _MockProgressContextImpl | None = None
        self._base = 0.0
        self._range = 1.0
        self._total = 1.0

    def set_wrapper(self, wrapper: "_MockProgressContextImpl") -> None:
        """Set the wrapper for tracking calls."""
        self._wrapper = wrapper

    def is_cancelled(self) -> bool:
        """Check if the operation has been cancelled."""
        return self._cancelled

    def set_cancelled(self, cancelled: bool) -> None:
        """Helper to change cancellation state during tests."""
        self._cancelled = cancelled

    def set_progress(self, progress: float) -> None:
        """Set progress as an absolute value."""
        normalized = progress / self._total if self._total > 0 else progress
        if self._wrapper:
            self._wrapper.progress_calls.append(normalized)

    def set_message(self, message: str) -> None:
        """Set a descriptive status message."""
        pass

    def flush(self) -> None:
        """Immediately send any pending updates."""
        pass

    def set_total(self, total: float) -> None:
        """Set the total value for progress normalization."""
        if total <= 0:
            self._total = 1.0
        else:
            self._total = float(total)

    def sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float,
    ) -> "_InnerMockProgressContext":
        """Create a sub-context for hierarchical progress reporting."""
        return self._create_sub_context(base_progress, progress_range, total)

    def _report_normalized_progress(self, progress: float) -> None:
        """Report a normalized progress value."""
        pass

    def _create_sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float,
    ) -> "_InnerMockProgressContext":
        """Factory method for creating sub-contexts."""
        sub_ctx = _InnerMockProgressContext(cancelled=self._cancelled)
        sub_ctx._base = base_progress
        sub_ctx._range = progress_range
        sub_ctx._total = total
        return sub_ctx


@pytest.fixture
def adopting_mock_proxy():
    """
    Creates a mock ExecutionContextProxy that adopts artifacts when
    send_event_and_wait is called.

    On Windows, shared memory is destroyed when all handles are closed.
    When tests run runner functions in-process (not in a real subprocess),
    the runner calls forget() after send_event_and_wait returns, which
    closes the only handle. This mock simulates the main process adopting
    the artifact before the runner forgets it.
    """
    from unittest.mock import MagicMock
    from rayforge.pipeline.artifact import create_handle_from_dict
    from rayforge.context import get_context

    artifact_store = get_context().artifact_store

    def mock_send_event_and_wait(event_name, data, logger=None):
        handle_dict = data.get("handle_dict")
        if handle_dict:
            handle = create_handle_from_dict(handle_dict)
            artifact_store.adopt(handle)
        return True

    proxy = MagicMock()
    proxy.send_event_and_wait.side_effect = mock_send_event_and_wait
    return proxy


@pytest.fixture
def mock_progress_context_v2():
    """Provides the unified MockProgressContext for testing.

    This fixture returns a new MockProgressContext instance that tracks
    all progress-related calls for verification in tests.
    """
    ctx = MockProgressContext()
    ctx._inner._progress_context.set_wrapper(ctx._inner)
    for sub in ctx._inner.sub_contexts:
        sub._progress_context.set_wrapper(sub)
    return ctx
