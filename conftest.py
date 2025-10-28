import pytest
import gettext
import asyncio
import logging
from functools import partial
import pytest_asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rayforge.machine.models.machine import Machine


# Set up gettext immediately at the module level.
# This ensures the '_' function is available in builtins before any
# application modules that use it are imported by pytest.
gettext.install("rayforge")

logger = logging.getLogger(__name__)


def pytest_configure(config):
    """
    Configure test-only components.
    This hook is called early in the pytest process after initial imports.
    """
    pass


@pytest_asyncio.fixture(scope="function")
async def task_mgr():
    """
    Provides a test-isolated TaskManager, configured to bridge its main-thread
    callbacks to the asyncio event loop.
    """
    from rayforge.shared.tasker.manager import TaskManager
    from rayforge.worker_init import initialize_worker

    main_loop = asyncio.get_running_loop()

    def asyncio_scheduler(callback, *args, **kwargs):
        main_loop.call_soon_threadsafe(partial(callback, *args, **kwargs))

    tm = TaskManager(
        main_thread_scheduler=asyncio_scheduler,
        worker_initializer=initialize_worker,
    )
    yield tm
    if tm.has_tasks():
        pytest.fail("Task manager still has tasks at end of test.")
    tm.shutdown()


@pytest_asyncio.fixture(scope="function")
async def context_initializer(tmp_path, task_mgr, monkeypatch, mocker):
    """
    A fixture that initializes the application context.
    """
    from rayforge import config
    from rayforge.context import get_context
    from rayforge import context as context_module
    from rayforge.shared import tasker as tasker_module

    # We are patching idle_add at its source to
    # guarantee that ALL calls to it (from Machine, MachineCmd, etc.)
    # will execute synchronously in the test environment, preventing deadlocks
    # caused by an un-run GLib main loop.
    mocker.patch(
        "rayforge.shared.util.glib.idle_add",
        side_effect=lambda func, *args, **kwargs: func(*args, **kwargs),
    )
    mocker.patch(
        "gi.repository.GLib.idle_add",
        side_effect=lambda func, *args, **kwargs: func(*args, **kwargs),
    )

    # 1. Isolate test configuration files
    temp_config_dir = tmp_path / "config"
    temp_machine_dir = temp_config_dir / "machines"
    monkeypatch.setattr(config, "CONFIG_DIR", temp_config_dir)
    monkeypatch.setattr(config, "MACHINE_DIR", temp_machine_dir)

    # 2. Patch the global task_mgr proxy to use our test-isolated instance.
    monkeypatch.setattr(tasker_module.task_mgr, "_instance", task_mgr)

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


@pytest.fixture
def machine(context_initializer) -> "Machine":
    """
    Provides a fresh, test-isolated Machine instance.
    The `context_initializer` fixture is responsible for patching `idle_add`
    so that this machine's signals work correctly.
    """
    from rayforge.machine.models.machine import Machine

    # The patch is already active thanks to context_initializer. We can just
    # create a new machine for the test to use.
    return Machine(context_initializer)


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
