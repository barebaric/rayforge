import pytest  # noqa: F401
import gettext
import asyncio
import logging
from functools import partial
import pytest_asyncio

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
    tm.shutdown()


@pytest_asyncio.fixture(scope="function")
async def context_initializer(tmp_path, task_mgr, monkeypatch):
    """
    A fixture that initializes the application context.
    """
    from rayforge import config
    from rayforge.context import get_context
    from rayforge import context as context_module
    from rayforge.shared import tasker as tasker_module

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
    await context.shutdown()
    context_module._context_instance = None


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
