"""
Pytest configuration for the webui builtin addon tests.

These tests verify the headless web service: the addon's service
entry point, and the FastAPI application it publishes.
"""

import sys
from unittest.mock import MagicMock

import pluggy
import pytest

from rayforge.addon_mgr.addon_manager import AddonManager
from rayforge.config import BUILTIN_ADDONS_DIR
from rayforge.core.hooks import RayforgeSpecs
from rayforge.core.service_registry import service_registry

ADDON_NAME = "webui"
ADDON_ROOT = str(BUILTIN_ADDONS_DIR / "rayforge-addon-webui")

if ADDON_ROOT not in sys.path:
    sys.path.insert(0, ADDON_ROOT)

_plugin_managers: list[pluggy.PluginManager] = []


def _direct_scheduler(callback, *args, **kwargs):
    callback(*args, **kwargs)


def _load_webui(worker_only: bool) -> None:
    plugin_mgr = pluggy.PluginManager("rayforge")
    plugin_mgr.add_hookspecs(RayforgeSpecs)
    _plugin_managers.append(plugin_mgr)
    mgr = AddonManager(
        [BUILTIN_ADDONS_DIR], BUILTIN_ADDONS_DIR, plugin_mgr, MagicMock()
    )
    mgr.set_registries({"service_registry": service_registry})
    assert mgr.load_addon_by_name(ADDON_NAME, worker_only=worker_only)


def _cleanup_webui() -> None:
    service_registry.unregister_all_from_addon(ADDON_NAME)
    while _plugin_managers:
        plugin_mgr = _plugin_managers.pop()
        for _, plugin in plugin_mgr.list_name_plugin():
            plugin_mgr.unregister(plugin)


@pytest.fixture
def task_mgr():
    from rayforge.shared.tasker.manager import TaskManager

    tm = TaskManager(main_thread_scheduler=_direct_scheduler)
    yield tm
    tm.shutdown()


@pytest.fixture
def headless_context(tmp_path, task_mgr, monkeypatch):
    from rayforge import config
    from rayforge import context as context_module
    from rayforge.context import get_context
    from rayforge.shared import tasker

    temp_config_dir = tmp_path / "config"
    temp_machine_dir = temp_config_dir / "machines"
    temp_dialect_dir = temp_config_dir / "dialects"
    monkeypatch.setattr(config, "CONFIG_DIR", temp_config_dir)
    monkeypatch.setattr(config, "MACHINE_DIR", temp_machine_dir)
    monkeypatch.setattr(config, "DIALECT_DIR", temp_dialect_dir)
    monkeypatch.setattr(tasker.task_mgr, "_instance", task_mgr)

    context = get_context()
    context._headless = True
    context.initialize_lite_context(temp_machine_dir)
    yield context

    context_module._context_instance = None


@pytest.fixture
def editor(headless_context, task_mgr):
    from rayforge.doceditor.editor import DocEditor

    ed = DocEditor(task_manager=task_mgr, context=headless_context)
    yield ed
    ed.cleanup()


@pytest.fixture
def webui_registry():
    """
    Loads the webui addon's service entry point the same way the
    headless main process does, and cleans the global registry after
    the test.
    """
    _load_webui(worker_only=True)
    yield service_registry
    _cleanup_webui()


@pytest.fixture
def gui_registry():
    """
    Loads the webui addon the way the GUI main process does (with
    frontend entry points, no services).
    """
    _load_webui(worker_only=False)
    yield service_registry
    _cleanup_webui()


@pytest.fixture
def app(editor, webui_registry):
    factory = webui_registry.get("webui")
    return factory(editor=editor, context=editor.context)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)
