"""
Tests that the core application can operate without GTK.

These tests verify that DocEditor, TaskManager, and RayforgeContext
can be instantiated, import files, save/load projects, and run layout
strategies — all without importing anything from gi.repository.
"""

import json
import threading
import zipfile
from pathlib import Path
from typing import Any, Callable

import pytest

from rayforge.context import get_context
from rayforge import context as context_module
from rayforge.core.doc import Doc
from rayforge.core.matrix import Matrix
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.editor import DocEditor
from rayforge.shared.tasker.manager import TaskManager


def _direct_scheduler(callback: Callable, *args: Any, **kwargs: Any):
    callback(*args, **kwargs)


@pytest.fixture
def task_mgr():
    tm = TaskManager(main_thread_scheduler=_direct_scheduler)
    yield tm
    tm.shutdown()


@pytest.fixture
def headless_context(tmp_path, task_mgr, monkeypatch):
    from rayforge import config
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
    ed = DocEditor(task_manager=task_mgr, context=headless_context)
    yield ed
    ed.cleanup()


class TestHeadlessDocEditor:
    """Core DocEditor operations without GTK."""

    def test_create_editor(self, editor):
        assert editor.doc is not None
        assert isinstance(editor.doc, Doc)

    def test_save_and_load_project(self, editor, tmp_path):
        project_path = tmp_path / "test.ryp"

        saved = editor.file.save_project_to_path(project_path)
        assert saved
        assert project_path.exists()

        data = json.loads(zipfile.ZipFile(project_path).read("project.json"))
        assert "uid" in data

        loaded = editor.file.load_project_from_path(project_path)
        assert loaded

    def test_add_workpiece_and_save(self, editor, tmp_path):
        wp = WorkPiece(name="test.svg")
        wp.matrix = wp.matrix @ Matrix.scale(20, 20)
        editor.doc.active_layer.add_workpiece(wp)

        project_path = tmp_path / "with_workpiece.ryp"
        editor.file.save_project_to_path(project_path)

        data = json.loads(zipfile.ZipFile(project_path).read("project.json"))
        assert "uid" in data
        assert len(data["children"]) > 0

    def test_save_load_roundtrip_preserves_workpiece(self, editor, tmp_path):
        wp = WorkPiece(name="roundtrip.svg")
        wp.matrix = wp.matrix @ Matrix.scale(30, 30)
        editor.doc.active_layer.add_workpiece(wp)

        path = tmp_path / "roundtrip.ryp"
        editor.file.save_project_to_path(path)

        new_doc = Doc.from_dict(
            json.loads(zipfile.ZipFile(path).read("project.json"))
        )
        wps = new_doc.all_workpieces
        assert len(wps) == 1
        assert wps[0].name == "roundtrip.svg"

    def test_load_existing_project(self, editor):
        assets = Path(__file__).parent / "assets"
        project = assets / "contour.ryp"
        if not project.exists():
            pytest.skip("contour.ryp test asset not found")

        loaded = editor.file.load_project_from_path(project)
        assert loaded
        assert len(editor.doc.all_workpieces) > 0


class TestHeadlessLayout:
    """Layout command works without GTK via schedule_on_main_thread."""

    def test_align_items(self, editor):
        wp1 = WorkPiece(name="a.svg")
        wp1.matrix = wp1.matrix @ Matrix.scale(10, 10)
        wp2 = WorkPiece(name="b.svg")
        wp2.matrix = (
            wp2.matrix @ Matrix.translation(50, 50) @ Matrix.scale(10, 10)
        )

        layer = editor.doc.active_layer
        layer.add_workpiece(wp1)
        layer.add_workpiece(wp2)

        items = layer.get_content_items()
        assert len(items) == 2

        editor.layout.align_left(items)

    def test_center_items(self, editor):
        wp = WorkPiece(name="c.svg")
        wp.matrix = wp.matrix @ Matrix.scale(10, 10)
        layer = editor.doc.active_layer
        layer.add_workpiece(wp)

        editor.layout.center_horizontally([wp], surface_width_mm=100.0)


class TestHeadlessTaskManager:
    """TaskManager works with a non-GLib scheduler."""

    def test_schedule_on_main_thread(self, task_mgr):
        results = []
        task_mgr.schedule_on_main_thread(results.append, 42)
        assert results == [42]

    def test_schedule_delayed(self, task_mgr):
        event = threading.Event()
        results = []

        def callback(value):
            results.append(value)
            event.set()

        task_mgr.schedule_delayed_on_main_thread(50, callback, "done")
        assert event.wait(timeout=2)
        assert results == ["done"]
