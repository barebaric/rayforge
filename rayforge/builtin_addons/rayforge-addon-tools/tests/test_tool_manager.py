"""
Tests for the ToolManager: CRUD, YAML persistence, changed signal.
"""

import pytest
from tool_library.manager import ToolManager
from tool_library.tool import CATEGORY_BY_NAME, Tool


@pytest.fixture
def mgr(tmp_path):
    return ToolManager(tmp_path)


def test_starts_empty(mgr):
    assert mgr.get_all() == []
    assert mgr.get("nope") is None


def test_save_persists_and_indexes(mgr, tmp_path):
    tool = Tool.create_default("First")
    mgr.save(tool)

    assert mgr.get(tool.uid) is tool
    assert [t.name for t in mgr.get_all()] == ["First"]
    assert (tmp_path / f"{tool.uid}.yaml").exists()


def test_load_reads_existing_files(tmp_path):
    tool = Tool.create_default("Persisted")
    mgr = ToolManager(tmp_path)
    mgr.save(tool)

    fresh = ToolManager(tmp_path)
    loaded = fresh.get(tool.uid)
    assert loaded is not None
    assert loaded.name == "Persisted"
    assert loaded.diameter() == pytest.approx(6.0)


def test_update_existing_tool(mgr):
    tool = Tool.create_default("Rename Me")
    mgr.save(tool)

    updated = Tool(
        uid=tool.uid,
        name="Renamed",
        max_rpm=tool.max_rpm,
        label=tool.label,
        category=CATEGORY_BY_NAME["CHAMFER"],
        tool_material=tool.tool_material,
        stickout=tool.stickout,
        coating=tool.coating,
        model=tool.model,
    )
    mgr.save(updated)

    assert mgr.get(tool.uid).name == "Renamed"
    assert mgr.get(tool.uid).category == CATEGORY_BY_NAME["CHAMFER"]
    assert len(mgr.get_all()) == 1


def test_delete_removes_from_memory_and_disk(mgr, tmp_path):
    tool = Tool.create_default("ToDelete")
    mgr.save(tool)
    assert (tmp_path / f"{tool.uid}.yaml").exists()

    assert mgr.delete(tool.uid) is True
    assert mgr.get(tool.uid) is None
    assert not (tmp_path / f"{tool.uid}.yaml").exists()
    assert mgr.delete(tool.uid) is False


def test_get_all_sorted_by_name(mgr):
    b = Tool.create_default("Banana")
    a = Tool.create_default("Apple")
    mgr.save(b)
    mgr.save(a)
    assert [t.name for t in mgr.get_all()] == ["Apple", "Banana"]


def test_changed_signal_emitted_on_mutations(mgr):
    fired = []

    def handler(_sender):
        fired.append(True)

    mgr.changed.connect(handler)

    tool = Tool.create_default()
    mgr.save(tool)
    assert len(fired) == 1

    mgr.delete(tool.uid)
    assert len(fired) == 2
