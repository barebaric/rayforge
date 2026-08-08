"""
Tests for the Tool dataclass and YAML (de)serialization.
"""

import pytest
from tool_library.tool import (
    CATEGORY_BY_NAME,
    CATEGORY_NAMES,
    TOOL_MATERIAL_BY_NAME,
    TOOL_MATERIAL_NAMES,
    Tool,
    category_to_name,
    tool_material_to_name,
)


def test_create_default_has_endmill_carbide():
    tool = Tool.create_default()
    assert tool.category == CATEGORY_BY_NAME["END_MILL"]
    assert tool.tool_material == TOOL_MATERIAL_BY_NAME["CARBIDE"]
    assert tool.diameter() == pytest.approx(6.0)
    assert tool.name


@pytest.mark.parametrize("cat_name", CATEGORY_NAMES)
def test_category_round_trip(cat_name):
    tool = Tool.create_default()
    tool = Tool(
        uid=tool.uid,
        name=tool.name,
        max_rpm=tool.max_rpm,
        label=tool.label,
        category=CATEGORY_BY_NAME[cat_name],
        tool_material=tool.tool_material,
        stickout=tool.stickout,
        coating=tool.coating,
        model=tool.model,
    )
    data = tool.to_dict()
    assert data["category"] == cat_name
    restored = Tool.from_dict(data)
    assert restored.category == CATEGORY_BY_NAME[cat_name]
    assert category_to_name(restored.category) == cat_name


@pytest.mark.parametrize("mat_name", TOOL_MATERIAL_NAMES)
def test_tool_material_round_trip(mat_name):
    tool = Tool.create_default()
    tool = Tool(
        uid=tool.uid,
        name=tool.name,
        max_rpm=tool.max_rpm,
        label=tool.label,
        category=tool.category,
        tool_material=TOOL_MATERIAL_BY_NAME[mat_name],
        stickout=tool.stickout,
        coating=tool.coating,
        model=tool.model,
    )
    restored = Tool.from_dict(tool.to_dict())
    assert restored.tool_material == TOOL_MATERIAL_BY_NAME[mat_name]
    assert tool_material_to_name(restored.tool_material) == mat_name


def test_full_round_trip_preserves_model_params():
    tool = Tool.create_default("My EM")
    tool = Tool(
        uid="abc-123",
        name="My EM",
        max_rpm=18000.0,
        label="6mm",
        category=CATEGORY_BY_NAME["BALL_NOSE"],
        tool_material=TOOL_MATERIAL_BY_NAME["HSS"],
        stickout=20.0,
        coating="AlTiN",
        model=tool.model,
    )
    data = tool.to_dict()
    restored = Tool.from_dict(data)

    assert restored.uid == "abc-123"
    assert restored.name == "My EM"
    assert restored.max_rpm == pytest.approx(18000.0)
    assert restored.label == "6mm"
    assert restored.category == CATEGORY_BY_NAME["BALL_NOSE"]
    assert restored.tool_material == TOOL_MATERIAL_BY_NAME["HSS"]
    assert restored.stickout == pytest.approx(20.0)
    assert restored.coating == "AlTiN"
    assert restored.diameter() == pytest.approx(6.0)
    assert restored.model.get_parameters() == tool.model.get_parameters()


def test_round_trip_with_no_coating():
    tool = Tool.create_default()
    tool = Tool(
        uid=tool.uid,
        name=tool.name,
        max_rpm=tool.max_rpm,
        label=tool.label,
        category=tool.category,
        tool_material=tool.tool_material,
        stickout=tool.stickout,
        coating=None,
        model=tool.model,
    )
    assert tool.to_dict()["coating"] is None
    assert Tool.from_dict(tool.to_dict()).coating is None


def test_from_dict_assigns_uid_when_missing():
    data = Tool.create_default().to_dict()
    del data["uid"]
    restored = Tool.from_dict(data)
    assert restored.uid


def test_from_dict_unknown_category_falls_back():
    data = Tool.create_default().to_dict()
    data["category"] = "Nonexistent"
    restored = Tool.from_dict(data)
    assert restored.category == CATEGORY_BY_NAME["END_MILL"]
