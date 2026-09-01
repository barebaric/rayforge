from sketcher.core.arrays import (
    Array,
    CircularArray,
    CurveAlongArray,
)


def test_circular_to_dict_format():
    """Locks the serialized format: keys, order-independent shape and
    value types must not change during the array refactoring."""
    array = CircularArray(
        uid="u1",
        guide_circle_id=7,
        members=[(0, [1]), (1, [2, 3])],
        count=8,
        total_angle_deg=270.0,
        rotate_copies=False,
    )
    assert array.to_dict() == {
        "uid": "u1",
        "mode": "circular",
        "guide_circle_id": 7,
        "members": [[0, [1]], [1, [2, 3]]],
        "standalone_pids": [],
        "count": 8,
        "total_angle_deg": 270.0,
        "rotate_copies": False,
    }


def test_curve_along_to_dict_format():
    array = CurveAlongArray(
        uid="u2",
        guide_circle_id=5,
        members=[(0, [1])],
        count=4,
        rotate_copies=False,
        path_entity_id=9,
        align_to_tangent=False,
        offset_to_start=2.0,
        spacing=5.0,
        template_anchor=((3.0, 4.0), 0.5),
    )
    assert array.to_dict() == {
        "uid": "u2",
        "mode": "curve_along",
        "guide_circle_id": 5,
        "members": [[0, [1]]],
        "standalone_pids": [],
        "count": 4,
        "rotate_copies": False,
        "path_entity_id": 9,
        "align_to_tangent": False,
        "offset_to_start": 2.0,
        "spacing": 5.0,
        "template_anchor": [[3.0, 4.0], 0.5],
    }


def test_curve_along_to_dict_omits_absent_anchor_and_fills_defaults():
    array = CurveAlongArray(
        uid="u3",
        guide_circle_id=5,
        members=[],
    )
    assert array.to_dict() == {
        "uid": "u3",
        "mode": "curve_along",
        "guide_circle_id": 5,
        "members": [],
        "standalone_pids": [],
        "count": 6,
        "rotate_copies": True,
        "path_entity_id": -1,
        "align_to_tangent": True,
        "offset_to_start": 0.0,
        "spacing": 0.0,
    }


def test_from_dict_defaults_circular():
    restored = Array.from_dict({"uid": "u4", "guide_circle_id": 7})
    assert isinstance(restored, CircularArray)
    assert restored.mode == "circular"
    assert restored.members == []
    assert restored.count == 6
    assert restored.total_angle_deg == 360.0
    assert restored.rotate_copies is True


def test_from_dict_defaults_curve_along():
    restored = Array.from_dict(
        {"uid": "u7", "mode": "curve_along", "guide_circle_id": 7}
    )
    assert isinstance(restored, CurveAlongArray)
    assert restored.mode == "curve_along"
    assert restored.members == []
    assert restored.count == 6
    assert restored.rotate_copies is True
    assert restored.path_entity_id == -1
    assert restored.align_to_tangent is True
    assert restored.offset_to_start == 0.0
    assert restored.spacing == 0.0
    assert restored.template_anchor is None


def test_from_dict_restores_anchor():
    restored = Array.from_dict(
        {
            "uid": "u6",
            "mode": "curve_along",
            "guide_circle_id": 5,
            "template_anchor": [[1.5, -2.5], 0.25],
        }
    )
    assert isinstance(restored, CurveAlongArray)
    assert restored.template_anchor == ((1.5, -2.5), 0.25)


def test_from_dict_coerces_member_ids_to_int():
    restored = Array.from_dict(
        {
            "uid": "u5",
            "mode": "circular",
            "guide_circle_id": 7,
            "members": [["0", ["1"]]],
            "standalone_pids": [["0", ["2", "3"]]],
        }
    )
    assert restored.members == [(0, [1])]
    assert restored.standalone_pids == {0: [2, 3]}
