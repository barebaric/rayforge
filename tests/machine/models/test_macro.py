from rayforge.machine.models.macro import Macro


def test_instantiation():
    """Test basic object creation with default values."""
    macro = Macro(name="My Macro")
    assert macro.name == "My Macro"
    assert macro.code == []
    assert macro.enabled is True
    assert isinstance(macro.uid, str)


def test_instantiation_with_values():
    """Test object creation with all values specified."""
    code_lines = ["G21", "G90"]
    macro = Macro(name="Test", code=code_lines, enabled=False)
    assert macro.name == "Test"
    assert macro.code == code_lines
    assert macro.enabled is False


def test_to_dict_serialization():
    """Verify that the to_dict method produces the correct structure."""
    code_lines = ["M5", "G0 X0 Y0"]
    macro = Macro(name="Go Home", code=code_lines, enabled=True)
    data = macro.to_dict()
    assert data["name"] == "Go Home"
    assert data["code"] == ["M5", "G0 X0 Y0"]
    assert data["enabled"] is True
    assert "uid" in data


def test_from_dict_deserialization():
    """Verify that from_dict correctly reconstructs an object."""
    data = {
        "uid": "test-uid-123",
        "name": "Test Macro",
        "code": ["( A comment )"],
        "enabled": False,
    }
    macro = Macro.from_dict(data)
    assert isinstance(macro, Macro)
    assert macro.uid == "test-uid-123"
    assert macro.name == "Test Macro"
    assert macro.code == ["( A comment )"]
    assert macro.enabled is False


def test_from_dict_with_defaults():
    """Test from_dict when optional fields are missing."""
    data = {"name": "Minimal Macro"}
    macro = Macro.from_dict(data)
    assert macro.name == "Minimal Macro"
    assert macro.code == []
    assert macro.enabled is True
    assert isinstance(macro.uid, str)


def test_forward_compatibility_unknown_fields():
    """
    Test that unknown fields are preserved during serialization and
    deserialization for forward compatibility.
    """
    original_macro = Macro(name="Test Macro", code=["G21"])

    data = original_macro.to_dict()

    # Simulate future version with additional fields
    future_fields = {
        "future_feature_enabled": True,
        "future_setting": "some_value",
        "future_list": [1, 2, 3],
    }
    data.update(future_fields)

    # Deserialize with unknown fields
    deserialized = Macro.from_dict(data)

    # Unknown fields should be in extra
    assert deserialized.extra == future_fields

    # Known fields should be preserved
    assert deserialized.name == "Test Macro"
    assert deserialized.code == ["G21"]

    # Reserialize should include unknown fields
    reserialized = deserialized.to_dict()
    assert reserialized["future_feature_enabled"] is True
    assert reserialized["future_setting"] == "some_value"
    assert reserialized["future_list"] == [1, 2, 3]


def test_forward_compatibility_roundtrip():
    """
    Test that a complete round-trip through to_dict and from_dict
    preserves all data including unknown fields.
    """
    original_data = {
        "uid": "test-uid-456",
        "name": "Roundtrip Test",
        "code": ["G21", "G90"],
        "enabled": False,
        "future_bool": False,
        "future_string": "test",
        "future_number": 42,
    }

    macro = Macro.from_dict(original_data)
    result = macro.to_dict()

    # All original fields should be preserved
    for key, value in original_data.items():
        assert result[key] == value
