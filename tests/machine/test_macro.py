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
