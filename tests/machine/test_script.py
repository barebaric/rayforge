from rayforge.machine.models.script import Script


class TestScript:
    def test_instantiation(self):
        """Test basic object creation with default values."""
        script = Script(name="My Script")
        assert script.name == "My Script"
        assert script.code == []
        assert script.enabled is True
        assert isinstance(script.uid, str)

    def test_instantiation_with_values(self):
        """Test object creation with all values specified."""
        code_lines = ["G21", "G90"]
        script = Script(name="Test", code=code_lines, enabled=False)
        assert script.name == "Test"
        assert script.code == code_lines
        assert script.enabled is False

    def test_to_dict_serialization(self):
        """Verify that the to_dict method produces the correct structure."""
        code_lines = ["M5", "G0 X0 Y0"]
        script = Script(name="Go Home", code=code_lines, enabled=True)
        data = script.to_dict()
        assert data["name"] == "Go Home"
        assert data["code"] == ["M5", "G0 X0 Y0"]
        assert data["enabled"] is True
        assert "uid" in data

    def test_from_dict_deserialization(self):
        """Verify that from_dict correctly reconstructs an object."""
        data = {
            "uid": "test-uid-123",
            "name": "Test Script",
            "code": ["( A comment )"],
            "enabled": False,
        }
        script = Script.from_dict(data)
        assert isinstance(script, Script)
        assert script.uid == "test-uid-123"
        assert script.name == "Test Script"
        assert script.code == ["( A comment )"]
        assert script.enabled is False

    def test_from_dict_with_defaults(self):
        """Test from_dict when optional fields are missing."""
        data = {"name": "Minimal Script"}
        script = Script.from_dict(data)
        assert script.name == "Minimal Script"
        assert script.code == []
        assert script.enabled is True
        assert isinstance(script.uid, str)
