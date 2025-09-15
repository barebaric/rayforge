import pytest
from rayforge.shared.util.template import TemplateFormatter
from rayforge.machine.models.script import Script


class TestTemplateFormatter:
    @pytest.fixture
    def context_and_machine(self):
        """
        Provides a nested object structure and a mock machine for testing.
        """

        class MockMacro(Script):
            pass

        class MockMachine:
            name = "MyLaser"
            dimensions = (200, 150)
            macros = {
                "macro1_uid": MockMacro(
                    name="First Macro", code=["G0 X10 Y10"]
                ),
                "macro2_uid": MockMacro(name="Second Macro", code=["G0 Z5"]),
                "disabled_uid": MockMacro(
                    name="Disabled Macro", code=["M5"], enabled=False
                ),
                "circular2_uid": MockMacro(
                    name="Circular2", code=["@include(Circular1)"]
                ),
                "wrapper_uid": MockMacro(
                    name="Wrapper",
                    code=["G90", "@include(First Macro)", "G91"],
                ),
            }

        MockMachine.macros["circular1_uid"] = MockMacro(
            name="Circular1", code=["@include(Circular2)"]
        )

        class Job:
            name = "Test Job"

        class Context:
            machine = MockMachine()
            job = Job()
            top_level_value = 42

        return Context(), MockMachine()

    def test_simple_replacement(self, context_and_machine):
        """Test replacement of a top-level attribute."""
        context, machine = context_and_machine
        formatter = TemplateFormatter(machine, context)
        result = formatter.format_string("Value is {top_level_value}")
        assert result == "Value is 42"

    def test_nested_replacement(self, context_and_machine):
        """Test replacement of a nested attribute."""
        context, machine = context_and_machine
        formatter = TemplateFormatter(machine, context)
        result = formatter.format_string(
            "Machine: {machine.name}, Job: {job.name}"
        )
        assert result == "Machine: MyLaser, Job: Test Job"

    def test_unresolved_path(self, context_and_machine):
        """Test that an invalid path is left as a placeholder."""
        context, machine = context_and_machine
        formatter = TemplateFormatter(machine, context)
        result = formatter.format_string(
            "Invalid: {machine.non_existent_attr}"
        )
        assert result == "Invalid: {machine.non_existent_attr}"

    def test_unresolved_top_level(self, context_and_machine):
        """Test that an invalid top-level path is left as a placeholder."""
        context, machine = context_and_machine
        formatter = TemplateFormatter(machine, context)
        result = formatter.format_string("Invalid: {invalid_key}")
        assert result == "Invalid: {invalid_key}"

    def test_mixed_resolved_and_unresolved(self, context_and_machine):
        """Test a string with both valid and invalid placeholders."""
        context, machine = context_and_machine
        formatter = TemplateFormatter(machine, context)
        template = (
            "Machine: {machine.name}, Invalid: {foo.bar}, Job: {job.name}"
        )
        expected = "Machine: MyLaser, Invalid: {foo.bar}, Job: Test Job"
        assert formatter.format_string(template) == expected

    def test_indexed_access(self, context_and_machine):
        """Test accessing elements of a tuple/list by index."""
        context, machine = context_and_machine
        formatter = TemplateFormatter(machine, context)
        template = (
            "Width: {machine.dimensions[0]}, Height: {machine.dimensions[1]}"
        )
        expected = "Width: 200, Height: 150"
        assert formatter.format_string(template) == expected

    def test_invalid_index(self, context_and_machine):
        """Test that an out-of-bounds index is left as a placeholder."""
        context, machine = context_and_machine
        formatter = TemplateFormatter(machine, context)
        result = formatter.format_string(
            "Invalid index: {machine.dimensions[2]}"
        )
        assert result == "Invalid index: {machine.dimensions[2]}"

    def test_no_placeholders(self, context_and_machine):
        """Test a string with no placeholders."""
        context, machine = context_and_machine
        formatter = TemplateFormatter(machine, context)
        template = "This is a simple string."
        assert formatter.format_string(template) == template

    def test_empty_string(self, context_and_machine):
        """Test formatting of an empty string."""
        context, machine = context_and_machine
        formatter = TemplateFormatter(machine, context)
        assert formatter.format_string("") == ""

    def test_expand_simple_script(self, context_and_machine):
        """Test that a script with no includes is formatted correctly."""
        context, machine = context_and_machine
        formatter = TemplateFormatter(machine, context)
        script = Script(
            name="Test", code=["Machine is {machine.name}", "G0 X0"]
        )
        result = formatter.expand_script(script)
        assert result == ["Machine is MyLaser", "G0 X0"]

    def test_expand_with_include(self, context_and_machine):
        """Test that @include directives are expanded."""
        context, machine = context_and_machine
        formatter = TemplateFormatter(machine, context)
        script = Script(
            name="Wrapper", code=["G90", "@include(First Macro)", "G91"]
        )
        result = formatter.expand_script(script)
        assert result == ["G90", "G0 X10 Y10", "G91"]

    def test_expand_with_formatting_and_include(self, context_and_machine):
        """Test that variables are formatted in the top-level script."""
        context, machine = context_and_machine
        formatter = TemplateFormatter(machine, context)
        script = Script(
            name="Wrapper", code=["Job: {job.name}", "@include(Second Macro)"]
        )
        result = formatter.expand_script(script)
        assert result == ["Job: Test Job", "G0 Z5"]

    def test_expand_unknown_macro(self, context_and_machine):
        """Test that including a non-existent macro produces a warning."""
        context, machine = context_and_machine
        formatter = TemplateFormatter(machine, context)
        script = Script(name="Test", code=["@include(Unknown Macro)"])
        result = formatter.expand_script(script)
        assert result == [
            "; WARNING: Macro 'Unknown Macro'  not found or disabled."
        ]

    def test_expand_disabled_macro(self, context_and_machine):
        """Test that including a disabled macro produces a warning."""
        context, machine = context_and_machine
        formatter = TemplateFormatter(machine, context)
        script = Script(name="Test", code=["@include(Disabled Macro)"])
        result = formatter.expand_script(script)
        assert result == [
            "; WARNING: Macro 'Disabled Macro'  not found or disabled."
        ]

    def test_circular_dependency(self, context_and_machine):
        """Test that circular dependencies are detected and reported."""
        context, machine = context_and_machine
        formatter = TemplateFormatter(machine, context)
        script = Script(name="Circular1", code=["@include(Circular2)"])
        result = formatter.expand_script(script)
        assert result == [
            "; ERROR: Circular dependency detected. Macro 'Circular1'"
            " was included again."
        ]
