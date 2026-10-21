from rayforge.core.expression import referenced_template_names


class TestReferencedTemplateNames:
    def test_finds_names_in_placeholders(self):
        names = referenced_template_names(["No. {counter:04d}", "{width * 2}"])
        assert {"counter", "width"} <= names

    def test_ignores_text_outside_placeholders(self):
        assert "counter" not in referenced_template_names(["counter"])

    def test_handles_function_calls_and_subscripts(self):
        names = referenced_template_names(['{csv["Name"]} {today()}'])
        assert {"csv", "today"} <= names
        assert "Name" not in names

    def test_invalid_expression_falls_back_to_identifiers(self):
        names = referenced_template_names(["{counter +}"])
        assert "counter" in names
