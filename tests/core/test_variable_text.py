from rayforge.core.variable_text import (
    SerialSettings,
    VariableText,
    referenced_template_names,
)


class TestSerialSettings:
    def test_defaults(self):
        s = SerialSettings()
        assert s.current == 1
        assert s.start == 1
        assert s.step == 1
        assert s.end is None
        assert s.auto_advance is True
        assert s.advance_on_export is False

    def test_value_at_offset_uses_step(self):
        s = SerialSettings(current=10, start=1, step=5)
        assert s.value_at(0) == 10
        assert s.value_at(1) == 15
        assert s.value_at(3) == 25

    def test_value_at_wraps_after_end(self):
        s = SerialSettings(current=4, start=1, step=1, end=5)
        assert [s.value_at(i) for i in range(4)] == [4, 5, 1, 2]

    def test_value_at_wraps_with_step(self):
        s = SerialSettings(current=90, start=10, step=40, end=95)
        # Range is 10, 50, 90; 130 is past the end and wraps to 10.
        assert [s.value_at(i) for i in range(3)] == [90, 10, 50]

    def test_value_at_negative_offset_wraps_backwards(self):
        s = SerialSettings(current=1, start=1, step=1, end=3)
        assert s.value_at(-1) == 3

    def test_zero_step_keeps_value(self):
        s = SerialSettings(current=7, step=0)
        assert s.value_at(5) == 7

    def test_advanced_moves_current(self):
        s = SerialSettings(current=3, step=2)
        assert s.advanced(4).current == 11
        assert s.advanced().current == 5
        assert s.current == 3

    def test_previous_moves_back_one_step(self):
        s = SerialSettings(current=3, step=2)
        assert s.advanced(-1).current == 1

    def test_reset_returns_to_start(self):
        s = SerialSettings(current=42, start=7)
        assert s.reset().current == 7

    def test_values_for_copies(self):
        s = SerialSettings(current=8, step=1, end=9, start=1)
        assert s.values_for(3) == [8, 9, 1]

    def test_round_trip(self):
        s = SerialSettings(
            current=12,
            start=10,
            step=2,
            end=30,
            auto_advance=False,
            advance_on_export=True,
        )
        assert SerialSettings.from_dict(s.to_dict()) == s

    def test_from_dict_ignores_missing_keys(self):
        assert SerialSettings.from_dict({}) == SerialSettings()

    def test_from_dict_rejects_end_below_start(self):
        s = SerialSettings.from_dict({"start": 10, "end": 5})
        assert s.end is None


class TestVariableText:
    def test_template_values_with_offset(self):
        vt = VariableText()
        vt.serial = SerialSettings(current=100, step=10)
        assert vt.template_values(0)["serial"] == 100
        assert vt.template_values(2)["serial"] == 120

    def test_setting_serial_emits_changed(self):
        vt = VariableText()
        received = []

        def on_changed(sender):
            received.append(sender)

        vt.changed.connect(on_changed)
        vt.serial = SerialSettings(current=5)
        assert received == [vt]

    def test_setting_equal_serial_is_silent(self):
        vt = VariableText()
        received = []

        def on_changed(sender):
            received.append(sender)

        vt.changed.connect(on_changed)
        vt.serial = SerialSettings()
        assert received == []

    def test_names(self):
        assert "serial" in VariableText().names

    def test_round_trip(self):
        vt = VariableText()
        vt.serial = SerialSettings(current=3, start=2, step=4)
        restored = VariableText.from_dict(vt.to_dict())
        assert restored.serial == vt.serial

    def test_from_dict_none(self):
        assert VariableText.from_dict(None).serial == SerialSettings()


class TestReferencedTemplateNames:
    def test_finds_names_in_placeholders(self):
        names = referenced_template_names(["No. {serial:04d}", "{width * 2}"])
        assert {"serial", "width"} <= names

    def test_ignores_text_outside_placeholders(self):
        assert "serial" not in referenced_template_names(["serial"])

    def test_handles_function_calls_and_subscripts(self):
        names = referenced_template_names(['{csv["Name"]} {today()}'])
        assert {"csv", "today"} <= names
        assert "Name" not in names

    def test_invalid_expression_falls_back_to_identifiers(self):
        names = referenced_template_names(["{serial +}"])
        assert "serial" in names
