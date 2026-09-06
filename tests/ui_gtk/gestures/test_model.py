# flake8: noqa: E402
"""Tests for the gesture data model."""

import gi

gi.require_version("Gtk", "4.0")

import pytest

pytestmark = pytest.mark.ui
from gi.repository import Gdk

from rayforge.ui_gtk.gestures.model import (
    BUTTON_MIDDLE,
    BUTTON_PRIMARY,
    BUTTON_SECONDARY,
    GestureKind,
    GestureSpec,
    normalize_modifiers,
)


class TestNormalizeModifiers:
    def test_masks_unrelated_flags(self):
        state = (
            Gdk.ModifierType.SHIFT_MASK
            | Gdk.ModifierType.LOCK_MASK
            | Gdk.ModifierType.BUTTON1_MASK
        )
        assert normalize_modifiers(state) == Gdk.ModifierType.SHIFT_MASK

    def test_combines_matched_modifiers(self):
        state = (
            Gdk.ModifierType.SHIFT_MASK
            | Gdk.ModifierType.CONTROL_MASK
            | Gdk.ModifierType.LOCK_MASK
        )
        expected = Gdk.ModifierType.SHIFT_MASK | Gdk.ModifierType.CONTROL_MASK
        assert normalize_modifiers(state) == expected

    def test_plain_state_yields_zero(self):
        assert normalize_modifiers(Gdk.ModifierType(0)) == (
            Gdk.ModifierType(0)
        )


class TestGestureSpec:
    def test_drag_spec_serialization_round_trip(self):
        spec = GestureSpec(
            GestureKind.DRAG,
            button=BUTTON_MIDDLE,
            modifiers=Gdk.ModifierType.SHIFT_MASK,
        )
        restored = GestureSpec.from_config_string(spec.to_config_string())
        assert restored == spec

    def test_click_spec_serialization_round_trip(self):
        spec = GestureSpec(GestureKind.CLICK, button=BUTTON_SECONDARY)
        assert spec.to_config_string() == "click+secondary"
        assert GestureSpec.from_config_string("click+secondary") == spec

    def test_scroll_spec_is_canonical(self):
        spec = GestureSpec(
            GestureKind.SCROLL,
            button=BUTTON_MIDDLE,
            modifiers=Gdk.ModifierType.SHIFT_MASK,
        )
        assert spec.button is None
        assert spec.modifiers == Gdk.ModifierType(0)
        assert spec.to_config_string() == "scroll"
        assert GestureSpec.from_config_string("scroll") == (
            GestureSpec(GestureKind.SCROLL)
        )

    def test_numeric_button_names_are_accepted(self):
        assert GestureSpec.from_config_string("drag+2") == (
            GestureSpec(GestureKind.DRAG, button=BUTTON_MIDDLE)
        )

    def test_modifier_order_is_insensitive(self):
        spec = GestureSpec(
            GestureKind.DRAG,
            button=BUTTON_PRIMARY,
            modifiers=(
                Gdk.ModifierType.SHIFT_MASK | Gdk.ModifierType.ALT_MASK
            ),
        )
        assert GestureSpec.from_config_string("drag+alt+shift+primary") == spec

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "nonsense",
            "click",
            "scroll+middle",
            "scroll+shift",
            "click+middle+secondary",
            "click+middle+bogus",
            "drag+middle+middle",
        ],
    )
    def test_malformed_specs_raise_value_error(self, value):
        with pytest.raises(ValueError):
            GestureSpec.from_config_string(value)

    def test_display_label_for_scroll(self):
        spec = GestureSpec(GestureKind.SCROLL)
        assert "Scroll" in spec.display_label()

    def test_display_label_for_shift_drag(self):
        spec = GestureSpec(
            GestureKind.DRAG,
            button=BUTTON_MIDDLE,
            modifiers=Gdk.ModifierType.SHIFT_MASK,
        )
        label = spec.display_label()
        assert "Shift" in label
        assert "Middle" in label
        assert "Drag" in label

    def test_display_label_for_right_click(self):
        spec = GestureSpec(GestureKind.CLICK, button=BUTTON_SECONDARY)
        assert "Right" in spec.display_label()
