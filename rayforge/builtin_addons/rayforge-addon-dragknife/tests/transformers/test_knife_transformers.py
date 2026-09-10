"""Tests for the knife compensation transformers."""

from unittest.mock import MagicMock

import pytest
from dragknife.transformers import (
    DragKnifeTransformer,
    TangentialKnifeTransformer,
)


class TestDragKnifeTransformer:
    def test_to_spec_returns_raygeo_spec(self):
        transformer = DragKnifeTransformer(
            offset_mm=0.8, swivel_angle_deg=60.0
        )
        spec = transformer.to_spec(None, None, None)
        assert spec.offset_mm == pytest.approx(0.8)
        assert spec.swivel_angle_deg == pytest.approx(60.0)

    def test_dict_roundtrip(self):
        transformer = DragKnifeTransformer(
            offset_mm=0.9, swivel_angle_deg=75.0
        )
        data = transformer.to_dict()
        assert data["name"] == "DragKnifeTransformer"
        assert data["enabled"] is True
        restored = DragKnifeTransformer.from_dict(data)
        assert restored.offset_mm == pytest.approx(0.9)
        assert restored.swivel_angle_deg == pytest.approx(75.0)
        assert restored.enabled is True

    def test_offset_clamps_negative(self):
        transformer = DragKnifeTransformer(offset_mm=-1.0)
        assert transformer.offset_mm == 0.0

    def test_swivel_clamps_to_180(self):
        transformer = DragKnifeTransformer(swivel_angle_deg=250.0)
        assert transformer.swivel_angle_deg == 180.0

    def test_setter_sends_signal_on_change(self):
        transformer = DragKnifeTransformer()
        handler = MagicMock()
        transformer.changed.connect(handler)
        transformer.offset_mm = 1.5
        handler.assert_called_once_with(transformer)

    def test_setter_no_signal_on_same_value(self):
        transformer = DragKnifeTransformer(offset_mm=1.0)
        handler = MagicMock()
        transformer.changed.connect(handler)
        transformer.offset_mm = 1.0
        handler.assert_not_called()

    def test_labels(self):
        transformer = DragKnifeTransformer()
        assert transformer.label
        assert transformer.description


class TestTangentialKnifeTransformer:
    def test_to_spec_returns_raygeo_spec(self):
        transformer = TangentialKnifeTransformer(
            angle_tolerance_deg=40.0,
            radius_tolerance_mm=2.0,
            safe_z=3.0,
        )
        spec = transformer.to_spec(None, None, None)
        assert spec.angle_tolerance_deg == pytest.approx(40.0)
        assert spec.radius_tolerance_mm == pytest.approx(2.0)
        assert spec.safe_z == pytest.approx(3.0)

    def test_dict_roundtrip(self):
        transformer = TangentialKnifeTransformer(
            angle_tolerance_deg=15.0,
            radius_tolerance_mm=0.5,
            safe_z=4.0,
        )
        data = transformer.to_dict()
        assert data["name"] == "TangentialKnifeTransformer"
        restored = TangentialKnifeTransformer.from_dict(data)
        assert restored.angle_tolerance_deg == pytest.approx(15.0)
        assert restored.radius_tolerance_mm == pytest.approx(0.5)
        assert restored.safe_z == pytest.approx(4.0)

    def test_angle_clamps(self):
        transformer = TangentialKnifeTransformer(angle_tolerance_deg=-5.0)
        assert transformer.angle_tolerance_deg == 0.0
        transformer.angle_tolerance_deg = 500.0
        assert transformer.angle_tolerance_deg == 180.0

    def test_labels(self):
        transformer = TangentialKnifeTransformer()
        assert transformer.label
        assert transformer.description
