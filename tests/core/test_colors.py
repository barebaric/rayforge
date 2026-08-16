import numpy as np
import pytest

from rayforge.core.color import ColorSet, colorize_rgb, normalize_color


class TestColorSetGetLut:
    def test_get_valid_lut(self):
        lut = np.zeros((256, 4), dtype=np.float32)
        lut[:, 0] = 1.0
        colorset = ColorSet(_data={"red": lut})

        result = colorset.get_lut("red")
        assert result is lut

    def test_get_missing_lut_returns_default(self):
        colorset = ColorSet(_data={})

        result = colorset.get_lut("missing")

        assert result.shape == (256, 4)
        assert result.dtype == np.float32
        assert result[0, 0] == 1.0
        assert result[0, 2] == 1.0

    def test_get_invalid_shape_returns_default(self):
        invalid_lut = np.zeros((100, 4), dtype=np.float32)
        colorset = ColorSet(_data={"invalid": invalid_lut})

        result = colorset.get_lut("invalid")

        assert result.shape == (256, 4)


class TestColorSetGetRgba:
    def test_get_valid_rgba(self):
        colorset = ColorSet(_data={"red": (1.0, 0.0, 0.0, 1.0)})

        result = colorset.get_rgba("red")
        assert result == (1.0, 0.0, 0.0, 1.0)

    def test_get_missing_rgba_returns_default(self):
        colorset = ColorSet(_data={})

        result = colorset.get_rgba("missing")

        assert result == (1.0, 0.0, 1.0, 1.0)

    def test_get_invalid_tuple_returns_default(self):
        colorset = ColorSet(_data={"invalid": (1.0, 0.0)})

        result = colorset.get_rgba("invalid")

        assert result == (1.0, 0.0, 1.0, 1.0)

    def test_get_non_tuple_returns_default(self):
        colorset = ColorSet(_data={"invalid": "not a tuple"})

        result = colorset.get_rgba("invalid")

        assert result == (1.0, 0.0, 1.0, 1.0)


class TestColorSetRepr:
    def test_repr_shows_sorted_keys(self):
        colorset = ColorSet(_data={"z": (0, 0, 0, 1), "a": (1, 1, 1, 1)})

        result = repr(colorset)

        assert result == "ColorSet(keys=['a', 'z'])"

    def test_repr_empty(self):
        colorset = ColorSet(_data={})

        result = repr(colorset)

        assert result == "ColorSet(keys=[])"


class TestColorSetSerialization:
    def test_to_dict_with_lut(self):
        lut = np.ones((256, 4), dtype=np.float32)
        colorset = ColorSet(_data={"my_lut": lut})

        result = colorset.to_dict()

        assert "_data" in result
        assert "my_lut" in result["_data"]
        assert result["_data"]["my_lut"]["__type__"] == "numpy"
        assert result["_data"]["my_lut"]["dtype"] == "float32"

    def test_to_dict_with_rgba(self):
        colorset = ColorSet(_data={"my_color": (1.0, 0.5, 0.0, 1.0)})

        result = colorset.to_dict()

        assert result["_data"]["my_color"]["__type__"] == "tuple"
        assert result["_data"]["my_color"]["data"] == (1.0, 0.5, 0.0, 1.0)

    def test_from_dict_with_lut(self):
        lut = np.ones((256, 4), dtype=np.float32)
        data = {
            "_data": {
                "my_lut": {
                    "__type__": "numpy",
                    "data": lut.tolist(),
                    "dtype": "float32",
                }
            }
        }

        result = ColorSet.from_dict(data)

        assert "my_lut" in result._data
        assert isinstance(result._data["my_lut"], np.ndarray)
        assert result._data["my_lut"].shape == (256, 4)

    def test_from_dict_with_rgba(self):
        data = {
            "_data": {
                "my_color": {"__type__": "tuple", "data": (1.0, 0.5, 0.0, 1.0)}
            }
        }

        result = ColorSet.from_dict(data)

        assert result._data["my_color"] == (1.0, 0.5, 0.0, 1.0)

    def test_roundtrip(self):
        lut = np.zeros((256, 4), dtype=np.float32)
        lut[:, 0] = 1.0
        original = ColorSet(
            _data={"my_lut": lut, "my_color": (0.5, 0.5, 0.5, 1.0)}
        )

        serialized = original.to_dict()
        restored = ColorSet.from_dict(serialized)

        assert "my_lut" in restored._data
        assert "my_color" in restored._data
        np.testing.assert_array_equal(restored._data["my_lut"], lut)
        assert restored._data["my_color"] == (0.5, 0.5, 0.5, 1.0)

    def test_from_dict_handles_data_without_wrapper(self):
        data = {
            "my_color": {"__type__": "tuple", "data": (1.0, 0.0, 0.0, 1.0)}
        }

        result = ColorSet.from_dict(data)

        assert result._data["my_color"] == (1.0, 0.0, 0.0, 1.0)


class TestColorSetImmutability:
    def test_frozen_dataclass(self):
        colorset = ColorSet(_data={"color": (1.0, 0.0, 0.0, 1.0)})

        with pytest.raises(AttributeError):
            colorset._data = {}  # type: ignore[assignment]


class TestNormalizeColor:
    def test_lowercase_hex(self):
        assert normalize_color("#e34c4c") == "#e34c4c"

    def test_uppercase_hex(self):
        assert normalize_color("#E34C4C") == "#e34c4c"

    def test_short_hex(self):
        assert normalize_color("#f00") == "#ff0000"

    def test_css_color_name(self):
        assert normalize_color("red") == "#ff0000"

    def test_rgb_string(self):
        assert normalize_color("rgb(227, 76, 76)") == "#e34c4c"

    def test_eight_digit_hex_drops_alpha(self):
        assert normalize_color("#e34c4cff") == "#e34c4c"

    def test_whitespace(self):
        assert normalize_color("  #E34C4C  ") == "#e34c4c"

    def test_invalid_returns_none(self):
        assert normalize_color("") is None
        assert normalize_color(None) is None
        assert normalize_color("not-a-color") is None
        assert normalize_color("   ") is None


class TestColorizeRgb:
    def test_white_shifts_exactly_to_tint(self):
        """A white pixel becomes exactly the tint color."""
        rgb = np.array([[[255, 255, 255]]], dtype=np.uint8)
        out = colorize_rgb(rgb, (1.0, 0.0, 0.0, 1.0))
        assert out[0, 0].tolist() == pytest.approx([255.0, 0.0, 0.0], abs=1.0)

    def test_colored_pixel_shifts_hue_preserving_brightness(self):
        """A colored (e.g. pale blue) pixel shifts to the tint hue."""
        rgb = np.array([[[200, 225, 230]]], dtype=np.uint8)  # pale blue
        out = colorize_rgb(rgb, (1.0, 0.0, 0.0, 1.0))
        # red channel carries all the energy, others are ~0
        assert out[0, 0, 1] < 1.0
        assert out[0, 0, 2] < 1.0
        assert out[0, 0, 0] > 200.0

    def test_darker_texel_darker_result(self):
        """Shading is preserved: darker texel -> darker tinted result."""
        a = colorize_rgb(
            np.array([[[255, 255, 255]]], dtype=np.uint8), (0.0, 1.0, 0.0, 1.0)
        )
        b = colorize_rgb(
            np.array([[[128, 128, 128]]], dtype=np.uint8), (0.0, 1.0, 0.0, 1.0)
        )
        assert a[0, 0, 1] == pytest.approx(255.0, abs=1.0)  # green channel
        assert b[0, 0, 1] == pytest.approx(128.0, abs=1.0)
        assert a[0, 0, 1] > b[0, 0, 1]
