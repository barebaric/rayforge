import numpy as np
import pytest

from rayforge.core.color import ColorSet, normalize_color


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

        with pytest.raises(Exception):
            colorset._data = {}


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
