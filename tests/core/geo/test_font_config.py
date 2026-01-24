import pytest
from rayforge.core.geo.font_config import FontConfig


class TestFontConfig:
    """Tests for FontConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = FontConfig()

        assert config.font_family == "sans-serif"
        assert config.font_size == 10.0
        assert config.bold is False
        assert config.italic is False
        assert config.extra == {}

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        config = FontConfig(
            font_family="Arial",
            font_size=12.0,
            bold=True,
            italic=True,
        )

        assert config.font_family == "Arial"
        assert config.font_size == 12.0
        assert config.bold is True
        assert config.italic is True
        assert config.extra == {}

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = FontConfig(
            font_family="serif",
            font_size=14.0,
            bold=True,
            italic=False,
        )

        result = config.to_dict()

        assert result == {
            "font_family": "serif",
            "font_size": 14.0,
            "bold": True,
            "italic": False,
        }

    def test_to_dict_with_extra(self):
        """Test serialization includes extra fields."""
        config = FontConfig(
            font_family="monospace",
            font_size=9.0,
        )
        config.extra["custom_field"] = "custom_value"

        result = config.to_dict()

        assert result["font_family"] == "monospace"
        assert result["font_size"] == 9.0
        assert result["custom_field"] == "custom_value"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "font_family": "serif",
            "font_size": 14.0,
            "bold": True,
            "italic": False,
        }

        config = FontConfig.from_dict(data)

        assert config.font_family == "serif"
        assert config.font_size == 14.0
        assert config.bold is True
        assert config.italic is False
        assert config.extra == {}

    def test_from_dict_none(self):
        """Test deserialization from None returns defaults."""
        config = FontConfig.from_dict(None)

        assert config.font_family == "sans-serif"
        assert config.font_size == 10.0
        assert config.bold is False
        assert config.italic is False
        assert config.extra == {}

    def test_from_dict_partial(self):
        """Test deserialization with partial data."""
        data = {
            "font_family": "Arial",
            "bold": True,
        }

        config = FontConfig.from_dict(data)

        assert config.font_family == "Arial"
        assert config.font_size == 10.0
        assert config.bold is True
        assert config.italic is False
        assert config.extra == {}

    def test_from_dict_with_extra(self):
        """Test deserialization stores unknown fields in extra."""
        data = {
            "font_family": "monospace",
            "font_size": 9.0,
            "future_field": "future_value",
            "another_future_field": 42,
        }

        config = FontConfig.from_dict(data)

        assert config.font_family == "monospace"
        assert config.font_size == 9.0
        assert config.extra == {
            "future_field": "future_value",
            "another_future_field": 42,
        }

    def test_roundtrip_serialization(self):
        """Test that serialization and deserialization are inverse."""
        original = FontConfig(
            font_family="Courier",
            font_size=11.0,
            bold=False,
            italic=True,
        )

        serialized = original.to_dict()
        deserialized = FontConfig.from_dict(serialized)

        assert deserialized.font_family == original.font_family
        assert deserialized.font_size == original.font_size
        assert deserialized.bold == original.bold
        assert deserialized.italic == original.italic

    def test_copy(self):
        """Test that copy creates independent instance."""
        original = FontConfig(
            font_family="Arial",
            font_size=12.0,
            bold=True,
        )

        original.extra["field1"] = "value1"

        copy = original.copy()

        assert copy.font_family == original.font_family
        assert copy.font_size == original.font_size
        assert copy.bold == original.bold
        assert copy.italic == original.italic
        assert copy.extra == original.extra

        copy.extra["field2"] = "value2"

        assert "field2" not in original.extra
        assert "field1" in copy.extra

    def test_equality(self):
        """Test equality comparison."""
        config1 = FontConfig(
            font_family="Arial",
            font_size=12.0,
            bold=True,
            italic=False,
        )

        config2 = FontConfig(
            font_family="Arial",
            font_size=12.0,
            bold=True,
            italic=False,
        )

        assert config1 == config2

    def test_inequality(self):
        """Test inequality comparison."""
        config1 = FontConfig(font_family="Arial", font_size=12.0)
        config2 = FontConfig(font_family="serif", font_size=12.0)

        assert config1 != config2

    def test_equality_with_extra(self):
        """Test equality includes extra fields."""
        config1 = FontConfig(font_family="Arial", font_size=12.0)
        config1.extra["custom"] = "value"

        config2 = FontConfig(font_family="Arial", font_size=12.0)
        config2.extra["custom"] = "value"

        assert config1 == config2

    def test_inequality_with_extra(self):
        """Test inequality with different extra fields."""
        config1 = FontConfig(font_family="Arial", font_size=12.0)
        config1.extra["custom"] = "value1"

        config2 = FontConfig(font_family="Arial", font_size=12.0)
        config2.extra["custom"] = "value2"

        assert config1 != config2

    def test_hash(self):
        """Test that hash is consistent with equality."""
        config1 = FontConfig(
            font_family="Arial",
            font_size=12.0,
            bold=True,
        )

        config2 = FontConfig(
            font_family="Arial",
            font_size=12.0,
            bold=True,
        )

        assert hash(config1) == hash(config2)

        config3 = FontConfig(font_family="serif", font_size=12.0)

        assert hash(config1) != hash(config3)

    def test_hash_with_extra(self):
        """Test hash includes extra fields."""
        config1 = FontConfig(font_family="Arial", font_size=12.0)
        config1.extra["custom"] = "value"

        config2 = FontConfig(font_family="Arial", font_size=12.0)
        config2.extra["custom"] = "value"

        assert hash(config1) == hash(config2)

    def test_get_font_metrics_returns_tuple(self):
        """Tests that font metrics returns a tuple."""
        config = FontConfig()
        metrics = config.get_font_metrics()
        assert isinstance(metrics, tuple)
        assert len(metrics) == 3

    def test_get_font_metrics_structure(self):
        """Tests that font metrics has (ascent, descent, height) structure."""
        config = FontConfig()
        ascent, descent, height = config.get_font_metrics()
        assert isinstance(ascent, float)
        assert isinstance(descent, float)
        assert isinstance(height, float)

    def test_get_font_metrics_positive_height(self):
        """Tests that total height is positive."""
        config = FontConfig()
        _, _, height = config.get_font_metrics()
        assert height > 0

    def test_get_font_metrics_descent_value(self):
        """Tests that descent has a numeric value."""
        config = FontConfig()
        _, descent, _ = config.get_font_metrics()
        assert isinstance(descent, float)

    def test_get_font_metrics_height_valid(self):
        """Tests that height is a positive value."""
        config = FontConfig()
        _, _, height = config.get_font_metrics()
        assert height > 0

    def test_get_font_metrics_font_size_scaling(self):
        """Tests that font size scales metrics approximately."""
        config_10 = FontConfig(font_size=10.0)
        config_20 = FontConfig(font_size=20.0)
        metrics_10 = config_10.get_font_metrics()
        metrics_20 = config_20.get_font_metrics()

        ascent_10, _, height_10 = metrics_10
        ascent_20, _, height_20 = metrics_20

        # Relaxed tolerance for CI environments where fonts may behave
        # differently
        assert ascent_20 == pytest.approx(2 * ascent_10, rel=0.1)
        assert height_20 == pytest.approx(2 * height_10, rel=0.1)

    def test_get_font_metrics_font_family(self):
        """Tests that font family parameter is accepted."""
        config_sans = FontConfig(font_family="sans-serif")
        config_serif = FontConfig(font_family="serif")
        metrics_sans = config_sans.get_font_metrics()
        metrics_serif = config_serif.get_font_metrics()

        assert len(metrics_sans) == 3
        assert len(metrics_serif) == 3

    def test_get_font_metrics_bold(self):
        """Tests that bold font has metrics."""
        config = FontConfig(bold=True)
        ascent, descent, height = config.get_font_metrics()
        assert height > 0

    def test_get_font_metrics_italic(self):
        """Tests that italic font has metrics."""
        config = FontConfig(italic=True)
        ascent, descent, height = config.get_font_metrics()
        assert height > 0

    def test_get_text_width_empty_string(self):
        """Tests that empty string returns zero width."""
        config = FontConfig()
        width = config.get_text_width("")
        assert width == 0.0

    def test_get_text_width_whitespace(self):
        """Tests that whitespace has positive width."""
        config = FontConfig()
        width = config.get_text_width(" ")
        assert width > 0.0

    def test_get_text_width_single_char(self):
        """Tests that single character has positive width."""
        config = FontConfig()
        width = config.get_text_width("A")
        assert width > 0.0

    def test_get_text_width_multiple_chars(self):
        """Tests that multiple characters have larger width."""
        config = FontConfig()
        width_single = config.get_text_width("A")
        width_multiple = config.get_text_width("AAA")
        assert width_multiple > width_single

    def test_get_text_width_font_size_scaling(self):
        """Tests that font size scales width approximately."""
        config_10 = FontConfig(font_size=10.0)
        config_20 = FontConfig(font_size=20.0)
        width_10 = config_10.get_text_width("A")
        width_20 = config_20.get_text_width("A")
        assert width_20 == pytest.approx(2 * width_10, rel=0.1)

    def test_get_text_width_different_chars(self):
        """Tests that different characters have different widths."""
        config = FontConfig()
        width_i = config.get_text_width("i")
        width_w = config.get_text_width("W")
        assert width_w > width_i

    def test_get_text_width_bold(self):
        """Tests that bold text has width."""
        config = FontConfig(bold=True)
        width = config.get_text_width("A")
        assert width > 0.0

    def test_get_text_width_italic(self):
        """Tests that italic text has width."""
        config = FontConfig(italic=True)
        width = config.get_text_width("A")
        assert width > 0.0

    def test_get_text_width_font_family(self):
        """Tests that different font families have different widths."""
        # Use 'W' to maximize difference between proportional and fixed width
        # fonts
        config_sans = FontConfig(font_family="sans-serif")
        config_mono = FontConfig(font_family="monospace")
        width_sans = config_sans.get_text_width("W")
        width_mono = config_mono.get_text_width("W")

        # In limited environments (CI), fonts might map to the same fallback.
        if width_sans == width_mono:
            pytest.skip(
                "System fonts for sans-serif and monospace appear identical"
            )

        assert width_sans != width_mono

    def test_get_font_metrics_default_params(self):
        """Tests that default parameters work."""
        config = FontConfig()
        metrics = config.get_font_metrics()
        assert len(metrics) == 3

    def test_get_text_width_default_params(self):
        """Tests that default parameters work."""
        config = FontConfig()
        width = config.get_text_width("A")
        assert width > 0.0
