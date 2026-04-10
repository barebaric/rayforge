from rayforge.core.ops.axis import Axis
from rayforge.machine.models.axis import (
    AxisConfig,
    AxisDirection,
    AxisSet,
    AxisType,
)
from rayforge.machine.models.rotary_module import RotaryModule


class TestAxisConfig:
    def test_construction_defaults(self):
        cfg = AxisConfig(
            letter=Axis.X,
            axis_type=AxisType.LINEAR,
            extents=(0, 200),
        )
        assert cfg.letter == Axis.X
        assert cfg.axis_type == AxisType.LINEAR
        assert cfg.extents == (0, 200)
        assert cfg.direction == AxisDirection.NORMAL
        assert cfg.gcode_letter is None
        assert cfg.resolution == 0.01
        assert cfg.rotary_diameter is None

    def test_construction_all_fields(self):
        cfg = AxisConfig(
            letter=Axis.A,
            axis_type=AxisType.ROTARY,
            extents=(0, 360),
            direction=AxisDirection.REVERSED,
            gcode_letter="A",
            resolution=0.1,
            rotary_diameter=25.0,
        )
        assert cfg.direction == AxisDirection.REVERSED
        assert cfg.gcode_letter == "A"
        assert cfg.resolution == 0.1
        assert cfg.rotary_diameter == 25.0

    def test_to_dict(self):
        cfg = AxisConfig(
            letter=Axis.X,
            axis_type=AxisType.LINEAR,
            extents=(0, 200),
        )
        d = cfg.to_dict()
        assert d == {
            "letter": "X",
            "axis_type": "linear",
            "extents": [0, 200],
            "direction": "normal",
            "resolution": 0.01,
        }
        assert "gcode_letter" not in d
        assert "rotary_diameter" not in d

    def test_to_dict_with_optional_fields(self):
        cfg = AxisConfig(
            letter=Axis.A,
            axis_type=AxisType.ROTARY,
            extents=(0, 360),
            gcode_letter="A",
            rotary_diameter=30.0,
        )
        d = cfg.to_dict()
        assert d["gcode_letter"] == "A"
        assert d["rotary_diameter"] == 30.0

    def test_from_dict(self):
        d = {
            "letter": "Y",
            "axis_type": "linear",
            "extents": [0, 300],
            "direction": "reversed",
            "resolution": 0.05,
        }
        cfg = AxisConfig.from_dict(d)
        assert cfg.letter == Axis.Y
        assert cfg.axis_type == AxisType.LINEAR
        assert cfg.extents == (0, 300)
        assert cfg.direction == AxisDirection.REVERSED
        assert cfg.resolution == 0.05
        assert cfg.gcode_letter is None
        assert cfg.rotary_diameter is None

    def test_roundtrip_linear(self):
        original = AxisConfig(
            letter=Axis.X,
            axis_type=AxisType.LINEAR,
            extents=(0, 200),
        )
        restored = AxisConfig.from_dict(original.to_dict())
        assert restored.letter == original.letter
        assert restored.axis_type == original.axis_type
        assert restored.extents == original.extents
        assert restored.direction == original.direction
        assert restored.gcode_letter == original.gcode_letter
        assert restored.resolution == original.resolution
        assert restored.rotary_diameter == original.rotary_diameter

    def test_roundtrip_rotary(self):
        original = AxisConfig(
            letter=Axis.A,
            axis_type=AxisType.ROTARY,
            extents=(0, 360),
            gcode_letter="A",
            rotary_diameter=50.0,
        )
        restored = AxisConfig.from_dict(original.to_dict())
        assert restored.gcode_letter == "A"
        assert restored.rotary_diameter == 50.0

    def test_from_dict_defaults(self):
        d = {
            "letter": "Z",
            "axis_type": "linear",
            "extents": [-50, 50],
        }
        cfg = AxisConfig.from_dict(d)
        assert cfg.direction == AxisDirection.NORMAL
        assert cfg.gcode_letter is None
        assert cfg.resolution == 0.01
        assert cfg.rotary_diameter is None


class TestAxisSet:
    def _make_configs(self):
        return [
            AxisConfig(
                letter=Axis.X,
                axis_type=AxisType.LINEAR,
                extents=(0, 200),
            ),
            AxisConfig(
                letter=Axis.Y,
                axis_type=AxisType.LINEAR,
                extents=(0, 300),
            ),
            AxisConfig(
                letter=Axis.Z,
                axis_type=AxisType.LINEAR,
                extents=(-50, 50),
            ),
            AxisConfig(
                letter=Axis.A,
                axis_type=AxisType.ROTARY,
                extents=(0, 360),
                rotary_diameter=25.0,
            ),
        ]

    def test_linear_axes_filtered(self):
        axset = AxisSet(self._make_configs())
        assert len(axset.linear_axes) == 3
        assert all(c.axis_type == AxisType.LINEAR for c in axset.linear_axes)

    def test_rotary_axes_filtered(self):
        axset = AxisSet(self._make_configs())
        assert len(axset.rotary_axes) == 1
        assert axset.rotary_axes[0].letter == Axis.A

    def test_get_existing(self):
        axset = AxisSet(self._make_configs())
        cfg = axset.get(Axis.X)
        assert cfg is not None
        assert cfg.letter == Axis.X

    def test_get_missing(self):
        axset = AxisSet(self._make_configs())
        assert axset.get(Axis.B) is None

    def test_configs_preserved(self):
        configs = self._make_configs()
        axset = AxisSet(configs)
        assert axset.configs is configs

    def test_to_dict(self):
        axset = AxisSet(self._make_configs())
        d = axset.to_dict()
        assert "configs" in d
        assert len(d["configs"]) == 4
        assert d["configs"][0]["letter"] == "X"

    def test_roundtrip(self):
        original = AxisSet(self._make_configs())
        restored = AxisSet.from_dict(original.to_dict())
        assert len(restored.configs) == len(original.configs)
        for orig, rest in zip(original.configs, restored.configs):
            assert rest.letter == orig.letter
            assert rest.axis_type == orig.axis_type
            assert rest.extents == orig.extents
            assert rest.direction == orig.direction
            assert rest.gcode_letter == orig.gcode_letter
            assert rest.resolution == orig.resolution
            assert rest.rotary_diameter == orig.rotary_diameter

    def test_roundtrip_preserves_filtered_subsets(self):
        original = AxisSet(self._make_configs())
        restored = AxisSet.from_dict(original.to_dict())
        assert len(restored.linear_axes) == 3
        assert len(restored.rotary_axes) == 1


class TestAxisSetFromLegacy:
    def test_no_rotary_modules(self):
        axset = AxisSet.from_legacy(
            axis_extents=(200, 300),
            reverse_x=False,
            reverse_y=False,
            reverse_z=False,
        )
        assert len(axset.configs) == 3
        assert len(axset.linear_axes) == 3
        assert len(axset.rotary_axes) == 0

        x = axset.get(Axis.X)
        assert x is not None
        assert x.axis_type == AxisType.LINEAR
        assert x.extents == (0, 200)
        assert x.direction == AxisDirection.NORMAL

        y = axset.get(Axis.Y)
        assert y is not None
        assert y.extents == (0, 300)

        z = axset.get(Axis.Z)
        assert z is not None
        assert z.extents == (-50, 50)

    def test_reversed_axes(self):
        axset = AxisSet.from_legacy(
            axis_extents=(400, 400),
            reverse_x=True,
            reverse_y=True,
            reverse_z=True,
        )
        x = axset.get(Axis.X)
        y = axset.get(Axis.Y)
        z = axset.get(Axis.Z)
        assert x is not None
        assert y is not None
        assert z is not None
        assert x.direction == AxisDirection.REVERSED
        assert y.direction == AxisDirection.REVERSED
        assert z.direction == AxisDirection.REVERSED

    def test_with_one_rotary_module(self):
        rm = RotaryModule()
        rm.axis = Axis.A
        rm.default_diameter = 40.0
        axset = AxisSet.from_legacy(
            axis_extents=(200, 200),
            reverse_x=False,
            reverse_y=False,
            reverse_z=False,
            rotary_modules={"uid": rm},
        )
        assert len(axset.configs) == 4
        assert len(axset.linear_axes) == 3
        assert len(axset.rotary_axes) == 1

        rotary = axset.get(Axis.A)
        assert rotary is not None
        assert rotary.axis_type == AxisType.ROTARY
        assert rotary.rotary_diameter == 40.0

    def test_with_multiple_rotary_modules(self):
        rm1 = RotaryModule()
        rm1.axis = Axis.A
        rm1.default_diameter = 25.0
        rm2 = RotaryModule()
        rm2.axis = Axis.B
        rm2.default_diameter = 30.0
        axset = AxisSet.from_legacy(
            axis_extents=(300, 300),
            reverse_x=False,
            reverse_y=False,
            reverse_z=False,
            rotary_modules={"a": rm1, "b": rm2},
        )
        assert len(axset.configs) == 5
        assert len(axset.rotary_axes) == 2

        a = axset.get(Axis.A)
        b = axset.get(Axis.B)
        assert a is not None
        assert b is not None
        assert a.rotary_diameter == 25.0
        assert b.rotary_diameter == 30.0

    def test_none_rotary_modules(self):
        axset = AxisSet.from_legacy(
            axis_extents=(200, 200),
            reverse_x=False,
            reverse_y=False,
            reverse_z=False,
            rotary_modules=None,
        )
        assert len(axset.configs) == 3

    def test_from_legacy_roundtrip(self):
        rm = RotaryModule()
        rm.axis = Axis.A
        rm.default_diameter = 50.0
        original = AxisSet.from_legacy(
            axis_extents=(200, 300),
            reverse_x=True,
            reverse_y=False,
            reverse_z=True,
            rotary_modules={"uid": rm},
        )
        restored = AxisSet.from_dict(original.to_dict())
        assert len(restored.configs) == 4
        assert len(restored.linear_axes) == 3
        assert len(restored.rotary_axes) == 1

        x = restored.get(Axis.X)
        assert x is not None
        assert x.direction == AxisDirection.REVERSED
        assert x.extents == (0, 200)

        z = restored.get(Axis.Z)
        assert z is not None
        assert z.direction == AxisDirection.REVERSED

        a = restored.get(Axis.A)
        assert a is not None
        assert a.axis_type == AxisType.ROTARY
        assert a.rotary_diameter == 50.0
