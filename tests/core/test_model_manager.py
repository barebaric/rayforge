import pytest
from pathlib import Path

from rayforge.core.model import Model, ModelLibrary
from rayforge.core.model_manager import ModelManager


@pytest.fixture
def model_mgr():
    return ModelManager()


class TestModel:
    def test_basic_creation(self):
        m = Model(name="Test Chuck", path=Path("test.glb"))
        assert m.name == "Test Chuck"
        assert m.path == Path("test.glb")
        assert m.description == ""
        assert m.extra == {}

    def test_creation_with_all_fields(self):
        m = Model(
            name="Chuck",
            path=Path("a.glb"),
            description="A 4-jaw chuck",
            extra={"scale": 1.0},
        )
        assert m.name == "Chuck"
        assert m.description == "A 4-jaw chuck"
        assert m.extra == {"scale": 1.0}

    def test_from_dict(self):
        data = {
            "name": "Standard Chuck",
            "path": "standard.glb",
            "description": "A standard 3-jaw chuck",
        }
        m = Model.from_dict(data)
        assert m.name == "Standard Chuck"
        assert m.path == Path("standard.glb")
        assert m.description == "A standard 3-jaw chuck"

    def test_from_dict_with_extra(self):
        data = {
            "name": "Chuck",
            "path": "test.glb",
            "scale": 0.5,
            "author": "test",
        }
        m = Model.from_dict(data)
        assert m.extra == {"scale": 0.5, "author": "test"}

    def test_from_dict_defaults(self):
        data = {"path": "test.glb"}
        m = Model.from_dict(data)
        assert m.name == ""
        assert m.description == ""

    def test_to_dict(self):
        m = Model(
            name="Chuck",
            path=Path("a.glb"),
            description="desc",
            extra={"scale": 1.0},
        )
        d = m.to_dict()
        assert d["name"] == "Chuck"
        assert d["path"] == "a.glb"
        assert d["description"] == "desc"
        assert d["scale"] == 1.0

    def test_roundtrip(self):
        original = Model(
            name="Chuck",
            path=Path("a.glb"),
            description="desc",
            extra={"key": "value"},
        )
        restored = Model.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.path == original.path
        assert restored.description == original.description
        assert restored.extra == original.extra

    def test_str(self):
        m = Model(name="Chuck", path=Path("a.glb"))
        assert "Chuck" in str(m)
        assert "a.glb" in str(m)


class TestModelLibrary:
    def test_defaults(self, tmp_path):
        lib = ModelLibrary(
            library_id="test",
            display_name="Test",
            path=tmp_path,
        )
        assert lib.read_only is False
        assert lib.icon_name == "folder-symbolic"


class TestModelManagerInit:
    def test_str_representation(self, model_mgr):
        result = str(model_mgr)
        assert "ModelManager" in result

    def test_starts_with_no_libraries(self, model_mgr):
        libs = model_mgr.get_libraries()
        assert len(libs) == 0


class TestModelManagerResolveAbsolute:
    def test_existing_absolute_path(self, model_mgr, tmp_path):
        model_file = tmp_path / "my_model.glb"
        model_file.write_text("fake glb data")
        m = Model(name="test", path=model_file)
        result = model_mgr.resolve(m)
        assert result == model_file

    def test_nonexistent_absolute_path(self, model_mgr, tmp_path):
        m = Model(name="test", path=tmp_path / "nonexistent.glb")
        result = model_mgr.resolve(m)
        assert result is None


class TestModelManagerResolveFromLibrary:
    def test_finds_file_in_registered_library(self, model_mgr, tmp_path):
        lib_root = tmp_path / "lib"
        lib_root.mkdir()
        model_file = lib_root / "custom_chuck.glb"
        model_file.write_text("fake glb data")

        model_mgr.add_library(
            ModelLibrary(
                library_id="test",
                display_name="Test",
                path=lib_root,
                read_only=True,
            )
        )

        m = Model(name="custom", path=Path("custom_chuck.glb"))
        result = model_mgr.resolve(m)
        assert result == model_file

    def test_returns_none_when_not_found(self, model_mgr):
        m = Model(name="test", path=Path("nonexistent.glb"))
        result = model_mgr.resolve(m)
        assert result is None


class TestModelManagerResolvePrecedence:
    def test_first_library_takes_precedence(self, model_mgr, tmp_path):
        lib_a_root = tmp_path / "lib_a"
        lib_a_root.mkdir()
        (lib_a_root / "chuck.glb").write_text("lib_a version")

        lib_b_root = tmp_path / "lib_b"
        lib_b_root.mkdir()
        (lib_b_root / "chuck.glb").write_text("lib_b version")

        model_mgr.add_library(
            ModelLibrary(
                library_id="a",
                display_name="A",
                path=lib_a_root,
                read_only=True,
            )
        )
        model_mgr.add_library(
            ModelLibrary(
                library_id="b",
                display_name="B",
                path=lib_b_root,
                read_only=True,
            )
        )

        m = Model(name="chuck", path=Path("chuck.glb"))
        result = model_mgr.resolve(m)
        assert result.read_text() == "lib_a version"


class TestModelManagerGetModels:
    def test_lists_models_in_library(self, model_mgr, tmp_path):
        lib_root = tmp_path / "lib"
        lib_root.mkdir()
        (lib_root / "a.glb").write_text("a")
        (lib_root / "b.glb").write_text("b")

        lib = ModelLibrary(
            library_id="test",
            display_name="Test",
            path=lib_root,
            read_only=True,
        )
        model_mgr.add_library(lib)

        models = model_mgr.get_models(lib)
        assert len(models) == 2
        names = {m.name for m in models}
        assert names == {"a", "b"}

    def test_returns_empty_for_empty_dir(self, model_mgr, tmp_path):
        lib_root = tmp_path / "lib"
        lib_root.mkdir()

        lib = ModelLibrary(
            library_id="test",
            display_name="Test",
            path=lib_root,
            read_only=True,
        )
        model_mgr.add_library(lib)

        models = model_mgr.get_models(lib)
        assert models == []

    def test_ignores_hidden_files(self, model_mgr, tmp_path):
        lib_root = tmp_path / "lib"
        lib_root.mkdir()
        (lib_root / "visible.glb").write_text("v")
        (lib_root / ".hidden.glb").write_text("h")

        lib = ModelLibrary(
            library_id="test",
            display_name="Test",
            path=lib_root,
            read_only=True,
        )
        model_mgr.add_library(lib)

        models = model_mgr.get_models(lib)
        assert len(models) == 1
        assert models[0].name == "visible"


class TestModelManagerGetAllModels:
    def test_returns_models_from_single_library(self, model_mgr, tmp_path):
        lib_root = tmp_path / "lib"
        lib_root.mkdir()
        (lib_root / "a.glb").write_text("a")
        (lib_root / "b.glb").write_text("b")

        model_mgr.add_library(
            ModelLibrary(
                library_id="test",
                display_name="Test",
                path=lib_root,
                read_only=True,
            )
        )

        models = model_mgr.get_all_models()
        assert len(models) == 2
        names = [m.name for m in models]
        assert "a" in names
        assert "b" in names

    def test_returns_empty_when_no_libraries(self, model_mgr):
        models = model_mgr.get_all_models()
        assert models == []

    def test_deduplicates_across_libraries(self, model_mgr, tmp_path):
        lib_a_root = tmp_path / "lib_a"
        lib_a_root.mkdir()
        (lib_a_root / "standard.glb").write_text("lib_a")

        lib_b_root = tmp_path / "lib_b"
        lib_b_root.mkdir()
        (lib_b_root / "standard.glb").write_text("lib_b")
        (lib_b_root / "extra.glb").write_text("extra")

        model_mgr.add_library(
            ModelLibrary(
                library_id="a",
                display_name="A",
                path=lib_a_root,
                read_only=True,
            )
        )
        model_mgr.add_library(
            ModelLibrary(
                library_id="b",
                display_name="B",
                path=lib_b_root,
                read_only=True,
            )
        )

        models = model_mgr.get_all_models()
        filenames = [m.path.name for m in models]
        assert "standard.glb" in filenames
        assert filenames.count("standard.glb") == 1
        assert "extra.glb" in filenames


class TestModelManagerLibraryRegistration:
    def test_add_library(self, model_mgr, tmp_path):
        lib = ModelLibrary(
            library_id="test_addon",
            display_name="Test Addon",
            path=tmp_path / "addon_models",
            read_only=True,
        )
        assert model_mgr.add_library(lib) is True
        libs = model_mgr.get_libraries()
        assert len(libs) == 1
        assert libs[0].library_id == "test_addon"

    def test_add_duplicate_library_fails(self, model_mgr, tmp_path):
        lib = ModelLibrary(
            library_id="test",
            display_name="Test",
            path=tmp_path,
        )
        model_mgr.add_library(lib)
        lib2 = ModelLibrary(
            library_id="test",
            display_name="Duplicate",
            path=tmp_path,
        )
        assert model_mgr.add_library(lib2) is False

    def test_add_library_from_path(self, model_mgr, tmp_path):
        addon_dir = tmp_path / "my_addon"
        addon_dir.mkdir()
        result = model_mgr.add_library_from_path(
            addon_dir, addon_name="test_addon"
        )
        assert result is not None

    def test_unregister_all_from_addon(self, model_mgr, tmp_path):
        lib1 = ModelLibrary(
            library_id="a1",
            display_name="A1",
            path=tmp_path / "a1",
            read_only=True,
        )
        lib2 = ModelLibrary(
            library_id="a2",
            display_name="A2",
            path=tmp_path / "a2",
            read_only=True,
        )
        model_mgr.add_library(lib1, addon_name="addon1")
        model_mgr.add_library(lib2, addon_name="addon2")

        count = model_mgr.unregister_all_from_addon("addon1")
        assert count == 1
        lib_ids = [lib.library_id for lib in model_mgr.get_libraries()]
        assert "a1" not in lib_ids
        assert "a2" in lib_ids

    def test_unregister_emits_changed(self, model_mgr, tmp_path):
        lib = ModelLibrary(
            library_id="addon",
            display_name="Addon",
            path=tmp_path,
        )
        model_mgr.add_library(lib, addon_name="my_addon")

        signals = []

        def on_changed(sender, **kwargs):
            signals.append(sender)

        model_mgr.changed.connect(on_changed)
        model_mgr.unregister_all_from_addon("my_addon")
        assert len(signals) == 1

    def test_unregister_no_signal_when_nothing_removed(self, model_mgr):
        signals = []

        def on_changed(sender, **kwargs):
            signals.append(sender)

        model_mgr.changed.connect(on_changed)
        model_mgr.unregister_all_from_addon("nonexistent_addon")
        assert len(signals) == 0

    def test_resolve_searches_libraries_in_order(self, model_mgr, tmp_path):
        addon_dir = tmp_path / "addon"
        addon_dir.mkdir()
        (addon_dir / "test.glb").write_text("addon")
        model_mgr.add_library(
            ModelLibrary(
                library_id="addon",
                display_name="Test",
                path=addon_dir,
                read_only=True,
            )
        )
        m = Model(name="test", path=Path("test.glb"))
        result = model_mgr.resolve(m)
        assert result is not None
        assert result.read_text() == "addon"


class TestModelManagerDevicePackageRegistration:
    def test_device_package_models_registered_as_library(
        self, model_mgr, tmp_path
    ):
        from rayforge.machine.device.manager import DevicePackageManager
        from rayforge.machine.device.package import DIALECT_FILENAME

        device_dir = tmp_path / "devices" / "test-device"
        device_dir.mkdir(parents=True)
        (device_dir / "device.yaml").write_text(
            "api_version: 1\ndevice:\n  name: Test Device\n"
        )
        _write_full_dialect(device_dir / DIALECT_FILENAME)

        models_dir = device_dir / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "head.glb").write_text("glb data")

        pkg_mgr = DevicePackageManager([tmp_path / "devices"])
        pkg_mgr.discover(context=_FakeContext(model_mgr))

        libs = model_mgr.get_libraries()
        lib_ids = [lib.library_id for lib in libs]
        assert "device:Test Device" in lib_ids

        m = Model(name="head", path=Path("head.glb"))
        result = model_mgr.resolve(m)
        assert result is not None
        assert result.name == "head.glb"

    def test_device_without_models_dir_not_registered(
        self, model_mgr, tmp_path
    ):
        from rayforge.machine.device.manager import DevicePackageManager
        from rayforge.machine.device.package import DIALECT_FILENAME

        device_dir = tmp_path / "devices" / "bare-device"
        device_dir.mkdir(parents=True)
        (device_dir / "device.yaml").write_text(
            "api_version: 1\ndevice:\n  name: Bare Device\n"
        )
        _write_full_dialect(device_dir / DIALECT_FILENAME)

        pkg_mgr = DevicePackageManager([tmp_path / "devices"])
        pkg_mgr.discover(context=_FakeContext(model_mgr))

        lib_ids = [lib.library_id for lib in model_mgr.get_libraries()]
        assert "device:Bare Device" not in lib_ids

    def test_discover_without_context(self, tmp_path):
        from rayforge.machine.device.manager import DevicePackageManager
        from rayforge.machine.device.package import DIALECT_FILENAME

        device_dir = tmp_path / "devices" / "test-device"
        device_dir.mkdir(parents=True)
        (device_dir / "device.yaml").write_text(
            "api_version: 1\ndevice:\n  name: Test Device\n"
        )
        _write_full_dialect(device_dir / DIALECT_FILENAME)

        models_dir = device_dir / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "head.glb").write_text("glb data")

        pkg_mgr = DevicePackageManager([tmp_path / "devices"])
        packages = pkg_mgr.discover()
        assert len(packages) == 1


def _write_full_dialect(path):
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(
            {
                "laser_on": "M4 S{power:.0f}",
                "laser_off": "M5",
                "focus_laser_on": "M4 S{power:.0f}",
                "tool_change": "T{tool_number}",
                "set_speed": "",
                "travel_move": "G0 X{x} Y{y}",
                "linear_move": "G1 X{x} Y{y}{f_command}",
                "arc_cw": "G2 X{x} Y{y} I{i} J{j}{f_command}",
                "arc_ccw": "G3 X{x} Y{y} I{i} J{j}{f_command}",
                "bezier_cubic": "",
                "air_assist_on": "M8",
                "air_assist_off": "M9",
                "home_all": "$H",
                "home_axis": "$H{axis_letter}",
                "move_to": "$J=G90 G21 F{speed} X{x} Y{y}",
                "jog": "$J=G91 G21 F{speed}",
                "clear_alarm": "$X",
                "set_wcs_offset": "G10 L2 P{p_num} X{x} Y{y} Z{z}",
                "probe_cycle": "G38.2 {axis_letter}{max_travel} F{feed_rate}",
                "preamble": ["G21", "G90"],
                "postscript": ["M5", "G0 X0 Y0"],
                "inject_wcs_after_preamble": True,
                "omit_unchanged_coords": True,
            },
            f,
            sort_keys=False,
        )


class _FakeContext:
    def __init__(self, model_mgr):
        self.model_mgr = model_mgr
