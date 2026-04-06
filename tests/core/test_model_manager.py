import pytest
from pathlib import Path

from rayforge.core.model import Model, ModelCategory, ModelLibrary
from rayforge.core.model_manager import ModelManager, CATEGORIES


MACHINES = ModelCategory(
    id="machines",
    display_name="Machines",
    description="Complete machine models.",
)
ROTARY = ModelCategory(
    id="rotary",
    display_name="Rotary",
    description="Rotary and workholding models.",
)


@pytest.fixture
def model_mgr(tmp_path):
    user_dir = tmp_path / "models"
    return ModelManager(user_dir)


class TestModel:
    def test_basic_creation(self):
        m = Model(name="Test Chuck", path=Path("rotary/test.glb"))
        assert m.name == "Test Chuck"
        assert m.path == Path("rotary/test.glb")
        assert m.description == ""
        assert m.file_path is None
        assert m.extra == {}

    def test_creation_with_all_fields(self):
        m = Model(
            name="Chuck",
            path=Path("rotary/a.glb"),
            description="A 4-jaw chuck",
            file_path=Path("/absolute/path.glb"),
            extra={"scale": 1.0},
        )
        assert m.name == "Chuck"
        assert m.description == "A 4-jaw chuck"
        assert m.file_path == Path("/absolute/path.glb")
        assert m.extra == {"scale": 1.0}

    def test_from_dict(self):
        data = {
            "name": "Standard Chuck",
            "path": "rotary/standard.glb",
            "description": "A standard 3-jaw chuck",
        }
        m = Model.from_dict(data)
        assert m.name == "Standard Chuck"
        assert m.path == Path("rotary/standard.glb")
        assert m.description == "A standard 3-jaw chuck"

    def test_from_dict_with_extra(self):
        data = {
            "name": "Chuck",
            "path": "rotary/test.glb",
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
            path=Path("rotary/a.glb"),
            description="desc",
            extra={"scale": 1.0},
        )
        d = m.to_dict()
        assert d["name"] == "Chuck"
        assert d["path"] == "rotary/a.glb"
        assert d["description"] == "desc"
        assert d["scale"] == 1.0

    def test_roundtrip(self):
        original = Model(
            name="Chuck",
            path=Path("rotary/a.glb"),
            description="desc",
            extra={"key": "value"},
        )
        restored = Model.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.path == original.path
        assert restored.description == original.description
        assert restored.extra == original.extra

    def test_str(self):
        m = Model(name="Chuck", path=Path("rotary/a.glb"))
        assert "Chuck" in str(m)
        assert "rotary/a.glb" in str(m)


class TestModelCategory:
    def test_str_returns_display_name(self):
        cat = ModelCategory(id="machines", display_name="Machines")
        assert str(cat) == "Machines"

    def test_equality(self):
        a = ModelCategory(id="x", display_name="X")
        b = ModelCategory(id="x", display_name="X")
        assert a == b

    def test_inequality(self):
        a = ModelCategory(id="x", display_name="X")
        b = ModelCategory(id="y", display_name="Y")
        assert a != b


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
    def test_creates_user_dir(self, tmp_path):
        user_dir = tmp_path / "models"
        assert not user_dir.exists()
        ModelManager(user_dir)
        assert user_dir.is_dir()

    def test_str_representation(self, model_mgr, tmp_path):
        result = str(model_mgr)
        assert "ModelManager" in result
        assert str(tmp_path / "models") in result

    def test_default_user_library(self, model_mgr):
        libs = model_mgr.get_libraries()
        assert len(libs) == 1
        assert libs[0].library_id == "user"
        assert libs[0].read_only is False


class TestModelManagerGetCategories:
    def test_returns_fixed_categories(self, model_mgr):
        categories = model_mgr.get_categories()
        assert len(categories) == 3
        ids = [c.id for c in categories]
        assert "machines" in ids
        assert "rotary" in ids
        assert "heads" in ids

    def test_categories_are_modelcategory_instances(self, model_mgr):
        for cat in model_mgr.get_categories():
            assert isinstance(cat, ModelCategory)

    def test_matches_module_constant(self, model_mgr):
        categories = model_mgr.get_categories()
        assert len(categories) == len(CATEGORIES)
        for cat, expected in zip(categories, CATEGORIES):
            assert cat.id == expected.id


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


class TestModelManagerResolveUserDir:
    def test_finds_file_in_user_dir(self, model_mgr):
        rotary_dir = model_mgr.user_dir / "rotary"
        rotary_dir.mkdir(parents=True)
        model_file = rotary_dir / "custom_chuck.glb"
        model_file.write_text("fake glb data")

        m = Model(name="custom", path=Path("rotary/custom_chuck.glb"))
        result = model_mgr.resolve(m)
        assert result == model_file

    def test_returns_none_when_not_found(self, model_mgr):
        m = Model(name="test", path=Path("rotary/nonexistent.glb"))
        result = model_mgr.resolve(m)
        assert result is None

    def test_finds_file_with_models_prefix(self, model_mgr):
        rotary_dir = model_mgr.user_dir / "models" / "rotary"
        rotary_dir.mkdir(parents=True)
        model_file = rotary_dir / "test.glb"
        model_file.write_text("data")

        m = Model(name="test", path=Path("models/rotary/test.glb"))
        result = model_mgr.resolve(m)
        assert result == model_file


class TestModelManagerResolveBundled:
    def test_finds_in_registered_library(self, model_mgr, tmp_path):
        bundled_root = tmp_path / "bundled"
        rotary_dir = bundled_root / "rotary"
        rotary_dir.mkdir(parents=True)
        model_file = rotary_dir / "standard.glb"
        model_file.write_text("bundled model")

        model_mgr.add_library(
            ModelLibrary(
                library_id="test_core",
                display_name="Test Core",
                path=bundled_root,
                read_only=True,
            )
        )

        m = Model(name="standard", path=Path("rotary/standard.glb"))
        result = model_mgr.resolve(m)
        assert result is not None
        assert result.name == "standard.glb"


class TestModelManagerResolvePrecedence:
    def test_user_dir_takes_precedence(self, model_mgr, tmp_path):
        user_rotary = model_mgr.user_dir / "rotary"
        user_rotary.mkdir(parents=True)
        user_file = user_rotary / "chuck.glb"
        user_file.write_text("user version")

        addon_root = tmp_path / "addon"
        addon_rotary = addon_root / "rotary"
        addon_rotary.mkdir(parents=True)
        (addon_rotary / "chuck.glb").write_text("addon version")

        model_mgr.add_library(
            ModelLibrary(
                library_id="addon",
                display_name="Addon",
                path=addon_root,
                read_only=True,
            )
        )

        m = Model(name="chuck", path=Path("rotary/chuck.glb"))
        result = model_mgr.resolve(m)
        assert result == user_file


class TestModelManagerGetModels:
    def test_lists_models_in_category(self, model_mgr):
        rotary_dir = model_mgr.user_dir / "rotary"
        rotary_dir.mkdir(parents=True)
        (rotary_dir / "a.glb").write_text("a")
        (rotary_dir / "b.glb").write_text("b")

        user_lib = model_mgr.get_libraries()[0]
        models = model_mgr.get_models(user_lib, ROTARY)
        assert len(models) == 2
        names = {m.name for m in models}
        assert names == {"a", "b"}

    def test_returns_empty_for_empty_dir(self, model_mgr):
        user_lib = model_mgr.get_libraries()[0]
        models = model_mgr.get_models(user_lib, ROTARY)
        assert models == []

    def test_ignores_hidden_files(self, model_mgr):
        rotary_dir = model_mgr.user_dir / "rotary"
        rotary_dir.mkdir(parents=True)
        (rotary_dir / "visible.glb").write_text("v")
        (rotary_dir / ".hidden.glb").write_text("h")

        user_lib = model_mgr.get_libraries()[0]
        models = model_mgr.get_models(user_lib, ROTARY)
        assert len(models) == 1
        assert models[0].name == "visible"

    def test_get_models_from_registered_library(self, model_mgr, tmp_path):
        addon_dir = tmp_path / "addon"
        rotary = addon_dir / "rotary"
        rotary.mkdir(parents=True)
        (rotary / "x.glb").write_text("x")
        (rotary / "y.glb").write_text("y")
        lib = ModelLibrary(
            library_id="addon",
            display_name="Test",
            path=addon_dir,
            read_only=True,
        )
        model_mgr.add_library(lib)

        models = model_mgr.get_models(lib, ROTARY)
        names = [m.name for m in models]
        assert "x" in names
        assert "y" in names


class TestModelManagerGetAllModels:
    def test_returns_user_models(self, model_mgr):
        rotary_dir = model_mgr.user_dir / "rotary"
        rotary_dir.mkdir()
        (rotary_dir / "a.glb").write_text("a")
        (rotary_dir / "b.glb").write_text("b")

        models = model_mgr.get_all_models(ROTARY)
        assert len(models) == 2
        names = [m.name for m in models]
        assert "a" in names
        assert "b" in names

    def test_returns_empty_for_category_with_no_files(self, model_mgr):
        models = model_mgr.get_all_models(MACHINES)
        assert models == []

    def test_deduplicates_across_libraries(self, model_mgr, tmp_path):
        rotary_dir = model_mgr.user_dir / "rotary"
        rotary_dir.mkdir()
        (rotary_dir / "standard.glb").write_text("user")

        addon_root = tmp_path / "addon"
        addon_rotary = addon_root / "rotary"
        addon_rotary.mkdir(parents=True)
        (addon_rotary / "standard.glb").write_text("addon")
        (addon_rotary / "extra.glb").write_text("addon extra")

        model_mgr.add_library(
            ModelLibrary(
                library_id="addon",
                display_name="Addon",
                path=addon_root,
                read_only=True,
            )
        )

        models = model_mgr.get_all_models(ROTARY)
        filenames = [m.path.name for m in models]
        assert "standard.glb" in filenames
        assert filenames.count("standard.glb") == 1
        assert "extra.glb" in filenames


class TestModelManagerGetUserModels:
    def test_lists_models_in_subdir(self, model_mgr):
        rotary_dir = model_mgr.user_dir / "rotary"
        rotary_dir.mkdir(parents=True)
        (rotary_dir / "a.glb").write_text("a")
        (rotary_dir / "b.glb").write_text("b")

        models = model_mgr.get_user_models(Path("rotary"))
        assert len(models) == 2
        names = {m.name for m in models}
        assert names == {"a", "b"}
        for m in models:
            assert m.file_path is not None

    def test_returns_empty_for_nonexistent_dir(self, model_mgr):
        models = model_mgr.get_user_models(Path("nonexistent"))
        assert models == []

    def test_ignores_subdirectories(self, model_mgr):
        rotary_dir = model_mgr.user_dir / "rotary"
        rotary_dir.mkdir(parents=True)
        (rotary_dir / "model.glb").write_text("m")
        (rotary_dir / "subdir").mkdir()

        models = model_mgr.get_user_models(Path("rotary"))
        assert len(models) == 1
        assert models[0].name == "model"


class TestModelManagerAddModel:
    def test_add_model_copies_file(self, model_mgr, tmp_path):
        src = tmp_path / "source.glb"
        src.write_text("model data")

        m = Model(
            name="my_chuck",
            path=Path("rotary/my_chuck.glb"),
        )
        result = model_mgr.add_model(src, m)
        assert result.file_path.is_file()
        assert result.file_path.read_text() == "model data"
        assert result.name == "my_chuck"
        assert result.path == Path("rotary/my_chuck.glb")

    def test_add_model_emits_changed(self, model_mgr, tmp_path):
        src = tmp_path / "source.glb"
        src.write_text("data")

        signals = []

        def on_changed(sender, **kwargs):
            signals.append(sender)

        model_mgr.changed.connect(on_changed)
        m = Model(name="test", path=Path("test.glb"))
        model_mgr.add_model(src, m)
        assert len(signals) == 1

    def test_add_model_creates_parent_dirs(self, model_mgr, tmp_path):
        src = tmp_path / "source.glb"
        src.write_text("data")

        m = Model(
            name="deep",
            path=Path("deep/nested/dir/model.glb"),
        )
        result = model_mgr.add_model(src, m)
        assert result.file_path.is_file()


class TestModelManagerRemoveModel:
    def test_remove_existing_model(self, model_mgr):
        model_file = model_mgr.user_dir / "test.glb"
        model_file.write_text("data")

        m = Model(name="test", path=Path("test.glb"))
        result = model_mgr.remove_model(m)
        assert result is True
        assert not model_file.exists()

    def test_remove_nonexistent_model(self, model_mgr):
        m = Model(name="test", path=Path("nonexistent.glb"))
        result = model_mgr.remove_model(m)
        assert result is False

    def test_remove_model_emits_changed(self, model_mgr):
        model_file = model_mgr.user_dir / "test.glb"
        model_file.write_text("data")

        signals = []

        def on_changed(sender, **kwargs):
            signals.append(sender)

        model_mgr.changed.connect(on_changed)
        m = Model(name="test", path=Path("test.glb"))
        model_mgr.remove_model(m)
        assert len(signals) == 1

    def test_remove_does_not_emit_on_failure(self, model_mgr):
        signals = []

        def on_changed(sender, **kwargs):
            signals.append(sender)

        model_mgr.changed.connect(on_changed)
        m = Model(name="test", path=Path("nonexistent.glb"))
        model_mgr.remove_model(m)
        assert len(signals) == 0


class TestModelManagerIsUserModel:
    def test_user_model(self, model_mgr):
        fp = model_mgr.user_dir / "test.glb"
        m = Model(name="test", path=Path("test.glb"), file_path=fp)
        assert model_mgr.is_user_model(m) is True

    def test_non_user_model(self, model_mgr):
        m = Model(
            name="test",
            path=Path("test.glb"),
            file_path=Path("/other/path/test.glb"),
        )
        assert model_mgr.is_user_model(m) is False

    def test_no_file_path(self, model_mgr):
        m = Model(name="test", path=Path("test.glb"))
        assert model_mgr.is_user_model(m) is False


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
        assert len(libs) == 2
        assert libs[1].library_id == "test_addon"

    def test_add_duplicate_library_fails(self, model_mgr, tmp_path):
        lib = ModelLibrary(
            library_id="user",
            display_name="Duplicate",
            path=tmp_path,
        )
        assert model_mgr.add_library(lib) is False

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

    def test_unregister_preserves_user_library(self, model_mgr, tmp_path):
        lib = ModelLibrary(
            library_id="addon",
            display_name="Addon",
            path=tmp_path,
        )
        model_mgr.add_library(lib, addon_name="my_addon")
        model_mgr.unregister_all_from_addon("my_addon")
        lib_ids = [lib.library_id for lib in model_mgr.get_libraries()]
        assert "user" in lib_ids

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
        (addon_dir / "rotary").mkdir(parents=True)
        (addon_dir / "rotary" / "test.glb").write_text("addon")
        model_mgr.add_library(
            ModelLibrary(
                library_id="addon",
                display_name="Test",
                path=addon_dir,
                read_only=True,
            )
        )
        m = Model(name="test", path=Path("rotary/test.glb"))
        result = model_mgr.resolve(m)
        assert result is not None
        assert result.read_text() == "addon"
