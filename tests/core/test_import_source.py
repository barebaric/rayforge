import pytest
from pathlib import Path
from unittest.mock import patch
from rayforge.core.vectorization_config import TraceConfig
from rayforge.core.import_source import ImportSource
from rayforge.image.svg.renderer import SvgRenderer


@pytest.fixture
def source_with_config() -> ImportSource:
    """Provides an ImportSource that requires vectorization."""
    return ImportSource(
        source_file=Path("test.png"),
        original_data=b"PNG_DATA",
        renderer=SvgRenderer(),
        vector_config=TraceConfig(threshold=0.8),
        metadata={"is_bilevel": False},
    )


@pytest.fixture
def source_without_config() -> ImportSource:
    """Provides an ImportSource that does not require vectorization."""
    return ImportSource(
        source_file=Path("test.svg"),
        original_data=b"SVG_DATA",
        renderer=SvgRenderer(),
    )


class TestImportSource:
    def test_init_with_config(self, source_with_config: ImportSource):
        """Tests initialization with a vectorization config."""
        src = source_with_config
        assert isinstance(src.uid, str)
        assert src.source_file == Path("test.png")
        assert src.original_data == b"PNG_DATA"
        assert src.working_data == b"PNG_DATA"
        assert src.data == b"PNG_DATA"  # Check property
        assert isinstance(src.renderer, SvgRenderer)
        assert src.vector_config is not None
        assert isinstance(src.vector_config, TraceConfig)
        assert src.vector_config.threshold == 0.8
        assert src.metadata == {"is_bilevel": False}

    def test_init_without_config(self, source_without_config: ImportSource):
        """Tests initialization without a vectorization config."""
        src = source_without_config
        assert src.source_file == Path("test.svg")
        assert src.original_data == b"SVG_DATA"
        assert src.working_data == b"SVG_DATA"
        assert src.data == b"SVG_DATA"
        assert isinstance(src.renderer, SvgRenderer)
        assert src.vector_config is None
        assert src.metadata == {}

    def test_data_setter(self, source_without_config: ImportSource):
        """Tests that the data property setter modifies working_data."""
        src = source_without_config
        src.data = b"MODIFIED_SVG_DATA"
        assert src.original_data == b"SVG_DATA"
        assert src.working_data == b"MODIFIED_SVG_DATA"
        assert src.data == b"MODIFIED_SVG_DATA"

    def test_serialization_with_config(self, source_with_config: ImportSource):
        """Tests serialization for a source with a config."""
        data_dict = source_with_config.to_dict()

        assert data_dict["uid"] == source_with_config.uid
        assert data_dict["source_file"] == "test.png"
        assert data_dict["original_data"] == b"PNG_DATA"
        assert data_dict["working_data"] == b"PNG_DATA"
        assert data_dict["renderer_name"] == "SvgRenderer"
        assert data_dict["vector_config"] == {"threshold": 0.8}
        assert data_dict["metadata"] == {"is_bilevel": False}

    def test_serialization_with_modified_data(
        self, source_without_config: ImportSource
    ):
        """Tests serialization for a source without a config."""
        source_without_config.data = b"MODIFIED_DATA"
        data_dict = source_without_config.to_dict()
        assert data_dict["uid"] == source_without_config.uid
        assert data_dict["original_data"] == b"SVG_DATA"
        assert data_dict["working_data"] == b"MODIFIED_DATA"
        assert data_dict["renderer_name"] == "SvgRenderer"
        assert data_dict["vector_config"] is None
        assert data_dict["metadata"] == {}

    def test_deserialization_with_config(self):
        """Tests deserialization for a source with a config."""
        data_dict = {
            "uid": "test-uid",
            "source_file": "test.png",
            "original_data": b"PNG_DATA",
            "working_data": b"MODIFIED_PNG_DATA",
            "renderer_name": "SvgRenderer",
            "vector_config": {"threshold": 0.9},
            "metadata": {"natural_size": [100, 50]},
        }
        with patch.dict(
            "rayforge.image.renderer_by_name",
            {"SvgRenderer": SvgRenderer()},
            clear=True,
        ):
            new_src = ImportSource.from_dict(data_dict)

        assert new_src.uid == "test-uid"
        assert new_src.original_data == b"PNG_DATA"
        assert new_src.working_data == b"MODIFIED_PNG_DATA"
        assert isinstance(new_src.renderer, SvgRenderer)
        assert isinstance(new_src.vector_config, TraceConfig)
        assert new_src.vector_config.threshold == 0.9
        assert new_src.metadata == {"natural_size": [100, 50]}

    def test_deserialization_without_working_data(self):
        """
        Tests deserialization when working_data is missing, relying on
        __post_init__ to populate it.
        """
        data_dict = {
            "uid": "test-uid",
            "source_file": "test.svg",
            "original_data": b"SVG_DATA",
            "renderer_name": "SvgRenderer",
            "vector_config": None,
            "metadata": {},
        }
        with patch.dict(
            "rayforge.image.renderer_by_name",
            {"SvgRenderer": SvgRenderer()},
            clear=True,
        ):
            new_src = ImportSource.from_dict(data_dict)

        assert new_src.uid == "test-uid"
        assert new_src.original_data == b"SVG_DATA"
        assert new_src.working_data == b"SVG_DATA"
        assert isinstance(new_src.renderer, SvgRenderer)
        assert new_src.vector_config is None
        assert new_src.metadata == {}

    def test_deserialization_backward_compatible(self):
        """
        Tests deserialization for a source saved in the old format with only
        a 'data' key.
        """
        data_dict = {
            "uid": "test-uid",
            "source_file": "test.svg",
            "data": b"OLD_SVG_DATA",
            "renderer_name": "SvgRenderer",
            "vector_config": None,
            "metadata": {},
        }
        with patch.dict(
            "rayforge.image.renderer_by_name",
            {"SvgRenderer": SvgRenderer()},
            clear=True,
        ):
            new_src = ImportSource.from_dict(data_dict)

        assert new_src.uid == "test-uid"
        assert new_src.original_data == b"OLD_SVG_DATA"
        assert new_src.working_data == b"OLD_SVG_DATA"
        assert new_src.vector_config is None
