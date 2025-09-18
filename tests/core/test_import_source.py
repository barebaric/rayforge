import pytest
from pathlib import Path
from rayforge.core.vectorization_config import TraceConfig
from rayforge.core.import_source import ImportSource


@pytest.fixture
def source_with_config() -> ImportSource:
    """Provides an ImportSource that requires vectorization."""
    return ImportSource(
        source_file=Path("test.png"),
        data=b"PNG_DATA",
        vector_config=TraceConfig(threshold=0.8),
    )


@pytest.fixture
def source_without_config() -> ImportSource:
    """Provides an ImportSource that does not require vectorization."""
    return ImportSource(source_file=Path("test.svg"), data=b"SVG_DATA")


class TestImportSource:
    def test_init_with_config(self, source_with_config: ImportSource):
        """Tests initialization with a vectorization config."""
        src = source_with_config
        assert isinstance(src.uid, str)
        assert src.source_file == Path("test.png")
        assert src.data == b"PNG_DATA"
        assert src.vector_config is not None
        assert isinstance(src.vector_config, TraceConfig)
        assert src.vector_config.threshold == 0.8

    def test_init_without_config(self, source_without_config: ImportSource):
        """Tests initialization without a vectorization config."""
        src = source_without_config
        assert src.source_file == Path("test.svg")
        assert src.data == b"SVG_DATA"
        assert src.vector_config is None

    def test_serialization_with_config(self, source_with_config: ImportSource):
        """Tests serialization for a source with a config."""
        data_dict = source_with_config.to_dict()

        assert data_dict["uid"] == source_with_config.uid
        assert data_dict["source_file"] == "test.png"
        assert data_dict["data"] == b"PNG_DATA"
        assert data_dict["vector_config"] == {"threshold": 0.8}

    def test_serialization_without_config(
        self, source_without_config: ImportSource
    ):
        """Tests serialization for a source without a config."""
        data_dict = source_without_config.to_dict()
        assert data_dict["uid"] == source_without_config.uid
        assert data_dict["vector_config"] is None

    def test_deserialization_with_config(self):
        """Tests deserialization for a source with a config."""
        data_dict = {
            "uid": "test-uid",
            "source_file": "test.png",
            "data": b"PNG_DATA",
            "vector_config": {"threshold": 0.9},
        }
        new_src = ImportSource.from_dict(data_dict)
        assert new_src.uid == "test-uid"
        assert isinstance(new_src.vector_config, TraceConfig)
        assert new_src.vector_config.threshold == 0.9

    def test_deserialization_without_config(self):
        """Tests deserialization for a source without a config."""
        data_dict = {
            "uid": "test-uid",
            "source_file": "test.svg",
            "data": b"SVG_DATA",
            "vector_config": None,
        }
        new_src = ImportSource.from_dict(data_dict)
        assert new_src.uid == "test-uid"
        assert new_src.vector_config is None
