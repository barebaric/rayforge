import base64
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from rayforge.core.source_asset import SourceAsset
from rayforge.image.registry import renderer_registry
from rayforge.image.svg.renderer import SvgRenderer


@pytest.fixture
def basic_asset() -> SourceAsset:
    """Provides a basic SourceAsset."""
    return SourceAsset(
        source_file=Path("test.svg"),
        original_data=b"SVG_DATA",
        renderer=SvgRenderer(),
        metadata={"key": "value"},
    )


class TestSourceAsset:
    def test_initialization(self, basic_asset: SourceAsset):
        """Tests basic initialization of a SourceAsset."""
        asset = basic_asset
        assert isinstance(asset.uid, str)
        assert asset.source_file == Path("test.svg")
        assert asset.name == "test.svg"
        assert asset.original_data == b"SVG_DATA"
        assert asset.base_render_data is None
        assert isinstance(asset.renderer, SvgRenderer)
        assert asset.metadata == {"key": "value"}

    def test_initialization_with_render_data(self):
        """Tests initialization with specific base_render_data."""
        asset = SourceAsset(
            source_file=Path("test.svg"),
            original_data=b"ORIGINAL",
            base_render_data=b"TRIMMED",
            renderer=SvgRenderer(),
        )
        assert asset.original_data == b"ORIGINAL"
        assert asset.base_render_data == b"TRIMMED"

    def test_serialization(self, basic_asset: SourceAsset):
        """Tests serialization to a dictionary."""
        basic_asset.base_render_data = b"RENDER_DATA"
        data_dict = basic_asset.to_dict()

        assert data_dict["uid"] == basic_asset.uid
        assert data_dict["type"] == "source"
        assert data_dict["name"] == "test.svg"
        assert data_dict["source_file"] == "test.svg"
        assert data_dict["original_data"] == base64.b64encode(
            b"SVG_DATA"
        ).decode("utf-8")
        assert data_dict["base_render_data"] == base64.b64encode(
            b"RENDER_DATA"
        ).decode("utf-8")
        assert data_dict["renderer_name"] == "SvgRenderer"
        assert data_dict["metadata"] == {"key": "value"}

    def test_deserialization(self):
        """Tests deserialization from a dictionary."""
        data_dict = {
            "uid": "test-uid",
            "type": "source",
            "name": "test.png",
            "source_file": "test.png",
            "original_data": base64.b64encode(b"PNG_DATA").decode("utf-8"),
            "base_render_data": base64.b64encode(b"RENDER_DATA").decode(
                "utf-8"
            ),
            "renderer_name": "SvgRenderer",
            "metadata": {"natural_size": [100, 50]},
        }
        with patch.object(
            renderer_registry,
            "_renderers",
            {"SvgRenderer": SvgRenderer()},
        ):
            new_asset = SourceAsset.from_dict(data_dict)

        assert new_asset.uid == "test-uid"
        assert new_asset.name == "test.png"
        assert new_asset.original_data == b"PNG_DATA"
        assert new_asset.base_render_data == b"RENDER_DATA"
        assert isinstance(new_asset.renderer, SvgRenderer)
        assert new_asset.metadata == {"natural_size": [100, 50]}

    def test_roundtrip_serialization(self, basic_asset: SourceAsset):
        """Tests that an asset can be serialized and deserialized back to an
        equivalent object."""
        basic_asset.base_render_data = b"RENDER_DATA"
        data_dict = basic_asset.to_dict()

        with patch.object(
            renderer_registry,
            "_renderers",
            {"SvgRenderer": SvgRenderer()},
        ):
            restored_asset = SourceAsset.from_dict(data_dict)

        assert restored_asset.uid == basic_asset.uid
        assert restored_asset.name == basic_asset.name
        assert restored_asset.source_file == basic_asset.source_file
        assert restored_asset.original_data == basic_asset.original_data
        assert restored_asset.base_render_data == basic_asset.base_render_data
        assert isinstance(restored_asset.renderer, SvgRenderer)
        assert restored_asset.metadata == basic_asset.metadata


class TestSourceFileFingerprint:
    """The fingerprint identifies the file on disk an asset came from."""

    def test_defaults_to_the_embedded_data(self, basic_asset):
        data = basic_asset.original_data
        assert (
            basic_asset.source_file_sha256 == hashlib.sha256(data).hexdigest()
        )
        assert basic_asset.source_file_size == len(data)
        assert basic_asset.matches_source_file(data)
        assert not basic_asset.matches_source_file(data + b" ")

    def test_converted_source_remembers_the_original_file(self, basic_asset):
        original = b"%!PS-Adobe-3.0 EPSF-3.0\n...original eps bytes..."
        basic_asset.set_source_file_fingerprint(original)

        assert (
            basic_asset.source_file_sha256
            == hashlib.sha256(original).hexdigest()
        )
        assert basic_asset.source_file_size == len(original)
        assert basic_asset.matches_source_file(original)
        assert not basic_asset.matches_source_file(basic_asset.original_data)

    def test_fingerprint_survives_serialization(self, basic_asset):
        original = b"original file bytes"
        basic_asset.set_source_file_fingerprint(original)
        restored = SourceAsset.from_dict(basic_asset.to_dict())
        assert restored.matches_source_file(original)
        assert restored.source_file_size == len(original)

    def test_projects_without_fingerprint_still_load(self, basic_asset):
        data = basic_asset.to_dict()
        data["metadata"] = {}
        restored = SourceAsset.from_dict(data)
        assert restored.matches_source_file(basic_asset.original_data)
