import pytest
from unittest.mock import MagicMock
from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset
from rayforge.core.matrix import Matrix
from rayforge.core.doc import Doc
from rayforge.core.geo import Geometry


@pytest.fixture
def mock_asset():
    """Provides a mock StockAsset."""
    asset = MagicMock(spec=StockAsset)
    asset.uid = "asset-123"
    asset.thickness = 5.0
    asset.material_uid = "wood"
    asset.geometry = Geometry()
    asset.display_icon_name = "stock-symbolic"
    asset.get_natural_size.return_value = (100.0, 50.0)
    return asset


@pytest.fixture
def mock_doc(mock_asset):
    """Provides a mock Doc that can retrieve the mock asset."""
    doc = MagicMock(spec=Doc)
    doc.get_asset_by_uid.return_value = mock_asset
    # A real Doc's .doc property returns self. The mock must do the same.
    doc.doc = doc
    return doc


def test_stock_item_initialization():
    """Tests that a StockItem initializes with an asset UID."""
    item = StockItem(stock_asset_uid="asset-abc", name="My Stock")
    assert item.name == "My Stock"
    assert item.stock_asset_uid == "asset-abc"
    assert item.matrix.is_identity()
    assert item.visible is True


def test_stock_item_property_delegation(mock_doc, mock_asset):
    """Tests that properties correctly delegate to the associated asset."""
    item = StockItem(stock_asset_uid="asset-123")
    item.parent = mock_doc  # Set parent to allow .doc -> .stock_asset to work

    assert item.stock_asset is mock_asset
    assert item.thickness == 5.0
    assert item.material_uid == "wood"
    assert item.geometry is mock_asset.geometry
    assert item.display_icon_name == "stock-symbolic"
    assert item.get_default_size() == (100.0, 50.0)


def test_stock_item_to_dict_serialization(mock_doc):
    """Tests serializing a StockItem to a dictionary."""
    item = StockItem(stock_asset_uid="asset-123", name="Serialized Stock")
    item.matrix = Matrix.translation(5, 15)
    item.visible = False
    item.parent = mock_doc

    data = item.to_dict()

    assert data["type"] == "stockitem"
    assert data["uid"] == item.uid
    assert data["name"] == "Serialized Stock"
    assert data["matrix"] == Matrix.translation(5, 15).to_list()
    assert data["stock_asset_uid"] == "asset-123"
    assert data["visible"] is False


def test_stock_item_from_dict_deserialization():
    """Tests creating a StockItem instance from a dictionary."""
    item_dict = {
        "uid": "item-uid-456",
        "type": "stockitem",
        "name": "Deserialized Stock",
        "matrix": [[1, 0, 10], [0, 1, 20], [0, 0, 1]],
        "stock_asset_uid": "asset-xyz",
        "visible": True,
    }

    item = StockItem.from_dict(item_dict)

    assert isinstance(item, StockItem)
    assert item.uid == "item-uid-456"
    assert item.name == "Deserialized Stock"
    assert item.matrix == Matrix.translation(10, 20)
    assert item.stock_asset_uid == "asset-xyz"
    assert item.visible is True
