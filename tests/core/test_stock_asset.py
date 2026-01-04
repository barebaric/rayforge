from unittest.mock import MagicMock
from rayforge.core.stock_asset import StockAsset
from rayforge.core.geo import Geometry


def test_stock_asset_initialization():
    """Tests that a StockAsset initializes correctly."""
    asset = StockAsset(name="My Asset")
    assert asset.name == "My Asset"
    assert isinstance(asset.geometry, Geometry)
    assert asset.geometry.is_empty()


def test_stock_asset_initialization_with_geometry():
    """Tests initializing a StockAsset with a pre-existing Geometry object."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(100, 0)
    geo.line_to(100, 50)
    geo.line_to(0, 50)
    geo.close_path()

    asset = StockAsset(geometry=geo)
    assert asset.geometry is geo
    assert len(asset.geometry) == 5


def test_stock_asset_to_dict_serialization():
    """Tests serializing a StockAsset to a dictionary."""
    geo = Geometry()
    geo.move_to(10, 20)
    geo.line_to(30, 40)

    asset = StockAsset(name="Serialized Asset", geometry=geo)
    asset.thickness = 12.5
    asset.material_uid = "wood-mahogany"
    asset_uid = asset.uid

    data = asset.to_dict()

    assert data["uid"] == asset_uid
    assert data["name"] == "Serialized Asset"
    assert data["thickness"] == 12.5
    assert data["material_uid"] == "wood-mahogany"
    assert "commands" in data["geometry"]
    assert len(data["geometry"]["commands"]) == 2


def test_stock_asset_from_dict_deserialization():
    """Tests creating a StockAsset instance from a dictionary."""
    asset_dict = {
        "uid": "test-uid-123",
        "name": "Deserialized Asset",
        "thickness": 3.0,
        "material_uid": "mdf-3mm",
        "geometry": {
            "commands": [["M", 0.0, 0.0, 0.0]],
            "last_move_to": [0.0, 0.0, 0.0],
        },
        "type": "stock",
    }

    asset = StockAsset.from_dict(asset_dict)

    assert isinstance(asset, StockAsset)
    assert asset.uid == "test-uid-123"
    assert asset.name == "Deserialized Asset"
    assert asset.thickness == 3.0
    assert asset.material_uid == "mdf-3mm"
    assert len(asset.geometry) == 1


def test_property_setters_fire_updated_signal():
    """Tests that property setters fire the 'updated' signal."""
    asset = StockAsset(name="Signal Test")
    handler = MagicMock()
    asset.updated.connect(handler)

    asset.name = "New Name"
    handler.assert_called_once_with(asset)
    handler.reset_mock()

    asset.set_thickness(5.0)
    handler.assert_called_once_with(asset)
    handler.reset_mock()

    asset.set_material("new-material-uid")
    handler.assert_called_once_with(asset)
