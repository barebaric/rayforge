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
    item.parent = mock_doc

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


def test_stock_item_forward_compatibility_with_extra_fields():
    """
    Tests that from_dict() preserves extra fields from newer versions
    and to_dict() re-serializes them.
    """
    item_dict = {
        "uid": "item-forward-456",
        "type": "stockitem",
        "name": "Future Stock",
        "matrix": Matrix.identity().to_list(),
        "stock_asset_uid": "asset-future",
        "visible": True,
        "future_field_string": "some value",
        "future_field_number": 42,
        "future_field_dict": {"nested": "data"},
    }

    item = StockItem.from_dict(item_dict)

    # Verify extra fields are stored
    assert item.extra["future_field_string"] == "some value"
    assert item.extra["future_field_number"] == 42
    assert item.extra["future_field_dict"] == {"nested": "data"}

    # Verify extra fields are re-serialized
    data = item.to_dict()
    assert data["future_field_string"] == "some value"
    assert data["future_field_number"] == 42
    assert data["future_field_dict"] == {"nested": "data"}


def test_stock_item_backward_compatibility_with_missing_optional_fields():
    """
    Tests that from_dict() handles missing optional fields gracefully
    (simulating data from an older version).
    """
    minimal_dict = {
        "uid": "item-backward-789",
        "type": "stockitem",
        "matrix": Matrix.identity().to_list(),
        "stock_asset_uid": "asset-old",
    }

    item = StockItem.from_dict(minimal_dict)

    # Verify defaults are applied for missing optional fields
    assert item.name == "Stock"
    assert item.visible is True
    assert item.extra == {}


def test_stock_item_get_local_bbox(mock_doc, mock_asset):
    """Tests that get_local_bbox returns a unit square (0,0,1,1)."""
    item = StockItem(stock_asset_uid="asset-123")
    item.parent = mock_doc

    bbox = item.get_local_bbox()
    assert bbox == (0.0, 0.0, 1.0, 1.0)


def test_stock_item_get_local_bbox_independent_of_transform(
    mock_doc, mock_asset
):
    """Tests that get_local_bbox is independent of any transforms."""
    item = StockItem(stock_asset_uid="asset-123")
    item.parent = mock_doc
    item.matrix = Matrix.translation(10, 20) @ Matrix.scale(2, 3)

    bbox = item.get_local_bbox()
    assert bbox == (0.0, 0.0, 1.0, 1.0)


def test_stock_item_get_world_geometry_empty(mock_doc):
    """Tests that get_world_geometry handles empty geometry."""
    asset = MagicMock(spec=StockAsset)
    asset.uid = "asset-empty"
    asset.geometry = Geometry()

    doc = MagicMock(spec=Doc)
    doc.get_asset_by_uid.return_value = asset
    doc.doc = doc

    item = StockItem(stock_asset_uid="asset-empty")
    item.parent = doc

    world_geo = item.get_world_geometry()
    assert world_geo.is_empty()


def test_stock_item_get_world_geometry_identity_matrix():
    """Tests get_world_geometry with identity matrix (no transform)."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 10)

    asset = MagicMock(spec=StockAsset)
    asset.uid = "asset-geo"
    asset.geometry = geo
    asset.get_natural_size.return_value = (10.0, 10.0)

    doc = MagicMock(spec=Doc)
    doc.get_asset_by_uid.return_value = asset
    doc.doc = doc
    doc.get_world_transform.return_value = Matrix.identity()

    item = StockItem(stock_asset_uid="asset-geo")
    item.parent = doc
    item.matrix = Matrix.scale(10.0, 10.0)

    world_geo = item.get_world_geometry()
    assert not world_geo.is_empty()

    rect = world_geo.rect()
    assert rect[0] == pytest.approx(0.0)
    assert rect[1] == pytest.approx(0.0)
    assert rect[2] == pytest.approx(10.0)
    assert rect[3] == pytest.approx(10.0)


def test_stock_item_get_world_geometry_translation():
    """Tests get_world_geometry with translation."""
    geo = Geometry()
    geo.move_to(5, 5)
    geo.line_to(15, 15)

    asset = MagicMock(spec=StockAsset)
    asset.uid = "asset-trans"
    asset.geometry = geo
    asset.get_natural_size.return_value = (10.0, 10.0)

    doc = MagicMock(spec=Doc)
    doc.get_asset_by_uid.return_value = asset
    doc.doc = doc
    doc.get_world_transform.return_value = Matrix.identity()

    item = StockItem(stock_asset_uid="asset-trans")
    item.parent = doc
    item.matrix = Matrix.translation(100, 200) @ Matrix.scale(10.0, 10.0)

    world_geo = item.get_world_geometry()
    rect = world_geo.rect()

    assert rect[0] == pytest.approx(100.0)
    assert rect[1] == pytest.approx(200.0)
    assert rect[2] == pytest.approx(110.0)
    assert rect[3] == pytest.approx(210.0)


def test_stock_item_get_world_geometry_scale():
    """Tests get_world_geometry with scale transformation."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 5)

    asset = MagicMock(spec=StockAsset)
    asset.uid = "asset-scale"
    asset.geometry = geo
    asset.get_natural_size.return_value = (10.0, 5.0)

    doc = MagicMock(spec=Doc)
    doc.get_asset_by_uid.return_value = asset
    doc.doc = doc
    doc.get_world_transform.return_value = Matrix.identity()

    item = StockItem(stock_asset_uid="asset-scale")
    item.parent = doc
    item.matrix = Matrix.scale(20.0, 15.0)

    world_geo = item.get_world_geometry()
    rect = world_geo.rect()

    assert rect[0] == pytest.approx(0.0)
    assert rect[1] == pytest.approx(0.0)
    assert rect[2] == pytest.approx(20.0)
    assert rect[3] == pytest.approx(15.0)


def test_stock_item_get_world_geometry_offset_geometry():
    """Tests get_world_geometry normalizes geometry that doesn't start at
    origin.
    """
    geo = Geometry()
    geo.move_to(10, 20)
    geo.line_to(30, 40)

    asset = MagicMock(spec=StockAsset)
    asset.uid = "asset-offset"
    asset.geometry = geo
    asset.get_natural_size.return_value = (20.0, 20.0)

    doc = MagicMock(spec=Doc)
    doc.get_asset_by_uid.return_value = asset
    doc.doc = doc
    doc.get_world_transform.return_value = Matrix.identity()

    item = StockItem(stock_asset_uid="asset-offset")
    item.parent = doc
    item.matrix = Matrix.scale(50.0, 50.0)

    world_geo = item.get_world_geometry()
    rect = world_geo.rect()

    assert rect[0] == pytest.approx(0.0)
    assert rect[1] == pytest.approx(0.0)
    assert rect[2] == pytest.approx(50.0)
    assert rect[3] == pytest.approx(50.0)


def test_stock_item_get_world_rect_geometry_identity_matrix():
    """Tests get_world_rect_geometry with identity matrix."""
    asset = MagicMock(spec=StockAsset)
    asset.uid = "asset-rect"
    asset.geometry = Geometry()
    asset.get_natural_size.return_value = (100.0, 50.0)

    doc = MagicMock(spec=Doc)
    doc.get_asset_by_uid.return_value = asset
    doc.doc = doc
    doc.get_world_transform.return_value = Matrix.identity()

    item = StockItem(stock_asset_uid="asset-rect")
    item.parent = doc
    item.matrix = Matrix.identity()

    world_geo = item.get_world_rect_geometry()
    rect = world_geo.rect()

    assert rect[0] == pytest.approx(0.0)
    assert rect[1] == pytest.approx(0.0)
    assert rect[2] == pytest.approx(1.0)
    assert rect[3] == pytest.approx(1.0)


def test_stock_item_get_world_rect_geometry_translation():
    """Tests get_world_rect_geometry with translation."""
    asset = MagicMock(spec=StockAsset)
    asset.uid = "asset-rect-trans"
    asset.geometry = Geometry()
    asset.get_natural_size.return_value = (100.0, 50.0)

    doc = MagicMock(spec=Doc)
    doc.get_asset_by_uid.return_value = asset
    doc.doc = doc
    doc.get_world_transform.return_value = Matrix.identity()

    item = StockItem(stock_asset_uid="asset-rect-trans")
    item.parent = doc
    item.matrix = Matrix.translation(50, 75)

    world_geo = item.get_world_rect_geometry()
    rect = world_geo.rect()

    assert rect[0] == pytest.approx(50.0)
    assert rect[1] == pytest.approx(75.0)
    assert rect[2] == pytest.approx(51.0)
    assert rect[3] == pytest.approx(76.0)


def test_stock_item_get_world_rect_geometry_scale():
    """Tests get_world_rect_geometry with scale transformation."""
    asset = MagicMock(spec=StockAsset)
    asset.uid = "asset-rect-scale"
    asset.geometry = Geometry()
    asset.get_natural_size.return_value = (200.0, 100.0)

    doc = MagicMock(spec=Doc)
    doc.get_asset_by_uid.return_value = asset
    doc.doc = doc
    doc.get_world_transform.return_value = Matrix.identity()

    item = StockItem(stock_asset_uid="asset-rect-scale")
    item.parent = doc
    item.matrix = Matrix.scale(200.0, 100.0)

    world_geo = item.get_world_rect_geometry()
    rect = world_geo.rect()

    assert rect[0] == pytest.approx(0.0)
    assert rect[1] == pytest.approx(0.0)
    assert rect[2] == pytest.approx(200.0)
    assert rect[3] == pytest.approx(100.0)


def test_stock_item_get_world_rect_geometry_combined_transform():
    """Tests get_world_rect_geometry with combined transforms."""
    asset = MagicMock(spec=StockAsset)
    asset.uid = "asset-rect-combined"
    asset.geometry = Geometry()
    asset.get_natural_size.return_value = (100.0, 50.0)

    doc = MagicMock(spec=Doc)
    doc.get_asset_by_uid.return_value = asset
    doc.doc = doc
    doc.get_world_transform.return_value = Matrix.identity()

    item = StockItem(stock_asset_uid="asset-rect-combined")
    item.parent = doc
    item.matrix = Matrix.translation(10, 20) @ Matrix.scale(100, 50)

    world_geo = item.get_world_rect_geometry()
    rect = world_geo.rect()

    assert rect[0] == pytest.approx(10.0)
    assert rect[1] == pytest.approx(20.0)
    assert rect[2] == pytest.approx(110.0)
    assert rect[3] == pytest.approx(70.0)


def test_stock_item_world_geometry_vs_rect_geometry():
    """
    Tests that get_world_geometry and get_world_rect_geometry
    produce equivalent results for rectangular assets.
    """
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)
    geo.line_to(10, 10)
    geo.line_to(0, 10)
    geo.close_path()

    asset = MagicMock(spec=StockAsset)
    asset.uid = "asset-rect-geo"
    asset.geometry = geo
    asset.get_natural_size.return_value = (10.0, 10.0)

    doc = MagicMock(spec=Doc)
    doc.get_asset_by_uid.return_value = asset
    doc.doc = doc
    doc.get_world_transform.return_value = Matrix.identity()

    item = StockItem(stock_asset_uid="asset-rect-geo")
    item.parent = doc
    item.matrix = Matrix.translation(50, 100) @ Matrix.scale(20, 15)

    world_geo = item.get_world_geometry()
    world_rect_geo = item.get_world_rect_geometry()

    geo_rect = world_geo.rect()
    rect_rect = world_rect_geo.rect()

    assert geo_rect[0] == pytest.approx(rect_rect[0])
    assert geo_rect[1] == pytest.approx(rect_rect[1])
    assert geo_rect[2] == pytest.approx(rect_rect[2])
    assert geo_rect[3] == pytest.approx(rect_rect[3])
