from rayforge.core.stock import StockItem
from rayforge.core.geo import Geometry
from rayforge.core.matrix import Matrix


def test_stock_item_initialization():
    """Tests that a StockItem initializes correctly."""
    stock = StockItem(name="My Stock")
    assert stock.name == "My Stock"
    assert stock.parent is None
    assert not stock.children  # DocItem property
    assert isinstance(stock.geometry, Geometry)
    assert stock.geometry.is_empty()
    assert stock.matrix.is_identity()


def test_stock_item_initialization_with_geometry():
    """Tests initializing a StockItem with a pre-existing Geometry object."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(100, 0)
    geo.line_to(100, 50)
    geo.line_to(0, 50)
    geo.close_path()

    stock = StockItem(geometry=geo)
    assert stock.geometry is geo
    assert len(stock.geometry.commands) == 5


def test_stock_item_to_dict_serialization():
    """Tests serializing a StockItem to a dictionary."""
    geo = Geometry()
    geo.move_to(10, 20)
    geo.line_to(30, 40)

    stock = StockItem(name="Serialized Stock", geometry=geo)
    stock.matrix = Matrix.translation(5, 15)
    stock_uid = stock.uid  # capture for later comparison

    data = stock.to_dict()

    assert data["type"] == "stockitem"
    assert data["uid"] == stock_uid
    assert data["name"] == "Serialized Stock"
    assert data["matrix"] == Matrix.translation(5, 15).to_list()
    assert "commands" in data["geometry"]
    assert len(data["geometry"]["commands"]) == 2
    assert data["geometry"]["commands"][0]["type"] == "MoveToCommand"
    assert data["geometry"]["commands"][0]["end"] == (10.0, 20.0, 0.0)


def test_stock_item_from_dict_deserialization():
    """Tests creating a StockItem instance from a dictionary."""
    stock_dict = {
        "uid": "test-uid-123",
        "type": "stockitem",
        "name": "Deserialized Stock",
        "matrix": [[1, 0, 10], [0, 1, 20], [0, 0, 1]],
        "geometry": {
            "commands": [
                {"type": "MoveToCommand", "end": [0, 0, 0]},
                {"type": "LineToCommand", "end": [50, 50, 0]},
            ],
            "last_move_to": [0, 0, 0],
        },
    }

    stock = StockItem.from_dict(stock_dict)

    assert isinstance(stock, StockItem)
    assert stock.uid == "test-uid-123"
    assert stock.name == "Deserialized Stock"
    assert stock.matrix == Matrix.translation(10, 20)
    assert isinstance(stock.geometry, Geometry)
    assert len(stock.geometry.commands) == 2
    assert stock.geometry.rect() == (0.0, 0.0, 50.0, 50.0)


def test_stock_item_deserialization_with_no_geometry():
    """Tests deserializing a StockItem that has null/no geometry data."""
    stock_dict = {
        "uid": "test-uid-456",
        "type": "stockitem",
        "name": "No-Geo Stock",
        "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "geometry": None,
    }

    stock = StockItem.from_dict(stock_dict)
    assert isinstance(stock.geometry, Geometry)
    assert stock.geometry.is_empty()
