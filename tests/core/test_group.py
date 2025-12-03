import pytest
from rayforge.core.group import Group
from rayforge.core.workpiece import WorkPiece
from rayforge.core.layer import Layer
from rayforge.core.matrix import Matrix
from rayforge.core.stock import StockItem


@pytest.fixture
def workpiece_factory():
    """Provides a factory to create dummy WorkPiece instances for testing."""

    def _create_workpiece(name="test_wp"):
        return WorkPiece(name=f"{name}.svg")

    return _create_workpiece


@pytest.fixture
def stock_item_factory():
    """
    Provides a factory to create dummy StockItem instances for testing.
    These tests only care about the type, so a dummy UID is sufficient.
    """

    def _create_stock_item(name="test_stock"):
        return StockItem(stock_asset_uid=f"asset-for-{name}", name=name)

    return _create_stock_item


def test_group_initialization():
    """Tests that a Group initializes correctly."""
    group = Group(name="Test Group")
    assert group.name == "Test Group"
    assert group.parent is None
    assert not group.children
    assert group.matrix.is_identity()


def test_group_layer_property():
    """
    Tests the recursive lookup of the 'layer' property.
    Verifies it handles immediate parents, nested ancestors, and detached
    states.
    """
    # 1. Test with no parent
    group = Group("Orphan Group")
    assert group.layer is None

    # 2. Test with direct Layer parent
    layer = Layer("Target Layer")
    group.parent = None  # Reset just in case
    layer.add_child(group)

    assert group.layer is not None
    assert group.layer is layer
    assert group.layer.name == "Target Layer"

    # 3. Test with nested Group (Layer -> OuterGroup -> InnerGroup)
    outer_group = Group("Outer Group")
    inner_group = Group("Inner Group")

    # Re-arrange hierarchy
    layer.remove_child(group)  # clean up previous test

    layer.add_child(outer_group)
    outer_group.add_child(inner_group)

    assert outer_group.layer is layer
    assert inner_group.layer is layer

    # 4. Test that breaking the chain updates the property result
    layer.remove_child(outer_group)

    # outer_group is now orphaned, so inner_group (its child) should also
    # find no layer
    assert outer_group.layer is None
    assert inner_group.layer is None


def test_group_add_and_remove_child(workpiece_factory):
    """Tests adding and removing children from a group."""
    group = Group()
    wp1 = workpiece_factory("wp1")
    wp2 = workpiece_factory("wp2")

    group.add_child(wp1)
    assert len(group.children) == 1
    assert group.children[0] is wp1
    assert wp1.parent is group

    group.add_child(wp2)
    assert len(group.children) == 2
    assert group.children[1] is wp2
    assert wp2.parent is group

    group.remove_child(wp1)
    assert len(group.children) == 1
    assert group.children[0] is wp2
    assert wp1.parent is None


def test_group_transformation_hierarchy(workpiece_factory):
    """Tests that world transforms are calculated correctly through groups."""
    layer = Layer("Test Layer")
    group = Group()
    wp = workpiece_factory()

    layer.matrix = Matrix.translation(10, 20)
    group.matrix = Matrix.rotation(90) @ Matrix.translation(5, 0)
    wp.matrix = Matrix.scale(2, 3)

    layer.add_child(group)
    group.add_child(wp)

    # Expected transform: Layer @ Group @ WorkPiece
    # T(10, 20) @ R(90) @ T(5, 0) @ S(2, 3)
    expected_matrix = layer.matrix @ group.matrix @ wp.matrix
    actual_matrix = wp.get_world_transform()

    assert actual_matrix == expected_matrix

    # Test that a point is transformed correctly
    # Point (0,0) in WP space -> S -> (0,0) -> T(5,0) -> (5,0)
    #     -> R(90) -> (0,5) -> T(10,20) -> (10, 25)
    transformed_point = actual_matrix.transform_point((0, 0))
    assert transformed_point[0] == pytest.approx(10)
    assert transformed_point[1] == pytest.approx(25)


def test_nested_group_transformation(workpiece_factory):
    """Tests world transform calculation with nested groups."""
    group1 = Group("Outer Group")
    group2 = Group("Inner Group")
    wp = workpiece_factory()

    group1.matrix = Matrix.translation(100, 100)
    group2.matrix = Matrix.scale(2, 2)
    wp.matrix = Matrix.translation(5, 10)

    group1.add_child(group2)
    group2.add_child(wp)

    # Expected: Group1 @ Group2 @ WorkPiece
    expected_matrix = group1.matrix @ group2.matrix @ wp.matrix
    actual_matrix = wp.get_world_transform()
    assert actual_matrix == expected_matrix

    # Test a point:
    # (0,0) -> T(5,10) -> (5,10) -> S(2,2)
    #   -> (10,20) -> T(100,100) -> (110, 120)
    transformed_point = actual_matrix.transform_point((0, 0))
    assert transformed_point == pytest.approx((110, 120))


def test_get_all_workpieces_simple(workpiece_factory):
    """Tests get_all_workpieces on a group with only workpiece children."""
    group = Group()
    wp1 = workpiece_factory("wp1")
    wp2 = workpiece_factory("wp2")
    group.add_child(wp1)
    group.add_child(wp2)

    found_wps = group.all_workpieces
    assert len(found_wps) == 2
    assert wp1 in found_wps
    assert wp2 in found_wps


def test_get_all_workpieces_nested(workpiece_factory):
    """Tests get_all_workpieces on nested groups."""
    g1 = Group("g1")
    g2 = Group("g2")
    g3 = Group("g3_empty")  # An empty group
    wp1 = workpiece_factory("wp1")
    wp2 = workpiece_factory("wp2")
    wp3 = workpiece_factory("wp3")

    g1.add_child(wp1)
    g1.add_child(g2)
    g2.add_child(wp2)
    g2.add_child(wp3)
    g1.add_child(g3)

    found_wps = g1.all_workpieces
    assert len(found_wps) == 3
    assert wp1 in found_wps
    assert wp2 in found_wps
    assert wp3 in found_wps


def test_get_all_workpieces_empty():
    """Tests get_all_workpieces on an empty group."""
    group = Group()
    assert not group.all_workpieces


def test_signal_bubbling_on_add_child(workpiece_factory, mocker):
    """
    Verify that adding a child to a group bubbles the descendant_added signal.
    """
    layer = Layer("L1")
    group = Group("G1")
    wp = workpiece_factory("wp1")
    layer.add_child(group)

    # Spy on the layer's signal handler
    layer_handler = mocker.Mock()
    layer.descendant_added.connect(layer_handler)

    # Action: add a workpiece to the group
    group.add_child(wp)

    # Assert: the layer's signal was fired correctly, with the workpiece
    # as the origin of the event.
    layer_handler.assert_called_once()
    args, kwargs = layer_handler.call_args
    assert args[0] is layer
    assert kwargs["origin"] is wp


def test_factory_with_empty_list():
    """Tests that the factory returns None for an empty item list."""
    layer = Layer("test")
    result = Group.create_from_items([], layer)
    assert result is None


def test_factory_with_single_item(workpiece_factory):
    """
    Tests creating a group from a single item. The new group's transform
    should match the item's, and the item's new local transform should
    be identity.
    """
    layer = Layer("test")
    wp = workpiece_factory()
    wp.matrix = Matrix.translation(50, 60) @ Matrix.scale(2, 3)
    layer.add_child(wp)

    # Action
    result = Group.create_from_items([wp], layer)
    assert result is not None
    new_group = result.new_group
    new_child_matrix = result.child_matrices[wp.uid]

    # Assert: The group's matrix should perfectly encapsulate the workpiece.
    # It's a 1x1 unit square, so its size matrix is S(1,1).
    expected_group_matrix = wp.matrix @ Matrix.scale(1, 1)
    assert new_group.matrix == expected_group_matrix

    # Assert: The child's new matrix relative to the group should be identity.
    assert new_child_matrix.is_identity()


def test_factory_preserves_world_transform(workpiece_factory):
    """
    The most critical test: ensures that after grouping, the items'
    world transforms are perfectly preserved.
    """
    layer = Layer("Test Layer")
    wp1 = workpiece_factory("wp1")
    wp2 = workpiece_factory("wp2")

    # Setup with different transformations
    wp1.matrix = Matrix.translation(10, 20)
    wp2.matrix = Matrix.translation(100, 20) @ Matrix.rotation(90)
    layer.add_child(wp1)
    layer.add_child(wp2)

    # Capture original world transforms BEFORE the operation
    original_wp1_world = wp1.get_world_transform()
    original_wp2_world = wp2.get_world_transform()

    # Action: Calculate the grouping result
    result = Group.create_from_items([wp1, wp2], layer)
    assert result is not None

    # Assert: Calculate the NEW world transform and compare to the original.
    # new_world_transform = group's_world_transform @ new_child_local_transform
    # Since the layer is at the origin, the group's world transform is just
    # its matrix.
    new_wp1_world = result.new_group.matrix @ result.child_matrices[wp1.uid]
    new_wp2_world = result.new_group.matrix @ result.child_matrices[wp2.uid]

    assert new_wp1_world == original_wp1_world
    assert new_wp2_world == original_wp2_world


def test_factory_with_nested_item(workpiece_factory):
    """
    Tests that the factory correctly calculates transforms for an item
    that is already nested in another group.
    """
    layer = Layer("Test Layer")
    existing_group = Group("Existing")
    wp = workpiece_factory()

    # Setup a deep hierarchy: layer -> existing_group -> wp
    layer.matrix = Matrix.translation(1000, 2000)
    existing_group.matrix = Matrix.scale(2, 2)
    wp.matrix = Matrix.translation(50, 0)
    layer.add_child(existing_group)
    existing_group.add_child(wp)

    # Capture original state
    original_wp_world = wp.get_world_transform()

    # Action: Group the workpiece, but make the new group a child of the layer
    result = Group.create_from_items([wp], layer)
    assert result is not None

    # Assert: The world transform must be preserved.
    # The new group's world transform will be its local matrix relative to
    # the layer.
    new_group_world_transform = (
        layer.get_world_transform() @ result.new_group.matrix
    )
    new_wp_world = new_group_world_transform @ result.child_matrices[wp.uid]

    assert new_wp_world == original_wp_world


def test_deserialization_skips_stock_item_child():
    """
    Tests that Group.from_dict skips over any StockItem found in its
    children list, as they are not allowed to be grouped.
    """
    group_dict = {
        "uid": "group-uid",
        "type": "group",
        "name": "My Group",
        "matrix": Matrix.identity().to_list(),
        "children": [
            {
                "uid": "wp-uid",
                "type": "workpiece",
                "name": "test.svg",
                "matrix": Matrix.identity().to_list(),
                "vectors": None,
            },
            {
                "uid": "stock-uid",
                "type": "stockitem",
                "name": "My Stock",
                "matrix": Matrix.identity().to_list(),
                "stock_asset_uid": "some-asset-uid",
            },
        ],
    }

    group = Group.from_dict(group_dict)

    assert len(group.children) == 1
    assert isinstance(group.children[0], WorkPiece)
    assert group.children[0].uid == "wp-uid"


def test_factory_ignores_stock_items(workpiece_factory, stock_item_factory):
    """
    Tests that the group factory method ignores any StockItems in the list
    of items to be grouped.
    """
    layer = Layer("Test Layer")
    wp1 = workpiece_factory("wp1")
    stock1 = stock_item_factory("stock1")

    wp1.matrix = Matrix.translation(10, 20)
    stock1.matrix = Matrix.translation(500, 500)
    layer.add_child(wp1)
    layer.add_child(stock1)

    # Action: try to group a workpiece and a stock item
    result = Group.create_from_items([wp1, stock1], layer)

    assert result is not None

    # Assert: The group's bbox should only encompass the workpiece,
    # not the stock.
    # Bbox of wp1: (10, 20, 1, 1)
    new_group = result.new_group
    group_x, group_y, group_w, group_h = new_group.matrix.transform_rectangle(
        (0, 0, 1, 1)
    )

    assert group_x == pytest.approx(10)
    assert group_y == pytest.approx(20)
    assert group_w == pytest.approx(1)
    assert group_h == pytest.approx(1)

    # Assert: The child matrices dict should only contain the workpiece
    assert wp1.uid in result.child_matrices
    assert stock1.uid not in result.child_matrices


def test_factory_returns_none_if_only_stock_items(stock_item_factory):
    """
    Tests that the group factory returns None if the only items to group are
    StockItems.
    """
    layer = Layer("Test Layer")
    stock1 = stock_item_factory("stock1")
    stock2 = stock_item_factory("stock2")
    layer.add_child(stock1)
    layer.add_child(stock2)

    result = Group.create_from_items([stock1, stock2], layer)
    assert result is None


def test_group_to_dict_serialization():
    """Tests serializing a Group to a dictionary."""
    group = Group(name="Test Group")
    group.matrix = Matrix.translation(10, 20) @ Matrix.scale(2, 3)

    wp1 = WorkPiece("wp1.svg")
    wp2 = WorkPiece("wp2.svg")
    group.add_child(wp1)
    group.add_child(wp2)

    data = group.to_dict()

    expected_matrix = Matrix.translation(10, 20) @ Matrix.scale(2, 3)

    assert data["type"] == "group"
    assert data["name"] == "Test Group"
    assert data["matrix"] == expected_matrix.to_list()
    assert "children" in data
    assert len(data["children"]) == 2


def test_group_to_dict_with_nested_groups():
    """Tests serializing a Group with nested Groups."""
    outer_group = Group("Outer Group")
    inner_group = Group("Inner Group")
    wp = WorkPiece("test.svg")

    inner_group.add_child(wp)
    outer_group.add_child(inner_group)

    data = outer_group.to_dict()

    assert data["type"] == "group"
    assert data["name"] == "Outer Group"
    assert len(data["children"]) == 1
    assert data["children"][0]["type"] == "group"
    assert data["children"][0]["name"] == "Inner Group"
    assert len(data["children"][0]["children"]) == 1


def test_group_roundtrip_serialization():
    """Tests that to_dict() and from_dict() produce equivalent objects."""
    # Create a group with children
    original = Group("Roundtrip Group")
    original.matrix = Matrix.translation(10, 20) @ Matrix.rotation(45)

    wp1 = WorkPiece("wp1.svg")
    wp2 = WorkPiece("wp2.svg")
    original.add_child(wp1)
    original.add_child(wp2)

    # Serialize and deserialize
    data = original.to_dict()
    restored = Group.from_dict(data)

    # Check that the restored object has the same properties
    assert restored.uid == original.uid
    assert restored.name == original.name
    assert restored.matrix == original.matrix
    assert len(restored.children) == len(original.children)
    assert all(
        child.name in ["wp1.svg", "wp2.svg"] for child in restored.children
    )


def test_group_aspect_ratio():
    """Tests that Groups (as DocItems) now have aspect ratio capabilities."""
    group = Group()
    # Groups default to identity (1x1 scale), so aspect is 1.0
    assert group.get_current_aspect_ratio() == pytest.approx(1.0)

    group.matrix = Matrix.scale(100, 50)
    assert group.get_current_aspect_ratio() == pytest.approx(2.0)
