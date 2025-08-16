import pytest
from pathlib import Path
from rayforge.core.group import Group
from rayforge.core.workpiece import WorkPiece
from rayforge.core.layer import Layer
from rayforge.core.matrix import Matrix
from rayforge.importer import SvgImporter


@pytest.fixture
def workpiece_factory():
    """Provides a factory to create dummy WorkPiece instances for testing."""

    def _create_workpiece(name="test_wp"):
        # The actual data doesn't matter for these tests.
        return WorkPiece(Path(f"{name}.svg"), b"<svg></svg>", SvgImporter)

    return _create_workpiece


def test_group_initialization():
    """Tests that a Group initializes correctly."""
    group = Group(name="Test Group")
    assert group.name == "Test Group"
    assert group.parent is None
    assert not group.children
    assert group.matrix.is_identity()


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
