import pytest
import math
from typing import Optional, Dict
from blinker import Signal
from rayforge.core.item import DocItem
from rayforge.core.group import Group
from rayforge.core.matrix import Matrix
from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece


class GroupItem(Group):
    """
    A generic container item for testing hierarchy. It can hold children
    but is not a workpiece itself.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name or "test")


class ConcreteItem(DocItem):
    """
    Another distinct DocItem type for testing type-based filtering.
    This item is NOT a workpiece.
    """

    def to_dict(self) -> Dict:
        return {"name": self.name}


class SimpleWorkPiece(WorkPiece):
    """
    A WorkPiece subclass for testing that avoids complex dependencies.
    """

    def __init__(self, name: str = "SimpleWorkPiece"):
        DocItem.__init__(self, name=name)


class SignalCatcher:
    """A simple callable class to catch and store signal emissions."""

    def __init__(self):
        self.calls = []

    def __call__(self, sender, **kwargs):
        origin = kwargs.get("origin", sender)
        self.calls.append({"sender": sender, "origin": origin})

    def reset(self):
        self.calls.clear()

    @property
    def call_count(self):
        return len(self.calls)

    def last_origin(self):
        return self.calls[-1]["origin"] if self.calls else None

    def last_sender(self):
        return self.calls[-1]["sender"] if self.calls else None


def test_initialization():
    """Tests the initial state of a DocItem."""
    item = GroupItem(name="TestItem")
    assert isinstance(item.uid, str)
    assert len(item.uid) == 36  # UUID4 string length
    assert item.name == "TestItem"
    assert item.parent is None
    assert item.children == []
    assert item.matrix == Matrix.identity()
    assert isinstance(item.updated, Signal)


def test_parent_and_doc_properties():
    """Tests parent-child relationships and the .doc property."""
    doc = Doc()
    item1 = GroupItem()
    item2 = GroupItem()

    doc.add_child(item1)
    item1.add_child(item2)

    assert item1.parent is doc
    assert item2.parent is item1

    assert doc.doc is doc
    assert item1.doc is doc
    assert item2.doc is doc

    standalone_item = GroupItem()
    assert standalone_item.doc is None


def test_add_child_and_reparenting():
    """Tests adding children, including re-parenting from another item."""
    p1 = GroupItem()
    p2 = GroupItem()
    c1 = GroupItem()
    c2 = GroupItem()

    p1.add_child(c1)
    assert c1 in p1.children
    assert c1.parent is p1

    p1.add_child(c2, index=0)
    assert p1.children == [c2, c1]

    # Re-parent c1 to p2
    p2.add_child(c1)
    assert c1 not in p1.children
    assert c1 in p2.children
    assert c1.parent is p2

    # Adding the same child again should not change anything
    children_before = list(p2.children)
    p2.add_child(c1)
    assert p2.children == children_before


def test_remove_child():
    """Tests removing a child from its parent."""
    parent = GroupItem()
    child = GroupItem()
    parent.add_child(child)
    assert child in parent.children

    parent.remove_child(child)
    assert child not in parent.children
    assert child.parent is None

    # Removing a non-child should not raise an error.
    # pytest will fail the test if any unexpected exception is raised.
    parent.remove_child(GroupItem())


def test_set_children():
    """Tests replacing the entire list of children."""
    parent = GroupItem()
    other_parent = GroupItem()
    c1, c2, c3, c4 = GroupItem(), GroupItem(), GroupItem(), GroupItem()

    parent.add_child(c1)
    parent.add_child(c2)
    other_parent.add_child(c4)

    parent.set_children([c2, c3, c4])

    assert parent.children == [c2, c3, c4]
    assert c1.parent is None
    assert c2.parent is parent
    assert c3.parent is parent
    assert c4.parent is parent
    assert c4 not in other_parent.children

    parent.set_children([])
    assert parent.children == []
    assert c2.parent is None
    assert c3.parent is None
    assert c4.parent is None


def test_get_depth():
    """Tests the calculation of an item's depth in the hierarchy."""
    root = GroupItem()
    child = GroupItem()
    grandchild = GroupItem()

    root.add_child(child)
    child.add_child(grandchild)

    assert root.get_depth() == 0
    assert child.get_depth() == 1
    assert grandchild.get_depth() == 2


def test_get_descendants():
    """
    Tests recursive fetching of all descendants, with and without
    type filtering.
    """
    root = GroupItem(name="root")
    c1 = GroupItem(name="c1")
    c2 = ConcreteItem(name="c2")
    gc1 = ConcreteItem(name="gc1")

    root.add_child(c1)
    root.add_child(c2)
    c1.add_child(gc1)

    descendants = root.get_descendants()
    assert descendants == [c1, gc1, c2]
    assert gc1.get_descendants() == []
    assert root.get_descendants(of_type=GroupItem) == [c1]
    assert root.get_descendants(of_type=ConcreteItem) == [gc1, c2]
    assert root.get_descendants(of_type=DocItem) == [c1, gc1, c2]
    assert root.get_descendants(of_type=Doc) == []


def test_find_descendant_by_uid():
    """Tests the recursive search for a descendant by its UID."""
    root = GroupItem(name="root")
    c1 = GroupItem(name="c1")
    c2 = ConcreteItem(name="c2")
    gc1 = ConcreteItem(name="gc1")

    root.add_child(c1)
    root.add_child(c2)
    c1.add_child(gc1)

    # Find a deeply nested item
    assert root.find_descendant_by_uid(gc1.uid) is gc1
    # Find a direct child
    assert root.find_descendant_by_uid(c2.uid) is c2
    # Search should not find the item itself
    assert root.find_descendant_by_uid(root.uid) is None
    # Search should return None for an unknown UID
    assert root.find_descendant_by_uid("non-existent-uid") is None
    # Search on a leaf node should return None
    assert gc1.find_descendant_by_uid(root.uid) is None


def test_matrix_property_and_signal():
    """Tests setting the matrix and the firing of `transform_changed`."""
    item = GroupItem()
    catcher = SignalCatcher()
    item.transform_changed.connect(catcher)

    new_matrix = Matrix(((2, 0, 10), (0, 1, 5), (0, 0, 1)))

    item.matrix = new_matrix
    assert item.matrix == new_matrix
    assert catcher.call_count == 1
    assert catcher.last_sender() is item

    catcher.reset()
    item.matrix = new_matrix
    assert catcher.call_count == 0


def test_get_world_transform():
    """Tests the calculation of the cumulative world transformation matrix."""
    m_root = Matrix(((2, 0, 0), (0, 2, 0), (0, 0, 1)))
    m_child = Matrix(((1, 0, 10), (0, 1, 5), (0, 0, 1)))

    root = GroupItem()
    child = GroupItem()
    grandchild = GroupItem()

    root.matrix = m_root
    child.matrix = m_child

    root.add_child(child)
    child.add_child(grandchild)

    assert root.get_world_transform() == m_root
    assert child.get_world_transform() == m_root @ m_child
    assert (
        grandchild.get_world_transform()
        == m_root @ m_child @ Matrix.identity()
    )


def test_signal_bubbling():
    """
    Tests that signals from descendants correctly bubble up the hierarchy.
    """
    gp = GroupItem(name="grandparent")
    p = GroupItem(name="parent")
    c = GroupItem(name="child")
    gp.add_child(p)
    p.add_child(c)

    gp_catcher = SignalCatcher()
    p_catcher = SignalCatcher()

    gp.descendant_updated.connect(gp_catcher)
    gp.descendant_transform_changed.connect(gp_catcher)
    gp.descendant_added.connect(gp_catcher)
    gp.descendant_removed.connect(gp_catcher)
    p.descendant_updated.connect(p_catcher)
    p.descendant_transform_changed.connect(p_catcher)
    p.descendant_added.connect(p_catcher)
    p.descendant_removed.connect(p_catcher)

    c.updated.send(c)
    assert p_catcher.call_count == 1
    assert p_catcher.last_origin() is c
    assert gp_catcher.call_count == 1
    assert gp_catcher.last_origin() is c

    p_catcher.reset()
    gp_catcher.reset()
    c.matrix = Matrix(((1, 0, 1), (0, 1, 0), (0, 0, 1)))
    assert p_catcher.call_count == 1
    assert p_catcher.last_origin() is c
    assert gp_catcher.call_count == 1
    assert gp_catcher.last_origin() is c

    p_catcher.reset()
    gp_catcher.reset()
    grandchild = GroupItem()
    p.add_child(grandchild)
    assert gp_catcher.call_count == 1
    assert gp_catcher.last_origin() is grandchild

    p_catcher.reset()
    gp_catcher.reset()
    p.remove_child(grandchild)
    assert gp_catcher.call_count == 1
    assert gp_catcher.last_origin() is grandchild


def test_signal_disconnection_on_remove():
    """Ensures that a removed child's signals are disconnected."""
    parent = GroupItem()
    child = GroupItem()
    parent.add_child(child)

    catcher = SignalCatcher()
    parent.descendant_updated.connect(catcher)

    parent.remove_child(child)

    child.updated.send(child)
    assert catcher.call_count == 0


def test_pos_property():
    """Tests the world-space pos property getter and setter."""
    parent = GroupItem()
    parent.matrix = Matrix.translation(100, 200) @ Matrix.rotation(90)
    item = GroupItem()
    parent.add_child(item)

    # Initial position should be the parent's translation
    # Rotated (0,0) is still (0,0), then translated by (100, 200)
    assert item.pos == pytest.approx((100, 200))

    # Set new world position
    item.pos = (50, 50)
    assert item.pos == pytest.approx((50, 50))

    # Test that setting the same position doesn't change the matrix
    old_matrix = item.matrix.copy()
    item.pos = (50, 50)
    assert item.matrix == old_matrix


def test_size_property():
    """Tests the world-space size property setter."""
    parent = GroupItem()
    parent.matrix = Matrix.scale(2, 3)  # Parent has non-uniform scale
    item = GroupItem()
    item.matrix = Matrix.rotation(45)  # Item has rotation
    parent.add_child(item)

    # Initial world size depends on parent scale and child rotation.
    # The `size` property returns the decomposed scale factors, which are
    # not trivial when rotation and non-uniform scale are mixed.
    # sx = sqrt((2*cos45)^2 + (3*sin45)^2) = sqrt(2 + 4.5) = sqrt(6.5)
    # sy = det / sx = (2*3) / sqrt(6.5) = 6 / sqrt(6.5)
    size_x = math.sqrt(6.5)
    size_y = 6 / size_x
    assert item.size == pytest.approx((size_x, size_y))

    # Get center point before resize
    center_before = item.get_world_transform().transform_point((0.5, 0.5))

    # Set new world size
    item.set_size(10, 20)
    assert item.size == pytest.approx((10, 20))

    # Check that center point is preserved
    center_after = item.get_world_transform().transform_point((0.5, 0.5))
    assert center_before == pytest.approx(center_after)


def test_angle_property():
    """Tests the local angle property getter and setter."""
    parent = GroupItem()
    parent.matrix = Matrix.rotation(30)
    item = GroupItem()
    parent.add_child(item)

    # Initial angle is 0
    assert item.angle == pytest.approx(0)

    # Get center point before rotation
    center_before = item.get_world_transform().transform_point((0.5, 0.5))

    # Set new local angle
    item.angle = 45
    assert item.angle == pytest.approx(45)

    # World rotation is parent + child
    world_angle = item.get_world_transform().get_rotation()
    assert world_angle == pytest.approx(30 + 45)

    # Center point should be preserved
    center_after = item.get_world_transform().transform_point((0.5, 0.5))
    assert center_before == pytest.approx(center_after)

    # Test setting the same angle doesn't fire signal
    # (implicitly tested by matrix change)
    old_matrix = item.matrix.copy()
    item.angle = 45
    assert item.matrix == old_matrix


def test_shear_property():
    """Tests the local shear property getter and setter."""
    parent = GroupItem()
    parent.matrix = Matrix.translation(10, 20) @ Matrix.rotation(30)
    item = GroupItem()
    item.matrix = Matrix.scale(2, 3)  # Give it some scale
    parent.add_child(item)

    # 1. Test Getter
    # Initial shear is 0
    assert item.shear == pytest.approx(0)
    # Manually set a matrix with shear
    item.matrix = Matrix.shear(0.5, 0)  # shear_factor = 0.5
    expected_shear_angle = math.degrees(math.atan(0.5))
    assert item.shear == pytest.approx(expected_shear_angle)

    # 2. Test Setter and center preservation
    # Reset item matrix
    item.matrix = Matrix.scale(2, 3)
    assert item.shear == pytest.approx(0)

    # Get center point before shear
    center_before = item.get_world_transform().transform_point((0.5, 0.5))

    catcher = SignalCatcher()
    item.transform_changed.connect(catcher)

    # Set new local shear
    new_shear_angle = 15.0
    item.shear = new_shear_angle
    assert item.shear == pytest.approx(new_shear_angle)
    assert catcher.call_count == 1

    # Center point should be preserved
    center_after = item.get_world_transform().transform_point((0.5, 0.5))
    assert center_before == pytest.approx(center_after)

    # 3. Test that setting the same value doesn't change the matrix/fire signal
    catcher.reset()
    old_matrix = item.matrix.copy()
    item.shear = 15.0
    assert item.matrix == old_matrix
    assert catcher.call_count == 0


def test_add_children_basic():
    """Tests adding multiple children at once."""
    parent = GroupItem()
    c1, c2, c3 = GroupItem(), GroupItem(), GroupItem()

    parent.add_children([c1, c2])
    assert parent.children == [c1, c2]
    assert c1.parent is parent
    assert c2.parent is parent

    parent.add_children([c3], index=1)
    assert parent.children == [c1, c3, c2]
    assert c3.parent is parent

    # Test adding an empty list
    children_before = list(parent.children)
    parent.add_children([])
    assert parent.children == children_before


def test_remove_children():
    """Tests removing multiple children at once."""
    parent = GroupItem()
    c1, c2, c3, c4 = GroupItem(), GroupItem(), GroupItem(), GroupItem()
    parent.set_children([c1, c2, c3, c4])

    parent.remove_children([c2, c4])
    assert parent.children == [c1, c3]
    assert c2.parent is None
    assert c4.parent is None
    assert c1.parent is parent
    assert c3.parent is parent

    # Test removing non-existent or already removed children
    children_before = list(parent.children)
    parent.remove_children([c2, GroupItem()])
    assert parent.children == children_before


def test_add_children_reparenting():
    """Tests that add_children correctly re-parents items."""
    p1 = GroupItem("p1")
    p2 = GroupItem("p2")
    c1, c2, c3 = GroupItem("c1"), GroupItem("c2"), GroupItem("c3")

    p1.add_children([c1, c2])
    assert c1.parent is p1
    assert c2.parent is p1
    assert len(p1.children) == 2

    p2.add_children([c2, c3])
    assert len(p1.children) == 1
    assert p1.children == [c1]
    assert len(p2.children) == 2
    assert p2.children == [c2, c3]
    assert c2.parent is p2
    assert c3.parent is p2


def test_bulk_operations_fire_single_signal():
    """
    Verifies that add_children and remove_children only fire a single
    'updated' signal, not one per child.
    """
    parent = GroupItem()
    c1, c2, c3 = GroupItem(), GroupItem(), GroupItem()

    catcher = SignalCatcher()
    parent.updated.connect(catcher)

    # Test add_children
    parent.add_children([c1, c2, c3])
    assert catcher.call_count == 1
    assert parent.children == [c1, c2, c3]

    # Test remove_children
    catcher.reset()
    parent.remove_children([c1, c3])
    assert catcher.call_count == 1
    assert parent.children == [c2]

    # Test adding empty list fires no signal
    catcher.reset()
    parent.add_children([])
    assert catcher.call_count == 0

    # Test removing empty list fires no signal
    catcher.reset()
    parent.remove_children([])
    assert catcher.call_count == 0
