from typing import Optional
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
