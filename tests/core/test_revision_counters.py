"""
Tests for the ``geometry_revision`` / ``transform_revision`` monotonic
counters on :class:`~rayforge.core.item.DocItem`.

These counters are bumped transparently whenever ``updated`` or
``transform_changed`` is sent, respectively, and are read by the
pipeline to build stable ``version_token`` values for raygeo
:class:`NodeRequest` objects (slice B0 of the migration plan).
"""

from typing import Dict

from raygeo.geo import Matrix

from rayforge.core.doc import Doc
from rayforge.core.group import Group
from rayforge.core.item import DocItem


class LeafItem(DocItem):
    """A minimal concrete DocItem for direct signal tests."""

    def to_dict(self) -> Dict:
        return {"name": self.name}

    @classmethod
    def from_dict(cls, data: Dict) -> "LeafItem":
        return cls(name=data.get("name", "leaf"))


def test_revisions_start_at_zero():
    item = LeafItem(name="leaf")
    assert item.geometry_revision == 0
    assert item.transform_revision == 0


def test_geometry_revision_bumps_on_updated_send():
    item = LeafItem(name="leaf")
    item.updated.send(item)
    assert item.geometry_revision == 1
    item.updated.send(item)
    assert item.geometry_revision == 2


def test_transform_revision_bumps_on_transform_changed_send():
    item = LeafItem(name="leaf")
    item.transform_changed.send(item, old_matrix=Matrix.identity())
    assert item.transform_revision == 1


def test_geometry_revision_does_not_bump_on_transform_changed():
    item = LeafItem(name="leaf")
    item.transform_changed.send(item, old_matrix=Matrix.identity())
    assert item.geometry_revision == 0
    assert item.transform_revision == 1


def test_transform_revision_does_not_bump_on_updated():
    item = LeafItem(name="leaf")
    item.updated.send(item)
    assert item.transform_revision == 0
    assert item.geometry_revision == 1


def test_name_setter_bumps_geometry_revision():
    item = LeafItem(name="leaf")
    item.name = "renamed"
    assert item.geometry_revision == 1
    assert item.transform_revision == 0


def test_matrix_setter_bumps_transform_revision():
    item = LeafItem(name="leaf")
    item.matrix = Matrix.translation(10, 20)
    assert item.transform_revision == 1
    assert item.geometry_revision == 0


def test_matrix_setter_no_bump_when_unchanged():
    item = LeafItem(name="leaf")
    item.matrix = Matrix.identity()  # same as default
    assert item.transform_revision == 0


def test_revision_is_read_only():
    item = LeafItem(name="leaf")
    try:
        item.geometry_revision = 5  # type: ignore[misc]
        assert False, "expected AttributeError"
    except AttributeError:
        pass
    try:
        item.transform_revision = 5  # type: ignore[misc]
        assert False, "expected AttributeError"
    except AttributeError:
        pass


def test_bubbling_signals_do_not_bump_parent_revision():
    """
    A child's ``updated`` emission bumps the *child's*
    ``geometry_revision`` only; the parent that relays via
    ``descendant_updated`` must not have its own revision bumped.
    """
    parent = Group(name="parent")
    child = LeafItem(name="child")
    parent.add_child(child)

    child.updated.send(child)
    assert child.geometry_revision == 1
    assert parent.geometry_revision == 0


def test_workpiece_revision_bumps():
    """End-to-end check using real WorkPiece / Doc / Layer wiring."""
    from rayforge.core.workpiece import WorkPiece

    doc = Doc()
    layer = doc.active_layer
    wp = WorkPiece(name="wp")
    layer.add_child(wp)

    assert wp.geometry_revision == 0
    # Setting a property that fires ``updated`` (e.g. via a subclass)
    # Direct send.
    wp.updated.send(wp)
    assert wp.geometry_revision == 1
    assert layer.geometry_revision == 0
    assert doc.geometry_revision == 0


def test_step_name_change_bumps_geometry_revision():
    """A real Step's name setter should bump geometry_revision.

    Step extends DocItem so it inherits the revision machinery.  We use
    a minimal concrete Step subclass to avoid pulling in addon loading.
    """
    from typing import Any, Dict

    from rayforge.core.step import Step

    class _DummyStep(Step):
        def to_dict(self) -> Dict[str, Any]:
            return {}

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "_DummyStep":
            return cls(typelabel="dummy")

    step = _DummyStep(typelabel="dummy", name="step")
    assert step.geometry_revision == 0
    step.name = "renamed"
    assert step.geometry_revision == 1
    assert step.transform_revision == 0


def test_signal_is_still_a_blinker_signal():
    """Ensure _RevisionSignal subclasses blinker.Signal so isinstance
    checks elsewhere in the codebase keep working."""
    item = LeafItem(name="leaf")
    from blinker import Signal

    assert isinstance(item.updated, Signal)
    assert isinstance(item.transform_changed, Signal)


def test_owner_garbage_collected_does_not_raise():
    """
    After the owning DocItem is GC'd, sending on a leftover reference
    to its signal must not raise.
    """
    import gc

    item = LeafItem(name="leaf")
    sig = item.updated
    del item
    gc.collect()
    sig.send(None)  # must not raise
