from rayforge.core.sketcher.registry import EntityRegistry
from rayforge.core.sketcher.selection import SketchSelection


class TestClear:
    def test_clears_all_selections(self):
        selection = SketchSelection()
        selection.point_ids.extend([1, 2])
        selection.entity_ids.extend([3, 4])
        selection.constraint_idx = 5
        selection.junction_pid = 6

        selection.clear()

        assert selection.point_ids == []
        assert selection.entity_ids == []
        assert selection.constraint_idx is None
        assert selection.junction_pid is None

    def test_emits_changed_signal(self):
        selection = SketchSelection()
        signal_received = []

        def on_changed(sender):
            signal_received.append(sender)

        selection.changed.connect(on_changed)
        selection.clear()

        assert len(signal_received) == 1
        assert signal_received[0] is selection


class TestCopy:
    def test_creates_shallow_copy(self):
        selection = SketchSelection()
        selection.point_ids.extend([1, 2])
        selection.entity_ids.extend([3, 4])
        selection.constraint_idx = 5
        selection.junction_pid = 6

        copy = selection.copy()

        assert copy is not selection
        assert copy.point_ids == selection.point_ids
        assert copy.entity_ids == selection.entity_ids
        assert copy.constraint_idx == selection.constraint_idx
        assert copy.junction_pid == selection.junction_pid

    def test_copy_is_independent(self):
        selection = SketchSelection()
        selection.point_ids.append(1)
        selection.entity_ids.append(2)

        copy = selection.copy()
        copy.point_ids.append(3)
        copy.entity_ids.append(4)

        assert 3 not in selection.point_ids
        assert 4 not in selection.entity_ids


class TestSelectConstraint:
    def test_selects_constraint_single_mode(self):
        selection = SketchSelection()
        selection.point_ids.append(1)
        selection.entity_ids.append(2)
        selection.junction_pid = 3

        selection.select_constraint(5, is_multi=False)

        assert selection.constraint_idx == 5
        assert selection.point_ids == []
        assert selection.entity_ids == []
        assert selection.junction_pid is None

    def test_selects_constraint_multi_mode(self):
        selection = SketchSelection()
        selection.point_ids.append(1)
        selection.entity_ids.append(2)
        selection.junction_pid = 3

        selection.select_constraint(5, is_multi=True)

        assert selection.constraint_idx == 5
        assert selection.point_ids == [1]
        assert selection.entity_ids == [2]
        assert selection.junction_pid == 3

    def test_emits_changed_signal(self):
        selection = SketchSelection()
        signal_received = []

        def on_changed(sender):
            signal_received.append(sender)

        selection.changed.connect(on_changed)
        selection.select_constraint(1, is_multi=False)

        assert len(signal_received) == 1


class TestSelectJunction:
    def test_selects_junction_single_mode(self):
        selection = SketchSelection()
        selection.point_ids.append(1)
        selection.entity_ids.append(2)
        selection.constraint_idx = 3

        selection.select_junction(5, is_multi=False)

        assert selection.junction_pid == 5
        assert selection.point_ids == []
        assert selection.entity_ids == []
        assert selection.constraint_idx is None

    def test_selects_junction_multi_mode(self):
        selection = SketchSelection()
        selection.point_ids.append(1)
        selection.entity_ids.append(2)
        selection.constraint_idx = 3

        selection.select_junction(5, is_multi=True)

        assert selection.junction_pid == 5
        assert selection.point_ids == [1]
        assert selection.entity_ids == [2]
        assert selection.constraint_idx == 3

    def test_emits_changed_signal(self):
        selection = SketchSelection()
        signal_received = []

        def on_changed(sender):
            signal_received.append(sender)

        selection.changed.connect(on_changed)
        selection.select_junction(1, is_multi=False)

        assert len(signal_received) == 1


class TestSelectPoint:
    def test_selects_point_single_mode(self):
        selection = SketchSelection()
        selection.entity_ids.append(2)
        selection.constraint_idx = 3
        selection.junction_pid = 4

        selection.select_point(5, is_multi=False)

        assert selection.point_ids == [5]
        assert selection.entity_ids == []
        assert selection.constraint_idx is None
        assert selection.junction_pid is None

    def test_selects_point_multi_mode_adds(self):
        selection = SketchSelection()

        selection.select_point(1, is_multi=True)
        selection.select_point(2, is_multi=True)

        assert selection.point_ids == [1, 2]

    def test_selects_point_multi_mode_toggles(self):
        selection = SketchSelection()
        selection.select_point(1, is_multi=True)
        selection.select_point(1, is_multi=True)

        assert selection.point_ids == []

    def test_selects_point_single_mode_no_change_if_same(self):
        selection = SketchSelection()
        selection.select_point(1, is_multi=False)

        selection.select_point(1, is_multi=False)

        assert selection.point_ids == [1]

    def test_emits_changed_signal(self):
        selection = SketchSelection()
        signal_received = []

        def on_changed(sender):
            signal_received.append(sender)

        selection.changed.connect(on_changed)
        selection.select_point(1, is_multi=False)

        assert len(signal_received) == 1


class TestSelectEntity:
    def test_selects_entity_single_mode(self):
        registry = EntityRegistry()
        p1 = registry.add_point(0, 0)
        p2 = registry.add_point(10, 0)
        line = registry.get_entity(registry.add_line(p1, p2))
        assert line is not None

        selection = SketchSelection()
        selection.point_ids.append(1)
        selection.constraint_idx = 2
        selection.junction_pid = 3

        selection.select_entity(line, is_multi=False)

        assert selection.entity_ids == [line.id]
        assert selection.point_ids == []
        assert selection.constraint_idx is None
        assert selection.junction_pid is None

    def test_selects_entity_multi_mode_adds(self):
        registry = EntityRegistry()
        p1 = registry.add_point(0, 0)
        p2 = registry.add_point(10, 0)
        p3 = registry.add_point(20, 0)
        line1 = registry.get_entity(registry.add_line(p1, p2))
        line2 = registry.get_entity(registry.add_line(p2, p3))
        assert line1 is not None
        assert line2 is not None

        selection = SketchSelection()
        selection.select_entity(line1, is_multi=True)
        selection.select_entity(line2, is_multi=True)

        assert set(selection.entity_ids) == {line1.id, line2.id}

    def test_selects_entity_multi_mode_toggles(self):
        registry = EntityRegistry()
        p1 = registry.add_point(0, 0)
        p2 = registry.add_point(10, 0)
        line = registry.get_entity(registry.add_line(p1, p2))
        assert line is not None

        selection = SketchSelection()
        selection.select_entity(line, is_multi=True)
        selection.select_entity(line, is_multi=True)

        assert selection.entity_ids == []

    def test_selects_entity_single_mode_no_change_if_same(self):
        registry = EntityRegistry()
        p1 = registry.add_point(0, 0)
        p2 = registry.add_point(10, 0)
        line = registry.get_entity(registry.add_line(p1, p2))
        assert line is not None

        selection = SketchSelection()
        selection.select_entity(line, is_multi=False)
        initial_id = line.id
        selection.select_entity(line, is_multi=False)

        assert selection.entity_ids == [initial_id]

    def test_emits_changed_signal(self):
        registry = EntityRegistry()
        p1 = registry.add_point(0, 0)
        p2 = registry.add_point(10, 0)
        line = registry.get_entity(registry.add_line(p1, p2))

        selection = SketchSelection()
        signal_received = []

        def on_changed(sender):
            signal_received.append(sender)

        selection.changed.connect(on_changed)
        selection.select_entity(line, is_multi=False)

        assert len(signal_received) == 1


class TestSelectConnected:
    def test_adds_to_existing_selection(self):
        registry = EntityRegistry()
        p1 = registry.add_point(0, 0)
        p2 = registry.add_point(10, 0)
        p3 = registry.add_point(20, 0)
        p4 = registry.add_point(30, 0)
        l1 = registry.add_line(p1, p2)
        l2 = registry.add_line(p2, p3)
        l3 = registry.add_line(p3, p4)

        selection = SketchSelection()
        selection.entity_ids.append(l3)
        selection.point_ids.append(p1)

        selection.select_connected_entities(l1, registry)

        assert set(selection.entity_ids) == {l1, l2, l3}
        assert selection.point_ids == []
        assert selection.constraint_idx is None
        assert selection.junction_pid is None

    def test_clears_constraint_and_junction(self):
        registry = EntityRegistry()
        p1 = registry.add_point(0, 0)
        p2 = registry.add_point(10, 0)
        l1 = registry.add_line(p1, p2)

        selection = SketchSelection()
        selection.constraint_idx = 0
        selection.junction_pid = p1

        selection.select_connected_entities(l1, registry)

        assert selection.constraint_idx is None
        assert selection.junction_pid is None

    def test_emits_changed_signal(self):
        registry = EntityRegistry()
        p1 = registry.add_point(0, 0)
        p2 = registry.add_point(10, 0)
        l1 = registry.add_line(p1, p2)

        selection = SketchSelection()
        signal_received = []

        def on_changed(sender):
            signal_received.append(sender)

        selection.changed.connect(on_changed)
        selection.select_connected_entities(l1, registry)

        assert len(signal_received) == 1
        assert signal_received[0] is selection


class TestUpdateList:
    def test_single_mode_adds_if_not_present(self):
        selection = SketchSelection()
        collection = []

        selection._update_list(collection, 1, is_multi=False)

        assert collection == [1]

    def test_single_mode_no_change_if_already_present(self):
        selection = SketchSelection()
        collection = [1]

        selection._update_list(collection, 1, is_multi=False)

        assert collection == [1]

    def test_single_mode_replaces_with_new_item(self):
        selection = SketchSelection()
        collection = [1, 2, 3]

        selection._update_list(collection, 4, is_multi=False)

        assert collection == [4]

    def test_multi_mode_adds_if_not_present(self):
        selection = SketchSelection()
        collection = [1]

        selection._update_list(collection, 2, is_multi=True)

        assert collection == [1, 2]

    def test_multi_mode_removes_if_present(self):
        selection = SketchSelection()
        collection = [1, 2, 3]

        selection._update_list(collection, 2, is_multi=True)

        assert collection == [1, 3]


class TestIsEmpty:
    def test_empty_selection_returns_true(self):
        selection = SketchSelection()
        assert selection.is_empty() is True

    def test_with_point_returns_false(self):
        selection = SketchSelection()
        selection.point_ids.append(1)
        assert selection.is_empty() is False

    def test_with_entity_returns_false(self):
        selection = SketchSelection()
        selection.entity_ids.append(1)
        assert selection.is_empty() is False

    def test_with_constraint_returns_false(self):
        selection = SketchSelection()
        selection.constraint_idx = 0
        assert selection.is_empty() is False

    def test_with_junction_returns_false(self):
        selection = SketchSelection()
        selection.junction_pid = 1
        assert selection.is_empty() is False

    def test_after_clear_returns_true(self):
        selection = SketchSelection()
        selection.point_ids.append(1)
        selection.entity_ids.append(2)
        selection.constraint_idx = 0
        selection.junction_pid = 3
        selection.clear()
        assert selection.is_empty() is True
