"""
Tests for the ArtifactNode class.
"""

from unittest.mock import Mock
from rayforge.pipeline.dag.node import ArtifactNode, NodeState
from rayforge.pipeline.artifact.key import ArtifactKey


WP_UID_1 = "550e8400-e29b-41d4-a716-446655440001"
WP_UID_2 = "550e8400-e29b-41d4-a716-446655440002"
STEP_UID_1 = "550e8400-e29b-41d4-a716-446655440003"


def create_manager_with_state(key, generation_id, state):
    """Helper to create a mock manager that returns a specific state."""
    manager = Mock()
    manager.get_state.return_value = state
    return manager


class TestArtifactNode:
    """Tests for the ArtifactNode class."""

    def test_node_creation(self):
        """Test creating a node with default state."""
        key = ArtifactKey.for_workpiece(WP_UID_1)
        manager = create_manager_with_state(key, 1, NodeState.DIRTY)
        node = ArtifactNode(
            key=key,
            generation_id=1,
            _artifact_manager=manager,
        )
        assert node.key == key
        assert node.state == NodeState.DIRTY
        assert len(node.dependencies) == 0
        assert len(node.dependents) == 0

    def test_node_state_from_manager(self):
        """Test that node state comes from the manager."""
        key = ArtifactKey.for_step(STEP_UID_1)
        manager = create_manager_with_state(key, 1, NodeState.VALID)
        node = ArtifactNode(
            key=key,
            generation_id=1,
            _artifact_manager=manager,
        )
        assert node.state == NodeState.VALID

    def test_node_state_set_calls_manager(self):
        """Test that setting node state calls the manager."""
        key = ArtifactKey.for_step(STEP_UID_1)
        manager = Mock()
        manager.get_state.return_value = NodeState.DIRTY
        node = ArtifactNode(
            key=key,
            generation_id=1,
            _artifact_manager=manager,
        )
        node.state = NodeState.PROCESSING
        manager.set_state.assert_called_once_with(key, 1, NodeState.PROCESSING)

    def test_node_raises_without_manager(self):
        """Test that accessing state without manager raises."""
        key = ArtifactKey.for_workpiece(WP_UID_1)
        node = ArtifactNode(key, generation_id=1)
        try:
            _ = node.state
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            assert "no ArtifactManager" in str(e)

    def test_add_dependency(self):
        """Test adding a dependency relationship."""
        wp_key = ArtifactKey.for_workpiece(WP_UID_1)
        step_key = ArtifactKey.for_step(STEP_UID_1)

        wp_manager = create_manager_with_state(wp_key, 1, NodeState.VALID)
        step_manager = create_manager_with_state(step_key, 1, NodeState.DIRTY)

        wp_node = ArtifactNode(
            key=wp_key, generation_id=1, _artifact_manager=wp_manager
        )
        step_node = ArtifactNode(
            key=step_key, generation_id=1, _artifact_manager=step_manager
        )

        step_node.add_dependency(wp_node)

        assert wp_node in step_node.dependencies
        assert step_node in wp_node.dependents

    def test_add_dependent(self):
        """Test adding a dependent relationship."""
        wp_key = ArtifactKey.for_workpiece(WP_UID_1)
        step_key = ArtifactKey.for_step(STEP_UID_1)

        wp_manager = create_manager_with_state(wp_key, 1, NodeState.VALID)
        step_manager = create_manager_with_state(step_key, 1, NodeState.DIRTY)

        wp_node = ArtifactNode(
            key=wp_key, generation_id=1, _artifact_manager=wp_manager
        )
        step_node = ArtifactNode(
            key=step_key, generation_id=1, _artifact_manager=step_manager
        )

        wp_node.add_dependent(step_node)

        assert step_node in wp_node.dependents
        assert wp_node in step_node.dependencies

    def test_mark_dirty_propagates_to_dependents(self):
        """Test that marking dirty propagates up the graph."""
        wp_key = ArtifactKey.for_workpiece(WP_UID_1)
        step_key = ArtifactKey.for_step(STEP_UID_1)
        job_key = ArtifactKey.for_job()

        wp_manager = create_manager_with_state(wp_key, 1, NodeState.VALID)
        step_manager = create_manager_with_state(step_key, 1, NodeState.VALID)
        job_manager = create_manager_with_state(job_key, 1, NodeState.VALID)

        wp_node = ArtifactNode(
            key=wp_key, generation_id=1, _artifact_manager=wp_manager
        )
        step_node = ArtifactNode(
            key=step_key, generation_id=1, _artifact_manager=step_manager
        )
        job_node = ArtifactNode(
            key=job_key, generation_id=1, _artifact_manager=job_manager
        )

        step_node.add_dependency(wp_node)
        job_node.add_dependency(step_node)

        wp_node.mark_dirty()

        wp_manager.set_state.assert_called_with(wp_key, 1, NodeState.DIRTY)
        step_manager.set_state.assert_called_with(step_key, 1, NodeState.DIRTY)
        job_manager.set_state.assert_called_with(job_key, 1, NodeState.DIRTY)

    def test_mark_dirty_idempotent(self):
        """Test that marking dirty multiple times is safe."""
        key = ArtifactKey.for_workpiece(WP_UID_1)
        manager = create_manager_with_state(key, 1, NodeState.VALID)
        node = ArtifactNode(
            key=key, generation_id=1, _artifact_manager=manager
        )

        node.mark_dirty()

        manager.set_state.assert_called_once_with(key, 1, NodeState.DIRTY)

    def test_is_ready_with_dirty_state(self):
        """Test is_ready returns True when node is DIRTY."""
        key = ArtifactKey.for_workpiece(WP_UID_1)
        manager = create_manager_with_state(key, 1, NodeState.DIRTY)
        node = ArtifactNode(
            key=key, generation_id=1, _artifact_manager=manager
        )

        assert node.is_ready() is True

    def test_is_ready_with_valid_state(self):
        """Test is_ready returns False when node is VALID."""
        key = ArtifactKey.for_workpiece(WP_UID_1)
        manager = create_manager_with_state(key, 1, NodeState.VALID)
        node = ArtifactNode(
            key=key, generation_id=1, _artifact_manager=manager
        )

        assert node.is_ready() is False

    def test_is_ready_with_processing_state(self):
        """Test is_ready returns False when node is PROCESSING."""
        key = ArtifactKey.for_workpiece(WP_UID_1)
        manager = create_manager_with_state(key, 1, NodeState.PROCESSING)
        node = ArtifactNode(
            key=key, generation_id=1, _artifact_manager=manager
        )

        assert node.is_ready() is False

    def test_is_ready_with_dirty_dependencies(self):
        """Test is_ready returns False when dependencies are DIRTY."""
        wp_key = ArtifactKey.for_workpiece(WP_UID_1)
        step_key = ArtifactKey.for_step(STEP_UID_1)

        wp_manager = create_manager_with_state(wp_key, 1, NodeState.DIRTY)
        step_manager = create_manager_with_state(step_key, 1, NodeState.DIRTY)

        wp_node = ArtifactNode(
            key=wp_key, generation_id=1, _artifact_manager=wp_manager
        )
        step_node = ArtifactNode(
            key=step_key, generation_id=1, _artifact_manager=step_manager
        )

        step_node.add_dependency(wp_node)

        assert step_node.is_ready() is False

    def test_is_ready_with_valid_dependencies(self):
        """Test is_ready returns True when all dependencies are VALID."""
        wp_key = ArtifactKey.for_workpiece(WP_UID_1)
        step_key = ArtifactKey.for_step(STEP_UID_1)

        wp_manager = create_manager_with_state(wp_key, 1, NodeState.VALID)
        step_manager = create_manager_with_state(step_key, 1, NodeState.DIRTY)

        wp_node = ArtifactNode(
            key=wp_key, generation_id=1, _artifact_manager=wp_manager
        )
        step_node = ArtifactNode(
            key=step_key, generation_id=1, _artifact_manager=step_manager
        )

        step_node.add_dependency(wp_node)

        assert step_node.is_ready() is True
