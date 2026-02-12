"""
Tests for the ArtifactNode class.
"""

from rayforge.pipeline.dag.node import ArtifactNode, NodeState
from rayforge.pipeline.artifact.key import ArtifactKey


WP_UID_1 = "550e8400-e29b-41d4-a716-446655440001"
WP_UID_2 = "550e8400-e29b-41d4-a716-446655440002"
STEP_UID_1 = "550e8400-e29b-41d4-a716-446655440003"


class TestArtifactNode:
    """Tests for the ArtifactNode class."""

    def test_node_creation(self):
        """Test creating a node with default state."""
        key = ArtifactKey.for_workpiece(WP_UID_1)
        node = ArtifactNode(key=key)
        assert node.key == key
        assert node.state == NodeState.DIRTY
        assert len(node.dependencies) == 0
        assert len(node.dependents) == 0

    def test_node_with_explicit_state(self):
        """Test creating a node with explicit state."""
        key = ArtifactKey.for_step(STEP_UID_1)
        node = ArtifactNode(key=key, state=NodeState.VALID)
        assert node.state == NodeState.VALID

    def test_add_dependency(self):
        """Test adding a dependency relationship."""
        wp_key = ArtifactKey.for_workpiece(WP_UID_1)
        step_key = ArtifactKey.for_step(STEP_UID_1)

        wp_node = ArtifactNode(key=wp_key, state=NodeState.VALID)
        step_node = ArtifactNode(key=step_key)

        step_node.add_dependency(wp_node)

        assert wp_node in step_node.dependencies
        assert step_node in wp_node.dependents

    def test_add_dependent(self):
        """Test adding a dependent relationship."""
        wp_key = ArtifactKey.for_workpiece(WP_UID_1)
        step_key = ArtifactKey.for_step(STEP_UID_1)

        wp_node = ArtifactNode(key=wp_key, state=NodeState.VALID)
        step_node = ArtifactNode(key=step_key)

        wp_node.add_dependent(step_node)

        assert step_node in wp_node.dependents
        assert wp_node in step_node.dependencies

    def test_mark_dirty_propagates_to_dependents(self):
        """Test that marking dirty propagates up the graph."""
        wp_key = ArtifactKey.for_workpiece(WP_UID_1)
        step_key = ArtifactKey.for_step(STEP_UID_1)
        job_key = ArtifactKey.for_job()

        wp_node = ArtifactNode(key=wp_key, state=NodeState.VALID)
        step_node = ArtifactNode(key=step_key, state=NodeState.VALID)
        job_node = ArtifactNode(key=job_key, state=NodeState.VALID)

        step_node.add_dependency(wp_node)
        job_node.add_dependency(step_node)

        wp_node.mark_dirty()

        assert wp_node.state == NodeState.DIRTY
        assert step_node.state == NodeState.DIRTY
        assert job_node.state == NodeState.DIRTY

    def test_mark_dirty_idempotent(self):
        """Test that marking dirty multiple times is safe."""
        key = ArtifactKey.for_workpiece(WP_UID_1)
        node = ArtifactNode(key=key, state=NodeState.DIRTY)

        node.mark_dirty()

        assert node.state == NodeState.DIRTY

    def test_is_ready_with_dirty_state(self):
        """Test is_ready returns True when node is DIRTY."""
        key = ArtifactKey.for_workpiece(WP_UID_1)
        node = ArtifactNode(key=key, state=NodeState.DIRTY)

        assert node.is_ready() is True

    def test_is_ready_with_valid_state(self):
        """Test is_ready returns False when node is VALID."""
        key = ArtifactKey.for_workpiece(WP_UID_1)
        node = ArtifactNode(key=key, state=NodeState.VALID)

        assert node.is_ready() is False

    def test_is_ready_with_processing_state(self):
        """Test is_ready returns False when node is PROCESSING."""
        key = ArtifactKey.for_workpiece(WP_UID_1)
        node = ArtifactNode(key=key, state=NodeState.PROCESSING)

        assert node.is_ready() is False

    def test_is_ready_with_dirty_dependencies(self):
        """Test is_ready returns False when dependencies are DIRTY."""
        wp_key = ArtifactKey.for_workpiece(WP_UID_1)
        step_key = ArtifactKey.for_step(STEP_UID_1)

        wp_node = ArtifactNode(key=wp_key, state=NodeState.DIRTY)
        step_node = ArtifactNode(key=step_key, state=NodeState.DIRTY)

        step_node.add_dependency(wp_node)

        assert step_node.is_ready() is False

    def test_is_ready_with_valid_dependencies(self):
        """Test is_ready returns True when all dependencies are VALID."""
        wp_key = ArtifactKey.for_workpiece(WP_UID_1)
        step_key = ArtifactKey.for_step(STEP_UID_1)

        wp_node = ArtifactNode(key=wp_key, state=NodeState.VALID)
        step_node = ArtifactNode(key=step_key, state=NodeState.DIRTY)

        step_node.add_dependency(wp_node)

        assert step_node.is_ready() is True
