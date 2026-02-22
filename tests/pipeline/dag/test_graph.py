"""
Tests for the PipelineGraph class.
"""

from rayforge.pipeline.dag.graph import PipelineGraph
from rayforge.pipeline.dag.node import ArtifactNode
from rayforge.pipeline.artifact.key import ArtifactKey
from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.workpiece import WorkPiece
from rayforge.core.step import Step


WP_UID_1 = "550e8400-e29b-41d4-a716-446655440001"
WP_UID_2 = "550e8400-e29b-41d4-a716-446655440002"
STEP_UID_1 = "550e8400-e29b-41d4-a716-446655440003"
STEP_UID_2 = "550e8400-e29b-41d4-a716-446655440004"


class TestPipelineGraph:
    """Tests for the PipelineGraph class."""

    def test_empty_graph(self):
        """Test creating an empty graph."""
        graph = PipelineGraph()
        assert len(graph.get_all_nodes()) == 0

    def test_add_node(self):
        """Test adding a node to the graph."""
        graph = PipelineGraph()
        key = ArtifactKey.for_workpiece(WP_UID_1)
        node = ArtifactNode(key, generation_id=1)

        graph.add_node(node)

        assert len(graph.get_all_nodes()) == 1
        assert graph.get_node(key) == node

    def test_get_node_not_found(self):
        """Test getting a non-existent node returns None."""
        graph = PipelineGraph()
        key = ArtifactKey.for_workpiece("550e8400-e29b-41d4-a716-446655449999")

        assert graph.get_node(key) is None
        assert graph.find_node(key) is None

    def test_get_nodes_by_group(self):
        """Test filtering nodes by group."""
        graph = PipelineGraph()

        wp_key1 = ArtifactKey.for_workpiece(WP_UID_1)
        wp_key2 = ArtifactKey.for_workpiece(WP_UID_2)
        step_key = ArtifactKey.for_step(STEP_UID_1)
        job_key = ArtifactKey.for_job()

        graph.add_node(ArtifactNode(wp_key1, generation_id=1))
        graph.add_node(ArtifactNode(wp_key2, generation_id=1))
        graph.add_node(ArtifactNode(step_key, generation_id=1))
        graph.add_node(ArtifactNode(job_key, generation_id=1))

        wp_nodes = graph.get_nodes_by_group("workpiece")
        step_nodes = graph.get_nodes_by_group("step")
        job_nodes = graph.get_nodes_by_group("job")

        assert len(wp_nodes) == 2
        assert len(step_nodes) == 1
        assert len(job_nodes) == 1

    def test_build_from_empty_doc(self):
        """Test building graph from an empty doc."""
        doc = Doc()
        graph = PipelineGraph()

        graph.build_from_doc(doc, generation_id=1)

        assert len(graph.get_all_nodes()) == 1
        job_nodes = graph.get_nodes_by_group("job")
        assert len(job_nodes) == 1

    def test_build_from_doc_with_workpieces(self):
        """Test building graph with workpieces but no steps."""
        doc = Doc()
        layer = Layer("Test Layer")

        wp1 = WorkPiece(name="wp1.svg")
        wp2 = WorkPiece(name="wp2.svg")

        layer.add_child(wp1)
        layer.add_child(wp2)

        doc.set_children([layer])

        graph = PipelineGraph()
        graph.build_from_doc(doc, generation_id=1)

        wp_nodes = graph.get_nodes_by_group("workpiece")
        assert len(wp_nodes) == 0

    def test_build_from_doc_with_workpieces_and_steps(self):
        """Test building graph with workpieces and steps."""
        doc = Doc()
        layer = Layer("Test Layer")

        wp1 = WorkPiece(name="wp1.svg")
        wp2 = WorkPiece(name="wp2.svg")

        layer.add_child(wp1)
        layer.add_child(wp2)

        from rayforge.core.step import Step

        step = Step(typelabel="contour")
        assert layer.workflow is not None
        layer.workflow.add_step(step)

        doc.set_children([layer])

        graph = PipelineGraph()
        graph.build_from_doc(doc, generation_id=1)

        wp_nodes = graph.get_nodes_by_group("workpiece")
        assert len(wp_nodes) == 2

    def test_build_from_doc_with_steps(self):
        """Test building graph with steps."""
        doc = Doc()
        layer = doc.active_layer

        wp = WorkPiece(name="wp.svg")
        step = Step(typelabel="contour")

        layer.add_child(wp)
        assert layer.workflow is not None
        layer.workflow.add_step(step)

        graph = PipelineGraph()
        graph.build_from_doc(doc, generation_id=1)

        wp_nodes = graph.get_nodes_by_group("workpiece")
        step_nodes = graph.get_nodes_by_group("step")

        assert len(wp_nodes) == 1
        assert len(step_nodes) == 1

        step_node = step_nodes[0]
        assert len(step_node.dependencies) == 1

    def test_build_from_doc_creates_job_node(self):
        """Test that building graph creates a job node."""
        doc = Doc()
        graph = PipelineGraph()

        graph.build_from_doc(doc, generation_id=1)

        job_nodes = graph.get_nodes_by_group("job")
        assert len(job_nodes) == 1

    def test_build_from_doc_job_depends_on_steps(self):
        """Test that job node depends on all step nodes."""
        doc = Doc()
        layer = doc.active_layer

        wp = WorkPiece(name="wp.svg")
        step1 = Step(typelabel="contour", name="step1")
        step2 = Step(typelabel="engrave", name="step2")

        layer.add_child(wp)
        assert layer.workflow is not None
        layer.workflow.add_step(step1)
        layer.workflow.add_step(step2)

        graph = PipelineGraph()
        graph.build_from_doc(doc, generation_id=1)

        job_nodes = graph.get_nodes_by_group("job")
        job_node = job_nodes[0]

        assert len(job_node.dependencies) == 2

    def test_build_from_doc_clears_existing_nodes(self):
        """Test that building graph clears previous nodes."""
        doc = Doc()
        graph = PipelineGraph()

        key = ArtifactKey.for_workpiece("550e8400-e29b-41d4-a716-446655449999")
        graph.add_node(ArtifactNode(key, generation_id=1))

        assert len(graph.get_all_nodes()) == 1

        graph.build_from_doc(doc, generation_id=1)

        assert len(graph.get_all_nodes()) == 1
        assert graph.get_node(key) is None
